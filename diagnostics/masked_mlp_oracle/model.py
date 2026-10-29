"""A shared-trunk MLP with static per-edge input masks, built for chart equivariance.

Every output coordinate ``i`` applies one shared trunk to its own masked input
``[z * m_z[i], a * m_a[i], theta * m_theta[i]]`` and reads the result with its
own linear readout. There is no output-index embedding. Mask logits are static
parameters, one per (output, input) edge and independent of input values, so the
thresholded masks are global functional graphs. Training samples hard logistic-
noise gates with a straight-through derivative; evaluation thresholds at zero.
Each block's sparsity term is the exact expected open-edge fraction,
``sigmoid(logit)`` averaged over that block, with SPARTAN's block normalization.

With every mask open this is exactly the global ``36-H-H-6`` MLP. With
block-equal state masks and fixed-open parameter masks (the baseline), any
within-block orthogonal re-charting of the state is absorbed exactly by the
trunk's state columns and the readouts. Gaussian initialization keeps that
symmetry in distribution, so the baseline cannot prefer one within-block chart.
Only learned parameter masks (full) can break it.
"""

import math
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from diagnostics.isotropic_oracle.model import ACTION_DIMENSION
from diagnostics.isotropic_oracle.model import build_control as build_native
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import GraphBlocks, GraphPenalties, Spartan, SpartanOutput
from scjepa.training.rng import preserve_rng

type Condition = Literal["dense", "baseline", "full"]

STATE_DIMENSION = 6
PARAMETER_DIMENSION = 6
INITIAL_LOGIT = 3.0
NATIVE_SPARTAN_PARAMETERS = 52_801


def trunk_width() -> int:
    """Return the width of the existing parameter-count-matched 24-action global MLP."""
    return min(range(1, 1025), key=lambda h: abs(h * h + 44 * h + 6 - NATIVE_SPARTAN_PARAMETERS))


def _gaussian_(tensor: Tensor, fan_in: int) -> None:
    """Match PyTorch's default Linear variance, 1/(3 fan_in), with an O(n)-invariant law."""
    with torch.no_grad():
        tensor.normal_(0.0, 1.0 / math.sqrt(3.0 * fan_in))


def _hard_gates(logits: Tensor, batch: int, temperature: float, training: bool) -> Tensor:
    """Sample independent per-example gates in training; threshold at zero otherwise."""
    if not training:
        return (logits > 0).to(logits.dtype).expand(batch, *logits.shape)
    uniform = torch.rand(batch, *logits.shape, device=logits.device, dtype=logits.dtype)
    uniform = uniform.clamp(1e-6, 1 - 1e-6)
    soft = torch.sigmoid((logits + uniform.log() - (-uniform).log1p()) / temperature)
    return (soft > 0.5).to(logits.dtype) + soft - soft.detach()


class MaskedTrunkPredictor(nn.Module):
    """Predict ``z_{t+1}`` from one state lag, observed actions and supplied theta."""

    history_len = 1

    def __init__(
        self,
        *,
        state_dim: int = STATE_DIMENSION,
        action_dim: int = ACTION_DIMENSION,
        param_dim: int = PARAMETER_DIMENSION,
        hidden_dim: int | None = None,
        learn_state: bool = True,
        learn_action: bool = True,
        learn_parameter: bool = True,
        initial_logit: float = INITIAL_LOGIT,
        temperature: float = 1.0,
    ) -> None:
        """Build the trunk, per-output readouts and the selected learned mask blocks."""
        super().__init__()
        if min(state_dim, action_dim, param_dim) < 1:
            raise ValueError("state, action and parameter dimensions must be positive")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(initial_logit):
            raise ValueError("initial_logit must be finite")
        hidden = trunk_width() if hidden_dim is None else hidden_dim
        if hidden < 1:
            raise ValueError("hidden_dim must be positive")
        self.state_dim, self.action_dim, self.param_dim = state_dim, action_dim, param_dim
        self.hidden_dim, self.temperature = hidden, temperature
        input_dim = state_dim + action_dim + param_dim
        self.first = nn.Linear(input_dim, hidden)
        self.second = nn.Linear(hidden, hidden)
        self.readout_weight = nn.Parameter(torch.empty(state_dim, hidden))
        self.readout_bias = nn.Parameter(torch.empty(state_dim))
        for layer in (self.first, self.second):
            _gaussian_(layer.weight, layer.in_features)
            _gaussian_(layer.bias, layer.in_features)
        _gaussian_(self.readout_weight, hidden)
        _gaussian_(self.readout_bias, hidden)
        # Constant initial logits consume no random draws, so arms stay paired.
        self.state_logits = self._logits(learn_state, state_dim, initial_logit)
        self.action_logits = self._logits(learn_action, action_dim, initial_logit)
        self.parameter_logits = self._logits(learn_parameter, param_dim, initial_logit)

    def _logits(self, learned: bool, inputs: int, initial: float) -> nn.Parameter | None:
        if not learned:
            return None
        return nn.Parameter(torch.full((self.state_dim, inputs), float(initial)))

    def _block(
        self, logits: nn.Parameter | None, inputs: int, reference: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return sampled/thresholded gates and the exact expected open fraction."""
        batch = len(reference)
        if logits is None:
            gates = reference.new_ones(batch, self.state_dim, inputs)
            return gates, reference.new_ones(())
        gates = _hard_gates(logits, batch, self.temperature, self.training)
        return gates, torch.sigmoid(logits).mean()

    def learned_logits(self) -> dict[str, Tensor]:
        """Return each learned logit block; fixed-open blocks are absent."""
        blocks = {
            "state": self.state_logits,
            "action": self.action_logits,
            "parameter": self.parameter_logits,
        }
        return {name: value for name, value in blocks.items() if value is not None}

    def thresholded_graphs(self) -> dict[str, Tensor]:
        """Return evaluation-time global graphs; fixed-open blocks are all ones."""
        shapes = {
            "state": self.state_dim,
            "action": self.action_dim,
            "parameter": self.param_dim,
        }
        logits = self.learned_logits()
        return {
            name: (logits[name] > 0)
            if name in logits
            else torch.ones(self.state_dim, inputs, dtype=torch.bool)
            for name, inputs in shapes.items()
        }

    def forward(
        self, states: Tensor, parameters: Tensor, actions: Tensor | None = None
    ) -> SpartanOutput:
        """Return SPARTAN-compatible output so the native objective and evaluator apply."""
        batch = len(states)
        if batch == 0 or states.shape != (batch, 1, self.state_dim):
            raise ValueError(f"expected nonempty states (B, 1, {self.state_dim})")
        if parameters.shape != (batch, self.param_dim):
            raise ValueError(f"expected parameters (B, {self.param_dim})")
        if actions is None or actions.shape != (batch, 1, self.action_dim):
            raise ValueError(f"expected actions (B, 1, {self.action_dim})")
        state_gates, state_open = self._block(self.state_logits, self.state_dim, states)
        action_gates, action_open = self._block(self.action_logits, self.action_dim, states)
        parameter_gates, parameter_open = self._block(self.parameter_logits, self.param_dim, states)
        inputs = torch.cat(
            (
                states[:, 0, None, :] * state_gates,
                actions[:, 0, None, :] * action_gates,
                parameters[:, None, :] * parameter_gates,
            ),
            dim=-1,
        )
        features = F.relu(self.second(F.relu(self.first(inputs))))
        prediction = (features * self.readout_weight).sum(-1) + self.readout_bias
        rows = torch.cat((state_gates, action_gates, parameter_gates), dim=-1)
        tokens = rows.shape[-1]
        path_matrix = torch.eye(tokens, dtype=rows.dtype, device=rows.device).repeat(batch, 1, 1)
        path_matrix[:, : self.state_dim] = rows
        logits = list(self.learned_logits().values())
        if logits:
            flat = torch.cat([value.flatten() for value in logits]).detach()
            probability = torch.sigmoid(flat)
            entropy = F.softplus(flat) - flat * probability
            mean_abs, mean_probability = flat.abs().mean(), probability.mean()
            mean_entropy = entropy.mean()
        else:
            mean_abs = mean_entropy = states.new_zeros(())
            mean_probability = states.new_ones(())
        return SpartanOutput(
            prediction=prediction,
            path_matrix=path_matrix,
            graphs=GraphBlocks(
                state_gates[:, :, None, :], action_gates[:, :, None, :], parameter_gates
            ),
            sparsity=GraphPenalties(state_open, action_open, parameter_open),
            logit_penalty=states.new_zeros(()),
            mean_abs_logit=mean_abs,
            mean_gate_probability=mean_probability,
            gate_entropy=mean_entropy,
        )


def build_predictor(condition: Condition) -> MaskedTrunkPredictor:
    """Dense opens every mask; baseline learns state/action; full learns all three."""
    if condition not in {"dense", "baseline", "full"}:
        raise ValueError("Unknown masked MLP condition")
    sparse = condition != "dense"
    return MaskedTrunkPredictor(
        learn_state=sparse, learn_action=sparse, learn_parameter=condition == "full"
    )


def build_control(condition: Condition, seed: int) -> SCJEPA:
    """Pair encoder, trunk and readout tensors across conditions for one seed.

    The native isotropic constructor supplies the oracle SCJEPA wrapper and its
    frozen, bypassed parameter encoder. The encoder head is then redrawn from a
    Gaussian law and copied to the EMA target; the SPARTAN predictor is replaced.
    """
    with preserve_rng(torch.device("cpu")):
        model = build_native("dense", seed)
        native = sum(parameter.numel() for parameter in model.predictor.parameters())
        if native != NATIVE_SPARTAN_PARAMETERS:
            raise RuntimeError("native 24-action SPARTAN size changed; trunk width is unmatched")
        torch.default_generator.manual_seed(seed)
        head = model.online_encoder.head
        if not isinstance(head, nn.Linear):
            raise TypeError("expected a linear state head")
        _gaussian_(head.weight, head.in_features)
        _gaussian_(head.bias, head.in_features)
        model.target_encoder.load_state_dict(model.online_encoder.state_dict())
        predictor = build_predictor(condition)
    # SCJEPA reads only dimensions and SpartanOutput fields from its predictor.
    model.predictor = cast(Spartan, predictor)
    return model


__all__ = [
    "INITIAL_LOGIT",
    "Condition",
    "MaskedTrunkPredictor",
    "build_control",
    "build_predictor",
    "trunk_width",
]
