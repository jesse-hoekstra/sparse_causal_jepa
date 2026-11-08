"""Framewise visual JEPA with two state/action lags and a learned mechanical code."""

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from diagnostics.masked_mlp_oracle.model import (
    MaskedTrunkPredictor,
    _gaussian_,  # pyright: ignore[reportPrivateUsage]
)
from scjepa.models.encoders import ImageEncoder
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.models.spartan import GraphBlocks, GraphPenalties, Spartan, SpartanOutput
from scjepa.training.rng import preserve_rng


class LaggedParameterEncoder(TransitionParameterEncoder):
    """Pool (z_previous,z_current,a_previous,a_current,z_next) within one context."""

    def __init__(self, context: int, code_dim: int = 4) -> None:
        """Retain the existing GELU transition/head MLP with a second-order input."""
        super().__init__(2, code_dim, 1, max_history=context)
        self.transition[0] = nn.Linear(8, 128)
        first = cast(nn.Linear, self.transition[0])
        _gaussian_(first.weight, 8)
        _gaussian_(first.bias, 8)
        linears = [module for module in self.modules() if isinstance(module, nn.Linear)]
        with torch.no_grad():
            for layer in linears[:-1]:
                layer.weight.mul_(math.sqrt(6))
            linears[-1].weight.mul_(math.sqrt(3))

    def features(self, states: Tensor, actions: Tensor | None = None) -> Tensor:
        """Return penultimate context features, using no transition beyond the context."""
        if states.ndim != 3 or states.shape[-1] != 2 or states.shape[1] < 3:
            raise ValueError("expected states (B,K>=3,2)")
        if actions is None or actions.shape != (len(states), states.shape[1] - 1, 1):
            raise ValueError("expected context actions (B,K-1,1)")
        inputs = torch.cat(
            (states[:, :-2], states[:, 1:-1], actions[:, :-1], actions[:, 1:], states[:, 2:]), -1
        )
        return self.head[:-1](self.transition(inputs).mean(1))

    def forward(self, states: Tensor, actions: Tensor | None = None) -> Tensor:
        """Map one context's pooled features to its complete parameter code."""
        return self.head[-1](self.features(states, actions))


class LaggedPredictor(MaskedTrunkPredictor):
    """Shared GELU trunk with coordinate masks repeated over both state/action lags."""

    history_len = 2

    def __init__(self, *, sparse: bool, code_dim: int = 4) -> None:
        """Keep parameter gates fixed open in every phase of the baseline screen."""
        super().__init__(
            state_dim=2,
            action_dim=1,
            param_dim=code_dim,
            hidden_dim=209,
            learn_state=sparse,
            learn_action=sparse,
            learn_parameter=False,
            activation="gelu",
        )
        self.first = nn.Linear(6 + code_dim, self.hidden_dim)
        _gaussian_(self.first.weight, self.first.in_features)
        _gaussian_(self.first.bias, self.first.in_features)

    def forward(
        self, states: Tensor, parameters: Tensor, actions: Tensor | None = None
    ) -> SpartanOutput:
        """Predict one future state from two lags, retaining attached donor codes."""
        batch = len(states)
        if states.shape != (batch, 2, 2) or parameters.shape != (batch, self.param_dim):
            raise ValueError(f"expected states (B,2,2), parameters (B,{self.param_dim})")
        if actions is None or actions.shape != (batch, 2, 1):
            raise ValueError("expected actions (B,2,1)")
        sg, so = self._block(self.state_logits, 2, states)
        ag, ao = self._block(self.action_logits, 1, states)
        pg, po = self._block(self.parameter_logits, self.param_dim, states)
        state_gates = sg[:, :, None].expand(-1, -1, 2, -1)
        action_gates = ag[:, :, None].expand(-1, -1, 2, -1)
        inputs = torch.cat(
            (
                (states[:, None] * state_gates).flatten(2),
                (actions[:, None] * action_gates).flatten(2),
                parameters[:, None] * pg,
            ),
            -1,
        )
        features = F.gelu(self.second(F.gelu(self.first(inputs))))
        prediction = (features * self.readout_weight).sum(-1) + self.readout_bias
        rows = torch.cat((state_gates.flatten(2), action_gates.flatten(2), pg), -1)
        paths = torch.eye(6 + self.param_dim, device=states.device, dtype=states.dtype).repeat(
            batch, 1, 1
        )
        paths[:, :2] = rows
        logits = list(self.learned_logits().values())
        zero = states.new_zeros(())
        flat = torch.cat([value.flatten() for value in logits]).detach() if logits else zero[None]
        return SpartanOutput(
            prediction,
            paths,
            GraphBlocks(state_gates, action_gates, pg),
            GraphPenalties(so, ao, po),
            zero,
            flat.abs().mean(),
            torch.sigmoid(flat).mean(),
            (F.softplus(flat) - flat * torch.sigmoid(flat)).mean(),
        )


def build_model(seed: int, context: int, *, sparse: bool = False, code_dim: int = 4) -> SCJEPA:
    """Construct the visual model without physical-state or parameter inputs."""
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(seed)
        encoder = ImageEncoder(1, 2, channels=(16, 32), pool_size=4)
        _gaussian_(encoder.head.weight, encoder.head.in_features)
        _gaussian_(encoder.head.bias, encoder.head.in_features)
        parameter_encoder = LaggedParameterEncoder(context, code_dim)
        predictor = LaggedPredictor(sparse=sparse, code_dim=code_dim)
        return SCJEPA(
            encoder,
            parameter_encoder,
            cast(Spartan, predictor),
            context,
            ema_decay=0.996,
            autoregressive=True,
        )
