"""State-to-state SCJEPA: true object states in, true next states out.

The predictive objective has two deliberately small branches. Teacher forcing
uses every observed suffix transition, while the auxiliary branch samples local
two-step compositions::

    L_pred = L_TF + lambda_rollout_t2 * L_AR2.

For an episode of length 60 with a 30-state context, ``L_TF`` covers the 30
transitions ``S_29 -> S_30, ..., S_58 -> S_59``. ``L_AR2`` samples true
anchors ``S_(29+r)`` with ``r in {0, ..., 28}``, predicts one state from the
anchor, feeds that prediction back once, and supervises only the endpoint.
Every transition and every sampled window reuses the one attached
``theta_hat`` inferred from ``S_0, ..., S_29``.

The model also exposes an evaluation-only autoregressive rollout. It refuses
to run while gradients are enabled or the module is in training mode, so the
K=30 observational-equivalence diagnostic cannot accidentally become a
training objective.
"""

from typing import NamedTuple

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from scjepa.models.parameter_encoder import ParameterEncoder
from scjepa.models.spartan import Spartan


class TransitionOutput(NamedTuple):
    """Teacher-forced predictions plus an optional batch of T=2 endpoints.

    Graph quantities are produced by the teacher-forced predictor call only.
    The auxiliary calls share the same predictor parameters and ``theta_hat``
    but do not add duplicate path/logit regularizers.
    """

    prediction: Float[Tensor, "bk n d"]
    target: Float[Tensor, "bk n d"]
    causal_params: Float[Tensor, "b n 1"]
    path_matrix: Float[Tensor, "bk m m"]
    sparsity: Float[Tensor, ""]
    logit_penalty: Float[Tensor, ""]
    mean_abs_logit: Float[Tensor, ""]
    mean_gate_probability: Float[Tensor, ""]
    gate_entropy: Float[Tensor, ""]
    rollout_t2_prediction: Float[Tensor, "b w n d"] | None = None
    rollout_t2_target: Float[Tensor, "b w n d"] | None = None
    rollout_t2_offsets: Int[Tensor, "b w"] | None = None
    rollout_t2_intermediate: Float[Tensor, "b w n d"] | None = None


def num_valid_rollout_t2_offsets(sequence_len: int, context_len: int) -> int:
    """Return the number of true anchors whose two-step target is in bounds.

    The first anchor is ``S_(context_len-1)``. An offset ``r`` is valid when
    ``context_len - 1 + r + 2 <= sequence_len - 1``. Hence the number of
    valid offsets is ``sequence_len - context_len - 1`` (29 for 60/30).
    """
    if not 1 <= context_len < sequence_len:
        raise ValueError(f"context_len={context_len} must be in [1, T-1={sequence_len - 1}]")
    return max(sequence_len - context_len - 1, 0)


def sample_rollout_t2_offsets(
    batch_size: int,
    num_valid_offsets: int,
    num_anchors: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> Int[Tensor, "b w"]:
    """Sample distinct offsets independently for every episode.

    Sampling uses the active PyTorch RNG when ``generator`` is omitted. The
    trainer checkpoints that RNG (CPU or CUDA), so resuming preserves the
    sampling stream to the same standard as Gumbel-gate sampling.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_valid_offsets < 1:
        raise ValueError(f"no valid two-step rollout offsets ({num_valid_offsets})")
    if not 1 <= num_anchors <= num_valid_offsets:
        raise ValueError(f"num_anchors={num_anchors} must be in [1, {num_valid_offsets}]")
    # Each row receives independent continuous random priorities. Taking the
    # lowest W priorities is uniform sampling without replacement per row.
    priorities = torch.rand(
        batch_size,
        num_valid_offsets,
        device=device,
        generator=generator,
    )
    return priorities.topk(num_anchors, dim=1, largest=False, sorted=False).indices


class StateToStateModel(nn.Module):
    """Parameter encoder ``P_eta`` plus SPARTAN state transition ``F_hat``."""

    def __init__(self, parameter_encoder: ParameterEncoder, predictor: Spartan) -> None:
        """Compose the parameter encoder and shared transition predictor."""
        super().__init__()
        self.parameter_encoder = parameter_encoder
        self.predictor = predictor

    def forward(
        self,
        states: Float[Tensor, "b t n d"],
        context_len: int | None = None,
        num_rollout_t2_anchors: int = 0,
        rollout_t2_offsets: Int[Tensor, "b w"] | None = None,
    ) -> TransitionOutput:
        """Infer one ``theta_hat``, then compute TF and optional T=2 branches.

        Args:
            states: Episode batch ``(B, T, N, d)``.
            context_len: Number of states used to infer ``theta_hat``. The
                teacher-forced anchors begin at ``context_len - 1``.
            num_rollout_t2_anchors: Number of distinct offsets sampled per
                episode. Zero disables the auxiliary branch without drawing
                from the RNG, which exactly recovers teacher forcing.
            rollout_t2_offsets: Optional explicit ``(B, W)`` offsets. This is
                used by deterministic evaluation/tests; normal training leaves
                it unset and samples with the active training RNG.
        """
        if states.ndim != 4 or states.shape[1] < 2:
            raise ValueError(f"expected (B, T>=2, N, d), got {tuple(states.shape)}")
        batch, length = states.shape[:2]
        tpar = context_len if context_len is not None else length - 1
        if not 1 <= tpar < length:
            raise ValueError(f"context_len={tpar} must be in [1, T-1={length - 1}]")
        if num_rollout_t2_anchors < 0:
            raise ValueError("num_rollout_t2_anchors must be non-negative")
        num_transitions = length - tpar

        # Exactly one episode-level parameter inference. The same attached
        # tensor is expanded (never recomputed or detached) everywhere below.
        causal_params = self.parameter_encoder(states[:, :tpar])
        anchors = states[:, tpar - 1 : -1].flatten(0, 1)
        targets = states[:, tpar:].flatten(0, 1)
        params = causal_params.repeat_interleave(num_transitions, dim=0)
        out = self.predictor(anchors, params)

        endpoint = endpoint_target = sampled_offsets = intermediate = None
        if rollout_t2_offsets is not None:
            if rollout_t2_offsets.ndim != 2 or rollout_t2_offsets.shape[0] != batch:
                raise ValueError(
                    "rollout_t2_offsets must have shape (B, W), got "
                    f"{tuple(rollout_t2_offsets.shape)} for B={batch}"
                )
            if rollout_t2_offsets.dtype == torch.bool or rollout_t2_offsets.is_floating_point():
                raise ValueError("rollout offsets must be integer indices")
            if num_rollout_t2_anchors not in (0, rollout_t2_offsets.shape[1]):
                raise ValueError(
                    "num_rollout_t2_anchors must be zero or match explicit offset width"
                )
            sampled_offsets = rollout_t2_offsets.to(device=states.device, dtype=torch.long)
        elif num_rollout_t2_anchors > 0:
            valid = num_valid_rollout_t2_offsets(length, tpar)
            sampled_offsets = sample_rollout_t2_offsets(
                batch,
                valid,
                num_rollout_t2_anchors,
                device=states.device,
            )

        if sampled_offsets is not None:
            endpoint, endpoint_target, intermediate = self.rollout_t2_from_offsets(
                states,
                tpar,
                causal_params,
                sampled_offsets,
            )

        return TransitionOutput(
            prediction=out.prediction,
            target=targets,
            causal_params=causal_params,
            path_matrix=out.path_matrix,
            sparsity=out.sparsity,
            logit_penalty=out.logit_penalty,
            mean_abs_logit=out.mean_abs_logit,
            mean_gate_probability=out.mean_gate_probability,
            gate_entropy=out.gate_entropy,
            rollout_t2_prediction=endpoint,
            rollout_t2_target=endpoint_target,
            rollout_t2_offsets=sampled_offsets,
            rollout_t2_intermediate=intermediate,
        )

    def rollout_t2_from_offsets(
        self,
        states: Float[Tensor, "b t n d"],
        context_len: int,
        causal_params: Float[Tensor, "b n 1"],
        offsets: Int[Tensor, "b w"],
    ) -> tuple[
        Float[Tensor, "b w n d"],
        Float[Tensor, "b w n d"],
        Float[Tensor, "b w n d"],
    ]:
        """Vectorize independent true-anchored two-step windows.

        Only the second prediction is supervised. The intermediate is retained
        for contract tests and is never detached before the second call.
        """
        batch, length = states.shape[:2]
        valid = num_valid_rollout_t2_offsets(length, context_len)
        if offsets.ndim != 2 or offsets.shape[0] != batch or offsets.shape[1] < 1:
            raise ValueError(f"offsets must be non-empty (B, W), got {tuple(offsets.shape)}")
        if offsets.dtype == torch.bool or offsets.is_floating_point():
            raise ValueError("rollout offsets must be integer indices")
        offsets = offsets.to(device=states.device, dtype=torch.long)
        if bool((offsets < 0).any()) or bool((offsets >= valid).any()):
            raise ValueError(f"rollout offsets must lie in [0, {valid - 1}]")
        sorted_offsets = offsets.sort(dim=1).values
        if sorted_offsets.shape[1] > 1 and bool(
            (sorted_offsets[:, 1:] == sorted_offsets[:, :-1]).any()
        ):
            raise ValueError("rollout offsets must be distinct within every episode")

        num_windows = offsets.shape[1]
        episode_index = torch.arange(batch, device=states.device)[:, None]
        anchor_index = context_len - 1 + offsets
        true_anchors = states[episode_index, anchor_index]
        window_params = (
            causal_params[:, None]
            .expand(-1, num_windows, -1, -1)
            .reshape(batch * num_windows, *causal_params.shape[1:])
        )
        first = self.predictor(true_anchors.flatten(0, 1), window_params).prediction
        # No detach: endpoint gradients traverse both transition calls.
        second = self.predictor(first, window_params).prediction
        target = states[episode_index, anchor_index + 2].detach()
        shape = (batch, num_windows, *states.shape[2:])
        return second.reshape(shape), target, first.reshape(shape)

    def rollout_for_evaluation(
        self,
        states: Float[Tensor, "b t n d"],
        context_len: int,
        horizon: int,
        causal_params: Float[Tensor, "b n 1"],
    ) -> tuple[Float[Tensor, "b k n d"], Float[Tensor, "b k n d"]]:
        """Run one open-loop trajectory strictly as a no-gradient diagnostic.

        The chain begins at the true state ``S_(context_len-1)`` and feeds every
        generated prediction into the next transition. This method is outside
        :meth:`forward` so no training call can request K=30 BPTT.
        """
        if self.training:
            raise RuntimeError("evaluation rollout requires model.eval()")
        if torch.is_grad_enabled():
            raise RuntimeError("evaluation rollout requires torch.no_grad()")
        if states.ndim != 4:
            raise ValueError(f"expected (B, T, N, d), got {tuple(states.shape)}")
        if horizon < 1:
            raise ValueError(f"evaluation horizon must be positive, got {horizon}")
        length = states.shape[1]
        if not 1 <= context_len < length:
            raise ValueError(f"invalid context_len={context_len} for T={length}")
        if context_len + horizon > length:
            raise ValueError(
                f"evaluation horizon {horizon} runs past T={length} from anchor S_{context_len - 1}"
            )
        if causal_params.shape[0] != states.shape[0]:
            raise ValueError("causal_params batch does not match states")

        state = states[:, context_len - 1]
        predictions: list[Tensor] = []
        for _ in range(horizon):
            state = self.predictor(state, causal_params).prediction
            predictions.append(state)
        return torch.stack(predictions, dim=1), states[:, context_len : context_len + horizon]


def build_state_to_state(
    state_dim: int = 4,
    num_slots: int = 5,
    param_encoder_dim: int = 32,
    param_encoder_heads: int = 4,
    max_history: int = 64,
    spartan_layers: int = 3,
    spartan_embed_dim: int = 512,
    spartan_mlp_hidden: int = 512,
    spartan_mlp_layers: int = 3,
    spartan_temperature: float = 1.0,
    spartan_dense: bool = False,
    spartan_identity: bool = False,
) -> StateToStateModel:
    """Build the state-to-state model from plain configuration values."""
    return StateToStateModel(
        parameter_encoder=ParameterEncoder(
            state_dim=state_dim,
            dim=param_encoder_dim,
            num_heads=param_encoder_heads,
            max_history=max_history,
        ),
        predictor=Spartan(
            state_dim=state_dim,
            param_dim=1,
            num_slots=num_slots,
            num_layers=spartan_layers,
            embed_dim=spartan_embed_dim,
            mlp_hidden_size=spartan_mlp_hidden,
            mlp_num_layers=spartan_mlp_layers,
            temperature=spartan_temperature,
            dense=spartan_dense,
            identity=spartan_identity,
        ),
    )


__all__ = [
    "StateToStateModel",
    "TransitionOutput",
    "build_state_to_state",
    "num_valid_rollout_t2_offsets",
    "sample_rollout_t2_offsets",
]
