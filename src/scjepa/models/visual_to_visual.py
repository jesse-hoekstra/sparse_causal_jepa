"""Experiment 2: anonymous video frames, online states and stopped-gradient EMA targets.

This extends Experiment 1's teacher forcing plus sampled T=2 endpoints to learned
states. One parameter vector per episode and one set of track keys are reused
by every transition. Full-horizon autoregression is evaluation-only.

EMA keeps feature coordinates close, but does not guarantee that recurrent slot
row i tracks the same object in both branches. We therefore match pre-head slot
trajectories over the observed context only, detach the assignment and keep it
fixed for every target frame in that episode. This is an implementation change
from the row-identity assumption in the outdated experiment proposal. It uses
neither simulator labels nor future frames; it cannot repair a mid-episode swap
or establish that slots represent objects in the first place.

Only the online visual path is copied to the target. Parameter inference and
SPARTAN remain predictor-side. Latent MSE is the gradient objective; the GECO
constraint normalizes it by detached target variance. Neither EMA nor a
variance denominator rules out representation collapse; held-out grounding,
state probes and collapse diagnostics must accompany prediction loss.
"""

import copy
from typing import NamedTuple, Self

import torch
from jaxtyping import Float, Int
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from scjepa.losses.alignment import align_to_assignment
from scjepa.models.parameter_encoder import ParameterEncoder
from scjepa.models.spartan import Spartan
from scjepa.models.state_to_state import num_valid_rollout_t2_offsets, sample_rollout_t2_offsets
from scjepa.models.visual import VisualStatePath

__all__ = [
    "VisualToVisualModel",
    "VisualToVisualOutput",
    "build_visual_to_visual",
    "context_target_assignment",
]


class VisualToVisualOutput(NamedTuple):
    """Teacher forcing, optional local endpoints and detached target alignment."""

    prediction: Float[Tensor, "bk n s"]
    target: Float[Tensor, "bk n s"]
    causal_params: Float[Tensor, "b n 1"]
    path_matrix: Float[Tensor, "bk m m"]
    sparsity: Float[Tensor, ""]
    logit_penalty: Float[Tensor, ""]
    mean_abs_logit: Float[Tensor, ""]
    mean_gate_probability: Float[Tensor, ""]
    gate_entropy: Float[Tensor, ""]
    target_variance: Float[Tensor, ""]
    context_states: Float[Tensor, "b t n s"]
    target_states: Float[Tensor, "b t n s"]
    """EMA states in online row order under the fixed context assignment."""
    target_assignment: Int[Tensor, "b n"]
    """assignment[b, i] is the EMA row matched to online row i."""
    episode_keys: Float[Tensor, "b n d"]
    rollout_t2_prediction: Float[Tensor, "b w n s"] | None = None
    rollout_t2_target: Float[Tensor, "b w n s"] | None = None
    rollout_t2_offsets: Int[Tensor, "b w"] | None = None
    rollout_t2_intermediate: Float[Tensor, "b w n s"] | None = None
    context_allocations: Float[Tensor, "b t n p"] | None = None
    target_allocations: Float[Tensor, "b t n p"] | None = None


def _content_variance(states: Tensor) -> Tensor:
    """Mean per-row, per-coordinate variance across episodes and timesteps."""
    return states.flatten(0, 1).var(dim=0, unbiased=False).mean()


@torch.no_grad()
def context_target_assignment(
    context_slots: Float[Tensor, "b c n d"],
    target_slots: Float[Tensor, "b c n d"],
) -> Int[Tensor, "b n"]:
    """Match two observed slot histories once, without supervision or future data.

    Cost is mean squared distance in pre-head feature coordinates. A shared
    detached coordinate scale estimated from both context histories prevents a
    high-amplitude feature from arbitrarily dominating the permutation. EMA
    makes comparison in this shared feature basis reasonable, not infallible.
    Assignment is discrete and detached. Identical histories resolve to row
    identity; degenerate slots still require explicit collapse diagnostics.
    """
    if context_slots.ndim != 4 or context_slots.shape != target_slots.shape:
        raise ValueError("context and target slot histories must share shape (B, C, N, D)")
    online = context_slots.detach().float()
    target = target_slots.detach().float()
    pooled = torch.cat((online.flatten(1, 2), target.flatten(1, 2)), dim=1)
    scales = pooled.var(dim=1, unbiased=False).clamp_min(1e-6).sqrt()
    difference = (online.unsqueeze(3) - target.unsqueeze(2)) / scales[:, None, None, None, :]
    cost = difference.square().mean(dim=(1, 4))
    # One small B x N x N transfer, not one device synchronization per episode.
    cost_cpu = cost.cpu().numpy()
    assignments = [linear_sum_assignment(episode)[1].tolist() for episode in cost_cpu]
    return torch.tensor(assignments, dtype=torch.long, device=context_slots.device)


class VisualToVisualModel(nn.Module):
    """Online visual path + EMA target path + parameter encoder + SPARTAN."""

    def __init__(
        self,
        online: VisualStatePath,
        parameter_encoder: ParameterEncoder,
        predictor: Spartan,
        ema_decay: float = 0.996,
        variance_floor: float = 1e-4,
    ) -> None:
        """Compose the learned-state model and initialize its frozen target."""
        super().__init__()
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        if variance_floor <= 0:
            raise ValueError("variance_floor must be positive")
        self.online = online
        self.target = copy.deepcopy(online)
        self.target.requires_grad_(False)
        self.target.eval()
        self.parameter_encoder = parameter_encoder
        self.predictor = predictor
        self.ema_decay = ema_decay
        self.variance_floor = variance_floor

    def train(self, mode: bool = True) -> Self:
        """Keep targets deterministic even while the online model trains."""
        super().train(mode)
        self.target.eval()
        return self

    @torch.no_grad()
    def update_target(self) -> None:
        """Advance the EMA only after an accepted optimizer update."""
        for target, online in zip(self.target.parameters(), self.online.parameters(), strict=True):
            target.mul_(self.ema_decay).add_(online.detach(), alpha=1.0 - self.ema_decay)
        for target_buffer, online_buffer in zip(
            self.target.buffers(), self.online.buffers(), strict=True
        ):
            target_buffer.copy_(online_buffer)

    def forward(
        self,
        frames: Float[Tensor, "b t c h w"],
        context_len: int | None = None,
        num_rollout_t2_anchors: int = 0,
        rollout_t2_offsets: Int[Tensor, "b w"] | None = None,
        capture_allocations: bool = False,
    ) -> VisualToVisualOutput:
        """Encode causal histories once and compute TF plus optional T=2 endpoints.

        Parameter inference and the target assignment use only frames 0..C-1.
        Teacher-forced online anchors at later t see frames 0..t; their EMA
        targets see frames 0..t+1. No online prediction sees its future image.
        A zero anchor count bypasses auxiliary sampling and predictor calls.
        """
        if frames.ndim != 5 or frames.shape[1] < 2:
            raise ValueError(f"expected (B, T>=2, C, H, W), got {tuple(frames.shape)}")
        batch, length = frames.shape[:2]
        tpar = context_len if context_len is not None else length - 1
        if not 1 <= tpar < length:
            raise ValueError(f"context_len={tpar} must be in [1, T-1={length - 1}]")
        if num_rollout_t2_anchors < 0:
            raise ValueError("num_rollout_t2_anchors must be non-negative")
        transitions = length - tpar
        context = self.online(frames[:, :-1], capture_allocations=capture_allocations)
        with torch.no_grad():
            target = self.target(frames, capture_allocations=capture_allocations)
        assignment = context_target_assignment(context.slots[:, :tpar], target.slots[:, :tpar])
        target_states = align_to_assignment(target.states, assignment, track_dim=2).detach()
        target_allocations = (
            None
            if target.allocations is None
            else align_to_assignment(target.allocations, assignment, track_dim=2)
        )

        # Infer theta exactly once and preserve its attached path in all calls.
        causal_params = self.parameter_encoder(context.slots[:, :tpar])
        sources = context.states[:, tpar - 1 :].flatten(0, 1)
        targets = target_states[:, tpar:].flatten(0, 1)
        params = causal_params.repeat_interleave(transitions, dim=0)
        episode_keys = self.predictor.sample_track_keys(batch)
        keys = episode_keys.repeat_interleave(transitions, dim=0)
        out = self.predictor(sources, params, track_keys=keys)

        endpoint = endpoint_target = sampled_offsets = intermediate = None
        if rollout_t2_offsets is not None:
            if rollout_t2_offsets.ndim != 2 or rollout_t2_offsets.shape[0] != batch:
                raise ValueError("rollout_t2_offsets must have shape (B, W)")
            if rollout_t2_offsets.dtype == torch.bool or rollout_t2_offsets.is_floating_point():
                raise ValueError("rollout offsets must be integer indices")
            if num_rollout_t2_anchors not in (0, rollout_t2_offsets.shape[1]):
                raise ValueError(
                    "num_rollout_t2_anchors must be zero or match explicit offset width"
                )
            sampled_offsets = rollout_t2_offsets.to(device=frames.device, dtype=torch.long)
        elif num_rollout_t2_anchors > 0:
            sampled_offsets = sample_rollout_t2_offsets(
                batch,
                num_valid_rollout_t2_offsets(length, tpar),
                num_rollout_t2_anchors,
                device=frames.device,
            )
        if sampled_offsets is not None:
            endpoint, endpoint_target, intermediate = self.rollout_t2_from_offsets(
                context.states, target_states, tpar, causal_params, sampled_offsets, episode_keys
            )

        return VisualToVisualOutput(
            prediction=out.prediction,
            target=targets,
            causal_params=causal_params,
            path_matrix=out.path_matrix,
            sparsity=out.sparsity,
            logit_penalty=out.logit_penalty,
            mean_abs_logit=out.mean_abs_logit,
            mean_gate_probability=out.mean_gate_probability,
            gate_entropy=out.gate_entropy,
            target_variance=_content_variance(target_states[:, tpar:]).detach(),
            context_states=context.states,
            target_states=target_states,
            target_assignment=assignment,
            episode_keys=episode_keys,
            rollout_t2_prediction=endpoint,
            rollout_t2_target=endpoint_target,
            rollout_t2_offsets=sampled_offsets,
            rollout_t2_intermediate=intermediate,
            context_allocations=context.allocations,
            target_allocations=target_allocations,
        )

    def rollout_t2_from_offsets(
        self,
        context_states: Float[Tensor, "b t n s"],
        target_states: Float[Tensor, "b t n s"],
        context_len: int,
        causal_params: Float[Tensor, "b n 1"],
        offsets: Int[Tensor, "b w"],
        episode_keys: Float[Tensor, "b n d"],
    ) -> tuple[Float[Tensor, "b w n s"], Float[Tensor, "b w n s"], Float[Tensor, "b w n s"]]:
        """Two attached transition calls from each observed online anchor.

        The target endpoint is the already aligned, detached EMA state. Path
        and logit penalties remain those of the teacher-forced call alone.
        """
        batch = context_states.shape[0]
        valid = num_valid_rollout_t2_offsets(target_states.shape[1], context_len)
        if offsets.ndim != 2 or offsets.shape[0] != batch or offsets.shape[1] < 1:
            raise ValueError(f"offsets must be non-empty (B, W), got {tuple(offsets.shape)}")
        if offsets.dtype == torch.bool or offsets.is_floating_point():
            raise ValueError("rollout offsets must be integer indices")
        offsets = offsets.to(device=context_states.device, dtype=torch.long)
        if bool((offsets < 0).any()) or bool((offsets >= valid).any()):
            raise ValueError(f"rollout offsets must lie in [0, {valid - 1}]")
        sorted_offsets = offsets.sort(dim=1).values
        if sorted_offsets.shape[1] > 1 and bool(
            (sorted_offsets[:, 1:] == sorted_offsets[:, :-1]).any()
        ):
            raise ValueError("rollout offsets must be distinct within every episode")
        windows = offsets.shape[1]
        episode_index = torch.arange(batch, device=context_states.device)[:, None]
        anchor_indices = context_len - 1 + offsets
        anchors = context_states[episode_index, anchor_indices].flatten(0, 1)
        params = causal_params.repeat_interleave(windows, dim=0)
        keys = episode_keys.repeat_interleave(windows, dim=0)
        first = self.predictor(anchors, params, track_keys=keys).prediction
        second = self.predictor(first, params, track_keys=keys).prediction
        target = target_states[episode_index, anchor_indices + 2].detach()
        shape = (batch, windows, *context_states.shape[2:])
        return second.reshape(shape), target, first.reshape(shape)

    def rollout_for_evaluation(
        self,
        context_states: Float[Tensor, "b t n s"],
        target_states: Float[Tensor, "b t n s"],
        context_len: int,
        horizon: int,
        causal_params: Float[Tensor, "b n 1"],
        episode_keys: Float[Tensor, "b n d"],
    ) -> tuple[Float[Tensor, "b j n s"], Float[Tensor, "b j n s"]]:
        """Recursively predict from the last context state, without gradients."""
        if self.training:
            raise RuntimeError("evaluation rollout requires model.eval()")
        if torch.is_grad_enabled():
            raise RuntimeError("evaluation rollout requires torch.no_grad()")
        if horizon < 1:
            raise ValueError("evaluation horizon must be positive")
        if not 1 <= context_len <= context_states.shape[1]:
            raise ValueError("context_len leaves no online anchor")
        if context_len + horizon > target_states.shape[1]:
            raise ValueError("evaluation horizon runs past the episode")
        state = context_states[:, context_len - 1]
        predictions: list[Tensor] = []
        for _ in range(horizon):
            state = self.predictor(state, causal_params, track_keys=episode_keys).prediction
            predictions.append(state)
        return torch.stack(predictions, dim=1), target_states[
            :, context_len : context_len + horizon
        ]


def build_visual_to_visual(
    num_slots: int = 5,
    slot_size: int = 32,
    state_dim: int = 32,
    resolution: int = 64,
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
    ema_decay: float = 0.996,
    variance_floor: float = 1e-4,
) -> VisualToVisualModel:
    """Build the visual-to-visual model from plain config values (Hydra-friendly)."""
    return VisualToVisualModel(
        online=VisualStatePath(
            num_slots=num_slots,
            slot_size=slot_size,
            state_dim=state_dim,
            resolution=resolution,
        ),
        # Eq. 91's visual input map is just the state-to-state parameter encoder
        # with a slot-width input instead of a 4-dimensional state.
        parameter_encoder=ParameterEncoder(
            state_dim=slot_size,
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
            output_dim=state_dim,  # Eq. 118: decode into the learned state width
        ),
        ema_decay=ema_decay,
        variance_floor=variance_floor,
    )
