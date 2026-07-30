"""Visual-to-visual regime: frames in, EMA-encoded next frame out (fully self-supervised).

The controlled change from the visual-to-state regime is the TARGET, and only the target. The
predictor-side construction is unchanged (Eq. 107): a causal SAVi encoder and a
shared state head produce S^ctx_t from X_{0:t}, the parameter encoder produces one
scalar per track from the first Tpar frames, and SPARTAN predicts one step ahead.
What changes is that the prediction is compared against a stopped-gradient
representation of X_{0:t+1} produced by an exponential-moving-average copy of the
online state path (Eqs. 108-114), instead of against the true physical state.

Consequences that shape this module:

* **No matching, anywhere in training.** The EMA copy is initialized from the
  online encoder and updated component-wise, so it never permutes slot rows:
  predicted row i is compared directly with target row i (Eq. 116). §6.6 forbids
  per-frame Hungarian matching or a learned context-to-target assignment here,
  because under EMA row correspondence it is unnecessary AND it could conceal a
  disagreement between the two recurrent trackers. The Hungarian machinery in
  ``scjepa.losses.alignment`` belongs to the visual-to-state regime's true-state target and to
  The visual-to-visual regime's EVALUATION only.
* **Only the state path has an EMA twin.** The parameter encoder, SPARTAN, the
  track keys and the gates are predictor-side and have no EMA copy (§6.6).
* **The dual sees a normalized scalar.** Because the learned target's scale can
  drift, the GECO controller is fed L_pred divided by the detached target content
  variance (Eqs. 122/123), while the gradient objective (Eq. 121) keeps the raw
  latent MSE. This normalization exists ONLY for the constraint.
* **Collapse is an empirical requirement.** The EMA asymmetry does not make a
  constant-representation fixed point impossible, so Eq. 124's diagnostics are
  computed every step and a run with vanishing content variance or degenerate
  effective rank is rejected as collapsed rather than reported.

During training, true states, masses and contact graphs are excluded from the
model, the objective and every selection decision; they are used only for
held-out evaluation.
"""

import copy
from typing import NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor, nn

from scjepa.models.parameter_encoder import ParameterEncoder
from scjepa.models.spartan import Spartan
from scjepa.models.visual import VisualStatePath

__all__ = ["VisualToVisualModel", "VisualToVisualOutput", "build_visual_to_visual"]


class VisualToVisualOutput(NamedTuple):
    """One forward pass over an episode batch, flattened to (B*K, ...) rows."""

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
    """V_tgt (Eq. 122): mean per-coordinate variance of the EMA target."""
    context_states: Float[Tensor, "b t n s"]
    """Online latent states, kept for the Eq. 124 collapse diagnostics."""
    target_states: Float[Tensor, "b t n s"]
    """EMA latent states, kept for the Eq. 124 collapse diagnostics."""
    rollout_prediction: Float[Tensor, "b j n s"] | None = None
    """Hybrid Eqs. 33-34: the K autoregressive prefixes. None when disabled."""
    rollout_target: Float[Tensor, "b j n s"] | None = None
    """EMA targets S^tgt_{t+1..t+K} for those prefixes (stop-gradient)."""


def _content_variance(states: Tensor) -> Tensor:
    """Eq. 122: mean over tracks and coordinates of the variance across (episode, time).

    A collapsed representation drives this to zero, which is exactly why it also
    guards the constraint denominator.
    """
    return states.flatten(0, 1).var(dim=0, unbiased=False).mean()


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
        """Compose §6.6's model; the target path is cloned from ``online``.

        Args:
            online: The trainable visual state path chi = (psi, omega).
            parameter_encoder: P_eta over the CONTEXT slots (Eq. 106).
            predictor: SPARTAN with a d_s state interface and a d_s output head.
            ema_decay: tau_EMA of Eq. 111. Fixed before the confirmatory runs;
                §6.6 requires reporting a faster and a slower control.
            variance_floor: epsilon_var of Eq. 123.
        """
        super().__init__()
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        if variance_floor <= 0:
            raise ValueError("variance_floor must be positive")
        self.online = online
        # Eq. 110: the target is INITIALIZED FROM the online parameters, which is
        # what makes row ancestry shared and matching unnecessary (Eq. 116).
        self.target = copy.deepcopy(online)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.parameter_encoder = parameter_encoder
        self.predictor = predictor
        self.ema_decay = ema_decay
        self.variance_floor = variance_floor

    @torch.no_grad()
    def update_target(self) -> None:
        """Eq. 111, applied after each optimizer step on the online parameters.

        Buffers are copied outright rather than averaged: they carry no learned
        content here, and averaging them would desynchronize the two branches'
        architectures instead of their weights.
        """
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
        rollout_len: int | None = None,
    ) -> VisualToVisualOutput:
        """Run both branches once, then make the |I| one-step latent predictions.

        Args:
            frames: (B, T, 3, H, W) episode batch.
            context_len: Tpar (the visual-to-visual regime: 30). None -> T-1, giving K = 1.
            rollout_len: K for the hybrid autoregressive branch (Eqs. 33-35).
                None disables it. Unlike the visual-to-state regime this is
                well-typed: the predictor maps the learned state width to
                itself (Eq. 118), so f can be composed with itself.
        """
        if frames.ndim != 5 or frames.shape[1] < 2:
            raise ValueError(f"expected (B, T>=2, C, H, W), got {tuple(frames.shape)}")
        batch, length = frames.shape[0], frames.shape[1]
        tpar = context_len if context_len is not None else length - 1
        if not 1 <= tpar < length:
            raise ValueError(f"context_len={tpar} must be in [1, T-1={length - 1}]")
        transitions = length - tpar  # K

        # Eq. 115: each recurrence is run ONCE over the episode. The context
        # branch never sees the final frame; the target branch does, and its
        # prefix notation only expresses causal dependence.
        context = self.online(frames[:, : length - 1])
        with torch.no_grad():
            target = self.target(frames)

        # Eq. 106: theta-hat comes from the CONTEXT SLOTS (pre-state-head) over
        # the parameter window only. Frames after Tpar-1 must not affect it.
        causal_params = self.parameter_encoder(context.slots[:, :tpar])

        # I = {Tpar-1, ..., T-2}: source states S^ctx_t, targets S^tgt_{t+1}.
        sources = context.states[:, tpar - 1 :].flatten(0, 1)
        targets = target.states[:, tpar:].flatten(0, 1).detach()  # Eq. 114 stop-gradient
        params = causal_params.repeat_interleave(transitions, dim=0)
        # §6.4: one key per episode, reused for every transition of that episode
        # — and, below, for every step of that episode's rollout. Resampling
        # mid-chain would give each step a different episode-level permutation.
        episode_keys = self.predictor.sample_track_keys(batch)
        keys = episode_keys.repeat_interleave(transitions, dim=0)
        out = self.predictor(sources, params, track_keys=keys)

        rollout_prediction = rollout_target = None
        if rollout_len is not None:
            rollout_prediction, rollout_target = self._rollout(
                context.states, target.states, tpar, rollout_len, causal_params, episode_keys
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
            target_variance=_content_variance(target.states[:, tpar:]).detach(),
            context_states=context.states,
            target_states=target.states,
            rollout_prediction=rollout_prediction,
            rollout_target=rollout_target,
        )

    def _rollout(
        self,
        context_states: Float[Tensor, "b t n s"],
        target_states: Float[Tensor, "b t n s"],
        tpar: int,
        horizon: int,
        causal_params: Float[Tensor, "b n 1"],
        episode_keys: Float[Tensor, "b n d"],
    ) -> tuple[Float[Tensor, "b j n s"], Float[Tensor, "b j n s"]]:
        """Hybrid Eqs. 33-34 in the learned latent space.

            Ŝ^[0]_t := S^on_t,   Ŝ^[k]_{t+k} := f_gamma(Ŝ^[k-1]_{t+k-1}, θ̂)

        The chain is anchored at t = Tpar-1 on the ONLINE path — Eq. 33's
        S^on_t — and supervised against the EMA target path, which has already
        been run over every frame, so no extra encoder pass is needed. The same
        θ̂ AND the same episode track keys enter at every step.

        Well-typed only because Eq. 118 gives the predictor ``output_dim =
        state_dim``: f maps the learned state width to itself. The
        visual-to-state regime decodes to the raw 4-dim state instead
        (Eq. 95), so it has no composable f and no rollout branch.
        """
        length = context_states.shape[1]
        # The online path never sees the final frame, so its states run to T-2;
        # the anchor and every fed-back state come from that branch.
        if tpar + horizon > target_states.shape[1]:
            raise ValueError(
                f"rollout_len={horizon} runs past the episode: need Tpar-1+K <= T-1, "
                f"got {tpar - 1}+{horizon} > {target_states.shape[1] - 1} (Tpar={tpar})"
            )
        if tpar - 1 >= length:
            raise ValueError(f"context_len={tpar} leaves no online anchor (T-1={length})")

        state = context_states[:, tpar - 1]  # Ŝ^[0] := S^on_t (Eq. 33)
        predictions: list[Tensor] = []
        for _ in range(horizon):
            state = self.predictor(state, causal_params, track_keys=episode_keys).prediction
            predictions.append(state)
        # Eq. 114 stop-gradient, exactly as the teacher-forced targets carry it.
        target = target_states[:, tpar : tpar + horizon].detach()
        return torch.stack(predictions, dim=1), target


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
