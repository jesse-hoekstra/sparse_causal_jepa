"""Visual-to-state regime: frames in, true next physical state out.

The middle rung of the ladder: the predictor-side observation becomes the causal
frame history X_{0:t}, while the prediction target stays the fixed physical state
Z_{t+1} (Eq. 85). Because the target is fixed data, this experiment has

* **no target encoder, no EMA teacher, no learned target geometry, and no
  target-side representation regularizer** (§6.5, stated verbatim), and
* **no representation-collapse regularizer at all** — a constant context state
  and a constant parameter representation cannot predict episode-varying future
  physical states, so collapse is not a trivial optimum here the way it is in
  The visual-to-visual regime.

That makes it the visual-to-visual regime minus the whole target branch, with two further
differences that are easy to miss:

1. SPARTAN decodes into the RAW 4-dimensional target space (Eq. 95), not the
   learned state width. Its input is the latent d_s state and its output is
   physical, so the input and output inhabit different spaces — which is exactly
   why §6.5 excludes open-loop rollout from the primary protocol.
2. Predictions are in visual-track order and targets are in simulator-row order,
   so a detached trajectory-level assignment is required before any loss is
   taken (Eqs. 98-100). The visual-to-visual regime needs no such thing, because EMA row
   ancestry aligns its two branches by construction.

The true states enter training ONLY as prediction targets and to define that
assignment; they are never predictor-side inputs and never supervise the latent
state directly. Masses and contact graphs stay out of the objective entirely.
"""

from typing import NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor, nn

from scjepa.models.parameter_encoder import ParameterEncoder
from scjepa.models.spartan import Spartan
from scjepa.models.visual import VisualStatePath

__all__ = ["VisualToStateModel", "VisualToStateOutput", "build_visual_to_state"]


class VisualToStateOutput(NamedTuple):
    """One forward pass over an episode batch.

    ``prediction`` and ``target`` keep the (B, K, N, 4) episode structure rather
    than flattening to rows, because the trajectory assignment of Eq. 99 is
    per-EPISODE and needs the whole window at once.
    """

    prediction: Float[Tensor, "b k n 4"]
    target: Float[Tensor, "b k n 4"]
    causal_params: Float[Tensor, "b n 1"]
    path_matrix: Float[Tensor, "bk m m"]
    sparsity: Float[Tensor, ""]
    logit_penalty: Float[Tensor, ""]
    mean_abs_logit: Float[Tensor, ""]
    mean_gate_probability: Float[Tensor, ""]
    gate_entropy: Float[Tensor, ""]
    context_states: Float[Tensor, "b t n s"]
    """Latent states S (Eq. 89), kept for the held-out probe diagnostics."""


class VisualToStateModel(nn.Module):
    """SAVi + state head + parameter encoder + SPARTAN, decoding to raw states."""

    coordinate_scales: Tensor

    def __init__(
        self,
        visual: VisualStatePath,
        parameter_encoder: ParameterEncoder,
        predictor: Spartan,
        state_dim: int = 4,
    ) -> None:
        """Compose §6.5's model. There is deliberately no second visual branch."""
        super().__init__()
        self.visual = visual
        self.parameter_encoder = parameter_encoder
        self.predictor = predictor
        # sigma_a of Eq. 98, frozen from the training split. It lives on the
        # model, not the trainer, so it rides along in the checkpoint: the
        # held-out constraint must be computed under the SAME assignment the
        # dual was trained against, or tau_2 calibrates against a quantity the
        # run never saw. Filled by VisualToStateTrainer; ones until then.
        self.register_buffer("coordinate_scales", torch.ones(state_dim))

    def forward(
        self,
        frames: Float[Tensor, "b t c h w"],
        states: Float[Tensor, "b t n k"],
        context_len: int | None = None,
    ) -> VisualToStateOutput:
        """Encode frames, infer theta-hat, predict the |I| next physical states.

        Args:
            frames: (B, T, 3, H, W) episode batch — the ONLY predictor input.
            states: (B, T, N, 4) true states. Used solely to carve out the
                targets; they are not fed to the encoder, the parameter encoder
                or SPARTAN, and they arrive in simulator-row order, so the
                caller must align them before taking a loss.
            context_len: Tpar (the visual-to-state regime: 30). None -> T-1, giving K = 1.
        """
        if frames.ndim != 5 or frames.shape[1] < 2:
            raise ValueError(f"expected frames (B, T>=2, C, H, W), got {tuple(frames.shape)}")
        if states.ndim != 4 or states.shape[:2] != frames.shape[:2]:
            raise ValueError(
                f"states {tuple(states.shape)} must be (B, T, N, k) matching frames "
                f"{tuple(frames.shape[:2])}"
            )
        batch, length = frames.shape[0], frames.shape[1]
        tpar = context_len if context_len is not None else length - 1
        if not 1 <= tpar < length:
            raise ValueError(f"context_len={tpar} must be in [1, T-1={length - 1}]")
        transitions = length - tpar  # K

        # The source state for transition t -> t+1 may depend only on X_{0:t},
        # so the encoder never sees the final frame (Eq. 86 + §6.5's causality
        # requirement). Running it over the prefix is what enforces that.
        context = self.visual(frames[:, : length - 1])
        # Eq. 90: theta-hat from the parameter window's SLOTS only. Frames after
        # Tpar-1 must leave it unchanged — a §6.5 continuation-gate requirement.
        causal_params = self.parameter_encoder(context.slots[:, :tpar])

        sources = context.states[:, tpar - 1 :].flatten(0, 1)
        params = causal_params.repeat_interleave(transitions, dim=0)
        # §6.4: one key per episode, shared by that track's state and parameter
        # token, reused across the episode's transitions.
        keys = self.predictor.sample_track_keys(batch).repeat_interleave(transitions, dim=0)
        out = self.predictor(sources, params, track_keys=keys)

        num_slots = causal_params.shape[1]
        return VisualToStateOutput(
            prediction=out.prediction.unflatten(0, (batch, transitions)),
            target=states[:, tpar:],
            causal_params=causal_params,
            path_matrix=out.path_matrix,
            sparsity=out.sparsity,
            logit_penalty=out.logit_penalty,
            mean_abs_logit=out.mean_abs_logit,
            mean_gate_probability=out.mean_gate_probability,
            gate_entropy=out.gate_entropy,
            context_states=context.states[:, :, :num_slots],
        )


def build_visual_to_state(
    state_dim: int = 4,
    num_slots: int = 5,
    slot_size: int = 32,
    latent_state_dim: int = 32,
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
) -> VisualToStateModel:
    """Build the visual-to-state model from plain config values (Hydra-friendly)."""
    return VisualToStateModel(
        visual=VisualStatePath(
            num_slots=num_slots,
            slot_size=slot_size,
            state_dim=latent_state_dim,
            resolution=resolution,
        ),
        parameter_encoder=ParameterEncoder(
            state_dim=slot_size,  # Eq. 91: the visual-slot input map
            dim=param_encoder_dim,
            num_heads=param_encoder_heads,
            max_history=max_history,
        ),
        state_dim=state_dim,
        predictor=Spartan(
            state_dim=latent_state_dim,  # Eq. 94: W_S in R^{D_sp x d_s}
            param_dim=1,
            num_slots=num_slots,
            num_layers=spartan_layers,
            embed_dim=spartan_embed_dim,
            mlp_hidden_size=spartan_mlp_hidden,
            mlp_num_layers=spartan_mlp_layers,
            temperature=spartan_temperature,
            dense=spartan_dense,
            identity=spartan_identity,
            output_dim=state_dim,  # Eq. 95: decode into the RAW 4-dim target
        ),
    )
