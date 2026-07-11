"""SCJepa: the full architecture of my_paper.pdf Fig. 1 as one composite module.

Wires together the already-tested parts — context SAVi encoder over the history,
single-frame target SAVi encoder (D9), channel split (D4), SPARTAN predictor —
and returns everything the training objective needs. No loss computation here;
that lives in ``scjepa.losses`` / the training loop (D6).

Data contract (batch): frames ``(B, Th+1, C, H, W)`` — the first Th frames are
the context window, the last frame is the prediction target.

Symbol table:
    frames[:, :-1]  history          (B, Th, C, H, W) → context encoder
    frames[:, -1:]  future frame     (B, 1, C, H, W)  → target encoder
    S̃               context slots    (B, Th, N, d)
    Ŝ^ph            causal params    (B, N, d)
    S_t             kinematic state  (B, N, d)
    S_{t+1}         target slots     (B, N, d)   raw target-encoder slots (D9)
    Ŝ_{t+1}         prediction       (B, N, d)
"""

from typing import NamedTuple

from jaxtyping import Float
from torch import Tensor, nn

from scjepa.models.channel_split import (
    AttnPooling,
    CrossSlotAttnPooling,
    KinematicHead,
    build_pooling,
)
from scjepa.models.savi import SAViEncoder
from scjepa.models.spartan import Spartan


class JepaOutput(NamedTuple):
    """One forward pass: predictions, targets, and everything the losses need."""

    prediction: Float[Tensor, "b n d"]
    target_slots: Float[Tensor, "b n d"]
    context_slots: Float[Tensor, "b th n d"]
    causal_params: Float[Tensor, "b n d"]
    kinematic_state: Float[Tensor, "b n d"]
    path_matrix: Float[Tensor, "b t t"]
    sparsity: Float[Tensor, ""]
    logit_penalty: Float[Tensor, ""]


class SCJepa(nn.Module):
    """Causally Inducing JEPA using a SPARTAN (my_paper.pdf Fig. 1).

    Joint training, no EMA, no stop-gradient (D7): both encoders, both heads,
    and the predictor are trained by ONE optimizer step; collapse prevention is
    the regularizer applied outside this module (D3).
    """

    def __init__(
        self,
        context_encoder: SAViEncoder,
        target_encoder: SAViEncoder,
        pooling: AttnPooling | CrossSlotAttnPooling,
        kinematic_head: KinematicHead,
        predictor: Spartan,
    ) -> None:
        """Compose the five submodules (built/configured by the caller)."""
        super().__init__()
        if not target_encoder.single_frame:
            raise ValueError("target encoder must be built with single_frame=True (D9)")
        if context_encoder.single_frame:
            raise ValueError("context encoder must be built with single_frame=False")
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.pooling = pooling
        self.kinematic_head = kinematic_head
        self.predictor = predictor

    def forward(
        self,
        frames: Float[Tensor, "b length c h w"],
        aux: Float[Tensor, "b m da"] | None = None,
        context_len: int | None = None,
    ) -> JepaOutput:
        """Sliding-window training (D15): one Ŝ^ph, K single-step predictions.

        Ŝ^ph is pooled ONCE from the first ``context_len`` steps of the slot
        history and held fixed while the window slides: for every t in
        [context_len-1, L-2] the model predicts frame t+1's target slots from
        (S_t, Ŝ^ph). The context encoder runs once over frames[:, :-1] (its
        recurrence makes slots at t a function of frames <= t, so slicing the
        one pass is exact); the single-frame target encoder (D9) runs on each
        predicted frame. ``context_len=None`` -> L-1 (K = 1, legacy behavior).
        Flattened outputs as in StateJepa; causal_params stays (B, N, d).
        """
        if frames.ndim != 5 or frames.shape[1] < 2:
            raise ValueError(f"expected (B, L>=2, C, H, W), got {tuple(frames.shape)}")
        length = frames.shape[1]
        th = context_len if context_len is not None else length - 1
        if not 1 <= th < length:
            raise ValueError(f"context_len={th} must be in [1, L-1={length - 1}]")
        all_context_slots = self.context_encoder(frames[:, :-1])  # (B, L-1, N, d)
        context_slots = all_context_slots[:, :th]
        causal_params = self.pooling(context_slots)  # (B, N, d) — pooled ONCE
        current = all_context_slots[:, th - 1 :]  # (B, K, N, d): slots at t = th-1..L-2
        kinematic_state = self.kinematic_head.project(current)
        num_transitions = current.shape[1]
        target_frames = frames[:, th:].flatten(0, 1).unsqueeze(1)  # (B*K, 1, C, H, W)
        target_slots = self.target_encoder(target_frames).squeeze(1)  # (B*K, N, d)
        flat_state = kinematic_state.flatten(0, 1)
        flat_params = causal_params.repeat_interleave(num_transitions, dim=0)
        flat_aux = aux.repeat_interleave(num_transitions, dim=0) if aux is not None else None
        predicted = self.predictor(flat_state, flat_params, flat_aux)
        return JepaOutput(
            prediction=predicted.prediction,
            target_slots=target_slots,
            context_slots=context_slots,
            causal_params=causal_params,
            kinematic_state=flat_state,
            path_matrix=predicted.path_matrix,
            sparsity=predicted.sparsity,
            logit_penalty=predicted.logit_penalty,
        )


def build_scjepa(
    resolution: int = 64,
    num_slots: int = 7,
    slot_size: int = 128,
    slot_mlp_size: int = 256,
    num_iterations: int = 2,
    enc_channels: tuple[int, ...] = (3, 64, 64, 64, 64),
    enc_out_channels: int = 128,
    pooling_heads: int = 4,
    pooling_type: str = "cross_slot",  # D14 default; "per_slot" = D4 ablation
    max_history: int = 64,
    spartan_layers: int = 3,
    spartan_embed_dim: int | None = 512,
    spartan_mlp_hidden: int = 512,
    spartan_mlp_layers: int = 3,
    spartan_temperature: float = 1.0,
    aux_dim: int | None = None,
) -> SCJepa:
    """Build the full model from plain config values (Hydra-friendly).

    Both encoders are independent instances with their own weights (D9 default);
    the target encoder is single-frame with no slot predictor.
    """
    context_encoder = SAViEncoder(
        resolution=(resolution, resolution),
        num_slots=num_slots,
        slot_size=slot_size,
        slot_mlp_size=slot_mlp_size,
        num_iterations=num_iterations,
        enc_channels=tuple(enc_channels),
        enc_out_channels=enc_out_channels,
    )
    target_encoder = SAViEncoder(
        resolution=(resolution, resolution),
        num_slots=num_slots,
        slot_size=slot_size,
        slot_mlp_size=slot_mlp_size,
        num_iterations=num_iterations,
        enc_channels=tuple(enc_channels),
        enc_out_channels=enc_out_channels,
        single_frame=True,
    )
    return SCJepa(
        context_encoder=context_encoder,
        target_encoder=target_encoder,
        pooling=build_pooling(pooling_type, slot_size, pooling_heads, max_history),
        kinematic_head=KinematicHead(slot_size=slot_size),  # state_size = d (D9)
        predictor=Spartan(
            slot_size=slot_size,
            num_layers=spartan_layers,
            embed_dim=spartan_embed_dim,
            mlp_hidden_size=spartan_mlp_hidden,
            mlp_num_layers=spartan_mlp_layers,
            temperature=spartan_temperature,
            aux_dim=aux_dim,
        ),
    )


__all__ = ["JepaOutput", "SCJepa", "build_scjepa"]
