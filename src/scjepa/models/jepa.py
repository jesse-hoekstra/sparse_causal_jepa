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

import torch
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
    kinematic_state: Float[Tensor, "b n d"]  # chain anchors S_t, (B·C, N, d)
    path_matrix: Float[Tensor, "b t t"]
    sparsity: Float[Tensor, ""]
    logit_penalty: Float[Tensor, ""]


def resolve_chains(num_transitions: int, rollout_horizon: int | None) -> tuple[int, int]:
    """Validate the horizon; return (chain_len, num_chains).

    ``rollout_horizon=None`` -> one chain covering all K transitions (the
    paper-literal S_Tp). An int Tp chunks the K transitions into consecutive
    chains of exactly Tp autoregressive steps (my_paper p16's invariance holds
    from ANY intermediate state, so multiple anchors per episode are theory-
    consistent); Tp must divide K so every prediction has a target.
    """
    chain_len = rollout_horizon if rollout_horizon is not None else num_transitions
    if not 1 <= chain_len <= num_transitions:
        raise ValueError(
            f"rollout_horizon={chain_len} must be in [1, K={num_transitions}] "
            "(K = clip_len - context_len transitions)"
        )
    if num_transitions % chain_len != 0:
        raise ValueError(
            f"rollout_horizon={chain_len} must divide K={num_transitions} so every "
            "autoregressive step has a target (adjust context_len/clip_len/horizon)"
        )
    return chain_len, num_transitions // chain_len


def rollout_predictions(
    predictor: Spartan,
    anchors: Float[Tensor, "b c n d"],
    causal_params: Float[Tensor, "b n d"],
    aux: Float[Tensor, "b m da"] | None,
    chain_len: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Autoregressive rollout (my_paper p7/p16): S_Tp = [f, f∘f, ..., f^Tp].

    Each chain starts from a TRUE encoded state (its anchor) and then feeds the
    predictor its OWN previous prediction, reusing the same Ŝ^ph at every step
    — the structure the identifiability theory is premised on (and Baumgartner
    App. B.4). Note the anchor lives in kinematic-head space while fed-back
    predictions live in target-embedding space; the predictive loss ties the
    two spaces together (prediction ≈ target slots), which is what makes the
    composition f∘f well-typed as training converges.

    Returns flattened (B·K, ...) prediction and per-transition path matrices in
    trajectory order (chain c covers transitions c·Tp .. c·Tp+Tp-1), plus
    sparsity/logit penalties averaged over the Tp sequential SPARTAN calls.
    """
    batch, num_chains, num_slots, dim = anchors.shape
    state = anchors.reshape(batch * num_chains, num_slots, dim)
    flat_params = causal_params.repeat_interleave(num_chains, dim=0)
    flat_aux = aux.repeat_interleave(num_chains, dim=0) if aux is not None else None
    predictions: list[Tensor] = []
    path_matrices: list[Tensor] = []
    sparsities: list[Tensor] = []
    logit_penalties: list[Tensor] = []
    for _ in range(chain_len):
        out = predictor(state, flat_params, flat_aux)
        predictions.append(out.prediction)
        path_matrices.append(out.path_matrix)
        sparsities.append(out.sparsity)
        logit_penalties.append(out.logit_penalty)
        state = out.prediction  # f∘f: the model's own output is the next input
    # (B·C, Tp, ...) -> (B, C·Tp = K, ...): trajectory order, matching both the
    # flattened targets and the eval harness's per-transition contact slices.
    tokens = path_matrices[0].shape[-1]
    prediction = torch.stack(predictions, dim=1).reshape(batch, -1, num_slots, dim)
    path_matrix = torch.stack(path_matrices, dim=1).reshape(batch, -1, tokens, tokens)
    return (
        prediction.flatten(0, 1),
        path_matrix.flatten(0, 1),
        torch.stack(sparsities).mean(),
        torch.stack(logit_penalties).mean(),
    )


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
        rollout_horizon: int | None = None,
    ) -> JepaOutput:
        """Autoregressive rollout training (D16): one Ŝ^ph, chained predictions.

        Ŝ^ph is pooled ONCE from the first ``context_len`` steps of the slot
        history. The remaining K = L - context_len transitions are covered by
        autoregressive chains (my_paper p7: S_Tp = [S_t, f(S_t, Ŝ^ph),
        f∘f(S_t, Ŝ^ph), ...]): each chain is anchored at a TRUE encoded state
        and then feeds its own predictions back, reusing the same Ŝ^ph —
        mass-blind dynamics can no longer satisfy the constraint one forgiven
        step at a time (the v2 empty-graph failure). ``rollout_horizon=None``
        -> one chain over all K transitions; Tp chunks K into K/Tp chains.
        The context encoder runs once over frames[:, :-1] (its recurrence
        makes slots at t a function of frames <= t, so slicing the one pass is
        exact); the single-frame target encoder (D9) embeds every target frame.
        ``context_len=None`` -> L-1 (K = 1: one single-step chain).
        Flattened outputs (B·K, N, d); causal_params stays (B, N, d);
        kinematic_state carries the chain anchors (B·C, N, d).
        """
        if frames.ndim != 5 or frames.shape[1] < 2:
            raise ValueError(f"expected (B, L>=2, C, H, W), got {tuple(frames.shape)}")
        length = frames.shape[1]
        th = context_len if context_len is not None else length - 1
        if not 1 <= th < length:
            raise ValueError(f"context_len={th} must be in [1, L-1={length - 1}]")
        chain_len, num_chains = resolve_chains(length - th, rollout_horizon)
        all_context_slots = self.context_encoder(frames[:, :-1])  # (B, L-1, N, d)
        context_slots = all_context_slots[:, :th]
        causal_params = self.pooling(context_slots)  # (B, N, d) — pooled ONCE
        anchor_steps = [th - 1 + c * chain_len for c in range(num_chains)]
        anchors = self.kinematic_head.project(all_context_slots[:, anchor_steps])
        target_frames = frames[:, th:].flatten(0, 1).unsqueeze(1)  # (B*K, 1, C, H, W)
        target_slots = self.target_encoder(target_frames).squeeze(1)  # (B*K, N, d)
        prediction, path_matrix, sparsity, logit_penalty = rollout_predictions(
            self.predictor, anchors, causal_params, aux, chain_len
        )
        return JepaOutput(
            prediction=prediction,
            target_slots=target_slots,
            context_slots=context_slots,
            causal_params=causal_params,
            kinematic_state=anchors.flatten(0, 1),
            path_matrix=path_matrix,
            sparsity=sparsity,
            logit_penalty=logit_penalty,
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
    spartan_dense: bool = False,
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
            dense=spartan_dense,
        ),
    )


__all__ = [
    "JepaOutput",
    "SCJepa",
    "build_scjepa",
    "resolve_chains",
    "rollout_predictions",
]
