"""StateJepa: the GT-embedding diagnostic regime (D11).

Same architecture as ``SCJepa`` downstream of the encoders — channel split (D4),
SPARTAN, identical ``JepaOutput`` contract — but the SAVi encoders are replaced
by shared per-object linear embeddings of the ground-truth kinematic states.
Purpose (my_paper experiment list item 1; SPARTAN App. D): isolate the
channel-split + SPARTAN identifiability question from encoder quality, with
slot i ≡ object i by construction (no alignment problem), and calibrate the
Lagrangian target τ.

Batch contract: ``states (B, Th+1, N, k)`` (bounce: k = 4, [x, y, vx, vy]);
the last step is the prediction target, exactly mirroring the frames contract.
"""

from jaxtyping import Float
from torch import Tensor, nn

from scjepa.models.channel_split import (
    AttnPooling,
    CrossSlotAttnPooling,
    KinematicHead,
    build_pooling,
)
from scjepa.models.jepa import JepaOutput, resolve_chains, rollout_predictions
from scjepa.models.spartan import Spartan


class StateJepa(nn.Module):
    """JEPA over ground-truth object states instead of pixels (diagnostic)."""

    def __init__(
        self,
        context_embed: nn.Module,
        target_embed: nn.Module,
        pooling: AttnPooling | CrossSlotAttnPooling,
        kinematic_head: KinematicHead,
        predictor: Spartan,
    ) -> None:
        """Compose the modules; embeddings are per-object, shared across objects."""
        super().__init__()
        self.context_embed = context_embed
        self.target_embed = target_embed
        self.pooling = pooling
        self.kinematic_head = kinematic_head
        self.predictor = predictor

    def forward(
        self,
        states: Float[Tensor, "b length n k"],
        aux: Float[Tensor, "b m da"] | None = None,
        context_len: int | None = None,
        rollout_horizon: int | None = None,
    ) -> JepaOutput:
        """Autoregressive rollout training (D16): one Ŝ^ph, chained predictions.

        Ŝ^ph is pooled ONCE from the first ``context_len`` frames. The
        remaining K = L - context_len transitions are covered by autoregressive
        chains (my_paper p7: S_Tp = [S_t, f(S_t, Ŝ^ph), f∘f(S_t, Ŝ^ph), ...]):
        each chain is anchored at a TRUE embedded state and then feeds its own
        predictions back, reusing the same Ŝ^ph at every step — the structure
        the identifiability theory (p16) is premised on. ``rollout_horizon``:
        None -> one chain over all K transitions; Tp chunks K into K/Tp chains.
        ``context_len=None`` means L-1 (K = 1: one single-step chain).
        Flattened outputs: prediction/target are (B·K, N, d); causal_params
        stays (B, N, d); kinematic_state carries the anchors (B·C, N, d).
        """
        if states.ndim != 4 or states.shape[1] < 2:
            raise ValueError(f"expected (B, L>=2, N, k), got {tuple(states.shape)}")
        length = states.shape[1]
        th = context_len if context_len is not None else length - 1
        if not 1 <= th < length:
            raise ValueError(f"context_len={th} must be in [1, L-1={length - 1}]")
        chain_len, num_chains = resolve_chains(length - th, rollout_horizon)
        context_slots = self.context_embed(states[:, :th])  # (B, Th, N, d)
        causal_params = self.pooling(context_slots)  # (B, N, d) — pooled ONCE
        # Chain anchors: TRUE states at t = th-1, th-1+Tp, ... (memoryless embeds).
        anchor_steps = [th - 1 + c * chain_len for c in range(num_chains)]
        anchors = self.kinematic_head.project(self.context_embed(states[:, anchor_steps]))
        target_slots = self.target_embed(states[:, th:])  # (B, K, N, d)
        prediction, path_matrix, sparsity, logit_penalty = rollout_predictions(
            self.predictor, anchors, causal_params, aux, chain_len
        )
        return JepaOutput(
            prediction=prediction,
            target_slots=target_slots.flatten(0, 1),
            context_slots=context_slots,
            causal_params=causal_params,
            kinematic_state=anchors.flatten(0, 1),
            path_matrix=path_matrix,
            sparsity=sparsity,
            logit_penalty=logit_penalty,
        )


def build_state_jepa(
    state_dim: int = 4,
    slot_size: int = 32,
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
) -> StateJepa:
    """Build the GT-embedding variant from plain config values.

    Context and target embeddings are separate linear maps (mirroring the
    separate encoders of the vision regime, D9 default).
    """
    return StateJepa(
        context_embed=nn.Linear(state_dim, slot_size),
        target_embed=nn.Linear(state_dim, slot_size),
        pooling=build_pooling(pooling_type, slot_size, pooling_heads, max_history),
        kinematic_head=KinematicHead(slot_size=slot_size),
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


__all__ = ["StateJepa", "build_state_jepa"]
