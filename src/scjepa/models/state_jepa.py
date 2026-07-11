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
from scjepa.models.jepa import JepaOutput
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
    ) -> JepaOutput:
        """Sliding-window training (D15): one Ŝ^ph, K single-step predictions.

        Ŝ^ph is pooled ONCE from the first ``context_len`` frames and held
        fixed while the window slides: for every t in [context_len-1, L-2] the
        model predicts frame t+1 from (S_t, Ŝ^ph) — K = L - context_len
        single-step predictions (D6 per prediction) sharing one parameter
        estimate, operationalizing the entanglement-map-independence premise.
        ``context_len=None`` means L-1 (K = 1, the legacy single-transition
        behavior). Flattened outputs: prediction/target/kinematic are
        (B·K, N, d); causal_params stays (B, N, d) — one estimate per episode.
        """
        if states.ndim != 4 or states.shape[1] < 2:
            raise ValueError(f"expected (B, L>=2, N, k), got {tuple(states.shape)}")
        length = states.shape[1]
        th = context_len if context_len is not None else length - 1
        if not 1 <= th < length:
            raise ValueError(f"context_len={th} must be in [1, L-1={length - 1}]")
        context_slots = self.context_embed(states[:, :th])  # (B, Th, N, d)
        causal_params = self.pooling(context_slots)  # (B, N, d) — pooled ONCE
        # S_t for every window position t = th-1 .. L-2 (memoryless embeds).
        current = self.context_embed(states[:, th - 1 : length - 1])  # (B, K, N, d)
        kinematic_state = self.kinematic_head.project(current)  # (B, K, N, d)
        target_slots = self.target_embed(states[:, th:])  # (B, K, N, d)
        num_transitions = target_slots.shape[1]
        flat_state = kinematic_state.flatten(0, 1)
        flat_params = causal_params.repeat_interleave(num_transitions, dim=0)
        flat_aux = aux.repeat_interleave(num_transitions, dim=0) if aux is not None else None
        predicted = self.predictor(flat_state, flat_params, flat_aux)
        return JepaOutput(
            prediction=predicted.prediction,
            target_slots=target_slots.flatten(0, 1),
            context_slots=context_slots,
            causal_params=causal_params,
            kinematic_state=flat_state,
            path_matrix=predicted.path_matrix,
            sparsity=predicted.sparsity,
            logit_penalty=predicted.logit_penalty,
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
        ),
    )


__all__ = ["StateJepa", "build_state_jepa"]
