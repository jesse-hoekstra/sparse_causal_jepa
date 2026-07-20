"""Channel split (D4): slot history → causal parameters Ŝ^ph and kinematic state S_t.

Implements the exact spec in docs/decisions.md D4:

- ``AttnPooling`` — per-slot temporal attention pooling (a PMA-style block, Lee et
  al. 2019), weights shared across slots, single learned query, learned temporal
  positional encodings; collapses the time axis inside the attention:
  ``(B, Th, N, d) → Ŝ^ph ∈ (B, N, d)``. Strictly slot-local: no cross-slot mixing —
  relational effects are SPARTAN's job.
- ``TrackAwareAttnPooling`` — first forms one temporal summary for each tracked
  object, then mixes those summaries with permutation-equivariant set attention
  and projects each result to a configurable parameter dimension. Unlike
  ``CrossSlotAttnPooling``, object identity is retained throughout the temporal
  stage and no final-state query is added as a residual shortcut.
- ``KinematicHead`` — linear layer on the *last-step* slots (which have seen all
  frames via SAVi's recurrence): ``(B, Th, N, d) → S_t ∈ (B, N, d)``.

Symbol table (paper ↔ code):
    S̃      slot history        (B, Th, N, d)   ``slot_history``
    p_k    temporal PE          (1, Th, d)      ``temporal_pe[:, :Th]``
    q      learned query        (1, 1, d)       ``query``
    Ŝ^ph   causal parameters    (B, N, d)       ``AttnPooling.forward`` output
    S_t    kinematic state      (B, N, d)       ``KinematicHead.forward`` output
"""

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


class AttnPooling(nn.Module):
    """Per-slot temporal attention pooling: slot history → causal parameters Ŝ^ph.

    For each slot i (batched, parameters shared across slots), a single learned
    query attends over that slot's Th timestep embeddings (+ temporal PE); the
    attention output passes through the standard PMA residual/LayerNorm/MLP
    block. Time-invariant per-object parameters (mass, charge, friction) are only
    observable from multi-frame behavior, hence pooling over the whole horizon.
    """

    def __init__(
        self,
        slot_size: int,
        num_heads: int = 4,
        mlp_hidden_size: int | None = None,
        max_history: int = 64,
    ) -> None:
        """Build the pooling block.

        Args:
            slot_size: d, slot embedding dimension (also the output dimension).
            num_heads: Attention heads (must divide ``slot_size``).
            mlp_hidden_size: Hidden width of the PMA feed-forward block
                (default ``2 * slot_size``).
            max_history: Maximum supported Th (length of the learned temporal PE
                table; forward slices the first Th entries).
        """
        super().__init__()
        if slot_size % num_heads != 0:
            raise ValueError(f"num_heads={num_heads} must divide slot_size={slot_size}")
        if mlp_hidden_size is None:
            mlp_hidden_size = 2 * slot_size
        self.max_history = max_history

        # Single learned query, shared across slots (D4).
        self.query = nn.Parameter(torch.empty(1, 1, slot_size))
        nn.init.normal_(self.query, std=slot_size**-0.5)
        # Learned temporal positional encodings p_1 … p_max_history (D4).
        self.temporal_pe = nn.Parameter(torch.empty(1, max_history, slot_size))
        nn.init.normal_(self.temporal_pe, std=0.02)

        self.mha = nn.MultiheadAttention(slot_size, num_heads, batch_first=True)
        self.norm_attn = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, slot_size),
        )

    def _pool(
        self, slot_history: Float[Tensor, "b t n d"]
    ) -> tuple[Float[Tensor, "b n d"], Float[Tensor, "b n t"]]:
        """Run the PMA block; return pooled slots and time-attention weights."""
        if slot_history.ndim != 4:
            raise ValueError(f"expected (B, Th, N, d), got shape {tuple(slot_history.shape)}")
        b, t = slot_history.shape[0], slot_history.shape[1]
        if t > self.max_history:
            raise ValueError(f"history length {t} exceeds max_history={self.max_history}")

        # Fold slots into the batch: per-slot pooling with shared weights.
        tokens = rearrange(slot_history, "b t n d -> (b n) t d")
        keys_values = tokens + self.temporal_pe[:, :t]  # K = V = s̃ᵢᵏ + p_k
        query = self.query.expand(tokens.shape[0], -1, -1)

        attn_out, attn = self.mha(query, keys_values, keys_values, need_weights=True)
        if attn is None:  # pragma: no cover - need_weights=True guarantees weights
            raise AssertionError("MultiheadAttention returned no weights")
        pooled = self.norm_attn(query + attn_out)  # time axis collapsed here
        pooled = self.norm_mlp(pooled + self.mlp(pooled))

        pooled = rearrange(pooled, "(b n) 1 d -> b n d", b=b)
        attn = rearrange(attn, "(b n) 1 t -> b n t", b=b)
        return pooled, attn

    def forward(self, slot_history: Float[Tensor, "b t n d"]) -> Float[Tensor, "b n d"]:
        """Pool each slot's history into its causal-parameter vector ŝ^ph_i."""
        return self._pool(slot_history)[0]

    def attention_over_time(self, slot_history: Float[Tensor, "b t n d"]) -> Float[Tensor, "b n t"]:
        """Per-slot attention weights over the time axis (rows sum to 1).

        Diagnostic readout (head-averaged): which timesteps informed each slot's
        parameters.
        """
        return self._pool(slot_history)[1]


class CrossSlotAttnPooling(nn.Module):
    """Cross-slot temporal attention pooling (D14; supersedes per-slot as default).

    One query per slot, projected from that slot's LAST-step embedding (the
    identity anchor), attends over ALL Th·N tokens of the history (+ learned
    temporal PE; deliberately NO slot-identity PE, so slot symmetry is kept).
    ŝ^ph_i therefore remains "the parameters of object i" via its anchor, while
    the evidence may come from any object's trajectory — required when
    parameters are identifiable only through relational events (bounce: m_i
    needs the partner's velocity at contact; per-slot pooling discards it —
    D13 caveat (a)). Permutation equivariance holds: permuting slots permutes
    the queries, and the key/value set is slot-order-free.
    """

    def __init__(
        self,
        slot_size: int,
        num_heads: int = 4,
        mlp_hidden_size: int | None = None,
        max_history: int = 64,
    ) -> None:
        """Build the pooling block (same knobs as the per-slot variant)."""
        super().__init__()
        if slot_size % num_heads != 0:
            raise ValueError(f"num_heads={num_heads} must divide slot_size={slot_size}")
        if mlp_hidden_size is None:
            mlp_hidden_size = 2 * slot_size
        self.max_history = max_history
        self.query_proj = nn.Linear(slot_size, slot_size)
        self.temporal_pe = nn.Parameter(torch.empty(1, max_history, 1, slot_size))
        nn.init.normal_(self.temporal_pe, std=0.02)
        self.mha = nn.MultiheadAttention(slot_size, num_heads, batch_first=True)
        self.norm_attn = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, slot_size),
        )

    def forward(self, slot_history: Float[Tensor, "b t n d"]) -> Float[Tensor, "b n d"]:
        """Pool the FULL history into per-slot parameter vectors Ŝ^ph."""
        if slot_history.ndim != 4:
            raise ValueError(f"expected (B, Th, N, d), got shape {tuple(slot_history.shape)}")
        history_len = slot_history.shape[1]
        if history_len > self.max_history:
            raise ValueError(f"history length {history_len} exceeds max_history={self.max_history}")
        queries = self.query_proj(slot_history[:, -1])  # (B, N, d): slot-identity anchors
        keys_values = rearrange(
            slot_history + self.temporal_pe[:, :history_len], "b t n d -> b (t n) d"
        )
        attn_out, _ = self.mha(queries, keys_values, keys_values, need_weights=False)
        pooled = self.norm_attn(queries + attn_out)
        return self.norm_mlp(pooled + self.mlp(pooled))


class TrackAwareAttnPooling(nn.Module):
    """Track-preserving parameter encoder with equivariant object mixing.

    The temporal and relational axes have deliberately separate stages:

    1. ``AttnPooling`` processes each object's tracked history independently,
       with shared weights and temporal positional encodings.
    2. Self-attention over the resulting object summaries lets collision
       partners exchange evidence. With no object-index positional embedding,
       this stage is permutation equivariant.
    3. A shared linear head emits ``param_dim`` values per object.

    The temporal query is learned and shared rather than projected from the
    final state. Consequently there is no last-state residual path by which
    instantaneous kinematics can bypass temporal parameter inference.
    """

    def __init__(
        self,
        slot_size: int,
        num_heads: int = 4,
        mlp_hidden_size: int | None = None,
        max_history: int = 64,
        param_dim: int | None = None,
    ) -> None:
        """Build the temporal-pooling, object-mixing, and parameter-head stages."""
        super().__init__()
        if slot_size % num_heads != 0:
            raise ValueError(f"num_heads={num_heads} must divide slot_size={slot_size}")
        if param_dim is not None and param_dim <= 0:
            raise ValueError(f"param_dim must be positive, got {param_dim}")
        if mlp_hidden_size is None:
            mlp_hidden_size = 2 * slot_size

        self.param_dim = slot_size if param_dim is None else param_dim
        self.temporal_pool = AttnPooling(
            slot_size=slot_size,
            num_heads=num_heads,
            mlp_hidden_size=mlp_hidden_size,
            max_history=max_history,
        )
        self.cross_slot_mha = nn.MultiheadAttention(slot_size, num_heads, batch_first=True)
        self.norm_attn = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, slot_size),
        )
        # Keep this projection unconstrained: identifiability is only up to an
        # element-wise diffeomorphism, so the learned scalar need not equal mass
        # numerically. In particular, do not put LayerNorm after a 1-D head.
        self.param_head = nn.Linear(slot_size, self.param_dim)

    def forward(self, slot_history: Float[Tensor, "b t n d"]) -> Float[Tensor, "b n p"]:
        """Encode tracked histories as per-object parameter vectors."""
        track_summaries = self.temporal_pool(slot_history)  # (B, N, d)
        mixed, _ = self.cross_slot_mha(
            track_summaries, track_summaries, track_summaries, need_weights=False
        )
        mixed = self.norm_attn(track_summaries + mixed)
        mixed = self.norm_mlp(mixed + self.mlp(mixed))
        return self.param_head(mixed)


class KinematicHead(nn.Module):
    """Linear kinematic-state head: last-step slots → S_t.

    Uses ONLY the final timestep of the slot history — SAVi's recurrence means
    the last-step slots have integrated the full clip — and disassociates the
    kinematic channel from Ŝ^ph via a learned linear map (D4).
    """

    def __init__(self, slot_size: int, state_size: int | None = None) -> None:
        """Build the head.

        Args:
            slot_size: d, slot embedding dimension.
            state_size: Output dimension of S_t (default: same as ``slot_size``).
        """
        super().__init__()
        self.proj = nn.Linear(slot_size, state_size if state_size is not None else slot_size)

    def forward(self, slot_history: Float[Tensor, "b t n d"]) -> Float[Tensor, "b n ds"]:
        """Project the last-step slots to the kinematic state S_t."""
        if slot_history.ndim != 4:
            raise ValueError(f"expected (B, Th, N, d), got shape {tuple(slot_history.shape)}")
        return self.proj(slot_history[:, -1])

    def project(self, slots: Float[Tensor, "b k n d"]) -> Float[Tensor, "b k n ds"]:
        """Project per-step slots to kinematic states (multi-transition path, D15)."""
        return self.proj(slots)


def build_pooling(
    pooling_type: str,
    slot_size: int,
    num_heads: int,
    max_history: int,
    param_dim: int | None = None,
) -> "AttnPooling | CrossSlotAttnPooling | TrackAwareAttnPooling":
    """Build a parameter pooling module selected by configuration.

    ``param_dim`` is used by ``track_aware``. The legacy poolers retain their
    original ``slot_size`` output and reject an incompatible explicit value.
    """
    if pooling_type == "track_aware":
        return TrackAwareAttnPooling(
            slot_size=slot_size,
            num_heads=num_heads,
            max_history=max_history,
            param_dim=param_dim,
        )
    if param_dim is not None and param_dim != slot_size:
        raise ValueError(
            f"pooling_type={pooling_type!r} outputs slot_size={slot_size}; "
            "set pooling_type='track_aware' to use a different param_dim"
        )
    if pooling_type == "cross_slot":
        return CrossSlotAttnPooling(
            slot_size=slot_size, num_heads=num_heads, max_history=max_history
        )
    if pooling_type == "per_slot":
        return AttnPooling(slot_size=slot_size, num_heads=num_heads, max_history=max_history)
    raise ValueError(
        f"unknown pooling_type {pooling_type!r} (cross_slot | per_slot | track_aware)"
    )


__all__ = [
    "AttnPooling",
    "CrossSlotAttnPooling",
    "KinematicHead",
    "TrackAwareAttnPooling",
    "build_pooling",
]
