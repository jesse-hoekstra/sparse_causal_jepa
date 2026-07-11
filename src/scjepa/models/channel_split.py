"""Channel split (D4): slot history → causal parameters Ŝ^ph and kinematic state S_t.

Implements the exact spec in docs/decisions.md D4:

- ``AttnPooling`` — per-slot temporal attention pooling (a PMA-style block, Lee et
  al. 2019), weights shared across slots, single learned query, learned temporal
  positional encodings; collapses the time axis inside the attention:
  ``(B, Th, N, d) → Ŝ^ph ∈ (B, N, d)``. Strictly slot-local: no cross-slot mixing —
  relational effects are SPARTAN's job.
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
    pooling_type: str, slot_size: int, num_heads: int, max_history: int
) -> "AttnPooling | CrossSlotAttnPooling":
    """Config-selectable pooling: "cross_slot" (D14 default) | "per_slot" (D4)."""
    if pooling_type == "cross_slot":
        return CrossSlotAttnPooling(
            slot_size=slot_size, num_heads=num_heads, max_history=max_history
        )
    if pooling_type == "per_slot":
        return AttnPooling(slot_size=slot_size, num_heads=num_heads, max_history=max_history)
    raise ValueError(f"unknown pooling_type {pooling_type!r} (cross_slot | per_slot)")


__all__ = ["AttnPooling", "CrossSlotAttnPooling", "KinematicHead", "build_pooling"]
