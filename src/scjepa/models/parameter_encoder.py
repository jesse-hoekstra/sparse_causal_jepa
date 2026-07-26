"""Parameter encoder P_η — experiments.pdf §6.2, Eqs. 16-26, implemented 1:1.

Maps the context window of tracked object states to one unconstrained scalar
parameter coordinate per track:

    P_η : Z_{0:Tpar-1} ∈ (B, Tpar, N, k)  →  θ̂ ∈ (B, N, 1).

Two stages, in the write-up's order:

1. Relational contextualization (Eqs. 18-20): at EVERY context timestep,
   permutation-equivariant multi-head self-attention across the N tracks, with
   parameters shared over timesteps and tracks. This is what lets h̃ⁱ_t encode
   synchronized interaction evidence — the states of both collision partners
   immediately before and after a contact.
2. Identity-preserving temporal pooling (Eqs. 21-24): one learned temporal
   query, shared across tracks and episodes, attends over each track's Tpar
   contextualized states independently (weights shared across tracks).

A shared unconstrained scalar head (Eq. 25) emits θ̂ᵢ = w^T rᵢ + b. No
normalization follows the head: identification is only required up to an
invertible element-wise reparameterization.

Both stages use post-norm residual blocks exactly as written:
u = LN(x + attn(x)), out = LN(u + FFN(u)), with FFN_rel a two-layer ReLU
network of hidden width 2d (Eq. 20's stated width).

Choices the write-up leaves open (flagged per project policy):
  * π^time is a LEARNED positional table (Eq. 16 writes π^time_t without
    specifying its construction).
  * FFN_time's hidden width is unstated; we reuse FFN_rel's 2d.

Permutation contract (§6.2): no absolute track-index embedding exists anywhere
in this module, so permuting the tracked-object axis permutes θ̂ identically.
The visual regimes reuse this module unchanged apart from the input map (Eq. 91):
pass ``state_dim`` = the visual slot width instead of 4.
"""

import torch
from jaxtyping import Float
from torch import Tensor, nn


class _PostNormBlock(nn.Module):
    """u = LN(x + sublayer_out); out = LN(u + FFN(u)) — Eqs. 20 and 24."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))

    def forward(self, residual: Tensor, sublayer_out: Tensor) -> Tensor:
        u = self.norm_attn(residual + sublayer_out)
        return self.norm_ffn(u + self.ffn(u))


class ParameterEncoder(nn.Module):
    """Track-attached scalar system-parameter coordinates from state history."""

    def __init__(
        self,
        state_dim: int = 4,
        dim: int = 32,
        num_heads: int = 4,
        max_history: int = 64,
    ) -> None:
        """Build the encoder.

        Args:
            state_dim: k, per-object input width (state-to-state: 4 = [x,y,vx,vy];
                the visual regimes pass the slot width instead, Eq. 91).
            dim: d = 32, working width (Eq. 16).
            num_heads: H = 4 attention heads, d_h = d/H = 8 (Eq. 18).
            max_history: Length of the learned temporal-PE table; forward
                slices the first Tpar entries.
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"num_heads={num_heads} must divide dim={dim}")
        self.max_history = max_history
        # Eq. 16: shared linear input map + temporal position encoding.
        self.input_map = nn.Linear(state_dim, dim)
        self.temporal_pe = nn.Parameter(torch.empty(1, max_history, 1, dim))
        nn.init.normal_(self.temporal_pe, std=0.02)
        # Eqs. 18-20: relational self-attention across tracks, per timestep.
        self.relational_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.relational_block = _PostNormBlock(dim)
        # Eqs. 21-24: single learned temporal query, per-track pooling.
        self.temporal_query = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.temporal_query, std=dim**-0.5)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.temporal_block = _PostNormBlock(dim)
        # Eq. 25: shared unconstrained scalar head — never add a norm here.
        self.head = nn.Linear(dim, 1)

    def forward(self, states: Float[Tensor, "b t n k"]) -> Float[Tensor, "b n 1"]:
        """Encode the context window into θ̂ ∈ (B, N, 1)."""
        if states.ndim != 4:
            raise ValueError(f"expected (B, Tpar, N, k), got {tuple(states.shape)}")
        batch, history, num_tracks, _ = states.shape
        if history > self.max_history:
            raise ValueError(f"history length {history} exceeds max_history={self.max_history}")

        # Eq. 16: hⁱ_t = W_c zⁱ_t + b_c + π^time_t.
        h = self.input_map(states) + self.temporal_pe[:, :history]  # (B, T, N, d)

        # Eqs. 18-20: self-attention over the track axis at each timestep.
        tokens = h.reshape(batch * history, num_tracks, -1)
        attn_out, _ = self.relational_attn(tokens, tokens, tokens, need_weights=False)
        contextualized = self.relational_block(tokens, attn_out)  # h̃ⁱ_t
        contextualized = contextualized.reshape(batch, history, num_tracks, -1)

        # Eqs. 21-24: pool each track's Tpar states with the shared query.
        per_track = contextualized.permute(0, 2, 1, 3).reshape(batch * num_tracks, history, -1)
        query = self.temporal_query.expand(per_track.shape[0], -1, -1)
        pooled, _ = self.temporal_attn(query, per_track, per_track, need_weights=False)
        summaries = self.temporal_block(query, pooled)  # rᵢ, (B·N, 1, d)

        # Eq. 25/26: θ̂ᵢ = w^T rᵢ + b, stacked to (B, N, 1).
        return self.head(summaries).reshape(batch, num_tracks, 1)


__all__ = ["ParameterEncoder"]
