"""Infer one attached, time-constant parameter vector from permitted context.

Attention, post-normalization blocks, positional embeddings, and learned-query
temporal pooling summarize whole latent state/action observations. Contextual
coordinate mixing is confined to parameter inference: SPARTAN's coordinate graph
describes predictor dependencies conditional on the inferred parameters.
"""

import torch
from torch import Tensor, nn


class _PostNormBlock(nn.Module):
    """Normalize an attention residual, then apply a residual two-layer FFN."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))

    def forward(self, residual: Tensor, sublayer_out: Tensor) -> Tensor:
        u = self.norm_attn(residual + sublayer_out)
        return self.norm_ffn(u + self.ffn(u))


class ParameterEncoder(nn.Module):
    """Attention and temporal pooling produce one ``(B, param_dim)`` vector."""

    def __init__(
        self,
        state_dim: int,
        param_dim: int,
        action_dim: int = 0,
        dim: int = 32,
        num_heads: int = 4,
        max_history: int = 64,
    ) -> None:
        """Configure independent state, parameter, and optional action dimensions."""
        super().__init__()
        if min(state_dim, param_dim, dim, num_heads, max_history) < 1 or action_dim < 0:
            raise ValueError("dimensions and max_history must be positive; action_dim may be zero")
        if dim % num_heads != 0:
            raise ValueError(f"num_heads={num_heads} must divide dim={dim}")
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.action_dim = action_dim
        self.max_history = max_history
        self.input_map = nn.Linear(state_dim + action_dim + int(action_dim > 0), dim)
        self.temporal_pe = nn.Parameter(torch.empty(1, max_history, dim))
        nn.init.normal_(self.temporal_pe, std=0.02)
        self.context_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.context_block = _PostNormBlock(dim)
        self.temporal_query = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.temporal_query, std=dim**-0.5)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.temporal_block = _PostNormBlock(dim)
        self.head = nn.Linear(dim, param_dim)

    def forward(self, states: Tensor, actions: Tensor | None = None) -> Tensor:
        """Pool states ``(B,K,d_z)`` and observed actions ``(B,K-1,d_a)``.

        All supplied context may interact; callers must provide only permitted
        context. The terminal observation has no context action; a zero placeholder
        and an observed-action indicator mark this explicitly. No parameters
        are detached, repeated as samples, or inferred from a target encoder.
        The unconstrained head has no output normalization.
        """
        if states.ndim != 3 or states.shape[-1] != self.state_dim:
            raise ValueError(f"expected states (B, K, {self.state_dim})")
        batch, history, _ = states.shape
        if batch < 1 or not 1 <= history <= self.max_history:
            raise ValueError(
                f"batch must be nonempty and context length in [1, {self.max_history}]"
            )
        if self.action_dim:
            if actions is None or actions.shape != (batch, history - 1, self.action_dim):
                raise ValueError(f"expected actions (B, K-1, {self.action_dim})")
            padded_actions = torch.cat(
                (actions, actions.new_zeros(batch, 1, self.action_dim)), dim=1
            )
            observed = states.new_ones(batch, history, 1)
            observed[:, -1] = 0
            inputs = torch.cat((states, padded_actions, observed), dim=-1)
        else:
            if actions is not None:
                raise ValueError("actions must be None when action_dim=0")
            inputs = states
        tokens = self.input_map(inputs) + self.temporal_pe[:, :history]
        attended, _ = self.context_attn(tokens, tokens, tokens, need_weights=False)
        contextualized = self.context_block(tokens, attended)
        query = self.temporal_query.expand(batch, -1, -1)
        pooled, _ = self.temporal_attn(query, contextualized, contextualized, need_weights=False)
        return self.head(self.temporal_block(query, pooled)).squeeze(1)


__all__ = ["ParameterEncoder"]
