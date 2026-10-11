"""Coordinate-level sparse attention for finite-lag, parameterized dynamics.

Inputs are scalar tokens in the fixed order ``[state lags | action lags | parameters]``.
Within the first two groups the order is lag-major, oldest to newest. Every
coordinate has its own learned identity; state/action identities also include a
lag embedding. No coordinates are mixed by the input projections, value
normalization, token MLPs, or scalar output head.

Attention logits and gates depend only on these identities, never on observation
values. This deliberate choice makes the extracted graph a complete structural
upper bound on information flow within the predictor, conditional on its
independently supplied parameter vector. It does not include the context-to-
parameter inference path. Content-dependent queries or keys would create control
paths through attention weights that a masked value graph could miss.
Sparse gates use straight-through binary sampling in training and deterministic
thresholding in evaluation. Dense and token-local reference modes share the same
modules. Stable masked normalization and residual path accounting are retained
from the previous implementation.
"""

import math
from typing import NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F


class AggregatedGraphBlocks(NamedTuple):
    """Any-lag maximum path counts; ``> 0`` gives any-lag reachability."""

    state: Float[Tensor, "b dz dz"]
    action: Float[Tensor, "b dz da"]
    parameter: Float[Tensor, "b dz dp"]


class GraphPenalties(NamedTuple):
    """Independent block means of residual-inclusive path counts.

    Disabling a block's loss weight removes that objective. Other source-block
    paths may still traverse the same intermediate token workspaces, and gate
    projections are shared; it does not freeze parameter-related gates.
    """

    state: Float[Tensor, ""]
    action: Float[Tensor, ""]
    parameter: Float[Tensor, ""]


class GraphBlocks(NamedTuple):
    """Paths from each input coordinate to each predicted state coordinate.

    State/action lag index zero is the oldest supplied input. Counts include
    every residual path and every sparse-attention layer. Nonzero entries are
    possible dependencies, not proof of causal influence or identifiability.
    """

    state: Float[Tensor, "b dz lag dz"]
    action: Float[Tensor, "b dz lag da"]
    parameter: Float[Tensor, "b dz dp"]

    def aggregate_lags(self) -> AggregatedGraphBlocks:
        """Take a maximum over lags; threshold at zero for any-lag dependencies."""
        return AggregatedGraphBlocks(
            self.state.amax(dim=2), self.action.amax(dim=2), self.parameter
        )

    def penalties(self) -> GraphPenalties:
        """Average each block independently; an absent action block costs zero."""
        action = self.action.mean() if self.action.numel() else self.state.new_zeros(())
        return GraphPenalties(self.state.mean(), action, self.parameter.mean())


class SpartanOutput(NamedTuple):
    """Prediction, structural path counts, and explicit regularization terms."""

    prediction: Float[Tensor, "b dz"]
    path_matrix: Float[Tensor, "b m m"]
    graphs: GraphBlocks
    sparsity: GraphPenalties
    logit_penalty: Float[Tensor, ""]
    mean_abs_logit: Float[Tensor, ""]
    mean_gate_probability: Float[Tensor, ""]
    gate_entropy: Float[Tensor, ""]


def _sample_hard_adjacency(logits: Tensor, temperature: float, training: bool) -> Tensor:
    """Sample hard gates with a straight-through derivative; threshold at eval."""
    if not training:
        return (logits > 0).to(logits.dtype)
    uniform = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
    noise = uniform.log() - (-uniform).log1p()
    soft = torch.sigmoid((logits + noise) / temperature)
    hard = (soft > 0.5).to(logits.dtype)
    return hard + soft - soft.detach()


class SpartanLayer(nn.Module):
    """Single-head attention with structure-only control and token-local values."""

    def __init__(
        self,
        dim: int,
        mlp_hidden_size: int,
        mlp_num_layers: int,
        temperature: float,
        dense: bool = False,
        identity: bool = False,
    ) -> None:
        """Build attention and a token-local residual MLP."""
        super().__init__()
        if dim <= 0 or mlp_hidden_size <= 0:
            raise ValueError("dim and mlp_hidden_size must be positive")
        if mlp_num_layers < 2:
            raise ValueError("mlp_num_layers must be >= 2")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if dense and identity:
            raise ValueError("dense and identity are mutually exclusive")
        self.temperature = temperature
        self.dense = dense
        self.identity = identity
        self.scale = 1.0 / math.sqrt(dim)
        self.norm = nn.LayerNorm(dim)
        self.structure_norm = nn.LayerNorm(dim)
        self.project_q = nn.Linear(dim, dim, bias=False)
        self.project_k = nn.Linear(dim, dim, bias=False)
        self.project_v = nn.Linear(dim, dim, bias=False)
        widths = [dim] + [mlp_hidden_size] * (mlp_num_layers - 1) + [dim]
        mlp_layers: list[nn.Module] = []
        for i in range(mlp_num_layers):
            mlp_layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < mlp_num_layers - 1:
                mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(
        self,
        tokens: Float[Tensor, "b m d"],
        identities: Float[Tensor, "1 m d"],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return updated tokens, sampled adjacency, and logit diagnostics."""
        structural = self.structure_norm(identities)
        q = self.project_q(structural)
        k = self.project_k(structural)
        v = self.project_v(self.norm(tokens))
        logits = (torch.einsum("bid,bjd->bij", q, k) * self.scale).expand(tokens.shape[0], -1, -1)
        if self.dense:
            adjacency = torch.ones_like(logits)
        elif self.identity:
            adjacency = torch.zeros_like(logits)
        else:
            adjacency = _sample_hard_adjacency(logits, self.temperature, self.training)

        # Subtract only the open-entry maximum. A closed row uses denominator
        # one instead of an epsilon floor that amplifies straight-through grads.
        unmasked_max = logits.masked_fill(adjacency == 0, float("-inf")).amax(dim=-1, keepdim=True)
        row_max = torch.where(torch.isfinite(unmasked_max), unmasked_max, 0.0).detach()
        weights = adjacency * (logits - row_max).clamp(max=0.0).exp()
        denom = weights.sum(dim=-1, keepdim=True)
        weights = weights / (denom + (denom.detach() < 0.5).to(weights.dtype))
        h = torch.einsum("bij,bjd->bid", weights, v)

        # exp(a) + exp(-a), with a linear continuation above |a|=30. Evaluate
        # in float32 under mixed precision so half-precision exp does not overflow.
        diagnostic_logits = logits.float()
        magnitude = diagnostic_logits.abs()
        core = magnitude.clamp(max=30.0)
        tail = (magnitude - 30.0).clamp(min=0.0)
        logit_penalty = (core.exp() + (-core).exp() + core.exp() * tail).mean()
        with torch.no_grad():
            detached = diagnostic_logits.detach()
            gate_probability = torch.sigmoid(detached)
            gate_entropy = (F.softplus(detached) - detached * gate_probability).mean()
        return (
            self.mlp(tokens + h),
            adjacency,
            logit_penalty,
            magnitude.detach().mean(),
            gate_probability.mean(),
            gate_entropy,
        )


class Spartan(nn.Module):
    """Predict a state vector from finite state/action history and one parameter vector."""

    def __init__(
        self,
        state_dim: int,
        param_dim: int,
        action_dim: int = 0,
        history_len: int = 1,
        num_layers: int = 2,
        embed_dim: int = 64,
        mlp_hidden_size: int = 64,
        mlp_num_layers: int = 3,
        temperature: float = 1.0,
        dense: bool = False,
        identity: bool = False,
    ) -> None:
        """Configure independent coordinate counts and the finite input-history length.

        ``history_len`` counts input lags, not rollout transitions. ``param_dim``
        need not match ``state_dim``. Set ``action_dim=0`` for autonomous dynamics.
        """
        super().__init__()
        if state_dim <= 0 or param_dim <= 0 or history_len <= 0 or num_layers <= 0:
            raise ValueError("state_dim, param_dim, history_len, num_layers must be positive")
        if action_dim < 0:
            raise ValueError("action_dim must be nonnegative")
        if dense and identity:
            raise ValueError("dense and identity are mutually exclusive")
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.embed_dim = embed_dim
        self.num_tokens = history_len * (state_dim + action_dim) + param_dim
        self.dense = dense
        self.identity = identity
        self.state_project = nn.Linear(1, embed_dim)
        self.action_project = nn.Linear(1, embed_dim) if action_dim else None
        self.param_project = nn.Linear(1, embed_dim)
        self.out_project = nn.Linear(embed_dim, 1)
        self.state_identity = nn.Parameter(torch.empty(state_dim, embed_dim))
        self.action_identity = nn.Parameter(torch.empty(action_dim, embed_dim))
        self.parameter_identity = nn.Parameter(torch.empty(param_dim, embed_dim))
        self.lag_identity = nn.Parameter(torch.empty(history_len, embed_dim))
        self.role_identity = nn.Parameter(torch.empty(3, embed_dim))
        for embedding in (
            self.state_identity,
            self.action_identity,
            self.parameter_identity,
            self.lag_identity,
            self.role_identity,
        ):
            nn.init.normal_(embedding, std=0.02)
        self.layers = nn.ModuleList(
            [
                SpartanLayer(
                    embed_dim,
                    mlp_hidden_size,
                    mlp_num_layers,
                    temperature,
                    dense=dense,
                    identity=identity,
                )
                for _ in range(num_layers)
            ]
        )

    def token_identities(self) -> Float[Tensor, "1 m d"]:
        """Return content-independent identities in the documented token order."""
        state = self.state_identity[None] + self.lag_identity[:, None] + self.role_identity[0]
        action = self.action_identity[None] + self.lag_identity[:, None] + self.role_identity[1]
        parameter = self.parameter_identity + self.role_identity[2]
        return torch.cat((state.flatten(0, 1), action.flatten(0, 1), parameter), dim=0).unsqueeze(0)

    def extract_graphs(self, path_matrix: Float[Tensor, "b m m"]) -> GraphBlocks:
        """Slice decoded latest-state rows into lag-resolved input blocks."""
        if path_matrix.ndim != 3 or path_matrix.shape[1:] != (self.num_tokens, self.num_tokens):
            raise ValueError("path_matrix must have shape (B, num_tokens, num_tokens)")
        state_end = self.history_len * self.state_dim
        action_end = state_end + self.history_len * self.action_dim
        rows = path_matrix[:, state_end - self.state_dim : state_end]
        batch = path_matrix.shape[0]
        return GraphBlocks(
            rows[:, :, :state_end].reshape(batch, self.state_dim, self.history_len, self.state_dim),
            rows[:, :, state_end:action_end].reshape(
                batch, self.state_dim, self.history_len, self.action_dim
            ),
            rows[:, :, action_end:],
        )

    def forward(
        self,
        states: Float[Tensor, "b lag dz"],
        parameters: Float[Tensor, "b dp"],
        actions: Float[Tensor, "b lag da"] | None = None,
    ) -> SpartanOutput:
        """Predict the next state; action at each lag accompanies that lag's state."""
        if states.ndim != 3 or states.shape[1:] != (self.history_len, self.state_dim):
            raise ValueError(f"expected states (B, {self.history_len}, {self.state_dim})")
        batch = states.shape[0]
        if parameters.shape != (batch, self.param_dim):
            raise ValueError(f"expected parameters (B, {self.param_dim})")
        if self.action_dim:
            if actions is None or actions.shape != (batch, self.history_len, self.action_dim):
                raise ValueError(f"expected actions (B, {self.history_len}, {self.action_dim})")
        elif actions is not None:
            raise ValueError("actions must be None when action_dim=0")
        if batch == 0:
            raise ValueError("batch must be nonempty")

        state_tokens = self.state_project(states.flatten(1).unsqueeze(-1))
        parameter_tokens = self.param_project(parameters.unsqueeze(-1))
        parts = [state_tokens]
        if self.action_project is not None and actions is not None:
            parts.append(self.action_project(actions.flatten(1).unsqueeze(-1)))
        parts.append(parameter_tokens)
        identities = self.token_identities()
        tokens = torch.cat(parts, dim=1) + identities
        adjacencies: list[Tensor] = []
        logit_penalties: list[Tensor] = []
        mean_abs_logits: list[Tensor] = []
        mean_gate_probabilities: list[Tensor] = []
        gate_entropies: list[Tensor] = []
        for layer in self.layers:
            tokens, adjacency, penalty, abs_logit, gate_prob, entropy = layer(tokens, identities)
            adjacencies.append(adjacency)
            logit_penalties.append(penalty)
            mean_abs_logits.append(abs_logit)
            mean_gate_probabilities.append(gate_prob)
            gate_entropies.append(entropy)

        # Every token-local residual is an identity edge. Left multiplication
        # follows receiver-row/source-column convention across successive layers.
        path_dtype = (
            torch.float32 if tokens.dtype in (torch.float16, torch.bfloat16) else tokens.dtype
        )
        eye = torch.eye(self.num_tokens, device=tokens.device, dtype=path_dtype)
        path_matrix = eye.expand(batch, -1, -1)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            for adjacency in adjacencies:
                path_matrix = (adjacency.to(path_dtype) + eye) @ path_matrix
        graphs = self.extract_graphs(path_matrix)
        state_end = self.history_len * self.state_dim
        return SpartanOutput(
            prediction=self.out_project(tokens[:, state_end - self.state_dim : state_end]).squeeze(
                -1
            ),
            path_matrix=path_matrix,
            graphs=graphs,
            sparsity=graphs.penalties(),
            logit_penalty=torch.stack(logit_penalties).mean(),
            mean_abs_logit=torch.stack(mean_abs_logits).mean(),
            mean_gate_probability=torch.stack(mean_gate_probabilities).mean(),
            gate_entropy=torch.stack(gate_entropies).mean(),
        )


__all__ = [
    "AggregatedGraphBlocks",
    "GraphBlocks",
    "GraphPenalties",
    "Spartan",
    "SpartanLayer",
    "SpartanOutput",
]
