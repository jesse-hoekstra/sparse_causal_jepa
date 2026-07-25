"""SPARTAN sparse transition predictor f_gamma — experiments.pdf §6.2, Eqs. 27-37.

Implements the paper's token layout exactly: N current-state tokens and N
track-attached scalar parameter tokens, M = 2N tokens total, L_sp single-head
hard-gated sparse-attention layers in a D_sp-dimensional workspace; only the
state-token positions are decoded (Eq. 37). Traceability:

    Eq. 27  x⁰ = W_Z z + b_Z + rho^Z + κᵢ  |  W_θ θ̂ + b_θ + rho^θ + κᵢ
    Eq. 31  g = q·k / √D_sp  (single head, bias-free q/k/v after pre-LN, Eq. 30)
    Eq. 32  A ~ Bernoulli(sigma(g)), hard {0,1}
    Eq. 33  straight-through binary Gumbel-softmax, τ_g = 1, fresh noise
            per transition (these are one-step predictions — no chains)
    Eq. 34  eval gates: A = 1{g > 0}, deterministic
    Eq. 35  masked softmax over OPEN incoming edges; fully closed row → h = 0
    Eq. 36  x^(l) = MLP(x^(l-1) + h): three D_sp-wide linears, ReLU after the
            first two, no post-MLP residual or LayerNorm
    Eq. 10  Ā = (A^(L)+I)···(A^(1)+I), path counts
    Eq. 11  L_path = Σ decoded state rows of Ā;  rho_path thresholds it

Track keys κᵢ (Eq. 27, §6.4): one key is SHARED by the state token and the
parameter token descended from track i — it records their common trajectory
origin without forcing any same-track edge open. Keys are drawn once from a
fixed non-trainable codebook (registered as a buffer: excluded from gradient
updates, independent of states and parameters). Experiment 1's tracks are the
ordered simulator rows, so codebook row i serves track i in every episode; the
visual experiments will add the episode-level codebook permutation of §6.4.
Key SCALE is unspecified upstream — matched to the role-embedding init (0.02).

Reference modes (§6.1.3, matched initialization — same modules, same codebook):
    dense:    A ≡ 1 in every layer, no gates sampled (τ-calibration reference).
    identity: A ≡ 0 — token-local: only each token's residual-MLP path remains.

Numerics kept from the audited implementation (they compute Eq. 35 exactly but
stably): row-max subtraction over unmasked entries only, and a denominator that
avoids the 1e-8-floor gradient amplification on fully masked rows.

Logit penalty is Eq. 11-of-Baumgartner / the write-up's §6.1.3 form:
mean of exp(a) + exp(-a) over layer logits, theoretical minimum 2, with a
linear continuation beyond |a| = 30 for fp32 stability (as declared there).
"""

import math
from typing import NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

# One fixed codebook for every model built at any size: dense reference,
# token-local reference, and sparse run share track keys by construction.
_TRACK_KEY_SEED = 0


class SpartanOutput(NamedTuple):
    """Everything the loss and the SHD/MCC evaluation need from one forward."""

    prediction: Float[Tensor, "b n k"]
    path_matrix: Float[Tensor, "b m m"]
    sparsity: Float[Tensor, ""]
    logit_penalty: Float[Tensor, ""]
    mean_abs_logit: Float[Tensor, ""]
    mean_gate_probability: Float[Tensor, ""]
    gate_entropy: Float[Tensor, ""]


def _sample_hard_adjacency(logits: Tensor, temperature: float, training: bool) -> Tensor:
    """Hard {0,1} gates: Eq. 33 in training, Eq. 34 deterministic in eval."""
    if not training:
        return (logits > 0).to(logits.dtype)
    uniform = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
    noise = uniform.log() - (-uniform).log1p()  # Logistic(0,1) = Gumbel difference
    soft = torch.sigmoid((logits + noise) / temperature)
    hard = (soft > 0.5).to(logits.dtype)
    return hard + soft - soft.detach()  # straight-through


class SpartanLayer(nn.Module):
    """One single-head hard-gated sparse-attention layer (Eqs. 30-36)."""

    def __init__(
        self,
        dim: int,
        mlp_hidden_size: int,
        mlp_num_layers: int,
        temperature: float,
        dense: bool = False,
        identity: bool = False,
    ) -> None:
        """Build one layer; ``dim`` = D_sp, other args as in ``Spartan``."""
        super().__init__()
        if mlp_num_layers < 2:
            raise ValueError("mlp_num_layers must be >= 2")
        if dense and identity:
            raise ValueError("dense and identity are mutually exclusive")
        self.temperature = temperature
        self.dense = dense
        self.identity = identity
        self.scale = 1.0 / math.sqrt(dim)
        self.norm = nn.LayerNorm(dim)
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
        self, tokens: Float[Tensor, "b m d"]
    ) -> tuple[
        Float[Tensor, "b m d"],
        Float[Tensor, "b m m"],
        Float[Tensor, ""],
        Float[Tensor, ""],
        Float[Tensor, ""],
        Float[Tensor, ""],
    ]:
        """Apply hard-masked attention; return outputs plus logit diagnostics."""
        normed = self.norm(tokens)  # Eq. 30 pre-LN
        q = self.project_q(normed)
        k = self.project_k(normed)
        v = self.project_v(normed)
        logits = torch.einsum("bid,bjd->bij", q, k) * self.scale  # Eq. 31

        if self.dense:
            adjacency = torch.ones_like(logits)
        elif self.identity:
            adjacency = torch.zeros_like(logits)
        else:
            adjacency = _sample_hard_adjacency(logits, self.temperature, self.training)

        # Eq. 35, computed stably: subtract the row max over UNMASKED entries
        # only (using the global max collapses the denominator to ~0 when the
        # row's largest logit is masked, amplifying gradients ~1e6x), and give
        # fully masked rows a denominator of 1 instead of a 1e-8 floor (which
        # amplified straight-through gradients ~1e8x during pruning).
        unmasked_max = logits.masked_fill(adjacency == 0, float("-inf")).max(dim=-1, keepdim=True)
        row_max = torch.where(
            torch.isfinite(unmasked_max.values), unmasked_max.values, 0.0
        ).detach()
        weights = adjacency * (logits - row_max).clamp(max=0.0).exp()
        denom = weights.sum(dim=-1, keepdim=True)
        weights = weights / (denom + (denom.detach() < 0.5).to(weights.dtype))
        h = torch.einsum("bij,bjd->bid", weights, v)

        # Logit penalty: exp(a) + exp(-a), exact within |a| <= 30, linear
        # continuation beyond (slope e^30 — an effectively hard wall that
        # stays fp32-finite), as declared in §6.1.3.
        magnitude = logits.abs()
        core = magnitude.clamp(max=30.0)
        tail = (magnitude - 30.0).clamp(min=0.0)
        logit_penalty = (core.exp() + (-core).exp() + core.exp() * tail).mean()
        with torch.no_grad():
            gate_probability = torch.sigmoid(logits.detach())
            gate_entropy = (F.softplus(logits.detach()) - logits.detach() * gate_probability).mean()
        return (
            self.mlp(tokens + h),  # Eq. 36: MLP(x + h)
            adjacency,
            logit_penalty,
            magnitude.detach().mean(),
            gate_probability.mean(),
            gate_entropy,
        )


class Spartan(nn.Module):
    """Sparse transition predictor over [state | parameter] tokens (Eq. 38)."""

    def __init__(
        self,
        state_dim: int = 4,
        param_dim: int = 1,
        num_slots: int = 5,
        num_layers: int = 3,
        embed_dim: int = 512,
        mlp_hidden_size: int = 512,
        mlp_num_layers: int = 3,
        temperature: float = 1.0,
        dense: bool = False,
        identity: bool = False,
    ) -> None:
        """Build the predictor (defaults are the Experiment-1 configuration).

        Args:
            state_dim: k = 4, raw state width (also the decoded output width).
            param_dim: Scalar parameter coordinates (Eq. 25): 1.
            num_slots: N tracked objects; the token count is M = 2N.
            num_layers: L_sp = 3.
            embed_dim: D_sp = 512 workspace width.
            mlp_hidden_size: Width of Eq. 36's hidden linears (512).
            mlp_num_layers: Linear layers per token MLP (Eq. 36: three).
            temperature: τ_g = 1 Gumbel-softmax temperature (Eq. 33).
            dense: A ≡ 1 — the fully connected τ-calibration reference.
            identity: A ≡ 0 — the token-local reference. Mutually exclusive.
        """
        super().__init__()
        if dense and identity:
            raise ValueError("dense and identity are mutually exclusive")
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.num_slots = num_slots
        self.dense = dense
        self.identity = identity
        # Eq. 27/28: separate input projections into the shared workspace.
        self.state_project = nn.Linear(state_dim, embed_dim)
        self.param_project = nn.Linear(param_dim, embed_dim)
        self.out_project = nn.Linear(embed_dim, state_dim)  # Eq. 37
        # Learned role embeddings rho^Z, rho^θ (Eq. 27).
        self.state_role = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.param_role = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.state_role, std=0.02)
        nn.init.normal_(self.param_role, std=0.02)
        # Track keys κᵢ (Eq. 27): fixed, non-trainable, shared by the state and
        # parameter token of track i. Buffer => excluded from gradient updates
        # and saved with the checkpoint. Scale 0.02 is our choice (unspecified).
        generator = torch.Generator().manual_seed(_TRACK_KEY_SEED)
        track_keys = 0.02 * torch.randn(1, num_slots, embed_dim, generator=generator)
        self.track_keys: Tensor
        self.register_buffer("track_keys", track_keys)
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

    def forward(
        self,
        state: Float[Tensor, "b n k"],
        params: Float[Tensor, "b n p"],
    ) -> SpartanOutput:
        """One transition: predict Ẑ_{t+1} from (Z_t, θ̂) and expose the graph."""
        if (
            state.ndim != 3
            or params.ndim != 3
            or state.shape[:2] != params.shape[:2]
            or state.shape[1] != self.num_slots
            or state.shape[2] != self.state_dim
            or params.shape[2] != self.param_dim
        ):
            raise ValueError(
                f"expected state (B, {self.num_slots}, {self.state_dim}) and params "
                f"(B, {self.num_slots}, {self.param_dim}), got {tuple(state.shape)} "
                f"vs {tuple(params.shape)}"
            )
        # Eq. 27: token construction, [state 0..N-1 | params N..2N-1].
        state_tokens = self.state_project(state) + self.state_role + self.track_keys
        param_tokens = self.param_project(params) + self.param_role + self.track_keys
        tokens = torch.cat((state_tokens, param_tokens), dim=1)  # (B, 2N, D_sp)

        adjacencies: list[Tensor] = []
        logit_penalties: list[Tensor] = []
        mean_abs_logits: list[Tensor] = []
        mean_gate_probabilities: list[Tensor] = []
        gate_entropies: list[Tensor] = []
        for layer in self.layers:
            tokens, adjacency, penalty, abs_logit, gate_prob, entropy = layer(tokens)
            adjacencies.append(adjacency)
            logit_penalties.append(penalty)
            mean_abs_logits.append(abs_logit)
            mean_gate_probabilities.append(gate_prob)
            gate_entropies.append(entropy)

        # Eq. 10: Ā = (A^(L)+I)···(A^(1)+I); entries are path counts.
        eye = torch.eye(tokens.shape[1], device=tokens.device, dtype=tokens.dtype)
        path_matrix = eye.expand(tokens.shape[0], -1, -1)
        for adjacency in adjacencies:
            path_matrix = (adjacency + eye) @ path_matrix
        # Eq. 11: the path objective covers only the N decoded state rows.
        sparsity = path_matrix[:, : self.num_slots].sum(dim=(1, 2)).mean()

        return SpartanOutput(
            prediction=self.out_project(tokens[:, : self.num_slots]),  # Eq. 37
            path_matrix=path_matrix,
            sparsity=sparsity,
            logit_penalty=torch.stack(logit_penalties).mean(),
            mean_abs_logit=torch.stack(mean_abs_logits).mean(),
            mean_gate_probability=torch.stack(mean_gate_probabilities).mean(),
            gate_entropy=torch.stack(gate_entropies).mean(),
        )


__all__ = ["Spartan", "SpartanLayer", "SpartanOutput"]
