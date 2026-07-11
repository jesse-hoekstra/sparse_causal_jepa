"""SPARTAN predictor (Lei, Schölkopf & Posner 2024; sources/SPARTAN.pdf, arXiv:2411.06890).

No public upstream code exists — implemented from the paper, adapted to this
project's token layout (my_paper.pdf Fig. 1): the transformer attends over
N kinematic-state tokens S_t, N causal-parameter tokens Ŝ^ph, and optional
auxiliary tokens U_t (actions; appended per my_paper §4.1 "{S_t, U_t} can be
seen as the new S_t"). Predictions Ŝ_{t+1} are read from the state-token
positions. The learned local causal graph is exposed as a first-class output.

Mechanics traceable to the SPARTAN paper:
    Eq. 3   A_ij ~ Bern(sigmoid(q_i·k_j)) — hard adjacency, sampled per layer;
            differentiable via the binary Gumbel-softmax (straight-through).
    Eq. 4   masked scaled dot-product attention; ŝ_i = MLP(h_i + s_i).
    Eq. 5   path matrix  Ā = (A_L + I)···(A_1 + I); token j is a local causal
            parent of output i  iff  Ā_ij ≥ 1 (I from the residual paths).
    Eq. 6   sparsity penalty |Ā| (sum of entries) — returned as
            ``SpartanOutput.sparsity``; the Lagrangian-relaxation schedule for
            its weight (App. A.2) lives in the training loop, not here.

Choices the paper leaves open (flagged per project policy, decisions D10):
    * Eq. 4's printed normalization is ambiguous; we renormalize the softmax
      over the UNMASKED entries only. Masking after normalization would leak
      masked tokens' information through the denominator, contradicting the
      paper's "adjacency ... disallows information flows" claim.
    * Single-head attention per layer (Eqs. 3-4 define one adjacency per layer).
    * Eval mode is deterministic: A_ij = 1 iff sigmoid(q_i·k_j) > 1/2.
    * Learned role embeddings (state/param/aux) are added to the tokens so the
      roles are distinguishable while slot-permutation equivariance is kept.

Symbol table:
    S_t     state tokens       (B, N, d)     ``state``
    Ŝ^ph    parameter tokens   (B, N, d)     ``params``
    U_t     auxiliary tokens   (B, M, d_u)   ``aux`` (optional)
    A_l     layer adjacency    (B, T, T)     T = 2N (+M); hard {0, 1}
    Ā       path matrix        (B, T, T)     ``SpartanOutput.path_matrix``
    Ŝ_{t+1} prediction         (B, N, d)     ``SpartanOutput.prediction``
"""

import math
from typing import NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor, nn


class SpartanOutput(NamedTuple):
    """Everything the losses and the SHD/MCC eval need from one forward pass."""

    prediction: Float[Tensor, "b n d"]
    path_matrix: Float[Tensor, "b t t"]
    sparsity: Float[Tensor, ""]
    logit_penalty: Float[Tensor, ""]


def _sample_hard_adjacency(logits: Tensor, temperature: float, training: bool) -> Tensor:
    """Hard {0,1} adjacency from Bernoulli logits (Eq. 3).

    Training: binary Gumbel-softmax (Logistic reparameterization) with a
    straight-through estimator — forward is exactly 0/1, gradients flow
    through the relaxed sigmoid. Eval: deterministic threshold sigmoid(logit) > 1/2.
    """
    if training:
        uniform = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
        logistic_noise = uniform.log() - (-uniform).log1p()  # Logistic(0, 1)
        soft = torch.sigmoid((logits + logistic_noise) / temperature)
        hard = (soft > 0.5).to(logits.dtype)
        return hard + soft - soft.detach()  # straight-through
    return (logits > 0).to(logits.dtype)


class SpartanLayer(nn.Module):
    """One sparse-attention transformer layer (Eqs. 3-4)."""

    def __init__(
        self, dim: int, mlp_hidden_size: int, mlp_num_layers: int, temperature: float
    ) -> None:
        """Build the layer.

        Args:
            dim: Working (embedding) dimension inside the transformer.
            mlp_hidden_size: Hidden width of the per-token MLP.
            mlp_num_layers: Number of Linear layers in the MLP (App. A.1: 3).
            temperature: Gumbel-softmax temperature for adjacency sampling.
        """
        super().__init__()
        if mlp_num_layers < 2:
            raise ValueError("mlp_num_layers must be >= 2")
        self.temperature = temperature
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
        self, tokens: Float[Tensor, "b t d"]
    ) -> tuple[Float[Tensor, "b t d"], Float[Tensor, "b t t"], Float[Tensor, ""]]:
        """Apply hard-masked attention; return (tokens, adjacency A_l, logit penalty).

        The logit penalty is Baumgartner et al. Eq. 11 per layer:
        mean_ij [exp(q_i·k_j) + exp(-q_i·k_j)] — penalises large attention
        logits so the softmax keeps gradient during the pruning phase (their
        F.4 ablation: without it the path loss plateaus). Logits are clamped
        at ±10 inside the penalty only, to keep exp() finite; the gradient
        still pushes oversized logits down.
        """
        normed = self.norm(tokens)
        q = self.project_q(normed)
        k = self.project_k(normed)
        v = self.project_v(normed)

        # Eq. 3 writes sigmoid(q_i·k_j) with no 1/sqrt(d), but an unscaled
        # 128-dim dot product is ~N(0, 4^2) at standard init: gates start
        # saturated and the Eq. 11 exp-penalty starts at e^10, which destroys
        # early training (bang-bang logit oscillation, Adam second-moment
        # blow-up; run rung1_seed0 2026-07-11). We read q·k in Eqs. 3/11 as the
        # conventionally scaled attention logit — the only init-trainable
        # reading, and the only one under which Eq. 11's "keep logits small"
        # aim is coherent. INTERPRETATION, not paper-literal (no public code).
        adjacency_logits = torch.einsum("bid,bjd->bij", q, k) * self.scale
        adjacency = _sample_hard_adjacency(adjacency_logits, self.temperature, self.training)

        # Eq. 4, masked BEFORE normalization (D10): softmax over unmasked j
        # only, sharing the scaled logits.
        attn_logits = adjacency_logits
        # Row-wise max subtraction over UNMASKED entries only (grad-neutral
        # softmax stabilization). Using the global max here is wrong: when the
        # row's largest logit is masked, every surviving term is exp(<<0), the
        # denominator collapses to the 1e-8 floor, and gradients are amplified
        # ~1e6x (observed as intermittent grad-norm spikes). With the unmasked
        # max the denominator is always >= 1. Fully masked rows fall back to
        # max 0, yielding weights 0 (h_i = 0, no information flow).
        unmasked_max = attn_logits.masked_fill(adjacency == 0, float("-inf")).max(
            dim=-1, keepdim=True
        )
        row_max = torch.where(
            torch.isfinite(unmasked_max.values), unmasked_max.values, 0.0
        ).detach()
        # clamp(max=0) is a no-op for unmasked entries (their logits are
        # <= row_max by construction); it only bounds the straight-through
        # gradient into gates of MASKED entries whose logit exceeds the
        # unmasked max, which would otherwise see exp(positive).
        weights = adjacency * (attn_logits - row_max).clamp(max=0.0).exp()
        denom = weights.sum(dim=-1, keepdim=True)
        # Non-empty rows have denom >= 1 (their max entry contributes
        # exp(0) = 1); only fully masked rows have denom == 0. Adding 1 there
        # (instead of a 1e-8 floor) keeps h_i = 0 while bounding the
        # straight-through gradient into the gates by exp(<=0) <= 1 — with the
        # 1e-8 floor, every fully masked row amplified that gradient by ~1e8,
        # firing exactly during the pruning phase when such rows are common.
        weights = weights / (denom + (denom.detach() < 0.5).to(weights.dtype))
        h = torch.einsum("bij,bjd->bid", weights, v)

        # Eq. 11 exactly within |logit| <= 30; linear continuation beyond
        # (slope exp(30) ~ 1e13) keeps the penalty fp32-finite while acting as
        # an effectively hard wall. The previous |logit| <= 10 cutoff (slope
        # ~2e4) was soft enough for the task gradient to push logits out to
        # ~100, causing the bang-bang logit-loss oscillation and grad-norm
        # spikes to 1e13+ observed in run rung1_seed0 (2026-07-11); logits sit
        # near 10 already at init, i.e. right at the old cutoff.
        magnitude = adjacency_logits.abs()
        core = magnitude.clamp(max=30.0)
        tail = (magnitude - 30.0).clamp(min=0.0)
        logit_penalty = (core.exp() + (-core).exp() + core.exp() * tail).mean()
        return self.mlp(h + tokens), adjacency, logit_penalty  # ŝ_i = MLP(h_i + s_i)


class Spartan(nn.Module):
    """Sparse transformer world model over (S_t, Ŝ^ph, optional U_t) tokens.

    Predicts Ŝ_{t+1} at the state-token positions and exposes the path matrix
    Ā (Eq. 5) whose thresholded state rows are the local causal graph read out
    for the SHD/MCC diagnostics: ``path_matrix[b, i, j] >= 1`` means token j is
    a local causal parent of prediction i in sample b.
    """

    def __init__(
        self,
        slot_size: int,
        num_layers: int = 3,
        embed_dim: int | None = 512,
        mlp_hidden_size: int = 512,
        mlp_num_layers: int = 3,
        temperature: float = 1.0,
        aux_dim: int | None = None,
    ) -> None:
        """Build the predictor.

        Args:
            slot_size: d, dimension of state and parameter tokens (= slot dim).
            num_layers: L, number of stacked sparse-attention layers (A.1: 3).
            embed_dim: Working dimension inside the transformer (App. A.1
                separates "token dimension" from a larger "embedding dimension",
                512); tokens are projected d -> embed_dim on entry and
                embed_dim -> d at the prediction head. None = work at d.
            mlp_hidden_size: Hidden width of each layer's MLP (A.1: 512/1024).
            mlp_num_layers: Linear layers per MLP (A.1: 3).
            temperature: Gumbel-softmax temperature (adjacency sampling).
            aux_dim: Dimension of auxiliary variables U_t; None disables the
                auxiliary pathway entirely (CLEVRER: None, Push-T: action dim).
        """
        super().__init__()
        self.slot_size = slot_size
        self.aux_dim = aux_dim
        dim = embed_dim if embed_dim is not None else slot_size
        self.in_project = nn.Linear(slot_size, dim) if dim != slot_size else nn.Identity()
        self.out_project = nn.Linear(dim, slot_size) if dim != slot_size else nn.Identity()
        self.layers = nn.ModuleList(
            [
                SpartanLayer(dim, mlp_hidden_size, mlp_num_layers, temperature)
                for _ in range(num_layers)
            ]
        )
        # Learned role embeddings (D10): distinguish token roles, keep slot symmetry.
        self.state_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.param_embed = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.state_embed, std=0.02)
        nn.init.normal_(self.param_embed, std=0.02)
        if aux_dim is not None:
            self.aux_project: nn.Linear | None = nn.Linear(aux_dim, dim)
            self.aux_embed: nn.Parameter | None = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.normal_(self.aux_embed, std=0.02)
        else:
            self.aux_project = None
            self.aux_embed = None

    def forward(
        self,
        state: Float[Tensor, "b n d"],
        params: Float[Tensor, "b n d"],
        aux: Float[Tensor, "b m da"] | None = None,
    ) -> SpartanOutput:
        """Predict next-step state slots and expose the causal graph.

        Args:
            state: Kinematic state S_t, (B, N, d).
            params: Causal parameters Ŝ^ph, (B, N, d).
            aux: Optional auxiliary variables U_t, (B, M, aux_dim).

        Returns:
            ``SpartanOutput(prediction, path_matrix, sparsity)`` — prediction
            (B, N, d) at state positions; Ā over the full token set with order
            [state 0..N-1 | params N..2N-1 | aux 2N..]; |Ā| averaged over batch.
        """
        if state.shape != params.shape or state.ndim != 3:
            raise ValueError(
                f"state/params must both be (B, N, d), got "
                f"{tuple(state.shape)} vs {tuple(params.shape)}"
            )
        num_slots = state.shape[1]
        pieces = [
            self.in_project(state) + self.state_embed,
            self.in_project(params) + self.param_embed,
        ]
        if aux is not None:
            if self.aux_project is None or self.aux_embed is None:
                raise ValueError("aux passed but model built with aux_dim=None")
            pieces.append(self.aux_project(aux) + self.aux_embed)
        tokens = torch.cat(pieces, dim=1)  # (B, T, embed_dim)

        adjacencies: list[Tensor] = []
        logit_penalties: list[Tensor] = []
        for layer in self.layers:
            tokens, adjacency, layer_logit_penalty = layer(tokens)
            adjacencies.append(adjacency)
            logit_penalties.append(layer_logit_penalty)

        # Eq. 5: Ā = (A_L + I)···(A_1 + I); Ā_ij = number of paths j → i.
        eye = torch.eye(tokens.shape[1], device=tokens.device, dtype=tokens.dtype)
        path_matrix = eye.expand(tokens.shape[0], -1, -1)
        for adjacency in adjacencies:
            path_matrix = (adjacency + eye) @ path_matrix

        # Eq. 6: |Ā| — includes the constant pure-residual diagonal contribution,
        # which carries zero gradient (documented; matches the paper's |Ā|).
        sparsity = path_matrix.sum(dim=(1, 2)).mean()

        return SpartanOutput(
            prediction=self.out_project(tokens[:, :num_slots]),
            path_matrix=path_matrix,
            sparsity=sparsity,
            logit_penalty=torch.stack(logit_penalties).mean(),
        )


__all__ = ["Spartan", "SpartanLayer", "SpartanOutput"]
