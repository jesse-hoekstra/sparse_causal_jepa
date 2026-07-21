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
    Eq. 6   sparsity penalty |Ā| (sum of paths into decoded outputs) —
            returned as ``SpartanOutput.sparsity``; the Lagrangian-relaxation
            schedule for its weight (App. A.2) lives in the training loop, not
            here. The full-token path matrix is retained for graph diagnostics.

Choices the paper leaves open (flagged per project policy, decisions D10):
    * D19 rollout gate coupling: the paper's objective (Eq. 6) is
      single-transition, so it prescribes only the per-step Bernoulli MARGINAL
      of Eq. 3 and is silent on how draws couple across a D16 rollout chain
      (a situation that does not exist upstream). We draw each layer's
      logistic thresholds once per chain and reuse them across its steps
      (``sample_gate_noise``): same marginals, common-threshold coupling
      within a chain, independent across chains. I.i.d. per-step redraws are
      catastrophically unstable under Tp=30 BPTT (decisions.md D19).
    * Eq. 4's printed normalization is ambiguous; we renormalize the softmax
      over the UNMASKED entries only. Masking after normalization would leak
      masked tokens' information through the denominator, contradicting the
      paper's "adjacency ... disallows information flows" claim.
    * Single-head attention per layer (Eqs. 3-4 define one adjacency per layer).
    * Eval mode is deterministic: A_ij = 1 iff sigmoid(q_i·k_j) > 1/2.
    * Learned role embeddings (state/param/aux) are added to the tokens so the
      roles are distinguishable while slot-permutation equivariance is kept.
    * Optional learned state-node and parameter-coordinate embeddings give
      persistent graph nodes distinct addresses.  The two tables are
      independent and encode no state-i/parameter-i correspondence; fixed node
      identities are appropriate for ordered GT states but require tracking or
      recurrent slot continuation before use with anonymous visual slots.
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
from torch.nn import functional as F


class SpartanOutput(NamedTuple):
    """Everything the losses and the SHD/MCC eval need from one forward pass."""

    prediction: Float[Tensor, "b n d"]
    path_matrix: Float[Tensor, "b t t"]
    sparsity: Float[Tensor, ""]
    logit_penalty: Float[Tensor, ""]
    mean_abs_logit: Float[Tensor, ""]
    mean_gate_probability: Float[Tensor, ""]
    gate_entropy: Float[Tensor, ""]


def _logistic_noise(like: Tensor) -> Tensor:
    """Logistic(0,1) threshold noise (= difference of two Gumbels), shaped like ``like``."""
    uniform = torch.rand_like(like).clamp(1e-6, 1 - 1e-6)
    return uniform.log() - (-uniform).log1p()


def _sample_hard_adjacency(
    logits: Tensor, temperature: float, training: bool, noise: Tensor | None = None
) -> Tensor:
    """Hard {0,1} adjacency from Bernoulli logits (Eq. 3).

    Training: binary Gumbel-softmax (Logistic reparameterization) with a
    straight-through estimator — forward is exactly 0/1, gradients flow
    through the relaxed sigmoid. ``noise`` optionally injects the logistic
    threshold tensor; None draws fresh noise (the single-transition behavior
    of the paper). D19 rollout chains draw the noise ONCE per chain and pass
    it to every step, so P(open) = sigmoid(logit) at every step (marginals
    exactly Eq. 3) but within one chain a gate flips only when its
    state-dependent logit crosses the chain's fixed threshold.
    Eval: deterministic threshold sigmoid(logit) > 1/2 (noise ignored).
    """
    if training:
        if noise is None:
            noise = _logistic_noise(logits)
        soft = torch.sigmoid((logits + noise) / temperature)
        hard = (soft > 0.5).to(logits.dtype)
        return hard + soft - soft.detach()  # straight-through
    return (logits > 0).to(logits.dtype)


class SpartanLayer(nn.Module):
    """One sparse-attention transformer layer (Eqs. 3-4)."""

    def __init__(
        self,
        dim: int,
        mlp_hidden_size: int,
        mlp_num_layers: int,
        temperature: float,
        dense: bool = False,
        identity: bool = False,
    ) -> None:
        """Build the layer.

        Args:
            dim: Working (embedding) dimension inside the transformer.
            mlp_hidden_size: Hidden width of the per-token MLP.
            mlp_num_layers: Number of Linear layers in the MLP (App. A.1: 3).
            temperature: Gumbel-softmax temperature for adjacency sampling.
            dense: A ≡ 1 — no gate sampling, standard softmax attention. This
                is SPARTAN's "fully connected model" (p.16), the reference
                whose loss defines τ. The gated model with sparsity disabled is
                NOT that reference: its gates keep sampling Bernoulli masks and inject
                masking noise, inflating the measured loss (audit F-8).
            identity: A ≡ 0 — attention output is zero for every token; only
                the residual + MLP path survives, so each token is predicted
                from itself alone (path matrix exactly I). This is the
                mass-blind reference of the D16 go/no-go: its converged loss is
                the best any model without cross-token (incl. param→state)
                edges can achieve, the floor τ must sit BELOW for sparsity to
                be forced to keep true edges. Mutually exclusive with dense.
        """
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
        self,
        tokens: Float[Tensor, "b t d"],
        gate_noise: Float[Tensor, "b t t"] | None = None,
    ) -> tuple[
        Float[Tensor, "b t d"],
        Float[Tensor, "b t t"],
        Float[Tensor, ""],
        Float[Tensor, ""],
        Float[Tensor, ""],
        Float[Tensor, ""],
    ]:
        """Apply hard-masked attention and return outputs plus logit diagnostics.

        ``gate_noise``: optional pre-drawn logistic thresholds for this layer's
        gates (D19: one draw per rollout chain, reused across its steps);
        None = fresh draw (single-step behavior). Ignored in dense/identity
        modes and in eval mode.

        The logit penalty is Baumgartner et al. Eq. 11 per layer:
        mean_ij [exp(q_i·k_j) + exp(-q_i·k_j)] — penalises large attention
        logits so the softmax keeps gradient during the pruning phase (their
        F.4 ablation: without it the path loss plateaus). The exact exponential
        is used through absolute logit 30 and continued linearly beyond that point to
        keep fp32 finite while retaining a strong restoring gradient.
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
        if self.dense:
            adjacency = torch.ones_like(adjacency_logits)
        elif self.identity:
            # A ≡ 0: every row fully masked -> h_i = 0 downstream (the
            # fully-masked-row path of the softmax below), tokens pass through
            # residual + MLP only. q/k still feed the Eq. 11 penalty so
            # constraint_loss stays the same quantity as in the other modes.
            adjacency = torch.zeros_like(adjacency_logits)
        else:
            adjacency = _sample_hard_adjacency(
                adjacency_logits, self.temperature, self.training, noise=gate_noise
            )

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
        # Read-only saturation diagnostics. Detaching avoids carrying an
        # otherwise unused entropy graph through every training step. The
        # stable Bernoulli identity is H(sigmoid(z)) = softplus(z) - z sigmoid(z).
        with torch.no_grad():
            diagnostic_logits = adjacency_logits.detach()
            gate_probability = torch.sigmoid(diagnostic_logits)
            gate_entropy = (
                F.softplus(diagnostic_logits) - diagnostic_logits * gate_probability
            ).mean()
        return (
            self.mlp(h + tokens),
            adjacency,
            logit_penalty,
            magnitude.detach().mean(),
            gate_probability.mean(),
            gate_entropy,
        )  # ŝ_i = MLP(h_i + s_i)


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
        param_size: int | None = None,
        dense: bool = False,
        identity: bool = False,
        node_embeddings: bool = False,
        num_slots: int | None = None,
    ) -> None:
        """Build the predictor.

        Args:
            slot_size: d_s, dimension of the STATE tokens (= prediction dim).
            param_size: Dimension of the parameter tokens Ŝ^ph when it differs
                from ``slot_size`` (D20 gt_states regime: state tokens are raw
                k-dim GT states while Ŝ^ph stays in slot space). None = params
                share ``slot_size`` and the state input projection (existing
                behavior; state_dict unchanged).
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
            dense: A ≡ 1 in every layer — SPARTAN's "fully connected model"
                (p.16), used ONLY for τ calibration (run with
                train.sparsity_enabled=false; the sparsity loss is meaningless
                here since |Ā| is a dense constant).
            identity: A ≡ 0 in every layer — path matrix exactly I, each state
                token predicted from its own past only: the mass-blind
                reference for the D16 go/no-go (run with
                train.sparsity_enabled=false, matched budget vs the dense
                reference; the gap between their constraint losses is the τ
                window). Mutually exclusive with dense.
            node_embeddings: Add independent learned addresses to the state
                nodes and parameter-coordinate nodes. This makes the nodes
                distinguishable without privileging any state/parameter edge.
            num_slots: Fixed number of state and parameter nodes. Required when
                ``node_embeddings=True``; the current architecture uses the
                same count for both blocks.
        """
        super().__init__()
        if dense and identity:
            raise ValueError("dense and identity are mutually exclusive")
        if node_embeddings and (num_slots is None or num_slots <= 0):
            raise ValueError("node_embeddings=True requires a positive num_slots")
        self.slot_size = slot_size
        self.param_size = param_size if param_size is not None else slot_size
        self.aux_dim = aux_dim
        self.dense = dense
        self.identity = identity
        self.node_embeddings = node_embeddings
        self.num_slots = num_slots
        dim = embed_dim if embed_dim is not None else slot_size
        self.in_project = nn.Linear(slot_size, dim) if dim != slot_size else nn.Identity()
        # Separate param-token projection ONLY when Ŝ^ph lives in a different
        # space (D20); None keeps the shared projection and an unchanged
        # state_dict, so pre-D20 checkpoints load strictly.
        if param_size is None or param_size == slot_size:
            self.param_project: nn.Module | None = None
        else:
            self.param_project = nn.Linear(param_size, dim) if dim != param_size else nn.Identity()
        self.out_project = nn.Linear(dim, slot_size) if dim != slot_size else nn.Identity()
        self.layers = nn.ModuleList(
            [
                SpartanLayer(
                    dim,
                    mlp_hidden_size,
                    mlp_num_layers,
                    temperature,
                    dense=dense,
                    identity=identity,
                )
                for _ in range(num_layers)
            ]
        )
        # Learned role embeddings (D10): distinguish token roles, keep slot symmetry.
        self.state_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.param_embed = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.state_embed, std=0.02)
        nn.init.normal_(self.param_embed, std=0.02)
        if node_embeddings:
            assert num_slots is not None
            # Separate tables expose node addresses only.  They are neither
            # shared nor initialized alike, so index equality carries no
            # built-in state←parameter preference.
            self.state_node_embed: nn.Parameter | None = nn.Parameter(
                torch.empty(1, num_slots, dim)
            )
            self.param_node_embed: nn.Parameter | None = nn.Parameter(
                torch.empty(1, num_slots, dim)
            )
            nn.init.normal_(self.state_node_embed, std=0.02)
            nn.init.normal_(self.param_node_embed, std=0.02)
        else:
            self.register_parameter("state_node_embed", None)
            self.register_parameter("param_node_embed", None)
        if aux_dim is not None:
            self.aux_project: nn.Linear | None = nn.Linear(aux_dim, dim)
            self.aux_embed: nn.Parameter | None = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.normal_(self.aux_embed, std=0.02)
        else:
            self.aux_project = None
            self.aux_embed = None

    def sample_gate_noise(
        self,
        state: Float[Tensor, "b n d"],
        params: Float[Tensor, "b n d"],
        aux: Float[Tensor, "b m da"] | None = None,
    ) -> list[Tensor] | None:
        """Draw one logistic-threshold tensor per layer for a rollout chain (D19).

        The rollout caller draws this ONCE per chain and passes it to every
        ``forward`` of that chain. Per-step marginals are exactly Eq. 3's
        Bernoulli — P(open) = sigmoid(logit) at every step — but the chain's
        draws share one threshold (common random numbers) instead of being
        independent: a gate then flips mid-chain only when its state-dependent
        logit crosses the fixed threshold (collision physics), never from
        re-rolled noise at unchanged state. The i.i.d. per-step coupling makes
        straight-through gradients through Tp x L stacked resampled masks
        explode at mid density (runs 7wupt6pw / 0ta5ymcw / u94wqvcb; decisions
        D19). Returns None in dense/identity mode — nothing is sampled there.
        """
        if state.shape[:2] != params.shape[:2]:
            raise ValueError(
                "state and params must share batch/slot axes, got "
                f"{tuple(state.shape[:2])} vs {tuple(params.shape[:2])}"
            )
        if self.dense or self.identity:
            return None
        num_tokens = 2 * state.shape[1] + (aux.shape[1] if aux is not None else 0)
        template = state.new_empty(state.shape[0], num_tokens, num_tokens)
        return [_logistic_noise(template) for _ in self.layers]

    def forward(
        self,
        state: Float[Tensor, "b n d"],
        params: Float[Tensor, "b n d"],
        aux: Float[Tensor, "b m da"] | None = None,
        gate_noise: list[Tensor] | None = None,
    ) -> SpartanOutput:
        """Predict next-step state slots and expose the causal graph.

        Args:
            state: Kinematic state S_t, (B, N, d).
            params: Causal parameters Ŝ^ph, (B, N, d).
            aux: Optional auxiliary variables U_t, (B, M, aux_dim).
            gate_noise: Optional per-layer gate thresholds from
                ``sample_gate_noise`` (D19 rollout chains); None = fresh noise
                per layer call (single-transition behavior, paper-literal).

        Returns:
            ``SpartanOutput(prediction, path_matrix, sparsity)`` — prediction
            (B, N, d) at state positions; Ā over the full token set with order
            [state 0..N-1 | params N..2N-1 | aux 2N..]; the sum of Ā's
            decoded state rows, averaged over the batch, as ``sparsity``.
        """
        if (
            state.ndim != 3
            or params.ndim != 3
            or state.shape[:2] != params.shape[:2]
            or state.shape[2] != self.slot_size
            or params.shape[2] != self.param_size
        ):
            raise ValueError(
                f"state must be (B, N, {self.slot_size}) and params "
                f"(B, N, {self.param_size}), got {tuple(state.shape)} vs "
                f"{tuple(params.shape)}"
            )
        num_slots = state.shape[1]
        param_project = self.param_project if self.param_project is not None else self.in_project
        state_tokens = self.in_project(state) + self.state_embed
        param_tokens = param_project(params) + self.param_embed
        if self.node_embeddings:
            if num_slots != self.num_slots:
                raise ValueError(
                    f"received {num_slots} slots, expected configured num_slots={self.num_slots}"
                )
            assert self.state_node_embed is not None
            assert self.param_node_embed is not None
            state_tokens = state_tokens + self.state_node_embed
            param_tokens = param_tokens + self.param_node_embed
        pieces = [
            state_tokens,
            param_tokens,
        ]
        if aux is not None:
            if self.aux_project is None or self.aux_embed is None:
                raise ValueError("aux passed but model built with aux_dim=None")
            pieces.append(self.aux_project(aux) + self.aux_embed)
        tokens = torch.cat(pieces, dim=1)  # (B, T, embed_dim)

        if gate_noise is not None and len(gate_noise) != len(self.layers):
            raise ValueError(
                f"gate_noise has {len(gate_noise)} tensors for {len(self.layers)} layers"
            )
        adjacencies: list[Tensor] = []
        logit_penalties: list[Tensor] = []
        mean_abs_logits: list[Tensor] = []
        mean_gate_probabilities: list[Tensor] = []
        gate_entropies: list[Tensor] = []
        for index, layer in enumerate(self.layers):
            (
                tokens,
                adjacency,
                layer_logit_penalty,
                layer_mean_abs_logit,
                layer_mean_gate_probability,
                layer_gate_entropy,
            ) = layer(
                tokens,
                gate_noise=gate_noise[index] if gate_noise is not None else None,
            )
            adjacencies.append(adjacency)
            logit_penalties.append(layer_logit_penalty)
            mean_abs_logits.append(layer_mean_abs_logit)
            mean_gate_probabilities.append(layer_mean_gate_probability)
            gate_entropies.append(layer_gate_entropy)

        # Eq. 5: Ā = (A_L + I)···(A_1 + I); Ā_ij = number of paths j → i.
        eye = torch.eye(tokens.shape[1], device=tokens.device, dtype=tokens.dtype)
        path_matrix = eye.expand(tokens.shape[0], -1, -1)
        for adjacency in adjacencies:
            path_matrix = (adjacency + eye) @ path_matrix

        # Eq. 6 adapted to this project's heterogeneous token layout: only the
        # first N rows are decoded as predictions. Paths ending at parameter or
        # auxiliary rows after the final layer cannot affect the loss, so
        # charging for them distorts the graph objective. Keep the complete
        # path matrix above for graph readout/debugging, but penalize precisely
        # the paths into decoded state outputs. The residual state diagonal is
        # constant and therefore contributes no gradient.
        sparsity = path_matrix[:, :num_slots].sum(dim=(1, 2)).mean()

        return SpartanOutput(
            prediction=self.out_project(tokens[:, :num_slots]),
            path_matrix=path_matrix,
            sparsity=sparsity,
            logit_penalty=torch.stack(logit_penalties).mean(),
            mean_abs_logit=torch.stack(mean_abs_logits).mean(),
            mean_gate_probability=torch.stack(mean_gate_probabilities).mean(),
            gate_entropy=torch.stack(gate_entropies).mean(),
        )


__all__ = ["Spartan", "SpartanLayer", "SpartanOutput"]
