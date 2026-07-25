"""Graph diagnostics: the ground-truth local causal graph, SPARTAN's readout, and SHD.

ONE graph metric, `shd` (D28): the Structural Hamming Distance between the
learned graph and the ground-truth causal graph, as reported by SPARTAN
(sources/SPARTAN.pdf Table 1 / §4.1 "Graph Learning": "we evaluate the Structural
Hamming Distance, a commonly used metric in graph structure learning, between the
learned graphs and the ground-truth"; App. D p.19 repeats the phrasing). Lower is
better — their Table 1 and Table 7 captions both say so. Baumgartner et al. do
not use SHD at all.

The graph is the whole decoded-output graph, not a sub-block: rows are the N
predicted next-states, columns are all 2N source tokens [state | params], so SHD
ranges over [0, 2N²] (= [0, 50] for five balls). This is the same index set as
the path objective and path_density (write-up Eq. 11), so the pruning curve and the
graph score describe the same object. Parameter-token rows are excluded because
they are never decoded and therefore carry no "parent of a prediction" meaning.

Conventions come from one source of truth each:
- Ground truth: ``scjepa.data.bounce`` docstring (D11), for the transition the
  model predicts (the LAST transition in the passed contact window).
  State edge j→i iff contact or i == j (free flight); parameter edge mass_j →
  state_i iff a contact involving i and j — including a ball's own mass at
  ball-ball collisions, and at wall bounces in the radius-proportional-to-mass variant (recorded
  on the contacts diagonal; audit G1).
- Learned graph: ``scjepa.models.spartan`` (D10) and SPARTAN Eq. 5 — Ā_ij counts
  paths j → i, and "s^t_j is a local causal parent of s^{t+1}_i iff Ā_ij >= 1".
  Token order is [state | params | aux]; path counts are integers, so the >= 1
  test is applied as >= 0.5.

What is compared is REACHABILITY agreement, not verified causal use: a retained
edge means information *can* flow, not that the model relies on it.

SHD is the Hamming distance between binary adjacency matrices (edge insertions +
deletions; no orientation term since both graphs are directed), averaged over
samples and left unnormalised. Node alignment is the CALLER's job. In the
GT-state experiment every axis is already in tracked-object order, so no
permutation is applied.

CAUTION (verified, see D28): on bounce the ground truth is sparse — ~7.9 true
edges out of 50 — so predicting the EMPTY graph scores ~2.9 while a saturated
graph scores ~42.1. Since lower is better, SHD alone rewards a model that learns
nothing. It is only meaningful read together with `mcc` and `pred_loss`.
"""

import torch
from jaxtyping import Bool, Float
from torch import Tensor


def gt_causal_graph_from_contacts(
    contacts: Bool[Tensor, "b tm1 n n"],
) -> Bool[Tensor, "b n n2"]:
    """Derive the true local causal graph into the decoded next-states.

    Args:
        contacts: Per-transition contact record, (B, T-1, N, N); the last
            transition is the one the model predicts (D6 single-step).

    Returns:
        graph[b, i, j] for j < N: state j influences state i (contact or i == j).
        graph[b, i, N + j]: mass j influences state i (any contact involving i
        and j; the mass diagonal is True iff ball i is in some contact —
        ball-ball, or a wall bounce recorded on the contacts diagonal).
        Column order matches SPARTAN's token order [state | params].
    """
    if contacts.ndim != 4:
        raise ValueError(f"expected (B, T-1, N, N), got {tuple(contacts.shape)}")
    last = contacts[:, -1]
    eye = torch.eye(last.shape[-1], dtype=torch.bool, device=last.device)
    state_graph = last | eye
    involved = last.any(dim=-1)  # (B, N): ball is in at least one contact
    param_graph = last | torch.diag_embed(involved)
    return torch.cat((state_graph, param_graph), dim=-1)


def read_learned_graph(
    path_matrix: Float[Tensor, "b t t"], num_slots: int
) -> Bool[Tensor, "b n n2"]:
    """Threshold SPARTAN's path matrix into the learned local causal graph.

    Keeps the ``num_slots`` decoded state rows and all ``2 * num_slots`` state
    and parameter source columns. ``path_matrix[i, j] >= 1`` means at least one
    unmasked path token j → prediction i (SPARTAN Eq. 5); entries are integer
    counts, so the test is applied as ``>= 0.5``.
    """
    if path_matrix.shape[-1] < 2 * num_slots:
        raise ValueError(f"path matrix has {path_matrix.shape[-1]} tokens, need >= {2 * num_slots}")
    return path_matrix[:, :num_slots, : 2 * num_slots] >= 0.5


def structural_hamming_distance(
    learned: Bool[Tensor, "b n n2"], target: Bool[Tensor, "b n n2"]
) -> Float[Tensor, ""]:
    """Mean per-sample Hamming distance between binary adjacency matrices."""
    if learned.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(learned.shape)} vs {tuple(target.shape)}")
    return (learned != target).sum(dim=(-2, -1)).float().mean()


__all__ = [
    "gt_causal_graph_from_contacts",
    "read_learned_graph",
    "structural_hamming_distance",
]
