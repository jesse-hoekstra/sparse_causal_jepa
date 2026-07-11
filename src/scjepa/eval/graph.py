"""Graph diagnostics: ground-truth local graphs, SPARTAN readout, and SHD.

Conventions come from one source of truth each:
- Ground truth: ``scjepa.data.bounce`` docstring (D11). For the training pair
  (context ending at frame t, target frame t+1) the relevant local graph is the
  LAST transition's contacts. State edge j→i iff contact (+ self-edges: free
  flight); parameter edge mass_j → state_i iff a contact involving i and j —
  including a ball's own mass, which matters only while colliding.
- Learned graph: ``scjepa.models.spartan`` (D10). Token order [state | params |
  aux]; token j is a local causal parent of prediction i iff path_matrix[i, j]
  >= 1. The state→state block is ``[:N, :N]``, the parameter→state block is
  ``[:N, N:2N]``.

SHD here is the Hamming distance between binary adjacency matrices (edge
insertions + deletions; no orientation term since both graphs are directed with
fixed node identity). Slot↔object alignment is the CALLER's job: with
ground-truth embeddings slot i is object i by construction; with learned slots,
align first (Hungarian / probe) and permute before calling.
"""

import torch
from jaxtyping import Bool, Float
from torch import Tensor


def gt_graphs_from_contacts(
    contacts: Bool[Tensor, "b tm1 n n"],
) -> tuple[Bool[Tensor, "b n n"], Bool[Tensor, "b n n"]]:
    """Derive (state_graph, param_graph) for the predicted transition.

    Args:
        contacts: Per-transition contact record, (B, T-1, N, N); the last
            transition is the one the model predicts (D6 single-step).

    Returns:
        state_graph[b, i, j]: state j influences state i (contact or i == j).
        param_graph[b, i, j]: mass j influences state i (any contact involving
            i and j; the diagonal is True iff ball i is in some contact).
    """
    if contacts.ndim != 4:
        raise ValueError(f"expected (B, T-1, N, N), got {tuple(contacts.shape)}")
    last = contacts[:, -1]
    eye = torch.eye(last.shape[-1], dtype=torch.bool, device=last.device)
    state_graph = last | eye
    involved = last.any(dim=-1)  # (B, N): ball is in at least one contact
    param_graph = last | torch.diag_embed(involved)
    return state_graph, param_graph


def read_learned_graphs(
    path_matrix: Float[Tensor, "b t t"], num_slots: int
) -> tuple[Bool[Tensor, "b n n"], Bool[Tensor, "b n n"]]:
    """Threshold SPARTAN's path matrix into (state_graph, param_graph).

    ``path_matrix[i, j] >= 1`` means at least one unmasked path token j →
    prediction i (Eq. 5 of the SPARTAN paper); entries are integer counts.
    """
    if path_matrix.shape[-1] < 2 * num_slots:
        raise ValueError(f"path matrix has {path_matrix.shape[-1]} tokens, need >= {2 * num_slots}")
    state_block = path_matrix[:, :num_slots, :num_slots]
    param_block = path_matrix[:, :num_slots, num_slots : 2 * num_slots]
    return state_block >= 0.5, param_block >= 0.5


def structural_hamming_distance(
    learned: Bool[Tensor, "b n n"], target: Bool[Tensor, "b n n"]
) -> Float[Tensor, ""]:
    """Mean per-sample Hamming distance between binary adjacency matrices."""
    if learned.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(learned.shape)} vs {tuple(target.shape)}")
    return (learned != target).sum(dim=(-2, -1)).float().mean()


__all__ = ["gt_graphs_from_contacts", "read_learned_graphs", "structural_hamming_distance"]
