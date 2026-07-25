"""Trajectory-level alignment of visual tracks to simulator rows (experiments.pdf Eqs. 98-100).

The state-to-state regime needs none of this: true states arrive in simulator-row order, so
parameter row ``i`` is attached to physical object ``i`` by construction and the
evaluation alignment is the identity (Eq. 132). The visual regimes read anonymous
recurrent visual tracks whose ordering may differ from the simulator's in every
episode, so the predictions and the targets live in different orders and must be
matched before any loss is taken.

The matching is deliberately constrained in three ways (S6.4, S6.5):

1. **One assignment per episode, not per frame.** Eq. 98 sums the cost over the
   whole prediction window before Eq. 99 solves it. Per-timestep rematching is
   explicitly forbidden, because a mid-episode identity switch would then be
   absorbed silently instead of showing up as loss.
2. **Detached.** No gradient flows through the discrete assignment; gradients
   reach the model only through the selected predictions.
3. **Standardized only to choose the permutation.** Eq. 98 divides by frozen
   training-split coordinate scales so position and velocity errors are
   comparable while ranking permutations, but the loss itself (Eq. 101) is raw
   unstandardized MSE.

The same assignment is reused for the prediction loss, the parameter evaluation
and every learned-graph axis in that episode, so those three never disagree
about which visual track is which physical object.
"""

import torch
from jaxtyping import Float, Int
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.utils.data import Dataset

__all__ = [
    "align_to_assignment",
    "assignment_cost",
    "coordinate_scales",
    "invert_assignment",
    "trajectory_assignment",
]


def invert_assignment(assignment: Int[Tensor, "b n"]) -> Int[Tensor, "b n"]:
    """Invert a per-episode permutation.

    ``zeta[i] = k`` reads "visual track i is physical object k". The inverse
    answers "which visual track is physical object k", which is what pooling
    learned coordinates across episodes needs: a per-episode relabelling is only
    safe once every episode has been mapped into the SAME global order, and the
    only global order available is the physical one (Eq. 139).
    """
    if assignment.ndim != 2:
        raise ValueError(f"assignment must be (B, N), got {tuple(assignment.shape)}")
    return assignment.argsort(dim=1)


def coordinate_scales(
    dataset: Dataset[dict[str, Tensor]],
    num_episodes: int = 2000,
    context_len: int = 30,
    eps: float = 1e-6,
) -> Float[Tensor, " k"]:
    """Frozen per-coordinate standard deviations sigma_a of Eq. 98.

    Computed once on the TRAINING split over the prediction window and then held
    fixed: a scale that drifted with the model would make the assignment cost
    non-comparable across steps. Only the ranking of permutations depends on it,
    so modest sampling error here is harmless.

    Args:
        dataset: Episode dataset yielding ``states`` of shape (T, N, k).
        num_episodes: Episodes to average over (capped at the dataset length).
        context_len: Tpar; only targets Z_{Tpar..T-1} enter the window.
        eps: Floor added to each std so a constant coordinate cannot divide by 0.
    """
    length = len(dataset)  # pyright: ignore[reportArgumentType]
    total = min(num_episodes, int(length))
    if total < 2:
        raise ValueError(f"need at least 2 episodes to estimate scales, got {total}")
    window = [dataset[i]["states"][context_len:] for i in range(total)]
    stacked = torch.stack(window).flatten(0, 2)  # (episodes * steps * tracks, k)
    return stacked.std(dim=0) + eps


def assignment_cost(
    predictions: Float[Tensor, "b k n c"],
    targets: Float[Tensor, "b k n c"],
    scales: Float[Tensor, " c"],
) -> Float[Tensor, "b n n"]:
    """Eq. 98's standardized trajectory cost ``D_e(i, j)``.

    Entry ``[b, i, j]`` is the mean squared standardized error between predicted
    visual track ``i`` and physical track ``j``, averaged over the whole window
    and all coordinates. Always detached — this quantity only ranks permutations.
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions {tuple(predictions.shape)} and targets {tuple(targets.shape)} "
            f"must have the same shape"
        )
    if predictions.ndim != 4:
        raise ValueError(f"expected (B, K, N, C), got {tuple(predictions.shape)}")
    if scales.shape != predictions.shape[-1:]:
        raise ValueError(f"scales must have shape {tuple(predictions.shape[-1:])}")
    with torch.no_grad():
        # (B, K, N_pred, N_true, C): predicted track on dim 2, physical on dim 3.
        difference = predictions.detach().unsqueeze(3) - targets.detach().unsqueeze(2)
        return (difference / scales).square().mean(dim=(1, 4))


def trajectory_assignment(
    predictions: Float[Tensor, "b k n c"],
    targets: Float[Tensor, "b k n c"],
    scales: Float[Tensor, " c"],
) -> Int[Tensor, "b n"]:
    """Eq. 99: one detached Hungarian assignment per episode.

    Returns ``pi`` with ``pi[b, i] = j``, meaning predicted visual track ``i`` of
    episode ``b`` is matched to physical track ``j``. Use it to move the TARGETS
    into visual-track order (:func:`align_to_assignment`); nothing is ever
    permuted inside the predictor, so latent state, parameter, key, prediction
    and graph node ``i`` all keep referring to visual track ``i``.
    """
    cost = assignment_cost(predictions, targets, scales)
    rows = [linear_sum_assignment(episode.cpu().numpy())[1] for episode in cost]
    return torch.as_tensor(
        [row.tolist() for row in rows], dtype=torch.long, device=predictions.device
    )


def align_to_assignment(
    values: Tensor,
    assignment: Int[Tensor, "b n"],
    track_dim: int,
) -> Tensor:
    """Reorder ``values`` along ``track_dim`` into visual-track order (Eq. 100).

    ``values`` holds a physical-track-indexed quantity (true states, masses, a
    ground-truth graph axis); the result is indexed by visual track, so
    ``result[b, ..., i, ...]`` is the physical object matched to visual track
    ``i``. Applying this to targets rather than to predictions is what keeps the
    predictor's own ordering untouched.
    """
    if assignment.ndim != 2:
        raise ValueError(f"assignment must be (B, N), got {tuple(assignment.shape)}")
    if values.shape[0] != assignment.shape[0]:
        raise ValueError(
            f"batch mismatch: values {values.shape[0]} vs assignment {assignment.shape[0]}"
        )
    axis = track_dim if track_dim >= 0 else values.ndim + track_dim
    if not 1 <= axis < values.ndim:
        raise ValueError(f"track_dim {track_dim} must select a non-batch axis of {values.ndim}D")
    batch, num_tracks = assignment.shape
    if values.shape[axis] != num_tracks:
        raise ValueError(
            f"axis {axis} has size {values.shape[axis]}, assignment covers {num_tracks}"
        )
    view = [batch] + [1] * (values.ndim - 1)
    view[axis] = num_tracks
    index = assignment.reshape(view).expand(values.shape)
    return values.gather(axis, index)
