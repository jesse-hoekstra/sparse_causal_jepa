"""Evaluation-only alignment of visual tracks to physical objects (experiments.pdf Eqs. 132-140).

The state-to-state regime needs nothing here: true states arrive in physical-object order, so
zeta = id (Eq. 132). In the visual experiments the recurrent tracks are anonymous
and their row ordering may differ between episodes even when tracking is perfect,
so learned parameter coordinates cannot be pooled across episodes until each
track is matched to a physical object.

This matching is deliberately DIFFERENT from the training-time assignment in
``scjepa.losses.alignment``:

* it uses trajectory GEOMETRY — the centroid of each slot's spatial allocation
  against the true rendered centre (Eqs. 134-137) — not prediction error, so it
  cannot be gamed by a model that predicts well for the wrong reason;
* it never sees the learned parameter values or the true masses, so it cannot be
  a permutation chosen to flatter the recovery score;
* it is computed once from the complete held-out trajectory and sends no
  gradient anywhere.

The visual-to-visual regime has no true-state prediction to match on at all, which is precisely
why the geometric route exists.
"""

import torch
from jaxtyping import Float, Int
from scipy.optimize import linear_sum_assignment
from torch import Tensor

__all__ = ["physical_assignment", "slot_centroids", "spatial_grid"]


def spatial_grid(resolution: int, device: torch.device | None = None) -> Float[Tensor, "p 2"]:
    """Normalized (x, y) coordinate of each spatial position, in renderer order.

    Position ``p = row * resolution + col`` and the renderer maps y to the image
    ROW (``grid_y`` from ``meshgrid(..., indexing='ij')``), so the returned pairs
    are (x from the column, y from the row) — the same order as the ``[x, y]``
    prefix of a true state. Getting this backwards would silently transpose every
    assignment.
    """
    axis = (torch.arange(resolution, device=device) + 0.5) / resolution
    rows, columns = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([columns.reshape(-1), rows.reshape(-1)], dim=-1)


def slot_centroids(
    allocations: Float[Tensor, "b t n p"], resolution: int
) -> Float[Tensor, "b t n 2"]:
    """Eq. 134: the spatial centroid of each slot's allocation map.

    The allocation is already normalized over positions (Eq. 71), so this is a
    plain weighted mean and needs no further division.
    """
    if allocations.ndim != 4:
        raise ValueError(f"expected (B, T, N, P), got {tuple(allocations.shape)}")
    if allocations.shape[-1] != resolution * resolution:
        raise ValueError(
            f"allocation covers {allocations.shape[-1]} positions, "
            f"expected {resolution}^2 = {resolution * resolution}"
        )
    grid = spatial_grid(resolution, device=allocations.device).to(allocations.dtype)
    return allocations @ grid


def physical_assignment(
    centroids: Float[Tensor, "b t n 2"],
    true_centres: Float[Tensor, "b t n 2"],
) -> Int[Tensor, "b n"]:
    """Eqs. 137-138: one geometric Hungarian assignment per held-out episode.

    Returns ``zeta`` with ``zeta[b, i] = k``, the physical object associated with
    recurrent visual track ``i``. Combine with
    ``scjepa.losses.alignment.align_to_assignment`` to express learned
    quantities in physical-object order (Eq. 139).
    """
    if centroids.shape != true_centres.shape:
        raise ValueError(
            f"centroids {tuple(centroids.shape)} and true centres "
            f"{tuple(true_centres.shape)} must have the same shape"
        )
    with torch.no_grad():
        # (B, T, N_slots, N_objects, 2) -> mean squared distance over the window.
        difference = centroids.unsqueeze(3) - true_centres.unsqueeze(2)
        cost = difference.square().sum(dim=-1).mean(dim=1)
        rows = [linear_sum_assignment(episode.cpu().numpy())[1] for episode in cost]
    return torch.as_tensor(
        [row.tolist() for row in rows], dtype=torch.long, device=centroids.device
    )
