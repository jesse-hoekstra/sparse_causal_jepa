"""Closed-form rank-constrained compression of a frozen predictor's code contribution."""

from typing import cast

import torch
from torch import Tensor


def orient_columns(vectors: Tensor) -> Tensor:
    """Match the native column convention by making its largest absolute entry positive."""
    pivots = torch.argmax(vectors.abs(), dim=0)
    signs = vectors[pivots, torch.arange(vectors.shape[1])].sign()
    return vectors * signs


def reduced_rank_map(
    total: Tensor, shared: Tensor, first_code_weight: Tensor, dimension: int
) -> tuple[Tensor, Tensor]:
    """Return encoder P and OLS decoder R minimizing E||W(c-RPc)||² for centered codes.

    The leading left singular vectors U of W Cov(c)^(1/2) give the optimal hidden
    response subspace. U'W are its linear scores. QR only changes their coordinates.
    Within that subspace use the native shared-variance ordering and sign convention.
    Equal within-subspace shared eigenvalues permit equivalent rotations. A numerical
    tie across the predictive rank cutoff is rejected because the subspace is ambiguous.
    Inputs are fitted to TRAIN groups by the caller. Outputs are CPU float64.
    """
    t, s, w = (x.detach().double().cpu() for x in (total, shared, first_code_weight))
    if (
        t.ndim != 2
        or t.shape[0] != t.shape[1]
        or s.shape != t.shape
        or w.ndim != 2
        or w.shape[1] != t.shape[0]
        or not 0 < dimension <= min(w.shape)
        or any(not torch.isfinite(x).all() for x in (s, t, w))
        or not torch.allclose(s, s.T, rtol=1e-10, atol=1e-12)
        or not torch.allclose(t, t.T, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError("expected finite symmetric covariances and compatible predictor weights")
    eig, vec = cast(tuple[Tensor, Tensor], torch.linalg.eigh((t + t.T) / 2))  # pyright: ignore[reportUnknownMemberType]
    eps = torch.finfo(t.dtype).eps
    if eig[0] < -eps * len(eig) * eig.abs().max():
        raise ValueError("total covariance is indefinite")
    root = (vec * eig.clamp_min(0).sqrt()) @ vec.T
    left, values, _ = cast(
        tuple[Tensor, Tensor, Tensor],
        torch.linalg.svd(w @ root, full_matrices=False),  # pyright: ignore[reportUnknownMemberType]
    )
    tolerance = eps * max(w.shape) * values[0]
    if values[dimension - 1] <= tolerance:
        raise ValueError("predictive rank below requested dimension")
    if dimension < len(values) and values[dimension - 1] - values[dimension] <= tolerance:
        raise ValueError("unresolved singular-value tie at compression cutoff")
    basis, _ = cast(tuple[Tensor, Tensor], torch.linalg.qr(w.T @ left[:, :dimension]))  # pyright: ignore[reportUnknownMemberType]
    _, rotations = cast(tuple[Tensor, Tensor], torch.linalg.eigh(basis.T @ s @ basis))  # pyright: ignore[reportUnknownMemberType]
    p = orient_columns(basis @ rotations).T
    gram = p @ t @ p.T
    cond = float(cast(Tensor, torch.linalg.cond(gram)))  # pyright: ignore[reportUnknownMemberType]
    if not cond < 1 / (eps * dimension):
        raise ValueError("selected TRAIN covariance is numerically singular")
    r = cast(Tensor, torch.linalg.solve(gram, p @ t)).T  # pyright: ignore[reportUnknownMemberType]
    return p, r
