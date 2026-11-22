"""Select the most episode-stable affine readout of fixed, trained endpoint features.

Let X be the centered TRAIN episode-by-feature matrix with G equal-sized systems and R
episodes per system. Let H subtract each system's episode mean. Define total covariance
T = X'X/(GR) and within covariance W = X'HX/(GR). Among p-by-d matrices A in the
resolved support of T satisfying A'TA = I, minimize tr(A'WA). If U is an orthonormal
basis of the resolved column space of X, every feasible centered readout is sqrt(GR) UQ
with Q'Q = I. The bottom d eigenvectors of U'HU attain the minimum sum of eigenvalues
(Ky Fan) and the minimum possible largest within/total ratio (Courant-Fischer).

A gap after the dth eigenvalue identifies a subspace, even if eigenvalues inside it tie.
Replacing U by UO conjugates the projected matrix by O. Any selected basis Z may
therefore rotate to ZJ. Least squares on system means gives (mean(Z)J)^+Y = J' mean(Z)^+Y
for orthogonal J. Thus the teacher-aligned episode predictions are basis-independent.
A full-rank alignment preserves every generalized W/T eigenvalue of the selected space.
The identity-total-variance normalization applies to Z before alignment.

These statements are empirical and concern the resolved TRAIN support. A nonsingular
feature reparameterization that preserves the resolved observation subspace gives the
same aligned codes there. Off-support values use the minimum-norm extension in the native
feature coordinates. Forecast error, physical information and validation stability are
measured by the caller. The fit consumes frozen TRAIN features and the original last layer.
"""

from typing import Any, cast

import torch
from torch import Tensor, nn


class StableReadoutError(ValueError):
    """A registered scientific prerequisite failed; diagnostics can enter a terminal receipt."""

    def __init__(self, reason: str, message: str, diagnostics: dict[str, Any]) -> None:
        """Store a stable reason and the measurements available at failure."""
        super().__init__(message)
        self.reason = reason
        self.diagnostics = dict(diagnostics)


class SingularMeanMap(StableReadoutError):
    """The selected system means cannot yield a resolved full-rank map into the teacher chart."""


def _svd(matrix: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return cast(tuple[Tensor, Tensor, Tensor], torch.linalg.svd(matrix, full_matrices=False))  # pyright: ignore[reportUnknownMemberType]


def _list(values: Tensor) -> list[float]:
    return cast(list[float], values.tolist())  # pyright: ignore[reportUnknownMemberType]


@torch.no_grad()
def fit_readout(
    features: Tensor, layer: nn.Linear
) -> tuple[dict[str, tuple[Tensor, Tensor]], dict[str, Any]]:
    """Return unchanged own and stable teacher-aligned treatment weights in CPU float64."""
    x = features.detach().cpu().double()
    if x.ndim != 3 or min(x.shape[:2]) < 2 or x.shape[-1] != layer.in_features:
        raise ValueError("expected grouped TRAIN features with at least two systems and episodes")
    if cast(Tensor | None, layer.bias) is None:
        raise ValueError("the forecast interface requires an affine endpoint layer")
    w, b = layer.weight.detach().cpu().double(), layer.bias.detach().cpu().double()
    if not all(bool(t.isfinite().all()) for t in (x, w, b)):
        raise ValueError("nonfinite endpoint features or readout")
    groups, episodes, width = x.shape
    n, d = groups * episodes, layer.out_features
    eps = torch.finfo(torch.float64).eps
    mean = x.mean((0, 1))
    centered = (x - mean).flatten(0, 1)
    u, singular, vh = _svd(centered)
    total_eigen = singular.square() / n
    covariance_cutoff = max(n, width) * eps * float(total_eigen[0])
    rank = int((total_eigen > covariance_cutoff).sum())
    stats: dict[str, Any] = {
        "groups": groups,
        "episodes": episodes,
        "features": width,
        "dimension": d,
        "design_rank": rank,
        "eigenvalue_cutoff": covariance_cutoff,
        "feature_spectrum": _list(total_eigen.flip(0)),
        "feature_spectrum_order": "ascending",
        "arm_definition": "own is the original readout. sibling is the aligned stable subspace",
    }
    if rank < d:
        raise StableReadoutError(
            "insufficient_feature_rank", "TRAIN feature rank below output dimension", stats
        )
    basis = u[:, :rank].reshape(groups, episodes, rank)
    group_basis = basis.mean(1)
    residual = (basis - group_basis[:, None, :]).flatten(0, 1)
    within = residual.T @ residual
    eigen, directions = cast(tuple[Tensor, Tensor], torch.linalg.eigh((within + within.T) * 0.5))  # pyright: ignore[reportUnknownMemberType]
    # The projector H has spectral norm one, independent of feature units.
    gap_cutoff = max(n, width) * eps
    gap = float(eigen[d] - eigen[d - 1]) if rank > d else None
    stats.update(
        within_over_total_spectrum=_list(eigen),
        boundary_gap=gap,
        gap_cutoff=gap_cutoff,
        entire_resolved_support=rank == d,
    )
    if gap is not None and gap <= gap_cutoff:
        raise StableReadoutError(
            "unresolved_boundary", "No resolved gap beyond the selected axes", stats
        )

    chosen = directions[:, :d]
    # mean(Z) and the original teacher's group means, both with the TRAIN intercept removed.
    selected_means = (group_basis @ chosen) * n**0.5
    selected_means = selected_means - selected_means.mean(0)
    teacher_mean = mean @ w.T + b
    teacher_means = centered.reshape(groups, episodes, width).mean(1) @ w.T
    teacher_means = teacher_means - teacher_means.mean(0)
    left_x, sx, right_x = _svd(selected_means)
    left_y, sy, _ = _svd(teacher_means)
    roundoff = max(groups, d) * eps
    # Selected codes have unit total variance. Teacher means are judged in teacher units.
    x_cut = groups**0.5 * roundoff
    teacher_total = _svd(singular[:, None] * (vh @ w.T))[1]
    y_cut = max(float(sy[0]), float(teacher_total[0]) / episodes**0.5) * roundoff
    rx, ry = int((sx > x_cut).sum()), int((sy > y_cut).sum())
    stats.update(
        selected_mean_rank=rx,
        teacher_mean_rank=ry,
        selected_mean_singular_values=_list(sx),
        teacher_mean_singular_values=_list(sy),
        teacher_total_singular_values=_list(teacher_total),
        selected_mean_cutoff=x_cut,
        teacher_mean_cutoff=y_cut,
    )
    if rx < d:
        raise SingularMeanMap(
            "selected_mean_rank", "Selected axes do not span d system means", stats
        )
    if ry < d:
        raise SingularMeanMap(
            "teacher_mean_rank", "Original teacher means do not span d outputs", stats
        )
    # The OLS chart is invertible iff the two full-rank column spaces have no right angle.
    cosines = _svd(left_x.T @ left_y)[1]
    stats.update(mean_canonical_correlations=_list(cosines), canonical_cutoff=roundoff)
    if float(cosines[-1]) <= roundoff:
        raise SingularMeanMap(
            "singular_alignment", "Selected and teacher means have a singular cross-map", stats
        )
    beta = right_x.T @ ((left_x.T @ teacher_means) / sx[:, None])
    beta_spectrum = _svd(beta)[1]
    chart_cut = max(d, groups) * eps * float(beta_spectrum[0])
    stats.update(chart_singular_values=_list(beta_spectrum), chart_cutoff=chart_cut)
    if float(beta_spectrum[-1]) <= chart_cut:
        raise SingularMeanMap(
            "singular_alignment", "Mean map unresolved in the forecast chart's native units", stats
        )
    into_selected = (vh[:rank].T / singular[:rank]) * n**0.5 @ chosen
    weight = (into_selected @ beta).T.contiguous()
    bias = teacher_mean - mean @ weight.T
    # Use the coefficients as stored in the native layer, with double accumulation for the assay.
    native_weight = weight.to(dtype=layer.weight.dtype).double()
    native_bias = bias.to(dtype=layer.bias.dtype).double()
    if not all(bool(t.isfinite().all()) for t in (native_weight, native_bias)):
        raise StableReadoutError(
            "native_precision", "Chart coefficients exceed native precision", stats
        )
    actual = centered @ native_weight.T
    ca, sa, _ = _svd(actual)
    actual_cut = float(sa[0]) * max(n, d) * eps
    if float(sa[-1]) <= actual_cut:
        stats.update(final_singular_values=_list(sa), final_singular_cutoff=actual_cut)
        raise SingularMeanMap(
            "singular_alignment", "Learned chart loses a numerical output after transport", stats
        )
    normalized = ca.reshape(groups, episodes, d)
    h = (normalized - normalized.mean(1, keepdim=True)).flatten(0, 1)
    actual_within = h.T @ h
    recovered = cast(Tensor, torch.linalg.eigvalsh((actual_within + actual_within.T) * 0.5))  # pyright: ignore[reportUnknownMemberType]
    aligned_mse = (selected_means @ beta - teacher_means).square().mean()
    stats.update(
        selected_spectrum=_list(eigen[:d]),
        aligned_train_spectrum=_list(recovered),
        spectrum_coefficient_dtype=str(layer.weight.dtype),
        final_singular_values=_list(sa),
        final_singular_cutoff=actual_cut,
        optimum_trace=float(eigen[:d].sum()),
        optimum_worst=float(eigen[d - 1]),
        spectrum_discrepancy=float((recovered - eigen[:d]).abs().max()),
        target_mse={"own": 0.0, "sibling": float(aligned_mse)},
        target_mse_definition="MSE to original TRAIN system means after centering",
    )
    return {"own": (w.clone(), b.clone()), "sibling": (weight, bias)}, stats
