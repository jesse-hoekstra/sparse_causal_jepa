"""Step-one sibling guidance from the detached codes of the current training minibatch."""

from typing import cast

import torch
from torch import Tensor, nn

from .consensus_gradient import add_capped_gradient


def sibling_loss(codes: Tensor) -> Tensor:
    """Whiten each residual to its other-episode mean using the detached batch covariance.

    The metric is the inverse of T + ridge * I, with T the total population covariance over
    the current TRAIN groups and episodes. The ridge is one millionth of the mean coordinate
    variance, floored at the smallest normal float64. This also defines the loss in an
    initially rank-deficient chart. Centering the targets around a detached per-group origin
    keeps exactly identical sibling codes at zero residual.
    """
    if codes.ndim != 3 or codes.shape[0] < 1 or codes.shape[1] < 2 or codes.shape[2] < 1:
        raise ValueError("expected one or more groups of multiple episodes with nonempty codes")
    student = codes.double()
    with torch.no_grad():
        frozen = student.detach()
        origin = frozen[:, :1]
        episodes = frozen - origin
        other_mean = (episodes.sum(dim=1, keepdim=True) - episodes) / (episodes.shape[1] - 1)

        rows = frozen.flatten(0, 1)
        centered = rows - rows[:1]
        centered = centered - centered.mean(0)
        covariance = centered.T @ centered / len(rows)
        diagonal = covariance.diagonal()
        ridge = (diagonal.mean() * 1e-6).clamp_min(torch.finfo(torch.float64).tiny)
        diagonal.add_(ridge)
        factor, info = cast(tuple[Tensor, Tensor], torch.linalg.cholesky_ex(covariance))  # pyright: ignore[reportUnknownMemberType]
        torch._assert_async(info == 0, "nonfinite or indefinite TRAIN code covariance")  # pyright: ignore[reportPrivateImportUsage, reportPrivateUsage]
    residual = (student - origin - other_mean).flatten(0, 1)
    whitened = cast(Tensor, torch.linalg.solve_triangular(factor, residual.T, upper=False))  # pyright: ignore[reportUnknownMemberType]
    return whitened.square().mean()


def add_sibling_gradient(
    forecast: Tensor, codes: Tensor, params: tuple[nn.Parameter, ...]
) -> dict[str, Tensor]:
    """Add only the capped code-encoder gradient before the native objective backward."""
    return add_capped_gradient(forecast, sibling_loss(codes), params)
