"""Sampled observational-equivalence diagnostics for state trajectories.

These helpers are evaluation-only. They measure approximate agreement on a
fixed held-out sample; they do not establish the paper's population
observational-equivalence assumption.
"""

from typing import NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class OeSummary(NamedTuple):
    """Tolerance satisfaction and quantiles of per-episode worst-step NRMSE."""

    satisfaction: float
    p50: float
    p95: float


@torch.no_grad()
def training_coordinate_std(
    dataset: Dataset[dict[str, Tensor]],
    *,
    batch_size: int = 256,
    eps: float = 1e-12,
) -> Float[Tensor, " d"]:
    """Compute fixed population standard deviations over all training states.

    Every episode, time step, and object in the finite training set contributes
    once. Float64 streaming moments avoid materializing another copy of the
    potentially 100k-episode preload. The finite training split is treated as
    the reference population (``correction=0``); only genuinely constant
    coordinates are floored.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(0),
    )
    total: Tensor | None = None
    total_square: Tensor | None = None
    count = 0
    for batch in loader:
        states = batch["states"]
        if states.ndim != 4:
            raise ValueError(f"dataset states must batch to (B,T,N,d), got {states.shape}")
        flat = states.reshape(-1, states.shape[-1]).to(dtype=torch.float64)
        batch_total = flat.sum(dim=0)
        batch_square = flat.square().sum(dim=0)
        total = batch_total if total is None else total + batch_total
        total_square = batch_square if total_square is None else total_square + batch_square
        count += flat.shape[0]
    if total is None or total_square is None or count < 1:
        raise ValueError("training dataset yielded no states")
    mean = total / count
    variance = (total_square / count - mean.square()).clamp_min(0.0)
    return variance.sqrt().clamp_min(eps).to(dtype=torch.float32)


def oe_worst_step_nrmse(
    prediction: Float[Tensor, "b k n d"],
    target: Float[Tensor, "b k n d"],
    coordinate_std: Float[Tensor, " d"],
) -> Float[Tensor, " b"]:
    """Return ``E_i = max_k sqrt(mean_(n,d) standardized_error²)``.

    The maximum is taken after the object/coordinate reduction, so one bad
    rollout step cannot be hidden by averaging over time.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            f"expected matching (B,K,N,d), got {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if coordinate_std.shape != prediction.shape[-1:]:
        raise ValueError(
            f"coordinate_std must have shape {tuple(prediction.shape[-1:])}, "
            f"got {tuple(coordinate_std.shape)}"
        )
    scales = coordinate_std.to(device=prediction.device, dtype=prediction.dtype)
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError("coordinate_std must contain finite positive values")
    per_step = ((prediction - target) / scales).square().mean(dim=(2, 3)).sqrt()
    return per_step.amax(dim=1)


def summarize_oe(errors: Float[Tensor, " n"], tolerance: float) -> OeSummary:
    """Summarize held-out worst-step errors at a fixed inclusive tolerance."""
    if errors.ndim != 1 or errors.numel() < 1:
        raise ValueError("errors must be a non-empty vector")
    if not bool(torch.isfinite(errors).all()):
        raise ValueError("errors must be finite")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    values = errors.float()
    return OeSummary(
        satisfaction=float((values <= tolerance).float().mean()),
        p50=float(torch.quantile(values, 0.50)),
        p95=float(torch.quantile(values, 0.95)),
    )


__all__ = [
    "OeSummary",
    "oe_worst_step_nrmse",
    "summarize_oe",
    "training_coordinate_std",
]
