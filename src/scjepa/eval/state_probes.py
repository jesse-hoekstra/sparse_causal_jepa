"""Frozen linear state decoding, fitted on training episodes for evaluation only.

A single shared linear map decodes every anonymous slot into (x, y, vx, vy).
Geometric assignments provide labels only after encoder training; they never
enter the representation objective. Episode-level separation prevents adjacent
frames of one trajectory from appearing in both probe fitting and scoring.
"""

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor


@dataclass(frozen=True)
class StateProbe:
    """A frozen map with input statistics estimated exclusively on fit episodes."""

    feature_mean: Tensor
    feature_std: Tensor
    target_mean: Tensor
    weight: Tensor

    def predict(self, features: Tensor) -> Tensor:
        """Decode detached slots without modifying the encoder or probe."""
        values = features.detach().cpu().double()
        return ((values - self.feature_mean) / self.feature_std) @ self.weight + self.target_mean


@torch.no_grad()
def fit_state_probe(features: Tensor, states: Tensor, ridge: float = 1e-3) -> StateProbe:
    """Fit one shared ridge regressor on flattened training-episode slot rows."""
    if features.ndim != 2 or states.ndim != 2 or features.shape[0] != states.shape[0]:
        raise ValueError("probe expects feature/state matrices with the same number of rows")
    if features.shape[0] < 2 or states.shape[1] != 4 or ridge <= 0:
        raise ValueError("probe requires at least two rows, four state coordinates, positive ridge")
    features, states = features.detach().cpu().double(), states.detach().cpu().double()
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp(min=1e-6)
    target_mean = states.mean(dim=0)
    design = (features - mean) / std
    count = features.shape[0]
    gram = design.T @ design / count + ridge * torch.eye(design.shape[1], dtype=design.dtype)
    weight = cast(Tensor, torch.linalg.solve(gram, design.T @ (states - target_mean) / count))  # pyright: ignore[reportUnknownMemberType]
    return StateProbe(mean, std, target_mean, weight)


def state_probe_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    """Report mean coordinate R² for position and velocity on held-out episodes.

    Negative values are retained: they mean the decoder is worse than the
    held-out mean baseline. High scores show linear decodability, not equality
    between individual learned coordinates and physical state coordinates.
    """
    prediction, target = prediction.detach().cpu().double(), target.detach().cpu().double()
    if prediction.shape != target.shape or prediction.ndim != 2 or target.shape[1] != 4:
        raise ValueError("state probe predictions and targets must both have shape (rows, 4)")
    residual = (prediction - target).square().sum(dim=0)
    total = (target - target.mean(dim=0)).square().sum(dim=0).clamp(min=1e-12)
    score = 1 - residual / total
    return {
        "position_probe_r2": float(score[:2].mean()),
        "velocity_probe_r2": float(score[2:].mean()),
    }
