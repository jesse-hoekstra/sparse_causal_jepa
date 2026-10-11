"""Frozen ridge probes for arbitrary feature and reference-state dimensions.

Fit and score on separate episodes supplied by the experiment. Probes never
modify the encoder and do not assign physical meanings to latent coordinates.
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
        """Decode detached features without modifying the encoder or probe."""
        values = features.detach().cpu().double()
        return ((values - self.feature_mean) / self.feature_std) @ self.weight + self.target_mean


@torch.no_grad()
def fit_state_probe(features: Tensor, states: Tensor, ridge: float = 1e-3) -> StateProbe:
    """Fit a ridge regressor on rows from the caller-selected training episodes."""
    if features.ndim != 2 or states.ndim != 2 or features.shape[0] != states.shape[0]:
        raise ValueError("probe expects feature/state matrices with the same number of rows")
    if features.shape[0] < 2 or min(features.shape[1], states.shape[1]) < 1 or ridge <= 0:
        raise ValueError("probe requires at least two rows, nonempty coordinates, positive ridge")
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
    """Report overall R² and MSE for arbitrary held-out reference coordinates."""
    prediction, target = prediction.detach().cpu().double(), target.detach().cpu().double()
    if prediction.shape != target.shape or prediction.ndim != 2 or target.shape[1] < 1:
        raise ValueError(
            "state probe predictions and targets must both have shape (rows, coordinates)"
        )
    residual = (prediction - target).square().sum(dim=0)
    total = (target - target.mean(dim=0)).square().sum(dim=0).clamp(min=1e-12)
    score = 1 - residual / total
    return {
        "r2": float(score.mean()),
        "mse": float((prediction - target).square().mean()),
    }
