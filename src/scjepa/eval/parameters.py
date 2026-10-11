"""Nonlinear coordinate recovery probes adapted from Baumgartner et al., App. F.1.

Each target coordinate is regressed on each learned coordinate independently.
The reported MCC is the mean of row-wise best held-out R² values, not a
one-to-one assignment. Finite probe fits are diagnostics, not guarantees of
invariance or identifiability. Rows must represent independent episodes; split
selection and scientific interpretation belong to the experiment.
"""

from typing import NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor, nn


class MccReport(NamedTuple):
    """MCC score plus the full R² matrix it summarises.

    ``matrix[i, j]`` is the held-out R² of learned coordinate ``j`` predicting
    true parameter ``i`` (Baumgartner's I x J orientation). ``score`` is the mean
    over ``i`` of the row maxima. The matrix is exposed only for the recovery
    grid and for saved artifacts; it is not a second metric.
    """

    score: Float[Tensor, ""]
    matrix: Float[Tensor, "p q"]
    num_samples: int


def _validate_pairs(learned: Tensor, target: Tensor) -> None:
    """Require two ``(episodes, coordinates)`` tensors with equal episode counts."""
    if learned.ndim != 2 or target.ndim != 2 or learned.shape[0] != target.shape[0]:
        raise ValueError(
            f"expected (S, d) and (S, P) with equal S, got "
            f"{tuple(learned.shape)} vs {tuple(target.shape)}"
        )
    if learned.shape[0] < 4 or min(learned.shape[1], target.shape[1]) < 1:
        raise ValueError("need at least 4 samples and nonempty coordinates for recovery probes")


def _r_squared(prediction: Tensor, target: Tensor) -> float:
    """Held-out coefficient of determination, clamped at 0.

    Negative R² means the fit is worse than predicting the mean — i.e. no
    explanatory power — so it is reported as 0, keeping MCC within [0, 1].
    """
    residual = (target - prediction).square().sum()
    total = (target - target.mean()).square().sum() + 1e-12
    return max(0.0, float(1.0 - residual / total))


def nonlinear_mcc(
    learned: Float[Tensor, "s q"],
    target: Float[Tensor, "s p"],
    hidden: int = 32,
    max_samples: int = 5000,
    epochs: int = 300,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> MccReport:
    """MCC per Baumgartner et al. App. F.1 (nonlinear MLP-R², mean over row maxima).

    Fits θ_i ≈ MLP(θ̂_j) for every (true, learned) pair with a one-hidden-layer
    MLP (width ``hidden``, Adam, full batch), scores R² on the held-out
    ``val_fraction`` split, and returns ``mean_i max_j R²_ij``. Deterministic
    given ``seed``. Intended to assess recovery under coordinate transformations,
    scale and shift only insofar as the finite probes can fit those maps.

    The probe MLPs are CPU modules whose initialization is isolated from the
    caller's global torch RNG, so periodic evaluation cannot alter the
    subsequent training trajectory.
    """
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        return _nonlinear_mcc_impl(
            learned,
            target,
            hidden=hidden,
            max_samples=max_samples,
            epochs=epochs,
            val_fraction=val_fraction,
            seed=seed,
        )


def _nonlinear_mcc_impl(
    learned: Float[Tensor, "s q"],
    target: Float[Tensor, "s p"],
    hidden: int,
    max_samples: int,
    epochs: int,
    val_fraction: float,
    seed: int,
) -> MccReport:
    """Fit the pairwise regressors inside an RNG-isolated caller context."""
    _validate_pairs(learned, target)
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    if hidden < 1 or max_samples < 4 or epochs < 1:
        raise ValueError("hidden/epochs must be positive and max_samples >= 4")
    generator = torch.Generator().manual_seed(seed)
    num_samples = min(learned.shape[0], max_samples)
    order = torch.randperm(learned.shape[0], generator=generator)[:num_samples]
    learned = learned.detach().cpu()[order].float()
    target = target.detach().cpu()[order].float()
    split = min(num_samples - 2, max(2, int(num_samples * val_fraction)))
    r2 = torch.zeros(target.shape[1], learned.shape[1])
    for i in range(target.shape[1]):
        for j in range(learned.shape[1]):
            torch.default_generator.manual_seed(seed * 7919 + i * 131 + j)
            x, y = learned[:, j : j + 1], target[:, i : i + 1]
            # Standardize with FIT-FOLD statistics only: computing them over the
            # full sample would leak the scored fold's first two moments into
            # the probe's input scaling.
            x_fit, y_fit = x[split:], y[split:]
            x = (x - x_fit.mean()) / (x_fit.std() + 1e-6)
            y_mean, y_std = y_fit.mean(), y_fit.std() + 1e-6
            y_norm = (y - y_mean) / y_std
            x_train, y_train = x[split:], y_norm[split:]
            x_val, y_val = x[:split], y_norm[:split]
            mlp = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-2)
            # enable_grad: this metric fits MLPs, so it must work even when the
            # caller (e.g. the eval harness) runs under torch.no_grad().
            with torch.enable_grad():
                for _ in range(epochs):
                    optimizer.zero_grad(set_to_none=True)
                    loss = (mlp(x_train) - y_train).square().mean()
                    loss.backward()  # pyright: ignore[reportUnknownMemberType]
                    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            with torch.no_grad():
                r2[i, j] = _r_squared(mlp(x_val), y_val)
    return MccReport(score=r2.max(dim=1).values.mean(), matrix=r2, num_samples=num_samples)


__all__ = ["MccReport", "nonlinear_mcc"]
