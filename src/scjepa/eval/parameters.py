"""Parameter-identification diagnostics: correlations, MCC/MCC, marginal recovery.

The identifiability claim (my_paper.pdf Thm; after Baumgartner et al.) is that
learned parameters Ŝ^ph match the ground truth up to permutation and
element-wise diffeomorphism. Diffeomorphisms are monotone, so |Pearson
correlation| between the right learned dimension and the true parameter is the
standard proxy (perfect only for affine maps — the marginal-recovery scatter is
the honest picture for nonlinear ones).

Sample convention: parameters are per-object with shared weights, so every
(episode, object) pair is one sample. Callers flatten to ``learned (S, d)`` and
``target (S, P)``. Slot↔object alignment is the caller's job (see graph.py).

Metrics:
    ``nonlinear_mcc``         — THE MCC (mean-max correlation coefficient) as
        defined by Baumgartner et al. App. F.1, used for exact comparison to
        their results: for every (true param i, learned dim j) pair, fit a
        one-hidden-layer MLP (width 32) predicting θ_i from θ̂_j on a 90%
        split, score nonlinear R² on the held-out 10%; MCC = mean_i max_j R².
        Diffeomorphism-invariant (unlike Pearson).
    ``mean_max_correlation``  — fast Pearson proxy of the same mean-max shape;
        under-reports strongly nonlinear diffeomorphisms (tanh caveat).
    ``marginal_recovery``     — the (true param, best learned dim) value pairs
        for the Fig.-5-style scatter plots.
"""

import torch
from jaxtyping import Float, Int64
from torch import Tensor, nn


def correlation_matrix(
    learned: Float[Tensor, "s d"], target: Float[Tensor, "s p"]
) -> Float[Tensor, "d p"]:
    """Pearson correlations between every learned dim and every true parameter."""
    if learned.ndim != 2 or target.ndim != 2 or learned.shape[0] != target.shape[0]:
        raise ValueError(
            f"expected (S, d) and (S, P) with equal S, got "
            f"{tuple(learned.shape)} vs {tuple(target.shape)}"
        )
    if learned.shape[0] < 2:
        raise ValueError("need at least 2 samples for correlations")
    x = learned - learned.mean(dim=0)
    y = target - target.mean(dim=0)
    x = x / (x.square().sum(dim=0).sqrt() + 1e-12)
    y = y / (y.square().sum(dim=0).sqrt() + 1e-12)
    return x.T @ y


def mean_max_correlation(
    learned: Float[Tensor, "s d"], target: Float[Tensor, "s p"]
) -> Float[Tensor, ""]:
    """MCC (mean-max): mean over true params of the best |corr| across dims."""
    return correlation_matrix(learned, target).abs().max(dim=0).values.mean()


def _r_squared(prediction: Tensor, target: Tensor) -> float:
    """Held-out coefficient of determination, clamped at 0.

    Negative R² means the fit is worse than predicting the mean — i.e. no
    explanatory power — so it is reported as 0, keeping MCC within [0, 1].
    """
    residual = (target - prediction).square().sum()
    total = (target - target.mean()).square().sum() + 1e-12
    return max(0.0, float(1.0 - residual / total))


def nonlinear_mcc(
    learned: Float[Tensor, "s d"],
    target: Float[Tensor, "s p"],
    hidden: int = 32,
    max_samples: int = 5000,
    epochs: int = 300,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Float[Tensor, ""]:
    """MCC per Baumgartner et al. App. F.1 (MLP-based nonlinear R², mean-max).

    Fits θ_i ≈ MLP(θ̂_j) for every pair with a one-hidden-layer MLP (width 32,
    Adam, full batch), R² scored on a held-out split; returns
    ``mean_i max_j R²_ij``. Deterministic given ``seed``. Slower than the
    Pearson proxy but invariant to element-wise diffeomorphisms — use this for
    reported numbers.
    """
    corr_input = correlation_matrix(learned, target)  # validates shapes cheaply
    del corr_input
    generator = torch.Generator().manual_seed(seed)
    num_samples = min(learned.shape[0], max_samples)
    order = torch.randperm(learned.shape[0], generator=generator)[:num_samples]
    learned = learned[order].float()
    target = target[order].float()
    split = max(1, int(num_samples * val_fraction))
    r2 = torch.zeros(target.shape[1], learned.shape[1])
    for i in range(target.shape[1]):
        for j in range(learned.shape[1]):
            torch.manual_seed(seed * 7919 + i * 131 + j)  # pyright: ignore[reportUnknownMemberType]
            x, y = learned[:, j : j + 1], target[:, i : i + 1]
            x = (x - x.mean()) / (x.std() + 1e-6)
            y_mean, y_std = y.mean(), y.std() + 1e-6
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
    return r2.max(dim=1).values.mean()


def marginal_recovery(
    learned: Float[Tensor, "s d"], target: Float[Tensor, "s p"], param_index: int = 0
) -> tuple[Int64[Tensor, ""], Float[Tensor, " s"], Float[Tensor, " s"]]:
    """Data for the recovery scatter: (best learned dim, its values, true values).

    A successful identification (up to element-wise diffeomorphism) shows up as
    a clean monotone curve when plotting the returned pairs.
    """
    corr = correlation_matrix(learned, target)
    best_dim = corr[:, param_index].abs().argmax()
    return best_dim, learned[:, best_dim], target[:, param_index]


__all__ = [
    "correlation_matrix",
    "marginal_recovery",
    "mean_max_correlation",
    "nonlinear_mcc",
]
