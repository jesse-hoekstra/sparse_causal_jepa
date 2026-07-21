"""Parameter-identification diagnostics for anonymous global coordinates.

The identifiability claim (my_paper.pdf Thm; after Baumgartner et al.) is that
learned parameters Ŝ^ph match the ground truth up to permutation and
element-wise diffeomorphism. Diffeomorphisms are monotone, so |Pearson
correlation| between the right learned dimension and the true parameter is the
standard proxy (perfect only for affine maps — the marginal-recovery scatter is
the honest picture for nonlinear ones).

The primary evaluation uses one sample per episode: learned ``(E, q)`` and
true ``(E, p)`` (``E x 5`` versus ``E x 5`` in Bounce).  A *single global*
Hungarian assignment is selected on a held-out alignment split and frozen
before scoring a disjoint test split.  This accepts a fixed permutation of the
learned coordinates, but does not accept episode-wise slot switching or let
one learned coordinate explain several true parameters.

Metrics:
    ``one_to_one_recovery``   — strict, global one-to-one nonlinear recovery.
    ``nonlinear_mcc``         — legacy mean-max correlation coefficient as
        defined by Baumgartner et al. App. F.1, used for metric comparison to
        their results: for every (true param i, learned dim j) pair, fit a
        one-hidden-layer MLP (width 32) predicting θ_i from θ̂_j on a 90%
        split, score nonlinear R² on the held-out 10%; MCC = mean_i max_j R².
        Diffeomorphism-invariant (unlike Pearson).
    ``mean_max_correlation``  — fast Pearson proxy of the same mean-max shape;
        under-reports strongly nonlinear diffeomorphisms (tanh caveat).
    ``marginal_recovery``     — the (true param, best learned dim) value pairs
        for the Fig.-5-style scatter plots.
"""

from typing import NamedTuple

import torch
from jaxtyping import Float, Int64
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn


class OneToOneRecovery(NamedTuple):
    """Held-out recovery scores and one frozen global coordinate assignment.

    Matrices use ``[learned_coordinate, true_parameter]`` order.
    ``target_to_learned[i]`` is the learned coordinate assigned to true
    parameter ``i``.  The assignment is selected from an alignment split;
    ``nonlinear_score``, ``linear_score`` and the exposed matrices are all
    measured on a separate test split.
    """

    nonlinear_score: Float[Tensor, ""]
    linear_score: Float[Tensor, ""]
    nonlinear_matrix: Float[Tensor, "q p"]
    linear_matrix: Float[Tensor, "q p"]
    target_to_learned: Int64[Tensor, " p"]
    num_samples: int


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


def optimal_one_to_one_assignment(
    scores: Float[Tensor, "q p"],
) -> Int64[Tensor, " p"]:
    """Maximise a square score matrix under a single bijective assignment.

    Returns a target-to-learned mapping: ``result[i] == j`` means learned
    coordinate ``j`` represents true parameter ``i``.  Requiring a square
    matrix makes the identifiability claim explicit: exactly one learned
    coordinate must account for each true factor.
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(
            "strict one-to-one recovery requires equally many learned and true "
            f"coordinates, got {tuple(scores.shape)}"
        )
    row_ind, col_ind = linear_sum_assignment(-scores.detach().cpu().numpy())
    target_to_learned = torch.empty(scores.shape[1], dtype=torch.long)
    target_to_learned[torch.as_tensor(col_ind)] = torch.as_tensor(row_ind)
    return target_to_learned


def _fold_correlation_matrix(learned: Tensor, target: Tensor) -> Tensor:
    """Absolute Pearson matrix, with a defined zero result for one-row folds."""
    if learned.shape[0] < 2:
        return torch.zeros(learned.shape[1], target.shape[1])
    return correlation_matrix(learned, target).abs()


def _split_for_recovery(
    learned: Tensor,
    target: Tensor,
    max_samples: int,
    holdout_fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Make deterministic probe-train, assignment and final-test folds."""
    correlation_matrix(learned, target)  # common shape/sample validation
    if learned.shape[1] != target.shape[1]:
        raise ValueError(
            "strict one-to-one recovery requires equally many learned and true "
            f"coordinates, got {learned.shape[1]} and {target.shape[1]}"
        )
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be in (0, 0.5)")
    num_samples = min(learned.shape[0], max_samples)
    if num_samples < 3:
        raise ValueError("strict recovery needs at least 3 episodes")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(learned.shape[0], generator=generator)[:num_samples]
    learned = learned.detach().cpu()[order].float()
    target = target.detach().cpu()[order].float()
    holdout = max(1, int(num_samples * holdout_fraction))
    # Preserve at least one probe-training sample for tiny wiring tests. Real
    # evaluations use hundreds or thousands of episodes, so both held-out
    # folds contain many samples.
    holdout = min(holdout, (num_samples - 1) // 2)
    align_end = holdout
    test_end = 2 * holdout
    return (
        learned[test_end:],
        target[test_end:],
        learned[:align_end],
        target[:align_end],
        learned[align_end:test_end],
        target[align_end:test_end],
    )


def one_to_one_recovery(
    learned: Float[Tensor, "s q"],
    target: Float[Tensor, "s p"],
    hidden: int = 32,
    max_samples: int = 5000,
    epochs: int = 300,
    holdout_fraction: float = 0.1,
    seed: int = 0,
) -> OneToOneRecovery:
    """Evaluate strict mass recovery up to one global coordinate permutation.

    A scalar MLP is fit for every learned/true pair on the probe-training
    split.  Nonlinear R² on a disjoint alignment split selects one Hungarian
    bijection.  That mapping is frozen before both nonlinear R² and absolute
    Pearson correlation are scored on the final test split.  The same mapping
    can therefore be reused to align causal-graph columns.

    Unlike Baumgartner's mean-max summary, this metric cannot reuse a learned
    coordinate for several masses.  It also cannot repair a different
    permutation in every episode.
    """
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        (
            learned_train,
            target_train,
            learned_align,
            target_align,
            learned_test,
            target_test,
        ) = _split_for_recovery(learned, target, max_samples, holdout_fraction, seed)
        num_learned = learned.shape[1]
        num_target = target.shape[1]
        align_r2 = torch.zeros(num_learned, num_target)
        test_r2 = torch.zeros_like(align_r2)
        for j in range(num_learned):
            for i in range(num_target):
                torch.default_generator.manual_seed(seed * 7919 + i * 131 + j)
                x_train = learned_train[:, j : j + 1]
                y_train = target_train[:, i : i + 1]
                x_mean = x_train.mean()
                y_mean = y_train.mean()
                x_std = x_train.std(unbiased=False) + 1e-6
                y_std = y_train.std(unbiased=False) + 1e-6
                x_train = (x_train - x_mean) / x_std
                y_train = (y_train - y_mean) / y_std
                mlp = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1))
                optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-2)
                with torch.enable_grad():
                    for _ in range(epochs):
                        optimizer.zero_grad(set_to_none=True)
                        loss = (mlp(x_train) - y_train).square().mean()
                        loss.backward()  # pyright: ignore[reportUnknownMemberType]
                        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
                with torch.no_grad():
                    align_prediction = (
                        mlp((learned_align[:, j : j + 1] - x_mean) / x_std) * y_std + y_mean
                    )
                    test_prediction = (
                        mlp((learned_test[:, j : j + 1] - x_mean) / x_std) * y_std + y_mean
                    )
                    align_r2[j, i] = _r_squared(align_prediction, target_align[:, i : i + 1])
                    test_r2[j, i] = _r_squared(test_prediction, target_test[:, i : i + 1])

        target_to_learned = optimal_one_to_one_assignment(align_r2)
        target_indices = torch.arange(num_target)
        test_linear = _fold_correlation_matrix(learned_test, target_test)
        return OneToOneRecovery(
            nonlinear_score=test_r2[target_to_learned, target_indices].mean(),
            linear_score=test_linear[target_to_learned, target_indices].mean(),
            nonlinear_matrix=test_r2,
            linear_matrix=test_linear,
            target_to_learned=target_to_learned,
            num_samples=(learned_train.shape[0] + learned_align.shape[0] + learned_test.shape[0]),
        )


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
    reported numbers. The metric's local model initialization is isolated from
    the caller's global torch RNG, so periodic evaluation cannot alter the
    subsequent training trajectory.
    """
    # The MLPs below are CPU modules. Fork only the CPU generator, and seed that
    # generator directly rather than calling torch.manual_seed (which also
    # mutates accelerator generators). The caller's state is restored on exit.
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
    learned: Float[Tensor, "s d"],
    target: Float[Tensor, "s p"],
    hidden: int,
    max_samples: int,
    epochs: int,
    val_fraction: float,
    seed: int,
) -> Float[Tensor, ""]:
    """Fit the pairwise regressors inside an RNG-isolated caller context."""
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
            torch.default_generator.manual_seed(seed * 7919 + i * 131 + j)
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
    "OneToOneRecovery",
    "correlation_matrix",
    "marginal_recovery",
    "mean_max_correlation",
    "nonlinear_mcc",
    "one_to_one_recovery",
    "optimal_one_to_one_assignment",
]
