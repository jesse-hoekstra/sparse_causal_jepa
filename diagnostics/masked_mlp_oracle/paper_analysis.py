"""Paper figures and tables for a multi-seed masked-MLP oracle batch.

Read-only: it authenticates completed run directories and reads their saved
validation endpoints, learning curves, matchings and final mask logits. It loads
no model for inference and never touches test data. Excluded from numerical
identity. Layout expected under ``--batch-dir``: ``{dense,baseline,full}-s{seed}/results``,
plus ``projected-s{seed}/results`` when initialization includes a learned-code projection.
An adapted dense endpoint additionally reports its authenticated projection-to-adaptation
comparison. When adaptation is present, that six-code endpoint initializes both sparse arms.

Outputs: box plots of MCC and block purity per arm with chance references, MCC
learning curves, edge-recovery frequencies of the learned graphs, matched
correlation matrices, forecast-quality panels, the GECO multiplier, LaTeX tables,
a per-run CSV, a JSON summary and a README with draft captions.
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment  # pyright: ignore[reportUnknownVariableType]

from diagnostics.isotropic_oracle.dgp import graphs as true_graphs
from diagnostics.masked_mlp_learned import run as learned
from diagnostics.masked_mlp_oracle import from_dense, run
from diagnostics.oracle_state_learning.data import read_json, sha
from scjepa.data.datasets import tensor_fingerprint
from scjepa.eval.recovery import pearson_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

ARMS = ("dense", "baseline", "full")
CODE_KEYS = (
    "code_r2_min",
    "code_r2_mean",
    "code_pair_purity",
    "code_mcc",
    "forecast/code_use_ratio",
)
CODE_ALIGNED_SHD = "masks/parameter_code_aligned_shd"
ENDPOINT_KEYS = (
    "state_mcc",
    "block_purity",
    "normalized_prediction_mse",
    "online_state_probe/r2",
    "masks/state_edges",
    "masks/state_aligned_shd",
    "masks/action_edges",
    "masks/action_aligned_shd",
    "masks/parameter_edges",
    "masks/parameter_aligned_shd",
    CODE_ALIGNED_SHD,
    "masks/parameter_code_aligned_missing_true_edges",
    "masks/parameter_code_aligned_false_edges",
    "geco/relative_prediction",
    *CODE_KEYS,
)
LABELS = {
    "dense": "Dense",
    "baseline": "State/action\nsparsity",
    "full": "+ parameter\nsparsity",
}
SHORT = {"dense": "Dense", "baseline": "Baseline", "full": "Full"}
# Validated categorical slots 1-3 (dataviz reference palette, light surface).
COLORS = {"dense": "#2a78d6", "baseline": "#eb6834", "full": "#1baf7a"}
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"
BLOCKS = ((0, 1, 2), (3, 4, 5))
type Array = NDArray[np.float64]

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


@dataclass(frozen=True)
class Run:
    """One authenticated endpoint and the series the figures need."""

    arm: str
    seed: int
    directory: Path
    metrics: dict[str, float]
    start: dict[str, float] | None
    curve: list[tuple[int, float, float]]
    penalty: list[tuple[int, float]]
    columns: list[int]
    correlation: Array
    graphs: dict[str, NDArray[np.bool_]]
    group_moments: Array
    physical_samples_sha256: str
    data_identity: str
    code_correlation: Array | None = None
    code_columns: list[int] | None = None


def _aligned_graphs(state: dict[str, Any], columns: list[int]) -> dict[str, NDArray[np.bool_]]:
    """Thresholded final masks in true-coordinate order; fixed-open blocks are all ones."""
    sizes = {"state": 6, "action": 24, "parameter": 6}
    order = np.asarray(columns)
    result: dict[str, NDArray[np.bool_]] = {}
    for name, inputs in sizes.items():
        logits = state.get(f"predictor.{name}_logits")
        graph = (
            np.ones((6, inputs), dtype=bool)
            if logits is None
            else (cast(torch.Tensor, logits).detach().cpu().numpy() > 0)
        )
        graph = graph[order]
        result[name] = graph[:, order] if name == "state" else graph
    return result


def history_series(
    rows: list[dict[str, Any]],
) -> tuple[list[tuple[int, float, float]], list[tuple[int, float]]]:
    """Keep the latest replay of each logged step, separately for validation and training.

    A resumed run can append repeated steps after restoring its last checkpoint.
    Raw logs remain intact; figures follow the retained optimization trajectory.
    """
    curves: dict[int, tuple[float, float]] = {}
    penalties: dict[int, float] = {}
    for row in rows:
        step = int(row["step"])
        if "validation/state_mcc" in row:
            curves[step] = (
                float(row["validation/state_mcc"]),
                float(row["validation/block_purity"]),
            )
        if "geco/penalty_weight" in row and "loss/prediction" in row:
            penalties[step] = math.log10(float(row["geco/penalty_weight"]))
    return [(step, *values) for step, values in sorted(curves.items())], sorted(penalties.items())


def load_run(directory: Path, arm: str, seed: int) -> Run:
    """Authenticate one run directory and extract its endpoint and curves."""
    verifiers = {
        run.SCHEMA: run.verify_completion,
        from_dense.SCHEMA: from_dense.verify_completion,
        learned.SCHEMA: learned.verify_completion,
    }
    schema = read_json(directory / "completion.json").get("schema")
    if schema not in verifiers:
        raise ValueError(f"{directory} has an unknown completion schema {schema}")
    completion = verifiers[schema](directory, kind="train")
    if completion["condition"] != arm or completion["seed"] != seed:
        raise ValueError(f"{directory} does not hold {arm} seed {seed}")
    metrics = cast(dict[str, float], read_json(directory / "validation.json")["metrics"])
    startup = read_json(directory / "startup.json")["metrics"]
    rows = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
    curve, penalty = history_series(rows)
    columns = [int(c) for c in read_json(directory / "matching.json")["online_columns"]]
    samples = cast(
        dict[str, Any],
        torch.load(directory / "validation.pt", map_location="cpu", weights_only=False),
    )["samples"]
    learned_states = cast(torch.Tensor, samples["online_states"]).reshape(-1, 6).double()
    truth = cast(torch.Tensor, samples["states"]).reshape(-1, 6).double()
    # pearson_matrix has true rows and learned columns; order columns by the matching.
    correlation = pearson_matrix(learned_states, truth)[:, columns].numpy()
    state = cast(
        dict[str, Any],
        torch.load(directory / "last.pt", map_location="cpu", weights_only=False),
    )["model"]
    graphs = _aligned_graphs(state, columns)
    code_correlation = code_columns = None
    if "codes" in samples:
        codes = cast(torch.Tensor, samples["codes"]).double()
        theta = (
            cast(torch.Tensor, samples["theta"]).double()[:, None].expand(-1, codes.shape[1], -1)
        )
        code_correlation, code_columns = _matched_code_correlation(
            codes.flatten(0, 1), theta.flatten(0, 1)
        )
        if "predictor.parameter_logits" in state:
            # The runner aligns only the state rows; a learned code's columns are unordered.
            order = _code_order(codes.flatten(0, 1), theta.flatten(0, 1))
            graphs["parameter"] = graphs["parameter"][:, order]
            metrics = metrics | _code_aligned_metrics(graphs["parameter"])
    return Run(
        arm=arm,
        seed=seed,
        directory=directory,
        metrics=metrics,
        start=None if arm == "dense" else cast(dict[str, float], startup["validation"]),
        curve=curve,
        penalty=penalty,
        columns=columns,
        correlation=correlation,
        graphs=graphs,
        group_moments=matched_group_moments(samples["online_states"], samples["states"], columns),
        physical_samples_sha256=tensor_fingerprint(samples["states"]),
        data_identity=completion["data_identity"],
        code_correlation=code_correlation,
        code_columns=code_columns,
    )


# Parameter pairs (1,4), (2,5), (3,6) placed next to each other for display.
PAIR_ORDER = (0, 3, 1, 4, 2, 5)


def _matched_code_correlation(codes: torch.Tensor, theta: torch.Tensor) -> tuple[Array, list[int]]:
    """|corr(theta_k, code_j)| with theta in pair order and six codes matched to it.

    A longer code contributes its six best-matched coordinates; their indices return too.
    """
    correlation = pearson_matrix(codes, theta).numpy()[list(PAIR_ORDER)]
    rows, cols = cast(tuple[Array, Array], linear_sum_assignment(-np.abs(correlation)))
    order = cols.astype(int)[np.argsort(rows.astype(int))]
    return np.abs(correlation[:, order]), [int(c) for c in order]


def _code_order(codes: torch.Tensor, theta: torch.Tensor) -> list[int]:
    """The code coordinate matched to each theta_k, by parameter pair first.

    Complementary parameter graphs identify a code only up to mixing within the pairs
    (1,4), (2,5), (3,6), and swapping the two coordinates of a pair leaves the true graph
    unchanged. Each pair therefore takes the two coordinates its linear fit explains
    best; |correlation| only orders them within the pair.
    """
    if codes.shape[1] != 6:
        raise ValueError("code alignment needs a six-dimensional code")
    codes = (codes - codes.mean(0)) / codes.std(0).clamp_min(1e-10)
    theta = (theta - theta.mean(0)) / theta.std(0).clamp_min(1e-10)
    pair_r2 = np.zeros((6, len(learned.PAIRS)))
    for p, pair in enumerate(learned.PAIRS):
        design = torch.cat([theta[:, list(pair)], torch.ones(len(theta), 1).double()], 1)
        weights = cast(
            torch.Tensor,
            torch.linalg.solve(design.T @ design, design.T @ codes),  # pyright: ignore[reportUnknownMemberType]
        )
        fit = design @ weights
        pair_r2[:, p] = (1 - ((fit - codes) ** 2).mean(0)).numpy()
    pair_of = [next(p for p, pair in enumerate(learned.PAIRS) if k in pair) for k in range(6)]
    correlation = np.abs(pearson_matrix(codes, theta).numpy()).T  # rows: codes
    score = pair_r2[:, pair_of] + 1e-3 * correlation
    rows, cols = cast(tuple[Array, Array], linear_sum_assignment(-score))
    return [int(c) for c in rows.astype(int)[np.argsort(cols.astype(int))]]


def _code_aligned_metrics(graph: NDArray[np.bool_]) -> dict[str, float]:
    expected = true_graphs().parameter.numpy()
    return {
        CODE_ALIGNED_SHD: float((graph != expected).sum()),
        "masks/parameter_code_aligned_missing_true_edges": float((~graph & expected).sum()),
        "masks/parameter_code_aligned_false_edges": float((graph & ~expected).sum()),
    }


def load_batch(batch: Path) -> dict[str, dict[int, Run]]:
    """Discover ``{arm}-s{seed}/results`` directories; every arm needs the same seeds."""
    runs: dict[str, dict[int, Run]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for path in sorted(batch.glob(f"{arm}-s*/results")):
            seed = int(path.parent.name.removeprefix(f"{arm}-s"))
            runs[arm][seed] = load_run(path, arm, seed)
    seeds = {arm: sorted(found) for arm, found in runs.items()}
    if not seeds["dense"] or any(value != seeds["dense"] for value in seeds.values()):
        raise ValueError(f"arms must share a nonempty seed set, found {seeds}")
    return runs


def random_block_rotation(sizes: tuple[int, ...], rng: np.random.Generator) -> Array:
    """Haar-random orthogonal matrix, block-diagonal with the given block sizes."""
    total = sum(sizes)
    matrix = np.zeros((total, total))
    start = 0
    for size in sizes:
        q, r = np.linalg.qr(rng.standard_normal((size, size)))
        matrix[start : start + size, start : start + size] = q * np.sign(np.diag(r))
        start += size
    return matrix


def matched_mcc(correlation: Array) -> float:
    """Mean absolute correlation under the optimal one-to-one assignment."""
    scores = np.abs(correlation)
    rows, cols = cast(tuple[Array, Array], linear_sum_assignment(-scores))
    return float(scores[rows.astype(int), cols.astype(int)].mean())


def chance_levels(sizes: tuple[int, ...], draws: int = 5000, seed: int = 0) -> dict[str, float]:
    """MCC and block purity of a random rotation of whitened independent coordinates.

    For ``z_hat = M z`` with orthogonal ``M`` and ``z ~ N(0, I)``, corr(z_hat_i, z_j)
    is ``M_ij`` and the best single-block R² of ``z_hat_i`` is ``max_B ||M_iB||²``.
    """
    rng = np.random.default_rng(seed)
    mcc: list[float] = []
    purity: list[float] = []
    for _ in range(draws):
        matrix = random_block_rotation(sizes, rng)
        mcc.append(matched_mcc(matrix))
        purity.append(
            float(
                np.mean(
                    [
                        max(float((matrix[i, np.asarray(b)] ** 2).sum()) for b in BLOCKS)
                        for i in range(len(matrix))
                    ]
                )
            )
        )
    return {
        "mcc_mean": float(np.mean(mcc)),
        "mcc_p05": float(np.percentile(mcc, 5)),
        "mcc_p95": float(np.percentile(mcc, 95)),
        "purity_mean": float(np.mean(purity)),
    }


def describe(values: list[float]) -> dict[str, float]:
    """Seed-level summary statistics."""
    array = np.asarray(values, dtype=float)
    return {
        "n": float(len(array)),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def paired(left: list[float], right: list[float], *, seed: int = 0) -> dict[str, float]:
    """Seed-paired difference ``left - right`` with an exact sign-flip test.

    The permutation distribution enumerates all 2^n sign assignments of the paired
    differences; the one-sided p-value is P(mean >= observed). The percentile
    bootstrap resamples seeds; with five seeds it is coarse and only descriptive.
    """
    diffs = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    n = len(diffs)
    observed = float(diffs.mean())
    signs = np.array(
        [[1.0 if (mask >> bit) & 1 else -1.0 for bit in range(n)] for mask in range(2**n)]
    )
    flipped = (signs * np.abs(diffs)).mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = rng.choice(diffs, size=(10_000, n), replace=True).mean(axis=1)
    return {
        "n": float(n),
        "mean": observed,
        "sd": float(diffs.std(ddof=1)) if n > 1 else 0.0,
        "positive": float((diffs > 0).sum()),
        "p_one_sided": float((flipped >= observed - 1e-12).mean()),
        "p_two_sided": float((np.abs(flipped) >= abs(observed) - 1e-12).mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def matched_group_moments(states: torch.Tensor, truth: torch.Tensor, columns: list[int]) -> Array:
    """Sufficient moments for fixed-matching Pearson MCC, keeping independent groups intact."""
    if states.shape != truth.shape or states.ndim < 3 or states.shape[-1] != 6:
        raise ValueError(
            "State tensors must share group/sample/coordinate axes with six coordinates"
        )
    if sorted(columns) != list(range(6)):
        raise ValueError("Require a fixed calibration permutation of six coordinates")
    x = states.detach().double().cpu().reshape(len(states), -1, 6).numpy()[:, :, columns]
    y = truth.detach().double().cpu().reshape(len(truth), -1, 6).numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite state samples")
    # A common translation leaves every resampled correlation unchanged and
    # avoids cancellation for nearly constant coordinates with large means.
    x = x - x.mean((0, 1))
    y = y - y.mean((0, 1))
    return np.stack(
        [x.mean(1), y.mean(1), (x * x).mean(1), (y * y).mean(1), (x * y).mean(1)], axis=1
    )


def _moment_mcc(moments: Array) -> Array:
    x, y, xx, yy, xy = (moments[..., index, :] for index in range(5))
    denominator = np.sqrt(np.maximum(xx - x * x, 0) * np.maximum(yy - y * y, 0))
    correlations = (xy - x * y) / np.maximum(denominator, 1e-30)
    return np.minimum(np.abs(correlations), 1).mean(-1)


def paired_group_mcc(
    left: Array, right: Array, *, draws: int = 1000, seed: int = 2026103001
) -> dict[str, float]:
    """Paired group bootstrap conditional on one training seed and fixed calibration matches."""
    if left.shape != right.shape or left.ndim != 3 or left.shape[1:] != (5, 6):
        raise ValueError("Require matching group-level MCC moments")
    groups = len(left)
    if groups < 2 or draws < 2 or seed < 0:
        raise ValueError("Require at least two groups/draws and a nonnegative seed")
    rng = np.random.default_rng(seed)
    indices = rng.integers(groups, size=(draws, groups))
    boot = _moment_mcc(left[indices].mean(1)) - _moment_mcc(right[indices].mean(1))
    return {
        "groups": float(groups),
        "bootstrap_draws": float(draws),
        "bootstrap_seed": float(seed),
        "estimate": float(_moment_mcc(left.mean(0)) - _moment_mcc(right.mean(0))),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def group_mcc_contrasts(runs: dict[str, dict[int, Run]], seeds: list[int]) -> dict[str, Any]:
    """Check group pairing and reproduce every endpoint MCC before reporting its differences."""
    contrasts = {"full - baseline": ("full", "baseline"), "baseline - dense": ("baseline", "dense")}
    result: dict[str, Any] = {}
    for name, (left, right) in contrasts.items():
        result[name] = {}
        for seed in seeds:
            a, b = runs[left][seed], runs[right][seed]
            if (
                a.data_identity != b.data_identity
                or a.physical_samples_sha256 != b.physical_samples_sha256
            ):
                raise ValueError("Paired MCC contrasts require identical ordered physical groups")
            for item in (a, b):
                if (
                    abs(float(_moment_mcc(item.group_moments.mean(0))) - item.metrics["state_mcc"])
                    > 1e-6
                ):
                    raise ValueError("Group moments do not reproduce the recorded fixed-match MCC")
            result[name][str(seed)] = paired_group_mcc(a.group_moments, b.group_moments)
    return result


def _metric(run_: Run, key: str) -> float:
    return float(run_.metrics[key])


def _start_mcc(run_: Run) -> float | None:
    return None if run_.start is None else float(run_.start["state_mcc"])


def _save(figure: Figure, output: Path, name: str) -> list[str]:
    paths: list[str] = []
    for suffix in ("pdf", "png"):
        path = output / f"{name}.{suffix}"
        cast(Any, figure).savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path.name)
    plt.close(figure)
    return paths


def _box_panel(
    target: Axes, runs: dict[str, dict[int, Run]], key: str, seeds: list[int], ylabel: str
) -> None:
    """Box per arm, seed points, and thin lines joining the same seed across arms."""
    axis: Any = target
    values = {arm: [_metric(runs[arm][s], key) for s in seeds] for arm in ARMS}
    positions = list(range(len(ARMS)))
    offsets = np.linspace(-0.12, 0.12, len(seeds)) if len(seeds) > 1 else np.zeros(1)
    for seed_index in range(len(seeds)):
        axis.plot(
            [position + offsets[seed_index] for position in positions],
            [values[arm][seed_index] for arm in ARMS],
            color=GRID,
            lw=0.8,
            zorder=1,
        )
    box = axis.boxplot(
        [values[arm] for arm in ARMS],
        positions=positions,
        widths=0.5,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": INK, "lw": 1.2},
        whiskerprops={"color": INK2, "lw": 0.8},
        capprops={"color": INK2, "lw": 0.8},
        boxprops={"lw": 0.8, "edgecolor": INK2},
        zorder=2,
    )
    for patch, arm in zip(box["boxes"], ARMS, strict=True):
        patch.set_facecolor(COLORS[arm])
        patch.set_alpha(0.28)
    for position, arm in zip(positions, ARMS, strict=True):
        axis.scatter(
            position + offsets,
            values[arm],
            s=22,
            color=COLORS[arm],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    axis.set_xticks(positions, [LABELS[arm] for arm in ARMS])
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color=GRID, lw=0.6)
    axis.set_axisbelow(True)


def _chance_reference(
    target: Axes, x: float, label_x: float, levels: dict[str, float], text: str
) -> None:
    """A dashed mean line plus a 5-95% range bar in the right margin, labelled."""
    axis: Any = target
    axis.axhline(levels["mcc_mean"], color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=0)
    axis.plot([x, x], [levels["mcc_p05"], levels["mcc_p95"]], color=INK2, lw=1.6)
    axis.plot([x], [levels["mcc_mean"]], marker="_", color=INK2, ms=9, mew=1.6)
    axis.text(label_x, levels["mcc_mean"], text, color=INK2, fontsize=6.8, va="center")


def figure_mcc(runs: dict[str, dict[int, Run]], seeds: list[int], chance: dict[str, Any]) -> Figure:
    """State MCC per arm with seed pairing and chance references."""
    figure, axis = cast(tuple[Any, Any], plt.subplots(figsize=(4.4, 3.0)))
    _chance_reference(axis, 2.5, 2.72, chance["any_rotation"], "chance:\nany rotation")
    _chance_reference(axis, 2.62, 2.72, chance["within_block"], "chance:\nwithin-block\nrotation")
    _box_panel(axis, runs, "state_mcc", seeds, "State MCC (validation)")
    axis.set_ylim(0.5, 1.01)
    axis.set_xlim(-0.5, 2.68)
    axis.spines["bottom"].set_bounds(-0.5, 2.4)
    axis.set_title(f"State recovery, {len(seeds)} training seeds per model", loc="left", color=INK)
    return figure


def figure_purity(
    runs: dict[str, dict[int, Run]], seeds: list[int], chance: dict[str, Any]
) -> Figure:
    """Block purity per arm with the any-rotation chance level."""
    figure, axis = cast(tuple[Any, Any], plt.subplots(figsize=(4.2, 3.0)))
    level = chance["any_rotation"]["purity_mean"]
    axis.axhline(level, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=0)
    axis.text(2.42, level, "chance:\nany rotation", color=INK2, fontsize=6.8, va="center")
    _box_panel(axis, runs, "block_purity", seeds, "Block purity (validation R²)")
    axis.set_ylim(0.5, 1.01)
    axis.set_xlim(-0.5, 2.4)
    axis.set_title("Separation of the two state blocks", loc="left", color=INK)
    return figure


def figure_curves(
    runs: dict[str, dict[int, Run]], seeds: list[int], chance: dict[str, Any]
) -> Figure:
    """Validation MCC during sparse training, per seed and median."""
    figure, axis = cast(tuple[Any, Any], plt.subplots(figsize=(5.2, 3.0)))
    block = chance["within_block"]
    axis.axhspan(block["mcc_p05"], block["mcc_p95"], color=BAND, alpha=0.45, lw=0, zorder=0)
    axis.axhline(block["mcc_mean"], color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=0)
    last_step = 0
    for arm in ("baseline", "full"):
        curves = [runs[arm][s].curve for s in seeds]
        common = {step for step, _, _ in curves[0]}
        for other in curves[1:]:
            common &= {step for step, _, _ in other}
        steps: list[int] = sorted(common)
        last_step = max(last_step, steps[-1])
        stacked = np.array(
            [[dict((s, m) for s, m, _ in c)[step] for step in steps] for c in curves]
        )
        for row in stacked:
            axis.plot(np.asarray(steps) / 1000, row, color=COLORS[arm], lw=0.7, alpha=0.35)
        axis.plot(
            np.asarray(steps) / 1000,
            np.median(stacked, axis=0),
            color=COLORS[arm],
            lw=2.0,
            label=f"{SHORT[arm]}: {LABELS[arm].replace(chr(10), ' ')} (median)",
        )
        axis.text(
            steps[-1] / 1000 + 2,
            float(np.median(stacked[:, -1])),
            SHORT[arm],
            color=INK,
            fontsize=7,
            va="center",
        )
    axis.text(1, block["mcc_mean"] + 0.012, "within-block chance", color=INK2, fontsize=6.8)
    axis.set_xlim(0, last_step / 1000 * 1.1)
    axis.set_ylim(0.5, 1.01)
    axis.set_xlabel("Sparse updates (thousands), starting from each seed's dense endpoint")
    axis.set_ylabel("State MCC (validation)")
    axis.grid(axis="y", color=GRID, lw=0.6)
    axis.legend(loc="lower right", frameon=False)
    axis.set_title("Learning curves (thin: individual seeds)", loc="left", color=INK)
    return figure


def figure_graphs(runs: dict[str, dict[int, Run]], seeds: list[int]) -> Figure:
    """Edge-recovery frequency of the aligned learned graphs."""
    learned_codes = runs["full"][seeds[0]].code_columns is not None
    truth = true_graphs()
    expected = {"state": truth.state[:, 0].numpy(), "parameter": truth.parameter.numpy()}
    figure, axes = cast(
        tuple[Any, Any], plt.subplots(2, 3, figsize=(6.4, 4.4), constrained_layout=True)
    )
    columns = (("truth", "true"), ("baseline", "baseline"), ("full", "full"))
    names = (("state", "State graph"), ("parameter", "Parameter graph"))
    image = None
    for row, (block, row_title) in enumerate(names):
        for col, (source, title) in enumerate(columns):
            axis = axes[row, col]
            if source == "truth":
                matrix = expected[block].astype(float)
            else:
                matrix = np.mean([runs[source][s].graphs[block] for s in seeds], axis=0)
            image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
            for i in range(6):
                for j in range(6):
                    if expected[block][i, j]:
                        axis.add_patch(
                            Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=0.9)
                        )
                    if source != "truth":
                        axis.text(
                            j,
                            i,
                            f"{matrix[i, j] * len(seeds):.0f}",
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="white" if matrix[i, j] > 0.6 else INK2,
                        )
            symbol = "z" if block == "state" else "θ"
            if block == "parameter" and source != "truth" and learned_codes:
                symbol = "q"
                axis.set_xlabel("Matched code positions", fontsize=7)
            labels = [f"{symbol}{k + 1}" for k in range(6)]
            axis.set_xticks(range(6), labels, fontsize=6.5)
            axis.set_yticks(range(6), [f"z{k + 1}'" for k in range(6)], fontsize=6.5)
            axis.tick_params(length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.set_title(f"{row_title}, {title}", loc="left", color=INK, fontsize=8)
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.6, label="Fraction of seeds with the edge")
    heading = (
        "Aligned input masks; q denotes code positions matched to parameter pairs."
        if learned_codes
        else "Aligned state and oracle-parameter input masks."
    )
    figure.suptitle(
        heading + "\nRows: next state; "
        f"outlined: true edges; numbers: seeds (of {len(seeds)}) with the edge",
        x=0.02,
        ha="left",
        color=INK,
        fontsize=8,
    )
    return figure


def figure_correlations(runs: dict[str, dict[int, Run]], seed: int) -> Figure:
    """Matched |correlation| matrices for one seed of every arm."""
    figure, axes = cast(
        tuple[Any, Any], plt.subplots(1, 3, figsize=(6.4, 2.4), constrained_layout=True)
    )
    image = None
    for axis, arm in zip(axes, ARMS, strict=True):
        run_ = runs[arm][seed]
        image = axis.imshow(np.abs(run_.correlation), vmin=0, vmax=1, cmap="Blues")
        axis.axhline(2.5, color=INK2, lw=0.6)
        axis.axvline(2.5, color=INK2, lw=0.6)
        axis.set_xticks(range(6), [f"ẑ{k + 1}" for k in range(6)], fontsize=6.5)
        axis.set_yticks(range(6), [f"z{k + 1}" for k in range(6)], fontsize=6.5)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(
            f"{SHORT[arm]}  (MCC {run_.metrics['state_mcc']:.3f})", loc="left", color=INK
        )
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.8, label="|Pearson correlation|")
    figure.suptitle(
        f"Learned vs true states, seed {seed} (learned coordinates ordered by the matching)",
        x=0.02,
        ha="left",
        color=INK,
        fontsize=8.5,
    )
    return figure


def figure_quality(runs: dict[str, dict[int, Run]], seeds: list[int]) -> Figure:
    """Forecast error and whole-state information per arm."""
    figure, axes = cast(tuple[Any, Any], plt.subplots(1, 2, figsize=(6.4, 2.7)))
    _box_panel(axes[0], runs, "normalized_prediction_mse", seeds, "Normalized forecast MSE")
    _box_panel(axes[1], runs, "online_state_probe/r2", seeds, "Whole-state linear R²")
    axes[0].set_title("Forecast error (two-step)", loc="left", color=INK)
    axes[1].set_title("Retained state information", loc="left", color=INK)
    figure.tight_layout()
    return figure


def figure_penalty(runs: dict[str, dict[int, Run]], seeds: list[int]) -> Figure:
    """GECO penalty weight over sparse training (supplementary)."""
    figure, axes = cast(
        tuple[Any, Any], plt.subplots(2, 1, figsize=(6.4, 3.6), sharex=True, sharey=True)
    )
    for axis, arm in zip(axes, ("baseline", "full"), strict=True):
        for seed in seeds:
            steps, values = zip(*runs[arm][seed].penalty, strict=True)
            axis.plot(np.asarray(steps) / 1000, values, color=COLORS[arm], lw=0.6, alpha=0.6)
        axis.set_ylabel("log10 penalty weight")
        axis.set_title(f"{SHORT[arm]}: GECO multiplier, all seeds", loc="left", color=INK)
        axis.grid(axis="y", color=GRID, lw=0.6)
    axes[-1].set_xlabel("Sparse updates (thousands)")
    figure.tight_layout()
    return figure


def figure_code_retention(runs: dict[str, dict[int, Run]], seeds: list[int]) -> Figure:
    """Whole-code retention R² per parameter, with every seed visible, including outliers."""
    figure, axis = cast(tuple[Any, Any], plt.subplots(figsize=(6.4, 2.8)))
    width = 0.24
    for offset, arm in zip((-width, 0.0, width), ARMS, strict=True):
        values = [
            [_metric(runs[arm][s], f"code_r2/theta_{k + 1}") for s in seeds] for k in range(6)
        ]
        positions = np.arange(6) + offset
        box = axis.boxplot(
            values,
            positions=positions,
            widths=width * 0.85,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": INK, "lw": 1.0},
            whiskerprops={"color": INK2, "lw": 0.7},
            capprops={"color": INK2, "lw": 0.7},
            boxprops={"lw": 0.7, "edgecolor": INK2},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(COLORS[arm])
            patch.set_alpha(0.45)
        jitter = np.linspace(-0.07, 0.07, len(seeds)) if len(seeds) > 1 else np.zeros(1)
        for position, scores in zip(positions, values, strict=True):
            axis.scatter(
                position + jitter,
                scores,
                s=12,
                color=COLORS[arm],
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        axis.plot([], [], color=COLORS[arm], lw=6, alpha=0.6, label=SHORT[arm])
    axis.set_xticks(range(6), [f"θ{k + 1}" for k in range(6)])
    axis.set_ylabel("Whole-code retention R² (validation)")
    axis.axhline(0.8, color=INK2, lw=0.7, ls=(0, (3, 2)))
    axis.text(5.55, 0.8, "  0.8", color=INK2, fontsize=6.8, va="center")
    axis.grid(axis="y", color=GRID, lw=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, ncol=3)
    axis.set_title(
        "Parameter retention from one context (dots: individual training seeds)",
        loc="left",
        color=INK,
    )
    return figure


def figure_code_correlations(runs: dict[str, dict[int, Run]], seed: int) -> Figure:
    """Matched |corr(theta, code)| for one seed; theta rows grouped by parameter pair."""
    figure, axes = cast(
        tuple[Any, Any], plt.subplots(1, 3, figsize=(6.4, 2.4), constrained_layout=True)
    )
    labels = [f"θ{k + 1}" for k in PAIR_ORDER]
    image = None
    for axis, arm in zip(axes, ARMS, strict=True):
        matrix = runs[arm][seed].code_correlation
        if matrix is None:
            raise ValueError("code correlations require a learned-code batch")
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        for boundary in (1.5, 3.5):
            axis.axhline(boundary, color=INK2, lw=0.6)
            axis.axvline(boundary, color=INK2, lw=0.6)
        columns = runs[arm][seed].code_columns or list(range(6))
        axis.set_xticks(range(6), [f"q{j + 1}" for j in columns], fontsize=6.5)
        axis.set_yticks(range(6), labels, fontsize=6.5)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(
            f"{SHORT[arm]}  (code MCC {runs[arm][seed].metrics['code_mcc']:.3f})",
            loc="left",
            color=INK,
        )
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.8, label="|Pearson correlation|")
    figure.suptitle(
        f"Learned code vs true parameters, seed {seed} (rows grouped by parameter pair)",
        x=0.02,
        ha="left",
        color=INK,
        fontsize=8.5,
    )
    return figure


def load_projections(batch: Path, seeds: list[int]) -> dict[int, dict[str, float]] | None:
    """Authenticated twelve-to-six code projections, one per seed, or None if absent."""
    found = sorted(batch.glob("projected-s*/results"))
    if not found:
        return None
    result: dict[int, dict[str, float]] = {}
    for path in found:
        completion = learned.verify_completion(path, kind="projection")
        seed = int(completion["seed"])
        record = read_json(path / "projection.json")
        metrics = read_json(path / "validation.json")["metrics"]
        source = record["source_validation"]
        result[seed] = {
            "source_code_dim": float(record["source"]["code_dim"]),
            "source_r2_min": float(source["code_r2_min"]),
            "source_r2_mean": float(source["code_r2_mean"]),
            "source_nmse": float(source["normalized_prediction_mse"]),
            "projected_r2_min": float(metrics["code_r2_min"]),
            "projected_r2_mean": float(metrics["code_r2_mean"]),
            "projected_nmse": float(metrics["normalized_prediction_mse"]),
            "projected_pair_purity": float(metrics["code_pair_purity"]),
            "reconstructed_variance": float(
                record["projection"]["reconstructed_variance_fraction"]
            ),
        }
    if sorted(result) != seeds:
        raise ValueError(f"projections cover seeds {sorted(result)}, runs cover {seeds}")
    return result


def write_projection_table(projections: dict[int, dict[str, float]], output: Path) -> None:
    """LaTeX table of retention and forecast error before and after the projection."""
    rows = [
        f"{seed} & {v['source_r2_min']:.3f} & {v['projected_r2_min']:.3f} & "
        f"{v['source_r2_mean']:.3f} & {v['projected_r2_mean']:.3f} & "
        f"{v['source_nmse']:.4f} & {v['projected_nmse']:.4f} & "
        f"{v['reconstructed_variance']:.3f} \\\\"
        for seed, v in sorted(projections.items())
    ]
    dimension = int(next(iter(projections.values()))["source_code_dim"])
    table = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lccccccc}",
            r"\toprule",
            r" & \multicolumn{2}{c}{Min.\ retention $R^2$} & "
            r"\multicolumn{2}{c}{Mean retention $R^2$} & \multicolumn{2}{c}{NMSE} & \\",
            f"Seed & {dimension}-code & 6-code & {dimension}-code & 6-code & "
            f"{dimension}-code & 6-code & Code variance kept \\\\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Projection of each acquired "
            + str(dimension)
            + r"-dimensional code onto the six leading eigenvectors of its TRAIN group-shared "
            r"code covariance in the initialization pipeline (validation). The "
            r"projection uses no parameter labels; code variance kept is the TRAIN fraction "
            r"of centered code variance its least-squares reconstruction explains.}",
            r"\label{tab:masked-mlp-learned-projection}",
            r"\end{table}",
            "",
        ]
    )
    (output / "table_projection.tex").write_text(table)


def load_adaptations(batch: Path, seeds: list[int]) -> dict[int, dict[str, float]] | None:
    """Authenticate a projected-to-adapted dense comparison without opening test data."""
    records = {seed: read_json(batch / f"dense-s{seed}/results/completion.json") for seed in seeds}
    if not any(record.get("adaptation_source") is not None for record in records.values()):
        return None
    result: dict[int, dict[str, float]] = {}
    for seed, record in records.items():
        source = record.get("adaptation_source")
        if not isinstance(source, dict):
            raise ValueError("all dense seeds must share the adaptation protocol")
        source = cast(dict[str, Any], source)
        directory = batch / f"dense-s{seed}/results"
        completion = learned.verify_completion(directory, kind="train")
        projection = batch / f"projected-s{seed}/results"
        if not projection.exists():
            projection = Path(source["directory"])
        parent = learned.verify_completion(projection, kind="projection")
        if (
            completion["condition"] != "dense"
            or completion["code_dim"] != 6
            or completion["seed"] != seed
            or parent["seed"] != seed
            or parent["data_identity"] != completion["data_identity"]
            or parent["identity"] != source["identity"]
            or parent["code_identity"] != source["code_identity"]
            or sha(projection / "completion.json") != source["completion_sha256"]
            or sha(projection / "last.pt") != source["checkpoint_sha256"]
        ):
            raise ValueError("adaptation and its declared projection differ")
        for arm in ("baseline", "full"):
            sparse = read_json(batch / f"{arm}-s{seed}/results/completion.json")
            sparse_parent = sparse.get("dense_endpoint")
            if not isinstance(sparse_parent, dict):
                raise ValueError("sparse arms must use the reported adapted dense endpoint")
            sparse_parent = cast(dict[str, Any], sparse_parent)
            if (
                sparse_parent.get("identity") != completion["identity"]
                or sparse_parent.get("completion_sha256") != sha(directory / "completion.json")
                or sparse_parent.get("checkpoint_sha256") != sha(directory / "last.pt")
            ):
                raise ValueError("sparse arms must use the reported adapted dense endpoint")
        row = {"updates": float(completion["steps"])}
        for stage, path in (("projected", projection), ("adapted", directory)):
            metrics = read_json(path / "validation.json")["metrics"]
            row[f"{stage}_r2_min"] = float(metrics["code_r2_min"])
            row[f"{stage}_nmse"] = float(metrics["normalized_prediction_mse"])
            row[f"{stage}_calibration_prediction"] = float(
                read_json(path / "calibration.json")["metrics"]["prediction"]
            )
        result[seed] = row
    return result


def write_adaptation_table(adaptations: dict[int, dict[str, float]], output: Path) -> None:
    """Show the fixed six-code adaptation budget, retained information and forecast change."""
    rows = [
        f"{seed} & {int(row['updates'])} & {row['projected_r2_min']:.3f} & "
        f"{row['adapted_r2_min']:.3f} & {row['projected_nmse']:.4f} & "
        f"{row['adapted_nmse']:.4f}" + r" \\"
        for seed, row in sorted(adaptations.items())
    ]
    table = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Seed & Updates & Min.\ $R^2$ before & after & NMSE before & after \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Six-code adaptation from the fixed projected endpoints (validation). "
            r"Both learned state encoders remain frozen; only the parameter encoder and "
            r"dense predictor learn from the ordinary label-free forecast objective. "
            r"All values use the final fixed-budget endpoint; retention is measured from "
            r"the complete code with the unchanged calibration-fitted probes.}",
            r"\label{tab:masked-mlp-learned-adaptation}",
            r"\end{table}",
            "",
        ]
    )
    (output / "table_adaptation.tex").write_text(table)


def write_code_table(runs: dict[str, dict[int, Run]], seeds: list[int], output: Path) -> None:
    """LaTeX table of code retention, alignment and predictive use per model."""
    keys = {
        "code_r2_min": "Min.\\ retention $R^2$",
        "code_r2_mean": "Mean retention $R^2$",
        "code_pair_purity": "Pair purity",
        "code_mcc": "Code MCC",
        "forecast/code_use_ratio": "Shuffled/factual NMSE",
    }
    rows: list[str] = []
    for arm in ARMS:
        cells = [SHORT[arm]] + [
            _fmt(describe([_metric(runs[arm][s], key) for s in seeds]), 2 if "ratio" in key else 3)
            for key in keys
        ]
        rows.append(" & ".join(cells) + r" \\")
    table = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{l" + "c" * len(keys) + "}",
            r"\toprule",
            "Model & " + " & ".join(keys.values()) + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Learned parameter codes, mean $\pm$ s.d.\ over "
            + str(len(seeds))
            + r" seeds (validation). Retention: linear or cubic probe of each true parameter "
            r"from the whole code, fitted and selected on calibration halves. Pair purity: "
            r"best single-pair linear $R^2$ of each code coordinate (pairs (1,4), (2,5), (3,6)). "
            r"Shuffled/factual: forecast error with codes from another parameter group, "
            r"relative to the factual error.}",
            r"\label{tab:masked-mlp-learned-codes}",
            r"\end{table}",
            "",
        ]
    )
    (output / "table_codes.tex").write_text(table)


def _fmt(stats: dict[str, float], digits: int = 3) -> str:
    # A spread that rounds to zero is printed as a bound, not as "0.000".
    floor = 0.5 * 10**-digits
    spread = f"$<${2 * floor:.{digits}f}" if stats["sd"] < floor else f"{stats['sd']:.{digits}f}"
    return f"{stats['mean']:.{digits}f} $\\pm$ {spread}"


def write_tables(
    runs: dict[str, dict[int, Run]], seeds: list[int], output: Path, chance: dict[str, Any]
) -> dict[str, Any]:
    """LaTeX tables of per-arm summaries and seed-paired MCC tests."""
    keys = {
        "state_mcc": "MCC",
        "block_purity": "Block purity",
        "normalized_prediction_mse": "NMSE",
        "online_state_probe/r2": "State $R^2$",
        "masks/state_aligned_shd": "SHD$_z$",
        "masks/parameter_aligned_shd": "SHD$_\\theta$",
    }

    code_aligned = all(CODE_ALIGNED_SHD in runs["full"][s].metrics for s in seeds)
    label_prefix = "masked-mlp-learned" if code_aligned else "masked-mlp-oracle"
    summary: dict[str, dict[str, dict[str, float]]] = {
        arm: {key: describe([_metric(runs[arm][s], key) for s in seeds]) for key in keys}
        for arm in ARMS
    }
    if code_aligned:
        # Keep the runner's state-row-only metric literal in machine-readable output.
        summary["full"][CODE_ALIGNED_SHD] = describe(
            [_metric(runs["full"][s], CODE_ALIGNED_SHD) for s in seeds]
        )
    rows: list[str] = []
    for arm in ARMS:
        cells = [SHORT[arm]]
        for key in keys:
            displayed_key = (
                CODE_ALIGNED_SHD
                if key == "masks/parameter_aligned_shd" and arm == "full" and code_aligned
                else key
            )
            stats = summary[arm][displayed_key]
            if (key == "masks/parameter_aligned_shd" and arm != "full") or (
                key.startswith("masks/") and arm == "dense"
            ):
                cells.append("open")
            elif key.startswith("masks/"):
                cells.append(f"{stats['mean']:.1f} $\\pm$ {stats['sd']:.1f}")
            elif key == "normalized_prediction_mse":
                cells.append(_fmt(stats, 4))
            else:
                cells.append(_fmt(stats))
        rows.append(" & ".join(cells) + r" \\")
    within, anywhere = chance["within_block"], chance["any_rotation"]
    table = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{l" + "c" * len(keys) + "}",
            r"\toprule",
            "Model & " + " & ".join(keys.values()) + r" \\",
            r"\midrule",
            *rows,
            r"\midrule",
            f"Chance (any rotation) & {anywhere['mcc_mean']:.3f} & {anywhere['purity_mean']:.3f}"
            + " & & & & "
            + r"\\",
            f"Chance (within-block) & {within['mcc_mean']:.3f} & 1.000 & & & & " + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Validation endpoints, mean $\pm$ s.d.\ over "
            + str(len(seeds))
            + r" training seeds. SHD: structural Hamming distance of the thresholded learned "
            r"graph to the true graph after the MCC matching; ``open'': the block is not "
            r"learned (all edges present). "
            + (
                r"The learned code's coordinates are first matched to the true parameter "
                r"pairs by linear pair $R^2$ (complementary graphs identify codes only up to "
                r"mixing within a pair, which leaves the parameter graph unchanged). "
                r"This descriptive code matching uses validation samples. "
                if code_aligned
                else ""
            )
            + r"Chance rows: random orthogonal re-charting of the true states.}",
            r"\label{tab:" + label_prefix + "}",
            r"\end{table}",
            "",
        ]
    )
    (output / "table_main.tex").write_text(table)
    comparisons = {
        "full - baseline": ("full", "baseline"),
        "baseline - dense": ("baseline", "dense"),
        "full - dense": ("full", "dense"),
    }
    tests = {
        name: paired(
            [_metric(runs[a][s], "state_mcc") for s in seeds],
            [_metric(runs[b][s], "state_mcc") for s in seeds],
        )
        for name, (a, b) in comparisons.items()
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Comparison & $\Delta$MCC & s.d. & seeds $>0$ & $p$ (one-sided) & 95\% CI \\",
        r"\midrule",
    ]
    for name, result in tests.items():
        lines.append(
            f"{name.replace('-', '$-$')} & {result['mean']:+.3f} & {result['sd']:.3f} & "
            f"{result['positive']:.0f}/{result['n']:.0f} & {result['p_one_sided']:.3f} & "
            f"[{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]" + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Seed-paired MCC differences. $p$: exact sign-flip permutation test over "
        r"all $2^n$ sign assignments (smallest attainable value $2^{-n}$). CI: percentile "
        r"bootstrap over seeds, descriptive only for small $n$.}",
        r"\label{tab:" + label_prefix + "-paired}",
        r"\end{table}",
        "",
    ]
    (output / "table_paired.tex").write_text("\n".join(lines))
    return {
        "summary": summary,
        "paired": tests,
        "parameter_graph_metric": CODE_ALIGNED_SHD
        if code_aligned
        else "masks/parameter_aligned_shd",
    }


def write_csv(runs: dict[str, dict[int, Run]], seeds: list[int], output: Path) -> None:
    """One row per run with its endpoint metrics and source directory."""
    keys = ENDPOINT_KEYS
    with (output / "per_run.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "seed", "start_state_mcc", *keys, "directory"])
        for arm in ARMS:
            for seed in seeds:
                run_ = runs[arm][seed]
                start = "" if run_.start is None else f"{run_.start['state_mcc']:.6f}"
                writer.writerow(
                    [arm, seed, start]
                    + [
                        "" if key not in run_.metrics else f"{run_.metrics[key]:.6f}"
                        for key in keys
                    ]
                    + [str(run_.directory)]
                )


def analyse(batch: Path, output: Path) -> dict[str, Any]:
    """Write every figure and table; return the JSON summary."""
    output.mkdir(parents=True, exist_ok=True)
    runs = load_batch(batch)
    seeds = sorted(runs["dense"])
    chance = {
        "any_rotation": chance_levels((6,)),
        "within_block": chance_levels((3, 3)),
    }
    full_mcc = {s: _metric(runs["full"][s], "state_mcc") for s in seeds}
    representative = sorted(seeds, key=lambda s: full_mcc[s])[len(seeds) // 2]
    figures = {
        "fig_mcc_box": figure_mcc(runs, seeds, chance),
        "fig_block_purity_box": figure_purity(runs, seeds, chance),
        "fig_mcc_curves": figure_curves(runs, seeds, chance),
        "fig_graph_recovery": figure_graphs(runs, seeds),
        "fig_correlations_median_seed": figure_correlations(runs, representative),
        "fig_forecast_quality": figure_quality(runs, seeds),
        "fig_geco_multiplier": figure_penalty(runs, seeds),
    }
    learned_codes = all(runs[arm][s].code_correlation is not None for arm in ARMS for s in seeds)
    if learned_codes:
        figures["fig_code_retention_box"] = figure_code_retention(runs, seeds)
        figures["fig_code_correlations_median_seed"] = figure_code_correlations(
            runs, representative
        )
        write_code_table(runs, seeds, output)
    projections = load_projections(batch, seeds)
    if projections is not None:
        write_projection_table(projections, output)
    adaptations = load_adaptations(batch, seeds)
    if adaptations is not None:
        write_adaptation_table(adaptations, output)
    files = {name: _save(figure, output, name) for name, figure in figures.items()}
    tables = write_tables(runs, seeds, output, chance)
    group_contrasts = group_mcc_contrasts(runs, seeds)
    (output / "group_mcc_contrasts.json").write_text(
        json.dumps(group_contrasts, indent=2, sort_keys=True)
    )
    write_csv(runs, seeds, output)
    result: dict[str, Any] = {
        "batch": str(batch),
        "seeds": seeds,
        "representative_seed": representative,
        "learned_codes": learned_codes,
        "projections": None
        if projections is None
        else {str(seed): values for seed, values in projections.items()},
        "adaptations": None
        if adaptations is None
        else {str(seed): values for seed, values in adaptations.items()},
        "chance": chance,
        **tables,
        "paired_group_mcc": group_contrasts,
        "figures": files,
        "per_run": {
            arm: {
                str(s): {
                    "start_state_mcc": _start_mcc(runs[arm][s]),
                    **{
                        key: _metric(runs[arm][s], key)
                        for key in ENDPOINT_KEYS
                        if key in runs[arm][s].metrics
                    },
                }
                for s in seeds
            }
            for arm in ARMS
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    write_readme(result, output)
    return result


def write_readme(result: dict[str, Any], output: Path) -> None:
    """README with the headline numbers and draft figure captions."""
    summary, tests, chance = result["summary"], result["paired"], result["chance"]
    n = len(result["seeds"])
    mcc = {arm: summary[arm]["state_mcc"] for arm in ARMS}
    purity = {arm: summary[arm]["block_purity"] for arm in ARMS}
    fb = tests["full - baseline"]
    summary_codes = {
        arm: {
            key: float(np.mean([row[key] for row in result["per_run"][arm].values()]))
            for key in CODE_KEYS
            if all(key in row for row in result["per_run"][arm].values())
        }
        for arm in ARMS
    }
    source = "learned parameter codes" if result["learned_codes"] else "oracle theta"
    text = (
        f"""# Masked-MLP batch ({source}): paper figures and tables

Source batch: `{result["batch"]}` ({n} training seeds per model, validation endpoints).

| Model | MCC mean ± s.d. | MCC range | Block purity mean |
|---|---|---|---|
"""
        + "\n".join(
            f"| {SHORT[arm]} | {mcc[arm]['mean']:.3f} ± {mcc[arm]['sd']:.3f} | "
            f"{mcc[arm]['min']:.3f}-{mcc[arm]['max']:.3f} | {purity[arm]['mean']:.3f} |"
            for arm in ARMS
        )
        + f"""

Chance MCC: any rotation {chance["any_rotation"]["mcc_mean"]:.3f}
(5-95% {chance["any_rotation"]["mcc_p05"]:.3f}-{chance["any_rotation"]["mcc_p95"]:.3f});
within-block rotation {chance["within_block"]["mcc_mean"]:.3f}
({chance["within_block"]["mcc_p05"]:.3f}-{chance["within_block"]["mcc_p95"]:.3f}).

Full - baseline MCC: {fb["mean"]:+.3f} (s.d. {fb["sd"]:.3f}), positive in
{fb["positive"]:.0f}/{fb["n"]:.0f} seeds, exact one-sided sign-flip p = {fb["p_one_sided"]:.3f}.

## Files and draft captions

- `fig_mcc_box` — State MCC per model; boxes over seeds, dots are seeds, grey lines
  join the same seed across models. Bands: 5-95% range of the MCC of a random
  rotation of the true states, either arbitrary (O(6)) or within the two state blocks.
- `fig_block_purity_box` — Best single-block linear R² per learned coordinate,
  averaged; 1 means every learned coordinate depends on one true block only.
- `fig_mcc_curves` — Validation MCC during sparse training from each seed's dense
  endpoint; thin lines are seeds, thick lines medians.
- `fig_graph_recovery` — Thresholded input masks with state rows/columns aligned
  by the calibration MCC matching; numbers count seeds with each edge.
- `fig_correlations_median_seed` — |Pearson| between true and matched learned
  coordinates for the seed with the median full-model MCC (seed {result["representative_seed"]}).
- `fig_forecast_quality` — Forecast error and whole-state information per model,
  checking prediction accuracy and retained state information.
- `fig_geco_multiplier` — Supplementary: GECO penalty weight over sparse training.
- `table_main.tex`, `table_paired.tex` — LaTeX (booktabs) tables; `per_run.csv`,
  `summary.json` — every number above.
- `group_mcc_contrasts.json` — Paired 95% group-bootstrap intervals for full minus
  baseline and baseline minus dense, separately for each training seed. Groups
  remain intact and calibration matchings remain fixed. These intervals are
  conditional on the trained models; the seed-level sign-flip tests are separate.
"""
        + (
            """- `fig_code_retention_box` — Whole-code retention R² of each true parameter, per
  model; dots show every training seed, including outliers, and the dashed line
  marks the repo's 0.8 retention criterion.
- `fig_code_correlations_median_seed` — |Pearson| between true parameters (grouped by
  pair) and matched code coordinates; complementary parameter graphs identify codes
  only up to mixing within each pair.
- In `fig_graph_recovery`, `table_main.tex` and `per_run.csv`
  (`masks/parameter_code_aligned_*`), the full model's parameter-graph columns are code
  coordinates matched to the true parameter pairs by validation linear pair R².
  The q labels denote matched positions, not pure physical parameters; the runner's own
  `masks/parameter_aligned_shd` aligns only the state rows.
- `table_codes.tex` — Code retention, pair purity, code MCC and predictive use.

Code summary (mean over seeds): """
            + "; ".join(
                f"{SHORT[arm]} min retention "
                f"{summary_codes[arm]['code_r2_min']:.3f}, pair purity "
                f"{summary_codes[arm]['code_pair_purity']:.3f}, shuffled/factual NMSE "
                f"{summary_codes[arm]['forecast/code_use_ratio']:.2f}"
                for arm in ARMS
            )
            + ".\n"
            if result["learned_codes"]
            else ""
        )
    )
    if result["projections"]:
        text += (
            "- `table_projection.tex` — Retention and forecast error of each acquired code "
            "before and after the label-free projection in the initialization pipeline.\n"
        )
    if result["adaptations"]:
        text += (
            "- `table_adaptation.tex` — The fixed six-code forecast-adaptation budget and "
            "retention/forecast accuracy before and after adaptation on frozen learned states. "
            "The adapted dense endpoints initialize the sparse comparison.\n"
        )
    (output / "README.md").write_text(text)


def main() -> None:
    """Write the paper outputs for one completed batch directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    result = analyse(Path(arguments.batch_dir), Path(arguments.output_dir))
    print(json.dumps({"figures": result["figures"], "paired": result["paired"]}, indent=2))


if __name__ == "__main__":
    main()
