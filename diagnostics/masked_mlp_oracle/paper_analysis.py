"""Paper figures and tables for a multi-seed masked-MLP oracle batch.

Read-only: it authenticates completed run directories and reads their saved
validation endpoints, learning curves, matchings and final mask logits. It loads
no model for inference and never touches test data. Excluded from numerical
identity. Layout expected under ``--batch-dir``: ``{dense,baseline,full}-s{seed}/results``.

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
from diagnostics.masked_mlp_oracle import from_dense, run
from diagnostics.oracle_state_learning.data import read_json
from scjepa.eval.recovery import pearson_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

ARMS = ("dense", "baseline", "full")
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


def load_run(directory: Path, arm: str, seed: int) -> Run:
    """Authenticate one run directory and extract its endpoint and curves."""
    verify = run.verify_completion if arm == "dense" else from_dense.verify_completion
    completion = verify(directory, kind="train")
    if completion["condition"] != arm or completion["seed"] != seed:
        raise ValueError(f"{directory} does not hold {arm} seed {seed}")
    metrics = cast(dict[str, float], read_json(directory / "validation.json")["metrics"])
    startup = read_json(directory / "startup.json")["metrics"]
    rows = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
    curve = [
        (int(r["step"]), float(r["validation/state_mcc"]), float(r["validation/block_purity"]))
        for r in rows
        if "validation/state_mcc" in r
    ]
    penalty = [
        (int(r["step"]), math.log10(float(r["geco/penalty_weight"])))
        for r in rows
        if "geco/penalty_weight" in r and "loss/prediction" in r
    ]
    columns = [int(c) for c in read_json(directory / "matching.json")["online_columns"]]
    samples = cast(
        dict[str, Any],
        torch.load(directory / "validation.pt", map_location="cpu", weights_only=False),
    )["samples"]
    learned = cast(torch.Tensor, samples["online_states"]).reshape(-1, 6).double()
    truth = cast(torch.Tensor, samples["states"]).reshape(-1, 6).double()
    # pearson_matrix has true rows and learned columns; order columns by the matching.
    correlation = pearson_matrix(learned, truth)[:, columns].numpy()
    state = cast(
        dict[str, Any],
        torch.load(directory / "last.pt", map_location="cpu", weights_only=False),
    )["model"]
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
        graphs=_aligned_graphs(state, columns),
    )


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
            labels = (
                [f"z{k + 1}" for k in range(6)]
                if block == "state"
                else [f"θ{k + 1}" for k in range(6)]
            )
            axis.set_xticks(range(6), labels, fontsize=6.5)
            axis.set_yticks(range(6), [f"z{k + 1}'" for k in range(6)], fontsize=6.5)
            axis.tick_params(length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.set_title(f"{row_title}, {title}", loc="left", color=INK, fontsize=8)
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.6, label="Fraction of seeds with the edge")
    figure.suptitle(
        "Learned global graphs aligned to the true coordinates. Rows: next state; "
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
    """Forecast error and whole-code state information per arm."""
    figure, axes = cast(tuple[Any, Any], plt.subplots(1, 2, figsize=(6.4, 2.7)))
    _box_panel(axes[0], runs, "normalized_prediction_mse", seeds, "Normalized forecast MSE")
    _box_panel(axes[1], runs, "online_state_probe/r2", seeds, "Whole-code linear state R²")
    axes[0].set_title("Forecast error (two-step)", loc="left", color=INK)
    axes[1].set_title("State information in the code", loc="left", color=INK)
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
    summary: dict[str, dict[str, dict[str, float]]] = {
        arm: {key: describe([_metric(runs[arm][s], key) for s in seeds]) for key in keys}
        for arm in ARMS
    }
    rows: list[str] = []
    for arm in ARMS:
        cells = [SHORT[arm]]
        for key in keys:
            stats = summary[arm][key]
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
            r"learned (all edges present). Chance rows: random orthogonal re-charting of the "
            r"true states.}",
            r"\label{tab:masked-mlp-oracle}",
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
        r"\label{tab:masked-mlp-oracle-paired}",
        r"\end{table}",
        "",
    ]
    (output / "table_paired.tex").write_text("\n".join(lines))
    return {"summary": summary, "paired": tests}


def write_csv(runs: dict[str, dict[int, Run]], seeds: list[int], output: Path) -> None:
    """One row per run with its endpoint metrics and source directory."""
    keys = [
        "state_mcc",
        "block_purity",
        "normalized_prediction_mse",
        "online_state_probe/r2",
        "masks/state_edges",
        "masks/state_aligned_shd",
        "masks/parameter_edges",
        "masks/parameter_aligned_shd",
        "geco/relative_prediction",
    ]
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
    files = {name: _save(figure, output, name) for name, figure in figures.items()}
    tables = write_tables(runs, seeds, output, chance)
    write_csv(runs, seeds, output)
    result: dict[str, Any] = {
        "batch": str(batch),
        "seeds": seeds,
        "representative_seed": representative,
        "chance": chance,
        **tables,
        "figures": files,
        "per_run": {
            arm: {
                str(s): {
                    "state_mcc": _metric(runs[arm][s], "state_mcc"),
                    "block_purity": _metric(runs[arm][s], "block_purity"),
                    "start_state_mcc": _start_mcc(runs[arm][s]),
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
    text = (
        f"""# Masked-MLP oracle batch: paper figures and tables

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
- `fig_graph_recovery` — Thresholded learned state and parameter graphs aligned to
  the true coordinates by the MCC matching; numbers count seeds with each edge.
- `fig_correlations_median_seed` — |Pearson| between true and matched learned
  coordinates for the seed with the median full-model MCC (seed {result["representative_seed"]}).
- `fig_forecast_quality` — Forecast error and whole-code state information per model,
  showing comparable prediction and retained information.
- `fig_geco_multiplier` — Supplementary: GECO penalty weight over sparse training.
- `table_main.tex`, `table_paired.tex` — LaTeX (booktabs) tables; `per_run.csv`,
  `summary.json` — every number above.
"""
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
