"""Figures, tables and number macros of the vision experiment (test split), for the paper.

Read-only on stored evaluation outputs; nothing is refitted. Inputs in ``--evaluation`` (all
written by ``diagnostics/vision_dimer/final_evaluation.sh``):

- ``test-seed{s}-{arm}.json``: ``train evaluate --split test`` of the fixed-budget endpoints
  (the reported numbers);
- ``figdata-seed{s}-{arm}.json``: correlation matrices and aligned graphs
  (``analysis figure-data``);
- ``summary-test.json`` and ``summary-validation.json``: rotation references, paired contrasts;
- ``chart-check.json`` (``analysis charts``) and ``code-invariance.json``
  (``analysis code-invariance``);
- ``data-manifest-with-test.json``: the cohort manifest including the test split.

``--curves`` is the validation learning-curve file written by ``analysis curves``.

Outputs: ``OUTPUT/figures/*.pdf`` (one file per figure, plus PNG previews) and
``OUTPUT/generated/*.tex`` (table bodies and ``numbers.tex`` macros). Style and conventions
follow Experiment 1's ``diagnostics/masked_mlp_oracle/paper_analysis.py``: Pearson MCC with
a calibration-fitted assignment, linear block purity, quartile boxes with 1.5-IQR whiskers,
rotation references as 5th-95th percentile bars.
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment  # pyright: ignore[reportUnknownVariableType]

ARMS = ("dense", "baseline", "full")
SEEDS = (0, 1, 2, 3, 4)
LABELS = {"dense": "Dense", "baseline": "State/action\nsparsity", "full": "+ parameter\nsparsity"}
SHORT = {"dense": "Dense", "baseline": "Baseline", "full": "Full"}
COLORS = {"dense": "#2a78d6", "baseline": "#eb6834", "full": "#1baf7a"}  # Experiment 1 palette
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"
STATES = ("u_{1a}", "v_{1a}", "u_{1b}", "v_{1b}", "u_{2a}", "v_{2a}", "u_{2b}", "v_{2b}")
ACTIONS = ("B", "E_{1x}", "E_{1y}", "E_{2x}", "E_{2y}")
DRAGS = ("c_{au}", "c_{av}", "c_{bu}", "c_{bv}")

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


def true_graphs() -> dict[str, np.ndarray]:
    """True lag-aggregated supports (rows: next state), as ``dgp.graphs()``."""
    index = np.arange(8)
    dimer, slot, axis = index // 4, (index // 2) % 2, index % 2
    state = dimer[:, None] == dimer[None, :]
    action = np.concatenate(
        (np.ones((8, 1), bool), dimer[:, None] == (np.arange(4) // 2)[None, :]), 1
    )
    parameter = (2 * slot + axis)[:, None] == np.arange(4)[None, :]
    return {"state": state, "action": action, "parameter": parameter}


def save(figure: Figure, out: Path, name: str) -> None:
    """One PDF per figure (vector), plus a PNG preview."""
    for suffix in ("pdf", "png"):
        cast(Any, figure).savefig(
            out / f"{name}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white"
        )
    plt.close(figure)


def box_panel(target: Axes, values: dict[str, list[float]], ylabel: str) -> None:
    """Quartile boxes, 1.5-IQR whiskers, points joining matched training seeds (Experiment 1)."""
    axis: Any = target
    positions = list(range(len(ARMS)))
    offsets = np.linspace(-0.12, 0.12, len(SEEDS))
    for k in range(len(SEEDS)):
        axis.plot(
            [p + offsets[k] for p in positions],
            [values[a][k] for a in ARMS],
            color=GRID,
            lw=0.8,
            zorder=1,
        )
    box = axis.boxplot(
        [values[a] for a in ARMS],
        positions=positions,
        widths=0.5,
        whis=1.5,
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
    axis.set_xticks(positions, [LABELS[a] for a in ARMS])
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color=GRID, lw=0.6)
    axis.set_axisbelow(True)


def chance_reference(
    target: Axes, x: float, label_x: float, level: dict[str, float], text: str
) -> None:
    """A dashed mean line plus a 5-95% range bar in the right margin, labelled (Experiment 1)."""
    axis: Any = target
    axis.axhline(level["mean"], color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=0)
    axis.plot([x, x], [level["p05"], level["p95"]], color=INK2, lw=1.6)
    axis.plot([x], [level["mean"]], marker="_", color=INK2, ms=9, mew=1.6)
    axis.text(label_x, level["mean"], text, color=INK2, fontsize=6.8, va="center")


def figure_mcc(
    test: dict[tuple[int, str], dict[str, Any]], reference: dict[str, Any], even: float, out: Path
) -> float:
    """Test state MCC per arm with seed pairing and rotation references; returns the y-axis floor.

    ``even`` is the MCC of an even within-dimer mixing (each bead pair replaced by its sum and
    difference along each axis), drawn because the baselines mix this way (``analysis charts``).
    """
    figure, axis = cast(tuple[Any, Any], plt.subplots(figsize=(4.4, 3.0)))
    unrestricted, within = reference["unrestricted_pearson"], reference["within_block_pearson"]
    chance_reference(
        axis, 2.5, 2.72, unrestricted, "Unrestricted\nrandom rotations\n5th-95th percentiles"
    )
    chance_reference(
        axis, 2.62, 2.72, within, "Within-block\nrandom rotations\n5th-95th percentiles"
    )
    axis.axhline(even, color=INK2, lw=0.9, ls=(0, (1, 1.5)), zorder=0)
    axis.text(
        -0.45, even + 0.006, "Even within-dimer mixing", color=INK2, fontsize=6.5, va="bottom"
    )
    values = {a: [float(test[(s, a)]["mcc"]) for s in SEEDS] for a in ARMS}
    box_panel(axis, values, r"State MCC $(\hat z^t,z^t)$ (test)")
    lowest = min(min(min(v) for v in values.values()), unrestricted["p05"])
    floor = float(np.floor((lowest - 0.02) * 20) / 20)
    axis.set_ylim(floor, 1.01)
    axis.set_xlim(-0.5, 2.68)
    axis.spines["bottom"].set_bounds(-0.5, 2.4)
    axis.set_title(f"State recovery, {len(SEEDS)} training seeds per model", loc="left", color=INK)
    figure.text(
        0.125,
        -0.06,
        "Boxes: middle 50% and median; whiskers: 1.5 IQR rule.\n"
        "Rotation bars: middle 90%; dashed lines: means.\n"
        "Dotted line: even within-dimer mixing (sum and difference of each bead pair).\n"
        "Values outside the rotation bars are possible.",
        fontsize=6.2,
        color=INK2,
        va="top",
    )
    save(figure, out, "vision_fig_mcc_box")
    return floor


def even_mixing_mcc(manifest: dict[str, Any], split: str) -> float:
    """Pearson MCC of the chart that replaces each bead pair by its sum and difference.

    Computed exactly from the split's state covariance in the data manifest (rows: chart
    coordinates), against the bead coordinates.
    """
    record = manifest["splits"][split]
    sd = np.asarray(record["state_std"], float)
    cov = np.asarray(record["correlation"], float) * np.outer(sd, sd)
    chart = np.zeros((8, 8))
    for dimer in range(2):
        for axis in range(2):
            a, b = 4 * dimer + axis, 4 * dimer + 2 + axis
            chart[4 * dimer + axis, [a, b]] = (1.0, 1.0)
            chart[4 * dimer + 2 + axis, [a, b]] = (1.0, -1.0)
    corr = np.abs(chart @ cov) / np.sqrt(np.diag(chart @ cov @ chart.T))[:, None] / sd[None, :]
    rows, cols = cast(tuple[Any, Any], linear_sum_assignment(-corr))
    return float(corr[rows, cols].mean())


def figure_curves(
    curves: dict[str, list[dict[str, Any]]],
    reference: dict[str, Any],
    even: float,
    floor: float,
    out: Path,
) -> None:
    """Validation MCC during sparse training, per seed and median, with the even-mixing line."""
    figure, axis = cast(tuple[Any, Any], plt.subplots(figsize=(5.2, 3.0)))
    block = reference["within_block_pearson"]
    axis.axhspan(block["p05"], block["p95"], color=BAND, alpha=0.45, lw=0, zorder=0)
    axis.axhline(block["mean"], color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=0)
    last = 0
    for arm in ("baseline", "full"):
        rows = [curves[f"seed{s}/{arm}"] for s in SEEDS]
        step_sets: list[set[int]] = [{int(r["step"]) for r in run} for run in rows]
        steps = sorted(step_sets[0].intersection(*step_sets[1:]))
        stacked = np.array([[{r["step"]: r["mcc"] for r in run}[t] for t in steps] for run in rows])
        last = max(last, steps[-1])
        x = np.asarray(steps) / 1000
        for row in stacked:
            axis.plot(x, row, color=COLORS[arm], lw=0.7, alpha=0.35)
        label = f"{SHORT[arm]}: {LABELS[arm].replace(chr(10), ' ')} (median)"
        axis.plot(x, np.median(stacked, axis=0), color=COLORS[arm], lw=2.0, label=label)
        axis.text(
            x[-1] + 4,
            float(np.median(stacked[:, -1])),
            SHORT[arm],
            color=INK,
            fontsize=7,
            va="center",
        )
    axis.text(4, block["mean"] + 0.012, "within-block chance", color=INK2, fontsize=6.8)
    axis.axhline(even, color=INK2, lw=0.9, ls=(0, (1, 1.5)), zorder=0)
    axis.text(
        last / 1000,
        even + 0.006,
        "even within-dimer mixing",
        color=INK2,
        fontsize=6.8,
        ha="right",
        va="bottom",
    )
    axis.set_xlim(0, last / 1000 * 1.1)
    axis.set_ylim(floor, 1.01)
    axis.set_xlabel("Sparse updates (thousands), starting from each seed's dense endpoint")
    axis.set_ylabel(r"State MCC $(\hat z^t,z^t)$ (validation)")
    axis.grid(axis="y", color=GRID, lw=0.6)
    axis.legend(loc="lower right", frameon=False)
    axis.set_title("Learning curves (thin: individual seeds)", loc="left", color=INK)
    save(figure, out, "vision_fig_curves")


def figure_correlations(
    figdata: dict[tuple[int, str], dict[str, Any]],
    test: dict[tuple[int, str], dict[str, Any]],
    seed: int,
    out: Path,
) -> None:
    """Matched |Pearson| matrices (true rows, matched learned columns) for one seed of every arm."""
    figure, axes = cast(
        tuple[Any, Any], plt.subplots(1, 3, figsize=(7.0, 2.6), constrained_layout=True)
    )
    image = None
    for axis, arm in zip(axes, ARMS, strict=True):
        matrix = np.asarray(figdata[(seed, arm)]["pearson"])
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        axis.axhline(3.5, color=INK2, lw=0.6)
        axis.axvline(3.5, color=INK2, lw=0.6)
        axis.set_xticks(range(8), [rf"$\hat z_{{{k + 1}}}^t$" for k in range(8)], fontsize=6.2)
        axis.set_yticks(range(8), [rf"${n}^t$" for n in STATES], fontsize=6.2)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(f"{SHORT[arm]}  (MCC {test[(seed, arm)]['mcc']:.3f})", loc="left", color=INK)
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.8, label="|Pearson correlation|")
    figure.suptitle(
        rf"Learned $\hat z^t$ vs true $z^t$, seed {seed} (test; calibration-matched)",
        x=0.02,
        ha="left",
        color=INK,
        fontsize=8.5,
    )
    save(figure, out, "vision_fig_correlations")


def figure_graphs(figdata: dict[tuple[int, str], dict[str, Any]], out: Path) -> None:
    """Edge-recovery counts of the aligned learned graphs over the five seeds (Experiment 1)."""
    truth = true_graphs()
    blocks = (
        ("state", "State graph", STATES),
        ("action", "Action graph", ACTIONS),
        ("parameter", "Drag graph", DRAGS),
    )
    figure: Any = cast(Any, plt).figure(figsize=(7.2, 7.4), constrained_layout=True)
    grid = figure.add_gridspec(3, 3, width_ratios=(1, 1, 1), height_ratios=(1, 1, 1))
    image = None
    for row, (key, row_title, columns) in enumerate(blocks):
        for col, source in enumerate(("truth", "baseline", "full")):
            axis = figure.add_subplot(grid[row, col])
            expected = truth[key]
            if source == "truth":
                matrix = expected.astype(float)
            else:
                matrix = np.mean(
                    [np.asarray(figdata[(s, source)]["graphs"][key], float) for s in SEEDS], axis=0
                )
            image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if expected[i, j]:
                        axis.add_patch(
                            Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=0.9)
                        )
                    if source != "truth":
                        axis.text(
                            j,
                            i,
                            f"{matrix[i, j] * len(SEEDS):.0f}",
                            ha="center",
                            va="center",
                            fontsize=5.6,
                            color="white" if matrix[i, j] > 0.6 else INK2,
                        )
            ticks = [rf"${c}^t$" if key == "state" else rf"${c}$" for c in columns]
            axis.set_xticks(
                range(len(columns)), ticks, fontsize=6.0, rotation=90 if key == "state" else 0
            )
            axis.set_yticks(range(8), [rf"${n}^{{t+1}}$" for n in STATES], fontsize=6.0)
            axis.tick_params(length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
            title = "true" if source == "truth" else source
            axis.set_title(f"{row_title}, {title}", loc="left", color=INK, fontsize=8)
    if image is not None:
        figure.colorbar(image, ax=figure.axes, shrink=0.45, label="Fraction of seeds with the edge")
    figure.suptitle(
        "Aligned learned graphs (calibration-fitted state and drag assignments).\n"
        f"Rows: next state; outlined: true edges; numbers: seeds (of {len(SEEDS)}) with the edge",
        x=0.02,
        ha="left",
        color=INK,
        fontsize=8,
    )
    save(figure, out, "fig_graph_recovery")


def figure_code_retention(test: dict[tuple[int, str], dict[str, Any]], out: Path) -> None:
    """Whole-code retention R^2 per drag coefficient, every seed visible (Experiment 1)."""
    figure, axis = cast(tuple[Any, Any], plt.subplots(figsize=(6.4, 2.8)))
    width = 0.24
    floor = 1.0
    for offset, arm in zip((-width, 0.0, width), ARMS, strict=True):
        values = [[float(test[(s, arm)]["parameter_r2"][k]) for s in SEEDS] for k in range(4)]
        floor = min(floor, min(min(v) for v in values))
        positions = np.arange(4) + offset
        box = axis.boxplot(
            values,
            positions=positions,
            widths=width * 0.85,
            whis=1.5,
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
        jitter = np.linspace(-0.07, 0.07, len(SEEDS))
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
    axis.set_xticks(range(4), [rf"${d}$" for d in DRAGS])
    axis.set_ylabel(r"Retention $R^2$ from $\hat\theta$ (test)")
    axis.axhline(0.95, color=INK2, lw=0.7, ls=(0, (3, 2)))
    axis.text(-0.42, 0.9506, "retention criterion 0.95", color=INK2, fontsize=6.8, va="bottom")
    axis.set_ylim(min(0.94, float(np.floor((floor - 0.005) * 100) / 100)), 1.0)
    axis.grid(axis="y", color=GRID, lw=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, ncol=3)
    axis.set_title(
        r"Drag retention from one context's $\hat\theta$ (dots: training seeds)",
        loc="left",
        color=INK,
    )
    save(figure, out, "fig_code_retention_box")


def f5(x: float) -> str:
    """Experiment 1 table format: five decimals without the leading zero."""
    text = f"{x:.5f}"
    return text[1:] if text.startswith("0.") else text


def f3(x: float) -> str:
    """Three decimals without the leading zero."""
    text = f"{x:.3f}"
    return text[1:] if text.startswith("0.") else text


def tables(
    test: dict[tuple[int, str], dict[str, Any]],
    summary: dict[str, Any],
    reference: dict[str, Any],
    out: Path,
) -> dict[str, Any]:
    """Main table, per-seed table and number macros; returns the macro values."""

    def mean(arm: str, key: str) -> float:
        return statistics.fmean(float(test[(s, arm)][key]) for s in SEEDS)

    def sd(arm: str, key: str) -> float:
        return statistics.stdev(float(test[(s, arm)][key]) for s in SEEDS)

    rows: list[str] = []
    for arm in ARMS:
        mcc = rf"${f5(mean(arm, 'mcc'))}\pm{f5(sd(arm, 'mcc'))}$"
        if arm == "full":
            mcc = rf"$\mathbf{{{f5(mean(arm, 'mcc'))}}}\pm{f5(sd(arm, 'mcc'))}$"
        shd = {k: f"{mean(arm, k + '_shd'):.1f}" for k in ("state", "action", "parameter")}
        if arm == "dense":
            shd = {k: "open" for k in shd}
        if arm == "baseline":
            shd["parameter"] = "open"
        rows.append(
            f"{SHORT[arm]} & {mcc} & {f5(mean(arm, 'block_purity_mean'))} & "
            f"{f5(mean(arm, 'nmse'))} & "
            f"{shd['state']} & {shd['action']} & {shd['parameter']}\\\\"
        )
    (out / "table_results.tex").write_text("\n".join(rows) + "\n")

    per_seed: list[str] = []
    for s in SEEDS:
        for arm in ARMS:
            r = test[(s, arm)]
            graph = (
                "open"
                if arm == "dense"
                else f"{r['state_shd']}/{r['action_shd']}/"
                + ("open" if arm == "baseline" else f"{r['parameter_shd']}")
            )
            per_seed.append(
                f"{s if arm == 'dense' else ''} & {SHORT[arm]} & {f3(r['mcc'])} & "
                f"{f3(r['spearman_mcc'])} & {f3(r['block_purity_mean'])} & "
                f"{f3(r['block_purity_knn_mean'])} & {f5(r['nmse'])} & "
                f"{f3(min(r['state_r2_cubic']))} & {f3(min(r['parameter_r2']))} & "
                f"{r['code_shuffle_ratio']:.1f} & {graph}\\\\"
            )
        if s != SEEDS[-1]:
            per_seed.append(r"\addlinespace[2pt]")
    (out / "table_per_seed.tex").write_text("\n".join(per_seed) + "\n")

    gaps = [float(test[(s, "full")]["mcc"]) - float(test[(s, "baseline")]["mcc"]) for s in SEEDS]
    exact = [
        s
        for s in SEEDS
        if all(int(test[(s, "full")][f"{k}_shd"]) == 0 for k in ("state", "action", "parameter"))
    ]
    contrast = summary["contrasts"]["mcc"]
    values: dict[str, Any] = {
        "VisDenseMCC": f5(mean("dense", "mcc")),
        "VisBaseMCC": f5(mean("baseline", "mcc")),
        "VisFullMCC": f5(mean("full", "mcc")),
        "VisFullMCCsd": f5(sd("full", "mcc")),
        "VisFullMCCmin": f5(min(float(test[(s, "full")]["mcc"]) for s in SEEDS)),
        "VisGap": f5(statistics.fmean(gaps)),
        "VisGapSD": f5(statistics.stdev(gaps)),
        "VisGapMin": f5(min(gaps)),
        "VisGapMax": f5(max(gaps)),
        "VisSignP": f"{contrast['sign_flip_p_one_sided']:.5f}",
        "VisWithinMean": f3(reference["within_block_pearson"]["mean"]),
        "VisWithinLow": f3(reference["within_block_pearson"]["p05"]),
        "VisWithinHigh": f3(reference["within_block_pearson"]["p95"]),
        "VisAnyMean": f3(reference["unrestricted_pearson"]["mean"]),
        "VisBasePurity": f5(mean("baseline", "block_purity_mean")),
        "VisFullPurity": f5(mean("full", "block_purity_mean")),
        "VisBaseKnn": f5(mean("baseline", "block_purity_knn_mean")),
        "VisDenseNMSE": f5(mean("dense", "nmse")),
        "VisBaseNMSE": f5(mean("baseline", "nmse")),
        "VisFullNMSE": f5(mean("full", "nmse")),
        "VisFullShuffle": f"{mean('full', 'code_shuffle_ratio'):.2f}",
        "VisBaseShuffle": f"{mean('baseline', 'code_shuffle_ratio'):.2f}",
        "VisFullParamRmin": f5(
            min(min(float(v) for v in test[(s, "full")]["parameter_r2"]) for s in SEEDS)
        ),
        "VisFullParamRmean": f5(
            statistics.fmean(float(v) for s in SEEDS for v in test[(s, "full")]["parameter_r2"])
        ),
        "VisStateRmin": f5(
            min(min(float(v) for v in test[(s, a)]["state_r2_cubic"]) for s in SEEDS for a in ARMS)
        ),
        "VisFullSpearman": f5(mean("full", "spearman_mcc")),
        "VisBaseSpearman": f5(mean("baseline", "spearman_mcc")),
        "VisExactSeeds": ", ".join(str(s) for s in exact),
        "VisExactCount": str(len(exact)),
    }
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in values.items()]
    (out / "numbers.tex").write_text(
        "% Generated by diagnostics.vision_dimer.paper_figures from the test evaluation.\n"
        + "\n".join(lines)
        + "\n"
    )
    return values


def main() -> None:
    """Build every figure and table from the stored evaluation outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True, help="final test evaluation")
    parser.add_argument("--curves", type=Path, required=True, help="validation learning curves")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluation = args.evaluation
    figures, generated = args.output_dir / "figures", args.output_dir / "generated"
    figures.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    test = {
        (s, a): json.loads((evaluation / f"test-seed{s}-{a}.json").read_text())
        for s in SEEDS
        for a in ARMS
    }
    for (s, a), r in test.items():
        if r["scored_split"] != "test":
            raise ValueError(f"seed {s} {a} is not a test evaluation")
    figdata = {
        (s, a): json.loads((evaluation / f"figdata-seed{s}-{a}.json").read_text())
        for s in SEEDS
        for a in ARMS
    }
    summary_test = json.loads((evaluation / "summary-test.json").read_text())
    summary_validation = json.loads((evaluation / "summary-validation.json").read_text())
    curves = json.loads(args.curves.read_text())
    chart = json.loads((evaluation / "chart-check.json").read_text())
    even = float(chart["references"]["medium_centre_separation_chart"])
    fits = [
        max(
            chart["runs"][f"seed{s}/baseline"][k]["mcc"]
            for k in ("medium_centre_separation", "camera_centre_separation")
        )
        for s in SEEDS
    ]
    invariance = json.loads((evaluation / "code-invariance.json").read_text())
    within = statistics.fmean(invariance[f"seed{s}/full"]["within_share"] for s in SEEDS)
    worst = max(invariance[f"seed{s}/full"]["worst_direction_share"] for s in SEEDS)
    floor = figure_mcc(test, summary_test["rotation_reference"], even, figures)
    manifest = json.loads((evaluation / "data-manifest-with-test.json").read_text())
    if abs(even_mixing_mcc(manifest, "test") - even) > 0.005:
        raise ValueError(
            "even-mixing MCC from the manifest covariance disagrees with the chart diagnostic"
        )
    even_validation = even_mixing_mcc(manifest, "validation")
    figure_curves(curves, summary_validation["rotation_reference"], even_validation, floor, figures)
    ranked = sorted(SEEDS, key=lambda s: float(test[(s, "full")]["mcc"]))
    median_seed = ranked[len(ranked) // 2]
    figure_correlations(figdata, test, median_seed, figures)
    figure_graphs(figdata, figures)
    figure_code_retention(test, figures)
    values = tables(test, summary_test, summary_test["rotation_reference"], generated)
    (generated / "numbers.tex").write_text(
        (generated / "numbers.tex").read_text()
        + f"\\newcommand{{\\VisMedianSeed}}{{{median_seed}}}\n"
        + f"\\newcommand{{\\VisEvenMix}}{{{f3(even)}}}\n"
        + f"\\newcommand{{\\VisEvenFitLow}}{{{f3(min(fits))}}}\n"
        + f"\\newcommand{{\\VisEvenFitHigh}}{{{f3(max(fits))}}}\n"
        + f"\\newcommand{{\\VisCodeWithin}}{{{100 * within:.2f}\\%}}\n"
        + f"\\newcommand{{\\VisCodeWorst}}{{{100 * worst:.2f}\\%}}\n"
    )
    print(json.dumps({"median_seed": median_seed, **values}, indent=1))


if __name__ == "__main__":
    main()
