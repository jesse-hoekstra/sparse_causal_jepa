"""Downstream (virtual tweezer) figures, tables and macros of the vision experiment, for the paper.

Read-only on the per-run JSON files written by ``diagnostics.vision_dimer.downstream`` (test
split). Outputs ``OUTPUT/figures/fig_downstream_*.pdf`` and ``OUTPUT/generated/downstream_*.tex``;
every macro is prefixed with ``Tw`` so it cannot clash with the paper's own.

usage: python -m diagnostics.vision_dimer.paper_downstream --downstream DIR --output-dir OUTPUT
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

NAMES = ("u1a", "v1a", "u1b", "v1b", "u2a", "v2a", "u2b", "v2b")
TEX = {n: f"${n[0]}_{{{n[1:]}}}$" for n in NAMES}
INK, INK2, RULE = "#0b0b0b", "#52514e", "#d9d8d3"
FULL, BASE = "#1baf7a", "#eb6834"  # Experiment 1 palette: full green, baseline orange
SEEDS = range(5)
ARMS = ("full", "baseline")


def style() -> None:
    """Recessive axes and grid, thin marks, text in ink."""
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.edgecolor": "#e4e3df",
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.solid_capstyle": "round",
        }
    )


def load(directory: Path) -> tuple[dict[str, Any], dict[tuple[int, str], dict[str, Any]]]:
    """Truth and per-run results."""
    truth = json.loads((directory / "truth.json").read_text())
    runs = {
        (s, a): json.loads((directory / f"seed{s}-{a}.json").read_text())
        for s in SEEDS
        for a in ARMS
    }
    return truth, runs


def partners(j: int) -> list[int]:
    """The held coordinate's three dimer partners."""
    return [i for i in range(4 * (j // 4), 4 * (j // 4) + 4) if i != j]


def others(j: int) -> list[int]:
    """The other dimer's four coordinates."""
    return [i for i in range(8) if i // 4 != j // 4]


def q1_scores(truth: dict[str, Any], run: dict[str, Any], j: int) -> dict[str, float]:
    """Within-dimer slope error (declared and per realized hold) and cross-dimer max |slope|."""
    t = np.array(truth["q1"][NAMES[j]]["slopes"])
    s = np.array(run["q1"][NAMES[j]]["slopes"])
    realized = s / s[j]
    return {
        "within": float(np.abs(s[partners(j)] - t[partners(j)]).mean()),
        "within_realized": float(np.abs(realized[partners(j)] - t[partners(j)]).mean()),
        "cross": float(np.abs(s[others(j)]).max()),
        "held_readback": float(s[j]),
    }


def q2_scores(
    truth: dict[str, Any], run: dict[str, Any], bead: str, label: str
) -> dict[str, float]:
    """Relative Frobenius error and cosine of the partner's 2x2 kick response."""
    m, t = np.array(run["q2"][bead][label]), np.array(truth["q2"][bead][label])
    return {
        "relerr": float(np.linalg.norm(m - t) / np.linalg.norm(t)),
        "cos": float((m * t).sum() / (np.linalg.norm(m) * np.linalg.norm(t))),
    }


def fig_headline(
    truth: dict[str, Any], runs: dict[tuple[int, str], dict[str, Any]], out: Path
) -> None:
    """Hold u2a at a: read-back of the held coordinate and three dimer partners at t+2."""
    panels = [
        (4, "held bead, $u_{2a}$ (read back)"),
        (6, "small bead, $u_{2b}$"),
        (7, "small bead, $v_{2b}$"),
        (5, "large bead, $v_{2a}$"),
    ]
    grid = np.array(truth["q1"]["u2a"]["grid"])
    fig, axes = cast(
        tuple[Any, Any], plt.subplots(1, 4, figsize=(7.0, 2.05), sharex=True, sharey=True)
    )
    for ax, (i, title) in zip(axes, panels, strict=True):
        m = np.array(truth["q1"]["u2a"]["means"])[:, i]
        ax.plot(
            grid,
            m - m[2],
            color=INK,
            lw=2.6,
            marker="o",
            ms=4.0,
            label="truth (simulator)",
            zorder=2,
        )
        for arm, colour in (("baseline", BASE), ("full", FULL)):
            for s in SEEDS:
                m = np.array(runs[(s, arm)]["q1"]["u2a"]["means"])[:, i]
                ax.plot(
                    grid,
                    m - m[2],
                    color=colour,
                    lw=0.9,
                    alpha=0.9,
                    label=arm if s == 0 else None,
                    zorder=3,
                )
        ax.set_title(title, color=INK)
        ax.set_xlabel("hold position $a$ of $u_{2a}$")
    axes[0].set_ylabel("change at $t+2$")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(x) for x in ("truth (simulator)", "full", "baseline")]
    fig.legend(
        [handles[k] for k in order],
        ["truth (simulator)", "full model (5 seeds)", "baseline (5 seeds)"],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.06),
    )
    fig.tight_layout()
    fig.savefig(out / "fig_downstream_headline.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_all_holds(
    truth: dict[str, Any], runs: dict[tuple[int, str], dict[str, Any]], out: Path
) -> None:
    """Per held coordinate: within-dimer slope errors and the cross-dimer max |slope|."""
    keys = [
        ("within", "within-dimer error"),
        ("within_realized", "per unit of realized hold"),
        ("cross", "effect on the other dimer"),
    ]
    fig, axes = cast(tuple[Any, Any], plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True))
    x = np.arange(8)
    for ax, (key, title) in zip(axes, keys, strict=True):
        for arm, colour, dx in (("full", FULL, -0.14), ("baseline", BASE, 0.14)):
            values = np.array(
                [[q1_scores(truth, runs[(s, arm)], j)[key] for j in range(8)] for s in SEEDS]
            )
            ax.scatter(
                np.repeat(x + dx, 5),
                values.T.ravel(),
                s=9,
                color=colour,
                linewidths=0,
                label="full model" if arm == "full" else "baseline",
            )
        ax.set_xticks(x, [TEX[n] for n in NAMES])
        ax.set_title(title, color=INK)
        ax.set_xlabel("held coordinate")
    axes[0].set_ylabel("slope error / |slope|")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08)
    )
    fig.tight_layout()
    fig.savefig(out / "fig_downstream_holds.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_kick(truth: dict[str, Any], runs: dict[tuple[int, str], dict[str, Any]], out: Path) -> None:
    """Hold 2a at rest; 2b displacement at t+2 per unit kick along camera x and y (medium frame)."""
    fig, axes = cast(
        tuple[Any, Any], plt.subplots(1, 2, figsize=(4.6, 2.35), sharex=True, sharey=True)
    )
    for k, ax in enumerate(axes):
        for arm, colour in (("baseline", BASE), ("full", FULL)):
            for s in SEEDS:
                m = np.array(runs[(s, arm)]["q2"]["2a"]["held"])[:, k]
                ax.annotate(
                    "",
                    xy=(m[0], m[1]),
                    xytext=(0, 0),
                    arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 0.9, "alpha": 0.85},
                )
        t = np.array(truth["q2"]["2a"]["held"])[:, k]
        ax.annotate(
            "",
            xy=(t[0], t[1]),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.8},
        )
        ax.axhline(0, color=INK2, lw=0.5)
        ax.axvline(0, color=INK2, lw=0.5)
        ax.set_title(f"unit kick along camera {'xy'[k]}", color=INK)
        ax.set_xlabel("change of $u_{2b}$")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("change of $v_{2b}$")
    everything = [np.array(runs[(s, a)]["q2"]["2a"]["held"]) for s in SEEDS for a in ARMS]
    everything.append(np.array(truth["q2"]["2a"]["held"]))
    lim = 1.15 * max(float(np.abs(m).max()) for m in everything)
    axes[0].set_xlim(-lim, lim)
    axes[0].set_ylim(-lim, lim)
    from matplotlib.lines import Line2D

    fig.legend(
        [Line2D([], [], color=c, lw=w) for c, w in ((INK, 1.8), (FULL, 1.0), (BASE, 1.0))],
        ["truth (simulator)", "full model (5 seeds)", "baseline (5 seeds)"],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.13),
    )
    fig.tight_layout()
    fig.savefig(out / "fig_downstream_kick.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def f3(x: float) -> str:
    """Three decimals with a typographic minus."""
    return f"{x:.3f}".replace("-", "$-$")


def f2(x: float) -> str:
    """Two decimals with a typographic minus."""
    return f"{x:.2f}".replace("-", "$-$")


def mean_sd(values: list[float], fmt: Callable[[float], str] = f2) -> str:
    """'mean (sd)'."""
    return f"{fmt(float(np.mean(values)))} ({fmt(float(np.std(values, ddof=1)))})"


def numbers(truth: dict[str, Any], runs: dict[tuple[int, str], dict[str, Any]], out: Path) -> None:
    """LaTeX macros and table rows."""
    lines: list[str] = []

    def macro(name: str, value: str) -> None:
        lines.append(f"\\newcommand{{\\Tw{name}}}{{{value}}}")

    t = truth["q1"]["u2a"]["slopes"]
    macro("TruthUb", f3(t[6]))
    macro("TruthVb", f3(t[7]))
    macro("TruthVa", f3(t[5]))
    macro("ObsUb", f3(truth["observational_slopes"][6][4]))
    grid = truth["q1"]["u2a"]["grid"]
    macro("HoldRange", f2(grid[-1] - grid[0]))
    macro("PlanTruth", f2(0.1 / t[6]))
    ub_full = float(np.mean([runs[(s, "full")]["q1"]["u2a"]["slopes"][6] for s in SEEDS]))
    ub_base = float(np.mean([runs[(s, "baseline")]["q1"]["u2a"]["slopes"][6] for s in SEEDS]))
    real_base = float(
        np.mean(
            [
                runs[(s, "baseline")]["q1"]["u2a"]["slopes"][6]
                / runs[(s, "baseline")]["q1"]["u2a"]["slopes"][4]
                for s in SEEDS
            ]
        )
    )
    macro("PlanFull", f2(0.1 / ub_full))
    macro("PlanBase", f2(0.1 / ub_base))
    macro("PlanBaseReal", f2(0.1 / real_base))
    rho = [runs[(s, "baseline")]["per_coordinate_spearman"][4] for s in SEEDS]
    macro("BaseRhoUa", f"{min(rho):.2f}--{max(rho):.2f}")
    rho = [runs[(s, "full")]["per_coordinate_spearman"][4] for s in SEEDS]
    macro("FullRhoUa", f"{min(rho):.2f}--{max(rho):.2f}")
    macro("Contexts", f"{truth['contexts']:,}".replace(",", "{,}"))
    for arm, tag in (("full", "Full"), ("baseline", "Base")):
        ub = [runs[(s, arm)]["q1"]["u2a"]["slopes"][6] for s in SEEDS]
        held = [runs[(s, arm)]["q1"]["u2a"]["slopes"][4] for s in SEEDS]
        ub_real = [u / h for u, h in zip(ub, held, strict=True)]
        macro(f"{tag}Ub", mean_sd(ub, f3))
        macro(f"{tag}UbReal", mean_sd(ub_real, f3))
        macro(f"{tag}Held", mean_sd(held))
        macro(f"{tag}UbSpread", f3(float(np.std(ub, ddof=1))))
        scores = [q1_scores(truth, runs[(s, arm)], j) for s in SEEDS for j in range(8)]
        macro(f"{tag}Within", f3(float(np.mean([x["within"] for x in scores]))))
        macro(f"{tag}WithinReal", f3(float(np.mean([x["within_realized"] for x in scores]))))
        macro(f"{tag}Cross", f3(float(np.max([x["cross"] for x in scores]))))
        macro(
            f"{tag}Nmse",
            f"{min(runs[(s, arm)]['stored_nmse'] for s in SEEDS):.4f}--"
            f"{max(runs[(s, arm)]['stored_nmse'] for s in SEEDS):.4f}",
        )
        kick = [
            q2_scores(truth, runs[(s, arm)], b, "held")
            for s in SEEDS
            for b in ("1a", "1b", "2a", "2b")
        ]
        macro(f"{tag}KickErr", f2(float(np.mean([x["relerr"] for x in kick]))))
        macro(f"{tag}KickCos", f2(float(np.mean([x["cos"] for x in kick]))))
        macro(f"{tag}KickCosMin", f2(float(np.min([x["cos"] for x in kick]))))
    rows: list[str] = []
    for s in SEEDS:
        f, b = runs[(s, "full")], runs[(s, "baseline")]
        fs, bs = f["q1"]["u2a"]["slopes"], b["q1"]["u2a"]["slopes"]
        rows.append(
            f"{s} & {f2(f['per_coordinate_spearman'][4])} & {f3(fs[6])} & {f3(fs[7])} & "
            f"{f3(fs[5])} & {f2(b['per_coordinate_spearman'][4])} & {f3(bs[4])} & {f3(bs[6])} & "
            f"{f3(bs[6] / bs[4])} & "
            f"{f3(bs[7])} & {f3(bs[5])} \\\\"
        )
    rows.append(
        f"\\midrule truth & -- & {f3(t[6])} & {f3(t[7])} & {f3(t[5])} & -- & 1.000 & {f3(t[6])} & "
        f"{f3(t[6])} & {f3(t[7])} & {f3(t[5])} \\\\"
    )
    (out / "downstream_table_headline.tex").write_text("\n".join(rows) + "\n")
    kick_rows: list[str] = []
    for bead in ("1a", "1b", "2a", "2b"):
        cells = [f"{bead[0]}{bead[1]} $\\to$ {bead[0]}{'b' if bead[1] == 'a' else 'a'}"]
        for label in ("held", "free"):
            for arm in ARMS:
                sc = [q2_scores(truth, runs[(s, arm)], bead, label) for s in SEEDS]
                cells.append(mean_sd([x["relerr"] for x in sc]))
                cells.append(mean_sd([x["cos"] for x in sc]))
        kick_rows.append(" & ".join(cells) + " \\\\")
    (out / "downstream_table_kick.tex").write_text("\n".join(kick_rows) + "\n")
    holds: list[str] = []
    for j in range(8):
        cells = [TEX[NAMES[j]]]
        for arm in ARMS:
            sc = [q1_scores(truth, runs[(s, arm)], j) for s in SEEDS]
            cells += [
                mean_sd([x["within"] for x in sc], f3),
                mean_sd([x["within_realized"] for x in sc], f3),
                f3(max(x["cross"] for x in sc)),
            ]
        holds.append(" & ".join(cells) + " \\\\")
    (out / "downstream_table_holds.tex").write_text("\n".join(holds) + "\n")
    checks: list[str] = []
    for a in ARMS:
        for s in SEEDS:
            r = runs[(s, a)]
            same = "yes" if r["assignment"] == r["stored_assignment"] else "no"
            checks.append(
                f"{a} & {s} & {r['stored_nmse']:.5f} & {r['nmse_first_window']:.5f} & "
                f"{r['nmse_two_step']:.5f} & {same} \\\\"
            )
    (out / "downstream_table_checks.tex").write_text("\n".join(checks) + "\n")
    (out / "downstream_numbers.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Build every figure and table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downstream", type=Path, required=True, help="downstream per-run JSONs")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.downstream
    figures, generated = args.output_dir / "figures", args.output_dir / "generated"
    figures.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    style()
    truth, runs = load(source)
    for run in runs.values():
        if run.get("split") != "test":
            raise ValueError("the vision document reports the test-split downstream analysis only")
    fig_headline(truth, runs, figures)
    fig_all_holds(truth, runs, figures)
    fig_kick(truth, runs, figures)
    numbers(truth, runs, generated)
    print(
        "written",
        sorted(
            p.name for p in (*figures.glob("fig_downstream_*"), *generated.glob("downstream_*"))
        ),
    )


if __name__ == "__main__":
    main()
