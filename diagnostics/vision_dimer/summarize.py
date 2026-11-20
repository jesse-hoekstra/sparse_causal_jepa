"""Paper-level summary of vision-dimer evaluations: per-run criteria, arm means, contrasts.

Reads evaluation JSON files (``final.json`` of a run or ``train evaluate`` output),
scored on validation during development and on test once at the end. Criteria are
the user-approved Experiment-1-level thresholds, with the theorem-compatible
(elementwise) state scores reported next to the Pearson/linear ones:

- both arms: every state cubic R2 >= .99 and every parameter R2 >= .95;
- full: Spearman MCC >= .98 and every monotone coordinatewise recovery >= .95
  (Pearson MCC >= .98 reported separately);
- baseline: kNN block purity >= .95 (linear purity reported) and Spearman MCC below
  the paired full arm.

usage: python -m diagnostics.vision_dimer.summarize --run SEED:ARM:PATH ... --output OUT
       [--reference DATA_DIR --split validation|test]
"""

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any, cast

import torch
from scipy.optimize import linear_sum_assignment  # pyright: ignore[reportUnknownVariableType]
from torch import Tensor

from diagnostics.vision_dimer import data as vdata
from diagnostics.vision_dimer import dgp

FIELDS = (
    "mcc", "spearman_mcc", "block_purity_mean", "block_purity_knn_mean", "state_r2_min",
    "parameter_r2_min", "coordinatewise_r2_monotone_min", "nmse", "code_shuffle_ratio",
    "state_edges", "action_edges", "parameter_edges", "state_shd", "action_shd", "parameter_shd",
)  # fmt: skip


def criteria(run: dict[str, Any], arm: str) -> dict[str, bool]:
    """Per-run checks of the declared thresholds (the baseline contrast is paired)."""
    checks = {
        "state_retention": min(run["state_r2_cubic"]) >= 0.99,
        "parameter_retention": min(run["parameter_r2"]) >= 0.95,
    }
    if arm == "full":
        checks["spearman_mcc"] = run["spearman_mcc"] >= 0.98
        checks["monotone_recovery"] = min(run["coordinatewise_r2_monotone"]) >= 0.95
        checks["pearson_mcc"] = run["mcc"] >= 0.98  # reported, not the elementwise claim
    else:
        checks["knn_purity"] = run["block_purity_knn_mean"] >= 0.95
        checks["linear_purity"] = run["block_purity_mean"] >= 0.95  # reported
    return checks


def sign_flip_p(differences: Sequence[float]) -> float:
    """Exact one-sided paired sign-flip p-value for a positive mean difference."""
    observed = sum(differences)
    flips = [
        sum(sign * value for sign, value in zip(signs, differences, strict=True))
        for signs in product((1, -1), repeat=len(differences))
    ]
    return sum(total >= observed - 1e-12 for total in flips) / len(flips)


def _mean_sd(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else math.nan,
    }


def summarize(runs: dict[tuple[int, str], dict[str, Any]]) -> dict[str, Any]:
    """Per-run fields and checks, arm means, and paired full-minus-baseline contrasts."""
    table = {
        f"seed{seed}/{arm}": {
            **{field: run.get(field) for field in FIELDS},
            "per_coordinate_spearman": run.get("per_coordinate_spearman"),
            "parameter_r2": run["parameter_r2"],
            "criteria": criteria(run, arm),
        }
        for (seed, arm), run in sorted(runs.items())
    }
    arms: dict[str, Any] = {}
    for arm in ("baseline", "full"):
        chosen = [run for (_, a), run in sorted(runs.items()) if a == arm]
        if chosen:
            arms[arm] = {
                field: _mean_sd([float(run[field]) for run in chosen])
                for field in FIELDS
                if all(run.get(field) is not None for run in chosen)
            }
    contrasts: dict[str, Any] = {}
    seeds = sorted(
        {seed for seed, _ in runs if (seed, "full") in runs and (seed, "baseline") in runs}
    )
    for field in ("spearman_mcc", "mcc"):
        differences = [runs[(s, "full")][field] - runs[(s, "baseline")][field] for s in seeds]
        if differences:
            contrasts[field] = {
                "seeds": seeds,
                "differences": differences,
                "positive": sum(d > 0 for d in differences),
                **_mean_sd(differences),
                "sign_flip_p_one_sided": sign_flip_p(differences),
            }
    for seed in seeds:  # paired baseline condition: below its own full arm
        checks = table[f"seed{seed}/baseline"]["criteria"]
        assert isinstance(checks, dict)
        checks["below_full_spearman"] = (
            runs[(seed, "baseline")]["spearman_mcc"] < runs[(seed, "full")]["spearman_mcc"]
        )
    return {"runs": table, "arms": arms, "contrasts": contrasts}


def rotation_reference(
    states: torch.Tensor, draws: int = 2000, seed: int = 0
) -> dict[str, dict[str, float]]:
    """MCC (Pearson and Spearman) of random within-block / unrestricted rotations."""
    z = states.reshape(-1, dgp.STATE_DIM).double()
    z = (z - z.mean(0)) / z.std(0)
    rank_z = z.argsort(0).argsort(0).double()
    generator = torch.Generator().manual_seed(seed)

    def orthogonal(size: int) -> torch.Tensor:
        gaussian = torch.randn(size, size, generator=generator, dtype=torch.float64)
        q, r = cast(tuple[Tensor, Tensor], torch.linalg.qr(gaussian))  # pyright: ignore[reportUnknownMemberType]
        return q * torch.sign(torch.diagonal(r))

    def mcc(x: torch.Tensor, truth: torch.Tensor) -> float:
        width = dgp.STATE_DIM
        scores = torch.corrcoef(torch.cat((x, truth), 1).T)[:width, width:].abs().numpy()
        rows, columns = linear_sum_assignment(-scores)
        return float(scores[rows, columns].mean())

    result: dict[str, dict[str, float]] = {}
    for name in ("within_block", "unrestricted"):
        pearson: list[float] = []
        spearman: list[float] = []
        for _ in range(draws):
            q = (
                cast(Tensor, torch.block_diag(*[orthogonal(len(b)) for b in dgp.BLOCKS]))  # pyright: ignore[reportUnknownMemberType]
                if name == "within_block"
                else orthogonal(dgp.STATE_DIM)
            )
            mixed: Tensor = z @ q.T
            pearson.append(mcc(mixed, z))
            spearman.append(mcc(mixed.argsort(0).argsort(0).double(), rank_z))
        for label, values in (("pearson", pearson), ("spearman", spearman)):
            tensor = torch.tensor(values)
            result[f"{name}_{label}"] = {
                "mean": float(tensor.mean()),
                "p05": float(tensor.quantile(0.05)),
                "p95": float(tensor.quantile(0.95)),
            }
    return result


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="SEED:ARM:PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="data directory for rotation references")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    runs: dict[tuple[int, str], dict[str, Any]] = {}
    sources: dict[str, str] = {}
    for item in args.run:
        seed, arm, path = item.split(":", 2)
        if arm not in ("baseline", "full") or (int(seed), arm) in runs:
            raise ValueError("each SEED:ARM appears once, with ARM baseline or full")
        runs[(int(seed), arm)] = json.loads(Path(path).read_text())
        sources[f"seed{seed}/{arm}"] = path
    summary = summarize(runs)
    summary["sources"] = sources
    if args.reference is not None:
        labels = vdata.load_labels(args.reference, args.split)
        frames = vdata.load_config(args.reference).frames
        summary["rotation_reference"] = rotation_reference(labels["states"][:, :, :frames])
        summary["reference_split"] = args.split
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"arms": summary["arms"], "contrasts": summary["contrasts"]}, indent=2))


if __name__ == "__main__":
    main()
