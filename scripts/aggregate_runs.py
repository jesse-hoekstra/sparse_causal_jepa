"""Aggregate metrics.json across seeded runs: mean ± SD + box-plot statistics.

Usage:
    python scripts/aggregate_runs.py 'outputs/bounce_example_rung1_seed*/main'
    python scripts/aggregate_runs.py dir1 dir2 dir3 ...

Each argument is a glob or path to a run dir containing metrics.json (written
by scripts/eval_identifiability.py). Prints per-metric mean, sample SD, and the
five-number summary (min/q1/median/q3/max — the statistics behind Baumgartner
Fig. 3's box plots), and writes aggregate.json next to the first run.
"""

import argparse
import json
import statistics
from glob import glob
from pathlib import Path


def five_number(values: list[float]) -> dict[str, float]:
    """Min, quartiles, max (matches box-plot whiskers/box of the source paper)."""
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "min": ordered[0],
        "q1": quartiles[0],
        "median": quartiles[1],
        "q3": quartiles[2],
        "max": ordered[-1],
    }


def main() -> None:
    """Collect metrics.json files, print and save the aggregate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run dirs or globs containing metrics.json")
    args = parser.parse_args()

    run_dirs: list[Path] = []
    for pattern in args.runs:
        run_dirs.extend(Path(p) for p in sorted(glob(pattern)))
    records: list[dict[str, float]] = []
    for run_dir in run_dirs:
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            records.append(json.loads(metrics_file.read_text()))
        else:
            print(f"  (skipping {run_dir}: no metrics.json)")
    if len(records) < 2:
        raise SystemExit(f"need >= 2 runs to aggregate, found {len(records)}")

    seeds = [r.get("seed") for r in records]
    print(f"aggregating {len(records)} runs (seeds {seeds}):\n")
    skip = {"seed", "step", "eval_seed_offset", "num_samples", "mass_mcc_linear"}
    aggregate: dict[str, dict[str, float]] = {}
    header = (
        f"{'metric':>14} | {'mean':>8} {'sd':>8} | "
        f"{'min':>7} {'q1':>7} {'med':>7} {'q3':>7} {'max':>7}"
    )
    print(header)
    print("-" * len(header))
    for key in records[0]:
        if key in skip:
            continue
        values = [float(r[key]) for r in records]
        stats = {"mean": statistics.mean(values), "sd": statistics.stdev(values)}
        stats.update(five_number(values))
        aggregate[key] = stats
        print(
            f"{key:>14} | {stats['mean']:8.4f} {stats['sd']:8.4f} | "
            f"{stats['min']:7.3f} {stats['q1']:7.3f} {stats['median']:7.3f} "
            f"{stats['q3']:7.3f} {stats['max']:7.3f}"
        )

    out_path = run_dirs[0].parent / "aggregate.json"
    out_path.write_text(json.dumps({"seeds": seeds, "metrics": aggregate}, indent=2))
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
