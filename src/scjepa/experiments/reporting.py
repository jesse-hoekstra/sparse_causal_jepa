"""Dense-reference and paired sparse summaries for completed controlled-vector runs."""

import csv
import io
import math
import statistics
from pathlib import Path
from typing import cast

from scjepa.experiments.comparison import paired_mcc_comparison
from scjepa.experiments.config import ExperimentConfig
from scjepa.experiments.storage import file_lock, read_json, write_json


def _statistics(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "standard_error": statistics.stdev(values) / math.sqrt(len(values))
        if len(values) > 1
        else None,
    }


def aggregate_results(config: ExperimentConfig) -> dict[str, object]:
    """Summarize actual finished pairs, explicitly retaining missing-run counts."""
    root = Path(config.output_dir)
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    missing_pairs: list[str] = []
    execution_path = root / "execution.json"
    expected_world_size = (
        read_json(execution_path).get("world_size") if execution_path.exists() else None
    )
    completed: dict[tuple[str, int, str], tuple[dict[str, object], dict[str, float]]] = {}
    with file_lock(root / ".report.lock", blocking=True):
        for graph in config.graphs:
            for seed in config.seeds:
                for condition in config.conditions:
                    slug = f"{graph}/seed-{seed}/{condition}"
                    directory = root / slug
                    status_path = directory / "status.json"
                    if (
                        not status_path.exists()
                        or read_json(status_path).get("status") != "completed"
                    ):
                        missing.append(slug)
                        continue
                    record = read_json(directory / "run_config.json")
                    processes = record.get("world_size")
                    if (
                        not isinstance(processes, int)
                        or isinstance(processes, bool)
                        or processes < 1
                        or (expected_world_size is not None and processes != expected_world_size)
                    ):
                        raise ValueError(f"unmatched or invalid world_size in {slug}")
                    expected_world_size = processes
                    values = read_json(directory / "evaluation" / "metrics.json")
                    if any(
                        not isinstance(v, int | float) or not math.isfinite(v)
                        for v in values.values()
                    ):
                        raise ValueError(f"nonfinite or nonscalar evaluation results in {slug}")
                    metrics = {
                        key: float(value) for key, value in cast(dict[str, float], values).items()
                    }
                    completed[(graph, seed, condition)] = (record, metrics)
                    rows.append({"graph": graph, "seed": seed, "condition": condition, **metrics})
        summaries: dict[str, object] = {}
        differences: dict[str, object] = {}
        group_mcc: dict[str, object] = {}
        missing_group_mcc: list[str] = []
        for graph in config.graphs:
            for condition in config.conditions:
                group = [
                    metrics
                    for (g, _, c), (_, metrics) in completed.items()
                    if g == graph and c == condition
                ]
                if group:
                    keys = set[str].intersection(*(set(metrics) for metrics in group))
                    summaries[f"{graph}/{condition}"] = {
                        key: _statistics([metrics[key] for metrics in group])
                        for key in sorted(keys)
                    }
            pairs: list[tuple[dict[str, float], dict[str, float]]] = []
            for seed in config.seeds:
                dense = completed.get((graph, seed, "dense"))
                baseline, full = (
                    completed.get((graph, seed, "baseline")),
                    completed.get((graph, seed, "full")),
                )
                if baseline is None or full is None:
                    missing_pairs.append(f"{graph}/seed-{seed}")
                    continue
                for key in ("initial_state_fingerprint", "dataset_identity", "experiment_identity"):
                    value = baseline[0].get(key)
                    if (
                        not isinstance(value, str)
                        or not value
                        or value != full[0].get(key)
                        or (
                            dense is not None
                            and key != "initial_state_fingerprint"
                            and value != dense[0].get(key)
                        )
                    ):
                        raise ValueError(f"unmatched {key} for {graph}, seed {seed}")
                calibration = baseline[0].get("tau_calibration")
                if (
                    not isinstance(calibration, dict)
                    or not calibration
                    or calibration != full[0].get("tau_calibration")
                    or (
                        dense is not None
                        and cast(dict[str, object], calibration).get("dense_run_identity")
                        != dense[0].get("run_identity")
                    )
                ):
                    raise ValueError(f"unmatched tau_calibration for {graph}, seed {seed}")
                pairs.append((baseline[1], full[1]))
                slug = f"{graph}/seed-{seed}"
                directories = {name: root / slug / name for name in ("dense", "baseline", "full")}
                if dense is None or any(
                    not (path / "evaluation" / "mcc_samples.pt").exists()
                    for path in directories.values()
                ):
                    # Legacy evaluations have no grouped samples. Scalar MCC
                    # cannot supply a valid independent-group uncertainty estimate.
                    missing_group_mcc.append(slug)
                else:
                    comparison = paired_mcc_comparison(directories, fixed_budget=config.train.steps)
                    group_mcc[slug] = comparison
                    write_json(root / "paired_mcc" / f"{slug}.json", comparison)
            if pairs:
                keys = set[str].intersection(*(set(a) & set(b) for a, b in pairs))
                differences[graph] = {
                    key: _statistics([full[key] - baseline[key] for baseline, full in pairs])
                    for key in sorted(keys)
                }
        result: dict[str, object] = {
            "format_version": 2,
            "complete": not missing,
            "missing_runs": missing,
            "missing_pairs": missing_pairs,
            "completed_runs": len(rows),
            "expected_runs": len(config.conditions) * len(config.graphs) * len(config.seeds),
            "per_condition": summaries,
            "paired_full_minus_baseline": differences,
            "paired_group_mcc": group_mcc,
            "missing_group_mcc": missing_group_mcc,
        }
        write_json(root / "summary.json", result)
        if rows:
            fields = [
                "graph",
                "seed",
                "condition",
                *sorted(
                    set[str].union(*(set(row) for row in rows)) - {"graph", "seed", "condition"}
                ),
            ]
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            temporary = root / ".results.csv.tmp"
            temporary.write_text(stream.getvalue())
            temporary.replace(root / "results.csv")
        return result
