"""Calibrate a dense JEPA reference, then compare two GECO-sparse models."""

import argparse
import json
from pathlib import Path

from scjepa.experiments.config import load_experiment_config
from scjepa.experiments.reporting import aggregate_results
from scjepa.experiments.runner import experiment_plan, run_experiment
from scjepa.experiments.storage import prepare_experiment
from scjepa.training.distributed import is_main_process


def main() -> None:
    """Resolve a strict experiment configuration before any run is launched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/experiments/controlled_vector.yaml",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--dry-run", action="store_true", help="print the plan without writing outputs"
    )
    action.add_argument(
        "--aggregate-only", action="store_true", help="refresh summaries of existing runs"
    )
    action.add_argument(
        "--evaluate-only", action="store_true", help="evaluate existing final (last.pt) checkpoints"
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume interrupted runs; skip completed runs"
    )
    parser.add_argument("--graph", choices=("dense", "complementary", "diagonal"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--condition", choices=("dense", "baseline", "full"))
    parser.add_argument("overrides", nargs="*", help="typed YAML overrides, e.g. train.device=cuda")
    args = parser.parse_args()
    config = load_experiment_config(args.config, args.overrides)
    plan = experiment_plan(config, graph=args.graph, seed=args.seed, condition=args.condition)
    if args.dry_run:
        print(json.dumps({"runs": len(plan), "plan": plan}, indent=2))
        return
    if args.aggregate_only:
        if not (Path(config.output_dir) / "experiment.json").is_file():
            parser.error("--aggregate-only requires an existing experiment manifest")
        prepare_experiment(config)
        result = aggregate_results(config)
    else:
        result = run_experiment(
            config,
            resume=args.resume,
            evaluate_only=args.evaluate_only,
            graph=args.graph,
            seed=args.seed,
            condition=args.condition,
        )
    if result and is_main_process():
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("complete", "completed_runs", "expected_runs", "missing_runs")
                },
                indent=2,
            )
        )
    if result.get("interrupted"):
        raise SystemExit(130)


if __name__ == "__main__":
    main()
