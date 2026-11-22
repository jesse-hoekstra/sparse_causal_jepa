"""Determine the native continuation device for a prospective combined seed."""

import argparse
from pathlib import Path
from typing import Any

from .changed_queries import Inputs
from .vision_acquisition import OUTPUT, read
from .vision_combined import native_gpu


def required(dataset: str, phase: str, seed: int, output: Path = OUTPUT) -> str:
    """Return CPU for receipt-only routes, or the recorded model for numerical replay."""
    if phase == "cohort":
        return "cpu"
    if dataset == "synthetic":
        warm = output / f"synthetic-warm-s{seed}"
        if read(warm / "metrics.json").get("reason") == "train_admission":
            return "cpu"
        config = read(warm / "full-350k/results/config.json")
        return str(config["runtime"]["gpu_name"])
    four: dict[str, Any] = read(output / f"rgb-four-s{seed}/metrics.json")
    if not four["terminal_paths"] or four["route"] == "reported":
        return "cpu"
    return native_gpu(Path(four["terminal_paths"]["full"]).parent, Inputs())


def main() -> None:
    """Print the declared device for a launcher's seed and phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("synthetic", "vision"))
    parser.add_argument("seed", type=int, choices=range(5))
    parser.add_argument("phase", choices=("cohort", "run"))
    args = parser.parse_args()
    print(required(args.dataset, args.phase, args.seed))


if __name__ == "__main__":
    main()
