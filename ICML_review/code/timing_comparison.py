"""Compare the two fixed-head timing schedules on the same held-out systems."""

import argparse
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from scjepa.experiments.storage import write_json

from .consensus_gradient import context_intervals
from .single_acquisition import PRIVATE
from .single_acquisition import _head as check_head  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import declared, read, sha


def samples(out: Path, dataset: str, split: str, record: dict[str, Any]) -> tuple[Tensor, Tensor]:
    """Load only samples identified in a terminal calibration or validation receipt."""
    if split not in {"calibration", "validation"}:
        raise ValueError("Timing comparisons require a calibration or validation split")
    if dataset == "synthetic":
        path = Path(record["directories"]["treatment"]) / f"{split}.pt"
        if sha(path) != record["records"]["treatment"][split]["bootstrap_samples_sha256"]:
            raise ValueError("Synthetic sample identity mismatch")
        arrays = cast(dict[str, Any], torch.load(path, weights_only=False, map_location="cpu"))[
            "samples"
        ]
        mse = (arrays["predictions"].double() - arrays["targets"].double()).square().flatten(1)
        return arrays["codes"].double(), mse.mean(1)
    if dataset != "rgb":
        raise ValueError("Unknown comparison dataset")
    path = out / "endpoint.json"
    if sha(path) != record["diagnostic_sha256"]["endpoint.json"]:
        raise ValueError("RGB sample identity mismatch")
    arrays = read(path)[split]
    return torch.tensor(arrays["codes"], dtype=torch.float64), torch.tensor(
        arrays["group_mse"], dtype=torch.float64
    )


def compare(revision: str, dataset: str) -> None:
    """Bootstrap paired differences with all episodes of a system kept together."""
    check_head(revision)
    torch.set_num_threads(1)
    seed = {"rgb": 3, "synthetic": 4}[dataset]
    folders = {s: PRIVATE / f"timing-{dataset}-{s}-s{seed}" for s in ("sparse", "entire")}
    folder = PRIVATE / f"timing-comparison-{dataset}-s{seed}"
    declared(
        folder / "NOTE.md",
        f"# Paired difference between sibling-guidance schedules for {dataset}\n\n"
        "Hypothesis: adding the early online sibling term changes within-system variation "
        "and forecast error among arms already trained with sparse sibling guidance. Fixed "
        "candidate zero and common held-out systems. Use 2,000 paired resamples of whole "
        "systems, retaining every sibling episode. Compare entire minus sparse on calibration "
        "and validation. An interval above zero favors sparse-only for the corresponding error. "
        "An interval below zero favors entire. Report uncertainty conditional on this seed. "
        "All terminal metrics and guard failures remain part of the verdict. These are the "
        "datasets and split identities recorded in the two original declarations.\n\n"
        + "\n".join(f"- {p}/NOTE.md SHA-256 {sha(p / 'NOTE.md')}" for p in folders.values())
        + f"\n\nCommit `{revision}`\n\n"
        f"    PYTHONPATH=$PWD:$PWD/src .venv/bin/python -m ICML_review.code.timing_comparison "
        f"{revision} {dataset}\n",
    )
    records = {s: read(d / "metrics.json") for s, d in folders.items()}
    for schedule, row in records.items():
        if (row["schedule"], row["seed"], row["commit"]) != (schedule, seed, revision):
            raise ValueError("Comparison requires the two predeclared endpoints")
    results: dict[str, Any] = {}
    for index, split in enumerate(("calibration", "validation")):
        data = {s: samples(d, dataset, split, records[s]) for s, d in folders.items()}
        sides = {"native": "sparse", "same": "sparse", "other": "entire"}
        try:
            results[split] = context_intervals(
                {k: data[v][0] for k, v in sides.items()},
                {k: data[v][1] for k, v in sides.items()},
                20261123 + index,
            )["other_minus_native"]
        except ValueError as exc:
            results[split] = {"undefined": str(exc)}
    check_head(revision)
    write_json(
        folder / "metrics.json",
        {
            "commit": revision,
            "seed": seed,
            "dataset": dataset,
            "entire_minus_sparse": results,
            "inputs": {s: sha(d / "metrics.json") for s, d in folders.items()},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision")
    parser.add_argument("dataset", choices=("synthetic", "rgb"))
    opts = parser.parse_args()
    compare(opts.revision, opts.dataset)
