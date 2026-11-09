"""Render the existing physical acquisition trajectories without changing their dynamics."""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.coupled_duffing.data import DataConfig, render
from diagnostics.shared.io import read_json, save_torch, sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import write_json


def prepare(directory: Path, source_dir: Path) -> dict[str, Any]:
    """Write image/action inputs and separate reference labels for the same three splits."""
    source = read_json(source_dir / "manifest.json")
    if (
        source.get("schema") != "duffing-physical-acquisition-data-v1"
        or canonical_fingerprint({k: v for k, v in source.items() if k != "identity"})
        != source["identity"]
        or source["test_generated"]
        or source["config"]["forecast"] < 2
    ):
        raise ValueError("expected an authenticated physical acquisition cache with two futures")
    settings = source["config"]
    config = DataConfig(
        **{
            key: settings[key]
            for key in (
                "seed",
                "train_groups",
                "calibration_groups",
                "validation_groups",
                "views",
                "context",
            )
        },
        forecast=2,
    )
    if directory.exists():
        record = read_json(directory / "manifest.json")
        if record["source_identity"] != source["identity"] or record["config"] != asdict(config):
            raise ValueError("existing visual cache has different source trajectories or settings")
        for name, digest in record["hashes"].items():
            if sha(directory / name) != digest:
                raise ValueError(f"changed visual artifact: {name}")
        return record
    directory.mkdir(parents=True)
    hashes: dict[str, str] = {}
    length = config.context + config.forecast
    for split, name in enumerate(("train", "calibration", "validation")):
        input_name, label_name = f"{name}-inputs.pt", f"{name}-labels.pt"
        for filename in (input_name, label_name):
            if sha(source_dir / filename) != source["hashes"][filename]:
                raise ValueError(f"changed source artifact: {filename}")
        physical = cast(dict[str, Tensor], torch.load(source_dir / input_name, weights_only=True))
        positions = physical["observations"][:, :, :length].contiguous()
        actions = physical["actions"][:, :, : length - 1].contiguous()
        observations = torch.empty(
            *positions.shape[:-1], 1, config.height, config.width, dtype=torch.float32
        )
        for start in range(0, len(positions), 16):
            observations[start : start + 16] = render(positions[start : start + 16], config)
        group_ids = torch.arange(len(positions), dtype=torch.int64) + split * 1000000
        save_torch(
            directory / input_name,
            {"observations": observations, "actions": actions, "group_ids": group_ids},
        )
        # Reference positions and parameters are stored separately, never in learner inputs.
        labels = cast(dict[str, Tensor], torch.load(source_dir / label_name, weights_only=True))
        save_torch(
            directory / label_name,
            {"states": positions, "theta": labels["theta"], "group_ids": group_ids},
        )
        for filename in (input_name, label_name):
            hashes[filename] = sha(directory / filename)
        print(f"rendered {name}: {len(positions)} systems, {length} frames", flush=True)
        del observations, physical, positions, actions, labels
    record = {
        "schema": "duffing-rendered-physical-data-v1",
        "config": asdict(config),
        "hashes": hashes,
        "source_directory": str(source_dir.resolve()),
        "source_identity": source["identity"],
        "source_manifest_sha256": sha(source_dir / "manifest.json"),
        "source_hashes": source["hashes"],
        "forcing": settings["forcing"],
        "true_states_in_model_inputs": False,
        "reserved_test_split": 3,
        "test_generated": False,
    }
    record["identity"] = canonical_fingerprint(record)
    write_json(directory / "manifest.json", record)
    return record


def main() -> None:
    """Prepare a visual cache from a completed physical diagnostic cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare(args.output_dir, args.source_dir)


if __name__ == "__main__":
    main()
