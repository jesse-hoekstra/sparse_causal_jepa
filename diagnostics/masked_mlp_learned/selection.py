"""Select completed acquisition restarts by their final TRAIN forecast loss only.

This selector never reads validation metrics, physical parameters, states or model
tensors. It authenticates the candidates and selects one fixed-budget endpoint per
state seed. A low training loss is an admission screen, not proof of retention.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from diagnostics.masked_mlp_learned.run import verify_completion
from diagnostics.shared.io import read_json, sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import write_json


def training_score(directory: Path, *, steps: int, window: int, log_every: int) -> float:
    """Average the uniformly logged forecast losses over a complete final window."""
    if not 0 < window <= steps or window % log_every or steps % log_every:
        raise ValueError("window and endpoint must contain a complete logging grid")
    losses: dict[int, float] = {}
    for line in (directory / "history.jsonl").read_text().splitlines():
        row = json.loads(line)
        step = int(row["step"])
        if "loss/prediction" not in row or not steps - window < step <= steps:
            continue
        value = float(row["loss/prediction"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("TRAIN prediction loss must be finite and nonnegative")
        if step in losses:
            raise ValueError("duplicate TRAIN loss in the selection window")
        losses[step] = value
    expected = set(range(steps - window + log_every, steps + 1, log_every))
    if set(losses) != expected:
        raise ValueError("incomplete TRAIN selection window")
    return math.fsum(losses.values()) / len(losses)


def select(
    directories: list[Path], *, window: int = 5000, maximum_loss: float = 0.025
) -> dict[str, Any]:
    """Select by TRAIN loss, breaking ties by acquisition seed; require matched parents."""
    if not directories or not math.isfinite(maximum_loss) or maximum_loss <= 0:
        raise ValueError("provide candidates and a finite positive loss threshold")
    candidates: list[dict[str, Any]] = []
    common: dict[str, Any] | None = None
    seeds: set[int] = set()
    for path in directories:
        directory = path.resolve()
        completion = verify_completion(directory, kind="train")
        config = read_json(directory / "config.json")["config"]
        source = completion.get("state_source")
        if completion["condition"] != "dense" or source is None:
            raise ValueError("selection requires completed frozen-state dense acquisitions")
        specification = {
            key: completion[key]
            for key in ("seed", "code_dim", "steps", "data_identity", "code_identity", "profile")
        } | {"state_source_identity": source["identity"]}
        if common is not None and common != specification:
            raise ValueError("acquisition candidates differ in parent, data, source or budget")
        common = specification
        seed = config.get("acquisition_seed")
        seed = int(config["seed"] if seed is None else seed)
        if seed in seeds:
            raise ValueError("acquisition initialization seeds must be distinct")
        seeds.add(seed)
        score = training_score(
            directory,
            steps=int(completion["steps"]),
            window=window,
            log_every=int(config["log_every"]),
        )
        candidates.append(
            {
                "directory": str(directory),
                "identity": completion["identity"],
                "completion_sha256": sha(directory / "completion.json"),
                "acquisition_seed": seed,
                "train_prediction_mean": score,
            }
        )
    selected = min(
        candidates, key=lambda row: (row["train_prediction_mean"], row["acquisition_seed"])
    )
    record: dict[str, Any] = {
        "schema": "masked-mlp-acquisition-selection-v1",
        "common": common,
        "window": window,
        "maximum_loss": maximum_loss,
        "policy": "minimum final-window TRAIN forecast loss; tie by initialization seed",
        "candidates": candidates,
        "selected": selected,
        "admitted": selected["train_prediction_mean"] <= maximum_loss,
        "selection_used_parameter_labels": False,
    }
    record["identity"] = canonical_fingerprint(record)
    return record


def main() -> None:
    """Write an immutable label-free selection receipt; exit nonzero on failed admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window", type=int, default=5000)
    parser.add_argument("--maximum-loss", type=float, default=0.025)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    result = select(arguments.runs, window=arguments.window, maximum_loss=arguments.maximum_loss)
    write_json(arguments.output, result)
    print(json.dumps(result, indent=2))
    if not result["admitted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
