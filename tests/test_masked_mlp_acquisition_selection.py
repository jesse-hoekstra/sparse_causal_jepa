"""Label-free endpoint selection must not follow retention or state-recovery scores."""

import json
from pathlib import Path
from typing import Any

import pytest

from diagnostics.masked_mlp_learned.run import SCHEMA
from diagnostics.masked_mlp_learned.selection import select, training_score
from diagnostics.oracle_state_learning.data import sha
from scjepa.experiments.storage import write_json


def _candidate(root: Path, seed: int, loss: float, retention: float) -> Path:
    directory = root / str(seed)
    directory.mkdir()
    history = [
        {"step": 0, "loss/prediction": 1000.0},
        *({"step": step, "loss/prediction": loss} for step in (100, 200, 300, 400)),
        {"step": 400, "validation/code_r2_min": retention, "validation/prediction": -1e10},
    ]
    (directory / "history.jsonl").write_text("\n".join(json.dumps(row) for row in history))
    write_json(
        directory / "config.json",
        {"config": {"seed": 0, "acquisition_seed": seed, "log_every": 100}},
    )
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": "train",
        "identity": str(seed),
        "condition": "dense",
        "seed": 0,
        "code_dim": 12,
        "steps": 400,
        "data_identity": "data",
        "code_identity": "source",
        "profile": "smoke",
        "state_source": {"identity": "parent"},
        "hashes": {name: sha(directory / name) for name in ("history.jsonl", "config.json")},
    }
    write_json(directory / "completion.json", record)
    return directory


def test_selection_uses_only_train_loss_and_authenticates_sources(tmp_path: Path) -> None:
    lower = _candidate(tmp_path, 1000, 0.02, -1.0)
    higher = _candidate(tmp_path, 1001, 0.05, 1.0)
    result = select([higher, lower], window=200)
    assert result["selected"]["directory"] == str(lower)
    assert result["admitted"]
    assert not result["selection_used_parameter_labels"]
    assert not select([higher], window=200)["admitted"]
    with (higher / "history.jsonl").open("a") as stream:
        stream.write("\n{}")
    with pytest.raises(ValueError, match="hash"):
        select([higher, lower], window=200)


def test_selection_rejects_unmatched_parents_and_incomplete_windows(tmp_path: Path) -> None:
    left = _candidate(tmp_path, 1000, 0.02, 0.5)
    right = _candidate(tmp_path, 1001, 0.02, 0.5)
    result = select([right, left], window=200)
    assert result["selected"]["acquisition_seed"] == 1000
    path = right / "completion.json"
    record = json.loads(path.read_text())
    record["state_source"]["identity"] = "different"
    write_json(path, record)
    with pytest.raises(ValueError, match="parent"):
        select([left, right], window=200)
    with pytest.raises(ValueError, match="incomplete"):
        training_score(left, steps=500, window=200, log_every=100)
