from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from diagnostics.oracle_state_learning.core import TRAINING_KEYS, build_data
from diagnostics.oracle_state_learning.data import (
    PrepareConfig,
    load_data,
    prepare,
    save_torch,
    sha,
    verify_completion,
)
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import write_json


def smoke_config(output: Path) -> PrepareConfig:
    """Keep original R10 trajectories while using tiny addressed software fixtures."""
    return PrepareConfig(
        str(output),
        train_groups=2,
        calibration_groups=2,
        validation_groups=2,
        test_groups=2,
        profile="smoke",
    )


def test_cache_preserves_raw_addressed_data_and_excludes_training_state_labels(
    tmp_path: Path,
) -> None:
    config = smoke_config(tmp_path / "cache")
    original_rng = torch.get_rng_state().clone()
    completion = prepare(config)
    torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
    data, restored = load_data(Path(config.output_dir))
    assert restored == completion
    assert set(data) == {"train", "calibration", "validation", "test"}
    assert data["train"].states is None
    seen: set[int] = set()
    for split, sample in data.items():
        original = build_data(split, 2)
        torch.testing.assert_close(sample.tensors, original.tensors, rtol=0, atol=0)
        assert sample.fingerprint == original.fingerprint
        ids = {int(value) for value in sample.tensors["episode_ids"]}
        assert not ids & seen
        seen.update(ids)
        assert set(sample.batch(torch.tensor([1]), "cpu")) == TRAINING_KEYS
        payload = cast(
            dict[str, Any],
            torch.load(Path(config.output_dir) / f"{split}.pt", weights_only=True),
        )
        assert payload["states"] is None
        if split != "train":
            torch.testing.assert_close(sample.states, original.states, rtol=0, atol=0)
            assert (Path(config.output_dir) / f"{split}_states.pt").is_file()
    assert not (Path(config.output_dir) / "train_states.pt").exists()
    assert prepare(config, resume=True) == completion
    with pytest.raises(ValueError, match="explicit cache resume"):
        prepare(config)
    with pytest.raises(ValueError, match="settings/code changed"):
        prepare(replace(config, train_groups=3), resume=True)


def test_cache_rejects_changed_artifact_and_metadata(tmp_path: Path) -> None:
    config = smoke_config(tmp_path / "cache")
    completion = prepare(config)
    directory = Path(config.output_dir)
    completion["settings"]["train_groups"] = 3
    write_json(directory / "completion.json", completion)
    with pytest.raises(ValueError, match="Completion metadata changed"):
        load_data(directory)
    completion["settings"]["train_groups"] = 2
    write_json(directory / "completion.json", completion)
    with (directory / "train.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Completed artifact changed"):
        load_data(directory)


def test_cache_rejects_state_label_in_input_payload_even_with_updated_file_hash(
    tmp_path: Path,
) -> None:
    config = smoke_config(tmp_path / "cache")
    completion = prepare(config)
    directory = Path(config.output_dir)
    payload = cast(dict[str, Any], torch.load(directory / "train.pt", weights_only=True))
    payload["states"] = torch.zeros(2, 10, 12, 6)
    save_torch(directory / "train.pt", payload)
    completion["hashes"]["train.pt"] = sha(directory / "train.pt")
    completion.pop("completion_fingerprint")
    completion["completion_fingerprint"] = canonical_fingerprint(completion)
    write_json(directory / "completion.json", completion)
    with pytest.raises(ValueError, match="must not contain evaluation state labels"):
        load_data(directory)


def test_preparation_guards_scientific_cohorts_and_completion_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="8192/512/256/512"):
        PrepareConfig(str(tmp_path), train_groups=2)
    with pytest.raises(ValueError, match="data seed 0"):
        replace(smoke_config(tmp_path), seed=1)
    with pytest.raises(ValueError, match="two to sixteen"):
        replace(smoke_config(tmp_path), train_groups=8192)
    outside = tmp_path / "outside.json"
    write_json(outside, {"saved": True})
    directory = tmp_path / "cache"
    write_json(directory / "completion.json", {"hashes": {"../outside.json": sha(outside)}})
    with pytest.raises(ValueError, match="direct children"):
        verify_completion(directory)
