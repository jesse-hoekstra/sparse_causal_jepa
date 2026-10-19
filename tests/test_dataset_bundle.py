"""Dataset bundle round trips, integrity validation, and training/evaluation separation."""

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from scjepa.data.bundle import load_dataset_bundle, save_dataset_bundle
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.datasets import EpisodeWindows, canonical_fingerprint
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.types import DGPConfig


def read_manifest(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text()))


@pytest.mark.parametrize("family", ["controlled_vector", "coupled_springs"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_bundle_roundtrip_and_separate_training_truth(
    tmp_path: Path, family: str, dtype: str
) -> None:
    dataset = build_dgp_dataset(
        DGPConfig(family=family, num_episodes=2, trajectory_length=3, dtype=dtype)
    )
    manifest_path = save_dataset_bundle(dataset, tmp_path / "bundle")
    saved = load_dataset_bundle(manifest_path)
    assert len(saved) == len(dataset)
    assert saved.fingerprint == dataset.fingerprint
    assert saved.observation_shape == dataset.observation_shape
    assert saved.config == dataset.config
    for index in range(len(dataset)):
        assert set(saved[index]) == {"observations", "actions", "episode_ids"}
        torch.testing.assert_close(saved[index], dataset[index], rtol=0, atol=0)
        torch.testing.assert_close(saved.ground_truth(index), dataset.ground_truth(index))
    torch.testing.assert_close(saved.observation_matrix, dataset.observation_matrix)
    torch.testing.assert_close(saved.graphs.state, dataset.graphs.state)
    manifest = read_manifest(manifest_path)
    assert manifest["bundle_version"] == 1
    descriptor = cast(dict[str, object], manifest["descriptor"])
    assert {"config", "specification", "graphs", "observation_matrix_fingerprint"} <= set(
        descriptor
    )
    if family == "controlled_vector":
        assert (manifest_path.parent / "observation_map.pt").is_file()
    else:
        assert not (manifest_path.parent / "observation_map.pt").exists()
        assert "settings" in cast(
            dict[str, object],
            cast(dict[str, object], descriptor["specification"])["observation_map"],
        )


def test_bundle_rejects_overwrite_and_detects_shard_changes_before_and_after_load(
    tmp_path: Path,
) -> None:
    dataset = build_dgp_dataset(DGPConfig(num_episodes=1, trajectory_length=3))
    directory = tmp_path / "bundle"
    save_dataset_bundle(dataset, directory)
    loaded = load_dataset_bundle(directory)
    with pytest.raises(FileExistsError):
        save_dataset_bundle(dataset, directory)
    shard = directory / "training/00000000.pt"
    shard.write_bytes(shard.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset_bundle(directory)
    with pytest.raises(ValueError, match="checksum"):
        loaded[0]


def test_paired_bundle_roundtrip_preserves_group_shape_and_shared_truth(tmp_path: Path) -> None:
    dataset = build_dgp_dataset(
        DGPConfig(num_episodes=2, trajectory_length=3, episodes_per_parameter=2)
    )
    manifest_path = save_dataset_bundle(dataset, tmp_path / "paired_bundle")
    saved = load_dataset_bundle(manifest_path)
    assert len(saved) == 2
    assert saved.fingerprint == dataset.fingerprint
    assert saved.observation_shape == (10,)
    assert saved.specification["episodes_per_parameter"] == 2
    for index in range(len(saved)):
        torch.testing.assert_close(saved[index], dataset[index], rtol=0, atol=0)
        torch.testing.assert_close(saved.ground_truth(index), dataset.ground_truth(index))
        assert saved[index]["observations"].shape == (2, 3, 10)
        assert saved[index]["actions"].shape == (2, 2, 10)
        assert saved.ground_truth(index)["parameters"].shape == (10,)
    torch.testing.assert_close(
        EpisodeWindows(saved, 2)[0]["observations"], saved[0]["observations"][:, :2]
    )


@pytest.mark.parametrize("change", ["version", "fingerprint", "provenance", "missing_checksum"])
def test_invalid_manifest_is_rejected(tmp_path: Path, change: str) -> None:
    dataset = build_dgp_dataset(DGPConfig(num_episodes=1, trajectory_length=3))
    manifest_path = save_dataset_bundle(dataset, tmp_path / "bundle")
    manifest = read_manifest(manifest_path)
    if change == "version":
        manifest["bundle_version"] = 987
    elif change == "fingerprint":
        manifest["fingerprint"] = "wrong"
    elif change == "missing_checksum":
        cast(dict[str, str], manifest["checksums"]).pop("ground_truth/00000000.pt")
    else:
        descriptor = cast(dict[str, object], manifest["descriptor"])
        cast(dict[str, object], descriptor["specification"])["history_len"] = 3
        manifest["fingerprint"] = canonical_fingerprint(descriptor)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="version|fingerprint|provenance|checksums"):
        load_dataset_bundle(manifest_path)


def test_labels_cannot_be_injected_into_training_shards(tmp_path: Path) -> None:
    dataset = build_dgp_dataset(DGPConfig(num_episodes=1, trajectory_length=3))
    manifest_path = save_dataset_bundle(dataset, tmp_path / "bundle")
    shard = manifest_path.parent / "training/00000000.pt"
    torch.save(dataset[0] | {"parameters": dataset.ground_truth(0)["parameters"]}, shard)
    manifest = read_manifest(manifest_path)
    cast(dict[str, str], manifest["checksums"])["training/00000000.pt"] = hashlib.sha256(
        shard.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    loaded = load_dataset_bundle(manifest_path)
    with pytest.raises(ValueError, match="training shards accept"):
        loaded[0]


def test_bundle_rejects_old_dynamics_with_unchanged_config(tmp_path: Path) -> None:
    dataset = build_dgp_dataset(DGPConfig(num_episodes=1, trajectory_length=3))
    manifest_path = save_dataset_bundle(dataset, tmp_path / "bundle")
    manifest = read_manifest(manifest_path)
    descriptor = cast(dict[str, object], manifest["descriptor"])
    specification = cast(dict[str, object], descriptor["specification"])
    specification.pop("dynamics")
    specification["generator_version"] = "paper-dgps-v1"
    specification["epsilon"] = 1 / 40
    manifest["fingerprint"] = canonical_fingerprint(descriptor)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance does not match the registered generator"):
        load_dataset_bundle(manifest_path)


def test_changed_mixing_matrix_is_rejected_even_with_updated_file_checksum(tmp_path: Path) -> None:
    dataset = build_dgp_dataset(DGPConfig(num_episodes=1, trajectory_length=3))
    manifest_path = save_dataset_bundle(dataset, tmp_path / "bundle")
    matrix_path = manifest_path.parent / "observation_map.pt"
    torch.save({"matrix": torch.eye(10, dtype=torch.float64)}, matrix_path)
    manifest = read_manifest(manifest_path)
    cast(dict[str, str], manifest["checksums"])["observation_map.pt"] = hashlib.sha256(
        matrix_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="observation matrix"):
        load_dataset_bundle(manifest_path)


def test_content_identity_changes_with_shards_even_when_provenance_is_retained(
    tmp_path: Path,
) -> None:
    dataset = build_dgp_dataset(DGPConfig(num_episodes=1, trajectory_length=3))
    manifest_path = save_dataset_bundle(dataset, tmp_path / "bundle")
    original = load_dataset_bundle(manifest_path)
    original_content_fingerprint = original.content_fingerprint
    original_windows = EpisodeWindows(original, 2)
    original_window_fingerprint = original_windows.fingerprint
    shard = manifest_path.parent / "training/00000000.pt"
    training = dataset[0]
    training["observations"][0, 0] += 1
    torch.save(training, shard)
    manifest = read_manifest(manifest_path)
    cast(dict[str, str], manifest["checksums"])["training/00000000.pt"] = hashlib.sha256(
        shard.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    changed = load_dataset_bundle(manifest_path)
    assert changed.fingerprint == original.fingerprint
    assert changed.content_fingerprint != original_content_fingerprint
    with pytest.raises(ValueError, match="checksum"):
        _ = original.content_fingerprint
    with pytest.raises(ValueError, match="checksum"):
        _ = original_windows.content_fingerprint
    assert EpisodeWindows(changed, 2).fingerprint != original_window_fingerprint


def test_persisted_map_is_authoritative_when_seed_based_realization_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = DGPConfig(num_episodes=1, trajectory_length=3)
    dataset = build_dgp_dataset(config)
    manifest_path = save_dataset_bundle(dataset, tmp_path / "bundle")
    original_init = VectorObservationMap.__init__

    def different_realization(
        self: VectorObservationMap, seed: int = 1729, matrix: Tensor | None = None
    ) -> None:
        original_init(self, seed=seed + 1 if matrix is None else seed, matrix=matrix)

    monkeypatch.setattr(VectorObservationMap, "__init__", different_realization)
    assert build_dgp_dataset(config).fingerprint != dataset.fingerprint
    loaded = load_dataset_bundle(manifest_path)
    assert loaded.fingerprint == dataset.fingerprint
    torch.testing.assert_close(
        loaded.observation_matrix, dataset.observation_matrix, rtol=0, atol=0
    )
    torch.testing.assert_close(loaded[0], dataset[0], rtol=0, atol=0)


def test_bundle_rejects_inconsistent_episode_identity(tmp_path: Path) -> None:
    dataset = build_dgp_dataset(DGPConfig(num_episodes=1, trajectory_length=3))
    manifest_path = save_dataset_bundle(dataset, tmp_path / "bundle")
    shard = manifest_path.parent / "training/00000000.pt"
    torch.save(dataset[0] | {"episode_ids": torch.tensor(42)}, shard)
    manifest = read_manifest(manifest_path)
    cast(dict[str, str], manifest["checksums"])["training/00000000.pt"] = hashlib.sha256(
        shard.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    loaded = load_dataset_bundle(manifest_path)
    with pytest.raises(ValueError, match="episode ID"):
        loaded[0]
