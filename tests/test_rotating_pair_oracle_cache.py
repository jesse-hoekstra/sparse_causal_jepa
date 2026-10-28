"""Paired addressed inputs, changed trajectories and sealed labels for the new phase."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.balanced_action_oracle import data as source_data
from diagnostics.isotropic_oracle import data as ancestor_data
from diagnostics.oracle_state_learning.data import read_json, sha
from diagnostics.rotating_pair_oracle import data, dgp


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _source(root: Path) -> Path:
    ancestor = root / "ancestor"
    ancestor_data.prepare(
        ancestor_data.PrepareConfig(
            str(ancestor),
            profile="smoke",
            train_groups=4,
            calibration_groups=2,
            validation_groups=2,
            test_groups=2,
        )
    )
    source = root / "source"
    source_data.prepare(source_data.PrepareConfig(str(source), str(ancestor), profile="smoke"))
    return source


def test_persisted_observation_matrix_survives_platform_qr_rounding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    original_map = data.VectorObservationMap
    persisted = cast(
        dict[str, Tensor], torch.load(source / "observation_map.pt", weights_only=True)
    )["matrix"]
    alternate = persisted.clone()
    alternate[0, 0] = torch.nextafter(alternate[0, 0], torch.tensor(float("inf")))
    assert not torch.equal(alternate, persisted)
    # Both matrices satisfy the real map's orthogonality validation; only their
    # final floating-point rounding differs, as with platform-dependent QR.
    original_map(matrix=alternate, dimension=6)
    supplied_matrices: list[Tensor | None] = []

    class PlatformMap(original_map):
        def __init__(
            self, seed: int = 1729, matrix: Tensor | None = None, *, dimension: int = 6
        ) -> None:
            supplied_matrices.append(matrix)
            super().__init__(
                seed=seed, matrix=alternate if matrix is None else matrix, dimension=dimension
            )

    monkeypatch.setattr(data, "VectorObservationMap", PlatformMap)
    output = tmp_path / "paired"
    data.prepare(data.PrepareConfig(str(output), str(source), profile="smoke"))
    assert len(supplied_matrices) == 1
    assert supplied_matrices[0] is not None
    assert torch.equal(supplied_matrices[0], persisted)
    assert sha(output / "observation_map.pt") == sha(source / "observation_map.pt")
    old = cast(dict[str, Any], torch.load(source / "train.pt", weights_only=True))
    new = cast(dict[str, Any], torch.load(output / "train.pt", weights_only=True))
    assert torch.equal(
        old["tensors"]["observations"][:, :, 0], new["tensors"]["observations"][:, :, 0]
    )


def test_regenerated_phase_preserves_inputs_and_initial_states_without_loading_test_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    source_hash = sha(source / "completion.json")
    source_identity = source_data.code_identity()
    original_load = torch.load
    loaded: list[str] = []

    def observe(path: Path, *, weights_only: bool) -> object:
        loaded.append(path.name)
        if path.name == "test_states.pt":
            raise AssertionError("Preparation and ordinary loading must leave test labels closed")
        return original_load(path, weights_only=weights_only)

    output = tmp_path / "paired"
    config = data.PrepareConfig(str(output), str(source), profile="smoke")
    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", observe)
        completion = data.prepare(config)
        loaded.clear()
        samples, authenticated = data.load_data(output)
        assert "test.pt" not in loaded
        assert "test_states.pt" not in loaded
    assert completion == authenticated
    assert source_identity == source_data.code_identity()
    assert source_hash == sha(source / "completion.json")
    assert set(samples) == {"train", "calibration", "validation"}
    assert samples["train"].states is None
    assert not (output / "train_states.pt").exists()
    assert completion["settings"]["candidate"] == "rotating_pair_innovation"
    assert completion["settings"]["phase_gain"] == dgp.PHASE_GAIN
    assert completion["settings"]["phase_formula"] == "rotating_pair"
    for split in data.SPLITS:
        old = cast(dict[str, Any], torch.load(source / f"{split}.pt", weights_only=True))
        new = cast(dict[str, Any], torch.load(output / f"{split}.pt", weights_only=True))
        assert set(new) == {"tensors", "metadata"}
        for name in ("actions", "oracle_parameters", "episode_ids"):
            assert torch.equal(old["tensors"][name], new["tensors"][name])
        assert torch.equal(
            old["tensors"]["observations"][:, :, 0], new["tensors"]["observations"][:, :, 0]
        )
        assert not torch.equal(
            old["tensors"]["observations"][:, :, 1:], new["tensors"]["observations"][:, :, 1:]
        )
        assert new["metadata"]["addressed_theta_actions_ids_initial_observations_verified"]
        if split != "train":
            # These are tiny software fixtures; the scientific test remains unopened.
            old_states = cast(
                dict[str, Tensor], torch.load(source / f"{split}_states.pt", weights_only=True)
            )["states"]
            new_states = cast(
                dict[str, Tensor], torch.load(output / f"{split}_states.pt", weights_only=True)
            )["states"]
            assert torch.equal(old_states[:, :, 0], new_states[:, :, 0])
            assert not torch.equal(old_states[:, :, 1:], new_states[:, :, 1:])
    for name in ("observation_map.pt", "action_map.pt"):
        assert sha(source / name) == sha(output / name)
    assert data.prepare(config, resume=True) == completion
    protocol = read_json(output / "protocol.json")
    assert protocol["source_completion_sha256"] == source_hash
    assert protocol["source_code_identity"] == source_identity


@pytest.mark.parametrize("stream", ["parameters", "initial", "actions"])
def test_mismatched_addressed_inputs_fail_before_replacing_the_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: str
) -> None:
    source = _source(tmp_path)
    original = data.generator

    def changed_address(seed: int, split: str, index: int, namespace: str) -> torch.Generator:
        return original(seed + int(namespace.endswith(stream)), split, index, namespace)

    monkeypatch.setattr(data, "generator", changed_address)
    with pytest.raises(ValueError, match="Regenerated paired inputs differ"):
        data.prepare(data.PrepareConfig(str(tmp_path / "paired"), str(source), profile="smoke"))
    assert not (tmp_path / "paired/completion.json").exists()


def test_source_hash_tampering_and_profile_mismatch_are_rejected(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="profiles differ"):
        data.prepare(data.PrepareConfig(str(tmp_path / "wrong_profile"), str(source)))
    with (source / "train.pt").open("ab") as stream:
        stream.write(b"tampered")
    output = tmp_path / "paired"
    with pytest.raises(ValueError, match="Completed artifact changed"):
        data.prepare(data.PrepareConfig(str(output), str(source), profile="smoke"))
    assert not output.exists()


def test_parent_cache_cannot_be_loaded_and_readonly_files_do_not_change_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="rotating pair cache schema"):
        data.load_data(source)
    before = data.code_identity()
    original_read = Path.read_bytes
    directory = Path(data.__file__).parent.resolve()

    def hypothetical_report_edit(path: Path) -> bytes:
        value = original_read(path)
        return (
            value + b"\n# read-only report edit\n"
            if path.parent.resolve() == directory and path.name in {"README.md", "analysis.py"}
            else value
        )

    monkeypatch.setattr(Path, "read_bytes", hypothetical_report_edit)
    assert data.code_identity() == before
