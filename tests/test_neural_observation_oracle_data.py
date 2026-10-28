"""Observation-only pairing, authenticated map provenance and fresh held-out addresses."""

import copy
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.neural_observation_oracle import data
from diagnostics.neural_observation_oracle import map as observation
from diagnostics.oracle_state_learning import data as source_data
from diagnostics.oracle_state_learning.core import build_control
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.datasets import canonical_fingerprint
from scjepa.data.identity import tensor_digest
from scjepa.data.random import episode_id


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _source(root: Path) -> tuple[Path, Path]:
    source = root / "original"
    source_data.prepare(
        source_data.PrepareConfig(
            str(source),
            profile="smoke",
            train_groups=4,
            calibration_groups=2,
            validation_groups=2,
            test_groups=2,
        )
    )
    base = root / "base_map.pt"
    source_data.save_torch(base, {"matrix": VectorObservationMap(seed=1729, dimension=6).matrix})
    return source, base


def _prepared(root: Path) -> tuple[Path, Path, Path]:
    source, base = _source(root)
    output = root / "new"
    data.prepare(data.PrepareConfig(str(output), str(source), str(base), profile="smoke"))
    return source, base, output


def test_observation_only_pairing_old_test_sealed_and_fresh_test_addresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, base = _source(tmp_path)
    output = tmp_path / "new"
    before_source = source_data.sha(source / "completion.json")
    load = torch.load
    accessed: list[Path] = []

    def guarded(path: Path, *, weights_only: bool) -> object:
        path = Path(path)
        accessed.append(path)
        if path.name in {"test.pt", "test_states.pt"}:
            raise AssertionError(
                "Preparing and default loading must not deserialize test artifacts"
            )
        return load(path, weights_only=weights_only)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", guarded)
        complete = data.prepare(
            data.PrepareConfig(str(output), str(source), str(base), profile="smoke")
        )
        paired, _ = data.load_data(output)
    assert set(paired) == {"train", "calibration", "validation"}
    assert source_data.sha(source / "completion.json") == before_source
    assert paired["train"].states is None
    assert not (output / "train_states.pt").exists()
    for split in data.PAIRED_SPLITS:
        original = cast(dict[str, Any], load(source / f"{split}.pt", weights_only=True))
        for name in ("actions", "oracle_parameters", "episode_ids"):
            assert torch.equal(paired[split].tensors[name], original["tensors"][name])
        assert not torch.equal(
            paired[split].tensors["observations"], original["tensors"]["observations"]
        )
        assert set(paired[split].batch(torch.tensor([0]), "cpu")) == {
            "observations",
            "actions",
            "oracle_parameters",
            "episode_ids",
        }
        if split != "train":
            assert source_data.sha(output / f"{split}_states.pt") == source_data.sha(
                source / f"{split}_states.pt"
            )
    assert source_data.sha(output / "observation_map.pt") == source_data.sha(base)
    assert complete["original_test_deserialized"] is False
    all_data, _ = data.load_data(output, include_test=True)
    fresh = all_data["test"]
    assert fresh.config.root_seed == data.TEST_ROOT_SEED
    assert torch.equal(
        fresh.tensors["episode_ids"],
        torch.tensor(
            [episode_id(data.TEST_ROOT_SEED, "test", index) for index in range(len(fresh))]
        ),
    )
    seen = [int(value) for sample in all_data.values() for value in sample.tensors["episode_ids"]]
    assert len(set(seen)) == len(seen)
    old_ids = {episode_id(0, "test", index) for index in range(2)}
    assert not old_ids & set(seen)
    network = observation.FrozenObservationMap.from_payload(
        cast(dict[str, Any], load(output / "neural_observation_map.pt", weights_only=True))
    )
    base_map = VectorObservationMap(
        matrix=cast(dict[str, Tensor], load(base, weights_only=True))["matrix"], dimension=6
    )
    assert fresh.states is not None
    for index in range(len(fresh)):
        expected = network.forward(base_map.forward(fresh.states[index]).float()).float()
        assert torch.equal(expected, fresh.tensors["observations"][index])
    assert all(
        path.parent != source or path.name not in {"test.pt", "test_states.pt"} for path in accessed
    )


def test_invalid_source_matrix_fails_before_output_written(tmp_path: Path) -> None:
    source, base = _source(tmp_path)
    wrong = VectorObservationMap(seed=1730, dimension=6).matrix
    source_data.save_torch(base, {"matrix": wrong})
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="original dataset identity"):
        data.prepare(data.PrepareConfig(str(output), str(source), str(base), profile="smoke"))
    assert not output.exists()


def test_persisted_base_map_tolerates_changed_platform_qr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, base = _source(tmp_path)
    original_qr = cast(
        Callable[[Tensor], tuple[Tensor, Tensor]],
        torch.linalg.qr,  # pyright: ignore[reportUnknownMemberType]
    )

    def perturbed_qr(value: Tensor) -> tuple[Tensor, Tensor]:
        q, r = original_qr(value)
        q = q.clone()
        q[0, 0] = torch.nextafter(q[0, 0], q.new_tensor(float("inf")))
        return q, r

    monkeypatch.setattr(torch.linalg, "qr", perturbed_qr)
    output = tmp_path / "new"
    data.prepare(data.PrepareConfig(str(output), str(source), str(base), profile="smoke"))
    assert source_data.sha(output / "observation_map.pt") == source_data.sha(base)
    data.load_data(output)


def test_cache_resume_reuses_exact_neural_weights_without_new_qr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, base, output = _prepared(tmp_path)
    before = source_data.sha(output / "neural_observation_map.pt")
    old_init = observation.FrozenObservationMap.__init__

    def forbid_generation(
        self: observation.FrozenObservationMap, matrices: Tensor | None = None
    ) -> None:
        if matrices is None:
            raise AssertionError("Resume must restore persisted neural matrices")
        old_init(self, matrices)

    monkeypatch.setattr(observation.FrozenObservationMap, "__init__", forbid_generation)
    config = data.PrepareConfig(str(output), str(source), str(base), profile="smoke")
    completed = data.prepare(config, resume=True)
    assert source_data.sha(output / "neural_observation_map.pt") == before
    assert completed == source_data.read_json(output / "completion.json")
    with pytest.raises(FileExistsError):
        data.prepare(config)


def test_changed_source_artifact_and_missing_required_cache_hash_are_rejected(
    tmp_path: Path,
) -> None:
    source, base, output = _prepared(tmp_path)
    complete = source_data.read_json(output / "completion.json")
    altered = copy.deepcopy(complete)
    del altered["hashes"]["test_states.pt"]
    altered.pop("completion_fingerprint")
    altered["completion_fingerprint"] = canonical_fingerprint(altered)
    source_data.write_json(output / "completion.json", altered)
    with pytest.raises(ValueError, match="required artifact hashes"):
        data.load_data(output)
    source_data.write_json(output / "completion.json", complete)
    with (source / "test_states.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        data.prepare(
            data.PrepareConfig(str(output), str(source), str(base), profile="smoke"), resume=True
        )


def test_candidate_and_map_settings_cannot_be_relabelled(tmp_path: Path) -> None:
    _, _, output = _prepared(tmp_path)
    protocol = source_data.read_json(output / "protocol.json")
    complete = source_data.read_json(output / "completion.json")
    protocol["settings"]["observation_map"]["alpha"] = 0.3
    protocol.pop("identity")
    protocol["identity"] = canonical_fingerprint(protocol)
    source_data.write_json(output / "protocol.json", protocol)
    complete.update(protocol)
    complete["hashes"]["protocol.json"] = source_data.sha(output / "protocol.json")
    complete.pop("completion_fingerprint")
    complete["completion_fingerprint"] = canonical_fingerprint(complete)
    source_data.write_json(output / "completion.json", complete)
    with pytest.raises(ValueError, match="scientific settings changed"):
        data.load_data(output)


def test_source_identity_includes_only_new_numeric_files_and_parent_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_before = source_data.code_identity()
    before = data.code_identity()
    read_bytes = Path.read_bytes

    def changed_read(path: Path) -> bytes:
        value = read_bytes(path)
        if path == Path(observation.__file__):
            return value + b"\n# simulated map change\n"
        return value

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    assert data.code_identity() != before
    assert source_data.code_identity() == parent_before


def test_map_and_cache_generation_preserve_exact_fresh_encoder_initialization(
    tmp_path: Path,
) -> None:
    initial = build_control("dense", 0).state_dict()
    _prepared(tmp_path)
    paired = build_control("dense", 0).state_dict()
    assert initial.keys() == paired.keys()
    assert all(
        torch.equal(value, paired[name]) if isinstance(value, Tensor) else value == paired[name]
        for name, value in initial.items()
    )
    assert tensor_digest(initial["online_encoder.head.weight"]) == tensor_digest(
        paired["online_encoder.head.weight"]
    )
