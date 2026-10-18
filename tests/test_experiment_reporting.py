import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments import storage
from scjepa.experiments.config import ExperimentConfig
from scjepa.experiments.reporting import aggregate_results
from scjepa.experiments.storage import (
    claim_execution,
    file_lock,
    prepare_experiment,
    read_json,
    write_json,
)


def _configuration(path: Path) -> ExperimentConfig:
    return ExperimentConfig(output_dir=str(path), graphs=("complementary",), seeds=(0, 1))


def _record(
    root: Path,
    seed: int,
    condition: str,
    value: float,
    *,
    identity_changes: dict[str, object] | None = None,
) -> Path:
    directory = root / f"complementary/seed-{seed}/{condition}"
    record: dict[str, object] = {
        "initial_state_fingerprint": f"initial-{seed}",
        "dataset_identity": "paired-dataset",
        "experiment_identity": "same-experiment",
        "world_size": 1,
        "run_identity": f"{condition}-{seed}",
        "tau_calibration": None
        if condition == "dense"
        else {"tau": 0.5, "dense_run_identity": f"dense-{seed}"},
    }
    record.update(identity_changes or {})
    write_json(directory / "run_config.json", record)
    write_json(directory / "evaluation/metrics.json", {"recovery": value})
    write_json(directory / "status.json", {"status": "completed"})
    return directory


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def test_paired_aggregation_uses_full_minus_baseline_and_sample_uncertainty(tmp_path: Path) -> None:
    config = _configuration(tmp_path)
    for seed, baseline, full in ((0, 1.0, 4.0), (1, 3.0, 8.0)):
        _record(tmp_path, seed, "dense", baseline)
        _record(tmp_path, seed, "baseline", baseline)
        _record(tmp_path, seed, "full", full)
    result = aggregate_results(config)
    paired = _mapping(_mapping(result["paired_full_minus_baseline"])["complementary"])
    scores = _mapping(paired["recovery"])
    assert scores["n"] == 2
    assert scores["mean"] == 4.0
    assert math.isclose(cast(float, scores["std"]), math.sqrt(2))
    assert math.isclose(cast(float, scores["standard_error"]), 1.0)
    assert result["complete"] is True
    assert result["missing_runs"] == []
    assert result["completed_runs"] == result["expected_runs"] == 6
    assert read_json(tmp_path / "summary.json") == result
    csv_rows = (tmp_path / "results.csv").read_text().splitlines()
    assert len(csv_rows) == 7


def test_missing_runs_and_single_seed_uncertainty_are_explicit(tmp_path: Path) -> None:
    config = _configuration(tmp_path)
    _record(tmp_path, 0, "dense", 0.0)
    _record(tmp_path, 0, "baseline", 1.0)
    _record(tmp_path, 0, "full", 4.0)
    unfinished = _record(tmp_path, 1, "baseline", 2.0)
    write_json(unfinished / "status.json", {"status": "running"})
    result = aggregate_results(config)
    assert result["complete"] is False
    assert result["completed_runs"] == 3
    assert result["expected_runs"] == 6
    assert result["missing_runs"] == [
        "complementary/seed-1/dense",
        "complementary/seed-1/baseline",
        "complementary/seed-1/full",
    ]
    assert result["missing_pairs"] == ["complementary/seed-1"]
    paired = _mapping(_mapping(result["paired_full_minus_baseline"])["complementary"])
    assert paired["recovery"] == {"n": 1, "mean": 3.0, "std": None, "standard_error": None}


@pytest.mark.parametrize("processes", [None, 0, True, 2])
def test_aggregation_rejects_changed_or_missing_world_size(
    tmp_path: Path, processes: object
) -> None:
    _record(tmp_path, 0, "baseline", 1.0)
    _record(tmp_path, 0, "full", 2.0, identity_changes={"world_size": processes})
    with pytest.raises(ValueError, match="world_size"):
        aggregate_results(_configuration(tmp_path))


def test_execution_claim_rejects_world_size_change_before_new_condition(tmp_path: Path) -> None:
    config = _configuration(tmp_path)
    claim_execution(config, "experiment-a", 1)
    claim_execution(config, "experiment-a", 1)
    with pytest.raises(ValueError, match="world_size"):
        claim_execution(config, "experiment-a", 2)
    with pytest.raises(ValueError, match="identity"):
        claim_execution(config, "experiment-b", 1)
    assert read_json(tmp_path / "execution.json") == {
        "experiment_identity": "experiment-a",
        "world_size": 1,
    }


def test_execution_claim_checks_existing_runs_if_ownership_record_is_missing(
    tmp_path: Path,
) -> None:
    _record(tmp_path, 0, "baseline", 1.0)
    with pytest.raises(ValueError, match="world_size"):
        claim_execution(_configuration(tmp_path), "same-experiment", 2)


@pytest.mark.parametrize(
    "identity", ["initial_state_fingerprint", "dataset_identity", "experiment_identity"]
)
def test_unmatched_pairs_are_rejected(tmp_path: Path, identity: str) -> None:
    _record(tmp_path, 0, "baseline", 1.0)
    _record(tmp_path, 0, "full", 2.0, identity_changes={identity: "different"})
    with pytest.raises(ValueError, match=f"unmatched {identity}"):
        aggregate_results(_configuration(tmp_path))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), "0.25", {"mean": 0.5}])
def test_nonfinite_or_nonscalar_metrics_are_rejected(tmp_path: Path, invalid: object) -> None:
    directory = _record(tmp_path, 0, "baseline", 1.0)
    (directory / "evaluation/metrics.json").write_text(json.dumps({"recovery": invalid}))
    with pytest.raises(ValueError, match="nonfinite or nonscalar"):
        aggregate_results(_configuration(tmp_path))


def test_paired_configs_only_change_parameter_penalty_and_enforce_sigreg(tmp_path: Path) -> None:
    config = _configuration(tmp_path)
    config.validate()
    baseline = asdict(config.training_config("baseline", 3, tmp_path / "run", tau=0.5))
    full = asdict(config.training_config("full", 3, tmp_path / "run", tau=0.5))
    baseline_objective, full_objective = (
        _mapping(baseline["objective"]),
        _mapping(full["objective"]),
    )
    assert baseline_objective.pop("parameter_sparsity_weight") == 0.0
    assert full_objective.pop("parameter_sparsity_weight") == 1.0
    assert baseline == full
    assert baseline_objective["state_regularizer"] == "sigreg"
    assert baseline_objective["parameter_regularizer"] == "sigreg"
    config.train.objective.state_regularizer = "visreg"
    with pytest.raises(ValueError, match="(?i)sigreg"):
        config.validate()
    config.train.objective.state_regularizer = "sigreg"
    config.train.objective.parameter_regularizer_weight = 0.0
    with pytest.raises(ValueError, match="(?i)sigreg"):
        config.validate()


def test_manifest_restores_actual_matrix_without_regenerating_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage, "source_fingerprint", lambda: "stable-source")
    config = _configuration(tmp_path)
    original, matrix = prepare_experiment(config)
    before_manifest = (tmp_path / "experiment.json").read_bytes()
    before_matrix = (tmp_path / "observation_map.pt").read_bytes()

    def require_saved(*, seed: int = 1729, matrix: Tensor | None = None) -> VectorObservationMap:
        assert matrix is not None, "resumption must restore the saved realization of W"
        return VectorObservationMap(seed=seed, matrix=matrix)

    monkeypatch.setattr(storage, "VectorObservationMap", require_saved)
    restored, restored_matrix = prepare_experiment(config)
    assert restored["identity"] == original["identity"]
    assert torch.equal(restored_matrix, matrix)
    assert (tmp_path / "experiment.json").read_bytes() == before_manifest
    assert (tmp_path / "observation_map.pt").read_bytes() == before_matrix


def test_manifest_rejects_matrix_config_or_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage, "source_fingerprint", lambda: "stable-source")
    config = _configuration(tmp_path)
    _, matrix = prepare_experiment(config)
    torch.save({"matrix": matrix.roll(1, dims=0)}, tmp_path / "observation_map.pt")
    with pytest.raises(ValueError, match="observation map changed"):
        prepare_experiment(config)
    torch.save({"matrix": matrix}, tmp_path / "observation_map.pt")
    with pytest.raises(ValueError, match="definition/source changed"):
        prepare_experiment(replace(config, data_seed=1))
    monkeypatch.setattr(storage, "source_fingerprint", lambda: "changed-source")
    with pytest.raises(ValueError, match="definition/source changed"):
        prepare_experiment(config)


def test_locks_release_and_json_failure_preserves_previous_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / ".run.lock"
    with (
        file_lock(lock),
        pytest.raises(RuntimeError, match="another process"),
        file_lock(lock),
    ):
        pytest.fail("a second open lock must not claim an active run")
    # Persistent lock files are harmless after the OS lock is released.
    with file_lock(lock):
        assert lock.exists()
    target = tmp_path / "record.json"
    write_json(target, {"complete": True})

    def failed_replace(self: Path, destination: Path) -> Path:
        assert self.name.endswith(".tmp")
        assert destination == target
        raise OSError("simulated interrupted publish")

    monkeypatch.setattr(Path, "replace", failed_replace)
    with pytest.raises(OSError, match="interrupted publish"):
        write_json(target, {"complete": False})
    assert read_json(target) == {"complete": True}
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "identity", ["initial_state_fingerprint", "dataset_identity", "experiment_identity"]
)
def test_missing_identity_in_both_conditions_cannot_count_as_a_pair(
    tmp_path: Path, identity: str
) -> None:
    for condition in ("baseline", "full"):
        directory = _record(tmp_path, 0, condition, 1.0)
        record = read_json(directory / "run_config.json")
        record.pop(identity)
        write_json(directory / "run_config.json", record)
    with pytest.raises(ValueError, match=f"unmatched {identity}"):
        aggregate_results(_configuration(tmp_path))


@pytest.mark.parametrize("invalid", [None, "", 123])
def test_paired_identity_requires_nonempty_strings(tmp_path: Path, invalid: object) -> None:
    for condition in ("baseline", "full"):
        _record(tmp_path, 0, condition, 1.0, identity_changes={"dataset_identity": invalid})
    with pytest.raises(ValueError, match="unmatched dataset_identity"):
        aggregate_results(_configuration(tmp_path))


@pytest.mark.parametrize("field", ["identity", "definition", "torch_version"])
def test_manifest_integrity_rejects_tampered_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    monkeypatch.setattr(storage, "source_fingerprint", lambda: "stable-source")
    config = _configuration(tmp_path)
    prepare_experiment(config)
    manifest = read_json(tmp_path / "experiment.json")
    if field == "definition":
        _mapping(manifest[field])["name"] = "changed-scientific-definition"
    else:
        manifest[field] = "modified"
    write_json(tmp_path / "experiment.json", manifest)
    with pytest.raises(ValueError, match="manifest integrity"):
        prepare_experiment(config)


def test_manifest_definition_must_match_even_with_recomputed_manifest_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage, "source_fingerprint", lambda: "stable-source")
    config = _configuration(tmp_path)
    prepare_experiment(config)
    manifest = read_json(tmp_path / "experiment.json")
    _mapping(manifest["definition"])["data_seed"] = 123
    manifest.pop("identity")
    manifest["identity"] = canonical_fingerprint(manifest)
    write_json(tmp_path / "experiment.json", manifest)
    with pytest.raises(ValueError, match="manifest integrity"):
        prepare_experiment(config)


@pytest.mark.parametrize("field", ["torch_version", "python_version"])
def test_exact_resume_rejects_changed_runtime_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    monkeypatch.setattr(storage, "source_fingerprint", lambda: "stable-source")
    config = _configuration(tmp_path)
    prepare_experiment(config)
    manifest = read_json(tmp_path / "experiment.json")
    manifest[field] = "different-runtime"
    manifest.pop("identity")
    manifest["identity"] = canonical_fingerprint(manifest)
    write_json(tmp_path / "experiment.json", manifest)
    with pytest.raises(ValueError, match="same Python/PyTorch versions"):
        prepare_experiment(config)


def test_dense_reference_provenance_and_sparse_tau_must_match(tmp_path: Path) -> None:
    config = _configuration(tmp_path)
    dense = _record(tmp_path, 0, "dense", 1.0)
    _record(tmp_path, 0, "baseline", 1.0)
    full = _record(tmp_path, 0, "full", 1.0)
    record = read_json(dense / "run_config.json")
    record["initial_state_fingerprint"] = "dense-random-initialization"
    write_json(dense / "run_config.json", record)
    assert aggregate_results(config)["completed_runs"] == 3
    for key in ("dataset_identity", "experiment_identity"):
        write_json(dense / "run_config.json", {**record, key: "different"})
        with pytest.raises(ValueError, match=f"unmatched {key}"):
            aggregate_results(config)
        write_json(dense / "run_config.json", record)
    record = read_json(full / "run_config.json")
    record["tau_calibration"] = {"tau": 0.6, "dense_run_identity": "dense-0"}
    write_json(full / "run_config.json", record)
    with pytest.raises(ValueError, match="unmatched tau_calibration"):
        aggregate_results(config)
