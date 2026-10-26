import copy
import json
import sys
from collections.abc import Generator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.multiquery_control.data import EvaluationData, module_fingerprint
from diagnostics.multiquery_control.run import frozen_hashes, sha
from diagnostics.signal_path import run as runner
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.models.model import SCJEPA
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.loop import ObjectiveConfig


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2))


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "finished"
    root.mkdir()
    model_config = ModelConfig(
        observation_dim=6,
        encoder_hidden_dims=(8,),
        state_dim=6,
        param_dim=6,
        action_dim=16,
        context_len=10,
        history_len=1,
        parameter_encoder_kind="transition",
        spartan_layers=1,
        spartan_embed_dim=4,
        spartan_mlp_hidden=8,
        spartan_mlp_layers=2,
        spartan_dense=True,
        autoregressive=True,
    )
    model = build_model(model_config)
    model.online_encoder.requires_grad_(False)
    model.target_encoder.requires_grad_(False)
    frozen = frozen_hashes(model)
    observation = VectorObservationMap(dimension=6)
    split = {
        "theta": torch.zeros(4, 6, dtype=torch.float64),
        "states_online": torch.zeros(4, 10, 12, 6),
        "states_target": torch.zeros(4, 10, 12, 6),
        "actions": torch.zeros(4, 10, 11, 16),
        "true_states": torch.zeros(4, 10, 12, 6, dtype=torch.float64),
        "true_actions": torch.zeros(4, 10, 11, 16, dtype=torch.float64),
    }
    evaluation: EvaluationData = {
        "metadata": {
            "generator_version": dgp.GENERATOR_VERSION,
            "observation_map_fingerprint": observation.fingerprint,
            "observation_matrix": observation.matrix,
            "state_online_sha256": module_fingerprint(model.online_encoder),
            "state_target_sha256": module_fingerprint(model.target_encoder),
        },
        "calibration": split,
        "validation": copy.deepcopy(split),
        "test": copy.deepcopy(split),
    }
    torch.save(evaluation, root / "evaluation_data.pt")
    objective = ObjectiveConfig(
        state_regularizer="sigreg",
        parameter_regularizer="sigreg",
        state_regularizer_weight=0.4,
        parameter_regularizer_weight=0.4,
        parameter_consistency_weight=1.0,
        regularizer_num_projections=8,
        logit_weight=0.0001,
    )
    protocol: dict[str, Any] = {
        "identity": "source-identity",
        "code_identity": runner.training_code_identity(),
        "config": {"seeds": [7], "prediction_weights": [1, 4, 16], "batch_size": 2, "steps": 2},
        "model_config": asdict(model_config),
        "objective_config": asdict(objective),
        "source": {
            "sha256": "source-checkpoint",
            "observation_map_fingerprint": observation.fingerprint,
        },
        "frozen_hashes": frozen,
        "initializations": {"7": {"sha256": "paired-initialization"}},
        "artifacts": {"evaluation_data.pt": sha(root / "evaluation_data.pt")},
    }
    write_json(root / "protocol.json", protocol)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    for weight in (1, 4, 16):
        condition = f"prediction_{weight}"
        dest = root / "seed-7" / condition
        dest.mkdir(parents=True)
        torch.save(
            {
                "identity": protocol["identity"],
                "condition": condition,
                "prediction_weight": weight,
                "seed": 7,
                "step": 2,
                "source_sha256": protocol["source"]["sha256"],
                "initial_sha256": protocol["initializations"]["7"]["sha256"],
                "parameter_regularizer_weight": 0.4,
                "consistency_weight": 1.0,
                "queries": 1,
                "frozen_hashes": frozen,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            dest / "last.pt",
        )
        write_json(dest / "gradient_history.json", {"history": []})
        write_json(
            dest / "results.json",
            {
                "identity": protocol["identity"],
                "condition": condition,
                "prediction_weight": weight,
                "seed": 7,
                "steps": 2,
                "checkpoint_sha256": sha(dest / "last.pt"),
                "metrics": {
                    "prediction_mse": 0.2,
                    "linear_r2": [0.0] * 6,
                    "parameter_readouts": {"selected_test_r2": [0.0] * 6},
                    "true_parameter_interventions": {"by_parameter": {}},
                },
            },
        )
        write_json(
            dest / "completed.json",
            {
                "identity": protocol["identity"],
                "checkpoint_sha256": sha(dest / "last.pt"),
                "results_sha256": sha(dest / "results.json"),
                "gradient_history_sha256": sha(dest / "gradient_history.json"),
            },
        )
    return root


def config(source: Path, output: Path) -> runner.Config:
    return runner.Config(
        str(source),
        str(output),
        seed=7,
        groups=2,
        donor_contexts=1,
        fit_queries=1,
        test_queries=1,
        gradient_groups=2,
        batch_size=2,
    )


def test_cli_selects_finished_checkpoints_without_running(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[runner.Config, bool]] = []

    def capture(settings: runner.Config, *, resume: bool = False) -> dict[str, Any]:
        calls.append((settings, resume))
        return {}

    monkeypatch.setattr(runner, "run", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--experiment-dir",
            "finished",
            "--output-dir",
            "diagnosis",
            "--device",
            "cuda:0",
            "--resume",
        ],
    )
    runner.main()
    assert len(calls) == 1
    settings, resume = calls[0]
    assert resume
    assert settings.prediction_weights == (1.0, 4.0, 16.0)
    assert settings.groups == settings.gradient_groups == 64
    assert settings.donor_contexts == 2
    assert settings.fit_queries == 4
    assert settings.test_queries == 5
    assert settings.device == "cuda:0"
    assert Path(settings.experiment_dir).is_absolute()


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_size": 1},
        {"gradient_groups": 3},
        {"fit_queries": 5, "test_queries": 5},
        {"donor_contexts": 11},
        {"rcond": 0.0},
        {"rcond": float("nan")},
        {"displacement": 0.0},
        {"prediction_weights": (1.0, 1.0)},
        {"prediction_weights": (2.0,)},
    ],
)
def test_config_rejects_invalid_diagnostic_settings(
    source: Path, tmp_path: Path, changes: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="must|Use at most|Prediction weight|prediction_weights"):
        replace(config(source, tmp_path / "diagnosis"), **changes)


def test_output_cannot_overwrite_or_nest_inside_source(source: Path) -> None:
    for output in (source, source / "nested"):
        with pytest.raises(ValueError, match="outside"):
            config(source, output)


def test_source_validation_loads_frozen_model_before_float64_conversion(
    source: Path, tmp_path: Path
) -> None:
    settings = config(source, tmp_path / "diagnosis")
    protocol, data, sources = runner.source_records(settings)
    assert len(data["test"]["theta"]) == 4
    assert set(sources) == {"prediction_1", "prediction_4", "prediction_16"}
    model, payload, prior = runner.load_checkpoint(
        settings, protocol, 4.0, torch.device("cpu"), data["metadata"]
    )
    assert not model.training
    assert next(model.parameters()).dtype == torch.float64
    assert not any(p.requires_grad for p in model.online_encoder.parameters())
    assert not any(p.requires_grad for p in model.target_encoder.parameters())
    assert all(p.requires_grad for p in model.parameter_encoder.parameters())
    assert payload["prediction_weight"] == 4.0
    assert prior["metrics"]["prediction_mse"] == 0.2


@pytest.mark.parametrize("filename", ["last.pt", "results.json", "gradient_history.json"])
def test_changed_finished_artifact_is_rejected(source: Path, tmp_path: Path, filename: str) -> None:
    path = source / "seed-7/prediction_4" / filename
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        runner.source_records(config(source, tmp_path / "diagnosis"))


def test_changed_evaluation_cache_is_rejected(source: Path, tmp_path: Path) -> None:
    path = source / "evaluation_data.pt"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="cache hash"):
        runner.source_records(config(source, tmp_path / "diagnosis"))


def test_changed_training_code_identity_is_rejected(source: Path, tmp_path: Path) -> None:
    protocol = read_json(source / "protocol.json")
    protocol["code_identity"] = "different-training-implementation"
    write_json(source / "protocol.json", protocol)
    with pytest.raises(ValueError, match="code"):
        runner.source_records(config(source, tmp_path / "diagnosis"))


def test_incompatible_frozen_metadata_is_rejected_even_with_updated_cache_hash(
    source: Path, tmp_path: Path
) -> None:
    path = source / "evaluation_data.pt"
    data = cast(dict[str, Any], torch.load(path, weights_only=True))
    data["metadata"]["state_online_sha256"] = "different-state-encoder"
    torch.save(data, path)
    protocol = read_json(source / "protocol.json")
    protocol["artifacts"]["evaluation_data.pt"] = sha(path)
    write_json(source / "protocol.json", protocol)
    settings = config(source, tmp_path / "diagnosis")
    protocol, data, _ = runner.source_records(settings)
    with pytest.raises(ValueError, match="metadata|encoder"):
        runner.load_checkpoint(settings, protocol, 1.0, torch.device("cpu"), data["metadata"])


def test_loading_nonfinal_checkpoint_is_rejected(source: Path, tmp_path: Path) -> None:
    settings = config(source, tmp_path / "diagnosis")
    protocol, data, _ = runner.source_records(settings)
    path = source / "seed-7/prediction_1/last.pt"
    payload = cast(dict[str, Any], torch.load(path, weights_only=False))
    payload["step"] = 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="provenance"):
        runner.load_checkpoint(settings, protocol, 1.0, torch.device("cpu"), data["metadata"])


@pytest.mark.parametrize("changes", [{"autoregressive": False}, {"context_len": 9}])
def test_checkpoint_prediction_flags_are_validated_after_state_load(
    source: Path, tmp_path: Path, changes: dict[str, Any]
) -> None:
    settings = config(source, tmp_path / "diagnosis")
    protocol, data, _ = runner.source_records(settings)
    path = source / "seed-7/prediction_1/last.pt"
    payload = cast(dict[str, Any], torch.load(path, weights_only=False))
    payload["model"]["_extra_state"].update(changes)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="original-MLP"):
        runner.load_checkpoint(settings, protocol, 1.0, torch.device("cpu"), data["metadata"])


def test_runner_is_read_only_restores_rng_and_verifies_completed_resume(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def geometry(
        model: SCJEPA, _data: EvaluationData, **_kwargs: object
    ) -> tuple[dict[str, Any], dict[str, Tensor]]:
        assert not any(p.requires_grad for p in model.parameters())
        calls.append("geometry")
        torch.randn(3)
        return {"kind": "geometry"}, {"A": torch.ones(2, 6, 6)}

    def gradients(
        model: SCJEPA,
        _data: EvaluationData,
        objective: ObjectiveConfig,
        weight: float,
        **kwargs: object,
    ) -> tuple[dict[str, Any], dict[str, Tensor]]:
        assert all(p.requires_grad for p in model.parameter_encoder.parameters())
        assert objective.parameter_regularizer_weight == 0.4
        assert objective.parameter_consistency_weight == 1.0
        assert "optimizer_state" in kwargs
        calls.append(f"gradients_{weight:g}")
        torch.randn(4)
        return {"kind": "gradients"}, {"norms": torch.ones(6)}

    monkeypatch.setattr(runner, "geometry_diagnostic", geometry)
    monkeypatch.setattr(runner, "gradient_path_diagnostic", gradients)
    settings = config(source, tmp_path / "diagnosis")
    before_files = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    rng = torch.get_rng_state()
    result = runner.run(settings)
    assert torch.equal(torch.get_rng_state(), rng)
    assert calls == [
        "geometry",
        "gradients_1",
        "geometry",
        "gradients_4",
        "geometry",
        "gradients_16",
    ]
    assert len(result["results"]) == 3
    after_files = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    assert after_files == before_files
    assert runner.run(settings, resume=True) == result
    assert len(calls) == 6
    with pytest.raises(FileExistsError, match="not empty"):
        runner.run(settings)
    with pytest.raises(ValueError, match="settings, code or source changed"):
        runner.run(replace(settings, rcond=1e-5), resume=True)
    tensor_path = Path(settings.output_dir) / "seed-7/prediction_1/signal_tensors.pt"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="diagnostic changed"):
        runner.run(settings, resume=True)


def test_mutating_diagnostic_is_detected(
    monkeypatch: pytest.MonkeyPatch, source: Path, tmp_path: Path
) -> None:
    def geometry(
        model: SCJEPA, _data: EvaluationData, **_kwargs: object
    ) -> tuple[dict[str, Any], dict[str, Tensor]]:
        with torch.no_grad():
            next(model.parameters()).add_(1)
        return {}, {}

    monkeypatch.setattr(runner, "geometry_diagnostic", geometry)
    with pytest.raises(RuntimeError, match="Geometry evaluation mutated"):
        runner.run(replace(config(source, tmp_path / "diagnosis"), prediction_weights=(1.0,)))
