"""Read-only paired controls authenticate endpoints without any training fixture."""

import math
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.balanced_action_oracle import paired_encoding_analysis as paired
from diagnostics.balanced_action_oracle import parameter_use
from diagnostics.balanced_action_oracle.data import StateData, code_identity
from diagnostics.balanced_action_oracle.run import SCHEMA, TrainConfig
from diagnostics.isotropic_oracle.model import build_control
from diagnostics.oracle_state_learning.core import EvaluationResult
from diagnostics.oracle_state_learning.data import read_json, save_torch, sha
from diagnostics.oracle_state_learning.run import recovery_metrics
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.experiments.storage import write_json


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _finish(directory: Path, complete: dict[str, Any]) -> None:
    complete["hashes"] = {
        path.name: sha(path)
        for path in directory.iterdir()
        if path.is_file() and path.name != "completion.json"
    }
    write_json(directory / "completion.json", complete)


def _health(values: Tensor, prefix: str) -> dict[str, float]:
    centered = values.double().reshape(-1, 6)
    centered = centered - centered.mean(0)
    covariance = centered.T @ centered / len(centered)
    eigenvalues = cast(Tensor, torch.linalg.eigvalsh(covariance))  # pyright: ignore[reportUnknownMemberType]
    probabilities = eigenvalues / eigenvalues.sum()
    return {
        f"{prefix}/variance_min": float(covariance.diagonal().min()),
        f"{prefix}/covariance_eigenvalue_min": float(eigenvalues.min()),
        f"{prefix}/effective_rank": float((-probabilities * probabilities.log()).sum().exp()),
    }


def _synthetic_endpoint(
    directory: Path,
    *,
    balanced: bool,
    truth_offset: float = 0.0,
    group_offset: int = 0,
    initial_offset: float = 0.0,
    actions: int = 24,
) -> None:
    """Write authenticated synthetic smoke artifacts; no model is trained or called."""
    directory.mkdir()
    initial_state = {
        "online_encoder.weight": torch.eye(6) + initial_offset,
        "predictor.weight": torch.ones(6, 36),
    }
    identities = {
        "initial_encoder_identity": canonical_fingerprint(
            {"weight": tensor_fingerprint(initial_state["online_encoder.weight"])}
        ),
        "initial_predictor_identity": canonical_fingerprint(
            {"weight": tensor_fingerprint(initial_state["predictor.weight"])}
        ),
    }
    save_torch(directory / "initial.pt", {"model": initial_state, **identities})
    identity = "synthetic-balanced" if balanced else "synthetic-isotropic"
    save_torch(
        directory / "last.pt",
        {
            "identity": identity,
            "step": 2,
            "initial_checkpoint_sha256": sha(directory / "initial.pt"),
            "optimizer": {"state": {0: {"step": torch.tensor(2)}}},
        },
    )
    settings: dict[str, Any] = {
        "candidate": "balanced_action_innovation" if balanced else "isotropic_innovation",
        "transport_chart": "isotropic",
        "rho": 0.7,
        "rotation_scale": 0.23865401979977377,
        "seed": 0,
        "profile": "smoke",
        "train_groups": 4,
        "calibration_groups": 4,
        "validation_groups": 4,
        "test_groups": 4,
    }
    if balanced:
        settings["action_mix"] = "paley24"
    record = {
        "identity": identity,
        "code_identity": "archived-balanced-source" if balanced else "archived-isotropic-source",
        "data_identity": "balanced-cache" if balanced else "isotropic-cache",
        "candidate_settings": settings,
        "scientific_config": {"condition": "dense", "seed": 0, "steps": 2, "profile": "smoke"},
        "runtime": {"gpu_name": "fixture2080" if balanced else "fixture3090"},
        **identities,
    }
    write_json(directory / "config.json", record)
    generator = torch.Generator().manual_seed(12)
    angle = 0.6 if balanced else 0.2
    transform = torch.eye(6)
    for start in (0, 2, 4):
        transform[start : start + 2, start : start + 2] = torch.tensor(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
    evaluations: dict[str, EvaluationResult] = {}
    for index, split in enumerate(("calibration", "validation")):
        truth = torch.randn(4, 10, 12, 6, generator=generator) + truth_offset
        online = truth @ transform
        target = online * 1.01
        samples = {
            "states": truth,
            "online_states": online,
            "target_states": target,
            "predictions": target[:, :, 10:12] + 0.01,
            "targets": target[:, :, 10:12],
            "group_ids": torch.arange(4) + index * 100 + group_offset,
            "graph_action": torch.ones(4, 6, actions),
        }
        prediction = float(
            (samples["predictions"].double() - samples["targets"].double()).square().mean()
        )
        variance = float(samples["targets"].double().flatten(0, 2).var(0, correction=0).mean())
        metrics = {
            "prediction": prediction,
            "forecast/normalized_mse": prediction / variance,
            **_health(online, "online"),
            **_health(target, "target"),
        }
        evaluations[split] = EvaluationResult(metrics, samples)
    for split, evaluation in evaluations.items():
        recovery, matching = recovery_metrics(evaluations["calibration"], evaluation)
        evaluation.metrics.update(recovery)
        payload = {"metrics": evaluation.metrics, "samples": evaluation.samples}
        save_torch(directory / f"{split}.pt", payload)
        write_json(directory / f"{split}.json", {"metrics": evaluation.metrics})
        write_json(directory / "matching.json", matching)
    write_json(directory / "startup.json", identities)
    write_json(directory / "wandb.json", {"mode": "disabled"})
    (directory / "history.jsonl").write_text('{"step":2}\n')
    _finish(
        directory,
        {
            "schema": SCHEMA if balanced else "isotropic-oracle-state-learning-v1",
            "kind": "train",
            "condition": "dense",
            "seed": 0,
            "profile": "smoke",
            "steps": 2,
            "successful_updates": 2,
            "skipped_updates": 0,
            "identity": identity,
            "code_identity": record["code_identity"],
            "data_identity": record["data_identity"],
            **identities,
        },
    )


def test_paired_saved_samples_recompute_scores_and_preserve_every_source_file(
    tmp_path: Path,
) -> None:
    original, balanced = tmp_path / "original", tmp_path / "balanced"
    _synthetic_endpoint(original, balanced=False)
    _synthetic_endpoint(balanced, balanced=True)
    hashes = {path: sha(path) for root in (original, balanced) for path in root.iterdir()}
    with pytest.raises(ValueError, match="100000-update"):
        paired.compare(original, balanced, tmp_path / "not-scientific")
    result = paired.compare(original, balanced, tmp_path / "report", draws=20, allow_smoke=True)
    assert result["paired_physical_samples"]
    assert result["paired_initial_encoder_predictor"]
    assert result["different_training_gpu"]
    assert len(result["comparisons"]) == 2
    for interval in result["comparisons"]:
        assert interval["upper_95"] < 0
    for arm in result["arms"].values():
        assert arm["metrics"]["online_state_probe/r2"] > 0.999
        assert arm["metrics"]["normalized_prediction_mse"] < 0.001
        assert len(arm["artifact_sha256"]) == 11
    assert all(sha(path) == digest for path, digest in hashes.items())
    assert not any((root / "test.pt").exists() for root in (original, balanced))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"truth_offset": 0.125}, "states are not bitwise paired"),
        ({"group_offset": 10}, "group_ids are not bitwise paired"),
        ({"initial_offset": 0.1}, "Initial encoder/predictor tensors are not paired"),
        ({"actions": 22}, "Malformed local routing"),
    ],
)
def test_pairing_and24_action_interface_reject_valid_but_different_artifacts(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    _synthetic_endpoint(tmp_path / "original", balanced=False)
    _synthetic_endpoint(tmp_path / "balanced", balanced=True, **change)
    with pytest.raises(ValueError, match=message):
        paired.compare(
            tmp_path / "original", tmp_path / "balanced", tmp_path / "report", allow_smoke=True
        )


def test_wrong_namespace_and_configuration_identity_are_rejected(tmp_path: Path) -> None:
    _synthetic_endpoint(tmp_path / "original", balanced=False)
    _synthetic_endpoint(tmp_path / "balanced", balanced=True)
    with pytest.raises(ValueError, match="schema"):
        paired.compare(
            tmp_path / "balanced", tmp_path / "original", tmp_path / "wrong", allow_smoke=True
        )
    record = read_json(tmp_path / "balanced/config.json")
    record["code_identity"] = "mismatched-record"
    write_json(tmp_path / "balanced/config.json", record)
    _finish(tmp_path / "balanced", read_json(tmp_path / "balanced/completion.json"))
    with pytest.raises(ValueError, match="identities disagree"):
        paired.compare(
            tmp_path / "original", tmp_path / "balanced", tmp_path / "wrong", allow_smoke=True
        )


def _validation_fixture() -> StateData:
    generator = torch.Generator().manual_seed(41)
    tensors = {
        "observations": torch.randn(4, 10, 12, 6, generator=generator),
        "actions": torch.randn(4, 10, 11, 24, generator=generator),
        "oracle_parameters": torch.rand(4, 6, generator=generator) - 0.5,
        "episode_ids": torch.arange(4),
    }
    return StateData("validation", tensors, None, {})


def test_theta_measure_preserves_model_inputs_modes_and_rng_with24_actions() -> None:
    data, model = _validation_fixture(), build_control("mlp", 0)
    before, inputs = deepcopy(model.state_dict()), deepcopy(data.tensors)
    modes, rng = [module.training for module in model.modules()], torch.get_rng_state().clone()
    metrics, samples = parameter_use.measure(model, data, batch_size=2)
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    torch.testing.assert_close(data.tensors, inputs, rtol=0, atol=0)
    assert modes == [module.training for module in model.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    assert model.predictor.action_dim == 24
    assert samples["targets"].shape == (4, 10, 2, 6)
    assert metrics["factual/normalized_prediction_change"] == 0
    assert metrics["permuted/normalized_prediction_change"] > 0
    assert metrics["zero/normalized_prediction_change"] > 0
    assert torch.equal(samples["donor_indices"].sort().values, torch.arange(4))
    assert bool((samples["donor_indices"] != torch.arange(4)).all())
    with pytest.raises(ValueError, match="validation"):
        parameter_use.measure(model, replace(data, split="train"))


def test_parameter_audit_own_schema_and_factual_reproduction_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    data, model = _validation_fixture(), build_control("mlp", 0)
    _, samples = parameter_use.measure(model, data, batch_size=2)
    config = TrainConfig(
        str(source), "fixture-cache", "mlp", profile="smoke", steps=2, batch_size=2
    )
    write_json(
        source / "config.json",
        {"config": asdict(config), "candidate_settings": {"action_mix": "paley24"}},
    )
    save_torch(source / "last.pt", {"identity": "fixture", "step": 2, "model": model.state_dict()})
    save_torch(
        source / "validation.pt",
        {"samples": {"predictions": samples["factual_predictions"], "targets": samples["targets"]}},
    )
    write_json(
        source / "matching.json",
        {"online_columns": list(range(6)), "target_columns": list(range(6))},
    )
    complete = {
        "schema": SCHEMA,
        "kind": "train",
        "identity": "fixture",
        "code_identity": code_identity(),
        "data_identity": "fixture-cache",
        "steps": 2,
    }
    _finish(source, complete)

    def fake_cache(directory: Path) -> tuple[dict[str, StateData], dict[str, Any]]:
        assert str(directory) == "fixture-cache"
        return {"validation": data}, {"identity": "fixture-cache"}

    monkeypatch.setattr(parameter_use, "load_data", fake_cache)
    hashes = {path: sha(path) for path in source.iterdir()}
    result = parameter_use.audit(
        parameter_use.AuditConfig(str(source), "fixture-cache", str(tmp_path / "audit"))
    )
    assert result["schema"] == "balanced-action-oracle-parameter-use-v1"
    assert result["candidate_settings"]["action_mix"] == "paley24"
    assert result["source_checkpoint_sha256"] == hashes[source / "last.pt"]
    assert all(sha(path) == digest for path, digest in hashes.items())
    saved = cast(dict[str, Tensor], torch.load(tmp_path / "audit/samples.pt", weights_only=True))
    torch.testing.assert_close(
        saved["factual_predictions"], samples["factual_predictions"], rtol=0, atol=0
    )
    for name, digest in result["hashes"].items():
        assert sha(tmp_path / "audit" / name) == digest
    complete["code_identity"] = "changed-numerical-source"
    _finish(source, complete)
    with pytest.raises(ValueError, match="numerical source"):
        parameter_use.audit(
            parameter_use.AuditConfig(str(source), "fixture-cache", str(tmp_path / "bad"))
        )


def test_readonly_adapters_do_not_change_frozen_source_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = code_identity()
    original_read = Path.read_bytes
    paths = {Path(paired.__file__), Path(parameter_use.__file__)}

    def changed_read(path: Path) -> bytes:
        value = original_read(path)
        return value + b"\n# hypothetical readonly update\n" if path in paths else value

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    assert code_identity() == before


def test_scientific_parameter_audit_rejects_nonfinal_budget_before_loading_data(
    tmp_path: Path,
) -> None:
    source = tmp_path / "early"
    source.mkdir()
    settings = TrainConfig(str(source), "unused", "dense", steps=30_000)
    write_json(source / "config.json", {"config": asdict(settings)})
    _finish(source, {"schema": SCHEMA, "kind": "train", "code_identity": code_identity()})
    with pytest.raises(ValueError, match="final 100000-update endpoint"):
        parameter_use.audit(
            parameter_use.AuditConfig(str(source), "unused", str(tmp_path / "audit"))
        )
