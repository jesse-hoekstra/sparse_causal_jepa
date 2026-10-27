import copy
import math
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from omegaconf import OmegaConf
from test_group_latent_runner import source_with_initialization
from test_multiquery_control import make_model
from torch import Tensor, nn

from diagnostics import group_latent_table_audit as audit_module
from diagnostics.encoder_gradient_control.source import SourceBundle, build_source_model
from diagnostics.group_latent_table_audit import (
    donor_zero_codes,
    prepare_audit_initialization,
    reconstruct_training_truth,
    verified_curriculum,
    verified_teacher_snapshot,
    verify_initial_table,
    verify_snapshot_hashes,
)
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.multiquery_control.data import build_cache, module_fingerprint
from diagnostics.multiquery_control.evaluation import parameter_readouts
from diagnostics.multiquery_control.run import sha
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.datasets import canonical_fingerprint
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.processes import SimplifiedGaussianDGP
from scjepa.data.types import ScientificEpisode
from scjepa.experiments.storage import read_json, write_json
from scjepa.models.model import SCJEPA


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def fixture_data() -> tuple[SCJEPA, dict[str, Tensor], dict[str, Any], Tensor, str]:
    model = make_model().float()
    observation = VectorObservationMap(matrix=torch.eye(6, dtype=torch.float64), dimension=6)
    config = {
        "family": "simplified_gaussian",
        "root_seed": 19,
        "split": "train",
        "num_episodes": 8,
        "episodes_per_parameter": 10,
        "trajectory_length": 12,
        "dtype": "float32",
    }
    train, evaluation = build_cache(
        model,
        observation,
        config,
        train_groups=8,
        calibration_groups=2,
        validation_groups=2,
        test_groups=2,
        batch_size=2,
    )
    return (
        model,
        train,
        config,
        observation.matrix,
        evaluation["metadata"]["training_dataset_fingerprint"],
    )


def test_training_labels_bind_only_to_exact_verified_cache() -> None:
    model, train, config, matrix, fingerprint = fixture_data()
    before = copy.deepcopy(train)
    theta, receipt = reconstruct_training_truth(
        model, train, config, matrix, fingerprint, batch_size=2
    )
    dataset = build_dgp_dataset(OmegaConf.create(config), observation_matrix=matrix)
    expected = torch.stack([dataset.ground_truth(index)["parameters"] for index in range(8)])
    torch.testing.assert_close(theta, expected, rtol=0, atol=0)
    assert receipt["groups_verified"] == 8
    assert receipt["all_actions_equal"]
    torch.testing.assert_close(train, before, rtol=0, atol=0)
    with pytest.raises(ValueError, match="fingerprint"):
        reconstruct_training_truth(model, train, {**config, "root_seed": 20}, matrix, fingerprint)
    changed_actions = {**train, "actions": train["actions"].clone()}
    changed_actions["actions"][-1, -1, -1, -1] += 0.001
    with pytest.raises(ValueError, match="actions differ"):
        reconstruct_training_truth(
            model, changed_actions, config, matrix, fingerprint, batch_size=2
        )
    changed_states = {**train, "target": train["target"].clone()}
    changed_states["target"][-1, -1, -1, -1] += 0.001
    with pytest.raises(ValueError, match="states differ"):
        reconstruct_training_truth(model, changed_states, config, matrix, fingerprint, batch_size=2)
    with pytest.raises(ValueError, match="label-free"):
        reconstruct_training_truth(model, {**train, "theta": theta}, config, matrix, fingerprint)


def test_encoder_audit_uses_individual_context_and_never_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, train, _, _, _ = fixture_data()
    model.eval()

    def forbidden_table(_self: nn.Embedding, _indices: Tensor) -> Tensor:
        raise AssertionError("Train group table accessed by context encoder audit")

    monkeypatch.setattr(nn.Embedding, "forward", forbidden_table)
    actual = donor_zero_codes(model, train, batch_size=2)
    with torch.no_grad():
        expected = torch.cat(
            [
                model.parameter_encoder(
                    train["online"][index : index + 1, 0, :10],
                    train["actions"][index : index + 1, 0, :9],
                )
                for index in range(8)
            ]
        )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    altered = {**train, "online": train["online"].clone()}
    altered["online"][:, 1:] += 100
    torch.testing.assert_close(
        donor_zero_codes(model, altered, batch_size=2), actual, rtol=0, atol=0
    )


def test_reconstructed_truth_must_reproduce_its_own_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, train, config, matrix, fingerprint = fixture_data()
    original = SimplifiedGaussianDGP.generate

    def mismatched_labels(process: SimplifiedGaussianDGP, index: int) -> ScientificEpisode:
        episode = original(process, index)
        truth = {**episode.ground_truth, "parameters": episode.ground_truth["parameters"] + 0.001}
        return replace(episode, ground_truth=truth)

    monkeypatch.setattr(SimplifiedGaussianDGP, "generate", mismatched_labels)
    with pytest.raises(ValueError, match="parameters do not reproduce"):
        reconstruct_training_truth(model, train, config, matrix, fingerprint, batch_size=2)


@pytest.mark.parametrize("mode", ["warm", "fresh_zero"])
def test_audit_authenticates_warm_and_fresh_initial_encoder_and_table(
    tmp_path: Path, mode: str
) -> None:
    source = source_with_initialization(tmp_path / "source")
    before = {str(path): sha(path) for path in source.rglob("*") if path.is_file()}
    bundle = load_primary_source(source, 7, expected_step=2, expected_checkpoint_sha256=None)
    model, _ = build_source_model(bundle, torch.device("cpu"))
    expected = copy.deepcopy(model)
    source_hashes = dict(bundle.hashes)
    initialization: dict[str, Any] | None = None
    if mode == "fresh_zero":
        payload = cast(dict[str, Any], torch.load(source / "initial_seed_7.pt", weights_only=False))
        expected.load_state_dict(payload["model"])
        source_hashes["initial_seed_7.pt"] = sha(source / "initial_seed_7.pt")
        initialization = {
            "mode": "fresh_zero",
            "source_initialization": {
                "path": "initial_seed_7.pt",
                "sha256": sha(source / "initial_seed_7.pt"),
            },
            "parameter_encoder_hash": module_fingerprint(expected.parameter_encoder),
            "predictor_hash": module_fingerprint(expected.predictor),
        }
        values = torch.zeros(bundle.train_groups, 6)
    else:
        values = donor_zero_codes(expected, bundle.train, bundle.batch_size)
    table = nn.Embedding(len(values), 6, _weight=values, sparse=True)
    curriculum: dict[str, Any] = {
        "identity": "authenticated-curriculum",
        "config": {"initialization": mode},
        "source_hashes": source_hashes,
        "initial_encoder_hash": module_fingerprint(expected.parameter_encoder),
        "initial_table_hash": module_fingerprint(table),
    }
    if initialization is not None:
        curriculum["initialization"] = initialization
    expected_codes, hashes = prepare_audit_initialization(model, bundle, curriculum)
    assert hashes == source_hashes
    torch.testing.assert_close(model.state_dict(), expected.state_dict(), rtol=0, atol=0)
    if expected_codes is None:
        expected_codes = donor_zero_codes(model, bundle.train, bundle.batch_size)
    initial = {"identity": curriculum["identity"], "weight": values}
    torch.testing.assert_close(verify_initial_table(initial, curriculum, expected_codes), values)
    assert {str(path): sha(path) for path in source.rglob("*") if path.is_file()} == before
    if mode == "fresh_zero":
        incorrect = values.clone()
        incorrect[0, 0] = 1e-6
        fake = {
            **curriculum,
            "initial_table_hash": module_fingerprint(
                nn.Embedding(len(values), 6, _weight=incorrect)
            ),
        }
        with pytest.raises(ValueError, match="exactly zero"):
            verify_initial_table({**initial, "weight": incorrect}, fake, expected_codes)
        (source / "initial_seed_7.pt").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="Initialization artifact hash"):
            prepare_audit_initialization(model, bundle, curriculum)
    else:
        mismatched = {**curriculum, "initial_encoder_hash": "wrong-encoder"}
        with pytest.raises(ValueError, match="initial encoder"):
            prepare_audit_initialization(model, bundle, mismatched)


@pytest.mark.parametrize("mode", ["warm", "fresh_zero"])
def test_completed_audit_accepts_both_fixed_budget_initializations(
    tmp_path: Path, mode: str
) -> None:
    config = {
        "initialization": mode,
        "verification": False,
        "teacher_steps": 30000,
        "amortization_steps": 10000,
        "washout_steps": 10000,
    }
    write_json(tmp_path / "protocol.json", {"identity": "completed", "config": config})
    write_json(tmp_path / "results.json", {"identity": "completed", "global_step": 50000})
    for name in (
        "initial_group_codes.pt",
        "teacher_final.pt",
        "amortization_final.pt",
        "washout_final.pt",
    ):
        (tmp_path / name).write_bytes(b"authenticated fixture artifact")
    hashes = {path.name: sha(path) for path in tmp_path.iterdir()}
    write_json(tmp_path / "completed.json", {"identity": "completed", "hashes": hashes})
    protocol, completion = verified_curriculum(tmp_path)
    assert protocol["config"] == config
    assert completion["hashes"] == hashes
    (tmp_path / "teacher_final.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        verified_curriculum(tmp_path)


def test_zero_initial_table_readout_reports_finite_no_information_scores() -> None:
    generator = torch.Generator().manual_seed(827)
    counts = {"calibration": 16, "validation": 8, "test": 8}
    codes = {name: torch.zeros(count, 1, 6) for name, count in counts.items()}
    splits = {
        name: {"theta": torch.randn(count, 6, generator=generator)}
        for name, count in counts.items()
    }
    report = parameter_readouts(codes, splits)
    assert report["calibration_code_covariance_eigenvalues"] == [0.0] * 6
    assert all(math.isfinite(value) and value <= 1e-8 for value in report["selected_test_r2"])


def snapshot_fixture(root: Path) -> tuple[Path, Path, SourceBundle, SCJEPA]:
    """Create exact-law caches and authenticated fresh-init phase inputs."""
    model, train, data, matrix, fingerprint = fixture_data()
    source, curriculum = root / "source", root / "curriculum"
    source.mkdir(parents=True)
    curriculum.mkdir()
    initial_name = "initial_seed_7.pt"
    torch.save(
        {"model": model.state_dict(), "seed": 7, "source_sha256": "fixture-source"},
        source / initial_name,
    )
    initial_sha = sha(source / initial_name)
    provenance = {
        "checkpoint": str(source / "source.pt"),
        "sha256": "fixture-source",
        "run_identity": "fixture-run",
        "dataset_identity": fingerprint,
    }
    bundle = SourceBundle(
        source,
        7,
        {
            "identity": "primary",
            "source": provenance,
            "model_config": {},
            "config": {"train_groups": 8, "batch_size": 2},
            "initializations": {"7": {"path": initial_name, "sha256": initial_sha}},
            "artifacts": {initial_name: initial_sha},
        },
        {},
        train,
        {"metadata": {"training_dataset_fingerprint": fingerprint, "observation_matrix": matrix}},
        {},
    )
    record = source / "run_config.json"
    write_json(
        record,
        {"run_identity": "fixture-run", "dataset_identity": fingerprint, "model": {}, "data": data},
    )
    config: dict[str, Any] = {
        "initialization": "fresh_zero",
        "verification": False,
        "teacher_steps": 30000,
        "amortization_steps": 10000,
        "washout_steps": 10000,
        "source_dir": str(source),
        "output_dir": str(curriculum),
        "training_source_root": None,
        "wandb_mode": "disabled",
        "wandb_entity": None,
        "wandb_project": "test",
        "seed": 7,
    }
    settings = {
        k: v
        for k, v in config.items()
        if k
        not in (
            "source_dir",
            "output_dir",
            "training_source_root",
            "wandb_mode",
            "wandb_entity",
        )
    }
    initial = torch.zeros(8, 6)
    evidence = {
        "settings": settings,
        "source_hashes": {initial_name: initial_sha},
        "initialization": {
            "mode": "fresh_zero",
            "source_initialization": {"path": initial_name, "sha256": initial_sha},
            "parameter_encoder_hash": module_fingerprint(model.parameter_encoder),
            "predictor_hash": module_fingerprint(model.predictor),
        },
        "source_step": 30000,
        "initial_table_hash": module_fingerprint(nn.Embedding.from_pretrained(initial)),  # pyright: ignore[reportUnknownMemberType]
        "initial_encoder_hash": module_fingerprint(model.parameter_encoder),
        "frozen_hashes": {},
        "code_identity": "fixture-code",
        "torch": str(torch.__version__),
    }
    identity = canonical_fingerprint(evidence)
    write_json(curriculum / "protocol.json", {"identity": identity, **evidence, "config": config})
    torch.save({"identity": identity, "weight": initial}, curriculum / "initial_group_codes.pt")
    torch.save(
        {
            "identity": identity,
            "phase": "teacher",
            "phase_step": 30000,
            "global_step": 30000,
            "model": model.state_dict(),
            "table": {"weight": torch.randn(8, 6)},
        },
        curriculum / "teacher_final.pt",
    )
    return curriculum, record, bundle, model


def test_snapshot_rejects_wrong_step_identity_and_mutation_without_parent_completion(
    tmp_path: Path,
) -> None:
    curriculum, _, _, _ = snapshot_fixture(tmp_path)
    protocol, receipt = verified_teacher_snapshot(curriculum)
    assert receipt["kind"] == "read_only_teacher_phase_snapshot"
    assert receipt["parent_completion_status"] == "not_asserted"
    assert set(receipt["hashes"]) == {"protocol.json", "initial_group_codes.pt", "teacher_final.pt"}
    assert not (curriculum / "completed.json").exists()
    with pytest.raises(FileNotFoundError):
        verified_curriculum(curriculum)
    teacher_path = curriculum / "teacher_final.pt"
    teacher = cast(dict[str, Any], torch.load(teacher_path, map_location="cpu", weights_only=False))
    for change in ({"phase_step": 29999}, {"global_step": 30001}, {"identity": "wrong"}):
        torch.save({**teacher, **change}, teacher_path)
        with pytest.raises(ValueError, match="identity or fixed phase counters"):
            verified_teacher_snapshot(curriculum)
    torch.save(teacher, teacher_path)
    _, receipt = verified_teacher_snapshot(curriculum)
    teacher["table"]["weight"][0, 0] += 0.01
    torch.save(teacher, teacher_path)
    with pytest.raises(ValueError, match="input changed"):
        verify_snapshot_hashes(curriculum, receipt["hashes"])
    protocol["config"]["teacher_steps"] = 29999
    write_json(curriculum / "protocol.json", protocol)
    with pytest.raises(ValueError, match="scientific 30000"):
        verified_teacher_snapshot(curriculum)


def test_snapshot_audits_only_initial_and_teacher_with_full_cache_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curriculum, record, bundle, model = snapshot_fixture(tmp_path)

    def load_source(*_args: object, **_kwargs: object) -> SourceBundle:
        return bundle

    def build_model(
        _bundle: SourceBundle, device: torch.device
    ) -> tuple[SCJEPA, torch.optim.AdamW]:
        restored = copy.deepcopy(model).to(device)
        return restored, torch.optim.AdamW(restored.parameters())

    monkeypatch.setattr(audit_module, "load_primary_source", load_source)
    monkeypatch.setattr(audit_module, "build_source_model", build_model)
    monkeypatch.setattr(audit_module, "READOUT_COUNTS", (4, 2, 2))
    before = {str(p): sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    result = audit_module.audit(curriculum, tmp_path / "snapshot", record, teacher_snapshot=True)
    assert set(result["reports"]) == {"initial_group_table", "learned_teacher_table"}
    assert result["reconstruction"]["groups_verified"] == 8
    assert result["reconstruction"]["all_actions_equal"]
    assert result["phase_snapshot"]["parent_completion_status"] == "not_asserted"
    completed = read_json(tmp_path / "snapshot/completed.json")
    assert completed["completion_scope"] == "teacher_phase_audit_only"
    assert completed["parent_complete"] is None
    assert not (curriculum / "completed.json").exists()
    assert not (curriculum / "amortization_final.pt").exists()
    assert all(sha(Path(name)) == expected for name, expected in before.items())
    original_readouts = audit_module.parameter_readouts

    def changed_input(
        codes: dict[str, Tensor], splits: dict[str, dict[str, Tensor]]
    ) -> dict[str, Any]:
        report = original_readouts(codes, splits)
        with (curriculum / "teacher_final.pt").open("ab") as handle:
            handle.write(b"changed during audit")
        return report

    monkeypatch.setattr(audit_module, "parameter_readouts", changed_input)
    with pytest.raises(ValueError, match="input changed"):
        audit_module.audit(curriculum, tmp_path / "changed", record, teacher_snapshot=True)
    assert not (tmp_path / "changed/completed.json").exists()
