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

from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.group_latent_table_audit import (
    donor_zero_codes,
    prepare_audit_initialization,
    reconstruct_training_truth,
    verified_curriculum,
    verify_initial_table,
)
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.multiquery_control.data import build_cache, module_fingerprint
from diagnostics.multiquery_control.evaluation import parameter_readouts
from diagnostics.multiquery_control.run import sha
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.processes import SimplifiedGaussianDGP
from scjepa.data.types import ScientificEpisode
from scjepa.experiments.storage import write_json
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
