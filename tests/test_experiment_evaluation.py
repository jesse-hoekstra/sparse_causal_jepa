"""Held-out scientific evaluation contracts using only tiny CPU tensor problems."""

import copy
import json
import pickle
import random
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor

from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.processes import ControlledVectorDGP
from scjepa.data.random import episode_id
from scjepa.data.types import DGPConfig, ScientificEpisode
from scjepa.eval.recovery import FrozenProbe, ProbeConfig, ProbePayload
from scjepa.experiments.evaluation import (
    EvaluationConfig,
    _collect,  # pyright: ignore[reportPrivateUsage]
    _evaluation_context,  # pyright: ignore[reportPrivateUsage]
    _paired_episodes,  # pyright: ignore[reportPrivateUsage]
    _physical_metrics,  # pyright: ignore[reportPrivateUsage]
    _rollout,  # pyright: ignore[reportPrivateUsage]
    evaluate_model,
    validate_model,
)
from scjepa.models.model import SCJEPA
from scjepa.training.factory import ModelConfig, build_model


def _model() -> SCJEPA:
    return build_model(
        ModelConfig(
            observation_dim=10,
            state_dim=10,
            param_dim=10,
            action_dim=10,
            history_len=1,
            context_len=2,
            encoder_hidden_dims=(8,),
            parameter_embed_dim=8,
            parameter_heads=2,
            spartan_embed_dim=8,
            spartan_mlp_hidden=8,
            spartan_layers=1,
            spartan_mlp_layers=2,
        )
    )


def _config() -> EvaluationConfig:
    return EvaluationConfig(
        calibration_episodes=8,
        validation_episodes=4,
        test_episodes=4,
        paired_episodes=2,
        batch_size=2,
        rollout_steps=2,
        validation_probe_steps=2,
        probe=ProbeConfig(hidden_dim=8, steps=2, batch_size=16),
    )


def _data() -> DGPConfig:
    return DGPConfig(trajectory_length=4, parameter_graph="complementary", root_seed=51)


def test_validation_never_reads_test_and_preserves_model_rng_and_existing_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model().train()
    model.parameter_encoder.eval()  # Preserve a deliberately mixed module-mode tree.
    for parameter in model.online_encoder.parameters():
        parameter.grad = torch.ones_like(parameter)
    modes = [module.training for module in model.modules()]
    state = copy.deepcopy(model.state_dict())
    gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    torch_state = torch.get_rng_state().clone()
    python_state = random.getstate()
    numpy_state = np.random.get_state()  # noqa: NPY002 — verify preservation of caller RNG
    seen: list[str] = []
    generate = ControlledVectorDGP.generate

    def checked_generate(process: ControlledVectorDGP, index: int) -> ScientificEpisode:
        seen.append(process.config.split)
        assert process.config.split in {"evaluation_calibration", "validation"}
        return generate(process, index)

    monkeypatch.setattr(ControlledVectorDGP, "generate", checked_generate)
    with torch.no_grad():
        result = validate_model(model, _data(), VectorObservationMap().matrix, _config())
    assert result["validation/physical_prediction_nrmse"] >= 0
    assert result["validation/encoded_target_decoding_nrmse"] >= 0
    assert seen.count("evaluation_calibration") == 8
    assert seen.count("validation") == 4
    assert [module.training for module in model.modules()] == modes
    torch.testing.assert_close(model.state_dict(), state, rtol=0, atol=0)
    for parameter, expected in zip(model.parameters(), gradients, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()  # noqa: NPY002 — compare legacy caller RNG state
    assert pickle.dumps(actual_numpy) == pickle.dumps(numpy_state)


def test_validation_is_deterministic_and_preserves_modes_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    data, config, matrix = _data(), _config(), VectorObservationMap().matrix
    first = validate_model(model, data, matrix, config)
    torch.randn(23)
    second = validate_model(model, data, matrix, config)
    assert first == second
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()

    def failing_generate(process: ControlledVectorDGP, index: int) -> ScientificEpisode:
        del process, index
        torch.randn(17)
        raise RuntimeError("injected evaluator failure")

    monkeypatch.setattr(ControlledVectorDGP, "generate", failing_generate)
    with pytest.raises(RuntimeError, match="injected evaluator failure"):
        validate_model(model, data, matrix, config)
    assert [module.training for module in model.modules()] == modes
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_paired_episodes_reuse_parameters_and_redraw_independent_histories() -> None:
    model, data, config = _model().eval(), _data(), _config()
    matrix = VectorObservationMap().matrix
    with _evaluation_context(model):
        test = _collect(model, data, matrix, config, "test", config.test_episodes)
    paired = list(_paired_episodes(data, matrix, test, config.paired_episodes))
    again = list(_paired_episodes(data, matrix, test, config.paired_episodes))
    for index, (episode, repeat) in enumerate(zip(paired, again, strict=True)):
        torch.testing.assert_close(
            episode.ground_truth["parameters"], test.parameters[index], rtol=0, atol=0
        )
        assert not torch.equal(episode.ground_truth["states"][0], test.states[index, 0])
        assert not torch.equal(episode.actions, test.actions[index])
        assert episode.episode_id == int(test.ids[index])
        torch.testing.assert_close(episode.observations, repeat.observations, rtol=0, atol=0)
        torch.testing.assert_close(episode.actions, repeat.actions, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["dense", "complementary", "diagonal"])
def test_final_evaluation_reports_metrics_and_frozen_separate_calibration_artifacts(
    variant: str,
    tmp_path: Path,
) -> None:
    model, config = _model(), replace(_config(), paired_episodes=3)
    data = replace(_data(), parameter_graph=variant)
    matrix = VectorObservationMap(seed=194).matrix
    state = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    metrics = evaluate_model(model, data, matrix, config, tmp_path)
    assert metrics["test/physical_prediction_nrmse"] >= 0
    assert metrics["test/encoded_target_decoding_nrmse"] >= 0
    assert metrics["test/rollout_nrmse"] >= 0
    assert "test/rollout_step_2_nrmse" in metrics
    assert "recovery/state/full/r2" in metrics
    assert "recovery/parameter/coordinate/spearman_mean" in metrics
    assert "consistency/different_parameter_swap_nrmse_change" in metrics
    assert 0 <= metrics["graph/state/density"] <= 1
    assert metrics["graph/state/edges"] >= 10  # Mandatory diagonal residual paths.
    assert not any(parameter.grad is not None for parameter in model.parameters())
    torch.testing.assert_close(model.state_dict(), state, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    assert protocol["calibration_data"]["split"] == "evaluation_calibration"
    assert protocol["test_data"]["split"] == "test"
    assert protocol["observation_matrix_sha256"] == VectorObservationMap(matrix=matrix).fingerprint
    refs: dict[str, Tensor] = torch.load(tmp_path / "evaluation_references.pt", weights_only=True)
    torch.testing.assert_close(refs["matrix"], matrix, rtol=0, atol=0)
    torch.testing.assert_close(
        refs["calibration_episode_ids"],
        torch.tensor([episode_id(data.root_seed, "evaluation_calibration", i) for i in range(8)]),
    )
    torch.testing.assert_close(
        refs["test_episode_ids"],
        torch.tensor([episode_id(data.root_seed, "test", i) for i in range(4)]),
    )
    paired: dict[str, Tensor] = torch.load(tmp_path / "parameter_consistency.pt", weights_only=True)
    torch.testing.assert_close(
        paired["true_parameters"], paired["paired_true_parameters"], rtol=0, atol=0
    )
    assert paired["original_codes"].shape == paired["paired_codes"].shape == (3, 10)
    payload = cast(ProbePayload, torch.load(tmp_path / "target_state_probe.pt", weights_only=True))
    probe = FrozenProbe.from_payload(payload)
    assert probe.predict(torch.zeros(2, 3, 10)).shape == (2, 3, 10)
    recovered: dict[str, object] = torch.load(tmp_path / "parameter_recovery.pt", weights_only=True)
    groups = cast(list[object], recovered["group_probes"])
    assert len(groups) == {"dense": 1, "complementary": 5, "diagonal": 10}[variant]


def test_rollout_uses_context_endpoint_action_and_appends_predicted_states() -> None:
    model, data, config = _model().eval(), _data(), _config()
    with _evaluation_context(model):
        test = _collect(model, data, VectorObservationMap().matrix, config, "test", 4)
        actual = _rollout(model, test, config)
        first = model.predictor(test.online[:, 1:2], test.codes, test.actions[:, 1:2]).prediction
        second = model.predictor(first[:, None], test.codes, test.actions[:, 2:3]).prediction
    torch.testing.assert_close(actual, torch.stack((first, second), 1))


def test_physical_normalization_is_coordinatewise_and_calibration_fixed() -> None:
    truth = torch.zeros(2, 2)
    predicted = torch.tensor([[2.0, 3.0], [2.0, 3.0]])
    result = _physical_metrics(predicted, truth, torch.tensor([4.0, 9.0]), "test")
    assert result == {"test_mse": 6.5, "test_nrmse": 1.0}


def test_evaluation_rejects_unsupported_model_and_short_prediction_window() -> None:
    model, data, matrix, config = _model(), _data(), VectorObservationMap().matrix, _config()
    with pytest.raises(ValueError, match="trajectory_length"):
        validate_model(model, replace(data, trajectory_length=3), matrix, config)
    with pytest.raises(ValueError, match="controlled_vector"):
        validate_model(model, DGPConfig(family="coupled_springs"), matrix, config)
    model.context_len = 1
    with pytest.raises(ValueError, match="K>=2"):
        validate_model(model, data, matrix, config)
    with pytest.raises(ValueError, match="paired_episodes"):
        replace(config, paired_episodes=5)


def test_nonfinite_encodings_fail_explicitly_and_restore_training_mode() -> None:
    model = _model().train()
    with torch.no_grad():
        next(model.online_encoder.parameters()).fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="nonfinite evaluation"):
        validate_model(model, _data(), VectorObservationMap().matrix, _config())
    assert model.training
    assert not model.target_encoder.training


def test_cpu_evaluation_does_not_touch_initialized_cuda_rngs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()

    def forbidden_cuda_rng(*_args: object, **_kwargs: object) -> Tensor:
        raise AssertionError("CPU evaluation touched another device's CUDA RNG")

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state", forbidden_cuda_rng)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden_cuda_rng)
    monkeypatch.setattr(torch.cuda, "device_count", forbidden_cuda_rng)
    metrics = validate_model(model, _data(), VectorObservationMap().matrix, _config())
    assert metrics["validation/physical_prediction_nrmse"] >= 0
