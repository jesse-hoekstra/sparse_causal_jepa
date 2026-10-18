"""Small CPU contracts for loss calibration, frozen MCC matching and routing SHD."""

import copy
import json
import math
import pickle
import random
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor

import scjepa.models.spartan as spartan_module
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.processes import ControlledVectorDGP
from scjepa.data.random import episode_id
from scjepa.data.types import DGPConfig, ScientificEpisode
from scjepa.eval.recovery import (
    CoordinateMatch,
    fit_coordinate_match,
    mean_correlation_coefficient,
    pearson_matrix,
)
from scjepa.experiments.evaluation import (
    EvaluationConfig,
    _aligned_support,  # pyright: ignore[reportPrivateUsage]
    _batch_sizes,  # pyright: ignore[reportPrivateUsage]
    _collect,  # pyright: ignore[reportPrivateUsage]
    _evaluation_context,  # pyright: ignore[reportPrivateUsage]
    calibrate_tau,
    evaluate_model,
    validate_model,
)
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.models.spartan import GraphBlocks, SpartanOutput
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.loop import ObjectiveConfig, TrainingObjective


def _model(*, dense: bool = False) -> SCJEPA:
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
            spartan_dense=dense,
        )
    )


def _config() -> EvaluationConfig:
    return EvaluationConfig(
        calibration_episodes=8,
        validation_episodes=4,
        test_episodes=4,
        batch_size=2,
    )


def _data() -> DGPConfig:
    return DGPConfig(trajectory_length=4, parameter_graph="complementary", root_seed=51)


def _objective() -> ObjectiveConfig:
    return ObjectiveConfig(
        state_regularizer="sigreg",
        state_regularizer_weight=0.1,
        parameter_regularizer="sigreg",
        parameter_regularizer_weight=0.2,
        logit_weight=0.001,
        regularizer_num_projections=8,
    )


def test_validation_excludes_test_and_preserves_rng_modes_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model().train()
    model.parameter_encoder.eval()
    for parameter in model.online_encoder.parameters():
        parameter.grad = torch.ones_like(parameter)
    modes = [module.training for module in model.modules()]
    state = copy.deepcopy(model.state_dict())
    gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    torch_state = torch.get_rng_state().clone()
    python_state = random.getstate()
    numpy_state = np.random.get_state()  # noqa: NPY002
    seen: list[str] = []
    generate = ControlledVectorDGP.generate

    def checked_generate(process: ControlledVectorDGP, index: int) -> ScientificEpisode:
        seen.append(process.config.split)
        assert process.config.split in {"evaluation_calibration", "validation"}
        return generate(process, index)

    monkeypatch.setattr(ControlledVectorDGP, "generate", checked_generate)
    result = validate_model(
        model, _data(), VectorObservationMap().matrix, _config(), objective=_objective()
    )
    assert set(result) == {
        "validation/prediction_mse",
        "validation/base_loss",
        "validation/sparsity",
        "validation/state_mcc",
        "validation/parameter_mcc",
        "validation/state_shd",
        "validation/action_shd",
        "validation/parameter_shd",
    }
    assert result["validation/base_loss"] > result["validation/prediction_mse"] >= 0
    assert seen == ["evaluation_calibration"] * 8 + ["validation"] * 4
    assert 0 <= result["validation/state_mcc"] <= 1
    assert 0 <= result["validation/parameter_mcc"] <= 1
    assert [module.training for module in model.modules()] == modes
    torch.testing.assert_close(model.state_dict(), state, rtol=0, atol=0)
    for parameter, expected in zip(model.parameters(), gradients, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()  # noqa: NPY002
    assert pickle.dumps(actual_numpy) == pickle.dumps(numpy_state)


def test_validation_deterministic_and_excludes_sparsity_terms() -> None:
    model, matrix = _model(), VectorObservationMap().matrix
    first = validate_model(model, _data(), matrix, _config(), objective=_objective())
    torch.randn(23)
    second = validate_model(
        model,
        _data(),
        matrix,
        _config(),
        objective=replace(
            _objective(),
            state_sparsity_weight=9,
            action_sparsity_weight=7,
            parameter_sparsity_weight=2,
        ),
    )
    assert first["validation/base_loss"] == second["validation/base_loss"]
    assert first["validation/prediction_mse"] == second["validation/prediction_mse"]
    assert second["validation/sparsity"] > first["validation/sparsity"]


def test_validation_preserves_modes_and_rng_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model().train()
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()

    def failing_generate(process: ControlledVectorDGP, index: int) -> ScientificEpisode:
        del process, index
        torch.randn(17)
        raise RuntimeError("injected evaluator failure")

    monkeypatch.setattr(ControlledVectorDGP, "generate", failing_generate)
    with pytest.raises(RuntimeError, match="injected evaluator failure"):
        validate_model(model, _data(), VectorObservationMap().matrix, _config())
    assert [module.training for module in model.modules()] == modes
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_tau_uses_calibration_only_and_handles_last_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    generate = ControlledVectorDGP.generate

    def checked_generate(process: ControlledVectorDGP, index: int) -> ScientificEpisode:
        seen.append(process.config.split)
        return generate(process, index)

    monkeypatch.setattr(ControlledVectorDGP, "generate", checked_generate)
    config = replace(_config(), calibration_episodes=3)
    result = calibrate_tau(
        _model(), _data(), VectorObservationMap().matrix, config, objective=_objective()
    )
    assert result["calibration/base_loss"] > result["calibration/prediction_mse"] >= 0
    assert seen == ["evaluation_calibration"] * 3
    assert _batch_sizes(3, 2) == [3]
    assert _batch_sizes(5, 2) == [2, 3]
    assert _batch_sizes(4, 2) == [2, 2]


def test_sampled_calibration_preserves_model_rng_and_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model().train()
    model.parameter_encoder.eval()
    for parameter in model.online_encoder.parameters():
        parameter.grad = torch.ones_like(parameter)
    modes = [module.training for module in model.modules()]
    state = copy.deepcopy(model.state_dict())
    gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    torch_state = torch.get_rng_state().clone()
    python_state = random.getstate()
    numpy_state = pickle.dumps(np.random.get_state())  # noqa: NPY002
    sample = spartan_module._sample_hard_adjacency  # pyright: ignore[reportPrivateUsage]
    calls = 0

    def checked_sample(logits: Tensor, temperature: float, training: bool) -> Tensor:
        nonlocal calls
        calls += 1
        assert training
        assert model.predictor.training
        assert not model.online_encoder.training
        assert not model.parameter_encoder.training
        assert not model.target_encoder.training
        return sample(logits, temperature, training)

    monkeypatch.setattr(spartan_module, "_sample_hard_adjacency", checked_sample)
    matrix = VectorObservationMap().matrix
    first = calibrate_tau(
        model, _data(), matrix, _config(), objective=_objective(), training_gates=True
    )
    second = calibrate_tau(
        model, _data(), matrix, _config(), objective=_objective(), training_gates=True
    )
    assert calls > 0
    assert first == second
    assert [module.training for module in model.modules()] == modes
    torch.testing.assert_close(model.state_dict(), state, rtol=0, atol=0)
    for parameter, expected in zip(model.parameters(), gradients, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    assert random.getstate() == python_state
    assert pickle.dumps(np.random.get_state()) == numpy_state  # noqa: NPY002


def test_dense_calibration_unchanged_with_training_gates() -> None:
    model, matrix = _model(dense=True), VectorObservationMap().matrix
    ordinary = calibrate_tau(model, _data(), matrix, _config(), objective=_objective())
    sampled = calibrate_tau(
        model, _data(), matrix, _config(), objective=_objective(), training_gates=True
    )
    assert ordinary == sampled


def test_sampled_calibration_keeps_sigreg_projection_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, matrix = _model(), VectorObservationMap().matrix
    states: list[Tensor] = []
    objective_forward = TrainingObjective.forward

    def checked_objective(
        objective: TrainingObjective, output: ModelOutput
    ) -> tuple[Tensor, dict[str, Tensor]]:
        states.append(torch.get_rng_state().clone())
        return objective_forward(objective, output)

    monkeypatch.setattr(TrainingObjective, "forward", checked_objective)
    calibrate_tau(model, _data(), matrix, _config(), objective=_objective())
    ordinary_states = states.copy()
    states.clear()
    calibrate_tau(model, _data(), matrix, _config(), objective=_objective(), training_gates=True)
    assert len(states) == 4
    for ordinary, sampled in zip(ordinary_states, states, strict=True):
        torch.testing.assert_close(ordinary, sampled, rtol=0, atol=0)


def test_pearson_mcc_recovers_permutation_and_sign_and_never_rematches_test() -> None:
    generator = torch.Generator().manual_seed(70)
    truth = torch.randn(1000, 3, generator=generator, dtype=torch.float64)
    features = truth[:, [2, 0, 1]] * torch.tensor([-4.0, 2.0, 3.0]) + 5
    match = fit_coordinate_match(features, truth, method="pearson")
    assert match.columns == (1, 2, 0)
    assert math.isclose(mean_correlation_coefficient(match, features, truth), 1.0)
    assert mean_correlation_coefficient(match, features[:, [2, 0, 1]], truth) < 0.1
    assert bool((pearson_matrix(torch.ones_like(features), truth) == 0).all())
    with pytest.raises(ValueError, match="dimensions"):
        mean_correlation_coefficient(CoordinateMatch((0, 1)), features, truth)


def test_graph_alignment_permutates_output_and_latent_inputs_but_not_actions_or_lags() -> None:
    state_order, parameter_order = (2, 0, 1), (1, 0)
    frequencies = {
        "state": torch.arange(18).reshape(3, 2, 3).double() / 18,
        "action": torch.arange(24).reshape(3, 2, 4).double() / 24,
        "parameter": torch.arange(6).reshape(3, 2).double() / 6,
    }
    actual = _aligned_support(
        frequencies, CoordinateMatch(state_order), CoordinateMatch(parameter_order), 0.5
    )
    for output in range(3):
        for lag in range(2):
            for source in range(3):
                assert actual["state"][output, lag, source] == (
                    frequencies["state"][state_order[output], lag, state_order[source]] > 0.5
                )
            torch.testing.assert_close(
                actual["action"][output, lag], frequencies["action"][state_order[output], lag] > 0.5
            )
        for source in range(2):
            assert actual["parameter"][output, source] == (
                frequencies["parameter"][state_order[output], parameter_order[source]] > 0.5
            )


def test_local_graph_aggregation_includes_every_batch_and_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model().eval()
    forward = model.predictor.forward
    calls = 0

    def local_graph(
        states: Tensor, parameters: Tensor, actions: Tensor | None = None
    ) -> SpartanOutput:
        nonlocal calls
        result = forward(states, parameters, actions)
        graph = GraphBlocks(*(torch.zeros_like(block) for block in result.graphs))
        graph.state[:, :, 0, :] = torch.eye(10)
        graph.parameter[:, 0, 1] = float(calls == 3)
        calls += 1
        return result._replace(graphs=graph)

    monkeypatch.setattr(model.predictor, "forward", local_graph)
    with _evaluation_context(model):
        result = _collect(model, _data(), VectorObservationMap().matrix, _config(), "test", 4)
    assert calls == 4
    assert result.graph_frequencies["parameter"][0, 1] == 0.25
    assert result.graph_frequencies["state"].sum() == 10
    matched = _aligned_support(
        result.graph_frequencies,
        CoordinateMatch(tuple(range(10))),
        CoordinateMatch(tuple(range(10))),
        0.0,
    )
    assert matched["parameter"][0, 1]


@pytest.mark.parametrize("variant", ["dense", "complementary", "diagonal"])
def test_final_evaluation_has_only_core_metrics_and_frozen_separate_calibration_artifacts(
    variant: str, tmp_path: Path
) -> None:
    model, config = _model(), _config()
    data, matrix = replace(_data(), parameter_graph=variant), VectorObservationMap(seed=194).matrix
    state, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
    metrics = evaluate_model(model, data, matrix, config, tmp_path, objective=_objective())
    assert set(metrics) == {
        "test/prediction_mse",
        "test/base_loss",
        "test/sparsity",
        "recovery/state_mcc",
        "recovery/parameter_mcc",
        "graph/state_shd",
        "graph/action_shd",
        "graph/parameter_shd",
    }
    assert 0 <= metrics["recovery/state_mcc"] <= 1
    assert 0 <= metrics["recovery/parameter_mcc"] <= 1
    assert all(
        0 <= metrics[f"graph/{name}_shd"] <= 100 for name in ("state", "action", "parameter")
    )
    assert not any(parameter.grad is not None for parameter in model.parameters())
    torch.testing.assert_close(model.state_dict(), state, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    assert protocol["calibration_data"]["split"] == "evaluation_calibration"
    assert protocol["test_data"]["split"] == "test"
    assert protocol["observation_matrix_sha256"] == VectorObservationMap(matrix=matrix).fingerprint
    refs = cast(
        dict[str, object], torch.load(tmp_path / "evaluation_references.pt", weights_only=True)
    )
    torch.testing.assert_close(cast(Tensor, refs["matrix"]), matrix, rtol=0, atol=0)
    torch.testing.assert_close(
        cast(Tensor, refs["calibration_episode_ids"]),
        torch.tensor([episode_id(data.root_seed, "evaluation_calibration", i) for i in range(8)]),
    )
    torch.testing.assert_close(
        cast(Tensor, refs["test_episode_ids"]),
        torch.tensor([episode_id(data.root_seed, "test", i) for i in range(4)]),
    )
    frequencies = cast(dict[str, Tensor], refs["routing_frequencies"])
    assert bool((frequencies["state"][:, 0].diagonal() == 1).all())
    assert cast(Tensor, refs["target_match"]).shape == (10,)
    assert sorted(item.name for item in tmp_path.iterdir()) == [
        "evaluation_references.pt",
        "metrics.json",
        "protocol.json",
    ]


def test_evaluation_rejects_unsupported_model_and_bad_config() -> None:
    model, matrix, config = _model(), VectorObservationMap().matrix, _config()
    with pytest.raises(ValueError, match="trajectory_length"):
        validate_model(model, replace(_data(), trajectory_length=2), matrix, config)
    with pytest.raises(ValueError, match="controlled_vector"):
        validate_model(model, DGPConfig(family="coupled_springs"), matrix, config)
    model.context_len = 1
    with pytest.raises(ValueError, match="K>=2"):
        validate_model(model, _data(), matrix, config)
    with pytest.raises(ValueError, match="graph_support_threshold"):
        replace(config, graph_support_threshold=1)
    with pytest.raises(ValueError, match="batch_size"):
        replace(config, batch_size=1)


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
    def forbidden_cuda_rng(*_args: object, **_kwargs: object) -> Tensor:
        raise AssertionError("CPU evaluation touched another device's CUDA RNG")

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state", forbidden_cuda_rng)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden_cuda_rng)
    monkeypatch.setattr(torch.cuda, "device_count", forbidden_cuda_rng)
    metrics = validate_model(_model(), _data(), VectorObservationMap().matrix, _config())
    assert metrics["validation/prediction_mse"] >= 0


def test_graph_output_rows_use_ema_matching_separately_from_online_state_columns() -> None:
    state_match = CoordinateMatch((2, 0, 1))
    target_match = CoordinateMatch((1, 2, 0))
    parameter_match = CoordinateMatch((1, 0))
    # Ground truth contains only output true-state 0 <- input true-state 2.
    state = torch.zeros(3, 1, 3)
    action = torch.zeros(3, 1, 2)
    parameter = torch.zeros(3, 2)
    state[target_match.columns[0], 0, state_match.columns[2]] = 1
    action[target_match.columns[0], 0, 1] = 1
    parameter[target_match.columns[0], parameter_match.columns[1]] = 1
    support = _aligned_support(
        {"state": state, "action": action, "parameter": parameter},
        state_match,
        parameter_match,
        0.0,
        target_match=target_match,
    )
    expected_state = torch.zeros_like(state, dtype=torch.bool)
    expected_state[0, 0, 2] = True
    expected_action = torch.zeros_like(action, dtype=torch.bool)
    expected_action[0, 0, 1] = True
    expected_parameter = torch.zeros_like(parameter, dtype=torch.bool)
    expected_parameter[0, 1] = True
    torch.testing.assert_close(support["state"], expected_state)
    torch.testing.assert_close(support["action"], expected_action)
    torch.testing.assert_close(support["parameter"], expected_parameter)


def test_validation_base_loss_is_independent_of_calibration_size() -> None:
    model, matrix = _model(), VectorObservationMap().matrix
    small = validate_model(model, _data(), matrix, _config(), objective=_objective())
    large = validate_model(
        model, _data(), matrix, replace(_config(), calibration_episodes=11), objective=_objective()
    )
    for key in ("prediction_mse", "base_loss", "sparsity"):
        assert small[f"validation/{key}"] == large[f"validation/{key}"]
