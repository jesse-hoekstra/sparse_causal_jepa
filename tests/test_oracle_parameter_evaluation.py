"""Raw-theta oracle evaluation, reference alignment, and shuffled-group controls."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import Tensor

from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.types import DGPConfig
from scjepa.experiments.evaluation import (
    EvaluationConfig,
    _collect,  # pyright: ignore[reportPrivateUsage]
    _evaluation_context,  # pyright: ignore[reportPrivateUsage]
    _recovery_metrics,  # pyright: ignore[reportPrivateUsage]
    _shuffled_code_prediction_mse,  # pyright: ignore[reportPrivateUsage]
    calibrate_tau,
    evaluate_model,
    validate_model,
)
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.models.spartan import SpartanOutput
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.loop import ObjectiveConfig


def _model() -> SCJEPA:
    return build_model(
        ModelConfig(
            observation_dim=6,
            state_dim=6,
            param_dim=6,
            action_dim=16,
            context_len=3,
            history_len=1,
            autoregressive=True,
            parameter_source="oracle",
            encoder_hidden_dims=(8,),
            parameter_embed_dim=8,
            parameter_heads=2,
            spartan_layers=1,
            spartan_embed_dim=8,
            spartan_mlp_hidden=8,
            spartan_mlp_layers=2,
            spartan_dense=True,
        )
    )


def _data(views: int = 4) -> DGPConfig:
    return DGPConfig(
        family="simplified_gaussian",
        trajectory_length=5,
        episodes_per_parameter=views,
        parameter_graph="complementary",
    )


def _config() -> EvaluationConfig:
    return EvaluationConfig(
        calibration_episodes=16, validation_episodes=8, test_episodes=8, batch_size=4
    )


@pytest.mark.parametrize("views", [2, 4])
def test_oracle_collection_uses_untransformed_theta_in_every_view(views: int) -> None:
    model, data, config = _model(), _data(views), _config()
    matrix = VectorObservationMap(dimension=6).matrix
    with _evaluation_context(model):
        calibration = _collect(
            model, data, matrix, config, "evaluation_calibration", config.calibration_episodes
        )
        scored = _collect(model, data, matrix, config, "test", config.test_episodes)
    process = build_dgp_dataset(
        replace(data, split="test", num_episodes=config.test_episodes), observation_matrix=matrix
    ).process
    expected = torch.stack(
        [
            process.generate(index).ground_truth["parameters"]
            for index in range(config.test_episodes)
        ]
    )
    assert bool((expected.std(dim=0) > 0).all())
    assert not torch.equal(expected[0], expected[1])
    torch.testing.assert_close(scored.parameters, expected.repeat_interleave(views, dim=0))
    torch.testing.assert_close(
        scored.codes, expected.to(scored.codes).repeat_interleave(views, dim=0), rtol=0, atol=0
    )
    metrics, matches, _ = _recovery_metrics(calibration, scored, 0)
    assert matches["parameter"].columns == tuple(range(6))
    assert math.isclose(metrics["parameter_mcc"], 1, abs_tol=1e-12)
    assert metrics["parameter_column_pair_linear_r2"] > 0.999
    assert metrics["parameter_within_theta_variance"] == 0
    assert metrics["parameter_total_variance"] > 0
    # Raw uniform theta is intentionally not Gaussianized before prediction.
    assert metrics["parameter_gaussianized_mcc"] < 0.999


def test_oracle_calibration_validation_and_test_document_provided_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, data, config = _model(), _data(), _config()
    matrix = VectorObservationMap(dimension=6).matrix
    objective = ObjectiveConfig(regularizer_num_projections=4)
    forward = model.forward
    received: list[Tensor] = []

    def checked_forward(
        observations: Tensor,
        actions: Tensor | None = None,
        prediction_start: int | None = None,
        *,
        oracle_parameters: Tensor | None = None,
    ) -> ModelOutput:
        assert oracle_parameters is not None
        assert oracle_parameters.shape == (observations.shape[0], 6)
        received.append(oracle_parameters.clone())
        return forward(observations, actions, prediction_start, oracle_parameters=oracle_parameters)

    monkeypatch.setattr(model, "forward", checked_forward)
    for training_gates in (False, True):
        received.clear()
        calibrated = calibrate_tau(
            model,
            data,
            matrix,
            config,
            objective=objective,
            training_gates=training_gates,
        )
        assert received
        assert math.isfinite(calibrated["calibration/base_loss"])
    received.clear()
    validation = validate_model(model, data, matrix, config, objective=objective)
    assert received
    assert math.isclose(validation["validation/parameter_mcc"], 1, abs_tol=1e-12)
    received.clear()
    metrics = evaluate_model(model, data, matrix, config, tmp_path, objective=objective)
    assert received
    assert math.isclose(metrics["recovery/parameter_mcc"], 1, abs_tol=1e-12)
    assert metrics["recovery/parameter_column_pair_linear_r2"] > 0.999
    assert metrics["recovery/parameter_within_theta_variance"] == 0
    assert math.isfinite(metrics["test/shuffled_parameter_prediction_mse"])
    metadata = json.loads((tmp_path / "protocol.json").read_text())
    assert metadata["parameter_source"] == "oracle"
    assert "not learned parameter recovery" in metadata["parameter_recovery_interpretation"]
    assert (
        "Gaussianized-reference MCC need not equal one"
        in metadata["parameter_recovery_interpretation"]
    )
    assert "parameter encoder bypassed" in metadata["prediction_context"]
    assert "provided raw ground-truth parameters" in metadata["graph_semantics"]
    assert (
        "provided raw theta shifted between groups"
        in metadata["additional_diagnostics"]["parameter_shuffle"]
    )


def test_oracle_shuffle_rolls_true_groups_and_reuses_raw_theta_at_each_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model().eval()
    theta = torch.linspace(-0.7, 0.7, 18).reshape(3, 6)
    codes = theta[:, None, :].expand(3, 4, 6)
    states = torch.zeros(3, 4, 5, 6)
    actions = torch.arange(3 * 4 * 4 * 16, dtype=torch.float32).reshape(3, 4, 4, 16)
    horizons = torch.tensor([1.0, 2.0])
    targets = codes[:, :, None] * horizons[None, None, :, None]
    result = ModelOutput(targets.clone(), targets, states, codes, (), 2)
    calls: list[tuple[Tensor, Tensor, Tensor]] = []
    forward = model.predictor.forward

    def observed_forward(
        history: Tensor, parameters: Tensor, observed_actions: Tensor | None = None
    ) -> SpartanOutput:
        assert observed_actions is not None
        calls.append((history.clone(), parameters.clone(), observed_actions.clone()))
        output = forward(history, parameters, observed_actions)
        return output._replace(prediction=history[:, -1] + parameters)

    monkeypatch.setattr(model.predictor, "forward", observed_forward)
    mse = _shuffled_code_prediction_mse(model, result, actions)
    expected = theta.roll(1, dims=0).repeat_interleave(4, dim=0)
    assert len(calls) == 2
    for step, (history, parameters, observed_actions) in enumerate(calls):
        torch.testing.assert_close(parameters, expected, rtol=0, atol=0)
        torch.testing.assert_close(history[:, 0], expected * step, rtol=0, atol=0)
        torch.testing.assert_close(
            observed_actions, actions[:, :, step + 2 : step + 3].flatten(0, 1), rtol=0, atol=0
        )
    expected_predictions = expected[:, None] * horizons[None, :, None]
    assert math.isclose(mse, float((expected_predictions - targets.flatten(0, 1)).square().mean()))
    assert mse > 0
