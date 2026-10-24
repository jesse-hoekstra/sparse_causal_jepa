"""Indexing, differentiable rollout, and evaluation contracts for the diagnostic."""

import pickle
import random
from copy import deepcopy
from math import isclose
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from diagnostics.oracle_state_predictor.data import build_data
from diagnostics.oracle_state_predictor.run import evaluate, forecast_metrics, rollout
from scjepa.models.spartan import GraphBlocks, GraphPenalties, Spartan, SpartanOutput


class _RecordingPredictor(nn.Module):
    """Expose rollout inputs and deliberately large irrelevant path penalties."""

    def __init__(self, *, consume_rng: bool = False, fail_on_second: bool = False) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.25))
        self.helper = nn.Identity()
        self.consume_rng = consume_rng
        self.fail_on_second = fail_on_second
        self.inputs: list[tuple[Tensor, Tensor, Tensor]] = []
        self.predictions: list[Tensor] = []
        self.modes: list[tuple[bool, ...]] = []

    def forward(self, states: Tensor, parameters: Tensor, actions: Tensor) -> SpartanOutput:
        """Apply a simple affine transition with an attached two-step gradient."""
        self.inputs.append((states, parameters, actions))
        self.modes.append(tuple(module.training for module in self.modules()))
        if self.consume_rng:
            random.random()
            np.random.random()  # noqa: NPY002 - intentionally exercise the global stream.
            torch.rand(3)
        if self.fail_on_second and len(self.inputs) == 2:
            raise RuntimeError("intentional prediction failure")
        prediction = self.scale * states[:, 0] + parameters + actions[:, 0, :6]
        if prediction.requires_grad:
            prediction.retain_grad()
        self.predictions.append(prediction)
        batch = len(prediction)
        zero = prediction.new_zeros(())
        return SpartanOutput(
            prediction=prediction,
            path_matrix=prediction.new_zeros(batch, 28, 28),
            graphs=GraphBlocks(
                prediction.new_zeros(batch, 6, 1, 6),
                prediction.new_zeros(batch, 6, 1, 16),
                prediction.new_zeros(batch, 6, 6),
            ),
            sparsity=GraphPenalties(*(prediction.new_tensor(1e6) for _ in range(3))),
            logit_penalty=self.scale.square(),
            mean_abs_logit=zero,
            mean_gate_probability=zero,
            gate_entropy=zero,
        )


def test_rollout_uses_two_observed_actions_one_theta_and_attached_predictions() -> None:
    """Horizon two must consume the first prediction, with its gradient intact."""
    model = _RecordingPredictor()
    initial = torch.arange(12, dtype=torch.float32).reshape(2, 1, 6) / 10
    parameters = torch.arange(12, dtype=torch.float32).reshape(2, 6) / 100
    actions = torch.arange(64, dtype=torch.float32).reshape(2, 2, 16) / 20
    predictions, logit = rollout(cast(Spartan, model), initial, parameters, actions)
    first = model.scale * initial[:, 0] + parameters + actions[:, 0, :6]
    second = model.scale * first + parameters + actions[:, 1, :6]
    torch.testing.assert_close(predictions, torch.stack((first, second), dim=1))
    torch.testing.assert_close(logit, model.scale.square())
    assert len(model.inputs) == 2
    torch.testing.assert_close(model.inputs[0][0], initial)
    torch.testing.assert_close(model.inputs[1][0][:, 0], first)
    for index, (_, supplied_theta, supplied_actions) in enumerate(model.inputs):
        assert supplied_theta is parameters
        torch.testing.assert_close(supplied_actions[:, 0], actions[:, index])

    # Backpropagating only horizon two must reach the intermediate prediction.
    predictions[:, 1].sum().backward()  # pyright: ignore[reportUnknownMemberType]
    intermediate_gradient = model.predictions[0].grad
    assert intermediate_gradient is not None
    torch.testing.assert_close(intermediate_gradient, torch.full_like(first, 1.25))
    expected_scale_gradient = (
        2 * model.scale.detach() * initial[:, 0] + parameters + actions[:, 0, :6]
    ).sum()
    assert model.scale.grad is not None
    torch.testing.assert_close(model.scale.grad, expected_scale_gradient)


def test_forecast_metrics_pool_examples_per_coordinate_and_keep_horizons_separate() -> None:
    """Large coordinate offsets must not inflate the target-variance denominator."""
    group = torch.arange(3, dtype=torch.float64)[:, None, None]
    horizon = torch.arange(2, dtype=torch.float64)[None, :, None]
    coordinate = torch.arange(6, dtype=torch.float64)[None, None, :]
    targets = 2 * group + 10 * horizon + 100 * coordinate
    predictions = targets + torch.tensor([1.0, 3.0], dtype=torch.float64)[None, :, None]
    metrics = forecast_metrics(predictions, targets)
    expected = {
        "prediction_mse": 5.0,
        "horizon_1_mse": 1.0,
        "horizon_2_mse": 9.0,
        "target_variance": 83 / 3,
        "horizon_1_target_variance": 8 / 3,
        "horizon_2_target_variance": 8 / 3,
        "normalized_prediction_mse": 15 / 83,
        "horizon_1_normalized_mse": 3 / 8,
        "horizon_2_normalized_mse": 27 / 8,
    }
    for key, value in expected.items():
        assert isclose(metrics[key], value)


@pytest.mark.parametrize("logit_weight", [0.0, 1e-4])
def test_evaluation_uses_global_metrics_and_only_prediction_plus_logit(
    logit_weight: float,
) -> None:
    """A short final batch must not change weighting or add graph penalties."""
    data = build_data("validation", 3, views=2)
    model = _RecordingPredictor()
    batch = data.batch(torch.arange(len(data)), "cpu")
    with torch.no_grad():
        first = (
            model.scale * batch.initial_states[:, 0] + batch.parameters + batch.actions[:, 0, :6]
        )
        second = model.scale * first + batch.parameters + batch.actions[:, 1, :6]
        expected = forecast_metrics(torch.stack((first, second), dim=1), batch.targets)
    actual = evaluate(cast(Spartan, model), data, 2, torch.device("cpu"), logit_weight)
    for key, value in expected.items():
        assert isclose(actual[key], value, rel_tol=1e-6, abs_tol=1e-8)
    assert isclose(actual["logit"], 1.25**2)
    assert isclose(actual["base_loss"], expected["prediction_mse"] + logit_weight * 1.25**2)
    whole_batch = evaluate(cast(Spartan, model), data, 3, torch.device("cpu"), logit_weight)
    assert actual.keys() == whole_batch.keys()
    for key, value in whole_batch.items():
        assert isclose(actual[key], value, rel_tol=1e-6, abs_tol=1e-8)


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_preserves_all_model_modes_weights_and_rng_even_on_failure(fail: bool) -> None:
    """Validation must not alter the following training step or consume its RNG."""
    data = build_data("validation", 3, views=2)
    model = _RecordingPredictor(consume_rng=True, fail_on_second=fail).train()
    model.helper.eval()
    initial_modes = tuple(module.training for module in model.modules())
    initial_weights = deepcopy(model.state_dict())
    python_state = random.getstate()
    numpy_state = pickle.dumps(np.random.get_state())  # noqa: NPY002
    torch_state = torch.get_rng_state().clone()
    if fail:
        with pytest.raises(RuntimeError, match="intentional prediction failure"):
            evaluate(cast(Spartan, model), data, 2, torch.device("cpu"), 1e-4)
    else:
        evaluate(cast(Spartan, model), data, 2, torch.device("cpu"), 1e-4)
    assert tuple(module.training for module in model.modules()) == initial_modes
    assert all(not any(modes) for modes in model.modes)
    torch.testing.assert_close(model.state_dict(), initial_weights, rtol=0, atol=0)
    assert random.getstate() == python_state
    assert pickle.dumps(np.random.get_state()) == numpy_state  # noqa: NPY002
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)


def test_real_dense_spartan_rollout_is_differentiable_and_gate_rng_free() -> None:
    """Exercise the existing two-layer predictor without any SCJEPA components."""
    model = Spartan(
        state_dim=6,
        param_dim=6,
        action_dim=16,
        history_len=1,
        embed_dim=8,
        num_layers=2,
        mlp_hidden_size=8,
        mlp_num_layers=3,
        dense=True,
    )
    data = build_data("train", 2, views=2)
    batch = data.batch(torch.arange(len(data)), "cpu")
    rng = torch.get_rng_state().clone()
    prediction, logit = rollout(model, batch.initial_states, batch.parameters, batch.actions)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert prediction.shape == batch.targets.shape == (4, 2, 6)
    assert logit.ndim == 0
    loss = (prediction - batch.targets).square().mean() + 1e-4 * logit
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert model.state_project.weight.grad is not None
    assert bool(model.state_project.weight.grad.abs().sum() > 0)
    assert model.param_project.weight.grad is not None
    assert bool(model.param_project.weight.grad.abs().sum() > 0)
    assert model.action_project is not None
    assert model.action_project.weight.grad is not None
    assert bool(model.action_project.weight.grad.abs().sum() > 0)
