"""Exact donor objective, attached gradients and hard optimizer-budget rollback."""

from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from diagnostics.global_code_calibration.objective import (
    TrainObjective,
    compose_translation,
    prediction_loss,
)
from diagnostics.global_code_calibration.optimization import fit
from diagnostics.mlp_oracle_code_control.model import Prediction
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.overcomplete_acquisition.model import OvercompletePredictor
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder


class Model(nn.Module):
    """Small source container with the actual twelve-code architecture."""

    def __init__(self) -> None:
        """Construct detached double-precision source E/P."""
        super().__init__()
        self.parameter_encoder = TransitionParameterEncoder(6, 12, 16)
        self.predictor = OvercompletePredictor()
        self.double().requires_grad_(False).eval()


def batch(groups: int) -> dict[str, Tensor]:
    """Generate label-free fixture tensors only."""
    generator = torch.Generator().manual_seed(15)
    return {
        "online": torch.randn(groups, 10, 12, 6, generator=generator, dtype=torch.float64),
        "target": torch.randn(groups, 10, 12, 6, generator=generator, dtype=torch.float64),
        "actions": torch.randn(groups, 10, 11, 16, generator=generator, dtype=torch.float64),
    }


def test_all_donors_and_full_gradient_match_brute_force() -> None:
    """All90 ordered within-group pairs reproduce the original expected donor loss."""
    torch.set_num_threads(1)
    model = cast(SCJEPA, Model())
    data = batch(3)
    codes = torch.randn(3, 10, 12, dtype=torch.float64)
    delta = torch.zeros(12, dtype=torch.float64, requires_grad=True)
    original = module_fingerprint(model)
    losses: list[Tensor] = []
    for shift in range(1, 10):
        donor = (codes + delta).roll(shift, 1).flatten(0, 1)
        history = data["online"][:, :, 9:10].flatten(0, 1)
        predictions: list[Tensor] = []
        for step in (9, 10):
            predicted = model.predictor(
                history, donor, data["actions"][:, :, step : step + 1].flatten(0, 1)
            ).prediction
            predictions.append(predicted)
            history = predicted[:, None]
        losses.append(
            (torch.stack(predictions, 1).reshape(3, 10, 2, 6) - data["target"][:, :, 10:12])
            .square()
            .mean()
        )
    expected = torch.stack(losses).mean()
    expected_gradient = torch.autograd.grad(expected, delta)[0]
    actual = prediction_loss(model, data, codes, delta)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, delta)[0], expected_gradient)
    for chunk in (1, 2, 3):
        loss, gradient = TrainObjective(model, data, codes, chunk)(delta)
        assert abs(loss - float(expected.detach())) < 1e-14
        torch.testing.assert_close(gradient, expected_gradient)
    assert module_fingerprint(model) == original
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError, match="label-free"):
        TrainObjective(model, {**data, "theta": torch.zeros(3, 6)}, codes, 2)


class Integrator(OvercompletePredictor):
    """Expose the first-to-second prediction derivative analytically."""

    def forward(self, states: Tensor, parameters: Tensor, actions: Tensor) -> Prediction:
        """Each step depends on the attached preceding prediction."""
        return Prediction(2 * states[:, 0] + parameters[:, :6] + 0 * actions[:, 0, :6])


def test_second_horizon_transmits_through_first_prediction() -> None:
    """Detaching first prediction would change the expected derivative by a factor of three."""
    model = Model()
    model.predictor = Integrator().double().requires_grad_(False)
    data = {key: torch.zeros_like(value) for key, value in batch(2).items()}
    data["target"][:, :, 11] = 1
    delta = torch.zeros(12, dtype=torch.float64, requires_grad=True)
    loss = prediction_loss(
        cast(SCJEPA, model), data, torch.zeros(2, 10, 12, dtype=torch.float64), delta
    )
    gradient = torch.autograd.grad(loss, delta)[0]
    torch.testing.assert_close(gradient[:6], torch.full((6,), -0.5, dtype=torch.float64))
    assert bool((gradient[6:] == 0).all())


def test_translation_composes_exactly_and_preserves_centered_information() -> None:
    """Only final bias changes; all centered probe features and finite code differences agree."""
    model = cast(SCJEPA, Model())
    before = module_fingerprint(model)
    data = batch(3)
    states, actions = (
        data["online"][:, :, :10].flatten(0, 1),
        data["actions"][:, :, :9].flatten(0, 1),
    )
    original = model.parameter_encoder(states, actions)
    delta = torch.randn(12, dtype=torch.float64)
    candidate = compose_translation(model, delta)
    shifted = candidate.parameter_encoder(states, actions)
    torch.testing.assert_close(shifted, original + delta, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(
        shifted - shifted.mean(0), original - original.mean(0), atol=1e-14, rtol=1e-14
    )
    assert module_fingerprint(candidate.predictor) == module_fingerprint(model.predictor)
    assert module_fingerprint(model) == before


def quadratic(point: Tensor) -> tuple[float, Tensor]:
    """A deterministic curved objective with a known minimizer."""
    weights = torch.arange(1, 13, device=point.device, dtype=point.dtype)
    difference = point - 1
    return float((weights * difference.square()).sum().detach()), (
        2 * weights * difference
    ).detach()


def test_budget_cap_rolls_back_an_unaccepted_trial() -> None:
    """A steep first trial consumes the cap; its translation must never be returned."""
    calls = 0

    def steep(point: Tensor) -> tuple[float, Tensor]:
        nonlocal calls
        calls += 1
        return float((1000 * point.square() - point).sum().detach()), (2000 * point - 1).detach()

    result = fit(steep, device=torch.device("cpu"), max_iterations=25, max_evaluations=2)
    assert calls == result["full_evaluations"] == 2
    assert result["stop_reason"] == "evaluation_budget_rollback"
    assert result["accepted_iterations"] == 0
    assert bool((result["translation"] == 0).all())
    assert result["accepted_loss"] == 0
    assert result["optimizer"]["state"] == {}


def test_resume_from_trial_cache_reproduces_fixed_endpoint() -> None:
    """Interrupted line search resumes accepted state and reuses already-budgeted evaluations."""
    saved: dict[str, Any] = {}

    def interrupt(state: dict[str, Any]) -> None:
        nonlocal saved
        saved = state
        if state["full_evaluations"] == 4:
            raise InterruptedError("synthetic interruption")

    expected = fit(quadratic, device=torch.device("cpu"), max_iterations=10, max_evaluations=20)
    with pytest.raises(InterruptedError):
        fit(
            quadratic,
            device=torch.device("cpu"),
            max_iterations=10,
            max_evaluations=20,
            on_state=interrupt,
        )
    actual = fit(
        quadratic, device=torch.device("cpu"), max_iterations=10, max_evaluations=20, state=saved
    )
    torch.testing.assert_close(actual["translation"], expected["translation"], atol=0, rtol=0)
    assert actual["full_evaluations"] == expected["full_evaluations"]
    assert actual["accepted_iterations"] == expected["accepted_iterations"]
    assert actual["accepted_loss"] == expected["accepted_loss"]
