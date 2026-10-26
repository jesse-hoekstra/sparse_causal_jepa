import math
from collections.abc import Generator
from typing import cast

import pytest
import torch
from torch import Tensor, nn

from diagnostics.encoder_gradient_control.updates import CONDITIONS, backward_and_clip
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective
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


def model_and_losses() -> tuple[SCJEPA, Tensor, Tensor]:
    torch.default_generator.manual_seed(713)
    model = build_model(
        ModelConfig(
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
    ).double()
    model.online_encoder.requires_grad_(False).eval()
    model.target_encoder.requires_grad_(False).eval()
    batch = {
        "online": torch.randn(4, 10, 12, 6, dtype=torch.float64),
        "target": torch.randn(4, 10, 12, 6, dtype=torch.float64),
        "actions": torch.randn(4, 10, 11, 16, dtype=torch.float64),
    }
    output = cached_multiquery_forward(model, batch, queries=1, shift=1)
    objective = PredictiveSupervisionObjective(
        ObjectiveConfig(
            state_regularizer="sigreg",
            parameter_regularizer="sigreg",
            state_regularizer_weight=0.4,
            parameter_regularizer_weight=0.4,
            parameter_consistency_weight=1.0,
            regularizer_num_projections=8,
            logit_weight=0.0001,
        ),
        1.0,
    )
    total, terms = objective(output)
    return model, total, terms["prediction"]


def gradient(loss: Tensor, parameters: tuple[Tensor, ...]) -> tuple[Tensor | None, ...]:
    return cast(
        tuple[Tensor | None, ...],
        torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True),
    )


def norm(values: tuple[Tensor | None, ...]) -> float:
    return float(sum(value.square().sum() for value in values if value is not None) ** 0.5)


@pytest.mark.parametrize("measured", [False, True])
def test_baseline_gradients_and_adam_update_match_original_exactly(measured: bool) -> None:
    model, total, prediction = model_and_losses()
    expected, expected_total, _ = model_and_losses()
    expected_parameters = tuple(p for p in expected.parameters() if p.requires_grad)
    expected_total.backward()  # pyright: ignore[reportUnknownMemberType]
    torch.nn.utils.clip_grad_norm_(expected_parameters, 1.0)
    state = torch.get_rng_state()
    result = backward_and_clip(
        model, total, prediction, condition="joint_clip", measure_prediction=measured
    )
    assert torch.equal(torch.get_rng_state(), state)
    for actual, target in zip(model.parameters(), expected.parameters(), strict=True):
        if target.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, target.grad, rtol=0, atol=0)
    for current in (model, expected):
        optimizer = torch.optim.AdamW(
            (p for p in current.parameters() if p.requires_grad), lr=1e-3, weight_decay=0.0
        )
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    for actual, target in zip(model.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    assert result["prediction_components_measured"] == float(measured)
    assert ("encoder/raw_prediction_gradient_norm" in result) == measured


@pytest.mark.parametrize("condition", CONDITIONS)
def test_expected_block_gradients_and_only_encoder_prediction_is_amplified(condition: str) -> None:
    model, total, prediction = model_and_losses()
    encoder = tuple(p for p in model.parameter_encoder.parameters() if p.requires_grad)
    predictor = tuple(p for p in model.predictor.parameters() if p.requires_grad)
    scale = 4.0 if condition == "separate_clip_encoder4" else 1.0
    # Independent reference: differentiate the encoder's scalar objective directly;
    # the predictor always differentiates the ordinary weight-one objective.
    encoder_values = gradient(total + (scale - 1) * prediction, encoder)
    predictor_values = gradient(total, predictor)
    predictive_values = gradient(prediction, encoder)
    auxiliary_values = gradient(total - prediction, encoder)
    expected_encoder = [nn.Parameter(p.detach().clone()) for p in encoder]
    expected_predictor = [nn.Parameter(p.detach().clone()) for p in predictor]
    for parameters, values in (
        (expected_encoder, encoder_values),
        (expected_predictor, predictor_values),
    ):
        for parameter, value in zip(parameters, values, strict=True):
            parameter.grad = value
    if condition == "joint_clip":
        torch.nn.utils.clip_grad_norm_(expected_encoder + expected_predictor, 1.0)
    else:
        torch.nn.utils.clip_grad_norm_(expected_encoder, 1.0)
        torch.nn.utils.clip_grad_norm_(expected_predictor, 1.0)
    state = torch.get_rng_state()
    before = [p.detach().clone() for p in model.parameters()]
    result = backward_and_clip(
        model, total, prediction, condition=condition, measure_prediction=True
    )
    assert torch.equal(torch.get_rng_state(), state)
    for actual, expected in zip(
        encoder + predictor, expected_encoder + expected_predictor, strict=True
    ):
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-10, atol=1e-12)
    for parameter, original in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)
        if not parameter.requires_grad:
            assert parameter.grad is None
    assert result["encoder/prediction_scale"] == scale
    assert math.isclose(result["encoder/raw_prediction_gradient_norm"], norm(predictive_values))
    assert math.isclose(result["encoder/auxiliary_gradient_norm_pre_clip"], norm(auxiliary_values))
    assert math.isclose(
        result["encoder/prediction_gradient_norm_before_clip"], scale * norm(predictive_values)
    )
    assert math.isclose(
        result["encoder/prediction_gradient_norm_after_clip"],
        scale * norm(predictive_values) * result["encoder/clip_coefficient"],
    )
    assert result["encoder/post_clip_norm"] <= 1 + 1e-12
    assert result["predictor/post_clip_norm"] <= 1 + 1e-12


@pytest.mark.parametrize("condition", CONDITIONS)
def test_zero_prediction_and_zero_objective_are_finite(condition: str) -> None:
    model, total, _ = model_and_losses()
    result = backward_and_clip(
        model, total * 0, total.new_zeros(()), condition=condition, measure_prediction=True
    )
    assert result["encoder/clip_coefficient"] == 1
    assert result["predictor/clip_coefficient"] == 1
    assert result["joint/post_clip_norm"] == 0
    assert result["encoder/raw_prediction_gradient_norm"] == 0


def test_nonfinite_loss_is_rejected_before_backward() -> None:
    model, total, prediction = model_and_losses()
    with pytest.raises(FloatingPointError, match="training loss"):
        backward_and_clip(model, total * float("nan"), prediction, condition="joint_clip")
    assert all(p.grad is None for p in model.parameters())


def test_nonfinite_derivative_is_rejected_before_clipping_or_update() -> None:
    model, total, _ = model_and_losses()
    singular_loss = (total - total.detach()).sqrt()
    with pytest.raises(FloatingPointError, match="gradients before clipping"):
        backward_and_clip(model, singular_loss, singular_loss, condition="separate_clip_encoder4")


def test_extra_trainable_parameter_is_rejected() -> None:
    model, total, prediction = model_and_losses()
    model.register_parameter("unexpected", nn.Parameter(torch.zeros(1)))
    with pytest.raises(ValueError, match="Only distinct"):
        backward_and_clip(model, total, prediction, condition="joint_clip")


def test_unfrozen_state_encoder_is_rejected() -> None:
    model, total, prediction = model_and_losses()
    model.online_encoder.requires_grad_(True)
    with pytest.raises(ValueError, match="state encoders"):
        backward_and_clip(model, total, prediction, condition="joint_clip")


def test_stale_gradient_buffers_are_rejected() -> None:
    model, total, prediction = model_and_losses()
    parameter = next(model.parameter_encoder.parameters())
    parameter.grad = torch.zeros_like(parameter)
    with pytest.raises(ValueError, match="zero_grad"):
        backward_and_clip(model, total, prediction, condition="joint_clip")


@pytest.mark.parametrize("scale", [0.0, 1.0, float("nan"), float("inf")])
def test_wrong_named_arm_scale_is_rejected(scale: float) -> None:
    model, total, prediction = model_and_losses()
    with pytest.raises(ValueError, match="scale=4"):
        backward_and_clip(
            model,
            total,
            prediction,
            condition="separate_clip_encoder4",
            encoder_prediction_scale=scale,
        )


def test_unknown_condition_is_rejected() -> None:
    model, total, prediction = model_and_losses()
    with pytest.raises(ValueError, match="Unknown"):
        backward_and_clip(model, total, prediction, condition="global_boost")
