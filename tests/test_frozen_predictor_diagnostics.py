import math
from collections.abc import Generator
from typing import cast

import pytest
import torch
from torch import Tensor

from diagnostics.frozen_predictor_control.diagnostics import (
    code_reference_stats,
    encoder_gradient_stats,
    predictor_code_sensitivity,
)
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


def model_and_batch() -> tuple[SCJEPA, dict[str, Tensor]]:
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
    for module in (model.online_encoder, model.target_encoder, model.predictor):
        module.requires_grad_(False).eval()
    return model, {
        "online": torch.randn(4, 10, 12, 6, dtype=torch.float64),
        "target": torch.randn(4, 10, 12, 6, dtype=torch.float64),
        "actions": torch.randn(4, 10, 11, 16, dtype=torch.float64),
    }


def objective_config() -> ObjectiveConfig:
    return ObjectiveConfig(
        state_regularizer="sigreg",
        parameter_regularizer="sigreg",
        state_regularizer_weight=0.4,
        parameter_regularizer_weight=0.4,
        parameter_consistency_weight=1.0,
        regularizer_num_projections=8,
        logit_weight=0.0001,
    )


def test_reference_stats_are_distribution_only_and_do_not_mutate_codes() -> None:
    values = torch.arange(24, dtype=torch.float64).reshape(2, 2, 6).requires_grad_(True)
    before = values.detach().clone()
    stats = code_reference_stats(values)
    assert stats["count"] == 4
    assert stats["mean"] == [9, 10, 11, 12, 13, 14]
    assert all(math.isclose(value, math.sqrt(45)) for value in stats["std"])
    assert math.isclose(stats["centered_rms"], math.sqrt(45))
    torch.testing.assert_close(
        torch.tensor(stats["covariance_eigenvalues"]),
        torch.tensor([0.0] * 5 + [270.0]),
        rtol=1e-10,
        atol=1e-12,
    )
    assert values.grad is None
    torch.testing.assert_close(values, before, rtol=0, atol=0)
    with pytest.raises(ValueError, match="dimension six"):
        code_reference_stats(values[..., :5])
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        code_reference_stats(values * float("nan"))


def test_gradient_stats_match_total_derivative_and_leave_backward_unchanged() -> None:
    model, batch = model_and_batch()
    output = cached_multiquery_forward(model, batch, queries=1, shift=1)
    config = objective_config()
    total, terms = PredictiveSupervisionObjective(config, 1.0)(output)
    parameters = tuple(model.parameter_encoder.parameters())
    expected = torch.autograd.grad(total, (output.parameters, *parameters), retain_graph=True)
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 0.25)
    buffers = [cast(Tensor, p.grad).clone() for p in parameters]
    weights = [p.detach().clone() for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    rng = torch.get_rng_state()
    result = encoder_gradient_stats(model, output, terms, config)
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == [m.training for m in model.modules()]
    assert result["code/prediction_norm"] > 0
    assert result["encoder/prediction_norm"] > 0
    assert math.isclose(
        result["code/total_norm"], float(expected[0].square().sum().sqrt()), rel_tol=1e-10
    )
    assert math.isclose(
        result["encoder/total_norm"],
        float(torch.cat([value.flatten() for value in expected[1:]]).square().sum().sqrt()),
        rel_tol=1e-10,
    )
    for parameter, buffer in zip(parameters, buffers, strict=True):
        torch.testing.assert_close(parameter.grad, buffer, rtol=0, atol=0)
    total.backward()  # pyright: ignore[reportUnknownMemberType]
    for parameter, value, buffer in zip(parameters, expected[1:], buffers, strict=True):
        torch.testing.assert_close(parameter.grad, value + buffer, rtol=1e-10, atol=1e-12)
    for parameter, original in zip(model.parameters(), weights, strict=True):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)
        if not parameter.requires_grad:
            assert parameter.grad is None


def test_constant_losses_have_defined_zero_norms_and_undefined_cosines() -> None:
    model, batch = model_and_batch()
    output = cached_multiquery_forward(model, batch, queries=1, shift=1)
    terms = {
        key: output.parameters.new_zeros(())
        for key in ("prediction", "parameter_regularizer", "parameter_consistency", "logit")
    }
    result = encoder_gradient_stats(model, output, terms, objective_config())
    assert all(value == 0 for value in result.values())
    model.predictor.requires_grad_(True)
    with pytest.raises(ValueError, match="must be frozen"):
        encoder_gradient_stats(model, output, terms, objective_config())


def test_sensitivity_matches_finite_differences_and_preserves_model_and_rng() -> None:
    model, batch = model_and_batch()
    for parameter in model.parameter_encoder.parameters():
        parameter.grad = torch.full_like(parameter, 0.125)
    weights = [p.detach().clone() for p in model.parameters()]
    gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    modes = [module.training for module in model.modules()]
    flags = [p.requires_grad for p in model.parameters()]
    groups = 2
    with torch.no_grad():
        codes = model.parameter_encoder(
            batch["online"][:groups, :, :10].flatten(0, 1),
            batch["actions"][:groups, :, :9].flatten(0, 1),
        ).reshape(groups, 10, 6)
    reference = code_reference_stats(codes)
    state = torch.get_rng_state()
    result = predictor_code_sensitivity(model, batch, reference_stats=reference, max_groups=groups)
    assert torch.equal(state, torch.get_rng_state())
    assert [m.training for m in model.modules()] == modes
    assert [p.requires_grad for p in model.parameters()] == flags
    for parameter, original, gradient in zip(model.parameters(), weights, gradients, strict=True):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)
        if gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    assert result["groups"] == 2
    assert result["queries"] == 20
    assert result["prediction_code_gradient_norm"] > 0
    assert math.isclose(result["source_code_comparison"]["rms_ratio"], 1)
    assert math.isclose(result["source_code_comparison"]["centered_rms_ratio"], 1)
    assert result["source_code_comparison"]["mean_shift_in_source_centered_rms"] == 0
    assert result["source_code_comparison"]["used_for_training"] is False

    # Numerical perturbations check the full two-step autoregressive derivative,
    # including the first prediction's contribution to the second prediction.
    donor = codes.roll(1, 1).flatten(0, 1)
    history = batch["online"][:groups, :, 9:10].flatten(0, 1)
    actions = batch["actions"][:groups].flatten(0, 1)

    def predict(values: Tensor) -> Tensor:
        first = model.predictor(history, values, actions[:, 9:10]).prediction
        second = model.predictor(first[:, None], values, actions[:, 10:11]).prediction
        return torch.stack((first, second), 1)

    columns: list[Tensor] = []
    with torch.no_grad():
        for coordinate in range(6):
            delta = torch.zeros_like(donor)
            delta[:, coordinate] = 1e-5
            columns.append((predict(donor + delta) - predict(donor - delta)) / 2e-5)
        numerical = torch.stack(columns, -1).reshape(groups * 10, 12, 6)
        singular = cast(Tensor, torch.linalg.svdvals(numerical))  # pyright: ignore[reportUnknownMemberType]
    assert math.isclose(
        result["predictor_jacobian_frobenius_rms"],
        float(numerical.square().sum((-1, -2)).mean().sqrt()),
        rel_tol=1e-7,
        abs_tol=1e-10,
    )
    torch.testing.assert_close(
        torch.tensor(result["predictor_jacobian_singular_values_median"], dtype=torch.float64),
        singular.median(0).values,
        rtol=1e-6,
        atol=1e-10,
    )


def test_sensitivity_handles_zero_reference_and_rejects_labels() -> None:
    model, batch = model_and_batch()
    result = predictor_code_sensitivity(
        model,
        batch,
        reference_stats=code_reference_stats(torch.zeros(2, 6)),
        max_groups=1,
    )
    comparison = result["source_code_comparison"]
    assert comparison["rms_ratio"] is None
    assert comparison["centered_rms_ratio"] is None
    assert comparison["mean_shift_in_source_centered_rms"] is None
    with pytest.raises(ValueError, match="label-free"):
        predictor_code_sensitivity(model, {**batch, "theta": torch.zeros(4, 6)})
