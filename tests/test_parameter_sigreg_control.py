import copy
import math
import sys
from collections.abc import Generator
from dataclasses import asdict
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.multiquery_control.objective import FrozenStateFastObjective
from diagnostics.parameter_sigreg_control import evaluation
from diagnostics.parameter_sigreg_control import run as runner
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.loop import ObjectiveConfig, TrainingObjective


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def tiny_model_and_batch() -> tuple[SCJEPA, dict[str, Tensor]]:
    torch.default_generator.manual_seed(4820)
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
    return model, {
        "online": torch.randn(4, 10, 12, 6, dtype=torch.float64),
        "target": torch.randn(4, 10, 12, 6, dtype=torch.float64),
        "actions": torch.randn(4, 10, 11, 16, dtype=torch.float64),
    }


def objective_config(weight: float) -> ObjectiveConfig:
    return ObjectiveConfig(
        state_regularizer="sigreg",
        parameter_regularizer="sigreg",
        state_regularizer_weight=0.4,
        parameter_regularizer_weight=weight,
        parameter_consistency_weight=1.0,
        regularizer_num_projections=8,
        logit_weight=0.0001,
    )


def test_only_parameter_sigreg_weight_changes_loss_and_gradient() -> None:
    model, batch = tiny_model_and_batch()
    output = cached_multiquery_forward(model, batch, queries=1, shift=1)
    losses: dict[float, Tensor] = {}
    gradients: dict[float, Tensor] = {}
    components: dict[float, dict[str, Tensor]] = {}
    assert isinstance(model.parameter_encoder, TransitionParameterEncoder)
    for weight in (0.4, 0.1, 0.0, 0.8, 1.6):
        torch.default_generator.manual_seed(948)
        losses[weight], components[weight] = FrozenStateFastObjective(
            objective_config(weight)
        ).forward(output)
        gradients[weight] = torch.autograd.grad(
            losses[weight], output.parameters, retain_graph=True
        )[0]
        torch.testing.assert_close(
            components[weight]["parameter_consistency"],
            2 * output.parameters.var(dim=1, correction=1).mean(),
            rtol=0,
            atol=0,
        )
    assert "parameter_regularizer" not in components[0.0]
    regularizer = components[0.4]["parameter_regularizer"]
    regularizer_gradient = torch.autograd.grad(regularizer, output.parameters)[0]
    for weight in (0.4, 0.1, 0.8, 1.6):
        torch.testing.assert_close(
            components[weight]["parameter_regularizer"], regularizer, rtol=0, atol=0
        )
        torch.testing.assert_close(
            losses[weight] - losses[0.0], weight * regularizer, rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            gradients[weight] - gradients[0.0],
            weight * regularizer_gradient,
            rtol=1e-10,
            atol=1e-12,
        )
    assert not any(parameter.requires_grad for parameter in model.online_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.target_encoder.parameters())


@pytest.mark.parametrize("weight", [0.4, 0.1, 0.0, 0.8, 1.6])
def test_fast_objective_only_omits_constant_frozen_state_sigreg(weight: float) -> None:
    model, batch = tiny_model_and_batch()
    output = cached_multiquery_forward(model, batch, queries=1, shift=1)
    full = TrainingObjective(objective_config(weight))
    fast = FrozenStateFastObjective(objective_config(weight))
    torch.default_generator.manual_seed(186)
    full_loss, full_terms = full.forward(output)
    rng = torch.get_rng_state()
    torch.default_generator.manual_seed(186)
    fast_loss, _ = fast.forward(output)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(
        full_loss - fast_loss, 0.4 * full_terms["state_regularizer"], rtol=1e-12, atol=1e-12
    )
    full_gradient = torch.autograd.grad(full_loss, output.parameters, retain_graph=True)[0]
    fast_gradient = torch.autograd.grad(fast_loss, output.parameters)[0]
    torch.testing.assert_close(full_gradient, fast_gradient, rtol=0, atol=0)


def test_zero_weight_never_evaluates_parameter_sigreg() -> None:
    class MustNotRun(nn.Module):
        def forward(self, _samples: Tensor) -> Tensor:
            raise AssertionError("Zero-weight parameter SIGReg was evaluated")

    model, batch = tiny_model_and_batch()
    output = cached_multiquery_forward(model, batch, queries=1, shift=1)
    objective = FrozenStateFastObjective(objective_config(0.0))
    objective.parameter_regularizer.loss = MustNotRun()
    loss, terms = objective.forward(output)
    assert torch.isfinite(loss)
    assert "parameter_regularizer" not in terms
    assert "parameter_consistency" in terms


def test_raw_consistency_can_shrink_without_prediction_or_information_change() -> None:
    model, batch = tiny_model_and_batch()
    model.eval()
    changed = copy.deepcopy(model)
    with torch.no_grad():
        assert isinstance(changed.parameter_encoder, TransitionParameterEncoder)
        head = changed.parameter_encoder.head[-1]
        assert isinstance(head, nn.Linear)
        head.weight.mul_(0.1)
        assert head.bias is not None
        head.bias.mul_(0.1)
        changed.predictor.param_project.weight.mul_(10)
        original = cached_multiquery_forward(model, batch, queries=1)
        scaled = cached_multiquery_forward(changed, batch, queries=1)
    torch.testing.assert_close(scaled.predictions, original.predictions, rtol=1e-10, atol=1e-12)
    before = original.parameters.var(dim=1, correction=1).mean()
    after = scaled.parameters.var(dim=1, correction=1).mean()
    torch.testing.assert_close(after, 0.01 * before, rtol=1e-12, atol=1e-15)
    before_total = original.parameters.flatten(0, 1).var(dim=0).mean()
    after_total = scaled.parameters.flatten(0, 1).var(dim=0).mean()
    torch.testing.assert_close(after / after_total, before / before_total, rtol=1e-12, atol=1e-12)


def test_sigreg_uses_independent_groups_per_view_without_pooling() -> None:
    class Capture(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.samples: Tensor | None = None

        def forward(self, samples: Tensor) -> Tensor:
            self.samples = samples
            return samples.square().mean()

    model, batch = tiny_model_and_batch()
    output = cached_multiquery_forward(model, batch, queries=1, shift=1)
    objective = TrainingObjective(objective_config(0.4))
    state_capture, parameter_capture = Capture(), Capture()
    objective.state_regularizer.loss = state_capture
    objective.parameter_regularizer.loss = parameter_capture
    objective.forward(output)
    assert parameter_capture.samples is not None
    assert state_capture.samples is not None
    assert parameter_capture.samples.shape == (10, 4, 6)
    assert state_capture.samples.shape[1] == 4
    torch.testing.assert_close(parameter_capture.samples, output.parameters.transpose(0, 1))


def test_config_and_cli_select_three_weights_one_seed_without_executing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocations: list[tuple[runner.Config, bool]] = []

    def capture(config: runner.Config, *, resume: bool = False) -> dict[str, Any]:
        invocations.append((config, resume))
        return {}

    monkeypatch.setattr(runner, "run", capture)
    monkeypatch.setattr(
        sys, "argv", ["run", "--checkpoint", "source.pt", "--output-dir", "comparison"]
    )
    runner.main()
    assert len(invocations) == 1
    config, resume = invocations[0]
    assert not resume
    assert config.parameter_sigreg_weights == (0.4, 0.1, 0.0)
    assert config.seeds == (202610261301,)
    assert config.evaluation_seed == 202610261601
    assert config.batch_size == 16
    assert config.steps == 30000
    assert config.wandb_group == "frozen-states-parameter-sigreg-control"
    assert [runner.condition_name(weight) for weight in config.parameter_sigreg_weights] == [
        "sigreg_0p4",
        "sigreg_0p1",
        "sigreg_0",
    ]
    with pytest.raises(ValueError, match="distinct"):
        runner.Config("source.pt", "comparison", parameter_sigreg_weights=(0.4, 0.4))
    with pytest.raises(ValueError, match="[Pp]arameter SIGReg weight"):
        runner.Config("source.pt", "comparison", parameter_sigreg_weights=(0.2,))


def test_cli_selects_only_two_requested_upper_weights_without_executing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocations: list[tuple[runner.Config, bool]] = []

    def capture(config: runner.Config, *, resume: bool = False) -> dict[str, Any]:
        invocations.append((config, resume))
        return {}

    monkeypatch.setattr(runner, "run", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--checkpoint",
            "source.pt",
            "--output-dir",
            "upper_comparison",
            "--parameter-sigreg-weights",
            "0.8",
            "1.6",
            "--device",
            "cuda:0",
        ],
    )
    runner.main()
    assert len(invocations) == 1
    config, resume = invocations[0]
    assert not resume
    assert config.parameter_sigreg_weights == (0.8, 1.6)
    assert [runner.condition_name(weight) for weight in config.parameter_sigreg_weights] == [
        "sigreg_0p8",
        "sigreg_1p6",
    ]
    reference = runner.Config(config.checkpoint, config.output_dir, device="cuda:0")
    expected = dict(asdict(reference), parameter_sigreg_weights=(0.8, 1.6))
    assert asdict(config) == expected


def test_arm_configs_only_change_parameter_weight_and_fix_consistency() -> None:
    source = objective_config(0.4)
    source.parameter_consistency_weight = 0.2
    protocol: dict[str, Any] = {"objective_config": asdict(source)}
    original = copy.deepcopy(protocol)
    for weight in (0.4, 0.1, 0.0, 0.8, 1.6):
        arm = runner.arm_objective(protocol, weight)
        expected = dict(asdict(source), parameter_regularizer_weight=weight)
        expected["parameter_consistency_weight"] = 1.0
        assert asdict(arm) == expected
        assert arm.state_regularizer_weight == 0.4
        assert protocol == original


def reference_objective() -> dict[str, Any]:
    row = {
        "prediction": 1.0,
        "parameter_regularizer": 2.0,
        "state_regularizer": 3.0,
        "parameter_consistency": 4.0,
        "total_at_lambda0": 16.0,
        "total_at_lambda1": 20.0,
        "total_at_training_lambda": 20.0,
        "base": 20.0,
    }
    return {
        "mean": row,
        "by_projection_seed": [dict(row, seed=43207.0)],
        "batch_groups": 4,
        "group_two_step_mse": [1.0] * 4,
    }


@pytest.mark.parametrize("weight", [0.4, 0.1, 0.0, 0.8, 1.6])
def test_objectives_are_scored_under_all_common_coefficients(weight: float) -> None:
    reference = reference_objective()
    saved = copy.deepcopy(reference)
    result = evaluation.objective_scores(reference, weight)
    for row in (result["mean"], result["by_projection_seed"][0]):
        assert math.isclose(row["total_at_sigreg_0p4"], 20.0)
        assert math.isclose(row["total_at_sigreg_0p1"], 19.4)
        assert math.isclose(row["total_at_sigreg_0"], 19.2)
        assert math.isclose(row["total_at_sigreg_0p8"], 20.8)
        assert math.isclose(row["total_at_sigreg_1p6"], 22.4)
        assert math.isclose(row["total_at_training_parameter_sigreg"], 20.0 + (weight - 0.4) * 2.0)
        assert row["parameter_consistency"] == 4.0
        assert row["state_regularizer"] == 3.0
        assert "total_at_lambda1" not in row
        assert "base" not in row
    assert result["consistency_weight"] == 1.0
    assert reference == saved


def test_probe_normalization_and_gain_are_invariant_to_small_uniform_code_scale() -> None:
    model, _ = tiny_model_and_batch()
    codes = {
        "calibration": torch.randn(4, 10, 6, dtype=torch.float64),
        "validation": torch.randn(4, 10, 6, dtype=torch.float64) + 3,
        "test": torch.randn(4, 10, 6, dtype=torch.float64) + 6,
    }
    normalized, normalization = evaluation.normalize_probe_codes(codes)
    scales = evaluation.code_scale_diagnostics(model, codes["test"])
    # The predictor shares a scalar-to-embedding projection across code coordinates.
    with torch.no_grad():
        projected = model.predictor.param_project(codes["test"].unsqueeze(-1))
        actual_token_std_rms = float(projected.flatten(0, 1).var(0, correction=0).mean().sqrt())
    assert math.isclose(
        scales["effective_parameter_token_std_rms"], actual_token_std_rms, rel_tol=1e-12
    )
    tiny_codes = {name: values * 1e-12 for name, values in codes.items()}
    normalized_tiny, tiny_normalization = evaluation.normalize_probe_codes(tiny_codes)
    with torch.no_grad():
        model.predictor.param_project.weight.mul_(1e12)
    tiny_scales = evaluation.code_scale_diagnostics(model, tiny_codes["test"])
    for name in codes:
        torch.testing.assert_close(normalized_tiny[name], normalized[name], rtol=1e-12, atol=1e-12)
    assert float(normalized["calibration"].mean().abs()) < 1e-15
    assert float(normalized["validation"].mean()) > 1
    assert math.isclose(
        tiny_normalization["calibration_global_std_rms"],
        normalization["calibration_global_std_rms"] * 1e-12,
        rel_tol=1e-12,
    )
    assert math.isclose(tiny_scales["code_rms"], scales["code_rms"] * 1e-12, rel_tol=1e-12)
    assert math.isclose(
        tiny_scales["parameter_projection_weight_norm"],
        scales["parameter_projection_weight_norm"] * 1e12,
    )
    assert math.isclose(
        tiny_scales["effective_parameter_token_std_rms"],
        scales["effective_parameter_token_std_rms"],
        rel_tol=1e-12,
    )
    assert not tiny_normalization["calibration_has_zero_variance"]


def test_constant_probe_codes_do_not_divide_by_zero() -> None:
    normalized, metadata = evaluation.normalize_probe_codes(
        {"calibration": torch.full((4, 10, 6), 3.0), "validation": torch.full((4, 10, 6), 3.0)}
    )
    assert metadata["calibration_has_zero_variance"]
    assert all(torch.equal(value, torch.zeros_like(value)) for value in normalized.values())


@pytest.mark.parametrize("weight", [0.4, 0.1, 0.0, 0.8, 1.6])
def test_evaluator_uses_reference_sigreg_even_for_zero_training_weight(
    monkeypatch: pytest.MonkeyPatch, weight: float
) -> None:
    model, _ = tiny_model_and_batch()
    metadata = {
        "state_online_sha256": module_fingerprint(model.online_encoder),
        "state_target_sha256": module_fingerprint(model.target_encoder),
    }
    split = {
        "theta": torch.randn(4, 6, dtype=torch.float64),
        "codes": torch.randn(4, 10, 6, dtype=torch.float64),
    }
    data: EvaluationData = {"metadata": metadata, "calibration": split, "validation": split}
    seen: list[ObjectiveConfig] = []

    def encode(_model: SCJEPA, data: TensorSplit, _batch_size: int) -> Tensor:
        return data["codes"]

    def common(
        model: SCJEPA,
        _split: TensorSplit,
        config: ObjectiveConfig,
        batch_size: int,
        final: bool,
    ) -> dict[str, Any]:
        assert not model.training
        assert batch_size == 4
        assert not final
        seen.append(config)
        torch.randn(5)
        return reference_objective()

    def linear(*_args: Tensor) -> list[float]:
        return [0.5] * 6

    monkeypatch.setattr(evaluation, "encode_codes", encode)
    monkeypatch.setattr(evaluation, "_common_objective", common)
    monkeypatch.setattr(evaluation, "_linear_progress", linear)
    config = objective_config(weight)
    saved = asdict(config)
    rng = torch.get_rng_state()
    result = evaluation.evaluate(model, data, config, batch_size=4)
    assert torch.equal(rng, torch.get_rng_state())
    assert asdict(config) == saved
    assert len(seen) == 1
    assert seen[0].parameter_regularizer_weight == 0.4
    assert seen[0].parameter_consistency_weight == 1.0
    assert result["raw_parameter_sigreg"] == 2.0
    assert result["raw_consistency"] == 4.0
    assert math.isclose(result["total_at_training_parameter_sigreg"], 20.0 + (weight - 0.4) * 2.0)
    totals = cast(dict[str, float], result["totals_at_parameter_sigreg"])
    for name, expected in {
        "sigreg_0p4": 20.0,
        "sigreg_0p1": 19.4,
        "sigreg_0": 19.2,
        "sigreg_0p8": 20.8,
        "sigreg_1p6": 22.4,
    }.items():
        assert math.isclose(totals[name], expected)
    assert model.training
    assert not model.online_encoder.training
    assert not model.target_encoder.training
