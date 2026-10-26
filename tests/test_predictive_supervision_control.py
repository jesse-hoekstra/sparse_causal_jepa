import copy
import math
import sys
from collections.abc import Generator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.multiquery_control.data import EvaluationData, TensorSplit
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.multiquery_control.objective import FrozenStateFastObjective
from diagnostics.predictive_supervision_control import evaluation
from diagnostics.predictive_supervision_control import run as runner
from diagnostics.predictive_supervision_control.objective import (
    PredictiveSupervisionObjective,
    gradient_diagnostics,
)
from scjepa.models.model import SCJEPA, ModelOutput
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


def tiny_config() -> ModelConfig:
    return ModelConfig(
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


def tiny_model_and_batch() -> tuple[SCJEPA, TensorSplit]:
    torch.default_generator.manual_seed(4820)
    model = build_model(tiny_config()).double()
    model.online_encoder.requires_grad_(False).eval()
    model.target_encoder.requires_grad_(False).eval()
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


def flat_gradient(loss: Tensor, inputs: tuple[Tensor, ...]) -> Tensor:
    gradients = cast(
        tuple[Tensor | None, ...],
        torch.autograd.grad(loss, inputs, retain_graph=True, allow_unused=True),
    )
    return torch.cat(
        [
            (torch.zeros_like(value) if gradient is None else gradient).flatten()
            for value, gradient in zip(inputs, gradients, strict=True)
        ]
    )


def test_weight_one_matches_original_value_gradients_and_projection_rng_exactly() -> None:
    model, batch = tiny_model_and_batch()
    output = cached_multiquery_forward(model, batch, queries=1, shift=1)
    inputs = (output.parameters, *(p for p in model.parameters() if p.requires_grad))
    config = objective_config()
    torch.default_generator.manual_seed(948)
    original, original_terms = FrozenStateFastObjective(config)(output)
    original_rng = torch.get_rng_state()
    torch.default_generator.manual_seed(948)
    baseline, terms = PredictiveSupervisionObjective(config, 1.0)(output)
    assert torch.equal(torch.get_rng_state(), original_rng)
    torch.testing.assert_close(baseline, original, rtol=0, atol=0)
    torch.testing.assert_close(
        flat_gradient(baseline, inputs), flat_gradient(original, inputs), rtol=0, atol=0
    )
    for name, value in original_terms.items():
        torch.testing.assert_close(terms[name], value, rtol=0, atol=0)
    torch.testing.assert_close(terms["weighted_prediction"], terms["prediction"], rtol=0, atol=0)


@pytest.mark.parametrize("weight", [4.0, 16.0])
def test_only_prediction_value_and_gradient_change_for_both_trainable_modules(
    weight: float,
) -> None:
    model, batch = tiny_model_and_batch()
    output = cached_multiquery_forward(model, batch, queries=1, shift=1)
    config = objective_config()
    torch.default_generator.manual_seed(948)
    baseline, original_terms = PredictiveSupervisionObjective(config, 1.0)(output)
    baseline_rng = torch.get_rng_state()
    torch.default_generator.manual_seed(948)
    scaled, terms = PredictiveSupervisionObjective(config, weight)(output)
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    torch.testing.assert_close(
        scaled - baseline, (weight - 1) * original_terms["prediction"], rtol=1e-12, atol=1e-12
    )
    for name in ("prediction", "parameter_regularizer", "parameter_consistency", "logit"):
        torch.testing.assert_close(terms[name], original_terms[name], rtol=0, atol=0)
    for inputs in (
        (output.parameters,),
        tuple(model.parameter_encoder.parameters()),
        tuple(p for p in model.predictor.parameters() if p.requires_grad),
    ):
        predictive_gradient = flat_gradient(terms["prediction"], inputs)
        assert float(predictive_gradient.square().sum().sqrt()) > 0
        torch.testing.assert_close(
            flat_gradient(scaled, inputs) - flat_gradient(baseline, inputs),
            (weight - 1) * predictive_gradient,
            rtol=1e-10,
            atol=1e-12,
        )
    assert not any(p.requires_grad for p in model.online_encoder.parameters())
    assert not any(p.requires_grad for p in model.target_encoder.parameters())


def test_gradient_measurement_preserves_rng_grad_buffers_and_training_backward() -> None:
    model, batch = tiny_model_and_batch()
    output = cached_multiquery_forward(model, batch, queries=1, shift=1)
    config = objective_config()
    loss, terms = PredictiveSupervisionObjective(config, 4.0)(output)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    expected = flat_gradient(loss, parameters)
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 0.123)
    rng = torch.get_rng_state()
    metrics = gradient_diagnostics(model, output, terms, config, 4.0)
    assert torch.equal(torch.get_rng_state(), rng)
    for parameter in parameters:
        assert parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad, torch.full_like(parameter, 0.123), rtol=0, atol=0
        )
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["encoder/prediction_norm"] > 0
    assert metrics["predictor/logit_norm"] > 0
    for block, inputs in (
        ("encoder", tuple(model.parameter_encoder.parameters())),
        ("predictor", tuple(p for p in model.predictor.parameters() if p.requires_grad)),
    ):
        expected_logit_norm = float(
            flat_gradient(config.logit_weight * terms["logit"], inputs).square().sum().sqrt()
        )
        assert math.isclose(metrics[f"{block}/logit_norm"], expected_logit_norm, rel_tol=1e-12)
    assert metrics["predictor/sigreg_norm"] == 0
    assert metrics["predictor/prediction_sigreg_cosine_defined"] == 0
    assert metrics["predictor/prediction_sigreg_cosine"] == 0
    assert metrics["code/prediction_logit_cosine_defined"] == 1
    model.zero_grad(set_to_none=True)
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    actual = torch.cat(
        [(torch.zeros_like(p) if p.grad is None else p.grad).flatten() for p in parameters]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gradient_measurement_handles_constant_and_connected_zero_losses() -> None:
    model, batch = tiny_model_and_batch()
    output = cached_multiquery_forward(model, batch, queries=1, shift=1)
    config = objective_config()
    _, terms = PredictiveSupervisionObjective(config, 1.0)(output)
    terms["parameter_regularizer"] = output.parameters.new_zeros(())
    terms["parameter_consistency"] = output.parameters.sum() * 0
    metrics = gradient_diagnostics(model, output, terms, config, 1.0)
    for block in ("code", "encoder", "predictor"):
        assert metrics[f"{block}/sigreg_norm"] == 0
        assert metrics[f"{block}/consistency_norm"] == 0
        assert metrics[f"{block}/sigreg_consistency_cosine"] == 0
        assert metrics[f"{block}/sigreg_consistency_cosine_defined"] == 0
    model.parameter_encoder.requires_grad_(False)
    with pytest.raises(ValueError, match="trainable"):
        gradient_diagnostics(model, output, terms, config, 1.0)


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf")])
def test_objective_rejects_invalid_prediction_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        PredictiveSupervisionObjective(objective_config(), weight)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state_regularizer_weight", 0.1),
        ("parameter_regularizer_weight", 0.1),
        ("parameter_consistency_weight", 0.5),
        ("state_sparsity_weight", 0.1),
        ("action_sparsity_weight", 0.1),
        ("parameter_sparsity_weight", 0.1),
    ],
)
def test_objective_rejects_changes_to_fixed_coefficients(field: str, value: float) -> None:
    config = objective_config()
    setattr(config, field, value)
    with pytest.raises(ValueError, match="dense SIGReg"):
        PredictiveSupervisionObjective(config, 4.0)


def test_source_logit_weight_is_preserved_by_prediction_intervention() -> None:
    model, batch = tiny_model_and_batch()
    output = cached_multiquery_forward(model, batch, queries=1, shift=1)
    config = objective_config()
    config.logit_weight = 0.03
    inputs = tuple(p for p in model.parameters() if p.requires_grad)
    torch.default_generator.manual_seed(948)
    original, _ = FrozenStateFastObjective(config)(output)
    torch.default_generator.manual_seed(948)
    scaled, terms = PredictiveSupervisionObjective(config, 4.0)(output)
    torch.testing.assert_close(
        flat_gradient(scaled, inputs),
        flat_gradient(original, inputs) + 3 * flat_gradient(terms["prediction"], inputs),
        rtol=1e-10,
        atol=1e-12,
    )


def test_cli_selects_paired_three_weight_control_without_executing(
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
    config, resume = invocations[0]
    assert len(invocations) == 1
    assert not resume
    assert config.prediction_weights == (1.0, 4.0, 16.0)
    assert config.seeds == (202610261301,)
    assert config.evaluation_seed == 202610261601
    assert config.batch_size == 16
    assert config.steps == 30000
    assert config.gradient_every == 1000
    assert [runner.condition_name(weight) for weight in config.prediction_weights] == [
        "prediction_1",
        "prediction_4",
        "prediction_16",
    ]
    with pytest.raises(ValueError, match="distinct"):
        replace(config, prediction_weights=(1.0, 1.0))
    with pytest.raises(ValueError, match="Prediction weight"):
        replace(config, prediction_weights=(0.0,))
    with pytest.raises(ValueError, match="gradient_every must be positive"):
        replace(config, gradient_every=0)


def test_arm_objectives_preserve_all_regularizer_settings() -> None:
    source = objective_config()
    protocol: dict[str, Any] = {"objective_config": asdict(source)}
    original = copy.deepcopy(protocol)
    for weight in (1.0, 4.0, 16.0):
        assert asdict(runner.arm_objective(protocol, weight)) == asdict(source)
        assert protocol == original


def reference_objective() -> dict[str, Any]:
    row = {
        "prediction": 2.0,
        "parameter_regularizer": 3.0,
        "state_regularizer": 4.0,
        "parameter_consistency": 5.0,
        "total_at_sigreg_0p4": 20.0,
        "total_at_training_parameter_sigreg": 20.0,
    }
    return {
        "mean": row,
        "by_projection_seed": [dict(row, seed=43207.0)],
        "group_two_step_mse": [1.0, 3.0, 1.0, 3.0],
    }


@pytest.mark.parametrize("weight", [1.0, 4.0, 16.0])
def test_common_scoring_changes_totals_without_changing_raw_prediction(weight: float) -> None:
    reference = reference_objective()
    original = copy.deepcopy(reference)
    scores = evaluation.objective_scores(reference, weight)
    for row in (scores["mean"], scores["by_projection_seed"][0]):
        assert row["prediction"] == 2.0
        assert row["total_at_prediction_1"] == 20.0
        assert row["total_at_prediction_4"] == 26.0
        assert row["total_at_prediction_16"] == 50.0
        assert row["total_at_training_prediction_weight"] == 20.0 + (weight - 1) * 2
    assert scores["group_two_step_mse"] == reference["group_two_step_mse"]
    assert reference == original


def test_evaluation_keeps_raw_errors_readouts_and_predictive_use_comparable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = tiny_model_and_batch()
    config = objective_config()
    original_config = asdict(config)
    reference: dict[str, Any] = {
        "prediction_mse": 2.0,
        "raw_parameter_sigreg": 3.0,
        "raw_consistency": 5.0,
        "parameter_sigreg_weight": 0.4,
        "objective": reference_objective(),
        "total_at_training_parameter_sigreg": 20.0,
        "totals_at_parameter_sigreg": {"sigreg_0p4": 20.0},
        "parameter_readouts": {"joint_r2": [0.5] * 6},
        "parameter_use_shuffle": {"delta_mse": 0.1},
        "true_parameter_interventions": {"cosines": [0.6] * 6},
    }
    calls: list[float] = []

    def base(
        _model: SCJEPA,
        _data: EvaluationData,
        objective: ObjectiveConfig,
        *,
        batch_size: int,
        final: bool,
    ) -> dict[str, Any]:
        assert batch_size == 4
        assert final
        assert asdict(objective) == original_config
        calls.append(objective.parameter_regularizer_weight)
        return copy.deepcopy(reference)

    monkeypatch.setattr(evaluation, "base_evaluate", base)
    for weight in (1.0, 4.0, 16.0):
        result = evaluation.evaluate(
            model, {}, config, prediction_weight=weight, batch_size=4, final=True
        )
        assert result["prediction_mse"] == 2.0
        assert result["weighted_prediction"] == weight * 2.0
        assert result["totals_at_prediction_weights"] == {
            "prediction_1": 20.0,
            "prediction_4": 26.0,
            "prediction_16": 50.0,
        }
        for key in ("parameter_readouts", "parameter_use_shuffle", "true_parameter_interventions"):
            assert result[key] == reference[key]
        assert "total_at_training_parameter_sigreg" not in result
    assert calls == [0.4] * 3
    assert asdict(config) == original_config


def test_tiny_training_resume_matches_uninterrupted_weights_optimizer_and_gradient_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model, batch = tiny_model_and_batch()
    model.float()
    train = {key: value.float() for key, value in batch.items()}
    initial = tmp_path / "initial.pt"
    torch.save({"model": model.state_dict()}, initial)
    frozen = runner.frozen_hashes(model)

    def prepare(config: runner.Config, device: torch.device, *, resume: bool) -> dict[str, Any]:
        assert device.type == "cpu"
        out = Path(config.output_dir)
        if not resume:
            (out / "initial.pt").write_bytes(initial.read_bytes())
            torch.save(train, out / "train_inputs.pt")
            torch.save({}, out / "evaluation_data.pt")
        return {
            "identity": "synthetic-test",
            "model_config": asdict(tiny_config()),
            "objective_config": asdict(objective_config()),
            "frozen_hashes": frozen,
            "source": {"sha256": "synthetic-source"},
            "initializations": {"7": {"path": "initial.pt", "sha256": runner.sha(initial)}},
        }

    def evaluate(
        candidate: SCJEPA,
        _data: EvaluationData,
        objective: ObjectiveConfig,
        *,
        prediction_weight: float,
        prediction_weights: tuple[float, ...],
        batch_size: int,
        final: bool,
    ) -> dict[str, Any]:
        assert runner.frozen_hashes(candidate) == frozen
        assert objective.parameter_regularizer_weight == 0.4
        assert objective.parameter_consistency_weight == 1.0
        assert prediction_weight == 4.0
        assert prediction_weights == (4.0,)
        assert batch_size == 2
        torch.randn(3)
        return {"prediction_mse": 0.25, "final": final}

    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(runner, "evaluate", evaluate)
    config = runner.Config(
        str(initial),
        str(tmp_path / "uninterrupted"),
        seeds=(7,),
        prediction_weights=(4.0,),
        steps=3,
        train_groups=4,
        calibration_groups=4,
        validation_groups=4,
        test_groups=4,
        batch_size=2,
        log_every=1,
        checkpoint_every=1,
        evaluate_every=1,
        gradient_every=1,
    )
    runner.run(config)
    interrupted = replace(config, output_dir=str(tmp_path / "interrupted"))
    calls = 0

    def interrupt(candidate: SCJEPA, values: TensorSplit, *, queries: int) -> ModelOutput:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return cached_multiquery_forward(candidate, values, queries=queries)

    monkeypatch.setattr(runner, "cached_multiquery_forward", interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        runner.run(interrupted)
    relative = Path("seed-7/prediction_4")
    interrupted_dir = Path(interrupted.output_dir) / relative
    checkpoint = cast(dict[str, Any], torch.load(interrupted_dir / "last.pt", weights_only=False))
    assert checkpoint["step"] == 1
    # A crash may leave the standalone history newer than the durable checkpoint.
    (interrupted_dir / "gradient_history.json").write_text('{"history": "stale"}')
    monkeypatch.setattr(runner, "cached_multiquery_forward", cached_multiquery_forward)
    result = runner.run(interrupted, resume=True)
    reference = cast(
        dict[str, Any],
        torch.load(Path(config.output_dir) / relative / "last.pt", weights_only=False),
    )
    restored = cast(dict[str, Any], torch.load(interrupted_dir / "last.pt", weights_only=False))
    assert restored["step"] == 3
    assert restored["gradient_history"] == reference["gradient_history"]
    assert restored["clipped_updates"] == reference["clipped_updates"]
    assert [row["step"] for row in restored["gradient_history"]] == [1, 2, 3]
    torch.testing.assert_close(restored["model"], reference["model"], rtol=0, atol=0)
    torch.testing.assert_close(restored["optimizer"], reference["optimizer"], rtol=0, atol=0)
    for name, value in model.state_dict().items():
        if name.startswith(("online_encoder.", "target_encoder.")):
            torch.testing.assert_close(restored["model"][name], value, rtol=0, atol=0)
    assert runner.run(interrupted, resume=True) == result
    (interrupted_dir / "gradient_history.json").write_text("{}")
    with pytest.raises(ValueError, match="artifacts changed"):
        runner.run(interrupted, resume=True)
