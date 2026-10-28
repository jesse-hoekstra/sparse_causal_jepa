from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import torch

from diagnostics.mlp_oracle_code_control.model import MLPPredictor
from diagnostics.oracle_state_learning.core import (
    TRAINING_KEYS,
    Condition,
    LiveStateData,
    build_control,
    build_data,
    evaluate,
    forward_loss,
    objective_config,
)
from scjepa.data.factory import build_dgp_dataset
from scjepa.eval.recovery import fit_coordinate_match, mean_correlation_coefficient
from scjepa.training.loop import TrainingObjective


@pytest.fixture(autouse=True)
def single_thread() -> Generator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


def test_controls_share_fresh_encoder_and_spartan_weights_without_changing_rng() -> None:
    before = torch.get_rng_state()
    models = {name: build_control(name, 7) for name in ("dense", "baseline", "full", "mlp")}
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    reference = models["dense"]
    for model in models.values():
        torch.testing.assert_close(
            model.online_encoder.state_dict(), reference.online_encoder.state_dict()
        )
        torch.testing.assert_close(
            model.target_encoder.state_dict(), model.online_encoder.state_dict()
        )
        assert model.parameter_source == "oracle"
        assert model.ema_decay == 0.996
        assert all(
            not parameter.requires_grad for parameter in model.parameter_encoder.parameters()
        )
        assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())
    for name in ("baseline", "full"):
        torch.testing.assert_close(
            models[name].predictor.state_dict(), reference.predictor.state_dict()
        )
    assert all(layer.dense for layer in reference.predictor.layers)
    assert models["baseline"].predictor.parameter_gates_open
    assert not models["full"].predictor.parameter_gates_open
    assert isinstance(models["mlp"].predictor, MLPPredictor)
    assert sum(parameter.numel() for parameter in models["mlp"].predictor.parameters()) == 52123


def test_cache_preserves_original_draws_and_keeps_truth_out_of_train(tmp_path: Path) -> None:
    train = build_data("train", 2)
    calibration = build_data("calibration", 2)
    assert train.states is None
    assert calibration.config.split == "evaluation_calibration"
    assert frozenset(train.batch(torch.arange(2), "cpu")) == TRAINING_KEYS
    assert not bool(
        (train.tensors["episode_ids"][:, None] == calibration.tensors["episode_ids"][None]).any()
    )
    original = build_dgp_dataset(train.config).process.generate(0)
    torch.testing.assert_close(
        train.tensors["observations"][0], original.observations, rtol=0, atol=0
    )
    torch.testing.assert_close(
        train.tensors["oracle_parameters"][0], original.ground_truth["parameters"].float()
    )
    path = tmp_path / "cache.pt"
    torch.save(calibration.payload(), path)
    payload: dict[str, Any] = torch.load(path, weights_only=True)
    restored = LiveStateData.from_payload(payload)
    assert restored.fingerprint == calibration.fingerprint
    torch.testing.assert_close(restored.states, calibration.states)
    contaminated = dict(train.tensors, states=torch.zeros(2, 10, 12, 6))
    with pytest.raises(ValueError, match="Training tensors"):
        LiveStateData(train.config, contaminated, None, train.fingerprint, train.metadata)


@pytest.mark.parametrize("condition", ["dense", "full", "baseline", "mlp"])
def test_two_step_state_gradient_and_oracle_freezing(condition: Condition) -> None:
    model = build_control(condition, 3).train()
    data = build_data("train", 2)
    batch = data.batch(torch.arange(2), "cpu")
    batch["oracle_parameters"].requires_grad_()
    objective = TrainingObjective(objective_config(condition))
    loss, terms, result = forward_loss(model, batch, objective)
    first = result.dynamics[0].prediction
    first.retain_grad()
    result.predictions[:, :, 1].square().mean().backward(retain_graph=True)  # pyright: ignore[reportUnknownMemberType]
    assert first.grad is not None
    assert float(first.grad.abs().sum()) > 0
    model.zero_grad(set_to_none=True)
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    for module in (model.online_encoder, model.predictor):
        assert any(
            parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
            for parameter in module.parameters()
        )
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.parameter_encoder.parameters())
    assert batch["oracle_parameters"].grad is None
    assert "parameter_consistency" not in terms
    assert "parameter_regularizer" not in terms
    target_before = [parameter.clone() for parameter in model.target_encoder.parameters()]
    with torch.no_grad():
        for parameter in model.online_encoder.parameters():
            parameter.add_(0.1)
    model.update_target()
    for before, target, online in zip(
        target_before,
        model.target_encoder.parameters(),
        model.online_encoder.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(target, before * 0.996 + online * 0.004)
    with pytest.raises(ValueError, match="Training batch"):
        forward_loss(model, dict(batch, states=torch.zeros(2, 10, 12, 6)), objective)


def test_mlp_uses_grouped_state_sigreg_without_attention_outputs() -> None:
    model = build_control("mlp", 2)
    batch = build_data("train", 2).batch(torch.arange(2), "cpu")
    objective = TrainingObjective(objective_config("mlp"))
    rng = torch.get_rng_state()
    loss, terms, result = forward_loss(model, batch, objective)
    torch.set_rng_state(rng)
    expected_regularizer = objective.state_regularizer(
        result.online_states.flatten(1, 2).transpose(0, 1)
    )
    torch.testing.assert_close(terms["state_regularizer"], expected_regularizer)
    torch.testing.assert_close(loss, terms["prediction"] + 0.4 * expected_regularizer)
    assert "logit" not in terms
    assert "state_sparsity" not in terms
    with pytest.raises(ValueError, match="no attention"):
        forward_loss(model, batch, TrainingObjective(objective_config("full")))


def test_forward_randomness_does_not_shift_shared_sigreg_projection_stream() -> None:
    batch = build_data("train", 2).batch(torch.arange(2), "cpu")
    states: list[torch.Tensor] = []
    for condition in ("dense", "baseline", "full", "mlp"):
        model = build_control(condition, 2)
        objective = TrainingObjective(objective_config(condition))
        torch.default_generator.manual_seed(123)
        forward_loss(model, batch, objective, forward_seed=456)
        states.append(torch.get_rng_state())
    for state in states[1:]:
        torch.testing.assert_close(state, states[0], rtol=0, atol=0)


@pytest.mark.parametrize("condition", ["full", "mlp"])
def test_evaluation_is_reproducible_preserves_modes_and_keeps_group_axis(
    condition: Condition,
) -> None:
    model = build_control(condition, 5).train()
    objective = TrainingObjective(objective_config(condition)).train()
    calibration = build_data("calibration", 4)
    before = torch.get_rng_state()
    first = evaluate(model, calibration, 2, "cpu", objective)
    second = evaluate(model, calibration, 2, "cpu", objective)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    assert model.training
    assert model.online_encoder.training
    assert objective.training
    assert not model.target_encoder.training
    assert not model.parameter_encoder.training
    assert first.metrics == second.metrics
    torch.testing.assert_close(first.samples, second.samples)
    assert first.samples["online_states"].shape == (4, 10, 12, 6)
    assert first.samples["target_states"].shape == (4, 10, 12, 6)
    assert first.samples["predictions"].shape == (4, 10, 2, 6)
    assert ("graph_state" in first.samples) == (condition == "full")
    match = fit_coordinate_match(
        first.samples["online_states"].flatten(0, -2),
        first.samples["states"].flatten(0, -2),
        method="pearson",
    )
    validation = evaluate(model, build_data("validation", 4), 2, "cpu", objective)
    mcc = mean_correlation_coefficient(
        match,
        validation.samples["online_states"].flatten(0, -2),
        validation.samples["states"].flatten(0, -2),
    )
    assert 0 <= mcc <= 1
    assert first.metrics["online/variance_min"] > 0
    assert 0 < first.metrics["online/effective_rank"] <= 6
