import copy
from collections.abc import Generator
from typing import Any

import pytest
import torch
from test_multiquery_control import make_batch, make_model, objective_config
from test_signal_path_geometry import tiny_data, tiny_model
from torch import Tensor, nn

from diagnostics.group_latent_control.run import (
    Phase,
    assess_model,
    initialize_table,
    phase_optimizers,
    training_loss,
)
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class NoTestAccess(dict[str, Any]):
    """Fail if a validation-only path asks for final-test data."""

    def __getitem__(self, key: str) -> object:
        """Expose calibration/validation data while poisoning the final-test slot."""
        if key == "test":
            raise AssertionError("Final test split was accessed during validation")
        return super().__getitem__(key)


def test_assessment_cannot_lookup_train_group_table_or_access_final_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model()
    data, _ = tiny_data(model)
    data["validation"] = data.pop("test")
    data["test"] = {"unreadable": True}
    guarded = NoTestAccess(data)
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()

    def forbidden_table_lookup(_self: nn.Embedding, _indices: Tensor) -> Tensor:
        raise AssertionError("A learned train-group table was used during evaluation")

    monkeypatch.setattr(nn.Embedding, "forward", forbidden_table_lookup)
    measured, raw = assess_model(
        model,
        guarded,
        DiagnosticSelection((0, 1), (0, 1)),
        bootstrap_samples=16,
        seed=21,
    )
    assert measured["response"]["split"] == "validation"
    assert len(measured["retention"]["selected_validation_r2"]) == 6
    assert [int(value) for value in raw["group_indices"]] == [0, 1]
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)


def test_amortization_updates_only_single_context_encoder_with_detached_teacher() -> None:
    model = make_model()
    _, batch = make_batch(model)
    table = initialize_table(model, batch, batch_size=2)
    with torch.no_grad():
        table.weight.add_(0.5)
    optimizer, latent_optimizer = phase_optimizers(model, "amortization", table)
    assert latent_optimizer is None
    assert not optimizer.state
    assert optimizer.param_groups[0]["lr"] == 0.001
    before_encoder = module_fingerprint(model.parameter_encoder)
    before_predictor = module_fingerprint(model.predictor)
    before_table = module_fingerprint(table)
    objective = PredictiveSupervisionObjective(objective_config(), 1.0)
    loss, terms = training_loss(
        model, batch, torch.tensor([0, 1]), "amortization", table, objective
    )
    assert terms["code_amortization"] > 0
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert table.weight.grad is None
    assert all(parameter.grad is None for parameter in model.predictor.parameters())
    assert any(
        parameter.grad is not None and bool(parameter.grad.square().sum() > 0)
        for parameter in model.parameter_encoder.parameters()
    )
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    assert module_fingerprint(model.parameter_encoder) != before_encoder
    assert module_fingerprint(model.predictor) == before_predictor
    assert module_fingerprint(table) == before_table
    assert all(not parameter.requires_grad for parameter in model.online_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())


def test_washout_equals_ordinary_objective_and_rejects_teacher_table() -> None:
    model = make_model()
    _, batch = make_batch(model)
    table = initialize_table(model, batch, batch_size=2)
    optimizer, latent_optimizer = phase_optimizers(model, "washout", None)
    assert not optimizer.state
    assert latent_optimizer is None
    model.train()
    objective = PredictiveSupervisionObjective(objective_config(), 1.0)
    torch.default_generator.manual_seed(678)
    actual, terms = training_loss(model, batch, torch.tensor([0, 1]), "washout", None, objective)
    actual_rng = torch.get_rng_state()
    torch.default_generator.manual_seed(678)
    expected, _ = objective(cached_multiquery_forward(model, batch, queries=1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), actual_rng)
    assert "code_amortization" not in terms
    with pytest.raises(ValueError, match="discard"):
        training_loss(model, batch, torch.tensor([0, 1]), "washout", table, objective)


@pytest.mark.parametrize("phase", ["teacher", "amortization", "washout"])
def test_no_phase_accepts_physical_parameter_labels(phase: Phase) -> None:
    model = make_model()
    _, batch = make_batch(model)
    table = None if phase == "washout" else initialize_table(model, batch, batch_size=2)
    objective = PredictiveSupervisionObjective(objective_config(), 1.0)
    with pytest.raises(ValueError, match="no physical parameters"):
        training_loss(
            model,
            {**batch, "theta": torch.zeros(2, 6)},
            torch.tensor([0, 1]),
            phase,
            table,
            objective,
        )
