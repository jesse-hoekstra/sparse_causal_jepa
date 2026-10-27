import copy
from collections.abc import Generator
from typing import cast

import pytest
import torch
from test_multiquery_control import make_batch, make_model, objective_config
from torch import Tensor, nn

from diagnostics.group_latent_control.forward import group_teacher_forward
from diagnostics.group_latent_control.run import (
    Phase,
    TeacherObjectiveArm,
    phase_optimizers,
    scale_metrics,
    training_loss,
)
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


@pytest.mark.parametrize("arm", ["original", "without_parameter_sigreg"])
def test_teacher_objective_matches_original_or_retained_graph_and_rng(
    arm: TeacherObjectiveArm,
) -> None:
    model = make_model()
    _, batch = make_batch(model)
    table = nn.Embedding.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        torch.randn(4, 6, dtype=torch.float64), freeze=False, sparse=True
    )
    phase_optimizers(model, "teacher", table)
    expected_model, expected_table = copy.deepcopy(model), copy.deepcopy(table)
    objective = PredictiveSupervisionObjective(objective_config(), 1.0)
    indices = torch.tensor([0, 2])
    torch.manual_seed(419)  # pyright: ignore[reportUnknownMemberType]
    original, terms = objective(
        group_teacher_forward(expected_model, batch, expected_table(indices))
    )
    expected = (
        original
        if arm == "original"
        else (
            terms["weighted_prediction"]
            + objective.config.logit_weight * terms["logit"]
            + objective.config.parameter_consistency_weight * terms["parameter_consistency"]
        )
    )
    expected.backward()  # pyright: ignore[reportUnknownMemberType]
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(419)  # pyright: ignore[reportUnknownMemberType]
    actual, reported = training_loss(
        model, batch, indices, "teacher", table, objective, teacher_objective_arm=arm
    )
    if arm == "without_parameter_sigreg":
        derivative = torch.autograd.grad(
            actual, reported["parameter_regularizer"], allow_unused=True, retain_graph=True
        )[0]
        assert derivative is None
        assert actual != reported["original_total"]
    assert actual == expected
    assert reported["original_total"] == original
    assert float(reported["effective_parameter_regularizer_weight"]) == (
        0.4 if arm == "original" else 0.0
    )
    actual.backward()  # pyright: ignore[reportUnknownMemberType]
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(table.weight.grad, expected_table.weight.grad, rtol=0, atol=0)
    for actual_parameter, expected_parameter in zip(
        model.parameters(), expected_model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0)


@pytest.mark.parametrize("phase", ["amortization", "washout"])
def test_teacher_omission_cannot_change_later_phase_loss_gradients_or_rng(phase: Phase) -> None:
    model = make_model()
    _, batch = make_batch(model)
    table = (
        nn.Embedding.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            torch.randn(2, 6, dtype=torch.float64), freeze=False, sparse=True
        )
        if phase == "amortization"
        else None
    )
    phase_optimizers(model, phase, table)
    expected_model, expected_table = copy.deepcopy(model), copy.deepcopy(table)
    objective = PredictiveSupervisionObjective(objective_config(), 1.0)
    torch.manual_seed(731)  # pyright: ignore[reportUnknownMemberType]
    expected, expected_terms = training_loss(
        expected_model, batch, torch.tensor([0, 1]), phase, expected_table, objective
    )
    expected.backward()  # pyright: ignore[reportUnknownMemberType]
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(731)  # pyright: ignore[reportUnknownMemberType]
    actual, terms = training_loss(
        model,
        batch,
        torch.tensor([0, 1]),
        phase,
        table,
        objective,
        teacher_objective_arm="without_parameter_sigreg",
    )
    actual.backward()  # pyright: ignore[reportUnknownMemberType]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(terms, expected_terms, rtol=0, atol=0)
    assert float(terms["effective_parameter_regularizer_weight"]) == 0.4
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for first, second in zip(model.parameters(), expected_model.parameters(), strict=True):
        torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)


def test_scale_metrics_match_full_table_covariance_and_preserve_state_rng() -> None:
    model = make_model()
    table = nn.Embedding.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        torch.randn(8, 6, dtype=torch.float64), freeze=False, sparse=True
    )
    before = {name: module_fingerprint(module) for name, module in model.named_children()}
    table_before = table.weight.detach().clone()
    rng = torch.get_rng_state().clone()
    metrics = scale_metrics(model, table)
    assert torch.equal(torch.get_rng_state(), rng)
    assert before == {name: module_fingerprint(module) for name, module in model.named_children()}
    torch.testing.assert_close(table.weight, table_before, rtol=0, atol=0)
    assert table.weight.grad is None
    assert metrics["scale/teacher_table_rms"] == float(table_before.square().mean().sqrt())
    expected = cast(Tensor, torch.linalg.eigvalsh(torch.cov(table_before.T)))  # pyright: ignore[reportUnknownMemberType]
    actual = torch.tensor(
        [metrics[f"scale/teacher_table_covariance_eigenvalue_{i + 1}"] for i in range(6)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert scale_metrics(model, None)["scale/teacher_table_available"] == 0.0
