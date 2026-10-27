import copy
from collections.abc import Generator
from typing import Any

import pytest
import torch
from test_multiquery_control import make_batch, make_model, objective_config
from torch import nn

from diagnostics.group_latent_control.forward import clip_teacher_gradients
from diagnostics.group_latent_control.run import phase_optimizers, training_loss
from diagnostics.group_teacher_gradient_audit import (
    FRACTIONS,
    analyze_batch,
    component_gradients,
    optimizer_direction,
    summarize,
)
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


@pytest.mark.parametrize("warm_moments", [False, True])
def test_component_gradients_and_cloned_update_match_actual_teacher(warm_moments: bool) -> None:
    model = make_model()
    _, batch = make_batch(model)
    table = nn.Embedding.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        torch.randn(7, 6, dtype=torch.float64), freeze=False, sparse=True
    )
    indices = torch.tensor([4, 1])
    optimizer, latent = phase_optimizers(model, "teacher", table)
    assert latent is not None
    objective = PredictiveSupervisionObjective(objective_config(), 1.0)
    if warm_moments:
        loss, _ = training_loss(model, batch, indices, "teacher", table, objective)
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        clip_teacher_gradients(model.predictor.parameters(), table)
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        latent.step()  # pyright: ignore[reportUnknownMemberType]
    model.zero_grad(set_to_none=True)
    table.weight.grad = None
    original_weights = table.weight.detach().clone()
    original_parameters = tuple(p.detach().clone() for p in model.predictor.parameters())
    adam = copy.deepcopy(optimizer.state_dict()) if warm_moments else None
    sparse = copy.deepcopy(latent.state_dict()) if warm_moments else None
    gradients, terms = component_gradients(
        model, batch, table.weight[indices], objective_config(), 92
    )
    assert terms["consistency"] == 0.0
    assert all(bool(value.eq(0).all()) for value in gradients["sigreg"][1:])
    direction, _ = optimizer_direction(
        model, table.weight, indices, gradients["total"], adam, sparse
    )
    if warm_moments:
        torch.testing.assert_close(optimizer.state_dict(), adam, rtol=0, atol=0)
        torch.testing.assert_close(latent.state_dict(), sparse, rtol=0, atol=0)
    torch.manual_seed(92)  # pyright: ignore[reportUnknownMemberType]
    actual, _ = training_loss(model, batch, indices, "teacher", table, objective)
    actual.backward()  # pyright: ignore[reportUnknownMemberType]
    assert table.weight.grad is not None
    torch.testing.assert_close(table.weight.grad.to_dense()[indices], gradients["total"][0])
    for parameter, gradient in zip(
        model.predictor.parameters(), gradients["total"][1:], strict=True
    ):
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, gradient)
    clip_teacher_gradients(model.predictor.parameters(), table)
    latent.step()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    torch.testing.assert_close(table.weight[indices] - original_weights[indices], direction[0])
    for new, old, delta in zip(
        model.predictor.parameters(), original_parameters, direction[1:], strict=True
    ):
        torch.testing.assert_close(new - old, delta, rtol=0, atol=0)
    torch.testing.assert_close(
        table.weight[[0, 2, 3, 5, 6]], original_weights[[0, 2, 3, 5, 6]], rtol=0, atol=0
    )


def test_audit_preserves_sources_rng_and_matches_finite_prediction_slope() -> None:
    model = make_model()
    model.parameter_encoder.requires_grad_(False)
    _, batch = make_batch(model)
    weights = torch.randn(5, 6, dtype=torch.float64)
    original = weights.clone()
    before = copy.deepcopy(model.state_dict())
    state = torch.get_rng_state().clone()
    report, _ = analyze_batch(model, weights, batch, torch.tensor([1, 3]), objective_config(), 42)
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(original, weights)
    for branch in report["forecast"]["directions"].values():
        for block in branch.values():
            assert block["finite"]["0.01"]["linearization_valid"]
    with pytest.raises(ValueError, match="label-free"):
        analyze_batch(
            model,
            weights,
            {**batch, "theta": torch.zeros(2, 6)},
            torch.tensor([1, 3]),
            objective_config(),
            42,
        )


def test_bootstrap_averages_projection_replicates_before_paired_batch_sampling() -> None:
    records: list[list[dict[str, Any]]] = []
    for batch in range(2):
        replicas: list[dict[str, Any]] = []
        for projection in range(3):
            factual = float(batch + projection)
            replica: dict[str, Any] = {"forecast": {"directions": {}}}
            for name, value in (("factual", factual), ("without_sigreg", factual - 2)):
                replica["forecast"]["directions"][name] = {
                    block: {
                        "finite": {
                            str(fraction): {"actual_prediction_change": value}
                            for fraction in FRACTIONS
                        }
                    }
                    for block in ("joint", "table", "predictor")
                }
            replicas.append(replica)
        records.append(replicas)
    summary = summarize(records)
    assert summary["joint/1.0"]["factual"]["mean"] == 1.5
    assert summary["joint/1.0"]["paired_difference"] == {"mean": -2.0, "ci95": [-2.0, -2.0]}
