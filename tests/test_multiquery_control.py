import copy
from collections.abc import Generator
from typing import cast

import pytest
import torch
from torch import Tensor

from diagnostics.multiquery_control.forward import cached_multiquery_forward, query_shifts
from diagnostics.multiquery_control.objective import FrozenStateFastObjective
from scjepa.models.model import SCJEPA
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.loop import ObjectiveConfig, TrainingObjective


@pytest.fixture(autouse=True)
def one_cpu_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def make_model() -> SCJEPA:
    torch.default_generator.manual_seed(4781)
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
    model.train()
    model.online_encoder.requires_grad_(False).eval()
    model.target_encoder.requires_grad_(False).eval()
    return model


def make_batch(model: SCJEPA) -> tuple[Tensor, dict[str, Tensor]]:
    observations = torch.randn(2, 10, 12, 6, dtype=torch.float64)
    actions = torch.randn(2, 10, 11, 16, dtype=torch.float64)
    with torch.no_grad():
        online = model.online_encoder.encode_sequence(observations.flatten(0, 1)).reshape(
            2, 10, 12, 6
        )
        target = torch.zeros_like(online)
        target[:, :, 10:12] = model.target_encoder.encode_sequence(
            observations[:, :, 10:12].flatten(0, 1)
        ).reshape(2, 10, 2, 6)
    return observations, {"online": online, "target": target, "actions": actions}


def objective_config(consistency: float = 1.0) -> ObjectiveConfig:
    return ObjectiveConfig(
        state_regularizer="sigreg",
        parameter_regularizer="sigreg",
        state_regularizer_weight=0.4,
        parameter_regularizer_weight=0.4,
        parameter_consistency_weight=consistency,
        regularizer_num_projections=8,
        logit_weight=0.0001,
    )


@pytest.mark.parametrize("consistency", [0.0, 1.0])
def test_q1_matches_production_forward_gradients_adam_and_rng(consistency: float) -> None:
    production = make_model()
    cached = copy.deepcopy(production)
    observations, batch = make_batch(production)
    objective = TrainingObjective(objective_config(consistency))
    optimizers = [
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        for model in (production, cached)
    ]
    torch.default_generator.manual_seed(679)
    original = production.forward(observations, batch["actions"])
    original_loss, _ = objective.forward(original)
    original_rng = torch.get_rng_state()
    torch.default_generator.manual_seed(679)
    copied = cached_multiquery_forward(cached, batch, queries=1)
    copied_loss, _ = objective.forward(copied)
    assert torch.equal(original_rng, torch.get_rng_state())
    for expected, actual in (
        (original.predictions, copied.predictions),
        (original.parameters, copied.parameters),
        (original.targets, copied.targets),
        (original_loss, copied_loss),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for loss, model, optimizer in zip(
        (original_loss, copied_loss), (production, cached), optimizers, strict=True
    ):
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    for expected, actual in zip(production.parameters(), cached.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if expected.grad is None:
            assert actual.grad is None
        else:
            assert actual.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


@pytest.mark.parametrize("queries", [1, 3, 9])
def test_balanced_nonself_pairs_and_regularizers_see_only_original_codes(queries: int) -> None:
    counts = torch.zeros(10, 10, dtype=torch.int64)
    for start in range(1, 10):
        shifts = query_shifts(queries, start)
        assert len(set(shifts)) == queries
        for recipient in range(10):
            for shift in shifts:
                source = (recipient - shift) % 10
                assert source != recipient
                counts[source, recipient] += 1
    assert bool((counts[~torch.eye(10, dtype=torch.bool)] == queries).all())
    model = make_model()
    _, batch = make_batch(model)
    objective = TrainingObjective(objective_config())
    torch.default_generator.manual_seed(817)
    expected = cached_multiquery_forward(model, batch, queries=1)
    _, expected_terms = objective.forward(expected)
    expected_rng = torch.get_rng_state()
    torch.default_generator.manual_seed(817)
    output = cached_multiquery_forward(model, batch, queries=queries)
    _, terms = objective.forward(output)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert output.parameters.shape == (2, 10, 6)
    assert output.online_states.shape == (2, 10, 12, 6)
    assert output.predictions.shape == (2, 10 * queries, 2, 6)
    for name in ("parameter_regularizer", "parameter_consistency", "state_regularizer"):
        torch.testing.assert_close(terms[name], expected_terms[name], rtol=0, atol=0)


def test_query_partitions_preserve_expected_loss_and_gradient_scale() -> None:
    model = make_model()
    _, batch = make_batch(model)
    objective = TrainingObjective(objective_config())
    parameters = [p for p in model.parameters() if p.requires_grad]

    def value_and_gradient(queries: int, shift: int) -> tuple[Tensor, Tensor]:
        torch.default_generator.manual_seed(765)
        output = cached_multiquery_forward(model, batch, queries=queries, shift=shift)
        loss, _ = objective.forward(output)
        gradients = cast(
            tuple[Tensor | None, ...], torch.autograd.grad(loss, parameters, allow_unused=True)
        )
        return loss.detach(), torch.cat(
            [
                (torch.zeros_like(p) if g is None else g).flatten()
                for p, g in zip(parameters, gradients, strict=True)
            ]
        )

    single = [value_and_gradient(1, shift) for shift in range(1, 10)]
    triple = [value_and_gradient(3, shift) for shift in range(1, 4)]
    all_pairs = value_and_gradient(9, 1)
    for part in (0, 1):
        expected = torch.stack([row[part] for row in single]).mean(dim=0)
        averaged = torch.stack([row[part] for row in triple]).mean(dim=0)
        torch.testing.assert_close(averaged, expected, rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(all_pairs[part], expected, rtol=1e-8, atol=1e-10)


def test_context_boundary_and_two_step_gradient_attachment() -> None:
    model = make_model()
    _, batch = make_batch(model)
    original = cached_multiquery_forward(model, batch, queries=3, shift=1)
    changed = {name: value.clone() for name, value in batch.items()}
    changed["online"][:, :, 10:] += 10
    changed["actions"][:, :, 9:] -= 10
    changed["target"] += 10
    output = cached_multiquery_forward(model, changed, queries=3, shift=1)
    torch.testing.assert_close(output.parameters, original.parameters, rtol=0, atol=0)
    changed["online"][:, 1:] += 100
    changed["actions"][:, 1:] -= 100
    other_contexts = cached_multiquery_forward(model, changed, queries=3, shift=1)
    torch.testing.assert_close(
        other_contexts.parameters[:, 0], original.parameters[:, 0], rtol=0, atol=0
    )
    second = original.dynamics[1].prediction.square().mean()
    gradients = torch.autograd.grad(second, (original.dynamics[0].prediction, original.parameters))
    assert all(bool(torch.isfinite(g).all()) and bool(g.square().sum() > 0) for g in gradients)


@pytest.mark.parametrize("consistency", [0.0, 1.0])
def test_fast_objective_preserves_full_gradients_rng_and_zero_consistency(
    consistency: float,
) -> None:
    model = make_model()
    _, batch = make_batch(model)
    output = cached_multiquery_forward(model, batch, queries=3, shift=1)
    parameters = [p for p in model.parameters() if p.requires_grad]
    full = TrainingObjective(objective_config(consistency))
    fast = FrozenStateFastObjective(objective_config(consistency))
    torch.default_generator.manual_seed(809)
    full_loss, full_terms = full.forward(output)
    expected_rng = torch.get_rng_state()
    torch.default_generator.manual_seed(809)
    fast_loss, fast_terms = fast.forward(output)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert ("parameter_consistency" in fast_terms) == bool(consistency)
    torch.testing.assert_close(
        full_loss - fast_loss, 0.4 * full_terms["state_regularizer"], rtol=1e-12, atol=1e-12
    )
    expected = cast(
        tuple[Tensor | None, ...],
        torch.autograd.grad(full_loss, parameters, allow_unused=True, retain_graph=True),
    )
    actual = cast(
        tuple[Tensor | None, ...], torch.autograd.grad(fast_loss, parameters, allow_unused=True)
    )
    for a, b in zip(actual, expected, strict=True):
        if a is None or b is None:
            assert a is None
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_rejects_labels_trainable_cache_and_unsupported_query_count() -> None:
    model = make_model()
    _, batch = make_batch(model)
    with pytest.raises(ValueError, match="only frozen states"):
        cached_multiquery_forward(model, {**batch, "theta": torch.zeros(2, 6)})
    with pytest.raises(ValueError, match="must remain frozen"):
        cached_multiquery_forward(model, {**batch, "online": batch["online"].requires_grad_()})
    with pytest.raises(ValueError, match="Q=1,3,9"):
        query_shifts(2, 1)
