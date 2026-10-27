import math
from collections.abc import Generator

import pytest
import torch
from test_multiquery_control import make_batch, make_model, objective_config
from torch import Tensor, nn

from diagnostics.group_latent_control.forward import clip_teacher_gradients, group_teacher_forward
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.multiquery_control.objective import FrozenStateFastObjective


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def test_teacher_matches_repeated_known_codes_objective_and_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_model()
    _, batch = make_batch(model)
    codes = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)

    def known_codes(_states: Tensor, _actions: Tensor) -> Tensor:
        return codes[:, None].expand(-1, 10, -1).reshape(20, 6)

    monkeypatch.setattr(model.parameter_encoder, "forward", known_codes)
    objective = FrozenStateFastObjective(objective_config())
    torch.default_generator.manual_seed(843)
    expected = cached_multiquery_forward(model, batch, queries=1)
    expected_loss, expected_terms = objective(expected)
    expected_rng = torch.get_rng_state()
    torch.default_generator.manual_seed(843)
    actual = group_teacher_forward(model, batch, codes)
    actual_loss, actual_terms = objective(actual)
    assert torch.equal(expected_rng, torch.get_rng_state())
    for original, teacher in (
        (expected.predictions, actual.predictions),
        (expected.parameters, actual.parameters),
        (expected.targets, actual.targets),
        (expected_loss, actual_loss),
    ):
        torch.testing.assert_close(original, teacher, rtol=0, atol=0)
    for name, original in expected_terms.items():
        torch.testing.assert_close(original, actual_terms[name], rtol=0, atol=0)
    assert actual_terms["parameter_consistency"] == 0
    expected_grad = torch.autograd.grad(expected_loss, codes)[0]
    actual_grad = torch.autograd.grad(actual_loss, codes)[0]
    torch.testing.assert_close(expected_grad, actual_grad, rtol=1e-12, atol=1e-14)


def test_teacher_two_step_rollout_attaches_code_prediction_and_predictor() -> None:
    model = make_model()
    _, batch = make_batch(model)
    codes = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
    output = group_teacher_forward(model, batch, codes)
    gradients = torch.autograd.grad(
        output.dynamics[1].prediction.square().mean(),
        (output.dynamics[0].prediction, codes, next(model.predictor.parameters())),
    )
    assert all(bool(torch.isfinite(g).all()) and bool(g.square().sum() > 0) for g in gradients)
    assert all(p.grad is None for p in model.parameter_encoder.parameters())
    with pytest.raises(ValueError, match="only frozen states"):
        group_teacher_forward(model, {**batch, "theta": codes}, codes)
    with pytest.raises(ValueError, match="must remain frozen"):
        group_teacher_forward(model, {**batch, "online": batch["online"].requires_grad_()}, codes)


def test_global_sparse_clip_and_unvisited_adam_rows_remain_unchanged() -> None:
    table = nn.Embedding(5, 6, sparse=True, dtype=torch.float64)
    optimizer = torch.optim.SparseAdam(table.parameters(), lr=0.01)
    predictor = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    table(torch.tensor([1])).square().sum().backward()
    clip_teacher_gradients([predictor], table)
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    optimizer.zero_grad(set_to_none=True)
    before_weight = table.weight.detach().clone()
    before_avg = optimizer.state[table.weight]["exp_avg"].clone()
    before_sq = optimizer.state[table.weight]["exp_avg_sq"].clone()
    table(torch.tensor([0, 0, 2])).sum().backward()
    predictor.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
    assert table.weight.grad is not None
    sparse_before = table.weight.grad.coalesce().values().clone()
    expected = float(torch.cat((sparse_before.flatten(), predictor.grad)).square().sum().sqrt())
    norm = clip_teacher_gradients([predictor], table)
    assert math.isclose(norm, expected, rel_tol=1e-14)
    coefficient = 1.0 / (expected + 1e-6)
    torch.testing.assert_close(table.weight.grad.values(), sparse_before * coefficient)
    torch.testing.assert_close(
        predictor.grad, torch.tensor([3.0, 4.0], dtype=torch.float64) * coefficient
    )
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    unvisited = torch.tensor([1, 3, 4])
    torch.testing.assert_close(table.weight[unvisited], before_weight[unvisited], rtol=0, atol=0)
    for key, before in (("exp_avg", before_avg), ("exp_avg_sq", before_sq)):
        torch.testing.assert_close(
            optimizer.state[table.weight][key][unvisited], before[unvisited], rtol=0, atol=0
        )
    assert not torch.equal(table.weight[[0, 2]], before_weight[[0, 2]])
