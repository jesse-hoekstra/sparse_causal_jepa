import math
from collections.abc import Generator

import pytest
import torch
from test_multiquery_control import make_batch, make_model
from torch import Tensor

from diagnostics.group_contrastive_pretext.forward import (
    cached_codes,
    contrastive_loss,
    training_loss,
)
from diagnostics.multiquery_control.data import module_fingerprint


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_contrastive_matches_independent_anchor_loop_and_gradients() -> None:
    codes = torch.randn(3, 10, 6, generator=torch.Generator().manual_seed(17), dtype=torch.float64)
    codes.requires_grad_()
    flat = codes.flatten(0, 1)
    rows: list[Tensor] = []
    for anchor in range(len(flat)):
        indices = [other for other in range(len(flat)) if other != anchor]
        logits = torch.stack(
            [-0.5 * (flat[anchor] - flat[other]).square().sum() for other in indices]
        )
        probabilities = torch.log_softmax(logits, dim=0)
        positive = [slot for slot, other in enumerate(indices) if other // 10 == anchor // 10]
        assert len(positive) == 9
        rows.append(-probabilities[positive].mean())
    expected = torch.stack(rows).mean() - math.log(9)
    actual = contrastive_loss(codes)
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    actual_gradient = torch.autograd.grad(actual, codes, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, codes)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-14)
    assert bool(torch.isfinite(actual_gradient).all())
    assert float(actual_gradient.square().sum()) > 0


def test_group_view_translation_and_orthogonal_invariance() -> None:
    generator = torch.Generator().manual_seed(171)
    codes = torch.randn(3, 10, 6, generator=generator, dtype=torch.float64)
    expected = contrastive_loss(codes)
    groups = torch.tensor([2, 0, 1])
    shuffled = torch.stack(
        [codes[group, torch.randperm(10, generator=generator)] for group in groups]
    )
    torch.testing.assert_close(contrastive_loss(shuffled), expected, rtol=1e-14, atol=1e-14)
    translation = torch.randn(6, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(
        contrastive_loss(codes + translation), expected, rtol=1e-14, atol=1e-14
    )
    # Compose a coordinate permutation with a genuine rotation of a two-dimensional plane.
    basis = torch.eye(6, dtype=torch.float64)[:, torch.tensor([3, 1, 5, 0, 4, 2])]
    basis[:, [1, 3, 5]] *= -1
    rotation = torch.eye(6, dtype=torch.float64)
    angle = 0.73
    rotation[:2, :2] = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=torch.float64,
    )
    basis = basis @ rotation
    torch.testing.assert_close(contrastive_loss(codes @ basis), expected, rtol=1e-14, atol=1e-14)


def test_radial_information_is_retained_and_matches_closed_form_scale_gradient() -> None:
    # Same angular direction, different radii: normalization would erase all group information.
    codes = torch.zeros(2, 10, 6, dtype=torch.float64)
    codes[0, :, 0], codes[1, :, 0] = 1, 2
    scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    actual = contrastive_loss(scale * codes)
    expected = torch.log(9 + 10 * torch.exp(-scale.square() / 2)) - math.log(9)
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    observed_gradient = torch.autograd.grad(actual, scale, retain_graph=True)[0]
    exact_gradient = torch.autograd.grad(expected, scale)[0]
    torch.testing.assert_close(observed_gradient, exact_gradient, rtol=1e-14, atol=1e-14)
    assert float(observed_gradient) < 0
    assert float(contrastive_loss(codes * 3)) < float(actual.detach())
    collapsed = contrastive_loss(torch.zeros_like(codes))
    assert math.isclose(float(collapsed), math.log(19 / 9), rel_tol=1e-14)


def test_context_encoder_uses_only_individual_observed_context_and_no_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_model()
    model.requires_grad_(False)
    model.parameter_encoder.requires_grad_(True)
    _, batch = make_batch(model)
    modules = (model.online_encoder, model.target_encoder, model.predictor)
    before = [module_fingerprint(module) for module in modules]

    def forbidden(*args: object, **kwargs: object) -> Tensor:
        del args, kwargs
        raise AssertionError("Pretext used predictor or re-encoded cached states")

    for module in modules:
        monkeypatch.setattr(module, "forward", forbidden)
    codes = cached_codes(model, batch)
    assert codes.shape == (2, 10, 6)
    assert not torch.equal(codes[:, 0], codes[:, 1])
    altered = {name: value.clone() for name, value in batch.items()}
    altered["online"][:, :, 10:] += 100
    altered["actions"][:, :, 9:] += 100
    altered["target"] += 100
    torch.testing.assert_close(cached_codes(model, altered), codes, rtol=0, atol=0)
    altered["online"][:, 3, :10] += 10
    updated_codes = cached_codes(model, altered)
    other_views = [view for view in range(10) if view != 3]
    torch.testing.assert_close(updated_codes[:, other_views], codes[:, other_views], rtol=0, atol=0)
    assert not torch.equal(updated_codes[:, 3], codes[:, 3])
    torch.default_generator.manual_seed(211)
    rng = torch.get_rng_state()
    loss, terms = training_loss(model, batch)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(loss, contrastive_loss(codes), rtol=0, atol=0)
    torch.testing.assert_close(terms["optimized_total"], loss, rtol=0, atol=0)
    encoder_before = module_fingerprint(model.parameter_encoder)
    optimizer = torch.optim.Adam(model.parameter_encoder.parameters(), lr=0.001)
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    assert module_fingerprint(model.parameter_encoder) != encoder_before
    assert [module_fingerprint(module) for module in modules] == before
    assert all(value.grad is None for module in modules for value in module.parameters())


@pytest.mark.parametrize("extra", ["theta", "group_ids"])
def test_pretext_rejects_label_fields_and_attached_caches(extra: str) -> None:
    model = make_model()
    _, batch = make_batch(model)
    with pytest.raises(ValueError, match="excludes physical labels and group IDs"):
        cached_codes(model, {**batch, extra: torch.zeros(2, 6)})
    with pytest.raises(ValueError, match="must stay frozen"):
        cached_codes(model, {**batch, "online": batch["online"].requires_grad_()})


def test_contrastive_rejects_invalid_grouping_and_nonfinite_codes() -> None:
    for codes in (torch.zeros(2, 9, 6), torch.zeros(1, 10, 6), torch.zeros(2, 10, 1, 6)):
        with pytest.raises(ValueError, match="Bx10x6"):
            contrastive_loss(codes)
    with pytest.raises(ValueError, match="finite floating-point"):
        contrastive_loss(torch.full((2, 10, 6), float("nan")))
