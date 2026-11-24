"""Compare the step-one sibling update against explicit disjoint targets and native gradients."""

import pytest
import torch
from torch import Tensor, nn

from .timing_loss import add_sibling_gradient, sibling_loss


def test_excludes_own_episode_and_freezes_targets_and_metric() -> None:
    """An independent scalar calculation checks both the value and the per-episode adjoint."""
    codes = torch.tensor(
        [[[1.0], [5.0], [2.0]], [[7.0], [8.0], [4.0]]], dtype=torch.float64, requires_grad=True
    )
    values = codes.detach()
    targets = torch.stack(
        [
            values[:, [other for other in range(3) if other != current]].mean(dim=1)
            for current in range(3)
        ],
        dim=1,
    )
    variance = values.var(correction=0) * (1 + 1e-6)
    residual = values - targets
    loss = sibling_loss(codes)
    (gradient,) = torch.autograd.grad(loss, codes)
    torch.testing.assert_close(loss, residual.square().mean() / variance)
    torch.testing.assert_close(gradient, 2 * residual / (variance * codes.numel()))


def test_invariant_to_rotation_and_common_rescaling_in_rank_deficient_chart() -> None:
    """Ridge-regularized Mahalanobis disagreement works with duplicate and constant axes."""
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(6, 5, 2, generator=generator, dtype=torch.float64)
    codes = torch.cat((x, x[..., :1], torch.ones_like(x[..., :1])), dim=-1)
    rotation = (
        torch.tensor([[3, 0, 4, 0], [0, 3, 0, 4], [-4, 0, 3, 0], [0, -4, 0, 3]], dtype=codes.dtype)
        / 5
    )
    torch.testing.assert_close(sibling_loss(codes), sibling_loss(3 * codes @ rotation + 2.2))


def test_identical_siblings_are_stationary_including_constant_chart() -> None:
    """The exactly collapsed initial chart and equally coded episodes give finite zeros."""
    for groups in ([[0.2, 0.7], [0.2, 0.7]], [[0.2, 0.7], [0.3, 0.8]]):
        codes = torch.tensor(groups, dtype=torch.float64)[:, None].repeat(1, 10, 1).requires_grad_()
        loss = sibling_loss(codes)
        (gradient,) = torch.autograd.grad(loss, codes)
        assert loss.item() == 0.0
        torch.testing.assert_close(gradient, torch.zeros_like(codes), atol=0.0, rtol=0.0)
    with pytest.raises(ValueError, match="multiple episodes"):
        sibling_loss(torch.zeros(1, 1, 2))


def test_capped_update_reaches_only_designated_encoder_and_consumes_no_rng() -> None:
    """Preserve native gradients at the input state and predictor weights through backward."""
    generator = torch.Generator().manual_seed(7)
    state = nn.Parameter(torch.randn(8, 4, 5, generator=generator))
    encoder = nn.Parameter(torch.randn(5, 3, generator=generator))
    predictor = nn.Parameter(torch.randn(3, 1, generator=generator))
    codes = state @ encoder
    forecast = (codes @ predictor).square().mean() * 1e-5
    native_state, native_encoder, native_predictor = torch.autograd.grad(
        forecast, (state, encoder, predictor), retain_graph=True
    )
    rng = torch.random.get_rng_state()
    stats = add_sibling_gradient(forecast, codes, (encoder,))
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert state.grad is None
    assert predictor.grad is None
    assert stats["consensus_cap_active"].item() == 1
    assert isinstance(encoder.grad, Tensor)
    torch.testing.assert_close(
        encoder.grad.square().sum().sqrt(),
        native_encoder.square().sum().sqrt(),
        atol=1e-9,
        rtol=1e-5,
    )
    forecast.backward()  # pyright: ignore[reportUnknownMemberType]
    torch.testing.assert_close(state.grad, native_state)
    torch.testing.assert_close(predictor.grad, native_predictor)
