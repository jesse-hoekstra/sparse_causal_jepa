"""Invariant tests for the loss package: Hungarian predictive loss + regularizer."""

import pytest
import torch

from scjepa.losses import RegularizerKind, SlotRegularizer, hungarian_mse, match_slots

B, N, D = 4, 5, 16


@pytest.fixture
def pred() -> torch.Tensor:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    return torch.randn(B, N, D, requires_grad=True)


@pytest.fixture
def target() -> torch.Tensor:
    torch.manual_seed(1)  # pyright: ignore[reportUnknownMemberType]
    return torch.randn(B, N, D, requires_grad=True)


def test_zero_at_identity(pred: torch.Tensor) -> None:
    loss = hungarian_mse(pred, pred.detach().clone())
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_permutation_invariance(pred: torch.Tensor, target: torch.Tensor) -> None:
    """D6: permuting target slots must not change the matched loss."""
    baseline = hungarian_mse(pred, target)
    for _ in range(3):
        perm = torch.randperm(N)
        torch.testing.assert_close(hungarian_mse(pred, target[:, perm]), baseline)


def test_matches_aligned_mse_when_slots_correspond(pred: torch.Tensor) -> None:
    """With slots already aligned (small noise), matching must pick the identity."""
    target = pred.detach() + 0.01 * torch.randn_like(pred)
    expected = torch.nn.functional.mse_loss(pred, target)
    torch.testing.assert_close(hungarian_mse(pred, target), expected)
    assignment = match_slots(pred, target)
    identity = torch.arange(N).expand(B, N)
    assert torch.equal(assignment, identity)


def test_recovers_planted_permutation(pred: torch.Tensor) -> None:
    """match_slots must invert a planted slot shuffle."""
    perm = torch.randperm(N)
    shuffled = pred.detach()[:, perm]
    assignment = match_slots(pred, shuffled)
    # target slot assignment[i] equals pred slot i ⇒ assignment must be argsort-free
    # inverse: shuffled[:, assignment[b]] == pred[b]
    for b in range(B):
        torch.testing.assert_close(shuffled[b, assignment[b]], pred.detach()[b])


def test_gradients_flow_to_both_sides(pred: torch.Tensor, target: torch.Tensor) -> None:
    """Joint training (D7): the loss must send gradients into pred AND target."""
    loss = hungarian_mse(pred, target)
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0
    assert target.grad is not None
    assert target.grad.abs().sum() > 0


def test_rejects_shape_mismatch(pred: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="expected matching"):
        hungarian_mse(pred, torch.randn(B, N + 1, D))


@pytest.mark.parametrize("kind", ["visreg", "sigreg"])
def test_regularizer_detects_collapse(kind: RegularizerKind) -> None:
    """D3: a collapsed batch must be penalized far more than a well-spread one."""
    torch.manual_seed(2)  # pyright: ignore[reportUnknownMemberType]
    reg = SlotRegularizer(kind=kind)
    collapsed = torch.ones(B, N, D)  # every slot identical
    spread = torch.randn(64, N, D)  # ~ the loss's Gaussian reference
    collapsed_penalty = reg(collapsed)
    spread_penalty = reg(spread)
    assert torch.isfinite(collapsed_penalty)
    assert torch.isfinite(spread_penalty)
    assert collapsed_penalty > 10 * spread_penalty


@pytest.mark.parametrize("kind", ["visreg", "sigreg"])
def test_regularizer_grads_and_leading_axes(kind: RegularizerKind) -> None:
    torch.manual_seed(3)  # pyright: ignore[reportUnknownMemberType]
    reg = SlotRegularizer(kind=kind)
    slots = torch.randn(B, 3, N, D, requires_grad=True)  # (B, T, N, d) also allowed
    penalty = reg(slots)
    assert penalty.shape == ()
    penalty.backward()
    assert slots.grad is not None
    assert slots.grad.abs().sum() > 0


def test_regularizer_input_guards() -> None:
    reg = SlotRegularizer()
    with pytest.raises(ValueError, match="at least 2"):
        reg(torch.randn(1, D))
    with pytest.raises(ValueError, match="expected"):
        reg(torch.randn(D))
