"""Tests for the Experiment-1 prediction loss (experiments.pdf Eq. 39)."""

import pytest
import torch

from scjepa.losses import aligned_mse


def test_zero_at_identity() -> None:
    pred = torch.randn(4, 5, 4)
    assert aligned_mse(pred, pred).item() == 0


def test_matches_eq39_elementwise_mean() -> None:
    """L_pred = (1/(|I| N k)) sum of squared errors == element-wise MSE."""
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    pred, target = torch.randn(6, 5, 4), torch.randn(6, 5, 4)
    manual = (pred - target).square().sum() / pred.numel()
    torch.testing.assert_close(aligned_mse(pred, target), manual)


def test_keeps_known_object_correspondence() -> None:
    """Row identity is tracked: swapping rows changes the loss (no matching)."""
    torch.manual_seed(1)  # pyright: ignore[reportUnknownMemberType]
    target = torch.randn(2, 3, 4)
    pred = target[:, [1, 0, 2]]
    assert aligned_mse(pred, target).item() > 0


def test_gradients_flow() -> None:
    pred = torch.randn(2, 3, 4, requires_grad=True)
    aligned_mse(pred, torch.randn(2, 3, 4)).backward()  # pyright: ignore[reportUnknownMemberType]
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="matching"):
        aligned_mse(torch.randn(2, 3, 4), torch.randn(2, 4, 4))
    with pytest.raises(ValueError, match="matching"):
        aligned_mse(torch.randn(2, 3, 4, 1), torch.randn(2, 3, 4, 1))
