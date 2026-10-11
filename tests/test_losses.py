"""Direct latent comparison and episode-wise representation statistics."""

import pytest
import torch
from torch import Tensor

from scjepa.losses import RepresentationRegularizer, prediction_mse


def test_prediction_mse_matches_coordinatewise_mean_and_stops_target_gradient() -> None:
    prediction = torch.randn(3, 4, 5, requires_grad=True)
    target = torch.randn(3, 4, 5, requires_grad=True)
    loss = prediction_mse(prediction, target)
    torch.testing.assert_close(loss, (prediction - target).square().mean())
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert prediction.grad is not None
    assert target.grad is None
    assert prediction_mse(target, target).item() == 0
    assert prediction_mse(target[:, :, [2, 1, 0, 3, 4]], target).item() > 0


def test_prediction_mse_rejects_broadcasting_or_empty_inputs() -> None:
    for prediction, target in (
        (torch.randn(2, 3, 4), torch.randn(2, 1, 4)),
        (torch.randn(2, 4), torch.randn(2, 4)),
        (torch.empty(2, 0, 4), torch.empty(2, 0, 4)),
    ):
        with pytest.raises(ValueError, match="matching|nonempty"):
            prediction_mse(prediction, target)


def test_sigreg_is_finite_and_differentiable() -> None:
    regularizer = RepresentationRegularizer("sigreg", num_projections=8)
    samples = torch.randn(2, 4, 5, requires_grad=True)
    loss: Tensor = regularizer(samples)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert samples.grad is not None
    assert torch.isfinite(samples.grad).all()
    assert samples.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="two episodes"):
        regularizer(torch.randn(2, 1, 5))


def test_no_regularizer_is_exact_zero() -> None:
    loss: Tensor = RepresentationRegularizer("none")(torch.randn(1, 1, 3))
    assert loss.item() == 0
    assert not loss.requires_grad


def test_visreg_is_not_an_active_regularizer() -> None:
    with pytest.raises(ValueError, match="none or sigreg"):
        RepresentationRegularizer("visreg")  # pyright: ignore[reportArgumentType]
