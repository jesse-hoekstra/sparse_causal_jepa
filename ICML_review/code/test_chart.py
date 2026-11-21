"""Check physical-coordinate surgery in a mixed latent basis."""

import pytest
import torch

from .chart import AffineChart


def test_hold_preserves_other_physical_coordinates() -> None:
    """A nonorthogonal chart with translations supports the same hold in both lags."""
    physical = torch.randn(
        (100, 3), generator=torch.Generator().manual_seed(1), dtype=torch.float64
    )
    basis = physical.new_tensor([[1, 2, 0], [0, 3, 1], [-1, 0, 2]])
    latent = physical @ basis + physical.new_tensor([5, 2, -1])
    chart = AffineChart.fit(latent, physical)
    expected = physical.reshape(50, 2, 3).clone()
    expected[..., 1] = 0.75
    held = chart.clamp(latent.reshape(50, 2, 3), 1, 0.75)
    torch.testing.assert_close(chart.decode(held), expected)
    torch.testing.assert_close(chart.decode(latent), physical)


def test_rank_deficient_chart_fails() -> None:
    """A physical direction absent from the codes cannot be independently set."""
    x = torch.eye(4, dtype=torch.float64).repeat(2, 1)
    x[:, 0] = x[:, 1]
    with pytest.raises(ValueError, match="unique inverse"):
        AffineChart.fit(x, x)
