"""Tests for the parameter encoder P_eta (experiments.pdf Eqs. 16-26)."""

import pytest
import torch

from scjepa.models import ParameterEncoder

B, T, N, K = 2, 6, 4, 4


@pytest.fixture
def encoder() -> ParameterEncoder:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    return ParameterEncoder(state_dim=K, dim=16, num_heads=4, max_history=8)


@pytest.fixture
def states() -> torch.Tensor:
    torch.manual_seed(1)  # pyright: ignore[reportUnknownMemberType]
    return torch.randn(B, T, N, K)


def test_scalar_per_track(encoder: ParameterEncoder, states: torch.Tensor) -> None:
    """One unconstrained scalar coordinate per tracked object (Eq. 26)."""
    theta = encoder(states)
    assert theta.shape == (B, N, 1)
    assert torch.isfinite(theta).all()


def test_track_permutation_equivariance(encoder: ParameterEncoder, states: torch.Tensor) -> None:
    """No absolute track-index embedding: permuting tracks permutes theta (§6.2)."""
    encoder.eval()
    perm = torch.randperm(N)
    with torch.no_grad():
        theta = encoder(states)
        theta_perm = encoder(states[:, :, perm])
    torch.testing.assert_close(theta_perm, theta[:, perm], atol=1e-5, rtol=1e-4)


def test_relational_evidence_reaches_other_tracks(
    encoder: ParameterEncoder, states: torch.Tensor
) -> None:
    """Eq. 18: per-timestep cross-track attention lets partner states affect theta_i."""
    states = states.clone().requires_grad_(True)
    encoder(states)[0, 0, 0].backward()  # pyright: ignore[reportUnknownMemberType]
    assert states.grad is not None
    # Gradient w.r.t. a DIFFERENT track's state must be nonzero.
    assert states.grad[0, :, 1].abs().sum() > 0


def test_time_order_awareness(encoder: ParameterEncoder, states: torch.Tensor) -> None:
    """The learned temporal PE makes theta depend on the order of observations."""
    encoder.eval()
    with torch.no_grad():
        theta = encoder(states)
        theta_reversed = encoder(states.flip(dims=[1]))
    assert not torch.allclose(theta, theta_reversed, atol=1e-5)


def test_history_length_guard(encoder: ParameterEncoder) -> None:
    with pytest.raises(ValueError, match="max_history"):
        encoder(torch.randn(1, 9, N, K))
    with pytest.raises(ValueError, match="expected"):
        encoder(torch.randn(1, 9, K))


def test_no_normalization_after_scalar_head(encoder: ParameterEncoder) -> None:
    """Eq. 25: the head is a bare linear map (no norm may follow it)."""
    assert isinstance(encoder.head, torch.nn.Linear)
    assert encoder.head.out_features == 1
