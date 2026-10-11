import pytest
import torch
from torch import nn

from scjepa.models import ParameterEncoder


@pytest.mark.parametrize("action_dim", [0, 2])
@pytest.mark.parametrize("history", [1, 5])
def test_independent_dimensions_and_context_lengths(action_dim: int, history: int) -> None:
    encoder = ParameterEncoder(7, 3, action_dim, dim=16, num_heads=4, max_history=6)
    states = torch.randn(2, history, 7, requires_grad=True)
    actions = (
        None if action_dim == 0 else torch.randn(2, history - 1, action_dim, requires_grad=True)
    )
    parameters = encoder(states, actions)
    assert parameters.shape == (2, 3)
    assert torch.isfinite(parameters).all()
    parameters.square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
    assert states.grad is not None
    assert states.grad.abs().sum() > 0
    if actions is not None and history > 1:
        assert actions.grad is not None
        assert actions.grad.abs().sum() > 0
    assert isinstance(encoder.head, nn.Linear)
    assert encoder.head.out_features == 3


def test_parameter_context_has_time_order() -> None:
    encoder = ParameterEncoder(5, 2, dim=16, max_history=6).eval()
    states = torch.randn(2, 6, 5)
    assert not torch.allclose(encoder(states), encoder(states.flip(1)), atol=1e-6)


def test_episode_independence() -> None:
    encoder = ParameterEncoder(5, 2, action_dim=3, dim=16)
    states, actions = torch.randn(3, 4, 5), torch.randn(3, 3, 3)
    first = encoder(states[:1], actions[:1])
    torch.testing.assert_close(first, encoder(states, actions)[:1])


def test_parameter_shape_guards() -> None:
    encoder = ParameterEncoder(5, 2, action_dim=3, max_history=4)
    with pytest.raises(ValueError, match="expected states"):
        encoder(torch.randn(2, 4, 7), torch.randn(2, 3, 3))
    with pytest.raises(ValueError, match="context length"):
        encoder(torch.randn(2, 5, 5), torch.randn(2, 4, 3))
    with pytest.raises(ValueError, match="K-1"):
        encoder(torch.randn(2, 4, 5), torch.randn(2, 4, 3))
    with pytest.raises(ValueError, match="K-1"):
        encoder(torch.randn(2, 4, 5))
    with pytest.raises(ValueError, match="action_dim=0"):
        ParameterEncoder(5, 2)(torch.randn(2, 4, 5), torch.randn(2, 3, 1))
    with pytest.raises(ValueError, match="positive"):
        ParameterEncoder(5, 0)
