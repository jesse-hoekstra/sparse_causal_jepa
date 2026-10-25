import pytest
import torch

from scjepa.models.parameter_encoder import (
    TransitionAttentionParameterEncoder,
    TransitionParameterEncoder,
)


@pytest.mark.parametrize("action_dim", [0, 3])
@pytest.mark.parametrize("history", [2, 6])
def test_independent_dimensions_and_attached_context(action_dim: int, history: int) -> None:
    torch.default_generator.manual_seed(40)
    encoder = TransitionAttentionParameterEncoder(
        5, 7, action_dim, dim=16, num_heads=4, max_history=6
    )
    states = torch.randn(2, history, 5, requires_grad=True)
    actions = torch.randn(2, history - 1, action_dim, requires_grad=True) if action_dim else None

    code = encoder(states, actions)
    assert code.shape == (2, 7)
    assert torch.isfinite(code).all()
    code.square().mean().backward()  # pyright: ignore[reportUnknownMemberType]

    assert states.grad is not None
    assert torch.isfinite(states.grad).all()
    assert (states.grad.abs().sum(dim=-1) > 0).all()
    if actions is not None:
        assert actions.grad is not None
        assert torch.isfinite(actions.grad).all()
        assert (actions.grad.abs().sum(dim=-1) > 0).all()
    for parameter in encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_episodes_do_not_share_context() -> None:
    torch.default_generator.manual_seed(41)
    encoder = TransitionAttentionParameterEncoder(5, 7, 3, dim=16).eval()
    states, actions = torch.randn(3, 6, 5), torch.randn(3, 5, 3)
    first_code = encoder(states[:1], actions[:1])
    torch.testing.assert_close(first_code, encoder(states, actions)[:1])

    states[1:] *= 100
    actions[1:] -= 100
    torch.testing.assert_close(first_code, encoder(states, actions)[:1])


def test_shared_mlp_and_downstream_initialization_remain_paired() -> None:
    torch.default_generator.manual_seed(43)
    old_encoder = TransitionParameterEncoder(6, 6, 16)
    old_rng = torch.get_rng_state()
    torch.default_generator.manual_seed(43)
    new_encoder = TransitionAttentionParameterEncoder(6, 6, 16)
    torch.testing.assert_close(torch.get_rng_state(), old_rng, rtol=0, atol=0)
    for name in ("transition", "head"):
        old_state = getattr(old_encoder, name).state_dict()
        new_state = getattr(new_encoder, name).state_dict()
        assert old_state.keys() == new_state.keys()
        for key in old_state:
            torch.testing.assert_close(old_state[key], new_state[key], rtol=0, atol=0)


def test_reordering_complete_transition_pairs_preserves_code() -> None:
    """Reorder closed excursions, retaining exactly the same directed transitions."""
    torch.default_generator.manual_seed(42)
    encoder = TransitionAttentionParameterEncoder(5, 7, 3, dim=16).eval()
    landmarks = torch.randn(2, 4, 5)
    # Each excursion starts at and returns to landmark zero; its two action
    # vectors remain attached to the same directed source/destination states.
    states = landmarks[:, [0, 1, 0, 2, 0, 3, 0]]
    actions = torch.randn(2, 6, 3)
    reordered_states = landmarks[:, [0, 3, 0, 1, 0, 2, 0]]
    reordered_actions = actions[:, [4, 5, 0, 1, 2, 3]]

    torch.testing.assert_close(
        encoder(states, actions), encoder(reordered_states, reordered_actions)
    )
    assert not torch.allclose(encoder(states, actions), encoder(states, actions.flip(1)))


def test_context_shape_guards() -> None:
    encoder = TransitionAttentionParameterEncoder(5, 7, 3, dim=16, max_history=4)
    with pytest.raises(ValueError, match="expected states"):
        encoder(torch.randn(2, 4, 6), torch.randn(2, 3, 3))
    with pytest.raises(ValueError, match="expected states"):
        encoder(torch.randn(2, 5))
    for history in (0, 1, 5):
        with pytest.raises(ValueError, match="context length"):
            encoder(torch.randn(2, history, 5))
    with pytest.raises(ValueError, match="nonempty"):
        encoder(torch.randn(0, 4, 5), torch.randn(0, 3, 3))
    with pytest.raises(ValueError, match="K-1"):
        encoder(torch.randn(2, 4, 5), torch.randn(2, 4, 3))
    with pytest.raises(ValueError, match="K-1"):
        encoder(torch.randn(2, 4, 5))
    with pytest.raises(ValueError, match="action_dim=0"):
        TransitionAttentionParameterEncoder(5, 7, dim=16)(
            torch.randn(2, 4, 5), torch.randn(2, 3, 1)
        )
    with pytest.raises(ValueError, match="positive"):
        TransitionAttentionParameterEncoder(5, 0)
    with pytest.raises(ValueError, match="max_history"):
        TransitionAttentionParameterEncoder(5, 7, max_history=1)
    with pytest.raises(ValueError, match="divide"):
        TransitionAttentionParameterEncoder(5, 7, dim=15, num_heads=4)
