from pathlib import Path

import pytest
import torch
from torch import Tensor

from scjepa.models import (
    SCJEPA,
    FramewiseEncoder,
    ImageEncoder,
    ParameterEncoder,
    Spartan,
    SpartanOutput,
    VectorEncoder,
)


class RecordingSpartan(Spartan):
    """Capture actual call inputs to verify finite-lag and attachment contracts."""

    def __init__(self, history_len: int, action_dim: int) -> None:
        """Create a deterministic dense predictor with independently sized inputs."""
        super().__init__(
            state_dim=5,
            param_dim=3,
            action_dim=action_dim,
            history_len=history_len,
            embed_dim=8,
            num_layers=1,
            dense=True,
        )
        self.calls: list[tuple[Tensor, Tensor, Tensor | None]] = []

    def forward(
        self, states: Tensor, parameters: Tensor, actions: Tensor | None = None
    ) -> SpartanOutput:
        """Record each untouched argument before delegating to the predictor."""
        self.calls.append((states, parameters, actions))
        return super().forward(states, parameters, actions)


class RecordingParameterEncoder(ParameterEncoder):
    """Count parameter inference independently of the number of predictions."""

    def __init__(self, action_dim: int) -> None:
        """Initialize a small encoder and an inference counter."""
        super().__init__(5, 3, action_dim=action_dim, dim=8, num_heads=2, max_history=8)
        self.calls = 0

    def forward(self, states: Tensor, actions: Tensor | None = None) -> Tensor:
        """Count one context inference before normal attention and pooling."""
        self.calls += 1
        return super().forward(states, actions)


def make_model(history_len: int = 1, action_dim: int = 0, context_len: int = 3) -> SCJEPA:
    return SCJEPA(
        VectorEncoder(7, 5, hidden_dims=(12,)),
        RecordingParameterEncoder(action_dim),
        RecordingSpartan(history_len, action_dim),
        context_len=context_len,
        ema_decay=0.8,
    )


@pytest.mark.parametrize("image", [False, True])
def test_frame_encoding_is_independent_in_training_mode(image: bool) -> None:
    encoder: FramewiseEncoder
    if image:
        encoder = ImageEncoder(3, 5, channels=(8, 8))
        observations = torch.randn(2, 4, 3, 12, 12, requires_grad=True)
    else:
        encoder = VectorEncoder(7, 5, hidden_dims=(12,))
        observations = torch.randn(2, 4, 7, requires_grad=True)
    encoder.train()
    together = encoder.encode_sequence(observations)
    alone = encoder(observations[1:2, 2])
    torch.testing.assert_close(together[1:2, 2], alone)
    altered = observations.detach().clone()
    altered[:, 3] += 10
    altered[0] -= 20
    torch.testing.assert_close(together[1:2, 2], encoder.encode_sequence(altered)[1:2, 2])
    together[1, 2].sum().backward()  # pyright: ignore[reportUnknownMemberType]
    assert observations.grad is not None
    assert observations.grad[1, 2].abs().sum() > 0
    unrelated_grad = observations.grad.clone()
    unrelated_grad[1, 2] = 0
    assert torch.count_nonzero(unrelated_grad) == 0


@pytest.mark.parametrize("history_len", [1, 2, 4])
@pytest.mark.parametrize("action_dim", [0, 2])
def test_history_action_indexing_and_single_parameter_inference(
    history_len: int, action_dim: int
) -> None:
    model = make_model(history_len, action_dim)
    observations = torch.randn(2, 7, 7)
    actions = None if action_dim == 0 else torch.randn(2, 6, action_dim)
    out = model(observations, actions)
    assert isinstance(model.parameter_encoder, RecordingParameterEncoder)
    assert model.parameter_encoder.calls == 1
    start = max(3, history_len) - 1
    assert out.predictions.shape == out.targets.shape == (2, 6 - start, 5)
    assert out.online_states.shape == (2, 7, 5)
    assert out.parameters.shape == (2, 3)
    assert out.prediction_start == start
    predictor = model.predictor
    assert isinstance(predictor, RecordingSpartan)
    for offset, (states, parameters, window_actions) in enumerate(predictor.calls):
        source = start + offset
        torch.testing.assert_close(
            states, out.online_states[:, source - history_len + 1 : source + 1]
        )
        assert parameters is out.parameters
        assert parameters.requires_grad
        if actions is not None:
            torch.testing.assert_close(
                window_actions, actions[:, source - history_len + 1 : source + 1]
            )
        else:
            assert window_actions is None
    expected_params = model.parameter_encoder(
        out.online_states[:, :3], None if actions is None else actions[:, :2]
    )
    torch.testing.assert_close(out.parameters, expected_params)
    with torch.no_grad():
        expected_targets = model.target_encoder.encode_sequence(observations[:, start + 1 :])
    torch.testing.assert_close(out.targets, expected_targets)


def test_future_frames_and_actions_do_not_leak_into_context_or_first_prediction() -> None:
    model = make_model(history_len=2, action_dim=2).eval()
    observations, actions = torch.randn(2, 7, 7), torch.randn(2, 6, 2)
    out = model(observations, actions)
    changed_observations, changed_actions = observations.clone(), actions.clone()
    changed_observations[:, 3:] += 7
    changed_actions[:, 3:] -= 4
    changed = model(changed_observations, changed_actions)
    torch.testing.assert_close(out.parameters, changed.parameters)
    torch.testing.assert_close(out.predictions[:, 0], changed.predictions[:, 0])
    changed_actions[:, 2] += 20  # a_2 predicts x_3; it is not part of parameter context.
    torch.testing.assert_close(out.parameters, model(observations, changed_actions).parameters)


@pytest.mark.parametrize("history_len", [1, 3])
@pytest.mark.parametrize("action_dim", [0, 2])
def test_rollout_shifts_history_and_reuses_attached_parameters(
    history_len: int, action_dim: int
) -> None:
    model = make_model(history_len, action_dim)
    states = torch.randn(2, history_len, 5)
    parameters = torch.randn(2, 3, requires_grad=True)
    actions = None if action_dim == 0 else torch.randn(2, history_len + 2, action_dim)
    predictions = model.rollout(states, parameters, steps=3, actions=actions)
    assert predictions.shape == (2, 3, 5)
    predictor = model.predictor
    assert isinstance(predictor, RecordingSpartan)
    expected_history = states
    for step, (history, used_parameters, used_actions) in enumerate(predictor.calls):
        assert used_parameters is parameters
        torch.testing.assert_close(history, expected_history)
        if actions is not None:
            torch.testing.assert_close(used_actions, actions[:, step : step + history_len])
        else:
            assert used_actions is None
        expected_history = torch.cat(
            (expected_history[:, 1:], predictions[:, step : step + 1]), dim=1
        )
    predictions.square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
    assert parameters.grad is not None
    assert parameters.grad.abs().sum() > 0


def test_prediction_gradients_reach_all_online_modules_and_never_target() -> None:
    model = make_model(history_len=2, action_dim=2)
    out = model(torch.randn(3, 6, 7), torch.randn(3, 5, 2))
    assert not out.targets.requires_grad
    (out.predictions - out.targets).square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
    for module in (model.online_encoder, model.parameter_encoder, model.predictor):
        gradients = [
            parameter.grad for parameter in module.parameters() if parameter.grad is not None
        ]
        assert gradients
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())


def test_ema_initialization_update_buffers_and_evaluation_mode() -> None:
    encoder = VectorEncoder(7, 5, hidden_dims=(12,))
    encoder.register_buffer("floating_counter", torch.tensor(1.0))
    encoder.register_buffer("integer_counter", torch.tensor(2))
    model = SCJEPA(encoder, ParameterEncoder(5, 3, dim=8), RecordingSpartan(1, 0), 3, 0.8)
    for online, target in zip(encoder.parameters(), model.target_encoder.parameters(), strict=True):
        assert online is not target
        torch.testing.assert_close(online, target, rtol=0, atol=0)
    model.train()
    assert model.training
    assert model.online_encoder.training
    assert not model.target_encoder.training
    before = [parameter.clone() for parameter in model.target_encoder.parameters()]
    with torch.no_grad():
        for parameter in model.online_encoder.parameters():
            parameter.add_(2)
        for buffer in model.online_encoder.buffers():
            buffer.add_(3)
    # Forward calls and backward calls never advance EMA.
    model(torch.randn(2, 5, 7)).predictions.sum().backward()  # pyright: ignore[reportUnknownMemberType]
    for previous, target in zip(before, model.target_encoder.parameters(), strict=True):
        torch.testing.assert_close(previous, target, rtol=0, atol=0)
    model.update_target()
    for previous, target in zip(before, model.target_encoder.parameters(), strict=True):
        torch.testing.assert_close(target, previous + 0.4)
    for online_buffer, target_buffer in zip(
        model.online_encoder.buffers(), model.target_encoder.buffers(), strict=True
    ):
        torch.testing.assert_close(online_buffer, target_buffer, rtol=0, atol=0)


def test_ema_checkpoint_restoration(tmp_path: Path) -> None:
    model = make_model()
    with torch.no_grad():
        assert isinstance(model.online_encoder, VectorEncoder)
        model.online_encoder.head.weight.add_(1)
    model.update_target()
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = make_model(context_len=1)
    restored.ema_decay = 0.2
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert restored.ema_decay == 0.8
    assert restored.context_len == 3
    assert not restored.target_encoder.training
    observations = torch.randn(2, 5, 7)
    torch.testing.assert_close(model(observations).predictions, restored(observations).predictions)
    torch.testing.assert_close(model(observations).targets, restored(observations).targets)
    for original, loaded in zip(
        model.target_encoder.parameters(), restored.target_encoder.parameters(), strict=True
    ):
        torch.testing.assert_close(original, loaded, rtol=0, atol=0)
        assert not loaded.requires_grad


def test_model_and_rollout_shape_guards() -> None:
    model = make_model(history_len=3, action_dim=2)
    with pytest.raises(ValueError, match="prediction_start"):
        model(torch.randn(2, 3, 7), torch.randn(2, 2, 2))
    with pytest.raises(ValueError, match="prediction_start"):
        model(torch.randn(2, 5, 7), torch.randn(2, 4, 2), prediction_start=1)
    with pytest.raises(ValueError, match="actions"):
        model(torch.randn(2, 5, 7))
    with pytest.raises(ValueError, match="states_history"):
        model.rollout(torch.randn(2, 1, 5), torch.randn(2, 3), 2, torch.randn(2, 4, 2))
    with pytest.raises(ValueError, match="rollout actions"):
        model.rollout(torch.randn(2, 3, 5), torch.randn(2, 3), 2, torch.randn(2, 3, 2))
