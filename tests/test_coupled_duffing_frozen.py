import pytest
import torch

from diagnostics.coupled_duffing.frozen import FrozenStateReplay
from diagnostics.coupled_duffing.model import build_model
from diagnostics.coupled_duffing.run import forward_loss, objective_for


@pytest.mark.parametrize(("code_dim", "synchronized"), [(8, False), (4, True)])
def test_frozen_replay_preserves_outputs_losses_and_parameter_gradients(
    code_dim: int, synchronized: bool
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(87)  # pyright: ignore[reportUnknownMemberType]
    model = build_model(0, 4, code_dim=code_dim, parameter_encoder="temporal-cnn").double()
    model.online_encoder.requires_grad_(False)
    if not synchronized:
        with torch.no_grad():
            next(model.target_encoder.parameters()).add_(0.1)
    inputs = {
        "observations": torch.randn(4, 3, 6, 1, 16, 32, dtype=torch.float64),
        "actions": torch.randn(4, 3, 5, 1, dtype=torch.float64),
    }
    replay = FrozenStateReplay(model, inputs, 2)
    objective = objective_for(
        "joint", 0.4, torch.device("cpu"), state_regularization="projected-aligned"
    ).double()
    objective.requires_grad_(False)
    model.train()
    model.online_encoder.eval()
    selection = torch.tensor([3, 1])
    torch.manual_seed(101)  # pyright: ignore[reportUnknownMemberType]
    direct_loss, direct_terms, direct = forward_loss(
        model, {k: v[selection] for k, v in inputs.items()}, objective
    )
    torch.manual_seed(101)  # pyright: ignore[reportUnknownMemberType]
    cached_loss, cached_terms, cached = replay.forward_loss(
        {k: v[selection] for k, v in replay.inputs.items()}, objective
    )
    for key in ("online_states", "targets", "parameters", "predictions"):
        torch.testing.assert_close(
            getattr(cached, key), getattr(direct, key), rtol=1e-10, atol=1e-12
        )
    torch.testing.assert_close(cached_loss, direct_loss, rtol=1e-10, atol=1e-12)
    for key in direct_terms:
        torch.testing.assert_close(cached_terms[key], direct_terms[key], rtol=1e-10, atol=1e-12)
    if not synchronized:
        assert not torch.allclose(cached.targets, cached.online_states[:, :, 4:])
    parameters = [p for p in model.parameters() if p.requires_grad]
    direct_gradients = torch.autograd.grad(direct_loss, parameters)
    cached_gradients = torch.autograd.grad(cached_loss, parameters)
    assert sum(float(g.square().sum()) for g in direct_gradients) > 0
    for actual, expected in zip(cached_gradients, direct_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-12)
    # Replaying must recompute codes after updates, rather than retaining stale codes.
    before = cached.parameters.detach().clone()
    with torch.no_grad():
        for parameter, gradient in zip(parameters, cached_gradients, strict=True):
            parameter.add_(gradient, alpha=-0.01)
    torch.manual_seed(101)  # pyright: ignore[reportUnknownMemberType]
    _, _, after = replay.forward_loss(
        {k: v[selection] for k, v in replay.inputs.items()}, objective
    )
    assert not torch.equal(before, after.parameters)


def test_frozen_replay_rejects_trainable_state_encoders_and_labels() -> None:
    model = build_model(0, 3)
    inputs = {"observations": torch.zeros(2, 3, 5, 1, 16, 32), "actions": torch.zeros(2, 3, 4, 1)}
    with pytest.raises(ValueError, match="fixed state encoders"):
        FrozenStateReplay(model, inputs, 2)
    model.online_encoder.requires_grad_(False)
    with pytest.raises(ValueError, match="only observations and actions"):
        FrozenStateReplay(model, {**inputs, "theta": torch.zeros(2, 4)}, 2)
    model.parameter_source = "oracle"
    with pytest.raises(ValueError, match="learned parameters"):
        FrozenStateReplay(model, inputs, 2)


def test_replay_does_not_consume_rng_when_encoding_fixed_images() -> None:
    model = build_model(0, 3)
    model.online_encoder.requires_grad_(False)
    inputs = {"observations": torch.zeros(2, 3, 5, 1, 16, 32), "actions": torch.zeros(2, 3, 4, 1)}
    before = torch.get_rng_state().clone()
    replay = FrozenStateReplay(model, inputs, 2)
    assert torch.equal(before, torch.get_rng_state())
    assert not any(value.requires_grad for value in replay.inputs.values())
    assert replay.replay.parameter_encoder is model.parameter_encoder
    assert replay.replay.predictor is model.predictor
    assert replay.replay.context_len == model.context_len
