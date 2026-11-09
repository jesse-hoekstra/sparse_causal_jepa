import pytest
import torch
from torch import Tensor

from diagnostics.coupled_duffing.model import build_model
from diagnostics.coupled_duffing.regularization import (
    StateRegularizationObjective,
    mix_state_samples,
    state_variance_floor,
)
from diagnostics.coupled_duffing.run import forward_loss, objective_for
from scjepa.training.rng import preserve_rng


def test_time_mixture_preserves_each_system_and_the_prediction_rng() -> None:
    states = torch.arange(4 * 30 * 2).reshape(4, 30, 2).float().requires_grad_()
    torch.manual_seed(19)  # pyright: ignore[reportUnknownMemberType]
    before = torch.get_rng_state().clone()
    mixed = mix_state_samples(states)
    assert torch.equal(torch.get_rng_state(), before)
    assert not torch.equal(mixed, states)
    for group in range(4):
        torch.testing.assert_close(mixed[group].sort(dim=0).values, states[group])
    # Independent permutations, not the same time shuffle for every system.
    assert not torch.equal(mixed[0, :, 0], mixed[1, :, 0] - 60)
    mixed.square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
    torch.testing.assert_close(states.grad, 2 * states.detach())


def test_variance_floor_anchors_scale_without_penalizing_dependent_coordinates() -> None:
    first = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    dependent = torch.stack((first, first), dim=-1)[:, None, :]
    assert state_variance_floor(dependent) == 0
    small = (dependent * 0.01).requires_grad_()
    penalty = state_variance_floor(small)
    assert penalty > 0.9
    penalty.backward()  # pyright: ignore[reportUnknownMemberType]
    assert small.grad is not None
    assert bool((small.grad * small).sum() < 0)


@pytest.mark.parametrize("mode", ["mixture", "mixture-variance", "projected-mixture-variance"])
def test_auxiliary_head_cannot_bypass_raw_state_prediction_or_context(mode: str) -> None:
    torch.set_num_threads(1)
    model = build_model(0, 6, code_dim=8, parameter_encoder="temporal-cnn")
    inputs = {"observations": torch.randn(4, 3, 8, 1, 16, 32), "actions": torch.randn(4, 3, 7, 1)}
    objective = objective_for("joint", 0.4, torch.device("cpu"), state_regularization=mode)
    assert isinstance(objective, StateRegularizationObjective)
    with preserve_rng(torch.device("cpu")):
        loss, terms, output = forward_loss(model, inputs, objective)
    expected: Tensor = (output.predictions - output.targets).square().mean()
    torch.testing.assert_close(terms["prediction"], expected, rtol=0, atol=0)
    assert output.online_states.shape == (4, 3, 8, 2)
    assert output.parameters.shape == (4, 3, 8)
    assert not output.targets.requires_grad
    second_to_first = torch.autograd.grad(
        output.predictions[:, :, 1].square().mean(),
        output.dynamics[0].prediction,
        retain_graph=True,
    )[0]
    assert bool(second_to_first.abs().sum() > 0)
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    for module in (model.online_encoder, model.parameter_encoder, model.predictor):
        assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())
    if mode == "projected-mixture-variance":
        assert objective.projector(output.online_states).shape == output.online_states.shape
        assert all(p.grad is not None for p in objective.projector.parameters())


def test_unknown_regularization_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="state regularization"):
        objective_for("joint", 0.4, torch.device("cpu"), state_regularization="typo")
