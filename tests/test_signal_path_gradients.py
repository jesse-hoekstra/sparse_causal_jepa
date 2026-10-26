import copy
import json
from collections.abc import Generator, Iterator
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.signal_path.gradients import (
    adamw_next_direction,
    gradient_path_diagnostic,
    independent_factor_directions,
    projected_losses,
)
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.models.model import SCJEPA
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.loop import ObjectiveConfig


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_unique_tangents_are_orthogonal_to_other_factors_and_ignore_zero_directions() -> None:
    generator = torch.Generator().manual_seed(81)
    jacobian = torch.randn(2, 3, 12, 6, generator=generator, dtype=torch.float64)
    unit, valid, fraction = independent_factor_directions(jacobian)
    assert bool(valid.all())
    assert bool((fraction <= 1 + 1e-12).all())
    torch.testing.assert_close(unit.square().sum(-2), torch.ones(2, 3, 6, dtype=torch.float64))
    for factor in range(6):
        other = jacobian[..., [i for i in range(6) if i != factor]]
        inner = torch.einsum("...o,...ok->...k", unit[..., factor], other)
        torch.testing.assert_close(inner, torch.zeros_like(inner), atol=1e-12, rtol=0)
    jacobian[..., 1] = jacobian[..., 0]
    duplicate_unit, duplicate_valid, duplicate_fraction = independent_factor_directions(jacobian)
    assert not bool(duplicate_valid[..., :2].any())
    assert bool(duplicate_valid[..., 2:].all())
    assert torch.count_nonzero(duplicate_unit[..., :2]) == 0
    assert bool((duplicate_fraction[..., :2] < 1e-12).all())
    zero = torch.zeros_like(jacobian)
    zero_unit, zero_valid, zero_fraction = independent_factor_directions(zero)
    assert torch.count_nonzero(zero_unit) == 0
    assert not bool(zero_valid.any())
    assert torch.count_nonzero(zero_fraction) == 0


def test_projected_loss_has_exact_stopped_projector_gradient_and_normalization() -> None:
    generator = torch.Generator().manual_seed(19)
    residual = torch.randn(2, 10, 2, 6, generator=generator, dtype=torch.float64).requires_grad_()
    jacobian = torch.randn(2, 10, 12, 6, generator=generator, dtype=torch.float64)
    directions, _, _ = independent_factor_directions(jacobian)
    directions.requires_grad_()
    losses, per_group = projected_losses(residual, directions, 4.0)
    torch.testing.assert_close(losses, per_group.mean(0))
    factor = 2
    gradient, projector_gradient = torch.autograd.grad(
        losses[factor], (residual, directions), allow_unused=True
    )
    assert projector_gradient is None
    coefficient = (residual.detach().flatten(-2) * directions.detach()[..., factor]).sum(-1)
    expected = (
        8 * coefficient[..., None] * directions.detach()[..., factor] / residual.numel()
    ).reshape_as(residual)
    torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)
    change = torch.randn(residual.shape, generator=generator, dtype=torch.float64)
    step = 1e-6
    plus, _ = projected_losses(residual.detach() + step * change, directions, 4)
    minus, _ = projected_losses(residual.detach() - step * change, directions, 4)
    torch.testing.assert_close(
        (plus[factor] - minus[factor]) / (2 * step),
        (gradient * change).sum(),
        rtol=1e-8,
        atol=1e-9,
    )


@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("maximize", [False, True])
def test_pure_adam_direction_matches_actual_copy_step_and_skips_none_gradients(
    amsgrad: bool, maximize: bool
) -> None:
    generator = torch.Generator().manual_seed(92)
    parameters = [
        nn.Parameter(torch.randn(3, 4, generator=generator, dtype=torch.float64)) for _ in range(3)
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=0.003,
        betas=(0.8, 0.95),
        eps=1e-7,
        weight_decay=0.07,
        amsgrad=amsgrad,
        maximize=maximize,
    )
    for _ in range(3):
        for p in parameters:
            p.grad = torch.randn(p.shape, generator=generator, dtype=torch.float64)
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    gradients: tuple[Tensor | None, ...] = (
        torch.randn(3, 4, generator=generator, dtype=torch.float64) * 5,
        None,
        torch.randn(3, 4, generator=generator, dtype=torch.float64) * 3,
    )
    snapshot = copy.deepcopy(optimizer.state_dict())
    original = [p.detach().clone() for p in parameters]
    saved_rng = torch.get_rng_state()
    update, metrics = adamw_next_direction(parameters, gradients, snapshot)
    torch.testing.assert_close(snapshot, optimizer.state_dict(), rtol=0, atol=0)
    assert torch.equal(saved_rng, torch.get_rng_state())
    for p, old in zip(parameters, original, strict=True):
        torch.testing.assert_close(p, old, rtol=0, atol=0)
    for p, gradient in zip(parameters, gradients, strict=True):
        p.grad = gradient.clone() if gradient is not None else None
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
    assert abs(norm - metrics["pre_clip_gradient_norm"]) < 1e-12
    assert metrics["clip_coefficient"] < 1
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    for p, old, delta in zip(parameters, original, update, strict=True):
        torch.testing.assert_close(p, old + delta, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(update[1], torch.zeros_like(update[1]), rtol=0, atol=0)


def _tiny_model_data() -> tuple[SCJEPA, EvaluationData, ObjectiveConfig]:
    torch.default_generator.manual_seed(314)
    model = (
        build_model(
            ModelConfig(
                observation_dim=6,
                encoder_hidden_dims=(8,),
                state_dim=6,
                param_dim=6,
                action_dim=16,
                context_len=10,
                history_len=1,
                parameter_encoder_kind="transition",
                spartan_layers=1,
                spartan_embed_dim=4,
                spartan_mlp_hidden=8,
                spartan_mlp_layers=2,
                spartan_dense=True,
                autoregressive=True,
            )
        )
        .double()
        .eval()
    )
    model.online_encoder.requires_grad_(False)
    model.target_encoder.requires_grad_(False)
    theta = torch.rand(4, 6, dtype=torch.float64) - 0.5
    initial = torch.randn(4, 10, 6, dtype=torch.float64)
    controls = torch.cat(
        (
            torch.randn(4, 10, 11, 10, dtype=torch.float64),
            2 * torch.rand(4, 10, 11, 6, dtype=torch.float64) - 1,
        ),
        dim=-1,
    )
    actions = dgp.encode_controls(controls)
    states = dgp.simulate(initial, theta[:, None], actions)
    observation_map = VectorObservationMap(matrix=torch.eye(6, dtype=torch.float64), dimension=6)
    with torch.no_grad():
        observations = observation_map.forward(states).flatten(0, 1)
        online = model.online_encoder.encode_sequence(observations).reshape(4, 10, 12, 6)
        target = model.target_encoder.encode_sequence(observations).reshape(4, 10, 12, 6)
    data: EvaluationData = {
        "metadata": {"observation_matrix": observation_map.matrix},
        "test": {
            "theta": theta,
            "true_states": states,
            "true_actions": actions,
            "states_online": online,
            "states_target": target,
            "actions": actions,
        },
    }
    objective = ObjectiveConfig(
        state_regularizer="sigreg",
        parameter_regularizer="sigreg",
        state_regularizer_weight=0.4,
        parameter_regularizer_weight=0.4,
        parameter_consistency_weight=1,
        logit_weight=0.0001,
        regularizer_num_projections=8,
    )
    return model, data, objective


def test_diagnostic_preserves_weights_modes_gradients_rng_and_optimizer_moments() -> None:
    model, data, objective = _tiny_model_data()
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0)
    optimizer_state: dict[str, Any] = copy.deepcopy(optimizer.state_dict())
    before_state = copy.deepcopy(model.state_dict())
    modules = list(cast(Iterator[tuple[str, nn.Module]], model.named_modules()))
    before_modes = {name: module.training for name, module in modules}
    for p in parameters:
        p.grad = torch.full_like(p, 0.123)
    rng = torch.get_rng_state()
    result, tensors = gradient_path_diagnostic(
        model, data, objective, 4, groups=4, batch_size=2, optimizer_state=optimizer_state
    )
    json.dumps(result, allow_nan=False)
    assert result["groups"] == 4
    assert len(result["batches"]) == 2
    assert len(result["by_parameter"]) == 6
    assert result["optimizer_direction_available"]
    assert tensors["factor_code_gradients"].shape == (4, 10, 6, 6)
    assert tensors["target_jacobian"].shape == (4, 10, 2, 6, 6)
    assert bool(tensors["unique_direction_valid"].all())
    assert float(tensors["factor_code_gradients"].square().sum()) > 0
    assert torch.equal(rng, torch.get_rng_state())
    assert before_modes == {name: module.training for name, module in modules}
    torch.testing.assert_close(model.state_dict(), before_state, rtol=0, atol=0)
    torch.testing.assert_close(optimizer_state, optimizer.state_dict(), rtol=0, atol=0)
    for p in parameters:
        assert p.grad is not None
        torch.testing.assert_close(p.grad, torch.full_like(p, 0.123), rtol=0, atol=0)
    repeat, repeat_tensors = gradient_path_diagnostic(
        model, data, objective, 4, groups=4, batch_size=2, optimizer_state=optimizer_state
    )
    assert repeat == result
    torch.testing.assert_close(repeat_tensors, tensors, rtol=0, atol=0)


def test_invalid_factor_directions_and_optimizer_metadata_are_rejected() -> None:
    with pytest.raises(ValueError, match="rcond"):
        independent_factor_directions(torch.eye(6), 0)
    with pytest.raises(ValueError, match="Jacobian"):
        independent_factor_directions(torch.zeros(3, 4))
    with pytest.raises(ValueError, match="parameter IDs"):
        adamw_next_direction(
            (torch.ones(2),),
            (torch.ones(2),),
            {
                "state": {},
                "param_groups": [{"params": []}],
            },
        )
