"""Independent equations, isotropy and fixed-output influence of Cayley dynamics."""

import math
from typing import cast

import pytest
import torch
from torch import Tensor

from diagnostics.isotropic_oracle import dgp


def _inputs() -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(193)
    return (
        torch.randn(6, dtype=torch.float64, generator=generator),
        1.5 * torch.rand(6, dtype=torch.float64, generator=generator) - 0.75,
        dgp.encode_controls(dgp.sample_controls((), generator)),
    )


def _rank(matrix: Tensor) -> int:
    return int(
        cast(
            Tensor,
            torch.linalg.matrix_rank(matrix, atol=1e-10, rtol=1e-10),  # pyright: ignore[reportUnknownMemberType]
        )
    )


def _quaternion_product(left: list[float], right: list[float]) -> list[float]:
    a, b, c, d = left
    e, f, g, h = right
    return [
        a * e - b * f - c * g - d * h,
        a * f + b * e + c * h - d * g,
        a * g - b * h + c * e + d * f,
        a * h + b * g - c * f + d * e,
    ]


def _reference(states: Tensor, parameters: Tensor, actions: Tensor, rho: float) -> Tensor:
    action = [float(value) for value in actions]
    controls = [value - sum(action) / 12 for value in action]
    state = [float(value) for value in states]
    theta = [float(value) for value in parameters]
    rotated: list[float] = []
    for block in range(2):
        vector = [dgp.ROTATION_SCALE * v for v in controls[3 * block : 3 * block + 3]]
        norm = math.sqrt(1 + sum(v * v for v in vector))
        quaternion = [value / norm for value in [1.0, *vector]]
        conjugate = [quaternion[0], *[-v for v in quaternion[1:]]]
        pure = [0.0, *state[3 * block : 3 * block + 3]]
        rotated.extend(_quaternion_product(_quaternion_product(quaternion, pure), conjugate)[1:])
    scale = math.sqrt(1 - rho * rho)
    result: list[float] = []
    for i in range(6):
        phase = theta[i] + controls[18 + i] * theta[(i + 3) % 6]
        innovation = math.cos(phase) * controls[6 + i] + math.sin(phase) * controls[12 + i]
        result.append(rho * rotated[i] + scale * innovation)
    return torch.tensor(result, dtype=torch.float64)


def test_transition_matches_independent_quaternion_reference() -> None:
    state, parameter, action = _inputs()
    torch.testing.assert_close(
        dgp.transition(state, parameter, action),
        _reference(state, parameter, action, 0.7),
        atol=3e-15,
        rtol=3e-15,
    )
    rotation = dgp.state_rotation(action)
    torch.testing.assert_close(rotation @ rotation.T, torch.eye(6, dtype=torch.float64))
    torch.testing.assert_close(torch.linalg.det(rotation), torch.tensor(1.0, dtype=torch.float64))  # pyright: ignore[reportUnknownMemberType]
    mixing = dgp.action_rotation()
    torch.testing.assert_close(mixing @ mixing, torch.eye(24, dtype=torch.float64))
    torch.testing.assert_close(dgp.encode_controls(dgp.decode_actions(action)), action)


def test_exact_jacobians_support_and_parameter_independent_state_transport() -> None:
    state, parameter, action = _inputs()
    actual = dgp.analytic_jacobians(state, parameter, action)
    automatic = cast(
        tuple[Tensor, Tensor, Tensor],
        torch.autograd.functional.jacobian(  # pyright: ignore[reportUnknownMemberType]
            dgp.transition, (state, parameter, action)
        ),
    )
    for calculated, expected in (
        (actual.state[:, 0], automatic[0]),
        (actual.parameter, automatic[1]),
        (actual.action[:, 0], automatic[2]),
    ):
        torch.testing.assert_close(calculated, expected, atol=1e-14, rtol=1e-13)
    masks = dgp.graphs()
    for derivative, mask, count in (
        (actual.state, masks.state, 18),
        (actual.action, masks.action, 144),
        (actual.parameter, masks.parameter, 12),
    ):
        assert int(mask.sum()) == count
        assert torch.equal(derivative != 0, mask)
    other = dgp.analytic_jacobians(state, -parameter, action)
    torch.testing.assert_close(actual.state, other.state, atol=0, rtol=0)
    assert not torch.allclose(actual.parameter, other.parameter)
    batched = dgp.analytic_jacobians(state.expand(2, 6), parameter, action.expand(2, 24))
    torch.testing.assert_close(batched.action[1], actual.action)


def test_rotation_is_equivariant_under_all_orthogonal_axis_changes() -> None:
    generator = torch.Generator().manual_seed(811)
    gamma = torch.randn(64, 3, generator=generator, dtype=torch.float64)
    rotation = dgp.rotation_matrix(gamma)
    # A dense reflection tests the det(Q) correction, not just coordinate permutations.
    reflection = torch.eye(3, dtype=torch.float64) - (2 / 3) * torch.ones(3, 3, dtype=torch.float64)
    proper = dgp.rotation_matrix(torch.tensor([1.1, -0.8, 0.5], dtype=torch.float64))
    for axes, determinant in ((reflection, -1), (proper, 1)):
        transformed = dgp.rotation_matrix(determinant * gamma @ axes.T)
        torch.testing.assert_close(transformed, axes @ rotation @ axes.T, atol=2e-15, rtol=2e-15)
    # An isotropic Gaussian gamma has the same distribution after either map.
    # Equivariance therefore removes a preferred population transport axis.


def test_rotation_mean_has_no_axis_preference_and_matches_predeclared_strength() -> None:
    generator = torch.Generator().manual_seed(1009)
    gamma = torch.randn(50_000, 3, generator=generator, dtype=torch.float64)
    rotation = dgp.rotation_matrix(gamma)
    expected = 0.8195489816055351 * torch.eye(3, dtype=torch.float64)
    assert float((rotation.mean(0) - expected).abs().max()) < 0.004
    torch.testing.assert_close(
        rotation @ rotation.transpose(-1, -2), torch.eye(3).double().expand(50_000, 3, 3)
    )
    zero = dgp.rotation_matrix(torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(zero, torch.eye(3, dtype=torch.float64), atol=0, rtol=0)


def test_simulation_broadcasts_and_attaches_all_parameters() -> None:
    state, parameter, action = _inputs()
    parameter.requires_grad_()
    initial = torch.stack((state, -state))
    actions = torch.stack((action, -action, 0.5 * action)).expand(2, 3, 24)
    result = dgp.simulate(initial, parameter, actions)
    assert result.shape == (2, 4, 6)
    expected = initial
    for step in range(3):
        expected = torch.stack(
            [_reference(expected[v], parameter.detach(), actions[v, step], 0.7) for v in range(2)]
        )
        torch.testing.assert_close(result[:, step + 1], expected)
    result[:, -1].square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
    assert parameter.grad is not None
    assert bool((parameter.grad.abs() > 0).all())
    assert dgp.transition(state.float(), parameter.float(), action.float()).dtype == torch.float64
    with pytest.raises(ValueError, match="time axis"):
        dgp.simulate(state, parameter, action)
    with pytest.raises(ValueError, match="dimension 24"):
        dgp.transition(state, parameter, action[:22])


@pytest.mark.parametrize("rho", [0.0, 1.0, -0.7, float("nan"), float("inf")])
def test_degenerate_or_nonfinite_rho_is_rejected(rho: float) -> None:
    with pytest.raises(ValueError, match="rho"):
        dgp.transition(*_inputs(), rho=rho)


@pytest.mark.parametrize("scale", [0.0, -0.5, float("nan"), float("inf")])
def test_degenerate_rotation_scale_is_rejected(scale: float) -> None:
    with pytest.raises(ValueError, match="rotation_scale"):
        dgp.rotation_matrix(torch.zeros(3), rotation_scale=scale)


@pytest.mark.parametrize("parameter_scale", [0.0, 0.7])
def test_gaussian_state_marginal_stationarity(parameter_scale: float) -> None:
    generator = torch.Generator().manual_seed(652)
    count = 16_000
    initial = torch.randn(count, 6, generator=generator, dtype=torch.float64)
    controls = dgp.sample_controls((count, 3), generator)
    parameter = parameter_scale * torch.linspace(-1, 1, 6, dtype=torch.float64)
    states = dgp.simulate(initial, parameter, dgp.encode_controls(controls))
    directions = torch.randn(6, 12, generator=generator, dtype=torch.float64)
    directions /= directions.square().sum(0).sqrt()
    for frame in range(1, 4):
        values = states[:, frame]
        centered = values - values.mean(0)
        covariance = centered.T @ centered / count
        assert float(values.mean(0).abs().max()) < 0.04
        assert float((covariance - torch.eye(6)).abs().max()) < 0.045
        projected = values @ directions
        assert float((projected.cos().mean(0) - math.exp(-0.5)).abs().max()) < 0.025
        assert float(projected.sin().mean(0).abs().max()) < 0.025


def test_state_influence_spans_eighteen_dimensions_on_fixed_point_fiber() -> None:
    generator = torch.Generator().manual_seed(832)
    count = 128
    output = torch.linspace(-0.9, 0.8, 6, dtype=torch.float64)
    parameter = torch.zeros(6, dtype=torch.float64)
    controls = torch.zeros(count, 24, dtype=torch.float64)
    controls[:, :6] = torch.randn(count, 6, generator=generator, dtype=torch.float64)
    rotation = dgp.state_rotation(dgp.encode_controls(controls))
    rotated = (rotation @ output[:, None]).squeeze(-1)
    controls[:, 6:12] = (output - 0.7 * rotated) / math.sqrt(0.51)
    actions = dgp.encode_controls(controls)
    torch.testing.assert_close(dgp.transition(output, parameter, actions), output.expand(count, 6))
    matrices = dgp.analytic_jacobians(output, parameter, actions).state[:, :, 0]
    assert _rank(matrices[:, dgp.graphs().state[:, 0]]) == 18


def test_parameter_influence_spans_twelve_dimensions_at_fixed_output_and_theta() -> None:
    output = torch.linspace(-0.8, 0.7, 6, dtype=torch.float64)
    parameter = torch.linspace(-0.6, 0.6, 6, dtype=torch.float64)
    state = torch.linspace(0.5, -0.3, 6, dtype=torch.float64)
    companion = torch.eye(6, dtype=torch.float64).repeat_interleave(2, 0)
    controls = torch.zeros(12, 24, dtype=torch.float64)
    controls[torch.arange(1, 12, 2), 18 + torch.arange(6)] = 0.5
    phase = parameter + controls[:, 18:] * parameter.roll(3)
    innovation = (output - 0.7 * state) / math.sqrt(0.51)
    controls[:, 6:12] = innovation * phase.cos() - companion * phase.sin()
    controls[:, 12:18] = innovation * phase.sin() + companion * phase.cos()
    actions = dgp.encode_controls(controls)
    torch.testing.assert_close(dgp.transition(state, parameter, actions), output.expand(12, 6))
    matrices = dgp.analytic_jacobians(state, parameter, actions).parameter
    assert _rank(matrices[:, dgp.graphs().parameter]) == 12


def test_all_twenty_four_action_columns_have_full_fixed_output_influence() -> None:
    output = torch.linspace(-0.7, 0.8, 6, dtype=torch.float64)
    basis = 0.1 * torch.eye(6, dtype=torch.float64)
    parameter = torch.cat((basis, -basis), 0)
    state = output / 0.7
    actions = torch.zeros(12, 24, dtype=torch.float64)
    torch.testing.assert_close(dgp.transition(state, parameter, actions), output.expand(12, 6))
    columns = dgp.analytic_jacobians(state, parameter, actions).action[:, :, 0]
    differences = columns[:6] - columns[6:]
    for j in range(24):
        assert _rank(differences[:, :, j]) == 6


def test_informative_anchor_identifies_all_six_parameters() -> None:
    _, parameter, _ = _inputs()
    controls = torch.zeros(24, dtype=torch.float64)
    controls[12:18] = 1
    output = dgp.transition(torch.zeros(6), parameter, dgp.encode_controls(controls))
    torch.testing.assert_close(
        torch.asin(output / math.sqrt(0.51)), parameter, atol=1e-15, rtol=1e-15
    )
