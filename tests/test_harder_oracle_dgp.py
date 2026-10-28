"""Equation, population-contract and fixed-output influence checks for harder data."""

import math
from typing import cast

import pytest
import torch
from torch import Tensor

from diagnostics.harder_oracle import dgp


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


def _reflect(vector: list[float], state: list[float]) -> list[float]:
    scale = 2 * sum(v * s for v, s in zip(vector, state, strict=True))
    scale /= sum(v * v for v in vector)
    return [s - scale * v for v, s in zip(vector, state, strict=True)]


def _reference(
    states: Tensor,
    parameters: Tensor,
    actions: Tensor,
    rho: float,
    transport_chart: dgp.TransportChart = "mixed",
) -> Tensor:
    action = [float(value) for value in actions]
    controls = [value - sum(action) / 11 for value in action]
    state = [float(value) for value in states]
    if transport_chart == "mixed":
        state = _reflect([1.0, 1.0, 1.0], state[:3]) + _reflect([1.0, 1.0, 1.0], state[3:])
    theta = [float(value) for value in parameters]
    rotated: list[float] = []
    for block in range(2):
        vector = [1.0, *controls[2 * block : 2 * block + 2]]
        scaled = [(index + 1) * value for index, value in enumerate(vector)]
        rotated.extend(_reflect(vector, _reflect(scaled, state[3 * block : 3 * block + 3])))
    if transport_chart == "mixed":
        rotated = _reflect([1.0, 1.0, 1.0], rotated[:3]) + _reflect([1.0, 1.0, 1.0], rotated[3:])
    scale = math.sqrt(1 - rho * rho)
    result: list[float] = []
    for i in range(6):
        phase = theta[i] + controls[16 + i] * theta[(i + 3) % 6]
        innovation = math.cos(phase) * controls[4 + i] + math.sin(phase) * controls[10 + i]
        result.append(rho * rotated[i] + scale * innovation)
    return torch.tensor(result, dtype=torch.float64)


@pytest.mark.parametrize("rho", [0.5, 0.7, 0.85])
@pytest.mark.parametrize("transport_chart", ["physical", "mixed"])
def test_transition_matches_independent_scalar_equations(
    rho: float, transport_chart: dgp.TransportChart
) -> None:
    state, parameter, action = _inputs()
    torch.testing.assert_close(
        dgp.transition(state, parameter, action, rho=rho, transport_chart=transport_chart),
        _reference(state, parameter, action, rho, transport_chart),
        atol=3e-15,
        rtol=3e-15,
    )
    rotation = dgp.state_rotation(action)
    torch.testing.assert_close(rotation @ rotation.T, torch.eye(6, dtype=torch.float64))
    mixing = dgp.action_rotation()
    torch.testing.assert_close(mixing @ mixing, torch.eye(22, dtype=torch.float64))
    torch.testing.assert_close(dgp.encode_controls(dgp.decode_actions(action)), action)


@pytest.mark.parametrize("transport_chart", ["physical", "mixed"])
def test_exact_jacobians_support_and_state_jacobian_parameter_invariance(
    transport_chart: dgp.TransportChart,
) -> None:
    state, parameter, action = _inputs()
    actual = dgp.analytic_jacobians(state, parameter, action, transport_chart=transport_chart)

    def mapped_transition(state: Tensor, parameter: Tensor, action: Tensor) -> Tensor:
        return dgp.transition(state, parameter, action, transport_chart=transport_chart)

    automatic = cast(
        tuple[Tensor, Tensor, Tensor],
        torch.autograd.functional.jacobian(  # pyright: ignore[reportUnknownMemberType]
            mapped_transition,
            (state, parameter, action),
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
        (actual.action, masks.action, 132),
        (actual.parameter, masks.parameter, 12),
    ):
        assert int(mask.sum()) == count
        assert torch.equal(derivative != 0, mask)
    other = dgp.analytic_jacobians(state, -parameter, action, transport_chart=transport_chart)
    torch.testing.assert_close(actual.state, other.state, atol=0, rtol=0)
    assert not torch.allclose(actual.parameter, other.parameter)
    batched = dgp.analytic_jacobians(
        state.expand(2, 6), parameter, action.expand(2, 22), transport_chart=transport_chart
    )
    torch.testing.assert_close(batched.action[1], actual.action)


def test_simulation_broadcasts_and_attaches_all_parameters() -> None:
    state, parameter, action = _inputs()
    parameter.requires_grad_()
    initial = torch.stack((state, -state))
    actions = torch.stack((action, -action, 0.5 * action)).expand(2, 3, 22)
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
    with pytest.raises(ValueError, match="dimension 22"):
        dgp.transition(state, parameter, action[:16])


@pytest.mark.parametrize("rho", [0.0, 1.0, -0.7, float("nan"), float("inf")])
def test_degenerate_or_nonfinite_rho_is_rejected(rho: float) -> None:
    with pytest.raises(ValueError, match="rho"):
        dgp.transition(*_inputs(), rho=rho)


@pytest.mark.parametrize("parameter_scale", [0.0, 0.7])
def test_gaussian_marginal_stationarity_in_multiple_directions(parameter_scale: float) -> None:
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
    controls = torch.zeros(count, 22, dtype=torch.float64)
    controls[:, :4] = torch.randn(count, 4, generator=generator, dtype=torch.float64)
    rotation = dgp.state_rotation(dgp.encode_controls(controls))
    rotated = (rotation @ output[:, None]).squeeze(-1)
    controls[:, 4:10] = (output - 0.7 * rotated) / math.sqrt(0.51)
    actions = dgp.encode_controls(controls)
    torch.testing.assert_close(dgp.transition(output, parameter, actions), output.expand(count, 6))
    matrices = dgp.analytic_jacobians(output, parameter, actions).state[:, :, 0]
    assert _rank(matrices[:, dgp.graphs().state[:, 0]]) == 18


def test_parameter_influence_spans_twelve_dimensions_at_fixed_output_and_theta() -> None:
    output = torch.linspace(-0.8, 0.7, 6, dtype=torch.float64)
    parameter = torch.linspace(-0.6, 0.6, 6, dtype=torch.float64)
    state = torch.linspace(0.5, -0.3, 6, dtype=torch.float64)
    companion = torch.eye(6, dtype=torch.float64).repeat_interleave(2, 0)
    controls = torch.zeros(12, 22, dtype=torch.float64)
    controls[torch.arange(1, 12, 2), 16 + torch.arange(6)] = 0.5
    phase = parameter + controls[:, 16:] * parameter.roll(3)
    innovation = (output - 0.7 * state) / math.sqrt(0.51)
    controls[:, 4:10] = innovation * phase.cos() - companion * phase.sin()
    controls[:, 10:16] = innovation * phase.sin() + companion * phase.cos()
    actions = dgp.encode_controls(controls)
    torch.testing.assert_close(dgp.transition(state, parameter, actions), output.expand(12, 6))
    matrices = dgp.analytic_jacobians(state, parameter, actions).parameter
    assert _rank(matrices[:, dgp.graphs().parameter]) == 12


def test_every_observed_action_column_has_full_fixed_output_influence() -> None:
    output = torch.linspace(-0.7, 0.8, 6, dtype=torch.float64)
    basis = 0.1 * torch.eye(6, dtype=torch.float64)
    parameter = torch.cat((basis, -basis), 0)
    state = output / 0.7
    actions = torch.zeros(12, 22, dtype=torch.float64)
    torch.testing.assert_close(dgp.transition(state, parameter, actions), output.expand(12, 6))
    columns = dgp.analytic_jacobians(state, parameter, actions).action[:, :, 0]
    differences = columns[:6] - columns[6:]
    for j in range(22):
        assert _rank(differences[:, :, j]) == 6


def test_informative_anchor_identifies_all_six_physical_parameters() -> None:
    _, parameter, _ = _inputs()
    state = torch.zeros(6, dtype=torch.float64)
    controls = torch.zeros(22, dtype=torch.float64)
    controls[10:16] = 1
    output = dgp.transition(state, parameter, dgp.encode_controls(controls))
    recovered = torch.asin(output / math.sqrt(0.51))
    torch.testing.assert_close(recovered, parameter, atol=1e-15, rtol=1e-15)


def test_transport_twist_keeps_parameter_innovations_in_physical_axes() -> None:
    state, parameter, action = _inputs()
    mixing = dgp.transport_rotation()
    torch.testing.assert_close(mixing @ mixing.T, torch.eye(6, dtype=torch.float64))
    physical_rotation = dgp.state_rotation(action, transport_chart="physical")
    mixed_rotation = dgp.state_rotation(action, transport_chart="mixed")
    torch.testing.assert_close(mixed_rotation, mixing.T @ physical_rotation @ mixing)
    physical_output = dgp.transition(state, parameter, action, transport_chart="physical")
    mixed_output = dgp.transition(state, parameter, action, transport_chart="mixed")
    torch.testing.assert_close(
        mixed_output - physical_output, 0.7 * (mixed_rotation - physical_rotation) @ state
    )
    assert not torch.allclose(physical_output, mixed_output)
    # This changes transport without rotating the parameter forcing itself.
    entirely_conjugated = mixing @ dgp.transition(mixing.T @ state, parameter, action)
    assert not torch.allclose(entirely_conjugated, physical_output)


def test_mixed_state_chart_preserves_state_action_graphs_but_densifies_parameters() -> None:
    state, parameter, action = _inputs()
    mixing = dgp.transport_rotation()
    physical_state = mixing.T @ state
    original = dgp.analytic_jacobians(physical_state, parameter, action)
    chart_state = mixing @ original.state[:, 0] @ mixing.T
    chart_action = mixing @ original.action[:, 0]
    chart_parameter = mixing @ original.parameter
    assert int((chart_state.abs() > 1e-12).sum()) == 18
    assert int((chart_action.abs() > 1e-12).sum()) == 132
    assert int((chart_parameter.abs() > 1e-12).sum()) == 36
    # Each population correlation row has maximum magnitude 2/3. A within-row
    # cyclic permutation attains all six maxima for Gaussian physical states.
    permutation = torch.tensor([1, 2, 0, 4, 5, 3])
    torch.testing.assert_close(
        mixing[torch.arange(6), permutation].abs().mean(),
        torch.tensor(2 / 3, dtype=torch.float64),
    )
