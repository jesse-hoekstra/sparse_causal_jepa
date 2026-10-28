"""Independent phase equations, derivatives, omission risks and informative witnesses."""

import math
from collections.abc import Iterator
from typing import cast

import pytest
import torch
from torch import Tensor

from diagnostics.balanced_action_oracle import dgp as source
from diagnostics.rotating_pair_oracle import dgp


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _inputs() -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(254)
    return (
        torch.randn(6, dtype=torch.float64, generator=generator),
        1.5 * torch.rand(6, dtype=torch.float64, generator=generator) - 0.75,
        dgp.encode_controls(dgp.sample_controls((), generator)),
    )


def _rank(matrix: Tensor) -> int:
    return int(
        cast(
            Tensor,
            torch.linalg.matrix_rank(  # pyright: ignore[reportUnknownMemberType]
                matrix, atol=1e-10, rtol=1e-10
            ),
        )
    )


def test_only_phase_changes_and_transport_action_helpers_are_reused() -> None:
    for name in (
        "encode_controls",
        "decode_actions",
        "state_rotation",
        "rotation_matrix",
        "sample_controls",
        "graphs",
        "action_rotation",
    ):
        assert getattr(dgp, name) is getattr(source, name)
    assert math.pi / 1.5 == dgp.PHASE_GAIN
    state, theta, action = _inputs()
    controls = dgp.decode_actions(action)
    expected = 0.7 * (source.state_rotation(action) @ state)
    for index in range(6):
        phase = (
            math.pi
            / 1.5
            * (
                math.cos(math.pi * float(controls[18 + index])) * float(theta[index])
                + math.sin(math.pi * float(controls[18 + index])) * float(theta[(index + 3) % 6])
            )
        )
        expected[index] += math.sqrt(0.51) * (
            float(controls[6 + index]) * math.cos(phase)
            + float(controls[12 + index]) * math.sin(phase)
        )
    torch.testing.assert_close(
        dgp.transition(state, theta, action), expected, atol=1e-15, rtol=1e-14
    )
    assert not torch.allclose(
        dgp.transition(state, theta, action), source.transition(state, theta, action)
    )


def test_analytic_jacobians_match_autograd_and_true_graph_supports() -> None:
    state, theta, action = _inputs()
    analytic = dgp.analytic_jacobians(state, theta, action)
    automatic = cast(
        tuple[Tensor, Tensor, Tensor],
        torch.autograd.functional.jacobian(  # pyright: ignore[reportUnknownMemberType]
            dgp.transition, (state, theta, action)
        ),
    )
    for actual, expected in (
        (analytic.state[:, 0], automatic[0]),
        (analytic.parameter, automatic[1]),
        (analytic.action[:, 0], automatic[2]),
    ):
        torch.testing.assert_close(actual, expected, atol=5e-14, rtol=2e-13)
    masks = dgp.graphs()
    for derivative, mask, count in (
        (analytic.state, masks.state, 18),
        (analytic.action, masks.action, 144),
        (analytic.parameter, masks.parameter, 12),
    ):
        assert int(mask.sum()) == count
        assert torch.equal(derivative != 0, mask)
    other = dgp.analytic_jacobians(state, -theta, action)
    torch.testing.assert_close(analytic.state, other.state, atol=0, rtol=0)
    batched = dgp.analytic_jacobians(state.expand(2, 6), theta, action.expand(2, 24))
    torch.testing.assert_close(batched.action[1], analytic.action)


def test_two_step_rollout_retains_the_first_step_parameter_gradient() -> None:
    state, theta, action = _inputs()
    theta.requires_grad_()
    actions = torch.stack((action, -action))
    rollout = dgp.simulate(state, theta, actions)
    assert rollout.shape == (3, 6)
    first = dgp.transition(state, theta, actions[0])
    second = dgp.transition(first, theta, actions[1])
    torch.testing.assert_close(rollout[-1], second, rtol=0, atol=0)
    full = torch.autograd.grad(rollout[-1].square().sum(), theta)[0]
    detached_second = dgp.transition(first.detach(), theta, actions[1])
    truncated = torch.autograd.grad(detached_second.square().sum(), theta)[0]
    assert bool((full.abs() > 1e-8).all())
    assert float((full - truncated).square().sum().sqrt()) > 0.1


def test_balanced_pair_omission_risks_and_directional_sensitivity() -> None:
    # Periodic trapezoidal quadrature is independent of the transition implementation.
    angle = 2 * math.pi * torch.arange(8192, dtype=torch.float64) / 8192
    coefficients = dgp.PHASE_GAIN * torch.stack((angle.cos(), angle.sin()), -1)
    torch.testing.assert_close(coefficients.mean(0), torch.zeros(2).double(), atol=1e-15, rtol=0)
    torch.testing.assert_close(
        coefficients.T @ coefficients / len(angle),
        (dgp.PHASE_GAIN**2 / 2) * torch.eye(2).double(),
        atol=1e-14,
        rtol=1e-14,
    )
    sinc_own = torch.sinc(0.5 * angle.cos())
    sinc_partner = torch.sinc(0.5 * angle.sin())
    scale = 0.51 * (1 + 0.49 / 2)
    own = scale * (1 - sinc_own.square().mean())
    partner = scale * (1 - sinc_partner.square().mean())
    blind = scale * (1 - (sinc_own * sinc_partner).square().mean())
    assert math.isclose(float(own), 0.2052982941500479, rel_tol=0, abs_tol=1e-13)
    assert math.isclose(float(partner), float(own), rel_tol=0, abs_tol=1e-13)
    assert math.isclose(float(blind), 0.3718506073019646, rel_tol=0, abs_tol=1e-13)
    assert float(own) > 0.15
    # A single omitted global theta affects two of six mechanisms: pooled utility
    # still cannot establish every individual factor's predictive fidelity.
    assert float(own) / 3 < 0.15


@pytest.mark.parametrize("parameter_scale", [0.0, 0.749])
def test_exact_gaussian_marginal_mechanism_has_stationary_empirical_moments(
    parameter_scale: float,
) -> None:
    generator = torch.Generator().manual_seed(781)
    count = 16_000
    initial = torch.randn(count, 6, generator=generator, dtype=torch.float64)
    controls = dgp.sample_controls((count, 3), generator)
    theta = parameter_scale * torch.linspace(-1, 1, 6, dtype=torch.float64)
    rollout = dgp.simulate(initial, theta, dgp.encode_controls(controls))
    for step in range(1, 4):
        value = rollout[:, step]
        centered = value - value.mean(0)
        covariance = centered.T @ centered / count
        assert float(value.mean(0).abs().max()) < 0.04
        assert float((covariance - torch.eye(6)).abs().max()) < 0.045


def test_parameter_influence_spans_all_twelve_entries_at_fixed_output() -> None:
    output = torch.linspace(-0.8, 0.7, 6, dtype=torch.float64)
    theta = torch.linspace(-0.72, 0.7, 6, dtype=torch.float64)
    state = torch.linspace(0.5, -0.3, 6, dtype=torch.float64)
    companion = torch.eye(6, dtype=torch.float64).repeat_interleave(2, 0)
    controls = torch.zeros(12, 24, dtype=torch.float64)
    controls[torch.arange(1, 12, 2), 18 + torch.arange(6)] = 0.5
    angle = math.pi * controls[:, 18:]
    phase = dgp.PHASE_GAIN * (angle.cos() * theta + angle.sin() * theta.roll(3))
    innovation = (output - 0.7 * state) / math.sqrt(0.51)
    controls[:, 6:12] = innovation * phase.cos() - companion * phase.sin()
    controls[:, 12:18] = innovation * phase.sin() + companion * phase.cos()
    actions = dgp.encode_controls(controls)
    torch.testing.assert_close(dgp.transition(state, theta, actions), output.expand(12, 6))
    matrices = dgp.analytic_jacobians(state, theta, actions).parameter
    assert _rank(matrices[:, dgp.graphs().parameter]) == 12


def test_state_and_all_action_columns_retain_fixed_output_influence() -> None:
    generator = torch.Generator().manual_seed(468)
    output = torch.linspace(-0.7, 0.8, 6, dtype=torch.float64)
    controls = torch.zeros(128, 24, dtype=torch.float64)
    controls[:, :6] = torch.randn(128, 6, generator=generator, dtype=torch.float64)
    rotation = dgp.state_rotation(dgp.encode_controls(controls))
    controls[:, 6:12] = (output - 0.7 * (rotation @ output[:, None]).squeeze(-1)) / math.sqrt(0.51)
    actions = dgp.encode_controls(controls)
    theta = torch.zeros(6, dtype=torch.float64)
    torch.testing.assert_close(dgp.transition(output, theta, actions), output.expand(128, 6))
    matrices = dgp.analytic_jacobians(output, theta, actions).state[:, :, 0]
    assert _rank(matrices[:, dgp.graphs().state[:, 0]]) == 18
    basis = 0.1 * torch.eye(6, dtype=torch.float64)
    theta = torch.cat((basis, -basis), 0)
    actions = torch.zeros(12, 24, dtype=torch.float64)
    torch.testing.assert_close(dgp.transition(output / 0.7, theta, actions), output.expand(12, 6))
    columns = dgp.analytic_jacobians(output / 0.7, theta, actions).action[:, :, 0]
    differences = columns[:6] - columns[6:]
    for column in range(24):
        assert _rank(differences[:, :, column]) == 6


def test_two_transition_sine_cosine_context_is_informative_near_parameter_boundaries() -> None:
    theta = torch.linspace(-0.749999, 0.749999, 31, dtype=torch.float64)[:, None].expand(31, 6)
    controls = torch.zeros(31, 2, 24, dtype=torch.float64)
    controls[:, 0, 12:18] = 1
    controls[:, 1, 6:12] = 1
    rollout = dgp.simulate(torch.zeros(31, 6), theta, dgp.encode_controls(controls))
    sine = rollout[:, 1] / math.sqrt(0.51)
    cosine = (rollout[:, 2] - 0.7 * rollout[:, 1]) / math.sqrt(0.51)
    recovered = torch.atan2(sine, cosine) / dgp.PHASE_GAIN
    torch.testing.assert_close(recovered, theta, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("rho", [0.0, 1.0, -0.7, float("nan")])
def test_invalid_rho_is_rejected(rho: float) -> None:
    with pytest.raises(ValueError, match="rho"):
        dgp.transition(*_inputs(), rho=rho)
