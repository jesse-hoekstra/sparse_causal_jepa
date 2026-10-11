"""Independent scientific checks for manuscript equations (87)-(99)."""

from typing import cast

import pytest
import torch
from torch import Tensor

from scjepa.data.controlled_vector import (
    ParameterGraph,
    VectorObservationMap,
    analytic_jacobians,
    coefficient_matrices,
    graphs,
    parameter_mask,
    recover_parameters,
    simulate,
    state_mask,
    sufficient_influence_diagnostics,
    transition,
)

VARIANTS: tuple[ParameterGraph, ...] = ("dense", "complementary", "diagonal")


def _inputs() -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(835)
    state = torch.randn(10, generator=generator, dtype=torch.float64)
    parameters = torch.randn(10, generator=generator, dtype=torch.float64)
    actions = 2 * torch.rand(10, generator=generator, dtype=torch.float64) - 1
    return state, parameters, actions


def _reference_matrices(actions: Tensor, variant: ParameterGraph) -> tuple[Tensor, Tensor]:
    # Deliberately scalar loops, with masks reconstructed independently.
    state_matrix = torch.empty(10, 10, dtype=torch.float64)
    parameter_matrix = torch.empty_like(state_matrix)
    action_sum = sum(float(actions[k]) for k in range(10))
    for output in range(10):
        for source in range(10):
            monomial = float(actions[output]) * float(actions[source]) ** 2
            diagonal = int(output == source)
            state_edge = output // 5 == source // 5
            parameter_edge = (
                variant == "dense"
                or (variant == "complementary" and output % 5 == source % 5)
                or (variant == "diagonal" and output == source)
            )
            state_matrix[output, source] = diagonal / 2 + state_edge * monomial / 40
            parameter_matrix[output, source] = (
                diagonal * (1 + action_sum / 40) + parameter_edge * monomial / 40
            )
    return state_matrix, parameter_matrix


@pytest.mark.parametrize("variant", VARIANTS)
def test_transition_and_coefficients_independent_float64_reference(variant: ParameterGraph) -> None:
    state, parameters, actions = _inputs()
    expected_a, expected_b = _reference_matrices(actions, variant)
    actual_a, actual_b = coefficient_matrices(actions, variant)
    torch.testing.assert_close(actual_a, expected_a, rtol=1e-14, atol=1e-15)
    torch.testing.assert_close(actual_b, expected_b, rtol=1e-14, atol=1e-15)
    expected = torch.tensor(
        [
            sum(
                float(expected_a[i, j]) * float(state[j])
                + float(expected_b[i, j]) * float(parameters[j])
                for j in range(10)
            )
            for i in range(10)
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(transition(state, parameters, actions, variant), expected)
    # A non-symmetric pair of actions catches transposing F_ij = a_i*a_j^2.
    assert not torch.isclose(actual_a[0, 1], actual_a[1, 0])


@pytest.mark.parametrize(
    ("variant", "count"), [("dense", 100), ("complementary", 20), ("diagonal", 10)]
)
def test_structural_graph_variants_and_aggregation(variant: ParameterGraph, count: int) -> None:
    graph = graphs(variant)
    assert graph.state.dtype == graph.action.dtype == graph.parameter.dtype == torch.bool
    assert graph.state.shape == graph.action.shape == (10, 1, 10)
    assert graph.parameter.shape == (10, 10)
    aggregated = graph.aggregate_lags()
    assert int(aggregated.state.sum()) == 50
    assert int(aggregated.action.sum()) == 100
    assert int(aggregated.parameter.sum()) == count
    torch.testing.assert_close(aggregated.state, state_mask())
    torch.testing.assert_close(aggregated.parameter, parameter_mask(variant))
    # Structural action edges remain present when all point derivatives vanish.
    zero = torch.zeros(10, dtype=torch.float64)
    assert analytic_jacobians(zero, zero, zero, variant).action.count_nonzero() == 0
    assert graph.action.all()
    if variant == "complementary":
        assert graph.parameter[0, 5]
        assert graph.parameter[5, 0]
        assert not graph.parameter[0, 1]
        assert not graph.parameter[0, 6]


@pytest.mark.parametrize("variant", VARIANTS)
def test_history_actions_and_constant_parameters_match_independent_recurrence(
    variant: ParameterGraph,
) -> None:
    initial, parameters, action = _inputs()
    actions = torch.stack((action, -action, action / 3, torch.zeros_like(action)))
    states = simulate(initial, parameters, actions, variant)
    assert states.dtype == torch.float64
    assert states.shape == (5, 10)
    torch.testing.assert_close(states[0], initial, rtol=0, atol=0)
    expected = initial
    for t, current_action in enumerate(actions):
        matrix_a, matrix_b = _reference_matrices(current_action, variant)
        expected = matrix_a @ expected + matrix_b @ parameters
        torch.testing.assert_close(states[t + 1], expected, rtol=1e-14, atol=1e-14)
    changed = actions.clone()
    changed[2] = -changed[2]
    changed_states = simulate(initial, parameters, changed, variant)
    torch.testing.assert_close(changed_states[:3], states[:3], rtol=0, atol=0)
    assert not torch.allclose(changed_states[3], states[3])


def test_simulation_broadcasts_batches_and_preserves_gradients() -> None:
    initial, parameters, action = _inputs()
    initial = initial.repeat(2, 1).requires_grad_()
    parameters = parameters.requires_grad_()
    actions = action.repeat(2, 4, 1).requires_grad_()
    states = simulate(initial, parameters, actions, "complementary")
    assert states.shape == (2, 5, 10)
    torch.testing.assert_close(
        states[1], simulate(initial[1], parameters, actions[1], "complementary")
    )
    states[:, -1].square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
    for value in (initial, parameters, actions):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
    # Empty action sequences preserve the explicit initial frame.
    empty = simulate(initial.detach(), parameters.detach(), torch.zeros(0, 10), "dense")
    assert empty.shape == (2, 1, 10)
    torch.testing.assert_close(empty[:, 0], initial.detach())


@pytest.mark.parametrize("variant", VARIANTS)
def test_analytic_jacobians_match_autodiff_and_structural_support(variant: ParameterGraph) -> None:
    state, parameters, actions = _inputs()

    def apply_transition(z: Tensor, theta: Tensor, a: Tensor) -> Tensor:
        return transition(z, theta, a, variant)

    automatic = cast(
        tuple[Tensor, Tensor, Tensor],
        torch.autograd.functional.jacobian(  # pyright: ignore[reportUnknownMemberType]
            apply_transition,
            (state, parameters, actions),
        ),
    )
    analytic = analytic_jacobians(state, parameters, actions, variant)
    graph = graphs(variant)
    for actual, expected, mask in (
        (analytic.state.squeeze(1), automatic[0], graph.state.squeeze(1)),
        (analytic.action.squeeze(1), automatic[2], graph.action.squeeze(1)),
        (analytic.parameter, automatic[1], graph.parameter),
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-14)
        assert torch.equal(actual.abs() > 1e-12, mask)
    batched = analytic_jacobians(state.repeat(2, 1), parameters, actions, variant)
    assert batched.state.shape == batched.action.shape == (2, 10, 1, 10)
    assert batched.parameter.shape == (2, 10, 10)
    torch.testing.assert_close(batched.action[1], analytic.action)


@pytest.mark.parametrize("variant", VARIANTS)
def test_inverse_observation_and_parameter_recovery_from_one_transition(
    variant: ParameterGraph,
) -> None:
    state, parameters, actions = _inputs()
    observation_map = VectorObservationMap(seed=94)
    next_state = transition(state, parameters, actions, variant)
    observed = observation_map.forward(torch.stack((state, next_state)))
    recovered_states = observation_map.inverse(observed)
    torch.testing.assert_close(
        recovered_states, torch.stack((state, next_state)), atol=2e-14, rtol=2e-14
    )
    recovered = recover_parameters(recovered_states[0], recovered_states[1], actions, variant)
    torch.testing.assert_close(recovered, parameters, atol=2e-14, rtol=2e-14)
    batched = recover_parameters(state.repeat(3, 1), next_state, actions.repeat(3, 1), variant)
    assert batched.shape == (3, 10)
    torch.testing.assert_close(batched, parameters.expand(3, -1))


def test_observation_map_fixed_framewise_rng_isolation_and_persistence() -> None:
    rng = torch.get_rng_state().clone()
    observation_map = VectorObservationMap(seed=997)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    same_map = VectorObservationMap(seed=997)
    restored_map = VectorObservationMap(seed=1, matrix=observation_map.matrix)
    assert observation_map.fingerprint == same_map.fingerprint == restored_map.fingerprint
    assert observation_map.fingerprint != VectorObservationMap(seed=998).fingerprint
    torch.testing.assert_close(
        observation_map.matrix @ observation_map.matrix.T, torch.eye(10, dtype=torch.float64)
    )
    assert observation_map.matrix.count_nonzero() == 100
    state, _, _ = _inputs()
    sequence = torch.stack((state, state * 2, -state))
    observed = observation_map.forward(sequence)
    for i in range(3):
        torch.testing.assert_close(
            observed[i], observation_map.forward(sequence[i]), rtol=1e-14, atol=1e-14
        )
    permutation = torch.tensor([2, 0, 1])
    torch.testing.assert_close(
        observation_map.forward(sequence[permutation]), observed[permutation]
    )
    # Matrix access cannot mutate the actual observation map.
    copy = observation_map.matrix
    copy.zero_()
    torch.testing.assert_close(observation_map.forward(sequence), observed, rtol=0, atol=0)
    torch.testing.assert_close(restored_map.forward(sequence), observed, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [0.0, 1e-12, 1.0, 1e3, 1e30])
def test_stable_cubic_inverse_over_wide_finite_scales(scale: float) -> None:
    observation_map = VectorObservationMap(seed=31)
    states = torch.linspace(-1, 1, 10, dtype=torch.float64) * scale
    recovered = observation_map.inverse(observation_map.forward(states))
    torch.testing.assert_close(recovered, states, rtol=2e-12, atol=2e-14 * max(scale, 1e-12))


def test_observation_roundtrip_and_recovery_remain_differentiable() -> None:
    state, parameters, action = _inputs()
    state.requires_grad_()
    parameters.requires_grad_()
    action.requires_grad_()
    observation_map = VectorObservationMap(seed=92)
    next_state = transition(state, parameters, action, "dense")
    recovered = recover_parameters(
        observation_map.inverse(observation_map.forward(state)),
        observation_map.inverse(observation_map.forward(next_state)),
        action,
        "dense",
    )
    recovered.sum().backward()  # pyright: ignore[reportUnknownMemberType]
    assert parameters.grad is not None
    assert state.grad is not None
    assert action.grad is not None
    torch.testing.assert_close(parameters.grad, torch.ones_like(parameters), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(state.grad, torch.zeros_like(state), atol=1e-13, rtol=0)
    torch.testing.assert_close(action.grad, torch.zeros_like(action), atol=1e-13, rtol=0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_sufficient_influence_constructions_with_explicit_tolerances(
    variant: ParameterGraph,
) -> None:
    rng = torch.get_rng_state().clone()
    report = sufficient_influence_diagnostics(variant, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert report.passed, report
    assert report.state_rank == 50
    assert report.parameter_rank == int(parameter_mask(variant).sum())
    assert report.action_ranks == (10,) * 10
    assert report.state_fixed_point_residual < 1e-13
    assert report.parameter_fixed_output_residual < 1e-13
    assert report.action_fixed_output_residual < 1e-13
    assert report.action_derivative_residual == 0
    assert sufficient_influence_diagnostics(variant, samples=2).passed is False


def test_invalid_inputs_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="unknown parameter graph"):
        parameter_mask(cast(ParameterGraph, "wrong"))
    with pytest.raises(ValueError, match="final dimension"):
        coefficient_matrices(torch.zeros(5), "dense")
    with pytest.raises(ValueError, match="positive and finite"):
        coefficient_matrices(torch.zeros(10), "dense", epsilon=0)
    with pytest.raises(ValueError, match="time axis"):
        simulate(torch.zeros(10), torch.zeros(10), torch.zeros(10), "dense")
    with pytest.raises(ValueError, match="orthogonal"):
        VectorObservationMap(matrix=torch.ones(10, 10))
    with pytest.raises(ValueError, match="must be positive"):
        sufficient_influence_diagnostics("dense", atol=0)
