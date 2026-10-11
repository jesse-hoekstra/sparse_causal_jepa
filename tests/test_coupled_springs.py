from typing import cast

import pytest
import torch
from torch import Tensor

from scjepa.data import coupled_springs as springs


def inputs() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.linspace(-0.8, 0.7, 10, dtype=torch.float64),
        torch.linspace(0.6, -0.9, 10, dtype=torch.float64),
        torch.linspace(0.55, 1.45, 10, dtype=torch.float64),
        torch.tensor([-0.7, 0.2, 0.8, -0.1, 0.4], dtype=torch.float64),
    )


def singular_values(matrix: Tensor) -> Tensor:
    return cast(Tensor, torch.linalg.svdvals(matrix))  # pyright: ignore[reportUnknownMemberType]


def numerical_rank(matrix: Tensor) -> int:
    return int((singular_values(matrix) > 1e-12).sum())


def reference_transition(
    previous: Tensor, current: Tensor, parameters: Tensor, action: Tensor
) -> Tensor:
    # Independent scalar reference: partner i xor 1 shares the pair's action.
    return torch.stack(
        [
            2 * current[i]
            - previous[i]
            + 0.01
            * parameters[i]
            * (-2 * current[i] + current[i ^ 1] - 2 * (current[i] - previous[i]) + action[i // 2])
            for i in range(10)
        ]
    )


def test_transition_matches_independent_float64_reference() -> None:
    previous, current, parameters, action = inputs()
    actual = springs.transition(previous, current, parameters, action)
    torch.testing.assert_close(
        actual, reference_transition(previous, current, parameters, action), rtol=1e-14, atol=1e-14
    )
    batch = springs.transition(previous[None], current[None].repeat(3, 1), parameters, action)
    torch.testing.assert_close(batch, actual[None].repeat(3, 1), rtol=1e-14, atol=1e-14)
    # Forces are independent of parameters and share the external action per pair.
    with_action = springs.forces(previous, current, action)
    without_action = springs.forces(previous, current, torch.zeros_like(action))
    torch.testing.assert_close(with_action - without_action, action.repeat_interleave(2))


def test_initialization_and_action_indices_are_not_shifted() -> None:
    position, velocity, parameters, action = inputs()
    actions = torch.stack([action + source / 20 for source in range(9)])
    states = springs.simulate(position, velocity, parameters, actions)
    assert states.shape == (10, 10)
    expected = [position, position + 0.1 * velocity]
    for source in range(1, actions.shape[0]):
        expected.append(
            reference_transition(expected[-2], expected[-1], parameters, actions[source])
        )
    torch.testing.assert_close(states, torch.stack(expected), rtol=1e-14, atol=1e-14)
    changed = actions.clone()
    changed[0] += 1000
    torch.testing.assert_close(states, springs.simulate(position, velocity, parameters, changed))
    changed[7] += 1
    later = springs.simulate(position, velocity, parameters, changed)
    torch.testing.assert_close(states[:8], later[:8])
    assert not torch.equal(states[8], later[8])
    # Eight-frame contexts recover using a1..a6; first prediction then uses a7.
    torch.testing.assert_close(springs.recover_parameters(states[:8], actions[:7]), parameters)
    torch.testing.assert_close(
        states[8], springs.transition(states[6], states[7], parameters, actions[7])
    )


def test_analytic_jacobians_match_autodiff_and_structural_support() -> None:
    previous, current, parameters, action = inputs()
    jacobians = springs.analytic_jacobians(previous, current, parameters, action)
    history = torch.stack((previous, current))
    actions = torch.stack((action * 7, action))

    def apply_transition(h: Tensor, p: Tensor, a: Tensor) -> Tensor:
        return springs.transition(h[0], h[1], p, a[1])

    state_jac, parameter_jac, action_jac = torch.func.jacrev(apply_transition, argnums=(0, 1, 2))(
        history, parameters, actions
    )
    for actual, expected in (
        (jacobians.state, state_jac),
        (jacobians.action, action_jac),
        (jacobians.parameter, parameter_jac),
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    graph = springs.graphs()
    assert torch.equal(jacobians.state != 0, graph.state)
    assert torch.equal(jacobians.action != 0, graph.action)
    assert torch.equal(jacobians.parameter != 0, graph.parameter)
    # At equilibrium all mass derivatives vanish, but the structural graph remains.
    equilibrium = springs.analytic_jacobians(
        torch.zeros_like(previous), torch.zeros_like(current), parameters, torch.zeros_like(action)
    )
    assert not bool(equilibrium.parameter.any())
    assert springs.graphs().parameter.sum() == 10


def test_graph_counts_union_and_oldest_lag_orientation() -> None:
    graph = springs.graphs()
    assert graph.state.shape == (10, 2, 10)
    assert graph.action.shape == (10, 2, 5)
    assert graph.parameter.shape == (10, 10)
    assert torch.equal(graph.state[:, 0], torch.eye(10, dtype=torch.bool))
    assert graph.state.sum() == 30
    assert graph.action[:, 0].sum() == 0
    aggregated = graph.aggregate_lags()
    assert (
        int(aggregated.state.sum()),
        int(aggregated.action.sum()),
        int(aggregated.parameter.sum()),
    ) == (
        20,
        10,
        10,
    )
    assert graph.action[3, 1, 1]
    assert not graph.action[3, 1, 3]


@pytest.mark.parametrize("length", [3, 8, 15])
def test_parameter_recovery_from_informative_contexts(length: int) -> None:
    position, velocity, parameters, action = inputs()
    actions = action[None].repeat(length - 1, 1)
    states = springs.simulate(position, velocity, parameters, actions)
    torch.testing.assert_close(
        springs.recover_parameters(states, actions), parameters, rtol=1e-11, atol=1e-11
    )
    assert springs.recover_parameters(states.float(), actions.float()).dtype == torch.float64


def test_parameter_recovery_rejects_zero_and_poorly_conditioned_forces() -> None:
    actions = torch.zeros(7, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="uninformative.*denominators"):
        springs.recover_parameters(torch.zeros(8, 10, dtype=torch.float64), actions)
    position, velocity, parameters, _ = inputs()
    position[:2], velocity[:2] = 1e-10, 1e-10
    states = springs.simulate(position, velocity, parameters, actions)
    with pytest.raises(ValueError, match=r"denominators: \[0, 1\]"):
        springs.recover_parameters(states, actions)
    with pytest.raises(ValueError, match="K >= 3"):
        springs.recover_parameters(states[:2], actions[:1])
    with pytest.raises(ValueError, match="K-1"):
        springs.recover_parameters(states, actions[1:])


def test_sufficient_influence_numerical_spans_use_both_state_lags() -> None:
    # Small deterministic implementation diagnostic; this is not an identifiability proof.
    # Two independent lag families are required to span all 20 pair-block entries.
    zero, action = torch.zeros(10, dtype=torch.float64), torch.zeros(5, dtype=torch.float64)
    parameters = torch.ones(10, dtype=torch.float64)
    baseline = springs.analytic_jacobians(zero, zero, parameters, action)
    support = springs.graphs().aggregate_lags().state
    previous_directions: list[Tensor] = []
    current_directions: list[Tensor] = []
    for coordinate in range(10):
        varied_parameters = parameters.clone()
        varied_parameters[coordinate] += 0.2
        varied = springs.analytic_jacobians(zero, zero, varied_parameters, action)
        previous_directions.append((varied.state[:, 0] - baseline.state[:, 0])[support])
        current_directions.append((varied.state[:, 1] - baseline.state[:, 1])[support])
    assert numerical_rank(torch.stack(previous_directions)) == 10
    assert numerical_rank(torch.stack(current_directions)) == 10
    both = torch.stack(previous_directions + current_directions)
    assert numerical_rank(both) == 20

    # Each action column spans its two children as inverse masses vary separately.
    for pair in range(5):
        parameters2 = parameters.clone()
        parameters2[2 * pair] += 0.3
        varied = springs.analytic_jacobians(zero, zero, parameters2, action)
        columns = torch.stack((baseline.action[:, 1, pair], varied.action[:, 1, pair]))
        assert numerical_rank(columns) == 2

    # Fix output and theta while varying current states and solving the other lag.
    parameter_directions: list[Tensor] = []
    for current in torch.eye(10, dtype=torch.float64):
        previous = (-baseline.state[:, 1] @ current) / baseline.state[:, 0].diagonal()
        torch.testing.assert_close(
            springs.transition(previous, current, parameters, action), zero, rtol=0, atol=1e-14
        )
        jac = springs.analytic_jacobians(previous, current, parameters, action)
        parameter_directions.append(jac.parameter.diagonal())
    assert numerical_rank(torch.stack(parameter_directions)) == 10
    # Lag invertibility and force-control construction on sampled parameter endpoints.
    for inverse_mass in (0.5, 1.5):
        jac = springs.analytic_jacobians(zero, zero, parameters * inverse_mass, action)
        for lag in range(2):
            assert singular_values(jac.state[:, lag]).min() > 0.9
    force_control = 2 * torch.eye(10, dtype=torch.float64) - springs.stiffness_matrix()
    assert singular_values(force_control).min() > 0.99


def test_invalid_simulation_inputs_fail_explicitly() -> None:
    position, velocity, parameters, action = inputs()
    with pytest.raises(ValueError, match="positive"):
        springs.simulate(position, velocity, parameters * 0, action[None])
    with pytest.raises(ValueError, match="T >= 2"):
        springs.simulate(position, velocity, parameters, action[None][:0])
    with pytest.raises(ValueError, match="dt must be positive"):
        springs.simulate(position, velocity, parameters, action[None], dt=0)
    with pytest.raises(ValueError, match="dtype"):
        springs.transition(position, velocity, parameters, action.float())
