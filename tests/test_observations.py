"""Observation-map invertibility, persistence and RNG isolation."""

import pytest
import torch

from scjepa.data.observations import VectorObservationMap


def test_observation_map_fixed_framewise_rng_isolation_and_persistence() -> None:
    rng = torch.get_rng_state().clone()
    observation_map = VectorObservationMap(seed=997)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    same_map = VectorObservationMap(seed=997)
    restored_map = VectorObservationMap(seed=1, matrix=observation_map.matrix)
    assert observation_map.fingerprint == same_map.fingerprint == restored_map.fingerprint
    assert observation_map.fingerprint != VectorObservationMap(seed=998).fingerprint
    torch.testing.assert_close(
        observation_map.matrix @ observation_map.matrix.T, torch.eye(6, dtype=torch.float64)
    )
    assert observation_map.matrix.count_nonzero() == 36
    state = torch.linspace(-0.8, 0.8, 6, dtype=torch.float64)
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
    states = torch.linspace(-1, 1, 6, dtype=torch.float64) * scale
    recovered = observation_map.inverse(observation_map.forward(states))
    torch.testing.assert_close(recovered, states, rtol=2e-12, atol=2e-14 * max(scale, 1e-12))
