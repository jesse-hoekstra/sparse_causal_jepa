"""Invertibility, conditioning and persisted randomness of the fixed observation network."""

import copy
import math
from collections.abc import Iterator
from typing import cast

import pytest
import torch
from torch import Tensor

from diagnostics.neural_observation_oracle import map as observation


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_smooth_activation_inverse_and_global_derivative_bounds() -> None:
    value = torch.tensor(
        [-1e8, -20, -1, -0.1, -1e-12, 0, 1e-12, 0.1, 1, 20, 1e8],
        dtype=torch.float64,
        requires_grad=True,
    )
    mapped = observation.smooth_leaky(value)
    inverse = observation.inverse_smooth_leaky(mapped)
    assert torch.allclose(inverse, value, atol=1e-13, rtol=2e-14)
    derivative = torch.autograd.grad(mapped.sum(), value)[0]
    assert bool((derivative >= observation.GAIN * observation.ALPHA - 1e-15).all())
    assert bool((derivative <= observation.GAIN + 1e-15).all())
    assert mapped[5] == 0
    assert bool((torch.diff(mapped) > 0).all())


def test_full_network_round_trip_and_float32_cache_precision() -> None:
    network = observation.FrozenObservationMap()
    generator = torch.Generator().manual_seed(72)
    physical = torch.randn(8192, 6, dtype=torch.float64, generator=generator)
    old_observations = (physical + 0.1 * physical.pow(3)).float()
    mapped = network.forward(old_observations)
    recovered = network.inverse(mapped)
    assert torch.allclose(recovered, old_observations.double(), atol=2e-13, rtol=2e-13)
    cached_recovery = network.inverse(mapped.float())
    norm_relative = (cached_recovery - old_observations).square().sum(-1).sqrt() / (
        1 + old_observations.double().square().sum(-1).sqrt()
    )
    assert float(norm_relative.max()) < 1e-5
    assert not torch.allclose(mapped, old_observations.double())
    assert torch.isfinite(mapped).all()


def test_network_jacobian_obeys_diffeomorphism_bounds_and_inverse_transmits_gradients() -> None:
    network = observation.FrozenObservationMap()
    value = torch.tensor([1.7, -0.2, 0.6, -2.1, 0.01, 0.4], dtype=torch.float64, requires_grad=True)
    mapped = network.forward(value)
    jacobian = torch.stack(
        [torch.autograd.grad(mapped[index], value, retain_graph=True)[0] for index in range(6)]
    )
    singular_values = cast(
        Tensor,
        torch.linalg.svdvals(jacobian),  # pyright: ignore[reportUnknownMemberType]
    )
    assert float(singular_values.min()) >= (observation.GAIN * observation.ALPHA) ** 3 - 1e-13
    assert float(singular_values.max()) <= observation.GAIN**3 + 1e-13
    assert float(singular_values.max() / singular_values.min()) <= observation.ALPHA**-3 + 1e-10
    round_trip = network.inverse(network.forward(value))
    gradient = torch.autograd.grad(round_trip.sum(), value)[0]
    assert torch.allclose(gradient, torch.ones(6, dtype=torch.float64), atol=2e-13, rtol=2e-13)


def test_realized_matrices_are_reproducible_defensive_and_do_not_change_global_rng() -> None:
    before = torch.get_rng_state().clone()
    first = observation.FrozenObservationMap()
    second = observation.FrozenObservationMap()
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(first.matrices, second.matrices)
    assert first.fingerprint == second.fingerprint
    returned = first.matrices
    returned[0].zero_()
    assert not torch.equal(returned, first.matrices)
    assert observation.settings()["gain"] == math.sqrt(2 / 1.04)


def test_loading_persisted_weights_never_runs_qr_and_rejects_relabeling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = observation.FrozenObservationMap()

    def unexpected_qr(_value: Tensor) -> None:
        raise AssertionError("Loading must use actual persisted weights")

    monkeypatch.setattr(torch.linalg, "qr", unexpected_qr)
    restored = observation.FrozenObservationMap.from_payload(network.payload())
    assert torch.equal(restored.matrices, network.matrices)
    payload = copy.deepcopy(network.payload())
    payload["settings"]["alpha"] = 0.3
    with pytest.raises(ValueError, match="settings changed"):
        observation.FrozenObservationMap.from_payload(payload)
    payload = network.payload()
    payload["matrices"][0, :, 0] *= -1
    with pytest.raises(ValueError, match="weights changed"):
        observation.FrozenObservationMap.from_payload(payload)


def test_invalid_shapes_and_nonorthogonal_weights_fail() -> None:
    with pytest.raises(ValueError, match="four floating"):
        observation.FrozenObservationMap(torch.eye(6))
    with pytest.raises(ValueError, match="orthogonal"):
        observation.FrozenObservationMap(torch.eye(6).expand(4, 6, 6) * 2)
    with pytest.raises(ValueError, match="final dimension six"):
        observation.FrozenObservationMap().forward(torch.zeros(3, 7))
    with pytest.raises(ValueError, match="final dimension six"):
        observation.FrozenObservationMap().inverse(torch.zeros(6, dtype=torch.int64))
