"""Scientific and data-contract checks for the independent-Gaussian DGP."""

import math
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from scjepa.data import independent_gaussian as dgp
from scjepa.data.bundle import load_dataset_bundle, save_dataset_bundle
from scjepa.data.compatibility import validate_model_dataset
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.datasets import EpisodeWindows
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.random import generator
from scjepa.data.sampling import sample_independent_gaussian
from scjepa.data.types import DGPConfig
from scjepa.training.factory import ModelConfig, build_model


def _inputs() -> tuple[Tensor, Tensor, Tensor]:
    rng = torch.Generator().manual_seed(913)
    state, parameter, action = [
        torch.randn(10, generator=rng, dtype=torch.float64) for _ in range(3)
    ]
    return state, parameter, action


def _reflect(vector: list[float], state: list[float]) -> list[float]:
    projection = sum(v * z for v, z in zip(vector, state, strict=True))
    norm = sum(v * v for v in vector)
    return [z - 2 * v * projection / norm for v, z in zip(vector, state, strict=True)]


def _reference_transition(states: Tensor, parameters: Tensor, actions: Tensor) -> Tensor:
    # Scalar reflection operations independently check order, raw controls, and pairing.
    a = [float(value) for value in actions]
    z = [float(value) for value in states]
    alpha = [0.1 * math.tanh(float(value)) for value in parameters]
    b: list[float] = []
    for block in range(2):
        vector = [1.0, *a[4 * block : 4 * block + 4]]
        scaled = [value * (i + 1) for i, value in enumerate(vector)]
        b.extend(_reflect(vector, _reflect(scaled, z[5 * block : 5 * block + 5])))
    c = [value - 0.2 * sum(a) for value in a]
    result: list[float] = []
    for i in range(10):
        r2 = b[i] ** 2 + c[i] ** 2
        phase = alpha[i] + 0.5 * r2 / (1 + r2) * alpha[(i + 5) % 10]
        result.append(b[i] * math.cos(phase) + c[i] * math.sin(phase))
    return torch.tensor(result, dtype=torch.float64)


def test_transition_matches_independent_reference_and_preserves_radial_energy() -> None:
    state, parameter, action = _inputs()
    expected = _reference_transition(state, parameter, action)
    actual = dgp.transition(state, parameter, action)
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)
    rotation = dgp.state_rotation(action)
    torch.testing.assert_close(rotation @ rotation.T, torch.eye(10, dtype=torch.float64))
    torch.testing.assert_close(
        dgp.action_rotation() @ dgp.action_rotation().T, torch.eye(10, dtype=torch.float64)
    )
    b = rotation @ state
    c = dgp.action_rotation() @ action
    r2 = b.square() + c.square()
    alpha = 0.1 * parameter.tanh()
    phase = alpha + 0.5 * r2 / (1 + r2) * alpha.roll(5)
    companion = -b * phase.sin() + c * phase.cos()
    torch.testing.assert_close(actual.square() + companion.square(), r2, rtol=1e-14, atol=1e-14)
    # Raw actions are deliberate: squaring the controls would make U even.
    assert not torch.allclose(dgp.state_rotation(action), dgp.state_rotation(-action))


def test_analytic_jacobians_match_autodiff_and_exact_structural_graphs() -> None:
    state, parameter, action = _inputs()
    actual = dgp.analytic_jacobians(state, parameter, action)
    automatic = cast(
        tuple[Tensor, Tensor, Tensor],
        torch.autograd.functional.jacobian(  # pyright: ignore[reportUnknownMemberType]
            dgp.transition, (state, parameter, action)
        ),
    )
    for calculated, reference in (
        (actual.state[:, 0], automatic[0]),
        (actual.action[:, 0], automatic[2]),
        (actual.parameter, automatic[1]),
    ):
        torch.testing.assert_close(calculated, reference, atol=2e-14, rtol=2e-14)
    masks = dgp.graphs()
    for derivative, mask, count in (
        (actual.state, masks.state, 50),
        (actual.action, masks.action, 100),
        (actual.parameter, masks.parameter, 20),
    ):
        assert int(mask.sum()) == count
        assert torch.equal(derivative != 0, mask)
    zero = torch.zeros(10, dtype=torch.float64)
    derivative_at_zero = cast(
        Tensor,
        torch.autograd.functional.jacobian(dgp.state_rotation, zero),  # pyright: ignore[reportUnknownMemberType]
    )
    # The final raw-action construction deliberately has DU(0) != 0.
    assert float(derivative_at_zero.abs().max()) == 8.0


def test_simulate_broadcasts_and_reuses_the_attached_parameter() -> None:
    state, parameter, action = _inputs()
    parameter.requires_grad_()
    actions = torch.stack((action, -action, 0.5 * action, 2 * action)).expand(2, 4, 10)
    initial = torch.stack((state, -state))
    result = dgp.simulate(initial, parameter, actions)
    assert result.shape == (2, 5, 10)
    expected = initial
    for step in range(4):
        expected = torch.stack(
            [
                _reference_transition(expected[view], parameter.detach(), actions[view, step])
                for view in range(2)
            ]
        )
        torch.testing.assert_close(result[:, step + 1], expected)
    result[:, -1].square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
    assert parameter.grad is not None
    assert bool((parameter.grad.abs() > 0).all())
    float_result = dgp.transition(state.float(), parameter.float(), action.float())
    assert float_result.dtype == torch.float64
    with pytest.raises(ValueError, match="time axis"):
        dgp.simulate(state, parameter, action)


@pytest.mark.parametrize("parameter_value", [0.0, 2.0])
def test_fixed_parameter_gaussian_distribution_is_preserved_over_time(
    parameter_value: float,
) -> None:
    rng = torch.Generator().manual_seed(421)
    count = 20_000
    initial = torch.randn(count, 10, generator=rng, dtype=torch.float64)
    actions = torch.randn(count, 4, 10, generator=rng, dtype=torch.float64)
    parameter = parameter_value * torch.linspace(-1, 1, 10, dtype=torch.float64)
    states = dgp.simulate(initial, parameter, actions)
    directions = torch.randn(10, 16, generator=rng, dtype=torch.float64)
    directions /= directions.square().sum(0).sqrt()
    for frame in range(1, 5):
        values = states[:, frame]
        centered = values - values.mean(0)
        covariance = centered.T @ centered / count
        assert float(values.mean(0).abs().max()) < 0.03
        assert float((covariance - torch.eye(10)).abs().max()) < 0.04
        # Joint projected normality catches failures invisible to covariance alone.
        projected = values @ directions
        assert float((projected.cos().mean(0) - math.exp(-0.5)).abs().max()) < 0.025
        assert float(projected.sin().mean(0).abs().max()) < 0.025


def test_addressed_gaussian_sampling_pairing_and_global_rng_are_preserved() -> None:
    config = DGPConfig(
        family="independent_gaussian",
        root_seed=27,
        num_episodes=3,
        trajectory_length=5,
        episodes_per_parameter=2,
    )
    before = torch.get_rng_state().clone()
    paired = sample_independent_gaussian(config, 1)
    single = sample_independent_gaussian(replace(config, episodes_per_parameter=1), 1)
    torch.testing.assert_close(paired.initial_position[0], single.initial_position, atol=0, rtol=0)
    torch.testing.assert_close(paired.parameters, single.parameters, atol=0, rtol=0)
    torch.testing.assert_close(paired.actions[0], single.actions, atol=0, rtol=0)
    assert not torch.equal(paired.initial_position[0], paired.initial_position[1])
    assert not torch.equal(paired.actions[0], paired.actions[1])
    expected_actions = torch.randn(
        4, 10, dtype=torch.float64, generator=generator(27, "train", 1, "actions")
    )
    torch.testing.assert_close(paired.actions[0], expected_actions, atol=0, rtol=0)
    dataset = build_dgp_dataset(config)
    original = dataset[1]
    dataset[2]
    torch.testing.assert_close(original, dataset[1], atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
    assert set(original) == {"observations", "actions", "episode_ids"}
    assert original["observations"].shape == (2, 5, 10)
    truth = dataset.ground_truth(1)
    assert truth["parameters"].shape == (10,)
    assert bool(truth["valid_transitions"].all())
    torch.testing.assert_close(
        truth["states"],
        dgp.simulate(paired.initial_position, paired.parameters, paired.actions),
        atol=0,
        rtol=0,
    )
    assert dataset.specification["minimum_context_len"] == 3
    assert dataset.specification["generator_version"] == dgp.GENERATOR_VERSION
    validation = build_dgp_dataset(replace(config, split="validation"))
    assert original["episode_ids"] != validation[1]["episode_ids"]
    assert not torch.equal(truth["parameters"], validation.ground_truth(1)["parameters"])
    assert dataset.fingerprint != validation.fingerprint
    with pytest.raises(IndexError):
        sample_independent_gaussian(config, config.num_episodes)


def test_bundle_roundtrip_and_windows_preserve_paired_data_and_realized_map(tmp_path: Path) -> None:
    config = DGPConfig(
        family="independent_gaussian", num_episodes=2, trajectory_length=5, episodes_per_parameter=2
    )
    matrix = torch.eye(10, dtype=torch.float64).roll(2, dims=0)
    dataset = build_dgp_dataset(config, observation_matrix=matrix)
    saved = load_dataset_bundle(save_dataset_bundle(dataset, tmp_path / "bundle"))
    assert saved.fingerprint == dataset.fingerprint
    torch.testing.assert_close(saved.observation_matrix, matrix, atol=0, rtol=0)
    for index in range(2):
        torch.testing.assert_close(saved[index], dataset[index], atol=0, rtol=0)
        torch.testing.assert_close(saved.ground_truth(index), dataset.ground_truth(index))
    truth = dataset.ground_truth(0)
    reference = VectorObservationMap(matrix=matrix).forward(truth["states"]).float()
    torch.testing.assert_close(saved[0]["observations"], reference, atol=0, rtol=0)
    window = EpisodeWindows(saved, window_length=3)[1]
    torch.testing.assert_close(window["observations"], saved[0]["observations"][:, 1:4])
    torch.testing.assert_close(window["actions"], saved[0]["actions"][:, 1:3])


def test_new_family_rejects_wrong_graphs_overrides_and_short_model_context() -> None:
    assert DGPConfig(family="independent_gaussian").parameter_graph == "complementary"
    for graph in ("dense", "diagonal"):
        with pytest.raises(ValueError, match="complementary"):
            DGPConfig(family="independent_gaussian", parameter_graph=graph)
    with pytest.raises(ValueError, match="Gaussian overrides"):
        DGPConfig(family="independent_gaussian", position_std=2.0)
    with pytest.raises(ValueError, match="at least 3"):
        DGPConfig(family="independent_gaussian", trajectory_length=2)
    dataset = build_dgp_dataset(
        DGPConfig(family="independent_gaussian", num_episodes=1, trajectory_length=5)
    )
    config = ModelConfig(observation_dim=10, action_dim=10, context_len=3)
    validate_model_dataset(build_model(config), dataset)
    with pytest.raises(ValueError, match="parameter-recovery minimum"):
        validate_model_dataset(build_model(replace(config, context_len=2)), dataset)


def test_new_generator_version_is_bound_to_dataset_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = build_dgp_dataset(DGPConfig(family="independent_gaussian", num_episodes=1))
    monkeypatch.setattr(dgp, "GENERATOR_VERSION", "different-scientific-definition")
    with pytest.raises(ValueError, match="definition changed"):
        dataset[0]
