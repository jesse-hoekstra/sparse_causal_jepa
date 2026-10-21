"""Independent equation, influence, distribution and data-contract checks for the new DGP."""

import hashlib
import math
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from scjepa.data import simplified_gaussian as dgp
from scjepa.data.bundle import load_dataset_bundle, save_dataset_bundle
from scjepa.data.compatibility import validate_model_dataset
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.datasets import EpisodeWindows
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.random import generator
from scjepa.data.sampling import sample_simplified_gaussian
from scjepa.data.types import DGPConfig
from scjepa.training.factory import ModelConfig, build_model


def _actions(shape: tuple[int, ...], rng: torch.Generator) -> Tensor:
    controls = torch.cat(
        (
            torch.randn(*shape, 10, dtype=torch.float64, generator=rng),
            2 * torch.rand(*shape, 6, dtype=torch.float64, generator=rng) - 1,
        ),
        -1,
    )
    return controls - controls.sum(-1, keepdim=True) / 8


def _inputs() -> tuple[Tensor, Tensor, Tensor]:
    rng = torch.Generator().manual_seed(801)
    return (
        torch.randn(6, dtype=torch.float64, generator=rng),
        1.5 * torch.rand(6, dtype=torch.float64, generator=rng) - 0.75,
        _actions((), rng),
    )


def _rank(matrix: Tensor) -> int:
    return int(
        cast(
            Tensor,
            torch.linalg.matrix_rank(matrix, atol=1e-10, rtol=1e-10),  # pyright: ignore[reportUnknownMemberType]
        )
    )


def _reflect(vector: list[float], state: list[float]) -> list[float]:
    scale = 2 * sum(v * z for v, z in zip(vector, state, strict=True))
    scale /= sum(v * v for v in vector)
    return [z - scale * v for v, z in zip(vector, state, strict=True)]


def _reference(states: Tensor, parameters: Tensor, actions: Tensor) -> Tensor:
    a = [float(value) for value in actions]
    controls = [value - sum(a) / 8 for value in a]
    z = [float(value) for value in states]
    theta = [float(value) for value in parameters]
    b: list[float] = []
    for block in range(2):
        vector = [1.0, *controls[2 * block : 2 * block + 2]]
        scaled = [(index + 1) * value for index, value in enumerate(vector)]
        b.extend(_reflect(vector, _reflect(scaled, z[3 * block : 3 * block + 3])))
    result: list[float] = []
    for i in range(6):
        phase = theta[i] + controls[10 + i] * theta[(i + 3) % 6]
        result.append(math.cos(phase) * b[i] + math.sin(phase) * controls[4 + i])
    return torch.tensor(result, dtype=torch.float64)


def test_transition_matches_scalar_reference_and_rotation_order() -> None:
    state, parameter, action = _inputs()
    torch.testing.assert_close(
        dgp.transition(state, parameter, action),
        _reference(state, parameter, action),
        atol=3e-15,
        rtol=3e-15,
    )
    rotation = dgp.state_rotation(action)
    torch.testing.assert_close(rotation @ rotation.T, torch.eye(6, dtype=torch.float64))
    mixing = dgp.action_rotation()
    torch.testing.assert_close(mixing @ mixing, torch.eye(16, dtype=torch.float64))
    torch.testing.assert_close(dgp.encode_controls(dgp.decode_actions(action)), action)
    assert not torch.allclose(dgp.state_rotation(action), dgp.state_rotation(-action))


def test_exact_jacobians_match_autodiff_and_structural_support() -> None:
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
        (actual.action, masks.action, 96),
        (actual.parameter, masks.parameter, 12),
    ):
        assert int(mask.sum()) == count
        assert torch.equal(derivative != 0, mask)
    batched = dgp.analytic_jacobians(state.expand(2, 6), parameter, action.expand(2, 16))
    torch.testing.assert_close(batched.action[1], actual.action)


def test_simulation_broadcasts_and_keeps_parameter_gradients_attached() -> None:
    state, parameter, action = _inputs()
    parameter.requires_grad_()
    initial = torch.stack((state, -state))
    actions = torch.stack((action, -action, 0.5 * action, -0.5 * action)).expand(2, 4, 16)
    result = dgp.simulate(initial, parameter, actions)
    assert result.shape == (2, 5, 6)
    expected = initial
    for step in range(4):
        expected = torch.stack(
            [_reference(expected[v], parameter.detach(), actions[v, step]) for v in range(2)]
        )
        torch.testing.assert_close(result[:, step + 1], expected)
    result[:, -1].square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
    assert parameter.grad is not None
    assert bool((parameter.grad.abs() > 0).all())
    assert dgp.transition(state.float(), parameter.float(), action.float()).dtype == torch.float64
    with pytest.raises(ValueError, match="time axis"):
        dgp.simulate(state, parameter, action)
    with pytest.raises(ValueError, match="dimension 16"):
        dgp.transition(state, parameter, action[:6])


@pytest.mark.parametrize("scale", [0.0, 0.7])
def test_conditional_gaussian_stationarity_over_four_transitions(scale: float) -> None:
    rng = torch.Generator().manual_seed(547)
    count = 20_000
    initial = torch.randn(count, 6, generator=rng, dtype=torch.float64)
    actions = _actions((count, 4), rng)
    parameter = scale * torch.linspace(-1, 1, 6, dtype=torch.float64)
    states = dgp.simulate(initial, parameter, actions)
    directions = torch.randn(6, 12, generator=rng, dtype=torch.float64)
    directions /= directions.square().sum(0).sqrt()
    for frame in range(1, 5):
        values = states[:, frame]
        centered = values - values.mean(0)
        covariance = centered.T @ centered / count
        assert float(values.mean(0).abs().max()) < 0.035
        assert float((covariance - torch.eye(6)).abs().max()) < 0.04
        projected = values @ directions
        assert float((projected.cos().mean(0) - math.exp(-0.5)).abs().max()) < 0.025
        assert float(projected.sin().mean(0).abs().max()) < 0.025


def test_state_influence_spans_eighteen_dimensions_on_fixed_point_fiber() -> None:
    rng = torch.Generator().manual_seed(90)
    count = 128
    output = torch.linspace(-0.9, 0.8, 6, dtype=torch.float64)
    parameter = torch.full((6,), 0.3, dtype=torch.float64)
    controls = torch.zeros(count, 16, dtype=torch.float64)
    controls[:, :4] = torch.randn(count, 4, generator=rng, dtype=torch.float64)
    rotation = dgp.state_rotation(dgp.encode_controls(controls))
    b = (rotation @ output[:, None]).squeeze(-1)
    controls[:, 4:10] = (output - math.cos(0.3) * b) / math.sin(0.3)
    actions = dgp.encode_controls(controls)
    torch.testing.assert_close(dgp.transition(output, parameter, actions), output.expand(count, 6))
    matrices = dgp.analytic_jacobians(output, parameter, actions).state[:, :, 0]
    masked = matrices[:, dgp.graphs().state[:, 0]]
    assert _rank(masked) == 18


def test_parameter_influence_spans_twelve_dimensions_at_fixed_output_and_parameter() -> None:
    output = torch.linspace(-0.8, 0.7, 6, dtype=torch.float64)
    parameter = torch.linspace(-0.6, 0.6, 6, dtype=torch.float64)
    companion = torch.eye(6, dtype=torch.float64).repeat_interleave(2, 0)
    controls = torch.zeros(12, 16, dtype=torch.float64)
    controls[torch.arange(1, 12, 2), 10 + torch.arange(6)] = 0.5
    phase = parameter + controls[:, 10:] * parameter.roll(3)
    state = output * phase.cos() - companion * phase.sin()
    controls[:, 4:10] = output * phase.sin() + companion * phase.cos()
    actions = dgp.encode_controls(controls)
    torch.testing.assert_close(dgp.transition(state, parameter, actions), output.expand(12, 6))
    matrices = dgp.analytic_jacobians(state, parameter, actions).parameter
    assert _rank(matrices[:, dgp.graphs().parameter]) == 12


def test_every_action_column_has_full_six_dimensional_influence_at_fixed_output() -> None:
    output = torch.linspace(-0.7, 0.8, 6, dtype=torch.float64)
    basis = 0.01 * torch.eye(6, dtype=torch.float64)
    parameter = torch.cat((basis, -basis), 0)
    state = output / parameter.cos()
    actions = torch.zeros(12, 16, dtype=torch.float64)
    torch.testing.assert_close(dgp.transition(state, parameter, actions), output.expand(12, 6))
    columns = dgp.analytic_jacobians(state, parameter, actions).action[:, :, 0]
    # Opposite parameter perturbations cancel the constant rotation term.
    differences = columns[:6] - columns[6:]
    for j in range(16):
        assert _rank(differences[:, :, j]) == 6


def test_two_transitions_resolve_real_one_transition_parameter_ambiguity() -> None:
    # This first transition has 64 distinct bounded parameter solutions despite
    # every angle being inside (-pi/2,pi/2). The second observed action separates them.
    parameter = torch.full((6,), 0.3, dtype=torch.float64)
    initial = torch.ones(6, dtype=torch.float64)
    rng = torch.Generator().manual_seed(618)
    actions = torch.stack((torch.zeros(16, dtype=torch.float64), _actions((), rng)))
    states = dgp.simulate(initial, parameter, actions)
    controls = dgp.decode_actions(actions[0])
    b = dgp.state_rotation(actions[0]) @ states[0]
    c = controls[4:10]
    phase_candidates: list[list[float]] = []
    for i in range(6):
        beta = math.atan2(float(c[i]), float(b[i]))
        angle = math.acos(float(states[1, i] / torch.hypot(b[i], c[i])))
        phase_candidates.append(
            [
                phase
                for sign in (-1, 1)
                for turn in (-1, 0, 1)
                if -1.5 < (phase := beta + sign * angle + 2 * math.pi * turn) < 1.5
            ]
        )
    phases = torch.tensor(list(product(*phase_candidates)), dtype=torch.float64)
    identity = torch.eye(6, dtype=torch.float64)
    matrix = identity + controls[10:, None] * identity.roll(3, 0)
    candidates = cast(
        Tensor,
        torch.linalg.solve(matrix, phases.T),  # pyright: ignore[reportUnknownMemberType]
    ).T
    candidates = candidates[(candidates.abs() < 0.75).all(-1)]
    assert len(candidates) == 64
    torch.testing.assert_close(
        dgp.transition(states[0], candidates, actions[0]), states[1].expand(64, 6)
    )
    residual = (dgp.transition(states[1], candidates, actions[1]) - states[2]).abs().amax(-1)
    recovered = candidates[residual < 1e-10]
    assert recovered.shape == (1, 6)
    torch.testing.assert_close(recovered[0], parameter, atol=1e-14, rtol=1e-14)


def test_addressed_sampling_preserves_pairing_and_exposes_only_mixed_actions() -> None:
    config = DGPConfig(
        family="simplified_gaussian",
        num_episodes=3,
        trajectory_length=5,
        episodes_per_parameter=2,
        root_seed=37,
    )
    before = torch.get_rng_state().clone()
    draws = sample_simplified_gaussian(config, 1)
    single = sample_simplified_gaussian(replace(config, episodes_per_parameter=1), 1)
    torch.testing.assert_close(draws.initial_position[0], single.initial_position, atol=0, rtol=0)
    torch.testing.assert_close(draws.actions[0], single.actions, atol=0, rtol=0)
    torch.testing.assert_close(draws.parameters, single.parameters, atol=0, rtol=0)
    assert not torch.equal(draws.initial_position[0], draws.initial_position[1])
    assert not torch.equal(draws.actions[0], draws.actions[1])
    expected_parameters = (
        1.5
        * torch.rand(
            6,
            dtype=torch.float64,
            generator=generator(37, "train", 1, "simplified_gaussian/parameters"),
        )
        - 0.75
    )
    torch.testing.assert_close(draws.parameters, expected_parameters, atol=2e-16, rtol=0)
    expected_actions = _actions((4,), generator(37, "train", 1, "simplified_gaussian/actions"))
    torch.testing.assert_close(draws.actions[0], expected_actions, atol=1e-15, rtol=1e-15)
    controls = dgp.decode_actions(draws.actions)
    assert bool((controls[..., 10:].abs() < 1).all())
    dataset = build_dgp_dataset(config)
    item = dataset[1]
    dataset[2]
    torch.testing.assert_close(item, dataset[1], atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
    assert set(item) == {"observations", "actions", "episode_ids"}
    assert item["observations"].shape == (2, 5, 6)
    assert item["actions"].shape == (2, 4, 16)
    truth = dataset.ground_truth(1)
    assert truth["parameters"].shape == (6,)
    assert bool(truth["valid_transitions"].all())
    torch.testing.assert_close(truth["controls"], controls, atol=0, rtol=0)
    torch.testing.assert_close(
        truth["states"], dgp.simulate(draws.initial_position, draws.parameters, draws.actions)
    )
    validation = build_dgp_dataset(replace(config, split="validation"))
    assert item["episode_ids"] != validation[1]["episode_ids"]
    assert dataset.fingerprint != validation.fingerprint
    assert not torch.equal(truth["parameters"], validation.ground_truth(1)["parameters"])
    with pytest.raises(IndexError):
        sample_simplified_gaussian(config, config.num_episodes)


def test_observation_dimensions_preserve_old_fingerprint_and_new_inverse() -> None:
    old = VectorObservationMap()
    expected = hashlib.sha256(
        b"controlled-vector-W:<f8:10,10:" + old.matrix.numpy().astype("<f8").tobytes()
    ).hexdigest()
    assert old.fingerprint == expected
    six = VectorObservationMap(dimension=6)
    state, _, _ = _inputs()
    torch.testing.assert_close(six.inverse(six.forward(state)), state)
    assert six.matrix.shape == (6, 6)
    assert six.fingerprint != old.fingerprint
    with pytest.raises(ValueError, match="6,6"):
        VectorObservationMap(matrix=old.matrix, dimension=6)


def test_bundle_roundtrip_restores_w6_and_keeps_labels_separate(tmp_path: Path) -> None:
    config = DGPConfig(
        family="simplified_gaussian",
        num_episodes=2,
        trajectory_length=5,
        episodes_per_parameter=2,
    )
    matrix = torch.eye(6, dtype=torch.float64).roll(2, dims=0)
    dataset = build_dgp_dataset(config, observation_matrix=matrix)
    saved = load_dataset_bundle(save_dataset_bundle(dataset, tmp_path / "bundle"))
    assert saved.fingerprint == dataset.fingerprint
    assert dataset.specification["generator_version"] == dgp.GENERATOR_VERSION
    torch.testing.assert_close(saved.observation_matrix, matrix, atol=0, rtol=0)
    for index in range(2):
        torch.testing.assert_close(saved[index], dataset[index], atol=0, rtol=0)
        torch.testing.assert_close(saved.ground_truth(index), dataset.ground_truth(index))
    observation = VectorObservationMap(matrix=matrix, dimension=6)
    torch.testing.assert_close(
        saved[0]["observations"],
        observation.forward(dataset.ground_truth(0)["states"]).float(),
        atol=0,
        rtol=0,
    )
    window = EpisodeWindows(saved, window_length=3)[1]
    torch.testing.assert_close(window["actions"], saved[0]["actions"][:, 1:3])


def test_config_context_and_version_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    assert DGPConfig(family="simplified_gaussian").parameter_graph == "complementary"
    for graph in ("dense", "diagonal"):
        with pytest.raises(ValueError, match="complementary"):
            DGPConfig(family="simplified_gaussian", parameter_graph=graph)
    with pytest.raises(ValueError, match="Gaussian overrides"):
        DGPConfig(family="simplified_gaussian", position_std=2)
    with pytest.raises(ValueError, match="at least 3"):
        DGPConfig(family="simplified_gaussian", trajectory_length=2)
    dataset = build_dgp_dataset(
        DGPConfig(family="simplified_gaussian", num_episodes=1, trajectory_length=5)
    )
    config = ModelConfig(observation_dim=6, state_dim=6, param_dim=6, action_dim=16, context_len=3)
    validate_model_dataset(build_model(config), dataset)
    with pytest.raises(ValueError, match="parameter-recovery minimum"):
        validate_model_dataset(build_model(replace(config, context_len=2)), dataset)
    monkeypatch.setattr(dgp, "GENERATOR_VERSION", "changed-scientific-definition")
    with pytest.raises(ValueError, match="definition changed"):
        dataset[0]
