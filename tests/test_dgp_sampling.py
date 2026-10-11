"""Paper distributions and independent addressable sampling, without rendering datasets."""

from dataclasses import replace

import pytest
import torch

from scjepa.data.random import episode_id, stream_seed
from scjepa.data.sampling import sample_controlled_vector, sample_coupled_springs
from scjepa.data.types import DGPConfig, RendererConfig


def test_controlled_draws_have_prescribed_distributions_and_independent_streams() -> None:
    config = DGPConfig(num_episodes=512, trajectory_length=5, root_seed=19)
    draws = [sample_controlled_vector(config, i) for i in range(config.num_episodes)]
    states = torch.stack([d.initial_position for d in draws]).flatten()
    parameters = torch.stack([d.parameters for d in draws]).flatten()
    actions = torch.stack([d.actions for d in draws]).flatten()
    # Fixed-seed distribution sanity checks, not statistical tests of PRNG quality.
    for values in (states, parameters):
        assert values.dtype == torch.float64
        assert abs(float(values.mean())) < 0.06
        assert abs(float(values.std()) - 1) < 0.06
    assert abs(float((states * parameters).mean())) < 0.06
    assert bool(((actions >= -1) & (actions < 1)).all())
    assert abs(float(actions.mean())) < 0.025
    assert abs(float(actions.square().mean()) - 1 / 3) < 0.025
    reference = draws[0]
    for graph in ("dense", "complementary", "diagonal"):
        matched = sample_controlled_vector(replace(config, parameter_graph=graph), 0)
        torch.testing.assert_close(
            matched.initial_position, reference.initial_position, rtol=0, atol=0
        )
        torch.testing.assert_close(matched.parameters, reference.parameters, rtol=0, atol=0)
        torch.testing.assert_close(matched.actions, reference.actions, rtol=0, atol=0)


def test_spring_distributions_initialization_choices_and_unused_action_sampling() -> None:
    config = DGPConfig(
        family="coupled_springs",
        num_episodes=512,
        trajectory_length=4,
        position_mean=2.0,
        position_std=0.3,
        velocity_mean=-1.0,
        velocity_std=0.7,
    )
    draws = [sample_coupled_springs(config, i) for i in range(config.num_episodes)]
    positions = torch.stack([d.initial_position for d in draws]).flatten()
    velocities = torch.stack(
        [d.initial_velocity for d in draws if d.initial_velocity is not None]
    ).flatten()
    parameters = torch.stack([d.parameters for d in draws]).flatten()
    actions = torch.stack([d.actions for d in draws])
    assert abs(float(positions.mean()) - 2) < 0.03
    assert abs(float(positions.std()) - 0.3) < 0.03
    assert abs(float(velocities.mean()) + 1) < 0.04
    assert abs(float(velocities.std()) - 0.7) < 0.04
    assert bool(((parameters >= 0.5) & (parameters < 1.5)).all())
    assert abs(float(parameters.mean()) - 1) < 0.025
    assert bool(((actions >= -1) & (actions < 1)).all())
    assert torch.count_nonzero(actions[:, 0]) > 0  # sampled but deliberately unused by simulation
    changed = sample_coupled_springs(replace(config, position_std=0.9), 0)
    torch.testing.assert_close(changed.parameters, draws[0].parameters, rtol=0, atol=0)
    torch.testing.assert_close(changed.actions, draws[0].actions, rtol=0, atol=0)
    torch.testing.assert_close(changed.initial_velocity, draws[0].initial_velocity, rtol=0, atol=0)


def test_stream_addresses_and_configs_reject_invalid_values() -> None:
    for kwargs in (
        {"root_seed": -1},
        {"split": "mixed"},
        {"dtype": "uint8"},
        {"position_std": 0.0},
    ):
        with pytest.raises(ValueError, match="root_seed|split|dtype|Gaussian"):
            DGPConfig(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only to coupled"):
        DGPConfig(position_mean=1)
    with pytest.raises(ValueError, match="diagonal"):
        DGPConfig(family="coupled_springs", parameter_graph="dense")
    with pytest.raises(ValueError, match="address"):
        episode_id(0, "train", 2**30)
    with pytest.raises(ValueError, match="stream"):
        stream_seed(0, "train", 0, "")
    assert RendererConfig(track_y=(20.0, 44.0)).track_y == (20.0, 44.0)
