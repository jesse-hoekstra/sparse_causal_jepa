"""Sampling responsibilities separated from deterministic dynamics and observation maps."""

from dataclasses import dataclass

import torch
from torch import Tensor

from scjepa.data.random import generator
from scjepa.data.types import DGPConfig


@dataclass(frozen=True)
class ExogenousDraws:
    """Episode-constant parameters, initial condition, and externally sampled actions."""

    initial_position: Tensor
    parameters: Tensor
    actions: Tensor
    initial_velocity: Tensor | None = None


def sample_controlled_vector(config: DGPConfig, index: int) -> ExogenousDraws:
    """Draw independent initial states/actions, sharing theta within an optional pair."""
    if not 0 <= index < config.num_episodes:
        raise IndexError(index)
    initial_generator = generator(config.root_seed, config.split, index, "initial_position")
    action_generator = generator(config.root_seed, config.split, index, "actions")
    # Draw one trajectory at a time so enabling a pair preserves the first view.
    initial_views = [
        torch.randn(10, dtype=torch.float64, generator=initial_generator)
        for _ in range(config.episodes_per_parameter)
    ]
    parameters = torch.randn(
        10,
        dtype=torch.float64,
        generator=generator(config.root_seed, config.split, index, "parameters"),
    )
    action_views = [
        (
            2
            * torch.rand(
                config.trajectory_length - 1,
                10,
                dtype=torch.float64,
                generator=action_generator,
            )
            - 1
        )
        for _ in range(config.episodes_per_parameter)
    ]
    initial = initial_views[0] if config.episodes_per_parameter == 1 else torch.stack(initial_views)
    actions = action_views[0] if config.episodes_per_parameter == 1 else torch.stack(action_views)
    return ExogenousDraws(initial, parameters, actions)


def sample_independent_gaussian(config: DGPConfig, index: int) -> ExogenousDraws:
    """Draw independent N(0,I10) states/actions and one parameter per group.

    Reuse the existing independently addressed streams. Drawing each view
    successively preserves the first view when a second episode is enabled.
    There is no filtering or conditioning on the sampled parameter or trajectory.
    """
    if not 0 <= index < config.num_episodes:
        raise IndexError(index)
    initial_generator = generator(config.root_seed, config.split, index, "initial_position")
    action_generator = generator(config.root_seed, config.split, index, "actions")
    initial_views = [
        torch.randn(10, dtype=torch.float64, generator=initial_generator)
        for _ in range(config.episodes_per_parameter)
    ]
    parameters = torch.randn(
        10,
        dtype=torch.float64,
        generator=generator(config.root_seed, config.split, index, "parameters"),
    )
    action_views = [
        torch.randn(
            config.trajectory_length - 1, 10, dtype=torch.float64, generator=action_generator
        )
        for _ in range(config.episodes_per_parameter)
    ]
    initial = initial_views[0] if config.episodes_per_parameter == 1 else torch.stack(initial_views)
    actions = action_views[0] if config.episodes_per_parameter == 1 else torch.stack(action_views)
    return ExogenousDraws(initial, parameters, actions)


def sample_coupled_springs(config: DGPConfig, index: int) -> ExogenousDraws:
    """Draw independent inverse masses, Gaussian history, and pair actions.

    Slot a0 is sampled from the same action distribution and retained for indexing,
    but the initialization equation does not use it. Nothing is resampled.
    """
    if not 0 <= index < config.num_episodes:
        raise IndexError(index)
    initial = config.position_mean + config.position_std * torch.randn(
        10,
        dtype=torch.float64,
        generator=generator(config.root_seed, config.split, index, "initial_position"),
    )
    velocity = config.velocity_mean + config.velocity_std * torch.randn(
        10,
        dtype=torch.float64,
        generator=generator(config.root_seed, config.split, index, "initial_velocity"),
    )
    parameters = 0.5 + torch.rand(
        10,
        dtype=torch.float64,
        generator=generator(config.root_seed, config.split, index, "parameters"),
    )
    actions = (
        2
        * torch.rand(
            config.trajectory_length - 1,
            5,
            dtype=torch.float64,
            generator=generator(config.root_seed, config.split, index, "actions"),
        )
        - 1
    )
    return ExogenousDraws(initial, parameters, actions, velocity)
