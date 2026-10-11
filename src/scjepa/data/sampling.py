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
    """Draw independent standard Gaussian states/parameters and uniform actions."""
    if not 0 <= index < config.num_episodes:
        raise IndexError(index)
    initial = torch.randn(
        10,
        dtype=torch.float64,
        generator=generator(config.root_seed, config.split, index, "initial_position"),
    )
    parameters = torch.randn(
        10,
        dtype=torch.float64,
        generator=generator(config.root_seed, config.split, index, "parameters"),
    )
    actions = (
        2
        * torch.rand(
            config.trajectory_length - 1,
            10,
            dtype=torch.float64,
            generator=generator(config.root_seed, config.split, index, "actions"),
        )
        - 1
    )
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
