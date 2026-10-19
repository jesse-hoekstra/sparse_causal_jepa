"""Compose scientific sampling, simulation, observation, and structural graph functions."""

from dataclasses import asdict
from functools import cache
from typing import cast

import torch
from torch import Tensor

from scjepa.data import controlled_vector, coupled_springs
from scjepa.data.random import STREAM_VERSION, episode_id
from scjepa.data.renderer import SpringRenderer
from scjepa.data.sampling import sample_controlled_vector, sample_coupled_springs
from scjepa.data.types import DGPConfig, GraphMasks, ScientificEpisode

GENERATOR_VERSION = "paper-dgps-v1"


def _common_specification(config: DGPConfig) -> dict[str, object]:
    return {
        "generator_version": GENERATOR_VERSION,
        "stream_version": STREAM_VERSION,
        "simulation_dtype": "float64",
        "storage_dtype": config.dtype,
        "state_dim": 10,
        "param_dim": 10,
        "trajectory_length": config.trajectory_length,
        "matrix_orientation": "rows=output,columns=input",
        "lag_order": "oldest_to_newest",
        "episode_id": "(root_seed << 32) | (split_index << 30) | episode_index",
        "stream_labels": ["initial_position", "initial_velocity", "parameters", "actions"],
        "split_order": ["train", "validation", "evaluation_calibration", "test"],
        "indexing": "a[t] is aligned to source frame t and next frame t+1",
        "noise": "none",
    }


class ControlledVectorDGP:
    """Positive action-dependent coupling with a fixed nonlinear observation map."""

    def __init__(self, config: DGPConfig) -> None:
        """Freeze this process's configuration and realize the separate observation map."""
        if config.family != "controlled_vector":
            raise ValueError("ControlledVectorDGP requires family=controlled_vector")
        self.config = config
        self.observation_map = controlled_vector.VectorObservationMap(seed=config.observation_seed)
        self.observation_matrix: Tensor | None = self.observation_map.matrix.clone()

    def restore_observation_matrix(self, matrix: Tensor) -> None:
        """Restore the realized orthogonal map rather than rerunning seeded QR."""
        self.observation_map = controlled_vector.VectorObservationMap(
            seed=self.config.observation_seed, matrix=matrix
        )
        self.observation_matrix = self.observation_map.matrix.clone()

    def specification(self) -> dict[str, object]:
        """Describe prescribed dynamics and fixed realized observation map."""
        return _common_specification(self.config) | {
            "generator_version": "controlled-vector-positive-paired-v3",
            "family": self.config.family,
            "parameter_graph": self.config.parameter_graph,
            "action_dim": 10,
            "history_len": 1,
            "minimum_prediction_start": 0,
            "minimum_context_len": 2,
            "observation_shape": [10],
            "episodes_per_parameter": self.config.episodes_per_parameter,
            "dataset_item": "one parameter draw with independent initial states and actions",
            "dynamics": {
                "interaction": "F[i,j]=(1+a[i]**2)*(1+a[j]**3)",
                "state_self": 0.5,
                "state_coupling": controlled_vector.STATE_COUPLING,
                "parameter_coupling": controlled_vector.PARAMETER_COUPLING,
                "parameter_gain": "exp(sum(a)/sqrt(10))",
                "action_scale": controlled_vector.ACTION_SCALE,
            },
            "initial_distribution": "independent N(0,I10) z0 and theta",
            "action_distribution": "independent Uniform(-1,1)",
            "observation_map": {
                "kind": "orthogonal_cubic",
                "formula": "psi(W z), psi(u)=u+0.1*u^3",
                "seed": self.config.observation_seed,
                "matrix_fingerprint": self.observation_map.fingerprint,
            },
            "initialization": "z0 supplied; all action transitions valid",
        }

    def graphs(self) -> GraphMasks:
        """Return structural masks, independent of all realized draws."""
        assert self.config.parameter_graph is not None
        return controlled_vector.graphs(
            cast(controlled_vector.ParameterGraph, self.config.parameter_graph)
        )

    def generate(self, episode_index: int) -> ScientificEpisode:
        """Generate one addressed trajectory and keep all labels in a separate payload."""
        draws = sample_controlled_vector(self.config, episode_index)
        assert self.config.parameter_graph is not None
        states = controlled_vector.simulate(
            draws.initial_position,
            draws.parameters,
            draws.actions,
            cast(controlled_vector.ParameterGraph, self.config.parameter_graph),
        )
        a_matrix, b_matrix = controlled_vector.coefficient_matrices(
            draws.actions, cast(controlled_vector.ParameterGraph, self.config.parameter_graph)
        )
        dtype = torch.float32 if self.config.dtype == "float32" else torch.float64
        return ScientificEpisode(
            self.observation_map.forward(states).to(dtype=dtype),
            draws.actions.to(dtype=dtype),
            episode_id(self.config.root_seed, self.config.split, episode_index),
            {
                "states": states,
                "parameters": draws.parameters,
                "initial_state": draws.initial_position,
                "actions": draws.actions,
                "state_matrices": a_matrix,
                "parameter_matrices": b_matrix,
                "valid_transitions": torch.ones(draws.actions.shape[:-1], dtype=torch.bool),
            },
        )


class CoupledSpringsDGP:
    """Two-state displacement dynamics and the fixed native-resolution smooth renderer."""

    def __init__(self, config: DGPConfig) -> None:
        """Bind explicit Gaussian initialization and renderer settings."""
        if config.family != "coupled_springs":
            raise ValueError("CoupledSpringsDGP requires family=coupled_springs")
        self.config = config
        self.renderer = SpringRenderer(config.renderer)
        self.observation_matrix: Tensor | None = None

    def restore_observation_matrix(self, matrix: Tensor) -> None:
        """Reject a matrix asset: this family's fixed map is its Gaussian renderer."""
        del matrix
        raise ValueError("coupled_springs has no observation mixing matrix")

    def specification(self) -> dict[str, object]:
        """Describe paper constants, chosen appearance, and unused initialization action."""
        return _common_specification(self.config) | {
            "family": self.config.family,
            "parameter_graph": "diagonal",
            "action_dim": 5,
            "history_len": 2,
            "minimum_prediction_start": 1,
            "minimum_context_len": 3,
            "observation_shape": [3, 320, 64],
            "dt": 0.1,
            "gamma": 0.2,
            "stiffness_block": [[2.0, -1.0], [-1.0, 2.0]],
            "parameter_distribution": "independent Uniform(0.5,1.5) inverse masses",
            "action_distribution": "independent Uniform(-1,1), shared within each pair",
            "initialization": "z1=z0+dt*v0; sampled a0 unused; recurrence t=1..T-2",
            "valid_transition_rule": "source t>=1; a0 is an unused slot, never shifted",
            "position_gaussian": [self.config.position_mean, self.config.position_std],
            "velocity_gaussian": [self.config.velocity_mean, self.config.velocity_std],
            "observation_map": {
                "kind": "gaussian_panels",
                "native_shape": [3, 320, 64],
                "horizontal_center": "32+20*tanh(z)",
                "settings": asdict(self.config.renderer),
                "pixel_coordinates": "integer grid 0..63; centers stay floating point",
                "channels": [0, 1],
            },
        }

    def graphs(self) -> GraphMasks:
        """Return the fixed structural lag masks, independent of sampled derivatives."""
        return coupled_springs.graphs()

    def generate(self, episode_index: int) -> ScientificEpisode:
        """Simulate and render only this episode; the training item has no labels."""
        draws = sample_coupled_springs(self.config, episode_index)
        assert draws.initial_velocity is not None
        states = coupled_springs.simulate(
            draws.initial_position, draws.initial_velocity, draws.parameters, draws.actions
        )
        forces = torch.zeros(self.config.trajectory_length - 1, 10, dtype=torch.float64)
        forces[1:] = coupled_springs.forces(states[:-2], states[1:-1], draws.actions[1:])
        valid = torch.ones(draws.actions.shape[0], dtype=torch.bool)
        valid[0] = False
        dtype = torch.float32 if self.config.dtype == "float32" else torch.float64
        return ScientificEpisode(
            self.renderer.forward(states).to(dtype=dtype),
            draws.actions.to(dtype=dtype),
            episode_id(self.config.root_seed, self.config.split, episode_index),
            {
                "states": states,
                "parameters": draws.parameters,
                "masses": draws.parameters.reciprocal(),
                "initial_state": draws.initial_position,
                "initial_velocity": draws.initial_velocity,
                "actions": draws.actions,
                "forces": forces,
                "valid_transitions": valid,
            },
        )


@cache
def register_builtin_processes() -> None:
    """Register the two implementations lazily, keeping the registry lightweight."""
    from scjepa.data.factory import register_dgp

    register_dgp("controlled_vector", ControlledVectorDGP)
    register_dgp("coupled_springs", CoupledSpringsDGP)
