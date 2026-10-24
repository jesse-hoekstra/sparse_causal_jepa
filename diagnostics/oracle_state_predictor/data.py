"""Precompute addressed true-state forecasts for the standalone predictor control.

This diagnostic intentionally reads scientific references. It does not change
the ordinary observation-only dataset interface or train an observation encoder.
"""

from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from scjepa.data.datasets import canonical_fingerprint
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.types import DGPConfig

CONTEXT_LENGTH = 10
FORECAST_STEPS = 2
STATE_DIMENSION = 6
ACTION_DIMENSION = 16


@dataclass(frozen=True)
class OracleStateBatch:
    """One minibatch, with views flattened and raw theta repeated per view."""

    initial_states: Tensor
    """True z9 with shape (groups * views, 1, 6)."""
    actions: Tensor
    """Observed actions a9/a10 with shape (groups * views, 2, 16)."""
    targets: Tensor
    """True z10/z11 with shape (groups * views, 2, 6)."""
    parameters: Tensor
    """Provided raw theta with shape (groups * views, 6)."""
    group_ids: Tensor
    """Original independent-group identities, shape (groups,), on the CPU."""
    groups: int
    views: int


@dataclass(frozen=True)
class OracleStateData:
    """CPU float32 forecast tensors retaining independent parameter groups."""

    initial_states: Tensor
    actions: Tensor
    targets: Tensor
    parameters: Tensor
    episode_ids: Tensor
    config: DGPConfig
    source_fingerprint: str
    fingerprint: str

    def __len__(self) -> int:
        """Return independent parameter groups, rather than flattened views."""
        return self.config.num_episodes

    @property
    def views(self) -> int:
        """Return independently initialized episodes sharing each raw theta."""
        return self.config.episodes_per_parameter

    @property
    def metadata(self) -> dict[str, object]:
        """Describe the scientific source and explicit diagnostic input interface."""
        return {
            "config": asdict(self.config),
            "source_fingerprint": self.source_fingerprint,
            "fingerprint": self.fingerprint,
            "source_state_index": CONTEXT_LENGTH - 1,
            "action_indices": list(range(CONTEXT_LENGTH - 1, CONTEXT_LENGTH + 1)),
            "target_state_indices": list(range(CONTEXT_LENGTH, CONTEXT_LENGTH + FORECAST_STEPS)),
            "parameter_source": "provided raw ground-truth theta",
            "state_source": "provided raw ground-truth states; no observation encoder",
            "dtype": "float32",
        }

    def batch(self, indices: Tensor, device: torch.device | str) -> OracleStateBatch:
        """Select CPU groups, then transfer only this flattened minibatch to device."""
        if indices.ndim != 1 or indices.dtype != torch.int64 or indices.device.type != "cpu":
            raise ValueError("indices must be a one-dimensional CPU int64 tensor")
        if indices.numel() == 0:
            raise ValueError("a minibatch must contain at least one parameter group")
        if bool((indices < 0).any()) or bool((indices >= len(self)).any()):
            raise IndexError("parameter-group index outside the precomputed split")
        groups = indices.numel()
        parameters = self.parameters.index_select(0, indices)
        return OracleStateBatch(
            initial_states=self.initial_states.index_select(0, indices).flatten(0, 1).to(device),
            actions=self.actions.index_select(0, indices).flatten(0, 1).to(device),
            targets=self.targets.index_select(0, indices).flatten(0, 1).to(device),
            parameters=parameters.repeat_interleave(self.views, dim=0).to(device),
            group_ids=self.episode_ids.index_select(0, indices),
            groups=groups,
            views=self.views,
        )


def build_data(
    split: str,
    groups: int,
    *,
    seed: int = 0,
    views: int = 10,
    progress: bool = False,
) -> OracleStateData:
    """Generate K=10/T=12 source episodes once and retain their two forecast steps.

    The seed, split, and group index address the existing generator. Initial
    states and targets come from its scientific references, actions retain their
    observed mixed coordinates, and theta is provided without a transformation.
    All stored model inputs and targets are CPU float32, as in the experiment.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("diagnostic split must be train, validation, or test")
    if type(groups) is not int or groups < 1:
        raise ValueError("groups must be a positive integer")
    config = DGPConfig(
        family="simplified_gaussian",
        parameter_graph="complementary",
        root_seed=seed,
        split=split,
        num_episodes=groups,
        episodes_per_parameter=views,
        trajectory_length=CONTEXT_LENGTH + FORECAST_STEPS,
        dtype="float32",
        observation_seed=1729,
    )
    dataset = build_dgp_dataset(config)
    initial_states = torch.empty(groups, views, 1, STATE_DIMENSION, dtype=torch.float32)
    actions = torch.empty(groups, views, FORECAST_STEPS, ACTION_DIMENSION, dtype=torch.float32)
    targets = torch.empty(groups, views, FORECAST_STEPS, STATE_DIMENSION, dtype=torch.float32)
    parameters = torch.empty(groups, STATE_DIMENSION, dtype=torch.float32)
    episode_ids = torch.empty(groups, dtype=torch.int64)
    start = CONTEXT_LENGTH - 1
    for index in range(groups):
        episode = dataset.process.generate(index)
        states = episode.ground_truth["states"].reshape(views, config.trajectory_length, -1)
        observed_actions = episode.actions.reshape(views, config.trajectory_length - 1, -1)
        initial_states[index] = states[:, start : start + 1]
        actions[index] = observed_actions[:, start : start + FORECAST_STEPS]
        targets[index] = states[:, start + 1 : start + 1 + FORECAST_STEPS]
        parameters[index] = episode.ground_truth["parameters"]
        episode_ids[index] = episode.episode_id
        if progress and ((index + 1) % 1024 == 0 or index + 1 == groups):
            print(f"Prepared {split} oracle-state groups: {index + 1}/{groups}", flush=True)
    source_fingerprint = dataset.fingerprint
    fingerprint = canonical_fingerprint(
        {
            "schema": "scjepa.diagnostics.oracle-state-predictor.v1",
            "source_dataset_fingerprint": source_fingerprint,
            "source_state_index": start,
            "action_indices": list(range(start, start + FORECAST_STEPS)),
            "target_state_indices": list(range(start + 1, start + 1 + FORECAST_STEPS)),
            "state_source": "raw_ground_truth",
            "parameter_source": "raw_ground_truth",
            "action_source": "observed_mixed_coordinates",
            "dtype": "float32",
        }
    )
    return OracleStateData(
        initial_states,
        actions,
        targets,
        parameters,
        episode_ids,
        config,
        source_fingerprint,
        fingerprint,
    )
