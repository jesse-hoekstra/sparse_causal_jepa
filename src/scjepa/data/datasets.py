"""Lazy deterministic episodes and training-only windows with explicit truth access."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from typing import Protocol, cast

import torch
from torch import Tensor
from torch.utils.data import Dataset

from scjepa.data.types import DGP, DGPConfig, GraphMasks


def canonical_fingerprint(value: object) -> str:
    """Hash a JSON descriptor using fixed ordering and no nonfinite numbers."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def tensor_fingerprint(value: Tensor) -> str:
    """Hash tensor values, shape and dtype independently of torch serialization."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def graph_descriptor(graphs: GraphMasks) -> dict[str, object]:
    """Serialize output-row structural masks, preserving their oldest-first lags."""
    return {
        "state": [[[bool(value) for value in lag] for lag in row] for row in graphs.state],
        "action": [[[bool(value) for value in lag] for lag in row] for row in graphs.action],
        "parameter": [[bool(value) for value in row] for row in graphs.parameter],
    }


def _process_descriptor(process: DGP) -> dict[str, object]:
    matrix = process.observation_matrix
    return {
        "schema": "scjepa.dgp.v1",
        "config": asdict(process.config),
        "specification": process.specification(),
        "graphs": graph_descriptor(process.graphs()),
        "observation_matrix_fingerprint": tensor_fingerprint(matrix)
        if matrix is not None
        else None,
    }


class EpisodeDataset(Protocol):
    """Training episode retrieval plus explicitly separate scientific metadata."""

    observation_shape: tuple[int, ...]
    action_dim: int
    history_len: int
    minimum_prediction_start: int
    trajectory_length: int
    specification: dict[str, object]

    @property
    def fingerprint(self) -> str:
        """Return the validated scientific definition identity."""
        ...

    def __len__(self) -> int:
        """Return the number of episodes or windows."""
        ...

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return observations, actions and original episode identity only."""
        ...

    def ground_truth(self, index: int) -> dict[str, Tensor]:
        """Explicitly request scientific references for one full episode."""
        ...


class GeneratedEpisodes(Dataset[dict[str, Tensor]]):
    """Generate each episode on demand without retaining observations in RAM."""

    def __init__(self, process: DGP) -> None:
        """Bind an immutable process definition and its reproducibility identity."""
        self.process = process
        self.config: DGPConfig = process.config
        self.specification = process.specification()
        graphs = process.graphs()
        self.graphs = GraphMasks(
            graphs.state.clone(), graphs.action.clone(), graphs.parameter.clone()
        )
        matrix = process.observation_matrix
        self._observation_matrix = None if matrix is None else matrix.detach().clone()
        self.descriptor = _process_descriptor(process)
        self._fingerprint = canonical_fingerprint(self.descriptor)
        shape = self.specification["observation_shape"]
        if not isinstance(shape, list | tuple):
            raise ValueError("process observation_shape must be a sequence of integers")
        dimensions = cast(list[object] | tuple[object, ...], shape)
        if not all(isinstance(v, int) for v in dimensions):
            raise ValueError("process observation_shape must be a sequence of integers")
        self.observation_shape = tuple(cast(int, value) for value in dimensions)
        self.action_dim = self._integer_specification("action_dim")
        self.history_len = self._integer_specification("history_len")
        self.minimum_prediction_start = self._integer_specification("minimum_prediction_start")
        self.trajectory_length = self.config.trajectory_length

    @property
    def fingerprint(self) -> str:
        """Reject changed generation settings before declaring checkpoint identity."""
        self._validate_definition()
        return self._fingerprint

    def _validate_definition(self) -> None:
        if (
            self.config != self.process.config
            or canonical_fingerprint(self.descriptor) != self._fingerprint
            or canonical_fingerprint(_process_descriptor(self.process)) != self._fingerprint
        ):
            raise ValueError("dataset definition changed after construction; build a new dataset")

    @property
    def observation_matrix(self) -> Tensor | None:
        """Return a defensive copy of the realized map used by this dataset."""
        return None if self._observation_matrix is None else self._observation_matrix.clone()

    def _integer_specification(self, key: str) -> int:
        value = self.specification[key]
        if not isinstance(value, int):
            raise ValueError(f"process {key} must be an integer")
        return value

    def __len__(self) -> int:
        """Return the configured count without generating any data."""
        return self.config.num_episodes

    def _index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("episode index outside dataset")
        return index

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return only the model's observation/action/identity interface."""
        self._validate_definition()
        episode = self.process.generate(self._index(index))
        return {
            "observations": episode.observations,
            "actions": episode.actions,
            "episode_ids": torch.tensor(episode.episode_id, dtype=torch.int64),
        }

    def ground_truth(self, index: int) -> dict[str, Tensor]:
        """Generate references through a separate, evaluation-only method."""
        self._validate_definition()
        return self.process.generate(self._index(index)).ground_truth


class EpisodeWindows(Dataset[dict[str, Tensor]]):
    """Contiguous observation/action windows retaining each original episode ID.

    No scientific labels enter ``__getitem__``. ``original_ground_truth`` returns
    the complete original episode; ``window_bounds`` gives its half-open frame
    bounds, avoiding ambiguous slicing of constant parameters or temporal labels.
    """

    def __init__(self, dataset: EpisodeDataset, window_length: int, stride: int = 1) -> None:
        """Enumerate regular windows arithmetically without caching episodes."""
        if not 2 <= window_length <= dataset.trajectory_length or stride < 1:
            raise ValueError("window_length must be in [2,T] and stride must be positive")
        self.dataset = dataset
        self.trajectory_length = window_length
        self.stride = stride
        self.windows_per_episode = 1 + (dataset.trajectory_length - window_length) // stride
        self.observation_shape = dataset.observation_shape
        self.action_dim = dataset.action_dim
        self.history_len = dataset.history_len
        self.minimum_prediction_start = dataset.minimum_prediction_start
        self._source_fingerprint = dataset.fingerprint
        self.specification = deepcopy(dataset.specification) | {
            "trajectory_length": window_length,
            "window_stride": stride,
            "source_dataset_fingerprint": dataset.fingerprint,
        }
        self.fingerprint = canonical_fingerprint(
            {
                "schema": "scjepa.windows.v1",
                "dataset": dataset.fingerprint,
                "source_content_fingerprint": getattr(dataset, "content_fingerprint", None),
                "window_length": window_length,
                "stride": stride,
            }
        )

    @property
    def content_fingerprint(self) -> str | None:
        """Propagate live source integrity checks to checkpointed window adapters."""
        if self.dataset.fingerprint != self._source_fingerprint:
            raise ValueError("window source dataset definition changed after construction")
        value: object = getattr(self.dataset, "content_fingerprint", None)
        return value if isinstance(value, str) else None

    def __len__(self) -> int:
        """Return the number of regularly spaced windows across all episodes."""
        return len(self.dataset) * self.windows_per_episode

    def window_bounds(self, index: int) -> tuple[int, int, int]:
        """Return original episode index and half-open observation frame bounds."""
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("window index outside dataset")
        episode_index, window_index = divmod(index, self.windows_per_episode)
        start = window_index * self.stride
        return episode_index, start, start + self.trajectory_length

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Slice frame t with its original action t, retaining original identity."""
        episode_index, start, end = self.window_bounds(index)
        episode = self.dataset[episode_index]
        time_axis = episode["observations"].ndim - len(self.observation_shape) - 1
        return {
            "observations": episode["observations"].narrow(time_axis, start, end - start),
            "actions": episode["actions"].narrow(time_axis, start, end - start - 1),
            "episode_ids": episode["episode_ids"],
        }

    def original_ground_truth(self, index: int) -> dict[str, Tensor]:
        """Return the full original episode's labels, not automatically sliced labels."""
        episode_index, _, _ = self.window_bounds(index)
        return self.dataset.ground_truth(episode_index)
