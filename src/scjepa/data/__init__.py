"""Training-safe tensor episodes; scientific generators live in separate data modules."""

from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import Dataset

from scjepa.data.identity import training_tensors_fingerprint


class TensorEpisodes(Dataset[dict[str, Tensor]]):
    """Adapt tensors to episodes without simulation, rendering, or scientific labels.

    Observations are ``(episodes, time, *observation_shape)``. Actions, when
    present, are ``(episodes, time - 1, action_dim)``: action t generates frame
    t+1. Each item includes a stable integer episode ID for regularization.
    Multiple clips from one episode must share its ID.
    """

    def __init__(
        self, observations: Tensor, actions: Tensor | None = None, episode_ids: Tensor | None = None
    ) -> None:
        """Validate sequence alignment and retain the supplied tensors."""
        if observations.ndim not in (3, 5) or min(observations.shape[:2]) < 1:
            raise ValueError("observations must be nonempty (E,T,D) or (E,T,C,H,W)")
        if observations.shape[1] < 2 or not observations.is_floating_point():
            raise ValueError("observations require at least two frames and floating point values")
        if actions is not None and (
            actions.ndim != 3
            or actions.shape[:2] != (observations.shape[0], observations.shape[1] - 1)
            or not actions.is_floating_point()
        ):
            raise ValueError("actions must be floating point (E,T-1,da)")
        if episode_ids is None:
            episode_ids = torch.arange(observations.shape[0], dtype=torch.int64)
        if episode_ids.shape != (observations.shape[0],) or episode_ids.dtype != torch.int64:
            raise ValueError("episode_ids must be int64 with shape (E,)")
        self.observations = observations
        self.actions = actions
        self.episode_ids = episode_ids

    @property
    def fingerprint(self) -> str:
        """Content identity for exact resume, independent of the source file path."""
        tensors = {"observations": self.observations, "episode_ids": self.episode_ids}
        if self.actions is not None:
            tensors["actions"] = self.actions
        return training_tensors_fingerprint(tensors)

    def __len__(self) -> int:
        """Return the number of supplied episodes or clips."""
        return self.observations.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return one sequence with its stable identity."""
        batch = {"observations": self.observations[index], "episode_ids": self.episode_ids[index]}
        if self.actions is not None:
            batch["actions"] = self.actions[index]
        return batch

    @classmethod
    def load(cls, path: str | Path) -> "TensorEpisodes":
        """Read a tensor-only mapping saved with torch.save; no object unpickling."""
        value: object = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, dict):
            raise ValueError("episode file must contain a tensor mapping")
        mapping = cast(dict[str, object], value)
        if set(mapping) - {"observations", "actions", "episode_ids"}:
            raise ValueError("episode file accepts observations, actions, and episode_ids only")
        if "observations" not in mapping or any(
            not isinstance(v, Tensor) for v in mapping.values()
        ):
            raise ValueError("episode file requires tensor observations and tensor optional fields")
        tensors = cast(dict[str, Tensor], mapping)
        return cls(tensors["observations"], tensors.get("actions"), tensors.get("episode_ids"))


__all__ = ["TensorEpisodes"]
