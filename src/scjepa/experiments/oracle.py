"""Explicit known-parameter control, separate from ordinary label-free datasets."""

import torch
from torch import Tensor

from scjepa.data.datasets import GeneratedEpisodes, canonical_fingerprint


class OracleParameterEpisodes(GeneratedEpisodes):
    """Expose only raw system parameters as additional inputs in the oracle control.

    States, graphs, inverse observations and other scientific references remain
    excluded. The underlying addressed draws and observations are unchanged.
    """

    def __init__(self, source: GeneratedEpisodes) -> None:
        """Bind the same process while recording a distinct training interface."""
        if source.config.family != "simplified_gaussian":
            raise ValueError("oracle parameters currently require simplified_gaussian")
        super().__init__(source.process)

    @property
    def fingerprint(self) -> str:
        """Distinguish supplied-parameter training from the label-free interface."""
        return canonical_fingerprint(
            {
                "schema": "scjepa.oracle-parameter-episodes.v1",
                "dataset": super().fingerprint,
                "parameter_source": "raw_ground_truth",
            }
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Generate once and return observations, actions, identity and raw theta."""
        self._validate_definition()
        episode = self.process.generate(self._index(index))
        return {
            "observations": episode.observations,
            "actions": episode.actions,
            "episode_ids": torch.tensor(episode.episode_id, dtype=torch.int64),
            "oracle_parameters": episode.ground_truth["parameters"]
            .detach()
            .to(episode.observations),
        }
