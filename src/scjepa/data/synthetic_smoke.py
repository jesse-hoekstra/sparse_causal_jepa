"""Random-clip dataset for smoke tests ONLY — no physics, no structure.

Exists so the training loop can be exercised end-to-end on CPU before the real
datasets (CLEVRER, Push-T, synthetic ground-truth systems) land. Batch contract
matches the real pipelines: ``{"frames": (Th+1, C, H, W)}`` per item, stacked
by the DataLoader to ``(B, Th+1, C, H, W)``.
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset


class RandomClipDataset(Dataset[dict[str, Tensor]]):
    """Fixed random video clips (deterministic per seed + index)."""

    def __init__(
        self,
        num_clips: int = 16,
        clip_len: int = 4,
        channels: int = 3,
        resolution: int = 64,
        seed: int = 0,
    ) -> None:
        """Pre-generate ``num_clips`` random clips of ``clip_len`` frames."""
        generator = torch.Generator().manual_seed(seed)
        self._clips = torch.randn(
            num_clips, clip_len, channels, resolution, resolution, generator=generator
        )

    def __len__(self) -> int:
        """Number of clips."""
        return self._clips.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return one clip under the shared batch contract."""
        return {"frames": self._clips[index]}


__all__ = ["RandomClipDataset"]
