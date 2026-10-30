"""Resumable group sampling and explicit accelerator selection."""

from typing import Any

import torch
from torch import Tensor


class GroupSampler:
    """Shuffle independent parameter groups without replacement; drop epoch remainders."""

    def __init__(self, groups: int, batch_size: int, seed: int) -> None:
        """Own the complete sampling stream so checkpoints resume its exact position."""
        self.groups, self.batch_size = groups, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.empty(0, dtype=torch.long)
        self.offset = 0

    def next(self) -> Tensor:
        """Return a full batch of distinct groups."""
        if self.offset + self.batch_size > len(self.order):
            self.order = torch.randperm(self.groups, generator=self.generator)
            self.offset = 0
        result = self.order[self.offset : self.offset + self.batch_size]
        self.offset += self.batch_size
        return result

    def state_dict(self) -> dict[str, object]:
        """Serialize the sampler without consuming its random stream."""
        return {"rng": self.generator.get_state(), "order": self.order, "offset": self.offset}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the next group batch from a trusted local checkpoint."""
        self.generator.set_state(state["rng"])
        self.order, self.offset = state["order"], state["offset"]


def resolve_device(requested: str) -> torch.device:
    """Resolve bare CUDA to the current visible GPU before selecting it."""
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        # Torch's stub omits None, but torch.device("cuda").index is None at runtime.
        if device.index is None:  # pyright: ignore[reportUnnecessaryComparison]
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    return device
