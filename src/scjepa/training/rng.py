"""Preserve host RNGs and the one CUDA device owned by this process."""

import random
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def preserve_rng(device: torch.device | None = None) -> Generator[None, None, None]:
    """Isolate auxiliary work without touching other ranks' visible CUDA devices.

    Host streams are always protected. If CUDA is already initialized, ``None``
    preserves the current CUDA device; an explicit CUDA device preserves only
    that device. An explicit CPU device does not access CUDA. This helper never
    initializes CUDA, enumerates devices, or seeds any random stream. Callers
    must keep their auxiliary tensor work on the selected device and the CPU.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()  # noqa: NPY002 - preserve callers' global stream
    torch_state = torch.get_rng_state()
    cuda_device: int | None = None
    if torch.cuda.is_initialized() and (device is None or device.type == "cuda"):
        cuda_device = (
            device.index
            if device is not None and device.index is not None  # pyright: ignore[reportUnnecessaryComparison]
            else torch.cuda.current_device()
        )
    cuda_state = torch.cuda.get_rng_state(cuda_device) if cuda_device is not None else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)  # noqa: NPY002
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            assert cuda_device is not None
            torch.cuda.set_rng_state(cuda_state, cuda_device)
