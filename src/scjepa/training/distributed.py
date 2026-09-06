"""Small torchrun helpers that preserve the experiment's total batch size.

Gradient synchronization belongs to DistributedDataParallel. These reductions
are detached statistics for GECO, logging, and all-or-nothing update decisions;
they must never replace DDP's differentiable local loss.
"""

import os
from collections.abc import Iterator
from datetime import timedelta
from typing import cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import Sampler


def world_size() -> int:
    """Return the active process count, or one outside a process group."""
    return dist.get_world_size() if dist.is_initialized() else 1


def rank() -> int:
    """Return the active global rank, or zero for ordinary Python execution."""
    return dist.get_rank() if dist.is_initialized() else 0


def is_main_process() -> bool:
    """Whether this process owns logging, evaluation, and checkpoint writes."""
    return rank() == 0


def initialize_distributed(device: str | torch.device) -> torch.device:
    """Initialize from torchrun's environment and select this rank's device.

    Normal single-process execution creates no process group. CPU workers use
    Gloo; CUDA workers use NCCL and ``LOCAL_RANK`` regardless of a generic
    ``cuda``/``cuda:0`` configuration shared by all workers.
    """
    requested = torch.device(device)
    processes = int(os.environ.get("WORLD_SIZE", "1"))
    if processes < 1:
        raise ValueError("WORLD_SIZE must be positive")
    if processes > 1 and requested.type not in {"cpu", "cuda"}:
        raise ValueError("distributed training requires CPU or CUDA devices")
    if requested.type == "cuda":
        if processes > 1:
            requested = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        elif requested.index is None:  # pyright: ignore[reportUnnecessaryComparison]
            requested = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(requested)
    if processes > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if requested.type == "cuda" else "gloo",
            # Other workers wait while rank zero runs the visual evaluation.
            timeout=timedelta(hours=4),
        )
    return requested


def distributed_barrier() -> None:
    """Wait for rank-zero evaluation/checkpoint work, if running distributed."""
    if world_size() > 1:
        dist.barrier()  # pyright: ignore[reportUnknownMemberType]


def destroy_distributed() -> None:
    """Release an initialized process group without requiring another barrier."""
    if dist.is_initialized():
        dist.destroy_process_group()


def mean_tensor(value: Tensor) -> Tensor:
    """Return the detached elementwise rank mean (equal local batch sizes)."""
    result = value.detach().clone()
    if world_size() > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)  # pyright: ignore[reportUnknownMemberType]
        result /= world_size()
    return result


def any_rank(value: bool | Tensor, device: str | torch.device) -> bool:
    """Reduce a rejection flag before making one device-to-host decision.

    Device boolean tensors avoid copying a flag to the CPU and immediately
    copying it back to the GPU for the collective. Python flags remain valid.
    """
    if world_size() == 1:
        return bool(value)
    result = torch.as_tensor(value, dtype=torch.int32, device=device).clone()
    dist.all_reduce(result, op=dist.ReduceOp.MAX)  # pyright: ignore[reportUnknownMemberType]
    return bool(result.item())


def mean_metrics(metrics: dict[str, float], device: str | torch.device) -> dict[str, float]:
    """Average matching scalar dictionaries; every rank must supply the same keys.

    Callers compute nonlinear representation diagnostics from gathered states
    before this reduction. Representation variance is monitoring only.
    """
    if world_size() == 1 or not metrics:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], dtype=torch.float64, device=device)
    reduced = cast(list[float], mean_tensor(values).tolist())  # pyright: ignore[reportUnknownMemberType]
    return {key: float(value) for key, value in zip(keys, reduced, strict=True)}


def gather_batch_tensor(value: Tensor) -> Tensor:
    """Concatenate detached equal-shaped local tensors along the episode axis."""
    if world_size() == 1:
        return value.detach()
    local = value.detach().contiguous()
    gathered = [torch.empty_like(local) for _ in range(world_size())]
    dist.all_gather(gathered, local)  # pyright: ignore[reportUnknownMemberType]
    return torch.cat(gathered, dim=0)


def gather_rank_objects(value: object) -> list[object]:
    """Gather trusted rank-local checkpoint metadata in global rank order."""
    if world_size() == 1:
        return [value]
    gathered: list[object] = [None] * world_size()
    dist.all_gather_object(gathered, value)  # pyright: ignore[reportUnknownMemberType]
    return gathered


class GlobalBatchSampler(Sampler[list[int]]):
    """Shard each deterministic global batch evenly, without padding episodes.

    ``batch_size`` is the TOTAL batch across GPUs. Concatenating each rank's
    local batch in rank order recovers the single-process shuffled batch.
    Incomplete final global batches are dropped consistently on every rank.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        epoch: int,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        """Validate an equal shard and freeze the epoch's global shuffle seed."""
        self.rank = rank if rank is not None else dist.get_rank() if dist.is_initialized() else 0
        self.world_size = (
            world_size
            if world_size is not None
            else (dist.get_world_size() if dist.is_initialized() else 1)
        )
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must satisfy 0 <= rank < a positive world_size")
        if batch_size < 1 or batch_size % self.world_size:
            raise ValueError("total batch_size must be positive and divisible by world_size")
        if dataset_size < 0 or epoch < 0:
            raise ValueError("dataset_size and epoch must be non-negative")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.local_batch_size = batch_size // self.world_size
        self.seed = seed
        self.epoch = epoch

    def __len__(self) -> int:
        """Number of full global batches in this epoch."""
        return self.dataset_size // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        """Yield this rank's slice in the legacy DataLoader's shuffle order."""
        generator = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        # The original DataLoader consumes one worker-base-seed draw before its
        # RandomSampler calls randperm. Preserve that exact episode order.
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        indices = cast(
            list[int],
            torch.randperm(self.dataset_size, generator=generator).tolist(),  # pyright: ignore[reportUnknownMemberType]
        )
        rank_offset = self.rank * self.local_batch_size
        for batch_index in range(len(self)):
            start = batch_index * self.batch_size + rank_offset
            yield indices[start : start + self.local_batch_size]
