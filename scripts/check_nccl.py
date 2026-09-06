"""Check local GPU communication independently of the experiment and its data.

Run with ``torchrun --standalone --nproc_per_node=2 scripts/check_nccl.py``.
The process-group timeout helps diagnose failed collectives; use the shell's
``timeout`` command as well when a hard wall-clock limit is required.
"""

import argparse
import os
from datetime import timedelta
from typing import cast

import torch
import torch.distributed as dist


def _check_tensor(
    elements: int,
    label: str,
    worker_rank: int,
    worker_count: int,
    device: torch.device,
) -> None:
    """Check exact broadcast and sum results for one tensor size."""
    value = torch.full((elements,), float(worker_rank + 1), device=device)
    print(f"[rank {worker_rank}] {label}: broadcast starting", flush=True)
    dist.broadcast(value, src=0)  # pyright: ignore[reportUnknownMemberType]
    if not bool(torch.all(value == 1.0)):
        raise RuntimeError(f"rank {worker_rank}: {label} broadcast returned incorrect values")
    print(f"[rank {worker_rank}] {label}: broadcast complete; all_reduce starting", flush=True)
    value.fill_(float(worker_rank + 1))
    dist.all_reduce(value, op=dist.ReduceOp.SUM)  # pyright: ignore[reportUnknownMemberType]
    expected = worker_count * (worker_count + 1) // 2
    if not bool(torch.all(value == expected)):
        raise RuntimeError(f"rank {worker_rank}: {label} all_reduce returned incorrect values")
    print(f"[rank {worker_rank}] {label}: all_reduce complete (sum={expected})", flush=True)


def main() -> None:
    """Bind local torchrun workers to separate GPUs and test NCCL collectives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout-seconds", type=int, default=60, help="Process-group timeout (default: 60)."
    )
    timeout_seconds = cast(int, parser.parse_args().timeout_seconds)
    if timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    worker_count = int(os.environ.get("WORLD_SIZE", "1"))
    local_count = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if worker_count < 2 or local_count != worker_count:
        parser.error("launch at least two workers on one machine with torchrun --standalone")
    worker_rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if not 0 <= worker_rank < worker_count or local_rank != worker_rank:
        parser.error("RANK and LOCAL_RANK must identify this single-machine torchrun worker")
    if not torch.cuda.is_available() or torch.cuda.device_count() < local_count:
        parser.error(f"need at least {local_count} visible CUDA GPUs; check CUDA_VISIBLE_DEVICES")
    if not dist.is_available() or not dist.is_nccl_available():
        parser.error("this PyTorch installation does not provide NCCL")

    device = torch.device("cuda", local_rank)
    print(f"[rank {worker_rank}] startup: binding to {device}", flush=True)
    torch.cuda.set_device(device)
    print(f"[rank {worker_rank}] init_process_group starting", flush=True)
    dist.init_process_group(
        backend="nccl",
        device_id=device,
        timeout=timedelta(seconds=timeout_seconds),
    )
    print(f"[rank {worker_rank}] init_process_group complete", flush=True)
    _check_tensor(1, "scalar", worker_rank, worker_count, device)
    _check_tensor(262_144, "1 MiB", worker_rank, worker_count, device)
    print(f"[rank {worker_rank}] destroying process group", flush=True)
    dist.destroy_process_group()
    print(f"[rank {worker_rank}] PASS: scalar and 1 MiB broadcast/all_reduce", flush=True)


if __name__ == "__main__":
    main()
