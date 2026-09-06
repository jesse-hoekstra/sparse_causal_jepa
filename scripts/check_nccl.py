"""Check local GPU communication independently of the experiment and its data.

Run with ``torchrun --standalone --nproc_per_node=2 scripts/check_nccl.py``.
Add ``--visual-ddp`` to check the dense Experiment 2 model's DDP constructor
instead of the scalar and 1 MiB collectives, without loading any data.
The process-group timeout helps diagnose failed collectives; use the shell's
``timeout`` command as well when a hard wall-clock limit is required.
"""

import argparse
import faulthandler
import os
from datetime import timedelta
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
from torch.cuda import nccl


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


def _check_visual_ddp(worker_rank: int, device: torch.device, timeout_seconds: int) -> None:
    """Exercise the benchmark model's real DDP initialization without a preload."""
    # Keep the basic communication test independent of model dependencies.
    from hydra import compose, initialize_config_dir
    from torch.nn.parallel import DistributedDataParallel

    from scjepa.training import seed_everything
    from scjepa.training.factory import build_model

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=bounce_visual_to_visual",
                "model.spartan_dense=true",
                "train.sparsity_enabled=false",
                "train.seed=0",
                "train.device=cuda",
                "wandb.enabled=false",
            ],
        )
    print(f"[rank {worker_rank}] constructing dense Experiment 2 model (seed=0)", flush=True)
    seed_everything(int(cfg.train.seed))
    model = build_model(cfg.model)
    parameters = list(model.parameters())
    numel = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    print(
        f"[rank {worker_rank}] model constructed: {len(parameters)} parameter tensors, "
        f"{numel:,} elements ({trainable:,} trainable)",
        flush=True,
    )
    # Match Trainer's rank-local reseeding and CUDA transfer before DDP.
    seed_everything(int(cfg.train.seed) + worker_rank * 1_000_003)
    print(f"[rank {worker_rank}] moving model to {device}", flush=True)
    model = model.to(device)
    torch.cuda.synchronize(device)
    print(f"[rank {worker_rank}] model on {device}; DDP constructor starting", flush=True)
    # Native communicator setup can stall before the process-group watchdog
    # becomes useful. Capture the Python call site for the failing DDP phase.
    faulthandler.dump_traceback_later(max(1, min(30, timeout_seconds // 2)), repeat=False)
    try:
        distributed_model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    finally:
        faulthandler.cancel_dump_traceback_later()
    torch.cuda.synchronize(device)
    print(f"[rank {worker_rank}] DDP constructor ready (no forward/backward test)", flush=True)
    del distributed_model


def main() -> None:
    """Bind local torchrun workers to separate GPUs and test NCCL collectives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout-seconds", type=int, default=60, help="Process-group timeout (default: 60)."
    )
    parser.add_argument(
        "--visual-ddp",
        action="store_true",
        help="Check the dense Experiment 2 model's DDP constructor without loading data.",
    )
    args = parser.parse_args()
    timeout_seconds = cast(int, args.timeout_seconds)
    visual_ddp = cast(bool, args.visual_ddp)
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

    print(
        f"[rank {worker_rank}] versions: torch={torch.__version__}, "
        f"CUDA={torch.version.cuda}, NCCL={nccl.version()}",
        flush=True,
    )
    for name in ("NCCL_P2P_DISABLE", "NCCL_CUMEM_ENABLE"):
        print(f"[rank {worker_rank}] {name}={os.environ.get(name, '(unset)')}", flush=True)
    device = torch.device("cuda", local_rank)
    print(f"[rank {worker_rank}] startup: binding to {device}", flush=True)
    torch.cuda.set_device(device)
    print(f"[rank {worker_rank}] init_process_group starting", flush=True)
    if visual_ddp:
        # Production initializes NCCL lazily: DDP encounters the first actual
        # collective after model construction and transfer. Do not warm it up.
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_seconds))
    else:
        dist.init_process_group(
            backend="nccl",
            device_id=device,
            timeout=timedelta(seconds=timeout_seconds),
        )
    print(f"[rank {worker_rank}] init_process_group complete", flush=True)
    if visual_ddp:
        _check_visual_ddp(worker_rank, device, timeout_seconds)
        result = "dense Experiment 2 DDP constructor (no data or training)"
    else:
        _check_tensor(1, "scalar", worker_rank, worker_count, device)
        _check_tensor(262_144, "1 MiB", worker_rank, worker_count, device)
        result = "scalar and 1 MiB broadcast/all_reduce"
    print(f"[rank {worker_rank}] destroying process group", flush=True)
    dist.destroy_process_group()
    print(f"[rank {worker_rank}] PASS: {result}", flush=True)


if __name__ == "__main__":
    main()
