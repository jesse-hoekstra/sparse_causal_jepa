"""Parallel one-off generation of the bounce TRAIN split to a preload file.

The bounce sim is single-threaded pure python (~0.27 s/episode on a Grace
core; 7.6h for 100k episodes, measured 2026-07-19) and the pipeline pays it
once per stage while the GPU idles. Episodes are deterministic per
(seed, index), so generation is embarrassingly parallel and cacheable:

    python scripts/pregenerate_bounce.py experiment=bounce_baumgartner

writes ``data/bounce_train_v<simulator>_<num_clips>.pt`` for seed 0, or a
``_seed<N>`` suffixed file for other seeds (~630MB at 100k; data/ is
gitignored) unless it already exists. Point the runs at it with
``data.preload=data/bounce_train_v<simulator>_<num_clips>[_seed<N>].pt`` — the SAME hydra overrides
must go to every reference and main run (D12); the file embeds its generation
settings and BounceDataset refuses a mismatch. Eval splits (seed offset)
always generate on the fly — the factory never applies ``preload`` to them.

Workers: $SLURM_CPUS_PER_TASK if set, else all cores.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scjepa.data import BounceDataset


def _dataset_kwargs(cfg: DictConfig) -> dict[str, object]:
    """The exact constructor args build_dataset uses for the train split."""
    data = cfg.data
    return {
        "num_episodes": data.num_clips,
        "clip_len": data.clip_len,
        "num_balls": data.num_balls,
        "resolution": data.resolution,
        "radius": data.radius,
        "mass_range": tuple(data.mass_range),
        "mass_normal": tuple(data.mass_normal) if data.get("mass_normal") is not None else None,
        "radius_from_mass": bool(data.get("radius_from_mass", False)),
        "speed": data.speed,
        "seed": data.seed,
        "render": False,
        "cache": False,
    }


def _generate_shard(args: tuple[dict[str, object], int, int]) -> dict[str, torch.Tensor]:
    """Simulate episodes [lo, hi) and return their stacked tensors."""
    kwargs, lo, hi = args
    dataset = BounceDataset(**kwargs)  # pyright: ignore[reportArgumentType]
    items = [dataset[i] for i in range(lo, hi)]
    keys = ("states", "params", "contacts")
    return {key: torch.stack([item[key] for item in items]) for key in keys}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate the train split in parallel and save it with its identity."""
    kwargs = _dataset_kwargs(cfg)
    num = int(kwargs["num_episodes"])  # pyright: ignore[reportArgumentType]
    meta = BounceDataset(**kwargs).generation_meta()  # pyright: ignore[reportArgumentType]
    simulator_version = int(meta["simulator_version"])  # pyright: ignore[reportArgumentType]
    seed = int(kwargs["seed"])  # pyright: ignore[reportArgumentType]
    seed_suffix = "" if seed == 0 else f"_seed{seed}"
    out = (
        Path(hydra.utils.get_original_cwd())
        / "data"
        / f"bounce_train_v{simulator_version}_{num}{seed_suffix}.pt"
    )
    if out.exists():
        try:
            BounceDataset(**kwargs, preload=str(out))  # pyright: ignore[reportArgumentType]
        except ValueError as error:
            raise RuntimeError(
                f"existing preload {out} does not match this run; move or delete it, "
                "then regenerate"
            ) from error
        print(f"{out} already exists and matches — nothing to do")
        return
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    bound = max(1, num // (workers * 4))  # small shards: even load, bounded worker RAM
    shards = [(kwargs, lo, min(lo + bound, num)) for lo in range(0, num, bound)]
    print(f"generating {num} episodes in {len(shards)} shards on {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_generate_shard, shards))
    tensors = {
        key: torch.cat([shard[key] for shard in results])
        for key in ("states", "params", "contacts")
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "tensors": tensors}, out)
    size = out.stat().st_size / 1e6
    shapes = ", ".join(f"{k} {tuple(v.shape)}" for k, v in tensors.items())
    print(f"wrote {out} ({size:.0f} MB): {shapes}")


if __name__ == "__main__":
    main()
