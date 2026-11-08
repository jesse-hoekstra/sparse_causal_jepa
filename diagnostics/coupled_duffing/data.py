"""A second-order discrete mechanical system with separate input and label files."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from diagnostics.shared.io import read_json, save_torch, sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import write_json


@dataclass(frozen=True)
class DataConfig:
    """Fixed physical units, excitation and split sizes for the baseline pilot."""

    seed: int = 20261108
    train_groups: int = 4096
    calibration_groups: int = 512
    validation_groups: int = 512
    views: int = 4
    context: int = 34
    forecast: int = 8
    dt: float = 0.1
    linear_stiffness: float = 1.0
    coupling: float = 0.5
    force_scale: float = 6.0
    height: int = 16
    width: int = 32


def transition(
    previous: Tensor, current: Tensor, action: Tensor, theta: Tensor, config: DataConfig
) -> Tensor:
    """Advance positions with observed command in [-1,1] and theta=(beta1,d1,beta2,d2)."""
    beta, damping = theta[..., 0::2], theta[..., 1::2]
    h = config.dt
    return (
        (2 + h * damping - h * h * (config.linear_stiffness + config.coupling)) * current
        - previous
        - h * h * beta * current.pow(3)
        + h * h * config.coupling * current.flip(-1)
        + h * h * config.force_scale * action
    ) / (1 + h * damping)


def render(states: Tensor, config: DataConfig) -> Tensor:
    """Render two soft carts on fixed separate rails; appearance contains no parameters.

    Smooth horizontal compression keeps finite positions visible without clipping.
    Independent vertical profiles and horizontal Gaussian ratios identify centers
    in exact arithmetic; finite precision is the numerical observation approximation.
    """
    horizontal = torch.linspace(-1, 1, config.width, device=states.device)
    vertical = torch.linspace(-1, 1, config.height, device=states.device)
    centers = 0.85 * states / torch.sqrt(1 + states.square())
    cart_x = torch.exp(-0.5 * ((horizontal - centers[..., None]) / 0.09).square())
    tracks = torch.tensor([-0.5, 0.5], device=states.device)
    cart_y = torch.exp(-0.5 * ((vertical[None] - tracks[:, None]) / 0.12).square())
    carts = (cart_x[..., :, None, :] * cart_y[:, :, None]).sum(-3)
    rails = 0.08 * torch.exp(-0.5 * ((vertical[None] - tracks[:, None]) / 0.035).square())
    return (carts + rails.sum(0)[:, None]).unsqueeze(-3).float()


def generate(
    groups: int, split: int, config: DataConfig
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Draw independent systems and episodes; expose only rendered frames and commands."""
    generator = torch.Generator().manual_seed(config.seed + 100003 * split)
    theta = torch.rand(groups, 4, generator=generator, dtype=torch.float64)
    theta[:, 0::2] = 0.5 + 1.5 * theta[:, 0::2]
    theta[:, 1::2] = 0.2 + theta[:, 1::2]
    length = config.context + config.forecast
    shape = (groups, config.views)
    actions = 2 * torch.rand(*shape, length - 1, 1, generator=generator, dtype=torch.float64) - 1
    start = 2.4 * torch.rand(*shape, 2, generator=generator, dtype=torch.float64) - 1.2
    velocity = 2 * torch.rand(*shape, 2, generator=generator, dtype=torch.float64) - 1
    states = [start, start + config.dt * velocity]
    for time in range(1, length - 1):
        states.append(
            transition(states[-2], states[-1], actions[:, :, time], theta[:, None], config)
        )
    state = torch.stack(states, dim=2)
    if not bool(torch.isfinite(state).all()):
        raise FloatingPointError("the fixed mechanical protocol produced nonfinite states")
    observations = torch.cat(
        [render(state[start : start + 32].float(), config) for start in range(0, groups, 32)]
    )
    ids = torch.arange(groups, dtype=torch.int64) + split * 1000000
    return (
        {"observations": observations, "actions": actions.float(), "group_ids": ids},
        {"states": state, "theta": theta, "group_ids": ids},
    )


def prepare(directory: Path, config: DataConfig) -> dict[str, Any]:
    """Create train/calibration/validation caches; do not generate or read test trajectories."""
    if directory.exists():
        manifest = read_json(directory / "manifest.json")
        if manifest["config"] != asdict(config):
            raise ValueError("existing data settings differ")
        for name, digest in manifest["hashes"].items():
            if sha(directory / name) != digest:
                raise ValueError(f"data artifact changed: {name}")
        return manifest
    directory.mkdir(parents=True)
    hashes: dict[str, str] = {}
    health: dict[str, Any] = {}
    for split, (name, count) in enumerate(
        (
            ("train", config.train_groups),
            ("calibration", config.calibration_groups),
            ("validation", config.validation_groups),
        )
    ):
        inputs, labels = generate(count, split, config)
        for suffix, payload in (("inputs", inputs), ("labels", labels)):
            path = directory / f"{name}-{suffix}.pt"
            save_torch(path, payload)
            hashes[path.name] = sha(path)
        states = labels["states"].reshape(-1, 2)
        health[name] = {
            "max_abs_position": float(states.abs().max()),
            "state_std": [float(value) for value in states.std(0)],
            "state_correlation": [
                [float(value) for value in row] for row in torch.corrcoef(states.T)
            ],
        }
        print(f"prepared {name}: {count} systems", flush=True)
    record: dict[str, Any] = {
        "config": asdict(config),
        "hashes": hashes,
        "health": health,
        "reserved_test_split": 3,
        "test_generated": False,
    }
    record["identity"] = canonical_fingerprint(record)
    write_json(directory / "manifest.json", record)
    return record
