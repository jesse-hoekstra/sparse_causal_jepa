"""Matched physical trajectories for acquisition diagnostics, with theta stored separately."""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.coupled_duffing.data import DataConfig, transition
from diagnostics.shared.io import read_json, save_torch, sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import write_json


@dataclass(frozen=True)
class ProbeDataConfig:
    """Share physical systems, initial histories and forecast boundaries across conditions."""

    seed: int = 20261109
    train_groups: int = 8192
    calibration_groups: int = 512
    validation_groups: int = 256
    views: int = 10
    context: int = 60
    forecast: int = 40
    forcing: str = "iid"

    def __post_init__(self) -> None:
        """Reject invalid sizes and undeclared excitation profiles."""
        if self.forcing not in {"iid", "driven"}:
            raise ValueError("forcing must be iid or driven")
        if min(self.train_groups, self.calibration_groups, self.validation_groups, self.views) < 2:
            raise ValueError("need at least two systems and episodes")
        if self.context < 3 or self.forecast < 1:
            raise ValueError("invalid trajectory lengths")


def draw(groups: int, split: int, config: ProbeDataConfig) -> tuple[dict[str, Tensor], Tensor]:
    """Generate reference-position observations; parameters never join the learner inputs."""
    if split not in {0, 1, 2}:
        raise ValueError("diagnostic does not generate the reserved test split")
    base = config.seed + 100003 * split
    physical = torch.Generator().manual_seed(base)
    forces = torch.Generator().manual_seed(base + 10000019)
    phases = torch.Generator().manual_seed(base + 20000033)
    theta = torch.rand(groups, 4, generator=physical, dtype=torch.float64)
    theta[:, 0::2] = 0.5 + 1.5 * theta[:, 0::2]
    theta[:, 1::2] = 0.2 + theta[:, 1::2]
    shape = (groups, config.views)
    first = 2.4 * torch.rand(*shape, 2, generator=physical, dtype=torch.float64) - 1.2
    velocity = 2 * torch.rand(*shape, 2, generator=physical, dtype=torch.float64) - 1
    length = config.context + config.forecast
    actions = 2 * torch.rand(*shape, length - 1, 1, generator=forces, dtype=torch.float64) - 1
    if config.forcing == "driven":
        # Nominal modal frequencies 1 and sqrt(2), plus their simple upper harmonic 2.
        # Independent continuous dither preserves an open set of action sequences.
        frequency = torch.tensor([1.0, math.sqrt(2.0), 2.0], dtype=torch.float64)
        phase = 2 * math.pi * torch.rand(*shape, 1, 3, generator=phases, dtype=torch.float64)
        clock = 0.1 * torch.arange(length - 1, dtype=torch.float64)
        waves = (clock[None, None, :, None] * frequency + phase).sin().mean(-1, keepdim=True)
        envelope = ((torch.arange(length - 1) // 20) % 2 == 0).double()[None, None, :, None]
        actions = 0.8 * envelope * waves + 0.2 * actions
    states = [first, first + 0.1 * velocity]
    for time in range(1, length - 1):
        states.append(
            transition(states[-2], states[-1], actions[:, :, time], theta[:, None], DataConfig())
        )
    observations = torch.stack(states, dim=2).float()
    if not bool(torch.isfinite(observations).all()):
        raise FloatingPointError("physical screen produced nonfinite trajectories")
    return {"observations": observations, "actions": actions.float()}, theta


def prepare(directory: Path, config: ProbeDataConfig) -> dict[str, Any]:
    """Persist separate inputs and parameter labels; reference states are a declared control."""
    if directory.exists():
        record = read_json(directory / "manifest.json")
        if record["config"] != asdict(config):
            raise ValueError("existing data settings differ")
        for name, digest in record["hashes"].items():
            if sha(directory / name) != digest:
                raise ValueError(f"changed data artifact: {name}")
        return record
    directory.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for split, (name, groups) in enumerate(
        (
            ("train", config.train_groups),
            ("calibration", config.calibration_groups),
            ("validation", config.validation_groups),
        )
    ):
        inputs, theta = draw(groups, split, config)
        for suffix, payload in (("inputs", inputs), ("labels", {"theta": theta})):
            path = directory / f"{name}-{suffix}.pt"
            save_torch(path, payload)
            hashes[path.name] = sha(path)
    record = {
        "schema": "duffing-physical-acquisition-data-v1",
        "config": asdict(config),
        "hashes": hashes,
        "oracle_state_diagnostic": True,
        "test_generated": False,
    }
    record["identity"] = canonical_fingerprint(record)
    write_json(directory / "manifest.json", record)
    return record


def window(
    inputs: dict[str, Tensor], boundary: int, context: int, horizon: int
) -> dict[str, Tensor]:
    """Change available history/horizon while keeping the forecast's physical start fixed."""
    if set(inputs) != {"observations", "actions"}:
        raise ValueError("acquisition inputs must exclude physical parameter labels")
    if (
        not 3 <= context <= boundary
        or not 1 <= horizon <= inputs["observations"].shape[2] - boundary
    ):
        raise ValueError("requested context/horizon is outside the saved trajectory")
    start, end = boundary - context, boundary + horizon
    return {
        "observations": inputs["observations"][:, :, start:end],
        "actions": inputs["actions"][:, :, start : end - 1],
    }


@torch.no_grad()
def reference_screen(config: ProbeDataConfig, groups: int = 512) -> dict[str, Any]:
    """Measure each parameter's forecast influence using the known physical recurrence."""
    inputs, theta = draw(groups, 0, config)
    states, actions = inputs["observations"].double(), inputs["actions"].double()
    means = theta.new_tensor([1.25, 0.7, 1.25, 0.7])
    variants = {"all_mean": means[None].expand_as(theta), "true": theta}
    for index in range(4):
        replacement = theta.clone()
        replacement[:, index] = means[index]
        variants[f"parameter_{index + 1}_mean"] = replacement
    truth = states[:, :, config.context :]
    variance = truth.flatten(0, 2).var(0, correction=0)
    scores: dict[str, Any] = {}
    for name, parameters in variants.items():
        previous, current = states[:, :, config.context - 2], states[:, :, config.context - 1]
        predictions: list[Tensor] = []
        for time in range(config.context - 1, states.shape[2] - 1):
            nxt = transition(
                previous, current, actions[:, :, time], parameters[:, None], DataConfig()
            )
            predictions.append(nxt)
            previous, current = current, nxt
        prediction = torch.stack(predictions, 2)
        mse = (prediction - truth).square().mean((0, 1, 2))
        scores[name] = {
            "coordinate_nmse": [float(v) for v in mse / variance],
            "mean_r2": float((1 - mse / variance).mean()),
        }
    return {
        "config": asdict(config),
        "groups": groups,
        "reference_only": True,
        "max_abs_position": float(states.abs().max()),
        "scores": scores,
    }


def main() -> None:
    """Prepare authenticated physical control data or report a short simulator diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "screen"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forcing", choices=("iid", "driven"), default="iid")
    args = vars(parser.parse_args())
    torch.set_num_threads(1)
    config = ProbeDataConfig(forcing=cast(str, args["forcing"]))
    if args["command"] == "prepare":
        result = prepare(args["output"], config)
    else:
        result = reference_screen(config)
        write_json(args["output"], result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
