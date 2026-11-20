"""Split generation, storage and label-free rendered inputs.

Each split stores float64 physical states, float64 parameters and float32
actions. ``RenderedInputs`` renders frames on the training device from the
stored trajectories (the RGB camera of ``dgp.Renderer``) and returns only
``observations`` and ``actions``. Physical references are read separately by
evaluation through ``load_labels``; they never enter a model batch.

Usage (``-m diagnostics.vision_dimer.data``):
    --data-dir DIR                     train, calibration and validation splits
    --data-dir DIR --include-test      the test split, generated once at the end
    --data-dir DIR --verify validation regenerate a stored split and compare bit for bit
"""

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.vision_dimer import dgp

SPLITS = {"train": 0, "calibration": 1, "validation": 2, "test": 3}
SCHEMA = "vision-dimer-data-v1"


@dataclass(frozen=True)
class DataConfig:
    """Cohort sizes, episode geometry and the declared physics/renderer."""

    seed: int = 20261115
    train_groups: int = 8192
    calibration_groups: int = 512
    validation_groups: int = 256
    test_groups: int = 512
    views: int = 10
    context: int = 10
    forecast: int = 2
    episode_frames: int = 40
    physics: dgp.Physics = field(default_factory=dgp.Physics)
    renderer: dgp.Renderer = field(default_factory=dgp.Renderer)

    def __post_init__(self) -> None:
        """Episodes must contain at least one complete context-plus-forecast window."""
        if self.episode_frames < self.frames:
            raise ValueError("episode_frames must be at least context + forecast")

    @property
    def frames(self) -> int:
        """Frames per model window: context followed by forecast targets."""
        return self.context + self.forecast

    def groups(self, split: str) -> int:
        """Number of systems in a split."""
        return {
            "train": self.train_groups,
            "calibration": self.calibration_groups,
            "validation": self.validation_groups,
            "test": self.test_groups,
        }[split]

    def as_record(self) -> dict[str, Any]:
        """JSON-normalized settings."""
        record = {
            "seed": self.seed,
            "groups": {split: self.groups(split) for split in SPLITS},
            "views": self.views,
            "context": self.context,
            "forecast": self.forecast,
            "episode_frames": self.episode_frames,
            "physics": self.physics.as_record(),
            "renderer": self.renderer.as_record(),
        }
        return cast(dict[str, Any], json.loads(json.dumps(record)))

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "DataConfig":
        """Inverse of ``as_record``."""

        def frozen(value: object) -> object:
            if isinstance(value, list):
                return tuple(frozen(item) for item in cast(list[object], value))
            return value

        physics = {key: frozen(value) for key, value in record["physics"].items()}
        return cls(
            seed=record["seed"],
            train_groups=record["groups"]["train"],
            calibration_groups=record["groups"]["calibration"],
            validation_groups=record["groups"]["validation"],
            test_groups=record["groups"]["test"],
            views=record["views"],
            context=record["context"],
            forecast=record["forecast"],
            episode_frames=record.get("episode_frames", record["context"] + record["forecast"]),
            physics=dgp.Physics(**cast(dict[str, Any], physics)),
            renderer=dgp.Renderer(
                **cast(dict[str, Any], {k: frozen(v) for k, v in record["renderer"].items()})
            ),
        )


def sha256(path: Path) -> str:
    """Hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_split(config: DataConfig, split: str) -> dict[str, Tensor]:
    """Deterministically generate one split on CPU in float64."""
    generator = torch.Generator().manual_seed(config.seed + 1_000_003 * SPLITS[split])
    theta = dgp.sample_theta(config.groups(split), config.physics, generator)
    states, actions = dgp.simulate(
        theta, config.views, config.episode_frames, config.physics, generator
    )
    return {"states": states, "theta": theta, "actions": actions.float()}


def prepare(directory: Path, config: DataConfig, *, include_test: bool = False) -> dict[str, Any]:
    """Write TRAIN/calibration/validation; test only when explicitly requested."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"schema": SCHEMA, "config": config.as_record(), "splits": {}}
    )
    if manifest["config"] != config.as_record():
        raise ValueError("existing data directory was generated with different settings")
    for split in (s for s in SPLITS if s != "test" or include_test):
        path = directory / f"{split}.pt"
        if split in manifest["splits"]:
            if sha256(path) != manifest["splits"][split]["sha256"]:
                raise ValueError(f"stored {split} split changed")
            continue
        tensors = generate_split(config, split)
        torch.save(tensors, path)
        rows = tensors["states"].reshape(-1, dgp.STATE_DIM)
        manifest["splits"][split] = {
            "sha256": sha256(path),
            "groups": config.groups(split),
            "state_std": [round(float(v), 6) for v in rows.std(0)],
            "max_abs_state": float(rows.abs().max()),
            "correlation": [[round(float(v), 4) for v in row] for row in torch.corrcoef(rows.T)],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"prepared {split}: {tuple(tensors['states'].shape)}", flush=True)
    return manifest


def load_config(directory: Path) -> DataConfig:
    """Rebuild the declared configuration from a manifest."""
    return DataConfig.from_record(json.loads((directory / "manifest.json").read_text())["config"])


def _load(directory: Path, split: str) -> dict[str, Tensor]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if split not in manifest["splits"]:
        raise FileNotFoundError(f"split {split} has not been generated")
    path = directory / f"{split}.pt"
    if sha256(path) != manifest["splits"][split]["sha256"]:
        raise ValueError(f"stored {split} split changed")
    return cast(dict[str, Tensor], torch.load(path, weights_only=True))


def load_labels(directory: Path, split: str) -> dict[str, Tensor]:
    """Physical states and parameters, for evaluation of non-TRAIN splits only."""
    if split == "train":
        raise ValueError("no learning or evaluation step needs TRAIN physical labels")
    tensors = _load(directory, split)
    return {"states": tensors["states"], "theta": tensors["theta"]}


def window(series: Tensor, starts: Tensor | None, length: int) -> Tensor:
    """Per-(group, view) windows ``(B, R, length, ...)`` of ``(B, R, E, ...)`` series."""
    if starts is None:
        return series[:, :, :length]
    frames = starts.to(series.device)[..., None] + torch.arange(length, device=series.device)
    index = frames.reshape(*frames.shape, *([1] * (series.ndim - 3)))
    return torch.gather(series, 2, index.expand(*frames.shape, *series.shape[3:]))


class RenderedInputs:
    """Label-free model inputs: rendered frames and observed actions only."""

    def __init__(self, directory: Path, split: str, device: torch.device) -> None:
        """Keep trajectories private on the device and render on demand."""
        tensors = _load(directory, split)
        config = load_config(directory)
        self.split = split
        self.renderer = config.renderer
        self._states = tensors["states"].float().to(device)
        self.actions = tensors["actions"].to(device)
        self.groups, self.views, self.episode_frames = self._states.shape[:3]
        self.frames = config.frames

    def __len__(self) -> int:
        """Number of systems."""
        return self.groups

    def batch(self, indices: Tensor, starts: Tensor | None = None) -> dict[str, Tensor]:
        """Return ``observations (B,R,T,C,H,W)`` and ``actions (B,R,T-1,A)`` only.

        ``starts (B,R)`` selects a window of ``T = context + forecast`` frames per
        episode; the default is the first window (evaluation convention).
        """
        indices = indices.to(self._states.device)
        states = window(self._states[indices], starts, self.frames)
        return {
            "observations": self.renderer(states),
            "actions": window(self.actions[indices], starts, self.frames - 1),
        }

    def episodes(self, indices: Tensor) -> dict[str, Tensor]:
        """Complete episodes (frozen-encoder caching only): all frames and actions."""
        indices = indices.to(self._states.device)
        return {
            "observations": self.renderer(self._states[indices]),
            "actions": self.actions[indices],
        }


def verify_split(directory: Path, split: str) -> dict[str, Any]:
    """Regenerate a stored split in memory from its manifest and compare it bit for bit."""
    stored = _load(directory, split)
    fresh = generate_split(load_config(directory), split)
    checks = {
        key: bool(fresh[key].dtype == stored[key].dtype and torch.equal(fresh[key], stored[key]))
        for key in ("states", "theta", "actions")
    }
    report: dict[str, Any] = {**checks, "identical": all(checks.values())}
    report["max_abs_state_difference"] = float((fresh["states"] - stored["states"]).abs().max())
    report["groups"] = len(stored["theta"])
    return report


def main() -> None:
    """Prepare the declared cohort; test generation requires an explicit flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--include-test", action="store_true", help="also generate the test split (once)"
    )
    parser.add_argument("--train-groups", type=int, default=8192)
    parser.add_argument("--episode-frames", type=int, default=40)
    parser.add_argument(
        "--verify", choices=tuple(SPLITS), help="regenerate a stored split; compare bit for bit"
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.verify is not None:
        report = verify_split(args.data_dir, args.verify)
        print(json.dumps(report))
        raise SystemExit(0 if report["identical"] else 1)
    config = DataConfig(train_groups=args.train_groups, episode_frames=args.episode_frames)
    manifest = prepare(args.data_dir, config, include_test=args.include_test)
    print(json.dumps({k: v["sha256"] for k, v in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
