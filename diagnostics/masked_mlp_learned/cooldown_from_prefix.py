"""Reuse an authenticated fixed 300k prefix of the declared 350k cooldown schedule.

The source run remains unchanged. A separate child copies its model, optimizer,
GECO, sampler and RNG state, then performs the prescribed final 50k updates with
a new W&B identity. Initial tensors and the complete history retain the common
dense starting point. Parent and adapter hashes enter the child's run identity.
"""

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.masked_mlp_learned import run
from diagnostics.masked_mlp_learned.budget_control import BudgetConfig
from diagnostics.masked_mlp_learned.cooldown_control import CooldownConfig, train_cooldown
from diagnostics.oracle_state_learning.data import read_json, save_torch, sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, write_json


@dataclass(frozen=True)
class PrefixConfig(CooldownConfig):
    """Bind an unchanged completed prefix to a separately authenticated child."""

    prefix_directory: str = ""
    prefix_identity: str = ""
    prefix_completion_sha256: str = ""
    prefix_checkpoint_sha256: str = ""
    prefix_control_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Keep the schedule guards and require explicit parent provenance."""
        super().__post_init__()
        if not Path(self.prefix_directory).is_absolute():
            raise ValueError("prefix_directory must be absolute")
        for value in (
            self.prefix_identity,
            self.prefix_completion_sha256,
            self.prefix_checkpoint_sha256,
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("prefix identities must be SHA256 digests")
        object.__setattr__(self, "prefix_control_sha256", sha(Path(__file__)))


def make_config(
    prefix: Path,
    output: Path,
    *,
    wandb_group: str,
    device: str | None = None,
    cooldown_updates: int = 50_000,
) -> PrefixConfig:
    """Inherit every numerical setting; only the declared tail and execution metadata differ."""
    prefix = prefix.resolve()
    completion = run.verify_completion(prefix, kind="train", require_current_source=True)
    source = read_json(prefix / "config.json")
    values = {item.name: source["config"][item.name] for item in fields(run.TrainConfig)}
    if asdict(BudgetConfig(**values)) != source["config"]:
        raise ValueError("prefix must use the unchanged authenticated 300k budget control")
    if completion["steps"] != values["steps"] or completion["skipped_updates"] != 0:
        raise ValueError("prefix must be its completed fixed-budget endpoint")
    start = int(values["steps"])
    values.update(
        output_dir=str(output.resolve()),
        steps=start + cooldown_updates,
        wandb_group=wandb_group,
        device=values["device"] if device is None else device,
    )
    return PrefixConfig(
        **values,
        cooldown_start=start,
        prefix_directory=str(prefix),
        prefix_identity=completion["identity"],
        prefix_completion_sha256=sha(prefix / "completion.json"),
        prefix_checkpoint_sha256=sha(prefix / "last.pt"),
    )


def train_from_prefix(config: PrefixConfig, *, resume: bool = False) -> dict[str, Any]:
    """Authenticate, copy the exact prefix once, then use strict ordinary child resume."""
    prefix, output = Path(config.prefix_directory), Path(config.output_dir)
    completion = run.verify_completion(prefix, kind="train", require_current_source=True)
    if (
        completion["identity"] != config.prefix_identity
        or sha(prefix / "completion.json") != config.prefix_completion_sha256
        or sha(prefix / "last.pt") != config.prefix_checkpoint_sha256
        or completion["steps"] != config.cooldown_start
    ):
        raise ValueError("prefix identity, checkpoint or fixed endpoint changed")
    source = read_json(prefix / "config.json")
    source_values = {item.name: source["config"][item.name] for item in fields(run.TrainConfig)}
    if asdict(BudgetConfig(**source_values)) != source["config"]:
        raise ValueError("prefix must use the unchanged authenticated 300k budget control")
    ignored = {"output_dir", "steps", "device", "wandb_group"}
    if any(
        getattr(config, item.name) != source["config"][item.name]
        for item in fields(run.TrainConfig)
        if item.name not in ignored
    ):
        raise ValueError("the cooldown must preserve every numerical prefix setting")
    # Reconstruct and verify the frozen trainer's source specification before
    # replacing its configuration. This fails closed if that interface changes.
    metadata = {
        "schema",
        "identity",
        "config",
        "physical_state_training",
        "true_parameter_training",
        "target_policy",
        "runtime",
        "initial_checkpoint_sha256",
    }
    specification = {key: value for key, value in source.items() if key not in metadata}
    if canonical_fingerprint(specification) != config.prefix_identity:
        raise ValueError("prefix specification does not reproduce its identity")
    specification["scientific_config"] = run._scientific_config(config)  # pyright: ignore[reportPrivateUsage]
    identity = canonical_fingerprint(specification)
    with file_lock(output.parent / f".{output.name}.lock"):
        if not resume:
            output.mkdir(parents=True, exist_ok=False)
            for name in ("initial.pt", "startup.json", "history.jsonl"):
                shutil.copy2(prefix / name, output / name)
            shutil.copy2(prefix / "last.pt", output / "prefix.pt")
            if sha(output / "prefix.pt") != config.prefix_checkpoint_sha256:
                raise ValueError("copied prefix checkpoint differs")
            checkpoint = cast(
                dict[str, Any],
                torch.load(output / "prefix.pt", weights_only=False, map_location="cpu"),
            )
            if checkpoint["identity"] != config.prefix_identity:
                raise ValueError("prefix checkpoint identity differs")
            if checkpoint["step"] != config.cooldown_start:
                raise ValueError("prefix checkpoint is not the declared schedule boundary")
            if any(group["lr"] != config.lr for group in checkpoint["optimizer"]["param_groups"]):
                raise ValueError("prefix optimizer learning rate differs")
            checkpoint["identity"] = identity
            checkpoint.pop("wandb", None)  # A new child run, never append to the completed parent.
            save_torch(output / "last.pt", checkpoint)
            write_json(
                output / "prefix-transfer.json",
                {
                    "parent_directory": str(prefix),
                    "parent_identity": config.prefix_identity,
                    "parent_completion_sha256": config.prefix_completion_sha256,
                    "parent_checkpoint_sha256": config.prefix_checkpoint_sha256,
                    "child_identity": identity,
                    "child_initial_checkpoint_sha256": sha(output / "initial.pt"),
                    "child_boundary_checkpoint_sha256": sha(output / "last.pt"),
                    "boundary_step": config.cooldown_start,
                    "copied_state": "model, optimizer, GECO, sampler, Python/NumPy/Torch/CUDA RNG",
                    "metadata_changes": "new scientific identity and new W&B run",
                    "adapter_sha256": config.prefix_control_sha256,
                },
            )
        else:
            receipt = read_json(output / "prefix-transfer.json")
            if receipt["child_identity"] != identity:
                raise ValueError("child continuation configuration or source differs")
            if sha(output / "prefix.pt") != config.prefix_checkpoint_sha256:
                raise ValueError("retained prefix checkpoint changed")
    result = train_cooldown(config, resume=True)
    # Include the retained parent and transfer receipt in ordinary completion
    # authentication, including when an already-completed child is resumed.
    for name in ("prefix.pt", "prefix-transfer.json"):
        result["hashes"][name] = sha(output / name)
    write_json(output / "completion.json", result)
    return result


def main() -> None:
    """Execute only the fixed cooldown suffix of an authenticated 300k parent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = make_config(
        args.prefix_dir, args.output_dir, wandb_group=args.wandb_group, device=args.device
    )
    result = train_from_prefix(config, resume=args.resume)
    print(json.dumps({key: result[key] for key in ("identity", "steps", "condition", "seed")}))


if __name__ == "__main__":
    main()
