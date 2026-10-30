"""Run a fixed 300k sparse phase followed by a 50k learning-rate cooldown.

Both phases train all ordinary sparse modules. The optimizer step hook changes
only AdamW's learning rate after update 300000, leaving the authenticated trainer
and its existing dense/calibration parents unchanged. This helper's exact source
and schedule enter the run identity. Use ``train_cooldown`` or this module's CLI.
"""

import argparse
import json
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from torch.optim import AdamW, Optimizer
from torch.optim.optimizer import register_optimizer_step_pre_hook

from diagnostics.masked_mlp_learned.run import TrainConfig, train
from diagnostics.oracle_state_learning.data import sha

SCHEMA = "masked-mlp-sparse-cooldown-control-v1"
STEPS = 350_000
COOLDOWN_START = 300_000
COOLDOWN_FACTOR = 0.1


@dataclass(frozen=True)
class CooldownConfig(TrainConfig):
    """Bind the fixed schedule while preserving every other ordinary setting guard."""

    steps: int = STEPS
    cooldown_start: int = COOLDOWN_START
    cooldown_factor: float = COOLDOWN_FACTOR
    cooldown_control_schema: str = field(default=SCHEMA, init=False)
    cooldown_control_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate the schedule and authenticate its implementation."""
        if self.condition not in {"baseline", "full"}:
            raise ValueError("the cooldown control is only for matched sparse arms")
        if type(self.steps) is not int or type(self.cooldown_start) is not int:
            raise ValueError("schedule boundaries must be integers")
        if not 0 < self.cooldown_start < self.steps:
            raise ValueError("cooldown must begin strictly inside the update budget")
        if self.cooldown_factor != COOLDOWN_FACTOR:
            raise ValueError("the declared cooldown factor is exactly 0.1")
        if self.profile == "exploratory" and (
            self.steps != STEPS or self.cooldown_start != COOLDOWN_START
        ):
            raise ValueError("the declared schedule is exactly 300k plus 50k updates")
        values = {item.name: getattr(self, item.name) for item in fields(TrainConfig)}
        if self.profile == "exploratory":
            values["steps"] = 200_000
        TrainConfig(**values)
        object.__setattr__(self, "cooldown_control_sha256", sha(Path(__file__)))


@contextmanager
def cooldown_schedule(config: CooldownConfig) -> Generator[None]:
    """Scope a deterministic step hook to the single optimizer of this training call.

    Adam's persisted update counter determines the rate, including after interruption.
    No scheduler RNG, extra optimizer state or altered resume identity is needed.
    The handle is removed on both successful completion and exceptions.
    """
    observed: list[Optimizer] = []

    def schedule(optimizer: Optimizer, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        if not isinstance(optimizer, AdamW):
            raise TypeError("the cooldown requires the ordinary AdamW optimizer")
        if not observed:
            observed.append(optimizer)
        elif observed[0] is not optimizer:
            raise RuntimeError("a cooldown training call must own exactly one optimizer")
        first = optimizer.param_groups[0]["params"][0]
        step = int(optimizer.state[first].get("step", 0)) + 1
        factor = config.cooldown_factor if step > config.cooldown_start else 1.0
        for group in optimizer.param_groups:
            group["lr"] = config.lr * factor

    handle = register_optimizer_step_pre_hook(schedule)
    try:
        yield
    finally:
        handle.remove()


def train_cooldown(config: CooldownConfig, *, resume: bool = False) -> dict[str, Any]:
    """Train a fresh matched arm or resume its exact recorded schedule and source."""
    with cooldown_schedule(config):
        return train(config, resume=resume)


def main() -> None:
    """Launch one predeclared 350k arm; no validation-dependent stopping or selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--condition", required=True, choices=("baseline", "full"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--init-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--wandb-mode", default="disabled")
    parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="masked-mlp-cooldown-control")
    parser.add_argument("--resume", action="store_true")
    arguments = vars(parser.parse_args())
    resume = bool(arguments.pop("resume"))
    result = train_cooldown(CooldownConfig(**arguments), resume=resume)
    print(json.dumps({key: result[key] for key in ("identity", "steps", "condition", "seed")}))


if __name__ == "__main__":
    main()
