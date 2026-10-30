"""Run an explicitly declared 300k sparse comparison with the unchanged trainer.

The ordinary runner retains its original budget guard. This separate control
binds its own source and policy into the scientific run identity, authenticates
the same dense initializer and calibration, and starts fresh sparse optimization.
It does not extend or relabel a completed shorter run.
"""

import argparse
import json
from dataclasses import dataclass, field, fields
from pathlib import Path

from diagnostics.masked_mlp_learned.run import TrainConfig, train
from diagnostics.shared.io import sha

SCHEMA = "masked-mlp-sparse-budget-control-v1"
STEPS = 300_000


@dataclass(frozen=True)
class BudgetConfig(TrainConfig):
    """Keep every ordinary setting guard except the declared sparse update budget."""

    steps: int = STEPS
    budget_control_schema: str = field(default=SCHEMA, init=False)
    budget_control_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate the base contract and bind this control's source to the run."""
        if self.condition not in {"baseline", "full"}:
            raise ValueError("the budget control is only for matched sparse arms")
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if self.profile == "exploratory" and self.steps != STEPS:
            raise ValueError(f"the declared sparse budget is exactly {STEPS}")
        values = {item.name: getattr(self, item.name) for item in fields(TrainConfig)}
        if self.profile == "exploratory":
            # Check all other frozen settings using the ordinary accepted budget.
            # The actual immutable control config and training loop retain STEPS.
            values["steps"] = 200_000
        TrainConfig(**values)
        object.__setattr__(self, "budget_control_sha256", sha(Path(__file__)))


def main() -> None:
    """Launch one fresh fixed-budget arm, or resume that exact control identity."""
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
    parser.add_argument("--wandb-group", default="masked-mlp-budget-control")
    parser.add_argument("--resume", action="store_true")
    arguments = vars(parser.parse_args())
    resume = bool(arguments.pop("resume"))
    result = train(BudgetConfig(**arguments), resume=resume)
    print(json.dumps({key: result[key] for key in ("identity", "steps", "condition", "seed")}))


if __name__ == "__main__":
    main()
