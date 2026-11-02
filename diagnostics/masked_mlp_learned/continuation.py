"""Continue a completed sparse endpoint without resetting its training state.

The parent stays immutable. A separate run binds the parent checkpoint and this
entry point into its identity, then uses the unchanged trainer's resume path.
Only the budget and W&B run identity change; optimizer, EMA, GECO, sampler and
random states are inherited. Ordinary resume keeps its fixed-budget guards.
"""

import argparse
import json
import shutil
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.masked_mlp_learned import run as runner
from diagnostics.shared.io import read_json, save_torch, sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, write_json

SCHEMA = "masked-mlp-sparse-continuation-v1"
ADDITIONAL_STEPS = 50_000
RECORD_ONLY = frozenset(
    {
        "schema",
        "identity",
        "config",
        "physical_state_training",
        "true_parameter_training",
        "target_policy",
        "runtime",
        "initial_checkpoint_sha256",
    }
)


@dataclass(frozen=True)
class ContinuationConfig(runner.TrainConfig):
    """Bind the completed parent and declared extra budget to a new run identity."""

    continuation_parent_identity: str = ""
    continuation_checkpoint_sha256: str = ""
    continuation_start_step: int = 0
    continuation_schema: str = field(default=SCHEMA, init=False)
    continuation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Preserve every ordinary setting except the explicitly extended budget."""
        if self.condition not in {"baseline", "full"}:
            raise ValueError("continuation requires a completed sparse arm")
        if type(self.continuation_start_step) is not int or self.continuation_start_step < 1:
            raise ValueError("continuation requires a positive parent update count")
        if type(self.steps) is not int or self.steps <= self.continuation_start_step:
            raise ValueError("continuation must add updates")
        if self.profile == "exploratory" and (
            self.continuation_start_step != 300_000
            or self.steps != self.continuation_start_step + ADDITIONAL_STEPS
        ):
            raise ValueError("declared continuation is exactly 300000 to 350000 updates")
        for value in (self.continuation_parent_identity, self.continuation_checkpoint_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("continuation requires authenticated parent identities")
        values = {item.name: getattr(self, item.name) for item in fields(runner.TrainConfig)}
        if self.profile == "exploratory":
            values["steps"] = 200_000
        runner.TrainConfig(**values)
        object.__setattr__(self, "continuation_sha256", sha(Path(__file__)))


def continue_training(
    source_dir: Path,
    output_dir: Path,
    *,
    additional_steps: int = ADDITIONAL_STEPS,
    device: str | None = None,
    threads: int | None = None,
    wandb_group: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Authenticate a completed parent and continue it in a separate output directory."""
    source, output = source_dir.resolve(), output_dir.resolve()
    if source == output:
        raise ValueError("continuation must preserve its parent in a separate directory")
    if type(additional_steps) is not int or additional_steps < 1:
        raise ValueError("additional_steps must be a positive integer")
    with file_lock(output.parent / f".{output.name}.lock"):
        parent = runner.verify_completion(source, kind="train", require_current_source=True)
        record = read_json(source / "config.json")
        specification = {k: v for k, v in record.items() if k not in RECORD_ONLY}
        if canonical_fingerprint(specification) != parent["identity"]:
            raise ValueError("parent specification and completion disagree")
        values = {item.name: record["config"][item.name] for item in fields(runner.TrainConfig)}
        if values["steps"] != parent["steps"]:
            raise ValueError("parent configuration and completed update count disagree")
        values.update(
            output_dir=str(output),
            steps=int(parent["steps"]) + additional_steps,
            continuation_parent_identity=parent["identity"],
            continuation_checkpoint_sha256=parent["hashes"]["last.pt"],
            continuation_start_step=parent["steps"],
        )
        for name, value in (("device", device), ("threads", threads), ("wandb_group", wandb_group)):
            if value is not None:
                values[name] = value
        config = ContinuationConfig(**values)
        specification["scientific_config"] = runner._scientific_config(config)  # pyright: ignore[reportPrivateUsage]
        identity = canonical_fingerprint(specification)
        receipt = {
            "schema": SCHEMA,
            "identity": identity,
            "parent_identity": parent["identity"],
            "parent_directory": str(source),
            "parent_completion_sha256": sha(source / "completion.json"),
            "parent_checkpoint_sha256": parent["hashes"]["last.pt"],
            "parent_step": parent["steps"],
            "final_step": config.steps,
            "additional_steps": additional_steps,
            "training_code_identity": runner.code_identity(),
            "continuation_sha256": config.continuation_sha256,
            "checkpoint_metadata_changes": ["identity", "wandb"],
            "state_policy": "inherit model, optimizer, objective, sampler and all RNG states",
            "test_evaluated": False,
        }
        if resume:
            if read_json(output / "continuation.json") != receipt:
                raise ValueError("continuation parent, source or budget differs")
            if sha(output / "parent.pt") != parent["hashes"]["last.pt"]:
                raise ValueError("preserved parent checkpoint differs")
        else:
            checkpoint = cast(
                dict[str, Any],
                torch.load(source / "last.pt", map_location="cpu", weights_only=False),
            )
            if (
                checkpoint.get("schema") != runner.SCHEMA
                or checkpoint.get("identity") != parent["identity"]
                or checkpoint.get("step") != parent["steps"]
                or checkpoint.get("initial_checkpoint_sha256") != sha(source / "initial.pt")
            ):
                raise ValueError("parent checkpoint does not match its completed run")
            runner._fresh_output(output)  # pyright: ignore[reportPrivateUsage]
            for name in ("initial.pt", "startup.json", "history.jsonl"):
                shutil.copy2(source / name, output / name)
            shutil.copy2(source / "last.pt", output / "parent.pt")
            checkpoint["identity"] = identity
            checkpoint["wandb"] = {}  # New run; never append to the completed parent's W&B run.
            save_torch(output / "last.pt", checkpoint)
            write_json(output / "continuation.json", receipt)
        result = runner._train(config, output, resume=True)  # pyright: ignore[reportPrivateUsage]
        runner.verify_completion(source, kind="train", require_current_source=True)
        if sha(Path(__file__)) != config.continuation_sha256:
            raise RuntimeError("continuation source changed while training")
        write_json(
            output / "continuation-completion.json",
            receipt
            | {
                "completion_sha256": sha(output / "completion.json"),
                "checkpoint_sha256": result["hashes"]["last.pt"],
                "parent_artifacts_unchanged": True,
            },
        )
        return result


def main() -> None:
    """Continue the declared extra budget, preserving every parent artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--additional-steps", type=int, default=ADDITIONAL_STEPS)
    parser.add_argument("--device")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--wandb-group")
    parser.add_argument("--resume", action="store_true")
    result = continue_training(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("identity", "steps", "condition", "seed")}))


if __name__ == "__main__":
    main()
