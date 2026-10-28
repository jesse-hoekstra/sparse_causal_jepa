"""Small identity helpers for the shared preparation and native Trainer checkpoints."""

import hashlib
from pathlib import Path
from typing import Any, cast

from diagnostics.joint_dense_continuation.run import code_identity as parent_identity
from diagnostics.multiquery_control.run import sha
from scjepa.experiments.storage import read_json, write_json
from scjepa.training.loop import Trainer


def code_identity() -> str:
    """Bind the new adapter to frozen ancestry and the native GECO/experiment implementation."""
    digest = hashlib.sha256(parent_identity().encode())
    root = Path(__file__).resolve().parents[2]
    for path in [
        *sorted(Path(__file__).parent.glob("*.py")),
        root / "src/scjepa/training/loop.py",
        root / "src/scjepa/training/lagrangian.py",
        root / "src/scjepa/experiments/evaluation.py",
        root / "src/scjepa/experiments/comparison.py",
    ]:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_checkpoint(trainer: Trainer, out: Path, identity: str) -> None:
    """Save native resumable state and its durable integrity receipt."""
    trainer.save_checkpoint(out / "last.pt")
    write_json(
        out / "last.receipt.json",
        {"identity": identity, "sha256": sha(out / "last.pt"), "step": trainer.step},
    )


def verify_checkpoint(out: Path, identity: str) -> dict[str, Any]:
    """Reject altered native checkpoints before restoring optimizer/controller state."""
    receipt = cast(dict[str, Any], read_json(out / "last.receipt.json"))
    if receipt["identity"] != identity or receipt["sha256"] != sha(out / "last.pt"):
        raise ValueError("Checkpoint hash or run identity changed")
    return receipt


def disjoint(output: Path, roots: list[Path]) -> None:
    """Reject outputs nested in, equal to, or enclosing any immutable input root."""
    output = output.resolve()
    for root in roots:
        source = root.resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Output must be disjoint from immutable source roots")
