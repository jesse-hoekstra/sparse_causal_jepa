"""Bind the omitted-logit comparison to its completed, unchanged oracle reference."""

from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.multiquery_control.run import sha
from diagnostics.oracle_code_control.run import code_identity, fixed_hashes
from diagnostics.oracle_code_control.source import OracleInputs
from diagnostics.table_settling_control.source import authenticated_directory
from scjepa.experiments.storage import read_json


def verify_reference(
    root: Path | None,
    inputs: OracleInputs,
    *,
    seed: int,
    steps: int,
    groups: int,
    bootstrap_samples: int,
    device: str,
    threads: int,
    verification: bool = False,
) -> dict[str, Any]:
    """Require the same fresh weights, original cache, truth, training and evaluation settings."""
    if root is None:
        if not verification:
            raise ValueError("Scientific comparison requires a completed original oracle run")
        return {"verification_only_without_reference": True}
    root = root.resolve()
    completion = authenticated_directory(root, {"protocol.json", "results.json", "last.pt"})
    protocol = cast(dict[str, Any], read_json(root / "protocol.json"))
    results = cast(dict[str, Any], read_json(root / "results.json"))
    expected_settings = {
        "seed": seed,
        "steps": steps,
        "groups": groups,
        "bootstrap_samples": bootstrap_samples,
        "device": device,
        "threads": threads,
        "verification": verification,
    }
    if any(protocol["config"][name] != value for name, value in expected_settings.items()):
        raise ValueError("Original oracle training/evaluation settings differ from comparison")
    if (
        results["step"] != steps
        or protocol["torch"] != str(torch.__version__)
        or protocol["source_hashes"] != inputs.hashes
        or protocol["initialization"] != inputs.initialization
        or protocol["frozen_hashes"] != fixed_hashes(inputs.model)
        or protocol["code_identity"] != code_identity()
        or protocol["objective"]
        != "Mean two-step forecast MSE + .0001 mean logit penalty. "
        "No SIGReg/consistency gradients or optimizer history."
    ):
        raise ValueError("Original oracle source, initialization, objective or code differs")
    return {
        "directory": str(root),
        "identity": completion["identity"],
        "completed_sha256": sha(root / "completed.json"),
        "artifact_hashes": completion["hashes"],
        "matched_settings": expected_settings,
        "matched_source_hashes": inputs.hashes,
        "matched_initialization": inputs.initialization,
        "reference_objective": protocol["objective"],
        "controlled_change": "Optimized logit weight 0.0001 -> 0; forecast unchanged",
    }
