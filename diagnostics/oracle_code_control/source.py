"""Authenticate original caches, archived fresh predictor and diagnostic training labels."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.encoder_gradient_control.source import SourceBundle, build_source_model
from diagnostics.group_latent_control.run import initialize_control
from diagnostics.group_latent_table_audit import STATE_TOLERANCE
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.multiquery_control.run import sha
from diagnostics.table_settling_control.source import authenticated_directory
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA


@dataclass(frozen=True)
class OracleInputs:
    """Diagnostic truth is explicitly separate from the original label-free cache."""

    bundle: SourceBundle
    model: SCJEPA
    theta: Tensor
    initialization: dict[str, Any]
    hashes: dict[str, str]


def load_inputs(
    source: Path,
    audit: Path,
    seed: int,
    *,
    training_source_root: Path | None = None,
    verification: bool = False,
) -> OracleInputs:
    """Expose training theta only after completed audit and every source hash authenticate."""
    bundle = load_primary_source(
        source,
        seed,
        expected_step=2 if verification else 30000,
        **({"expected_checkpoint_sha256": None} if verification else {}),
        training_source_root=training_source_root,
    )
    completion = authenticated_directory(
        audit, {"protocol.json", "results.json", "audit_inputs.pt"}
    )
    protocol = cast(dict[str, Any], read_json(audit / "protocol.json"))
    result = cast(dict[str, Any], read_json(audit / "results.json"))
    if canonical_fingerprint({k: v for k, v in protocol.items() if k != "identity"}) != completion[
        "identity"
    ] or any(protocol["source_hashes"].get(name) != value for name, value in bundle.hashes.items()):
        raise ValueError("Code audit does not bind the original source cache")
    fingerprint = bundle.evaluation["metadata"]["training_dataset_fingerprint"]
    receipt = result["reconstruction"]
    if (
        protocol["training_dataset_fingerprint"] != fingerprint
        or receipt["dataset_fingerprint"] != fingerprint
        or receipt["groups_verified"] != bundle.train_groups
        or not receipt["all_actions_equal"]
        or not receipt["all_parameter_payloads_reproduce_observations"]
        or receipt["state_tolerance"] != STATE_TOLERANCE
        or any(value > STATE_TOLERANCE for value in receipt["state_max_absolute_error"].values())
    ):
        raise ValueError("Code audit did not completely verify training theta/cache binding")
    model, discarded = build_source_model(bundle, torch.device("cpu"))
    del discarded
    unused_table, initialization, source_hashes = initialize_control(model, bundle, "fresh_zero")
    del unused_table
    for name, expected in protocol["source_hashes"].items():
        if source_hashes.get(name) != expected:
            raise ValueError("Audit source or initialization hash differs")
    payload = cast(
        dict[str, Any],
        torch.load(audit / "audit_inputs.pt", map_location="cpu", weights_only=False),
    )
    theta = cast(Tensor, payload["theta_evaluation_only"])
    if (
        payload["identity"] != completion["identity"]
        or theta.shape != (bundle.train_groups, 6)
        or not bool(torch.isfinite(theta).all())
        or bool((theta.abs() >= 0.75).any())
        or tensor_fingerprint(theta) != receipt["theta_tensor_fingerprint"]
    ):
        raise ValueError("Authenticated diagnostic training theta changed")
    hashes: dict[str, str] = {}
    for root, entries in ((source, source_hashes), (audit, completion["hashes"])):
        for name, expected in entries.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or sha(path) != expected:
                raise ValueError("Immutable source artifact changed")
            hashes[str(path)] = expected
    hashes[str((audit / "completed.json").resolve())] = sha(audit / "completed.json")
    return OracleInputs(bundle, model.requires_grad_(False).eval(), theta, initialization, hashes)
