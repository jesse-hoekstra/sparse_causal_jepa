"""Verified primary checkpoint and compatible source/current-code comparisons."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.encoder_gradient_control.source import SourceBundle, load_source
from diagnostics.frozen_predictor_control.source import (
    build_fresh_model,
    frozen_module_hashes,
    load_source_bundle,
)
from diagnostics.multiquery_control.data import EvaluationData, module_fingerprint
from diagnostics.multiquery_control.run import sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA

CURRENT_CHECKPOINT_SHA256 = "822e2ea5a485800fdbe422f3f0d32f25cf84817801a9383c41a71f1559fe3da2"
JOINT_CHECKPOINT_SHA256 = "d475564e24994f5264a6cb5d2fb44f89b50db8cf08aaa5183e3edc356743990c"


@dataclass(frozen=True)
class ComparisonSources:
    """Two actual encoders sharing exactly the frozen predictor and state maps."""

    source_model: SCJEPA
    current_model: SCJEPA
    evaluation: EvaluationData
    identity: dict[str, Any]


def load_primary_source(
    source_dir: Path,
    seed: int = 202610261301,
    *,
    expected_step: int = 30000,
    expected_checkpoint_sha256: str | None = JOINT_CHECKPOINT_SHA256,
    training_source_root: Path | None = None,
) -> SourceBundle:
    """Load the original joint endpoint and its coherent saved optimizer history."""
    bundle = load_source(source_dir, seed, training_source_root=training_source_root)
    if bundle.source_step != expected_step:
        raise ValueError(
            f"Requested original {expected_step}-step endpoint, got {bundle.source_step}"
        )
    checkpoint = source_dir / f"seed-{seed}/prediction_1/last.pt"
    if expected_checkpoint_sha256 is not None and sha(checkpoint) != expected_checkpoint_sha256:
        raise ValueError("Primary checkpoint differs from the recorded original endpoint hash")
    optimizer = cast(dict[str, Any], bundle.checkpoint["optimizer"])
    group = optimizer["param_groups"][0]
    if (
        group["params"] != list(range(len(group["params"])))
        or tuple(group["betas"]) != (0.9, 0.999)
        or group["eps"] != 1e-8
        or group.get("amsgrad", False)
        or group.get("maximize", False)
    ):
        raise ValueError("Saved AdamW parameter order or optimizer semantics are incompatible")
    return bundle


def load_comparison_sources(
    bundle_dir: Path,
    all_six_checkpoint: Path,
    current_checkpoint: Path,
    *,
    current_protocol: Path | None = None,
    expected_current_step: int = 100000,
    expected_current_sha256: str | None = CURRENT_CHECKPOINT_SHA256,
    device: torch.device | None = None,
) -> ComparisonSources:
    """Require the actual source encoder, verified source/current artifacts and shared maps."""
    bundle = load_source_bundle(bundle_dir)
    provenance = bundle.manifest["provenance"]
    if sha(all_six_checkpoint) != provenance["sha256"]["checkpoint"]:
        raise ValueError("Actual all-six checkpoint differs from the exported source provenance")
    source_payload = cast(
        dict[str, Any], torch.load(all_six_checkpoint, map_location="cpu", weights_only=False)
    )
    source_model = build_fresh_model(bundle, 0, torch.device("cpu"))
    source_model.load_state_dict(source_payload["model"], strict=True)
    if module_fingerprint(source_model.parameter_encoder) != provenance["source_q_sha256"]:
        raise ValueError("Source encoder fingerprint differs from the actual original checkpoint")
    protocol_path = current_protocol or current_checkpoint.parent.parent / "protocol.json"
    protocol = cast(dict[str, Any], read_json(protocol_path))
    completion = cast(dict[str, Any], read_json(current_checkpoint.parent / "completed.json"))
    if expected_current_sha256 is not None and sha(current_checkpoint) != expected_current_sha256:
        raise ValueError("Current checkpoint differs from the recorded frozen-predictor endpoint")
    expected_identity = canonical_fingerprint(
        {key: protocol[key] for key in ("settings", "source_identity", "code_identity", "torch")}
    )
    if expected_identity != protocol["identity"]:
        raise ValueError("Current protocol identity does not match its recorded configuration")
    current_payload = cast(
        dict[str, Any], torch.load(current_checkpoint, map_location="cpu", weights_only=False)
    )
    if (
        protocol["source_identity"] != bundle.manifest["identity"]
        or protocol["frozen_hashes"] != bundle.manifest["frozen_hashes"]
        or current_payload["identity"] != protocol["identity"]
        or completion["identity"] != protocol["identity"]
        or completion["hashes"]["last.pt"] != sha(current_checkpoint)
        or current_payload["step"] != expected_current_step
        or protocol["config"]["steps"] != expected_current_step
        or current_payload["seed"] != protocol["config"]["seed"]
        or current_payload["frozen_hashes"] != bundle.manifest["frozen_hashes"]
    ):
        raise ValueError("Current frozen-predictor checkpoint/protocol/source identity mismatch")
    current_model = build_fresh_model(bundle, 0, torch.device("cpu"))
    current_model.load_state_dict(current_payload["model"], strict=True)
    for model in (source_model, current_model):
        if frozen_module_hashes(model) != bundle.manifest["frozen_hashes"]:
            raise ValueError("Compatible code substitution requires identical frozen modules")
        if model.get_extra_state() != bundle.frozen_payload["extra_state"]:
            raise ValueError("Compatible code substitution requires identical prediction flags")
        model.to(device or torch.device("cpu")).double().eval().requires_grad_(False)
    return ComparisonSources(
        source_model,
        current_model,
        bundle.evaluation,
        {
            "bundle_identity": bundle.manifest["identity"],
            "source_checkpoint_sha256": sha(all_six_checkpoint),
            "current_checkpoint_sha256": sha(current_checkpoint),
            "current_protocol_sha256": sha(protocol_path),
            "frozen_hashes": bundle.manifest["frozen_hashes"],
            "optimizer_comparison": (
                "None: no predictor Adam history is fabricated for the frozen-predictor endpoint."
            ),
        },
    )
