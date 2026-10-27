"""Authenticate a fixed learned six-code teacher and expose raw TRAIN-only targets."""

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.encoder_gradient_control.source import SourceBundle
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.run import sha
from diagnostics.multiteacher_curriculum.forward import Heads
from diagnostics.table_settling_control.source import authenticated_directory
from diagnostics.teacher_code_compression.data import (
    CompressionInputs,
    load_reference,
    verify_reference,
)
from diagnostics.teacher_code_compression.model import CompressionEncoder, initialize_modules
from diagnostics.teacher_code_compression.run import code_identity as compression_code_identity
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA

SELECTED_COMPRESSION_HASHES = {
    "last.pt": "2cd7bd02ba4f7e708de7110f64149b9674780a01f8fde7adf13799321daa8d27",
    "protocol.json": "b3d08987a41d2346769622a7c29ec700d77229877586732674a643037ec83cac",
    "completed.json": "fccf2cd749fbbd987d72a6ea9795826292415953584aa23ea50a24c8b9138211",
}


@dataclass(frozen=True)
class Inputs:
    """Fresh original student and independently frozen compressor/head/source artifacts."""

    bundle: SourceBundle
    model: SCJEPA
    compression: CompressionInputs
    compressor: CompressionEncoder
    heads: Heads
    initialization: dict[str, Any]
    hashes: dict[str, str]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class CompressedTargets:
    """Raw six-vectors followed by detached nonself-context means, without rescaling."""

    raw: Tensor
    targets: Tensor
    receipt: dict[str, Any]


def load_inputs(compression_source: Path, *, verification: bool = False) -> Inputs:
    """Use the completed fixed-budget compressor; never substitute its initial or best weights."""
    root = compression_source.resolve()
    if not verification and any(
        sha(root / name) != expected for name, expected in SELECTED_COMPRESSION_HASHES.items()
    ):
        raise ValueError("Compressor differs from the newly selected fixed30k artifact")
    completed = authenticated_directory(
        root, {"protocol.json", "results.json", "last.pt", "initial_modules.pt", "validation.json"}
    )
    protocol = cast(dict[str, Any], read_json(root / "protocol.json"))
    config = protocol["config"]
    evidence = {
        key: protocol[key]
        for key in (
            "settings",
            "reference",
            "source_hashes",
            "reference_hashes",
            "initialization",
            "code_identity",
            "torch",
        )
    }
    settings = dict(config)
    for name in ("curriculum_dir", "output_dir", "wandb_mode", "wandb_entity"):
        settings.pop(name)
    if (
        canonical_fingerprint(evidence) != protocol["identity"]
        or settings != protocol["settings"]
        or protocol["code_identity"] != compression_code_identity()
        or config["verification"] != verification
        or protocol["training_parameter_labels_absent"] is not True
        or (verification and not 1 <= config["steps"] <= 3)
        or (
            not verification
            and (config["steps"], config["seed"], config["groups"]) != (30000, 202610261301, 64)
        )
        or protocol["objective"]
        != "Mean teacher MSE + .4 parameter SIGReg + consistency1; no predictor"
    ):
        raise ValueError("Compression protocol, code, labels or fixed budget differ")
    compression = load_reference(Path(config["curriculum_dir"]), verification=verification)
    if (
        compression.reference != protocol["reference"]
        or compression.hashes != protocol["reference_hashes"]
        or compression.parent.hashes != protocol["source_hashes"]
    ):
        raise ValueError("Compression is not bound to its original teacher/cached TRAIN inputs")
    compressor, heads, initialization = initialize_modules(
        config["seed"], dtype=compression.train.dtype
    )
    if initialization != protocol["initialization"]:
        raise ValueError("Compression architecture or seeded initialization differs")
    initial = cast(
        dict[str, Any],
        torch.load(root / "initial_modules.pt", map_location="cpu", weights_only=False),
    )
    compressor.load_state_dict(initial["encoder"], strict=True)
    heads.load_state_dict(initial["heads"], strict=True)
    if (
        initial["identity"] != protocol["identity"]
        or module_fingerprint(compressor) != initialization["encoder_sha256"]
        or module_fingerprint(heads) != initialization["heads_sha256"]
    ):
        raise ValueError("Compression initial snapshot differs")
    payload = cast(
        dict[str, Any], torch.load(root / "last.pt", map_location="cpu", weights_only=False)
    )
    result = cast(dict[str, Any], read_json(root / "results.json"))
    if (
        payload["identity"] != completed["identity"]
        or payload["step"] != config["steps"]
        or result["step"] != config["steps"]
        or payload["initial_sha256"] != sha(root / "initial_modules.pt")
    ):
        raise ValueError("Compression checkpoint is not the completed fixed endpoint")
    compressor.load_state_dict(payload["encoder"], strict=True)
    heads.load_state_dict(payload["heads"], strict=True)
    if any(
        not bool(torch.isfinite(value).all())
        for module in (compressor, heads)
        for value in module.state_dict().values()
    ):
        raise ValueError("Compressor/head contains nonfinite weights")
    compressor.requires_grad_(False).eval()
    heads.requires_grad_(False).eval()
    hashes = {
        **compression.parent.hashes,
        **compression.hashes,
        **{str(root / name): value for name, value in completed["hashes"].items()},
        str(root / "completed.json"): sha(root / "completed.json"),
    }
    provenance = {
        "compression_directory": str(root),
        "identity": completed["identity"],
        "step": payload["step"],
        "checkpoint_sha256": sha(root / "last.pt"),
        "compressor_sha256": module_fingerprint(compressor),
        "heads_sha256": module_fingerprint(heads),
        "teacher_provenance": compression.parent.teacher_provenance,
        "teacher_input_transform": compression.reference["teacher_target_receipt"],
        "output_transform": "identity; raw learned6D code, no additional normalization or clipping",
        "target_order": (
            "Compress each single context first, then average the other nine raw6D codes"
        ),
        "heads_role": (
            "Fixed readout features of studentq only; never a training head or teacher input"
        ),
        "primary_compression_validation_r2": result["final_full_validation"][
            "selected_validation_r2"
        ],
        "selection_scope": (
            "NEW adaptive curriculum selected from secondary fixed-head all-six recovery. "
            "The compressor's predeclared direct-code probe gate failed and remains failed. "
            "Original teacher selection used prior parameter/test reports; "
            "no blinded discovery claim."
        ),
        "evaluation_contract": (
            "Retain original directq probe result separately. Declare expanded features "
            "directq plus each of three frozen6D heads(q); fit on calibration and choose "
            "feature map/family/hyperparameters on validation only, then fixed70k test."
        ),
    }
    inputs = Inputs(
        compression.parent.bundle,
        copy.deepcopy(compression.parent.model),
        compression,
        compressor,
        heads,
        copy.deepcopy(compression.parent.initialization),
        hashes,
        provenance,
    )
    verify_sources(inputs)
    return inputs


def verify_sources(inputs: Inputs) -> None:
    """Allow the student to learn while freezing teacher modules and every ancestor artifact."""
    verify_reference(inputs.compression)
    if any(sha(Path(name)) != value for name, value in inputs.hashes.items()):
        raise ValueError("Immutable compressed-teacher source changed")
    for name, module in (("compressor", inputs.compressor), ("heads", inputs.heads)):
        if module_fingerprint(module) != inputs.provenance[name + "_sha256"] or any(
            parameter.requires_grad for parameter in module.parameters()
        ):
            raise ValueError("Frozen compressed teacher or evaluation heads changed")


def leave_one_out_codes(raw: Tensor) -> Tensor:
    """Average other contexts AFTER nonlinear compression, with no new normalization."""
    if (
        raw.ndim != 3
        or raw.shape[1:] != (10, 6)
        or not raw.is_floating_point()
        or not bool(torch.isfinite(raw).all())
    ):
        raise ValueError("Expected finite single-context compressed codes [groups,10,6]")
    values = raw.detach().cpu().double()
    means = torch.stack(
        [torch.cat((values[:, :view], values[:, view + 1 :]), dim=1).mean(1) for view in range(10)],
        dim=1,
    )
    return means.to(raw)


def prepare_targets(
    inputs: Inputs,
    *,
    device: torch.device,
    batch_size: int = 64,
    progress: Callable[[int, int], None] | None = None,
) -> CompressedTargets:
    """Read only authenticated TRAIN teacher features, never calibration/test or labels."""
    if batch_size < 1:
        raise ValueError("Target preparation batch size must be positive")
    verify_sources(inputs)
    compressor = copy.deepcopy(inputs.compressor).to(device).eval()
    features = inputs.compression.train
    if features.shape != (inputs.bundle.train_groups, 10, 18) or features.requires_grad:
        raise ValueError("Invalid detached TRAIN teacher-union features")
    pieces: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            pieces.append(compressor(features[start:end].to(device)).cpu())
            if progress is not None:
                progress(end, len(features))
    raw = torch.cat(pieces).detach()
    targets = leave_one_out_codes(raw)
    receipt: dict[str, Any] = {
        "fit_split": "train",
        "groups": len(raw),
        "contexts_per_group": 10,
        "compressor_sha256": inputs.provenance["compressor_sha256"],
        "input_feature_sha256": tensor_fingerprint(features),
        "raw_sha256": tensor_fingerprint(raw),
        "targets_sha256": tensor_fingerprint(targets),
        "output_normalization": "none",
        "target": "Mean of nine OTHER raw compressed6D codes",
        "raw_mean_report_only": [float(value) for value in raw.double().mean((0, 1))],
        "raw_std_report_only": [float(value) for value in raw.double().std((0, 1), correction=0)],
        "arithmetic": "Compressor sourcefloat32; nonself mean float64, cast back to source dtype",
    }
    receipt["identity"] = canonical_fingerprint(receipt)
    verify_sources(inputs)
    return CompressedTargets(raw, targets, receipt)
