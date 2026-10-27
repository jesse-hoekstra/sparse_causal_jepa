"""Read only the fixed distillation-boundary snapshot and authenticated TRAIN targets."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.run import sha
from diagnostics.multiteacher_curriculum.run import code_identity as curriculum_code_identity
from diagnostics.multiteacher_curriculum.source import (
    Inputs,
    load_inputs,
    normalize_targets,
    verify_sources,
)
from diagnostics.teacher_code_compression.model import initialize_modules
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.experiments.storage import read_json

REFERENCE_FILES = (
    "protocol.json",
    "initial_modules.pt",
    "target_manifest.json",
    "teacher_targets.pt",
    "distill_final.pt",
)


@dataclass(frozen=True)
class CompressionInputs:
    """Only code-derived tensors enter training; parent evaluation remains separate."""

    parent: Inputs
    train: Tensor
    targets: Tensor
    mean: Tensor
    scale: Tensor
    heads_state: dict[str, Tensor]
    reference: dict[str, Any]
    hashes: dict[str, str]


def load_reference(curriculum: Path, *, verification: bool = False) -> CompressionInputs:
    """Require immutable 30k-boundary artifacts, never mutable parent logs/checkpoints."""
    curriculum = curriculum.resolve()
    hashes = {str(curriculum / name): sha(curriculum / name) for name in REFERENCE_FILES}
    protocol = cast(dict[str, Any], read_json(curriculum / "protocol.json"))
    config = protocol["config"]
    evidence = {
        key: protocol[key]
        for key in (
            "settings",
            "source_hashes",
            "teacher_provenance",
            "source_initialization",
            "initial_encoder_sha256",
            "mlp_initialization",
            "heads_initialization",
            "state_hashes",
            "code_identity",
            "torch",
        )
    }
    settings = dict(config)
    for key in (
        "source_dir",
        "primary_teacher_dir",
        "fresh_teacher_dir",
        "mlp_teacher_dir",
        "output_dir",
        "training_source_root",
        "wandb_mode",
        "wandb_entity",
    ):
        settings.pop(key)
    if (
        canonical_fingerprint(evidence) != protocol["identity"]
        or settings != protocol["settings"]
        or protocol["code_identity"] != curriculum_code_identity()
        or config["verification"] != verification
        or protocol["training_parameter_labels_absent"] is not True
        or (
            not verification
            and (
                config["seed"] != 202610261301
                or config["distill_steps"] != 30000
                or config["groups"] != 64
            )
        )
        or (verification and not 1 <= config["distill_steps"] <= 3)
    ):
        raise ValueError("Curriculum protocol, code or fixed distillation settings differ")
    parent = load_inputs(
        Path(config["source_dir"]),
        Path(config["primary_teacher_dir"]),
        Path(config["fresh_teacher_dir"]),
        Path(config["mlp_teacher_dir"]),
        config["seed"],
        training_source_root=Path(config["training_source_root"])
        if config["training_source_root"]
        else None,
        verification=verification,
    )
    if (
        protocol["source_hashes"] != parent.hashes
        or protocol["teacher_provenance"] != parent.teacher_provenance
        or protocol["source_initialization"] != parent.initialization
    ):
        raise ValueError("Curriculum teacher or original-source ancestry differs")
    initial = cast(
        dict[str, Any],
        torch.load(curriculum / "initial_modules.pt", map_location="cpu", weights_only=False),
    )
    endpoint = cast(
        dict[str, Any],
        torch.load(curriculum / "distill_final.pt", map_location="cpu", weights_only=False),
    )
    manifest = cast(dict[str, Any], read_json(curriculum / "target_manifest.json"))
    payload = cast(
        dict[str, Any],
        torch.load(curriculum / "teacher_targets.pt", map_location="cpu", weights_only=False),
    )
    if (
        any(
            item["identity"] != protocol["identity"]
            for item in (initial, endpoint, manifest, payload)
        )
        or endpoint["step"] != config["distill_steps"]
        or endpoint["stage"] != "distill"
        or endpoint["target_manifest_sha256"] != sha(curriculum / "target_manifest.json")
        or manifest["sha256"] != sha(curriculum / "teacher_targets.pt")
        or payload["receipt"] != manifest["receipt"]
        or set(payload) != {"identity", "raw", "mean", "scale", "targets", "receipt"}
    ):
        raise ValueError("Distillation boundary does not authenticate the fixed teacher targets")
    reference_encoder = copy.deepcopy(parent.model.parameter_encoder)
    reference_encoder.load_state_dict(initial["encoder"], strict=True)
    _, heads, _ = initialize_modules(config["seed"], dtype=next(parent.model.parameters()).dtype)
    expected_heads = module_fingerprint(heads)
    heads.load_state_dict(initial["heads"], strict=True)
    if (
        module_fingerprint(reference_encoder) != protocol["initial_encoder_sha256"]
        or module_fingerprint(reference_encoder)
        != module_fingerprint(parent.model.parameter_encoder)
        or module_fingerprint(heads) != expected_heads
        or expected_heads != protocol["heads_initialization"]["module_sha256"]
    ):
        raise ValueError("Curriculum initial student or exact seeded head weights differ")
    raw = cast(Tensor, payload["raw"])
    if raw.shape != (parent.bundle.train_groups, 10, 3, 6) or raw.dtype != torch.float32:
        raise ValueError("Teacher inputs are not the original TRAIN individual-context union")
    rebuilt = normalize_targets(raw)
    for name in ("raw", "mean", "scale", "targets"):
        actual = cast(Tensor, payload[name])
        expected = cast(Tensor, getattr(rebuilt, name))
        if (
            actual.requires_grad
            or actual.dtype != expected.dtype
            or not torch.equal(actual, expected)
        ):
            raise ValueError("Saved teacher transform or LOO target arithmetic differs")
    receipt = payload["receipt"]
    if (
        canonical_fingerprint({key: value for key, value in receipt.items() if key != "identity"})
        != receipt["identity"]
        or any(
            receipt.get(key) != value for key, value in rebuilt.receipt.items() if key != "identity"
        )
        or receipt["train_inputs_sha256"] != parent.bundle.hashes["train_inputs.pt"]
        or receipt["teacher_encoders"]
        != [module_fingerprint(teacher) for teacher in parent.teachers]
    ):
        raise ValueError("Teacher TRAIN normalization receipt or encoder/cache binding differs")
    features = ((raw - rebuilt.mean) / rebuilt.scale).flatten(-2).detach().clone()
    reference = {
        "curriculum_directory": str(curriculum),
        "curriculum_identity": protocol["identity"],
        "distillation_step": endpoint["step"],
        "parent_completion_asserted": False,
        "immutable_files": {name: hashes[str(curriculum / name)] for name in REFERENCE_FILES},
        "seed": config["seed"],
        "device": config["device"],
        "threads": config["threads"],
        "groups": config["groups"],
        "teacher_provenance": parent.teacher_provenance,
        "teacher_target_receipt": receipt,
        "exact_initial_heads_sha256": expected_heads,
        "train_feature_sha256": tensor_fingerprint(features),
        "input_policy": (
            "Same context's three6D teacher vectors, stored TRAIN normalization, no averaging"
        ),
    }
    inputs = CompressionInputs(
        parent,
        features,
        rebuilt.targets,
        rebuilt.mean,
        rebuilt.scale,
        cast(dict[str, Tensor], initial["heads"]),
        reference,
        hashes,
    )
    verify_reference(inputs)
    return inputs


def verify_reference(inputs: CompressionInputs) -> None:
    """Reject mutation of the selected parent snapshot or original teacher/source artifacts."""
    verify_sources(inputs.parent)
    if any(sha(Path(name)) != expected for name, expected in inputs.hashes.items()):
        raise ValueError("Immutable distillation reference changed")
