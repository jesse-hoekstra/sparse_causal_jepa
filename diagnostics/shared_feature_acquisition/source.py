"""Restore authenticated initial modules, then fit one TRAIN-only output projection."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch.nn import functional as F

from diagnostics.encoder_gain_acquisition.model import apply_encoder_gains
from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.group_latent_control.run import initialize_control
from diagnostics.joint_acquisition.source import SourceBundle, load_primary_source
from diagnostics.mlp_oracle_code_control.model import initialize_mlp
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.run import sha
from diagnostics.overcomplete_acquisition.model import initialize_overcomplete
from diagnostics.overcomplete_acquisition.run import code_identity as first_code_identity
from diagnostics.overcomplete_replication.run import code_identity as second_code_identity
from diagnostics.overcomplete_replication.source import (
    SOURCE_SEED,
    TRAINING_SEED,
    seeded_original_model,
)
from diagnostics.shared_feature_acquisition.projection import (
    Projection,
    apply_projection,
    fit_projection,
    hidden_features,
)
from diagnostics.table_settling_control.source import authenticated_directory
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.training.rng import preserve_rng

PINNED = {
    SOURCE_SEED: {
        "initial_modules.pt": "8b7e3c24213dc6c8a8abbeaad55ba5c7c5a50eb1d7e35f92f4834455fb47dfc1",
        "protocol.json": "3b47b97f87cd7fe50d94174ed10413958e15fabfe6e2426533fdab30c775497f",
        "completed.json": "1d452f94a724aa6e802312aa6e018320756adf734d0a351fe75b9b5df684783f",
    },
    TRAINING_SEED: {
        "initial_modules.pt": "173071340e20d089abec9d02d37b5a99424762d7c2a6adab9fd0378972ffc3a9",
        "protocol.json": "022b7366cf48a859a261ae41f30fcec09387b946a6c30c695d3c385d9e2b6b48",
        "completed.json": "8f686b7ed2517c9ec1412fac58286d021cf74f72cd2ebb3cb93be722216564d1",
    },
}
AUDIT_HASHES = {
    SOURCE_SEED: "9e2f35501d577486301f8d5077b3c2a85376c1540a12d3d882e817e2dd9283f2",
    TRAINING_SEED: "abfe99f4e30b6c018f2bfd5459656ce932b90183996daf9a798a4074e753738f",
}


@dataclass(frozen=True)
class Inputs:
    """Prepared fresh modules and immutable ancestry, never trained parent E/P weights."""

    bundle: SourceBundle
    model: SCJEPA
    hashes: dict[str, str]
    provenance: dict[str, Any]
    projection: Projection


@torch.no_grad()
def compare_audit(
    model: SCJEPA, projection: Projection, bundle: SourceBundle, path: Path, seed: int
) -> dict[str, Any]:
    """Check numerical identity across CPU backends, without evaluation or label thresholds."""
    reference = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))
    if (
        reference["seed"] != seed
        or reference["train_only"] is not True
        or reference["source_initial_encoder_sha256"] != projection.receipt["source_encoder_sha256"]
    ):
        raise ValueError("Audit projection belongs to a different initialization")
    differences: dict[str, float] = {}
    for name, actual in (("weight", projection.weight), ("bias", projection.bias)):
        expected = reference[name]
        if expected.shape != actual.shape or not bool(torch.isfinite(expected).all()):
            raise ValueError("Invalid audited projection tensor")
        denominator = expected.double().square().sum().sqrt().clamp_min(1e-30)
        relative = float((actual.double() - expected.double()).square().sum().sqrt() / denominator)
        differences[name + "_relative_frobenius_difference"] = relative
        if relative > 1e-4:
            raise ValueError("Recomputed projection exceeds audited relative numerical tolerance")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        features = hidden_features(
            cast(TransitionParameterEncoder, model.parameter_encoder),
            {name: value[:103] for name, value in bundle.train.items()},
        )
    finally:
        torch.set_num_threads(old_threads)
    flat = features.flatten(0, 1)[:1024].float()
    actual_codes = F.linear(flat, projection.weight, projection.bias)
    reference_codes = F.linear(flat, reference["weight"], reference["bias"])
    difference = float((actual_codes.double() - reference_codes.double()).square().mean().sqrt())
    tolerance = 1e-4 * projection.receipt["original_code_centered_rms"] + 1e-7
    if difference > tolerance:
        raise ValueError("Recomputed projection code RMS exceeds audited numerical tolerance")
    return {
        **differences,
        "first_contexts": len(flat),
        "code_rms_difference": difference,
        "code_rms_tolerance": tolerance,
        "relative_tolerance": 1e-4,
        "audit_file_sha256": sha(path),
    }


def load_inputs(
    source: Path,
    initial_source: Path,
    seed: int,
    *,
    source_seed: int = SOURCE_SEED,
    audit_projection: Path | None = None,
    training_source_root: Path | None = None,
    verification: bool = False,
) -> Inputs:
    """Authenticate completed q12 ancestry, reconstruct exact initial tensors and fit TRAIN."""
    parent = initial_source.resolve()
    if not verification:
        if source_seed != SOURCE_SEED or seed not in PINNED or audit_projection is None:
            raise ValueError("Require one of two declared seeds and its fixed audit projection")
        if any(sha(parent / name) != value for name, value in PINNED[seed].items()):
            raise ValueError("Initial source is not the declared completed q12 parent")
        if sha(audit_projection) != AUDIT_HASHES[seed]:
            raise ValueError("Audited projection file changed")
    completion = authenticated_directory(parent, {"initial_modules.pt", "protocol.json"})
    protocol = cast(dict[str, Any], read_json(parent / "protocol.json"))
    config = protocol["config"]
    first = seed == source_seed
    evidence_keys = [
        "settings",
        "source_hashes",
        "source_initialization",
        "initial_encoder_sha256",
        "frozen_hashes",
        "code_identity",
        "torch",
    ]
    evidence_keys += (
        [
            "mlp_initialization",
            "encoder_gain_initialization",
            "overcomplete_initialization",
            "source_predictor_parameter_count",
        ]
        if first
        else ["replication_initialization", "initial_predictor_sha256"]
    )
    if (
        canonical_fingerprint({key: protocol[key] for key in evidence_keys}) != protocol["identity"]
        or config["seed"] != seed
        or config["verification"] != verification
        or protocol["code_identity"] != (first_code_identity() if first else second_code_identity())
        or (not first and (config["arm"] != "gain12" or config["source_seed"] != source_seed))
        or (not verification and config["steps"] != 100000)
        or (verification and not 1 <= config["steps"] <= 3)
    ):
        raise ValueError("Parent identity, seed, architecture or fixed budget differs")
    bundle = load_primary_source(
        source,
        source_seed,
        expected_step=2 if verification else 30000,
        **({"expected_checkpoint_sha256": None} if verification else {}),
        training_source_root=training_source_root,
    )
    model, discarded_optimizer = build_source_model(bundle, torch.device("cpu"))
    del discarded_optimizer
    table, initial_receipt, source_hashes = initialize_control(model, bundle, "fresh_zero")
    del table
    if (
        source_hashes != protocol["source_hashes"]
        or initial_receipt != protocol["source_initialization"]
    ):
        raise ValueError("Parent uses different original caches or archived source initialization")
    if not first:
        fresh = seeded_original_model(bundle.protocol["model_config"], seed)
        model.parameter_encoder.load_state_dict(fresh.parameter_encoder.state_dict(), strict=True)
    with preserve_rng():
        initialize_mlp(model, seed)
    apply_encoder_gains(model)
    initialize_overcomplete(model, seed)
    initial = cast(
        dict[str, Any],
        torch.load(parent / "initial_modules.pt", map_location="cpu", weights_only=False),
    )
    if initial["identity"] != protocol["identity"]:
        raise ValueError("Parent initial modules identity differs")
    for name in ("parameter_encoder", "predictor"):
        current = getattr(model, name)
        if current.state_dict().keys() != initial[name].keys() or any(
            not torch.equal(value, initial[name][key])
            for key, value in current.state_dict().items()
        ):
            raise ValueError(
                "Reconstructed module differs from parent's actual saved initialization"
            )
        current.load_state_dict(initial[name], strict=True)
    frozen = {
        name: module_fingerprint(getattr(model, name))
        for name in ("online_encoder", "target_encoder")
    }
    if frozen != protocol["frozen_hashes"]:
        raise ValueError("Parent initialization uses different frozen state maps")
    initial_hashes = {
        name: module_fingerprint(getattr(model, name))
        for name in ("parameter_encoder", "predictor")
    }
    projection = fit_projection(
        cast(TransitionParameterEncoder, model.parameter_encoder), bundle.train
    )
    numerical_audit = (
        compare_audit(model, projection, bundle, audit_projection, seed)
        if audit_projection
        else None
    )
    application = apply_projection(model, projection)
    hashes = {
        **{str((source / name).resolve()): value for name, value in source_hashes.items()},
        **{str(parent / name): value for name, value in completion["hashes"].items()},
        str(parent / "completed.json"): sha(parent / "completed.json"),
    }
    if audit_projection:
        hashes[str(audit_projection.resolve())] = sha(audit_projection)
    provenance = {
        "source_seed": source_seed,
        "training_seed": seed,
        "parent_identity": protocol["identity"],
        "parent_initial_hashes": initial_hashes,
        "parent_initial_file_sha256": sha(parent / "initial_modules.pt"),
        "source_initialization": initial_receipt,
        "frozen_hashes": frozen,
        "rollout_flags": model.get_extra_state(),
        "projection_application": application,
        "numerical_audit": numerical_audit,
        "optimizer": "Fresh Adam; no parent moments",
        "training_parameter_labels_absent": True,
        "scope": (
            "TRAIN-fitted initial output projection only; original hidden/P weights "
            "and per-context forward retained"
        ),
    }
    inputs = Inputs(bundle, model, hashes, provenance, projection)
    verify_sources(inputs)
    return inputs


def verify_sources(inputs: Inputs) -> None:
    """Preserve every ancestor file and frozen state map; learned modules may train."""
    if any(sha(Path(name)) != value for name, value in inputs.hashes.items()):
        raise ValueError("Immutable shared-feature source changed")
    if inputs.model.get_extra_state() != inputs.provenance["rollout_flags"] or any(
        module_fingerprint(getattr(inputs.model, name)) != value
        for name, value in inputs.provenance["frozen_hashes"].items()
    ):
        raise ValueError("Frozen state maps or rollout flags changed")
