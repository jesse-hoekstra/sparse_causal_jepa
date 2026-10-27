"""Bind a supervised accurate MLP predictor to fresh E without exposing training labels."""

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.encoder_gradient_control.source import SourceBundle, build_source_model
from diagnostics.group_latent_control.run import initialize_control
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.mlp_oracle_code_control.model import MLPPredictor
from diagnostics.mlp_oracle_code_control.run import code_identity as oracle_code_identity
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.run import sha
from diagnostics.oracle_code_control.evaluation import PARAMETER_STD
from diagnostics.oracle_code_control.run import fixed_hashes as oracle_fixed_hashes
from diagnostics.table_settling_control.source import authenticated_directory
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import Spartan
from scjepa.training.rng import preserve_rng

SELECTED_ORACLE_HASHES = {
    "last.pt": "21ba81f3140ab85b647967c0c40d5a0284d76dae482214f9d585406d61633192",
    "protocol.json": "025f69326d8440c4c9f5d68a1eb3edb8fc983dc23c83f3076541363c52451d07",
    "completed.json": "be50790ab1d17ec2c82e1232e7e2f0f8187ce21c01e6e5066a296213400c92ee",
}


@dataclass(frozen=True)
class Inputs:
    """Label-free cached inputs plus fresh E and an explicitly supervised frozen predictor."""

    bundle: SourceBundle
    model: SCJEPA
    initialization: dict[str, Any]
    hashes: dict[str, str]
    predictor_provenance: dict[str, Any]


def fixed_hashes(model: SCJEPA) -> dict[str, str]:
    """Only E may change; bind both frozen state maps and the actual trained predictor."""
    return {
        name: module_fingerprint(getattr(model, name))
        for name in ("online_encoder", "target_encoder", "predictor")
    }


def _bind_ancestors(
    protocol: dict[str, Any], source: Path, source_hashes: dict[str, str]
) -> dict[str, str]:
    """Verify oracle training ancestry by bytes, never deserializing its theta payload."""
    declared_source = Path(protocol["config"]["source_dir"])
    declared_audit = Path(protocol["config"]["audit_dir"])
    if not declared_source.is_absolute() or not declared_audit.is_absolute():
        raise ValueError("Oracle ancestry paths must be absolute")
    completion = authenticated_directory(
        declared_audit, {"protocol.json", "results.json", "audit_inputs.pt"}
    )
    expected = {
        **{str(declared_source / name): value for name, value in source_hashes.items()},
        **{str(declared_audit / name): value for name, value in completion["hashes"].items()},
        str(declared_audit / "completed.json"): sha(declared_audit / "completed.json"),
    }
    if protocol["source_hashes"] != expected:
        raise ValueError("Oracle predictor was not trained on the exact original source/cache")
    hashes = {str((source / name).resolve()): value for name, value in source_hashes.items()}
    hashes.update(
        {
            str((declared_audit / name).resolve()): value
            for name, value in completion["hashes"].items()
        }
    )
    hashes[str((declared_audit / "completed.json").resolve())] = expected[
        str(declared_audit / "completed.json")
    ]
    return hashes


def load_inputs(
    source: Path,
    predictor_source: Path,
    seed: int,
    *,
    training_source_root: Path | None = None,
    verification: bool = False,
) -> Inputs:
    """Restore exact oracle P and archived fresh E; do not load any training-theta tensor."""
    source, predictor_source = source.resolve(), predictor_source.resolve()
    if not verification and any(
        sha(predictor_source / name) != expected
        for name, expected in SELECTED_ORACLE_HASHES.items()
    ):
        raise ValueError("Oracle differs from the selected completed accurate predictor")
    completion = authenticated_directory(
        predictor_source, {"protocol.json", "results.json", "last.pt", "initial_predictor.pt"}
    )
    protocol = cast(dict[str, Any], read_json(predictor_source / "protocol.json"))
    config = protocol["config"]
    evidence = {
        key: protocol[key]
        for key in (
            "settings",
            "source_hashes",
            "source_initialization",
            "mlp_initialization",
            "source_predictor_parameter_count",
            "frozen_hashes",
            "code_identity",
            "torch",
        )
    }
    settings = dict(config)
    for key in (
        "source_dir",
        "audit_dir",
        "output_dir",
        "training_source_root",
        "wandb_mode",
        "wandb_entity",
    ):
        settings.pop(key)
    if (
        canonical_fingerprint(evidence) != protocol["identity"]
        or settings != protocol["settings"]
        or protocol["code_identity"] != oracle_code_identity()
        or config["seed"] != seed
        or config["verification"] != verification
        or (verification and not 1 <= config["steps"] <= 3)
        or (not verification and (config["steps"], seed) != (30000, 202610261301))
        or protocol["diagnostic_physical_labels_in_training"] is not True
        or protocol["parameter_population_std"] != PARAMETER_STD
        or protocol["objective"]
        != (
            "Mean two-step forecast MSE only. No attention logits exist; omitting "
            "the SPARTAN logit regularizer is an explicit comparison confound. "
            "No SIGReg, consistency gradients, or optimizer history."
        )
    ):
        raise ValueError(
            "Oracle identity, protocol, supervision or code-coordinate contract differs"
        )
    bundle = load_primary_source(
        source,
        seed,
        expected_step=2 if verification else 30000,
        **({"expected_checkpoint_sha256": None} if verification else {}),
        training_source_root=training_source_root,
    )
    model, discarded_optimizer = build_source_model(bundle, torch.device("cpu"))
    del discarded_optimizer
    original_parameter_count = sum(parameter.numel() for parameter in model.predictor.parameters())
    discarded_table, initialization, source_hashes = initialize_control(model, bundle, "fresh_zero")
    del discarded_table
    if (
        protocol["source_initialization"] != initialization
        or protocol["source_predictor_parameter_count"] != original_parameter_count
        or protocol["frozen_hashes"] != oracle_fixed_hashes(model)
    ):
        raise ValueError("Oracle archived initialization or frozen state maps differ from source")
    hashes = _bind_ancestors(protocol, source, source_hashes)
    # CPU-only seeding avoids changing CUDA generators during source authentication.
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(seed)
        fresh_predictor = MLPPredictor().to(next(model.parameters()))
    mlp_initialization = protocol["mlp_initialization"]
    if (
        mlp_initialization["seed"] != seed
        or mlp_initialization["parameter_count"] != 52123
        or mlp_initialization["module_sha256"] != module_fingerprint(fresh_predictor)
    ):
        raise ValueError("Oracle MLP initialization architecture or seed differs")
    initial = cast(
        dict[str, Any],
        torch.load(
            predictor_source / "initial_predictor.pt", map_location="cpu", weights_only=False
        ),
    )
    initial_copy = copy.deepcopy(fresh_predictor)
    initial_copy.load_state_dict(initial["predictor"], strict=True)
    if (
        initial["identity"] != protocol["identity"]
        or initial["initialization"] != mlp_initialization
        or module_fingerprint(initial_copy) != mlp_initialization["module_sha256"]
    ):
        raise ValueError("Oracle initial predictor snapshot differs")
    model.predictor = cast(Spartan, fresh_predictor)
    candidate = copy.deepcopy(model)
    payload = cast(
        dict[str, Any],
        torch.load(predictor_source / "last.pt", map_location="cpu", weights_only=False),
    )
    result = cast(dict[str, Any], read_json(predictor_source / "results.json"))
    if (
        payload["identity"] != completion["identity"]
        or payload["step"] != config["steps"]
        or result["step"] != config["steps"]
    ):
        raise ValueError("Oracle predictor is not its completed fixed-budget endpoint")
    candidate.load_state_dict(payload["model"], strict=True)
    # Direct tensor checks prevent a self-consistent forged metadata/hash pair in fixtures.
    for name in ("online_encoder", "target_encoder", "parameter_encoder"):
        actual = cast(dict[str, Tensor], getattr(candidate, name).state_dict())
        expected = cast(dict[str, Tensor], getattr(model, name).state_dict())
        if set(actual) != set(expected) or any(
            actual[key].dtype != value.dtype or not torch.equal(actual[key], value)
            for key, value in expected.items()
        ):
            raise ValueError(
                "Oracle checkpoint changed original state maps or archived fresh encoder"
            )
    if candidate.get_extra_state() != model.get_extra_state() or any(
        not bool(torch.isfinite(value).all()) for value in candidate.predictor.state_dict().values()
    ):
        raise ValueError("Oracle rollout settings or predictor values differ")
    model.predictor.load_state_dict(candidate.predictor.state_dict(), strict=True)
    model.requires_grad_(False).eval()
    hashes.update(
        {str(predictor_source / name): value for name, value in completion["hashes"].items()}
    )
    hashes[str(predictor_source / "completed.json")] = sha(predictor_source / "completed.json")
    provenance = {
        "source_directory": str(predictor_source),
        "identity": completion["identity"],
        "checkpoint_sha256": sha(predictor_source / "last.pt"),
        "step": payload["step"],
        "protocol_sha256": sha(predictor_source / "protocol.json"),
        "predictor_sha256": module_fingerprint(model.predictor),
        "frozen_hashes": fixed_hashes(model),
        "rollout_flags": model.get_extra_state(),
        "source_initialization": initialization,
        "predictor_training_used_physical_labels": True,
        "new_encoder_training_parameter_labels_absent": True,
        "predictor_training_code": "theta / sqrt(.1875), no learned encoder",
        "parameter_population_std": PARAMETER_STD,
        "predictor_training_code_support": [-math.sqrt(3), math.sqrt(3)],
        "encoder_output_transform": (
            "identity; no Gaussianization, scaling, clipping or learned map"
        ),
        "distribution_caveat": (
            "P was fitted to independent standardized Uniform parameters, not Gaussian codes. "
            "SIGReg favors standard Gaussian E outputs; equal unit variance does not remove "
            "this distribution mismatch or protect P from out-of-support codes."
        ),
        "scope": (
            "Privileged accurate-predictor diagnostic: E loss uses no theta labels, but P "
            "was trained with them. Success is not label-free end-to-end acquisition."
        ),
        "audit_payload_policy": (
            "Historical theta payload hashed for ancestry, never deserialized here"
        ),
    }
    inputs = Inputs(bundle, model, initialization, hashes, provenance)
    verify_sources(inputs)
    return inputs


def verify_sources(inputs: Inputs) -> None:
    """Recheck source bytes and frozen P/state maps while permitting E to learn."""
    if any(sha(Path(name)) != expected for name, expected in inputs.hashes.items()):
        raise ValueError("Immutable accurate-predictor source changed")
    if (
        fixed_hashes(inputs.model) != inputs.predictor_provenance["frozen_hashes"]
        or inputs.model.get_extra_state() != inputs.predictor_provenance["rollout_flags"]
        or any(
            parameter.requires_grad
            for module in (
                inputs.model.online_encoder,
                inputs.model.target_encoder,
                inputs.model.predictor,
            )
            for parameter in module.parameters()
        )
    ):
        raise ValueError("Accurate predictor, state maps or rollout settings changed")
