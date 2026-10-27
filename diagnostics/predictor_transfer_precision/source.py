"""Restore one fixed transfer arm on the canonical common chart without reading TRAIN."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from omegaconf import DictConfig, OmegaConf

from diagnostics.global_code_calibration.source import verify_files
from diagnostics.mlp_oracle_code_control.model import MLPPredictor
from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.multiquery_control.run import sha
from diagnostics.overcomplete_use_precision.run import code_identity as precision_code_identity
from diagnostics.predictor_transfer_control.model import audit_model_fingerprint
from diagnostics.predictor_transfer_control.run import code_identity as transfer_code_identity
from diagnostics.predictor_transfer_control.source import (
    AUDIT_HASHES,
    ENDPOINT_HASHES,
    SEED,
    fixed_hashes,
)
from diagnostics.table_settling_control.source import authenticated_directory
from scjepa.data import simplified_gaussian as dgp
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import Spartan
from scjepa.training.factory import build_model

Arm = Literal["spartan", "mlp", "reference"]
PRECISION_CACHE_SHA256 = "0e4ed63400ad03cd71c9a1653f77267fa66d964ce38efb629970513d6657b2dc"


@dataclass(frozen=True)
class Inputs:
    """One fixed model, original validation replay and the common existing precision cache."""

    model: SCJEPA
    original_validation: EvaluationData
    precision_data: EvaluationData
    archived_report: dict[str, Any]
    hashes: dict[str, str]
    source_protocol: dict[str, Any]
    evaluation_seed: int
    original_bootstrap_samples: int
    chart_provenance: dict[str, Any]


def load_inputs(
    endpoint: Path,
    source: Path,
    precision: Path,
    *,
    arm: Arm,
    checkpoint_sha256: str,
    verification: bool = False,
) -> Inputs:
    """Bind final weights and shared E/state/projection directly to the original PCA artifact."""
    if arm not in ("spartan", "mlp", "reference"):
        raise ValueError("Require one of the three predeclared arms")
    completed = authenticated_directory(
        endpoint, {"protocol.json", "last.pt", "initial_modules.pt", "reference_validation.json"}
    )
    protocol = cast(dict[str, Any], read_json(endpoint / "protocol.json"))
    if (
        protocol["identity"] != completed["identity"]
        or protocol["code_identity"] != transfer_code_identity()
    ):
        raise ValueError("Completed transfer identity/code changed")
    config, provenance = protocol["config"], protocol["provenance"]
    expected_parent = "spartan" if arm == "spartan" else "mlp"
    if config["arm"] != expected_parent or config["verification"] != verification:
        raise ValueError("Transfer parent arm or verification role differs")
    if sha(endpoint / "last.pt") != checkpoint_sha256:
        raise ValueError("Fixed-final checkpoint SHA differs from its declared pin")
    if not verification and (
        config["steps"],
        config["seed"],
        config["groups"],
        config["bootstrap_samples"],
    ) != (100000, SEED, 64, 1000):
        raise ValueError("Require the original fixed100k transfer endpoint")
    verify_files(protocol["source_hashes"])
    hashes = dict(protocol["source_hashes"])
    hashes.update(
        {str((endpoint / name).resolve()): value for name, value in completed["hashes"].items()}
    )
    hashes[str((endpoint / "completed.json").resolve())] = sha(endpoint / "completed.json")
    for name in ("protocol.json", "evaluation_data.pt"):
        declared = Path(config["source_dir"]) / name
        if protocol["source_hashes"].get(str(declared.resolve())) != sha(source / name):
            raise ValueError("Original source law/cache differs from transfer inputs")
        hashes[str((source / name).resolve())] = sha(source / name)
    source_protocol = cast(dict[str, Any], read_json(source / "protocol.json"))
    payload = cast(
        dict[str, Any], torch.load(endpoint / "last.pt", map_location="cpu", weights_only=False)
    )
    initial = cast(
        dict[str, Any],
        torch.load(endpoint / "initial_modules.pt", map_location="cpu", weights_only=False),
    )
    if payload["identity"] != protocol["identity"] or initial["identity"] != protocol["identity"]:
        raise ValueError("Checkpoint/initialization identity differs")
    if payload["step"] != config["steps"] or payload["records"][-1]["step"] != config["steps"]:
        raise ValueError("Require last fixed-budget checkpoint and its final validation record")
    audit = Path(config["pca_audit_dir"])
    if not verification and (
        provenance["source_endpoint_sha256"] != ENDPOINT_HASHES["last.pt"]
        or any(sha(audit / name) != value for name, value in AUDIT_HASHES.items())
    ):
        raise ValueError("Require the original canonical PCA6 source and chart")
    for name in ("results.json", "measurements.pt"):
        if protocol["source_hashes"].get(str((audit / name).resolve())) != sha(audit / name):
            raise ValueError("PCA chart artifact differs from transfer provenance")
    projection = cast(
        dict[str, Any],
        torch.load(audit / "measurements.pt", map_location="cpu", weights_only=False),
    )
    for name in ("mean", "basis", "eigenvalues"):
        torch.testing.assert_close(initial["projection"][name], projection[name], rtol=0, atol=0)
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        model = build_model(cast(DictConfig, OmegaConf.create(source_protocol["model_config"])))
        reference = build_model(cast(DictConfig, OmegaConf.create(source_protocol["model_config"])))
        reference.predictor = cast(Spartan, MLPPredictor())
        if arm != "spartan":
            model.predictor = cast(Spartan, MLPPredictor())
    reference.load_state_dict(
        projection["composed_model"], strict=True
    )  # Exact float64->float32 cast.
    reference_hash = audit_model_fingerprint(reference)
    reference.load_state_dict(initial["reference_model"], strict=True)
    if (
        reference_hash != audit_model_fingerprint(reference)
        or reference_hash != provenance["reference_model_sha256"]
    ):
        raise ValueError("Saved reference differs from canonical composed float32 model")
    model.load_state_dict(
        initial["reference_model"] if arm == "reference" else payload["model"], strict=True
    )
    if (
        fixed_hashes(model) != provenance["frozen_hashes"]
        or fixed_hashes(reference) != fixed_hashes(model)
        or model.get_extra_state() != provenance["rollout_flags"]
    ):
        raise ValueError("Canonical frozen E/state or attached rollout flags changed")
    if model.parameter_encoder.param_dim != 6 or model.predictor.param_dim != 6:
        raise ValueError("Require the original six-code interface")
    if arm == "spartan" and not all(
        layer.dense and not layer.identity for layer in model.predictor.layers
    ):
        raise ValueError("Original dense SPARTAN gates must remain open")
    if arm == "reference":
        archived = cast(dict[str, Any], read_json(endpoint / "reference_validation.json"))
        if archived["identity"] != protocol["identity"]:
            raise ValueError("Reference validation belongs to another run")
        report = archived["metrics"]["response"]
    else:
        report = payload["records"][-1]["response"]
    completion = authenticated_directory(
        precision, {"protocol.json", "validation_data.pt", "results.json"}
    )
    precision_protocol = cast(dict[str, Any], read_json(precision / "protocol.json"))
    if (
        precision_protocol["identity"] != completion["identity"]
        or precision_protocol["code_identity"] != precision_code_identity()
    ):
        raise ValueError("Original precision cache identity/code changed")
    if not verification and sha(precision / "validation_data.pt") != PRECISION_CACHE_SHA256:
        raise ValueError("Require the existing precision validation cache bytes")
    verify_files(precision_protocol["source_hashes"])
    hashes.update(precision_protocol["source_hashes"])
    hashes.update(
        {str((precision / name).resolve()): value for name, value in completion["hashes"].items()}
    )
    hashes[str((precision / "completed.json").resolve())] = sha(precision / "completed.json")
    old = cast(
        EvaluationData,
        torch.load(source / "evaluation_data.pt", map_location="cpu", weights_only=False),
    )
    data = cast(
        EvaluationData,
        torch.load(precision / "validation_data.pt", map_location="cpu", weights_only=False),
    )
    if set(data) != {"metadata", "validation"}:
        raise ValueError("Precision cache must be validation-only")
    for name in (
        "observation_map_fingerprint",
        "views_per_group",
        "context_length",
        "trajectory_length",
    ):
        if data["metadata"][name] != old["metadata"][name]:
            raise ValueError("Precision DGP/input geometry differs")
    if (
        not torch.equal(
            data["metadata"]["observation_matrix"], old["metadata"]["observation_matrix"]
        )
        or data["metadata"]["generator_version"] != dgp.GENERATOR_VERSION
    ):
        raise ValueError("Precision observation map/generator differs")
    frozen = fixed_hashes(model)
    for name in ("online", "target"):
        if (
            data["metadata"][f"state_{name}_sha256"] != frozen[f"{name}_encoder"]
            or old["metadata"][f"state_{name}_sha256"] != frozen[f"{name}_encoder"]
        ):
            raise ValueError("Precision cache and transfer frozen state charts differ")
        if data["validation"][f"states_{name}"].dtype != torch.float32:
            raise ValueError("Preserve original float32 state-cache encoding")
    if not verification and (len(data["validation"]["theta"]), data["metadata"]["seed"]) != (
        8192,
        202610271901,
    ):
        raise ValueError("Require original8192-group precision validation cache")
    original: EvaluationData = {
        "metadata": old["metadata"],
        "validation": {key: value[: config["groups"]] for key, value in old["validation"].items()},
    }
    chart = {
        "frozen_hashes": frozen,
        "reference_model_sha256": reference_hash,
        "pca_measurements_sha256": sha(audit / "measurements.pt"),
        "canonical_source_endpoint_sha256": provenance["source_endpoint_sha256"],
        "precision_cache_sha256": sha(precision / "validation_data.pt"),
    }
    return Inputs(
        model.requires_grad_(False).double().eval(),
        original,
        data,
        report,
        hashes,
        protocol,
        config["seed"],
        config["bootstrap_samples"],
        chart,
    )
