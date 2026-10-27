"""Expose only factual TRAIN tensors while authenticating a completed six-code endpoint."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from omegaconf import DictConfig, OmegaConf

from diagnostics.global_code_calibration.objective import check_train
from diagnostics.global_code_calibration.source import verify_files
from diagnostics.mlp_oracle_code_control.model import MLPPredictor
from diagnostics.multiquery_control.data import TensorSplit
from diagnostics.multiquery_control.run import sha
from diagnostics.overcomplete_use_precision.run import code_identity as precision_code_identity
from diagnostics.predictor_transfer_control.run import code_identity as transfer_code_identity
from diagnostics.predictor_transfer_control.source import (
    AUDIT_HASHES,
    ENDPOINT_HASHES,
    SEED,
    fixed_hashes,
)
from diagnostics.predictor_transfer_precision.source import PRECISION_CACHE_SHA256
from diagnostics.table_settling_control.source import authenticated_directory, model_fingerprint
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import Spartan
from scjepa.training.factory import build_model

Arm = Literal["spartan", "mlp"]


@dataclass(frozen=True)
class Inputs:
    """Immutable six-code model and original label-free TRAIN cache only."""

    model: SCJEPA
    train: TensorSplit
    hashes: dict[str, str]
    protocol: dict[str, Any]
    original_model_sha256: str


def load_training(
    endpoint: Path,
    source: Path,
    precision: Path,
    *,
    arm: Arm,
    checkpoint_sha256: str,
    verification: bool = False,
) -> Inputs:
    """Check complete source/cache hashes but deserialize no held-out input cache before fitting."""
    completed = authenticated_directory(
        endpoint, {"protocol.json", "last.pt", "initial_modules.pt"}
    )
    protocol = cast(dict[str, Any], read_json(endpoint / "protocol.json"))
    config, provenance = protocol["config"], protocol["provenance"]
    if (
        protocol["identity"] != completed["identity"]
        or protocol["code_identity"] != transfer_code_identity()
    ):
        raise ValueError("Completed transfer identity or implementation changed")
    if (
        config["arm"] != arm
        or config["verification"] != verification
        or arm not in ("spartan", "mlp")
    ):
        raise ValueError("Require one of both declared completed predictor arms")
    if sha(endpoint / "last.pt") != checkpoint_sha256:
        raise ValueError("Require the predeclared final-checkpoint SHA")
    if not verification and (config["steps"], config["seed"]) != (100000, SEED):
        raise ValueError("Require canonical fixed100k transfer endpoint")
    audit = Path(config["pca_audit_dir"])
    if not verification and (
        provenance["source_endpoint_sha256"] != ENDPOINT_HASHES["last.pt"]
        or any(sha(audit / name) != value for name, value in AUDIT_HASHES.items())
    ):
        raise ValueError("Canonical learned source/PCA chart changed")
    verify_files(protocol["source_hashes"])
    hashes = dict(protocol["source_hashes"])
    hashes.update(
        {str((endpoint / name).resolve()): value for name, value in completed["hashes"].items()}
    )
    hashes[str((endpoint / "completed.json").resolve())] = sha(endpoint / "completed.json")
    for name in ("protocol.json", "train_inputs.pt", "evaluation_data.pt"):
        declared = Path(config["source_dir"]) / name
        if protocol["source_hashes"].get(str(declared.resolve())) != sha(source / name):
            raise ValueError("Original transfer TRAIN/source/cache changed")
        hashes[str((source / name).resolve())] = sha(source / name)
    precision_complete = authenticated_directory(
        precision, {"protocol.json", "validation_data.pt", "results.json"}
    )
    precision_protocol = cast(dict[str, Any], read_json(precision / "protocol.json"))
    if (
        precision_complete["identity"] != precision_protocol["identity"]
        or precision_protocol["code_identity"] != precision_code_identity()
    ):
        raise ValueError("Original precision cache identity/implementation changed")
    if not verification and sha(precision / "validation_data.pt") != PRECISION_CACHE_SHA256:
        raise ValueError("Require the exact existing precision validation cache")
    verify_files(precision_protocol["source_hashes"])
    hashes.update(precision_protocol["source_hashes"])
    hashes.update(
        {
            str((precision / name).resolve()): value
            for name, value in precision_complete["hashes"].items()
        }
    )
    hashes[str((precision / "completed.json").resolve())] = sha(precision / "completed.json")
    source_protocol = cast(dict[str, Any], read_json(source / "protocol.json"))
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        model = build_model(cast(DictConfig, OmegaConf.create(source_protocol["model_config"])))
        if arm == "mlp":
            model.predictor = cast(Spartan, MLPPredictor())
    payload = cast(
        dict[str, Any], torch.load(endpoint / "last.pt", map_location="cpu", weights_only=False)
    )
    if (
        payload["identity"] != protocol["identity"]
        or payload["step"] != config["steps"]
        or payload["records"][-1]["step"] != config["steps"]
    ):
        raise ValueError("Require fixed-final checkpoint and its final validation record")
    model.load_state_dict(payload["model"], strict=True)
    if (
        fixed_hashes(model) != provenance["frozen_hashes"]
        or model.get_extra_state() != provenance["rollout_flags"]
    ):
        raise ValueError("Frozen E/state or attached rollout flags changed")
    if model.parameter_encoder.param_dim != 6 or model.predictor.param_dim != 6:
        raise ValueError("Require original six-code interface")
    if arm == "spartan" and not all(
        layer.dense and not layer.identity for layer in model.predictor.layers
    ):
        raise ValueError("Require original dense SPARTAN with all gates open")
    train = cast(
        TensorSplit, torch.load(source / "train_inputs.pt", map_location="cpu", weights_only=False)
    )
    check_train(train)
    if not verification and len(train["online"]) != 8192:
        raise ValueError("Require all8192 original TRAIN groups")
    model = model.requires_grad_(False).double().eval()
    return Inputs(model, train, hashes, protocol, model_fingerprint(model))
