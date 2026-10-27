"""Bind shared-feature endpoints and a precision cache through their common frozen state charts."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from diagnostics.global_code_calibration.objective import check_train
from diagnostics.global_code_calibration.source import verify_files as verify_files
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiquery_control.run import sha
from diagnostics.overcomplete_acquisition.model import OvercompletePredictor
from diagnostics.overcomplete_use_precision.run import ENDPOINT_SHA256
from diagnostics.overcomplete_use_precision.run import code_identity as precision_code_identity
from diagnostics.shared_feature_acquisition.run import code_identity as shared_code_identity
from diagnostics.table_settling_control.source import authenticated_directory, model_fingerprint
from scjepa.data import simplified_gaussian as dgp
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import Spartan
from scjepa.training.factory import build_model

PINNED = {
    202610261301: "5daecb051d2243fa5c747704bea7fa3b40836e3255180b13966124cb5a6b2301",
    202610262301: "2a40a7abdb6ec6953e005ab252a56ed8eeb87f55b8340a75d3de2797f51884ca",
}
EVALUATION_SEED = 202610261301


@dataclass(frozen=True)
class Inputs:
    """TRAIN-only fitting inputs; compatibility is checked before held-out tensors are loaded."""

    model: SCJEPA
    train: TensorSplit
    protocol: dict[str, Any]
    archived_validation: dict[str, Any]
    precision_protocol: dict[str, Any]
    hashes: dict[str, str]
    original_model_sha256: str
    evaluation_seed: int
    cache_provenance: dict[str, Any]


def load_inputs(
    endpoint: Path, source: Path, precision: Path, *, seed: int, verification: bool = False
) -> Inputs:
    """Authenticate both model ancestry and state-only cache reuse without loading labels."""
    completion = authenticated_directory(endpoint, {"protocol.json", "results.json", "last.pt"})
    protocol = cast(dict[str, Any], read_json(endpoint / "protocol.json"))
    if protocol["code_identity"] != shared_code_identity() or protocol["config"]["seed"] != seed:
        raise ValueError("Shared-feature code identity or training seed differs")
    if not verification and (seed not in PINNED or sha(endpoint / "last.pt") != PINNED[seed]):
        raise ValueError("Require one of the two selected completed100k shared endpoints")
    evaluation_seed = int(protocol["config"]["evaluation_seed"])
    if not verification and evaluation_seed != EVALUATION_SEED:
        raise ValueError("Original evaluation seed must remain fixed across training seeds")
    precision_completion = authenticated_directory(
        precision, {"protocol.json", "results.json", "validation_data.pt", "cache_receipt.json"}
    )
    precision_protocol = cast(dict[str, Any], read_json(precision / "protocol.json"))
    if precision_protocol["code_identity"] != precision_code_identity():
        raise ValueError("Original precision evaluation code changed")
    original_endpoint = Path(precision_protocol["config"]["endpoint_dir"])
    original_protocol_path = original_endpoint / "protocol.json"
    original_protocol = cast(dict[str, Any], read_json(original_protocol_path))
    original_endpoint_hash = precision_protocol["source_hashes"].get(
        str((original_endpoint / "last.pt").resolve())
    )
    if original_protocol["identity"] != precision_protocol["source_identity"] or precision_protocol[
        "source_hashes"
    ].get(str(original_protocol_path.resolve())) != sha(original_protocol_path):
        raise ValueError("Precision cache original endpoint provenance changed")
    if not verification and (
        original_endpoint_hash != ENDPOINT_SHA256
        or (precision_protocol["config"]["groups"], precision_protocol["config"]["seed"])
        != (8192, 202610271901)
    ):
        raise ValueError("Require the original authenticated8192-group precision cache")
    if original_protocol["frozen_hashes"] != protocol["frozen_hashes"]:
        raise ValueError("Precision cache and shared endpoint have different frozen state charts")
    hashes: dict[str, str] = {}
    for root, completed in ((endpoint, completion), (precision, precision_completion)):
        hashes.update(
            {str((root / name).resolve()): value for name, value in completed["hashes"].items()}
        )
        hashes[str((root / "completed.json").resolve())] = sha(root / "completed.json")
    # These are original precision input artifacts, not predictions from the new E/P.
    for name, expected in precision_protocol["source_hashes"].items():
        if sha(Path(name)) != expected:
            raise ValueError("Original precision input provenance changed")
        hashes[str(Path(name).resolve())] = expected
    declared_source = Path(protocol["config"]["source_dir"])
    precision_source = Path(precision_protocol["config"]["source_dir"])
    for name in ("protocol.json", "train_inputs.pt", "evaluation_data.pt"):
        expected = protocol["source_hashes"].get(str((declared_source / name).resolve()))
        if expected is None or sha(source / name) != expected:
            raise ValueError("Shared endpoint original TRAIN/state-cache source changed")
        if (
            name in ("protocol.json", "evaluation_data.pt")
            and precision_protocol["source_hashes"].get(str((precision_source / name).resolve()))
            != expected
        ):
            raise ValueError("Precision cache has a different DGP/observation-map source")
        hashes[str((source / name).resolve())] = expected
    payload = cast(
        dict[str, Any], torch.load(endpoint / "last.pt", map_location="cpu", weights_only=False)
    )
    expected_step = protocol["config"]["steps"] if verification else 100000
    if (
        payload["identity"] != protocol["identity"]
        or payload["step"] != expected_step
        or payload["records"][-1]["step"] != expected_step
    ):
        raise ValueError("Shared endpoint identity or final validation budget differs")
    source_protocol = cast(dict[str, Any], read_json(source / "protocol.json"))
    config = copy.deepcopy(source_protocol["model_config"])
    config["param_dim"] = 12
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        model = build_model(cast(DictConfig, OmegaConf.create(config)))
        model.predictor = cast(Spartan, OvercompletePredictor())
    model.load_state_dict(payload["model"], strict=True)
    for name, expected in protocol["frozen_hashes"].items():
        if module_fingerprint(getattr(model, name)) != expected:
            raise ValueError("Shared endpoint frozen state tensor bytes changed")
    original_hash = model_fingerprint(model)
    train = cast(
        TensorSplit, torch.load(source / "train_inputs.pt", map_location="cpu", weights_only=False)
    )
    check_train(train)
    if not verification and len(train["online"]) != 8192:
        raise ValueError("Require all8192 original TRAIN groups")
    provenance = {
        "precision_original_endpoint_identity": original_protocol["identity"],
        "shared_endpoint_identity": protocol["identity"],
        "frozen_hashes": protocol["frozen_hashes"],
        "common_source_protocol_sha256": sha(source / "protocol.json"),
        "common_original_evaluation_sha256": sha(source / "evaluation_data.pt"),
        "interpretation": (
            "Precision inputs depend on the DGP/observation map and frozen "
            "state charts, not the old parameter encoder or predictor. New source8192 "
            "responses must be measured, not replayed from the old E/P."
        ),
    }
    return Inputs(
        model.requires_grad_(False).double().eval(),
        train,
        protocol,
        payload["records"][-1]["response"],
        precision_protocol,
        hashes,
        original_hash,
        evaluation_seed,
        provenance,
    )


def validate_precision_data(data: EvaluationData, old: EvaluationData, inputs: Inputs) -> None:
    """After fitting, check actual cache law/chart metadata before any new prediction."""
    if set(data) != {"metadata", "validation"}:
        raise ValueError("Precision input must contain only its fixed validation sample")
    metadata, original = data["metadata"], old["metadata"]
    for name in (
        "observation_map_fingerprint",
        "views_per_group",
        "context_length",
        "trajectory_length",
    ):
        if metadata[name] != original[name]:
            raise ValueError(f"Precision cache law differs: {name}")
    if metadata["generator_version"] != dgp.GENERATOR_VERSION or not torch.equal(
        metadata["observation_matrix"], original["observation_matrix"]
    ):
        raise ValueError("Precision generator or observation matrix differs")
    if (
        int(metadata["seed"]) != inputs.precision_protocol["config"]["seed"]
        or len(data["validation"]["theta"]) != inputs.precision_protocol["config"]["groups"]
    ):
        raise ValueError("Precision seed or fixed group count differs")
    for name in ("online", "target"):
        if metadata[f"state_{name}_sha256"] != inputs.protocol["frozen_hashes"][f"{name}_encoder"]:
            raise ValueError("Precision cache state fingerprints differ")
        if data["validation"][f"states_{name}"].dtype != torch.float32:
            raise ValueError("Precision cache must preserve original float32 encoding policy")
