"""Authenticate the fixed all-six endpoint and expose only label-free TRAIN tensors to fitting."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from diagnostics.global_code_calibration.objective import check_train
from diagnostics.multiquery_control.data import TensorSplit, module_fingerprint
from diagnostics.multiquery_control.run import sha
from diagnostics.overcomplete_acquisition.model import OvercompletePredictor
from diagnostics.overcomplete_acquisition.run import code_identity as source_code_identity
from diagnostics.overcomplete_use_precision.run import ENDPOINT_SHA256
from diagnostics.table_settling_control.source import authenticated_directory, model_fingerprint
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import Spartan
from scjepa.training.factory import build_model


@dataclass(frozen=True)
class Inputs:
    """No held-out arrays or physical labels are loaded by this training source seam."""

    model: SCJEPA
    train: TensorSplit
    protocol: dict[str, Any]
    archived_validation: dict[str, Any]
    precision_protocol: dict[str, Any]
    hashes: dict[str, str]
    original_model_sha256: str


def load_inputs(
    endpoint: Path, source: Path, precision: Path, *, verification: bool = False
) -> Inputs:
    """Verify completed artifacts, read only TRAIN cache and restore frozen source values."""
    completion = authenticated_directory(endpoint, {"protocol.json", "results.json", "last.pt"})
    protocol = cast(dict[str, Any], read_json(endpoint / "protocol.json"))
    if protocol["code_identity"] != source_code_identity():
        raise ValueError("Completed source code differs from current dependencies")
    if not verification and sha(endpoint / "last.pt") != ENDPOINT_SHA256:
        raise ValueError("Require the selected all-six fixed100k endpoint")
    precision_completion = authenticated_directory(
        precision, {"protocol.json", "results.json", "validation_data.pt", "cache_receipt.json"}
    )
    precision_protocol = cast(dict[str, Any], read_json(precision / "protocol.json"))
    if precision_protocol["source_identity"] != protocol["identity"] or precision_protocol[
        "source_hashes"
    ].get(str((endpoint / "last.pt").resolve())) != sha(endpoint / "last.pt"):
        raise ValueError("Precision validation cache belongs to another source endpoint")
    if not verification and (
        precision_protocol["config"]["groups"],
        precision_protocol["config"]["seed"],
    ) != (8192, 202610271901):
        raise ValueError("Require the fixed independent precision validation sample")
    hashes: dict[str, str] = {}
    for root, completed in ((endpoint, completion), (precision, precision_completion)):
        hashes.update(
            {str((root / name).resolve()): value for name, value in completed["hashes"].items()}
        )
        hashes[str((root / "completed.json").resolve())] = sha(root / "completed.json")
    for name in ("protocol.json", "train_inputs.pt", "evaluation_data.pt"):
        if sha(source / name) != protocol["source_hashes"][name]:
            raise ValueError("Original source/cache changed")
        hashes[str((source / name).resolve())] = sha(source / name)
    payload = cast(
        dict[str, Any], torch.load(endpoint / "last.pt", map_location="cpu", weights_only=False)
    )
    expected_step = protocol["config"]["steps"] if verification else 100000
    if payload["identity"] != protocol["identity"] or payload["step"] != expected_step:
        raise ValueError("Endpoint identity or step differs")
    source_protocol = cast(dict[str, Any], read_json(source / "protocol.json"))
    config = copy.deepcopy(source_protocol["model_config"])
    config["param_dim"] = 12
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        model = build_model(cast(DictConfig, OmegaConf.create(config)))
        model.predictor = cast(Spartan, OvercompletePredictor())
    model.load_state_dict(payload["model"], strict=True)
    for name, expected in protocol["frozen_hashes"].items():
        if module_fingerprint(getattr(model, name)) != expected:
            raise ValueError("Frozen source chart changed")
    original_hash = model_fingerprint(model)
    train = cast(
        TensorSplit, torch.load(source / "train_inputs.pt", map_location="cpu", weights_only=False)
    )
    check_train(train)
    if not verification and len(train["online"]) != 8192:
        raise ValueError("Require all8192 original TRAIN groups")
    return Inputs(
        model.requires_grad_(False).double().eval(),
        train,
        protocol,
        payload["records"][-1]["response"],
        precision_protocol,
        hashes,
        original_hash,
    )


def verify_files(hashes: dict[str, str]) -> None:
    """Preserve every completed source/cache artifact throughout fitting and evaluation."""
    for name, expected in hashes.items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Immutable source changed: {name}")
