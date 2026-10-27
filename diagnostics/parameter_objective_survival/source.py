"""Authenticate the fixed canonical MLP6 transfer endpoint without inheriting its optimizer."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.joint_acquisition.source import SourceBundle, load_primary_source
from diagnostics.mlp_oracle_code_control.model import MLPPredictor
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.run import sha
from diagnostics.predictor_transfer_control.model import audit_model_fingerprint
from diagnostics.predictor_transfer_control.run import code_identity as parent_code_identity
from diagnostics.table_settling_control.source import authenticated_directory
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import Spartan
from scjepa.training.rng import preserve_rng

ENDPOINT_SHA256 = "a9024d8740911ea7d45ae59bc58fff921bf3e98e8cf8509e511c5a7382b8a481"


@dataclass(frozen=True)
class Inputs:
    """Frozen source identity, exact learned E/P and original data; no optimizer history."""

    model: SCJEPA
    bundle: SourceBundle
    hashes: dict[str, str]
    provenance: dict[str, Any]
    rng: dict[str, Any]
    archived_validation: dict[str, Any]


def load_inputs(
    source: Path,
    endpoint: Path,
    *,
    device: torch.device,
    training_source_root: Path | None = None,
    verification: bool = False,
) -> Inputs:
    """Restore only the declared endpoint's weights and verify all source dependencies."""
    completed = authenticated_directory(endpoint, {"protocol.json", "last.pt", "results.json"})
    protocol = cast(dict[str, Any], read_json(endpoint / "protocol.json"))
    config = protocol["config"]
    if protocol["code_identity"] != parent_code_identity() or config["arm"] != "mlp":
        raise ValueError("Require the unchanged completed MLP6 transfer code and arm")
    seed = int(config["seed"])
    if not verification and (seed != 202610261301 or sha(endpoint / "last.pt") != ENDPOINT_SHA256):
        raise ValueError("Require the declared canonical fixed100k MLP6 endpoint")
    bundle = load_primary_source(
        source,
        seed,
        expected_step=2 if verification else 30000,
        **({"expected_checkpoint_sha256": None} if verification else {}),
        training_source_root=training_source_root,
    )
    payload = cast(
        dict[str, Any], torch.load(endpoint / "last.pt", map_location="cpu", weights_only=False)
    )
    source_step = int(config["steps"]) if verification else 100000
    if (
        payload["identity"] != protocol["identity"]
        or payload["step"] != source_step
        or payload["records"][-1]["step"] != source_step
    ):
        raise ValueError("Source checkpoint is not the completed fixed-budget endpoint")
    model, discarded_optimizer = build_source_model(bundle, torch.device("cpu"))
    del discarded_optimizer
    with preserve_rng():
        model.predictor = cast(Spartan, MLPPredictor())
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False).eval()
    for name, expected in protocol["provenance"]["frozen_hashes"].items():
        if module_fingerprint(getattr(model, name)) != expected:
            raise ValueError("Parent's frozen E6 or state charts changed")
    if model.predictor.param_dim != 6 or model.parameter_encoder.param_dim != 6:
        raise ValueError("Require the unchanged six-code interface")
    if not verification and (bundle.train_groups, bundle.batch_size) != (8192, 16):
        raise ValueError("Require original8192 TRAIN groups and B16")
    declared_source = Path(config["source_dir"])
    for name in ("protocol.json", "train_inputs.pt", "evaluation_data.pt"):
        if protocol["source_hashes"].get(str((declared_source / name).resolve())) != sha(
            source / name
        ):
            raise ValueError("Original TRAIN/evaluation source changed")
    hashes: dict[str, str] = {
        **protocol["source_hashes"],
        **{str((endpoint / name).resolve()): value for name, value in completed["hashes"].items()},
        str((endpoint / "completed.json").resolve()): sha(endpoint / "completed.json"),
        **{
            str((source / name).resolve()): sha(source / name)
            for name in ("protocol.json", "train_inputs.pt", "evaluation_data.pt")
        },
    }
    provenance: dict[str, Any] = {
        "source_identity": protocol["identity"],
        "source_step": source_step,
        "endpoint_sha256": sha(endpoint / "last.pt"),
        "source_model_sha256": audit_model_fingerprint(model),
        "initial_encoder_sha256": module_fingerprint(model.parameter_encoder),
        "initial_predictor_sha256": module_fingerprint(model.predictor),
        "frozen_hashes": {
            name: module_fingerprint(getattr(model, name))
            for name in ("online_encoder", "target_encoder")
        },
        "rollout_flags": model.get_extra_state(),
        "ancestry": (
            "Fresh label-free shared-feature initialization,12D acquisition, TRAIN-PCA6, "
            "fresh MLP6 fitting; no physical-label or teacher ancestry."
        ),
        "optimizer": (
            "No parent optimizer loaded; both arms use fresh Adam .001 with E/P trainable."
        ),
    }
    result = Inputs(
        model.to(device), bundle, hashes, provenance, payload["rng"], payload["records"][-1]
    )
    verify_sources(result)
    return result


def verify_sources(inputs: Inputs) -> None:
    """Keep all prior sources immutable and both distinct state charts completely frozen."""
    if any(sha(Path(name)) != expected for name, expected in inputs.hashes.items()):
        raise ValueError("Immutable source artifact changed")
    for name, expected in inputs.provenance["frozen_hashes"].items():
        if module_fingerprint(getattr(inputs.model, name)) != expected:
            raise ValueError("Frozen state chart changed")
    if inputs.model.get_extra_state() != inputs.provenance["rollout_flags"]:
        raise ValueError("Original rollout or EMA flags changed")
