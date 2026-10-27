"""Authenticate final joint-state models and re-encode their observed TRAIN inputs."""

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from diagnostics.global_code_calibration.source import verify_files
from diagnostics.joint_state_survival.data import check_train, refresh_evaluation
from diagnostics.joint_state_survival.forward import Arm
from diagnostics.joint_state_survival.run import code_identity as survival_identity
from diagnostics.joint_state_survival.source import validate_source_mode
from diagnostics.multiquery_control.data import (
    EvaluationData,
    TensorSplit,
    encode_states,
    module_fingerprint,
)
from diagnostics.multiquery_control.run import sha
from diagnostics.predictor_transfer_precision.source import PRECISION_CACHE_SHA256
from diagnostics.table_settling_control.source import authenticated_directory, model_fingerprint
from scjepa.data import simplified_gaussian as dgp
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA
from scjepa.training.factory import build_model
from scjepa.training.rng import preserve_rng

PINNED = {
    "frozen_state": "04813afe350067ef578570a81b11ba242ef94de26e071dbf0cf478255c54ba66",
    "live_state": "9e551d96cfaf292d6353f6ab1268619bca686f29ac5fc7b819bcafc5f7f56643",
}


@dataclass(frozen=True)
class Inputs:
    """A frozen endpoint and observed TRAIN cache; no held-out input cache is deserialized."""

    model: SCJEPA
    observations: TensorSplit
    hashes: dict[str, str]
    protocol: dict[str, Any]
    original_model_sha256: str
    float32_model_sha256: str
    endpoint: Path
    archived_original: dict[str, Any]


@dataclass(frozen=True)
class Evaluation:
    """Post-fit original64 and precision8192 validation inputs in the endpoint's own charts."""

    original_validation: EvaluationData
    precision_data: EvaluationData
    archived_reports: dict[str, Any]
    evaluation_seed: int
    original_bootstrap_samples: int


def load_training(
    endpoint: Path,
    source: Path,
    precision: Path,
    *,
    arm: Arm,
    checkpoint_sha256: str,
    verification: bool = False,
) -> Inputs:
    """Bind a fixed110k model to authenticated observations; discard checkpoint monitoring."""
    if arm not in PINNED or (not verification and checkpoint_sha256 != PINNED[arm]):
        raise ValueError("Require the explicitly pinned fixed110k endpoint for its declared arm")
    complete = authenticated_directory(
        endpoint,
        {
            "protocol.json",
            "last.pt",
            "initial_model.pt",
            "train_observations.pt",
            "train_receipt.json",
            "endpoint_assessment.json",
        },
    )
    protocol = cast(dict[str, Any], read_json(endpoint / "protocol.json"))
    config = protocol["config"]
    if (
        protocol["identity"] != complete["identity"]
        or protocol["code_identity"] != survival_identity()
    ):
        raise ValueError("Completed survival identity or code differs")
    if (
        config["arm"] != arm
        or config["verification"] != verification
        or sha(endpoint / "last.pt") != checkpoint_sha256
    ):
        raise ValueError("Endpoint arm, checkpoint pin or verification role differs")
    if not verification and (
        config["steps"],
        config["seed"],
        config["groups"],
        config["bootstrap_samples"],
        config["source_gate_mode"],
    ) != (10000, 202610261301, 64, 1000, "qualified_spartan"):
        raise ValueError("Require the completed canonical paired10k continuation")
    validate_source_mode(
        config["source_gate_mode"],
        config["checkpoint_sha256"],
        config["calibrated_sha256"],
        verification=verification,
    )
    verify_files(protocol["source_hashes"])
    for name in ("protocol.json", "train_inputs.pt", "evaluation_data.pt"):
        declared = Path(config["source_dir"]) / name
        if protocol["source_hashes"].get(str(declared.resolve())) != sha(source / name):
            raise ValueError("Original source/cache differs from completed survival ancestry")
    declared_precision = Path(config["precision_dir"]) / "validation_data.pt"
    if protocol["source_hashes"].get(str(declared_precision.resolve())) != sha(
        precision / "validation_data.pt"
    ):
        raise ValueError("Precision validation cache differs from completed survival ancestry")
    if not verification and sha(precision / "validation_data.pt") != PRECISION_CACHE_SHA256:
        raise ValueError("Require unchanged original8192 precision validation")
    source_protocol = cast(dict[str, Any], read_json(source / "protocol.json"))
    payload = cast(
        dict[str, Any], torch.load(endpoint / "last.pt", map_location="cpu", weights_only=False)
    )
    initial = cast(
        dict[str, Any],
        torch.load(endpoint / "initial_model.pt", map_location="cpu", weights_only=False),
    )
    expected_step = int(protocol["source_step"]) + int(config["steps"])
    if (
        payload["identity"] != protocol["identity"]
        or initial["identity"] != protocol["identity"]
        or payload["updates"] != config["steps"]
        or payload["step"] != expected_step
        or payload["records"][-1]["step"] != expected_step
        or (not verification and expected_step != 110000)
    ):
        raise ValueError("Require last fixed-budget checkpoint and its final validation record")
    with preserve_rng(torch.device("cpu")):
        model = build_model(cast(DictConfig, OmegaConf.create(source_protocol["model_config"])))
    model.load_state_dict(initial["model"], strict=True)
    if model_fingerprint(model) != protocol["initial_model_sha256"] or any(
        module_fingerprint(getattr(model, name)) != value
        for name, value in protocol["source_provenance"]["initial_state_hashes"].items()
    ):
        raise ValueError("Initial model or inherited state charts changed")
    model.load_state_dict(payload["model"], strict=True)
    if (
        model.get_extra_state() != protocol["source_provenance"]["rollout_flags"]
        or model.parameter_encoder.param_dim != 6
        or model.predictor.param_dim != 6
        or not all(layer.dense and not layer.identity for layer in model.predictor.layers)
    ):
        raise ValueError("Endpoint interface, attached rollout flags or dense gates changed")
    if arm == "frozen_state" and any(
        module_fingerprint(getattr(model, name)) != value
        for name, value in protocol["source_provenance"]["initial_state_hashes"].items()
    ):
        raise ValueError("Frozen-state endpoint changed its state charts")
    receipt = cast(dict[str, Any], read_json(endpoint / "train_receipt.json"))
    if (
        receipt["identity"] != protocol["identity"]
        or receipt["sha256"] != sha(endpoint / "train_observations.pt")
        or receipt["dataset_fingerprint"] != source_protocol["source"]["dataset_identity"]
        or not all(
            receipt.get(name) is True
            for name in ("observations_only", "actions_exact", "all_initial_encodings_checked")
        )
    ):
        raise ValueError(
            "Observed TRAIN cache lost original dataset/action/encoding authentication"
        )
    observations = cast(
        TensorSplit,
        torch.load(endpoint / "train_observations.pt", map_location="cpu", weights_only=True),
    )
    check_train(observations)
    if len(observations["observations"]) != source_protocol["config"]["train_groups"] or (
        not verification and len(observations["observations"]) != 8192
    ):
        raise ValueError("Require all original TRAIN groups in unchanged order")
    hashes = {
        **protocol["source_hashes"],
        **{str((endpoint / name).resolve()): value for name, value in complete["hashes"].items()},
        str((endpoint / "completed.json").resolve()): sha(endpoint / "completed.json"),
        **{
            str((source / name).resolve()): sha(source / name)
            for name in ("protocol.json", "train_inputs.pt", "evaluation_data.pt")
        },
        str((precision / "validation_data.pt").resolve()): sha(precision / "validation_data.pt"),
    }
    float32_hash = model_fingerprint(model)
    model = model.requires_grad_(False).double().eval()
    # last.pt includes earlier monitoring tensors; they are not returned to the fitter.
    return Inputs(
        model,
        observations,
        hashes,
        protocol,
        model_fingerprint(model),
        float32_hash,
        endpoint,
        payload["records"][-1]["response"],
    )


@torch.no_grad()
def encode_observations(
    model: SCJEPA,
    observations: TensorSplit,
    *,
    chunk_size: int,
    progress: Callable[[int], None] | None = None,
) -> TensorSplit:
    """Re-encode every original TRAIN trajectory with the fixed final float32 state charts."""
    check_train(observations)
    if chunk_size < 1 or any(p.requires_grad for p in model.parameters()) or model.training:
        raise ValueError("Require positive chunks and completely frozen eval model")
    candidate = copy.deepcopy(model).float().requires_grad_(False).eval()
    fixed = model_fingerprint(candidate)
    online: list[torch.Tensor] = []
    target: list[torch.Tensor] = []
    for start in range(0, len(observations["observations"]), chunk_size):
        first, second = encode_states(
            candidate, observations["observations"][start : start + chunk_size], 32
        )
        online.append(first)
        target.append(second)
        if progress:
            progress(min(start + chunk_size, len(observations["observations"])))
    if model_fingerprint(candidate) != fixed:
        raise RuntimeError("TRAIN encoding changed a frozen endpoint state chart")
    return {
        "online": torch.cat(online),
        "target": torch.cat(target),
        "actions": observations["actions"],
    }


def load_evaluation(
    inputs: Inputs,
    source: Path,
    precision: Path,
    *,
    device: torch.device,
    verification: bool = False,
) -> Evaluation:
    """Only the runner's post-fit stage opens physical validation inputs and refreshes charts."""
    verify_files(inputs.hashes)
    old = cast(
        EvaluationData,
        torch.load(source / "evaluation_data.pt", map_location="cpu", weights_only=False),
    )
    prior = cast(
        EvaluationData,
        torch.load(precision / "validation_data.pt", map_location="cpu", weights_only=False),
    )
    if set(prior) != {"metadata", "validation"}:
        raise ValueError("Precision sample must remain validation-only")
    for name in (
        "observation_map_fingerprint",
        "views_per_group",
        "context_length",
        "trajectory_length",
        "generator_version",
    ):
        if old["metadata"][name] != prior["metadata"][name]:
            raise ValueError("Original and precision physical data laws differ")
    if (
        not torch.equal(
            old["metadata"]["observation_matrix"], prior["metadata"]["observation_matrix"]
        )
        or prior["metadata"]["generator_version"] != dgp.GENERATOR_VERSION
        or (
            not verification
            and (len(prior["validation"]["theta"]), prior["metadata"]["seed"])
            != (8192, 202610271901)
        )
    ):
        raise ValueError("Precision observation map, generator or fixed sample differs")
    model = copy.deepcopy(inputs.model).float().to(device).eval()
    groups = int(inputs.protocol["config"]["groups"])
    original = refresh_evaluation(
        model,
        {
            "metadata": old["metadata"],
            "validation": {key: value[:groups] for key, value in old["validation"].items()},
        },
        ("validation",),
    )
    refreshed = refresh_evaluation(model, prior, ("validation",))
    assessment = cast(dict[str, Any], read_json(inputs.endpoint / "endpoint_assessment.json"))
    if assessment["identity"] != inputs.protocol["identity"] or assessment[
        "checkpoint_sha256"
    ] != sha(inputs.endpoint / "last.pt"):
        raise ValueError("Archived endpoint assessment differs from final weights")
    return Evaluation(
        original,
        refreshed,
        {
            "original64": inputs.archived_original,
            "precision8192": assessment["metrics"]["precision_validation"]["response"],
        },
        int(inputs.protocol["config"]["seed"]),
        int(inputs.protocol["config"]["bootstrap_samples"]),
    )
