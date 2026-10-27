"""Restore the uncalibrated live endpoint and its exact ordinary AdamW history."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.joint_state_calibration.source import PINNED, load_training
from diagnostics.joint_state_survival.forward import configure_model, ordinary_objective
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit
from diagnostics.multiquery_control.run import sha
from diagnostics.table_settling_control.source import model_fingerprint
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA
from scjepa.training.loop import ObjectiveConfig, TrainingObjective

CHECKPOINT_SHA256 = PINNED["live_state"]


@dataclass(frozen=True)
class Inputs:
    """Original weights/moments and observed TRAIN; physical labels remain evaluation-only."""

    model: SCJEPA
    optimizer: torch.optim.AdamW
    objective: TrainingObjective
    observations: TensorSplit
    evaluation: EvaluationData
    precision_data: EvaluationData
    hashes: dict[str, str]
    provenance: dict[str, Any]
    rng: dict[str, Any]
    archived_validation: dict[str, Any]
    archived_endpoint: dict[str, Any]
    source_step: int
    optimizer_step: int
    batch_size: int
    seed: int


def load_inputs(
    endpoint: Path,
    source: Path,
    precision: Path,
    *,
    checkpoint_sha256: str,
    device: torch.device,
    verification: bool = False,
) -> Inputs:
    """Authenticate fixed110k and restore AdamW in the exact original trainable-parameter order."""
    restored = load_training(
        endpoint,
        source,
        precision,
        arm="live_state",
        checkpoint_sha256=checkpoint_sha256,
        verification=verification,
    )
    payload = cast(
        dict[str, Any], torch.load(endpoint / "last.pt", map_location="cpu", weights_only=False)
    )
    model = copy.deepcopy(restored.model).float().to(device)
    configure_model(model, "live_state")
    if model_fingerprint(model) != restored.float32_model_sha256:
        raise ValueError("Restored float32 live weights differ from the uncalibrated endpoint")
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    saved = cast(dict[str, Any], payload["optimizer"])
    groups = cast(list[dict[str, Any]], saved["param_groups"])
    if len(groups) != 1:
        raise ValueError("Require the original single AdamW parameter group")
    group = groups[0]
    ids = list(range(len(parameters)))
    if (
        group.get("foreach") is not None
        or group.get("fused") is not None
        or group["params"] != ids
        or (
            group["lr"],
            group["weight_decay"],
            tuple(group["betas"]),
            group["eps"],
            group["amsgrad"],
            group["maximize"],
            group.get("capturable", False),
            group.get("differentiable", False),
            group.get("decoupled_weight_decay", True),
        )
        != (0.001, 0.0, (0.9, 0.999), 1e-8, False, False, False, False, True)
    ):
        raise ValueError("Saved optimizer parameter order or ordinary AdamW settings differ")
    states = cast(dict[int, dict[str, Any]], saved["state"])
    optimizer_step = int(payload["updates"])
    if set(states) != set(ids):
        raise ValueError("Every live trainable parameter must retain its original AdamW moments")
    for pid, parameter in zip(ids, parameters, strict=True):
        entry = states[pid]
        count = float(entry["step"])
        if count != optimizer_step:
            raise ValueError("Saved AdamW moment step differs from source joint updates")
        for key in ("exp_avg", "exp_avg_sq"):
            moment = entry[key]
            if (
                not isinstance(moment, Tensor)
                or moment.shape != parameter.shape
                or moment.dtype != torch.float32
                or not bool(torch.isfinite(moment).all())
                or (key == "exp_avg_sq" and bool((moment < 0).any()))
            ):
                raise ValueError("Saved AdamW moment shape, dtype or values differ")
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.0)
    optimizer.load_state_dict(copy.deepcopy(saved))
    original = cast(dict[str, Any], read_json(source / "protocol.json"))
    objective = ordinary_objective(ObjectiveConfig(**original["objective_config"])).to(device)
    evaluation = cast(
        EvaluationData,
        torch.load(source / "evaluation_data.pt", map_location="cpu", weights_only=False),
    )
    precision_data = cast(
        EvaluationData,
        torch.load(precision / "validation_data.pt", map_location="cpu", weights_only=False),
    )
    if set(precision_data) != {"metadata", "validation"}:
        raise ValueError("Require existing precision validation only")
    for name in (
        "observation_map_fingerprint",
        "generator_version",
        "views_per_group",
        "context_length",
        "trajectory_length",
    ):
        if evaluation["metadata"][name] != precision_data["metadata"][name]:
            raise ValueError("Original and precision physical data laws differ")
    if not torch.equal(
        evaluation["metadata"]["observation_matrix"],
        precision_data["metadata"]["observation_matrix"],
    ):
        raise ValueError("Original and precision observation maps differ")
    assessment = cast(dict[str, Any], read_json(endpoint / "endpoint_assessment.json"))
    if (
        assessment["identity"] != restored.protocol["identity"]
        or assessment["checkpoint_sha256"] != checkpoint_sha256
        or assessment["raw_sha256"] != sha(endpoint / "endpoint_assessment.pt")
    ):
        raise ValueError("Archived fixed110k assessment differs")
    return Inputs(
        model,
        optimizer,
        objective,
        restored.observations,
        evaluation,
        precision_data,
        restored.hashes,
        {
            "source_identity": restored.protocol["identity"],
            "checkpoint_sha256": checkpoint_sha256,
            "source_model_sha256": restored.float32_model_sha256,
            "source_global_step": payload["step"],
            "source_optimizer_step": optimizer_step,
            "optimizer_parameter_names": names,
            "optimizer_group": group,
            "rollout_flags": model.get_extra_state(),
            "calibration_not_applied": True,
            "optimizer_history": (
                "Exact original live AdamW moments/counters; no reset or source offset"
            ),
        },
        payload["rng"],
        payload["records"][-1],
        assessment["metrics"],
        int(payload["step"]),
        optimizer_step,
        int(original["config"]["batch_size"]),
        int(restored.protocol["config"]["seed"]),
    )
