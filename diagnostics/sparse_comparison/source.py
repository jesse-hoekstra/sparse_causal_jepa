"""Authenticate the raw200k endpoint without selecting its post-fit calibration."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from diagnostics.global_code_calibration.source import verify_files
from diagnostics.joint_dense_continuation.run import code_identity as parent_identity
from diagnostics.joint_dense_continuation.source import load_inputs as load_parent
from diagnostics.joint_state_survival.source import response_status
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit
from diagnostics.multiquery_control.run import sha
from diagnostics.table_settling_control.source import authenticated_directory, model_fingerprint
from scjepa.data.factory import build_dgp_dataset, resolve_dgp_config
from scjepa.data.types import DGPConfig
from scjepa.experiments.evaluation import EvaluationConfig
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA
from scjepa.training.factory import ModelConfig
from scjepa.training.loop import ObjectiveConfig

CHECKPOINT_SHA256 = "ea0225c00f4556563ddfcf900cd0fc27f1ea06167ac868bc97d122c6a984cc98"
STREAM_SEED = 202610261301


@dataclass(frozen=True)
class Inputs:
    """Observed TRAIN, separate physical evaluation cohorts and the exact raw source."""

    model: SCJEPA
    observations: TensorSplit
    evaluation: EvaluationData
    precision_data: EvaluationData
    data_config: DGPConfig
    model_config: ModelConfig
    objective_config: ObjectiveConfig
    native_evaluation: EvaluationConfig
    hashes: dict[str, str]
    provenance: dict[str, Any]
    archived_validation: dict[str, Any]
    admission: dict[str, Any]
    source_step: int
    seed: int
    batch_size: int


def admission_status(report: dict[str, Any]) -> dict[str, Any]:
    """Keep amended information/response admission distinct from historical benefit success."""
    response = report["response"]
    status = response_status(response)
    rows = response.get("per_parameter", [])
    energies = (
        len(rows) == 6
        and sorted(row.get("parameter", 0) for row in rows) == list(range(1, 7))
        and all(
            all(
                isinstance(row.get(name), int | float)
                and math.isfinite(row[name])
                and row[name] > 0
                for name in ("exact_response_energy", "learned_response_energy")
            )
            for row in rows
        )
    )
    retentions = [
        report[name] for name in ("original_full_validation_retention", "precision_retention")
    ]
    retention = all(
        len(value.get("r2", [])) == 6
        and all(isinstance(x, int | float) and math.isfinite(x) and x >= 0.8 for x in value["r2"])
        and value.get("all_six_above_0_80") is True
        and math.isfinite(value["minimum_r2"])
        and value["minimum_r2"] == min(value["r2"])
        for value in retentions
    )
    passed = retention and energies and status["all_six_fidelity_pass"]
    return {
        "passed": passed,
        "all_six_retention": retention,
        "all_six_nonzero_response_energy": energies,
        "historical_all_six_use_passed": report["all_six_source_gate"],
        **status,
        "scope": (
            "Authenticated raw endpoint existing8192 validation; amended admission does not "
            "convert failed historical benefit into success"
        ),
    }


def load_inputs(endpoint: Path, *, device: torch.device, verification: bool = False) -> Inputs:
    """Restore immutable raw weights and derive the original data law from verified ancestry."""
    complete = authenticated_directory(
        endpoint,
        {
            "protocol.json",
            "last.pt",
            "last.receipt.json",
            "endpoint_assessment.json",
            "endpoint_assessment.pt",
            "results.json",
        },
    )
    protocol = cast(dict[str, Any], read_json(endpoint / "protocol.json"))
    config = protocol["config"]
    if (
        protocol["identity"] != complete["identity"]
        or protocol["code_identity"] != parent_identity()
        or config["verification"] != verification
    ):
        raise ValueError("Raw dense continuation identity/code/verification role differs")
    checkpoint_sha = sha(endpoint / "last.pt")
    if not verification and checkpoint_sha != CHECKPOINT_SHA256:
        raise ValueError("Require the pinned raw200k checkpoint, not its calibrated composition")
    verify_files(protocol["source_hashes"])
    parent = load_parent(
        Path(config["endpoint_dir"]),
        Path(config["source_dir"]),
        Path(config["precision_dir"]),
        checkpoint_sha256=config["checkpoint_sha256"],
        device=device,
        verification=verification,
    )
    payload = cast(
        dict[str, Any], torch.load(endpoint / "last.pt", map_location="cpu", weights_only=False)
    )
    receipt = read_json(endpoint / "last.receipt.json")
    if (
        receipt["identity"] != protocol["identity"]
        or receipt["sha256"] != checkpoint_sha
        or payload["identity"] != protocol["identity"]
        or payload["updates"] != config["steps"]
        or payload["step"] != parent.source_step + config["steps"]
        or payload["optimizer_steps"] != parent.optimizer_step + config["steps"]
    ):
        raise ValueError("Raw source is not the declared fixed final continuation")
    if not verification and (
        payload["step"],
        payload["optimizer_steps"],
        parent.seed,
        parent.batch_size,
    ) != (200000, 100000, STREAM_SEED, 16):
        raise ValueError("Require canonical200k/raw100k joint optimizer/B16/stream seed")
    model = parent.model
    model.load_state_dict(payload["model"], strict=True)
    if model.get_extra_state() != protocol["source_provenance"]["rollout_flags"] or any(
        not bool(torch.isfinite(p).all()) for p in model.parameters()
    ):
        raise ValueError("Raw source rollout or finite weights differ")
    original = cast(dict[str, Any], read_json(Path(config["source_dir"]) / "protocol.json"))
    live_protocol = cast(dict[str, Any], read_json(Path(config["endpoint_dir"]) / "protocol.json"))
    record_path = Path(
        live_protocol["config"].get("data_record")
        or Path(original["source"]["checkpoint"]).parent / "run_config.json"
    )
    if protocol["source_hashes"].get(str(record_path.resolve())) != sha(record_path):
        raise ValueError("Original data record is not authenticated by source ancestry")
    record = cast(dict[str, Any], read_json(record_path))
    if (
        record["run_identity"] != original["source"]["run_identity"]
        or record["dataset_identity"] != original["source"]["dataset_identity"]
    ):
        raise ValueError("Original data record identity differs")
    data_config = resolve_dgp_config(cast(DictConfig, OmegaConf.create(record["data"])))
    dataset = build_dgp_dataset(
        data_config, observation_matrix=parent.evaluation["metadata"]["observation_matrix"]
    )
    if dataset.fingerprint != record["dataset_identity"]:
        raise ValueError("Original TRAIN data law differs")
    assessment = cast(dict[str, Any], read_json(endpoint / "endpoint_assessment.json"))
    if (
        assessment["identity"] != protocol["identity"]
        or assessment["checkpoint_sha256"] != checkpoint_sha
        or assessment["raw_sha256"] != sha(endpoint / "endpoint_assessment.pt")
    ):
        raise ValueError("Raw endpoint assessment differs")
    admission = admission_status(assessment["metrics"]["precision_validation"])
    model_config = ModelConfig(**original["model_config"])
    native = EvaluationConfig(batch_size=parent.batch_size)
    if verification:
        native = EvaluationConfig(
            calibration_episodes=4,
            validation_episodes=4,
            test_episodes=4,
            batch_size=parent.batch_size,
        )
    hashes = {
        **parent.hashes,
        **protocol["source_hashes"],
        **{str((endpoint / name).resolve()): value for name, value in complete["hashes"].items()},
        str((endpoint / "completed.json").resolve()): sha(endpoint / "completed.json"),
    }
    verify_files(hashes)
    return Inputs(
        model,
        parent.observations,
        parent.evaluation,
        parent.precision_data,
        data_config,
        model_config,
        parent.objective.config,
        native,
        hashes,
        {
            "source_identity": protocol["identity"],
            "checkpoint_sha256": checkpoint_sha,
            "source_model_sha256": model_fingerprint(model),
            "dataset_identity": dataset.fingerprint,
            "raw_source_only": True,
            "rollout_flags": model.get_extra_state(),
            "historical_admission": admission,
        },
        payload["records"][-1],
        admission,
        int(payload["step"]),
        parent.seed,
        parent.batch_size,
    )
