"""Authenticate the conditional calibrated SPARTAN source and its exact original data law."""

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from omegaconf import OmegaConf

from diagnostics.global_code_calibration.source import verify_files
from diagnostics.joint_acquisition.source import SourceBundle, load_primary_source
from diagnostics.joint_state_survival.forward import ordinary_objective
from diagnostics.multiquery_control.data import EvaluationData, module_fingerprint
from diagnostics.multiquery_control.run import sha
from diagnostics.predictor_transfer_calibration.objective import compose_translation
from diagnostics.predictor_transfer_calibration.run import code_identity as calibration_identity
from diagnostics.predictor_transfer_precision.source import load_inputs as load_precision
from diagnostics.table_settling_control.source import authenticated_directory, model_fingerprint
from scjepa.data.factory import build_dgp_dataset
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng


@dataclass(frozen=True)
class Inputs:
    """Exact float32 starting model, frozen physical data and complete immutable provenance."""

    model: SCJEPA
    bundle: SourceBundle
    precision_data: EvaluationData
    data_config: dict[str, Any]
    hashes: dict[str, str]
    provenance: dict[str, Any]
    archived_response: dict[str, Any]
    source_step: int


SourceGateMode = Literal["all_six_use", "qualified_spartan"]
QUALIFIED_CHECKPOINT_SHA256 = "b85b70e110f6d00a1d5bbcdddcf9756d0288daf2ccf1c51eded92eff49d864ce"
QUALIFIED_CALIBRATED_SHA256 = "8225738a6db85e074bf95a0af9c40055db979a1ccdca31735ba61485f1335cb9"


def validate_source_mode(
    mode: SourceGateMode, checkpoint: str, calibrated: str, *, verification: bool = False
) -> None:
    """Restrict the prospective exception to the one explicitly authorized fixed source."""
    if mode not in ("all_six_use", "qualified_spartan"):
        raise ValueError("Unknown source gate mode")
    if (
        mode == "qualified_spartan"
        and not verification
        and (checkpoint, calibrated) != (QUALIFIED_CHECKPOINT_SHA256, QUALIFIED_CALIBRATED_SHA256)
    ):
        raise ValueError("Qualified mode requires the specifically pinned SPARTAN source")


def response_status(report: dict[str, Any]) -> dict[str, Any]:
    """Expose valid per-factor gates without converting an unresolved benefit to a pass."""
    rows = report.get("per_parameter", [])
    valid = len(rows) == 6 and {row.get("parameter") for row in rows} == set(range(1, 7))
    fidelity: list[int] = []
    benefit: list[int] = []
    for row in rows:
        for name in ("normalized_response_error", "predictive_benefit"):
            entry = row.get(name, {})
            interval = cast(list[Any] | tuple[Any, ...] | None, entry.get("interval_95"))
            if (
                entry.get("defined", True) is not True
                or not isinstance(interval, list | tuple)
                or len(interval) != 2
                or not all(
                    isinstance(value, int | float) and math.isfinite(value) for value in interval
                )
                or interval[0] > interval[1]
            ):
                valid = False
                continue
            if name == "normalized_response_error" and interval[1] < 1:
                fidelity.append(row["parameter"])
            if name == "predictive_benefit" and interval[0] > 0:
                benefit.append(row["parameter"])
    return {
        "valid": valid,
        "fidelity_pass_parameters": sorted(fidelity),
        "benefit_pass_parameters": sorted(benefit),
        "unmet_benefit_parameters": sorted(set(range(1, 7)) - set(benefit)),
        "all_six_fidelity_pass": valid and len(fidelity) == 6,
        "all_six_use_pass": valid and len(fidelity) == 6 and len(benefit) == 6,
    }


def response_gate(report: dict[str, Any]) -> bool:
    """Keep the unchanged all-six final success criterion and strict default admission."""
    return bool(response_status(report)["all_six_use_pass"])


def admission_gate(report: dict[str, Any], mode: SourceGateMode) -> bool:
    """Permit only the recorded theta2 uncertainty in the explicitly qualified diagnostic."""
    if mode == "all_six_use":
        return response_gate(report)
    if mode != "qualified_spartan":
        raise ValueError("Unknown source gate mode")
    status = response_status(report)
    if not status["all_six_fidelity_pass"] or not {1, 3, 4, 5, 6}.issubset(
        status["benefit_pass_parameters"]
    ):
        return False
    theta2 = next(row for row in report["per_parameter"] if row["parameter"] == 2)[
        "predictive_benefit"
    ]
    estimate = theta2.get("estimate")
    return (
        isinstance(estimate, int | float)
        and math.isfinite(estimate)
        and estimate > 0
        and theta2["interval_95"][0] <= 0 < theta2["interval_95"][1]
    )


@torch.no_grad()
def cast_replay(
    model: SCJEPA, observations: torch.Tensor, actions: torch.Tensor
) -> tuple[SCJEPA, dict[str, Any]]:
    """Materialize one float32 continuation with declared numerical replay tolerance."""
    reference = model.double().requires_grad_(False).eval()
    candidate = copy.deepcopy(reference).float().eval()
    before = cast(ModelOutput, reference(observations.double(), actions.double()))
    after = cast(ModelOutput, candidate(observations.float(), actions.float()))
    errors: dict[str, Any] = {}
    for name in ("online_states", "parameters", "predictions", "targets"):
        expected, actual = getattr(before, name).double(), getattr(after, name).double()
        if not bool(torch.isfinite(expected).all()) or not bool(torch.isfinite(actual).all()):
            raise ValueError(f"Nonfinite float32 numerical replay: {name}")
        rms = float(expected.square().mean().sqrt())
        maximum = float((expected - actual).abs().max())
        tolerance = 1e-6 + 1e-4 * rms
        if maximum > tolerance:
            raise ValueError(f"Float32 numerical replay failed: {name}")
        errors[name] = {"max_absolute_error": maximum, "reference_rms": rms, "tolerance": tolerance}
    return candidate, {
        "float64_sha256": model_fingerprint(reference),
        "float32_sha256": model_fingerprint(candidate),
        "tolerance": "max absolute <= 1e-6 + 1e-4 reference RMS",
        "replay": errors,
    }


def load_inputs(
    source: Path,
    endpoint: Path,
    calibration: Path,
    precision: Path,
    *,
    checkpoint_sha256: str,
    calibrated_sha256: str,
    device: torch.device,
    training_source_root: Path | None = None,
    data_record: Path | None = None,
    verification: bool = False,
    source_gate_mode: SourceGateMode = "all_six_use",
) -> Inputs:
    """Reject unapproved source gates, changed accepted calibration or a different data law."""
    validate_source_mode(
        source_gate_mode, checkpoint_sha256, calibrated_sha256, verification=verification
    )
    completion = authenticated_directory(
        calibration,
        {"protocol.json", "results.json", "composed_model.pt", "translation.pt", "fit_state.pt"},
    )
    protocol = cast(dict[str, Any], read_json(calibration / "protocol.json"))
    if (
        completion["identity"] != protocol["identity"]
        or protocol["code_identity"] != calibration_identity()
    ):
        raise ValueError("Completed calibration identity or code differs")
    if protocol["config"]["arm"] != "spartan" or protocol["config"]["verification"] != verification:
        raise ValueError("Require the completed original dense SPARTAN calibration")
    if (
        protocol["config"]["checkpoint_sha256"] != checkpoint_sha256
        or sha(calibration / "composed_model.pt") != calibrated_sha256
    ):
        raise ValueError("Require explicitly pinned transfer and calibrated endpoint hashes")
    evaluation = load_precision(
        endpoint,
        source,
        precision,
        arm="spartan",
        checkpoint_sha256=checkpoint_sha256,
        verification=verification,
    )
    if protocol["source_identity"] != evaluation.source_protocol["identity"]:
        raise ValueError("Calibration and transfer source identities differ")
    verify_files(protocol["source_hashes"])
    for path, expected in evaluation.hashes.items():
        if protocol["source_hashes"].get(path) != expected:
            raise ValueError("Calibration and evaluation source artifacts differ")
    results = cast(dict[str, Any], read_json(calibration / "results.json"))
    archived = results["reports"]["precision8192/calibrated"]
    if results["identity"] != protocol["identity"] or (
        not verification and not admission_gate(archived, source_gate_mode)
    ):
        raise ValueError(
            "Conditional source fails its declared precision-validation admission policy"
        )
    composed = cast(
        dict[str, Any],
        torch.load(calibration / "composed_model.pt", map_location="cpu", weights_only=False),
    )
    delta = cast(
        dict[str, Any],
        torch.load(calibration / "translation.pt", map_location="cpu", weights_only=False),
    )
    original = evaluation.model
    if any(
        payload["identity"] != protocol["identity"]
        or payload["source_model_sha256"] != model_fingerprint(original)
        for payload in (composed, delta)
    ):
        raise ValueError("Calibration model/translation source differs")
    model = compose_translation(original, delta["translation"])
    restored = copy.deepcopy(original)
    restored.load_state_dict(composed["model"], strict=True)
    if model_fingerprint(model) != model_fingerprint(restored) or composed[
        "composed_model_sha256"
    ] != model_fingerprint(restored):
        raise ValueError("Composed endpoint differs from accepted constant translation")
    bundle = load_primary_source(
        source,
        seed=int(evaluation.source_protocol["config"]["seed"]),
        expected_step=2 if verification else 30000,
        **({"expected_checkpoint_sha256": None} if verification else {}),
        training_source_root=training_source_root,
    )
    ordinary_objective(ObjectiveConfig(**bundle.protocol["objective_config"]))
    record_path = (
        data_record or Path(bundle.protocol["source"]["checkpoint"]).parent / "run_config.json"
    )
    record = cast(dict[str, Any], read_json(record_path))
    if (
        record["run_identity"] != bundle.protocol["source"]["run_identity"]
        or record["dataset_identity"] != bundle.protocol["source"]["dataset_identity"]
    ):
        raise ValueError("Original data record identity differs")
    data_config = cast(dict[str, Any], record["data"])
    dataset = build_dgp_dataset(
        OmegaConf.create(data_config),
        observation_matrix=bundle.evaluation["metadata"]["observation_matrix"],
    )
    if dataset.fingerprint != record["dataset_identity"]:
        raise ValueError("Original dataset configuration differs from its fingerprint")
    # The replay batch is reconstructed from the exact original TRAIN law, never labels.
    train_dataset = build_dgp_dataset(
        OmegaConf.create(dict(data_config, split="train", num_episodes=bundle.train_groups)),
        observation_matrix=bundle.evaluation["metadata"]["observation_matrix"],
    )
    with preserve_rng(device):
        episodes = [
            train_dataset.process.generate(index) for index in range(min(32, bundle.train_groups))
        ]
        observations = torch.stack([episode.observations for episode in episodes]).to(device)
        actions = torch.stack([episode.actions for episode in episodes]).to(device)
        model, replay = cast_replay(restored.to(device), observations, actions)
    for name in ("online", "target"):
        if (
            module_fingerprint(getattr(model, f"{name}_encoder"))
            != bundle.evaluation["metadata"][f"state_{name}_sha256"]
        ):
            raise ValueError("Initial distinct state chart changed during composition/cast")
    hashes = {
        **evaluation.hashes,
        **protocol["source_hashes"],
        **{
            str((calibration / name).resolve()): value
            for name, value in completion["hashes"].items()
        },
        str((calibration / "completed.json").resolve()): sha(calibration / "completed.json"),
        str(record_path.resolve()): sha(record_path),
    }
    verify_files(hashes)
    return Inputs(
        model,
        bundle,
        evaluation.precision_data,
        data_config,
        hashes,
        {
            "source_gate_mode": source_gate_mode,
            "archived_response_status": response_status(archived),
            "archived_response": {
                key: archived[key] for key in ("ordinary_mse_by_horizon", "per_parameter")
            },
            "calibration_identity": protocol["identity"],
            "checkpoint_sha256": checkpoint_sha256,
            "calibrated_sha256": calibrated_sha256,
            "cast": replay,
            "rollout_flags": model.get_extra_state(),
            "initial_state_hashes": {
                name: module_fingerprint(getattr(model, name))
                for name in ("online_encoder", "target_encoder")
            },
            "source_gate": (
                "Strict default requires all-six use. Explicit qualified_spartan mode admits only "
                "the pinned source with all-six fidelity and theta2 benefit uncertainty; "
                "fresh full512cal/256val retention and declared response gate remain required. "
                "The archived unmet benefit and original all-six gate are preserved."
            ),
        },
        archived,
        int(evaluation.source_protocol["config"]["steps"]),
    )
