"""Compare saved dense endpoints under two paired observed-action encodings.

This reads completed training artifacts only. It runs no model, loads no data
cache, and fits coordinate assignments/readouts only on saved calibration rows.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from diagnostics.balanced_action_oracle.run import verify_completion as verify_balanced
from diagnostics.isotropic_oracle.analysis import routing_metrics
from diagnostics.isotropic_oracle.run import verify_completion as verify_isotropic
from diagnostics.oracle_state_learning.analysis import audit_samples, paired_mcc_interval
from diagnostics.oracle_state_learning.data import read_json, sha
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.eval.recovery import CoordinateMatch, fit_coordinate_match, mean_correlation_coefficient
from scjepa.eval.state_probes import fit_state_probe, state_probe_metrics

SCHEMA = "balanced-action-oracle-paired-encoding-analysis-v1"
type Encoding = Literal["isotropic", "balanced"]


def _identity(state: dict[str, Any], prefix: str) -> str:
    values = {
        key[len(prefix) + 1 :]: tensor_fingerprint(value)
        for key, value in state.items()
        if key.startswith(prefix + ".") and isinstance(value, Tensor)
    }
    if not values:
        raise ValueError("Initial checkpoint lacks encoder or predictor tensors")
    return canonical_fingerprint(values)


def _equal(actual: float, expected: float, label: str, *, rounding: bool = False) -> None:
    if not math.isclose(
        actual, expected, rel_tol=2e-6 if rounding else 1e-9, abs_tol=1e-8 if rounding else 1e-10
    ):
        raise ValueError(f"Reported {label} differs from saved samples")


def _health(values: Tensor) -> dict[str, float]:
    rows = values.flatten(0, 2).double()
    centered = rows - rows.mean(0)
    covariance = centered.T @ centered / len(rows)
    eigenvalues = cast(Tensor, torch.linalg.eigvalsh(covariance)).clamp_min(0)  # pyright: ignore[reportUnknownMemberType]
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-15)
    return {
        "variance_min": float(covariance.diagonal().min()),
        "covariance_eigenvalue_min": float(eigenvalues.min()),
        "effective_rank": float(
            (-probabilities * probabilities.clamp_min(1e-15).log()).sum().exp()
        ),
    }


def _load(directory: Path, encoding: Encoding, *, allow_smoke: bool) -> dict[str, Any]:
    verify = verify_isotropic if encoding == "isotropic" else verify_balanced
    completed = verify(directory, kind="train", require_current_source=False)
    required = {
        "config.json",
        "initial.pt",
        "last.pt",
        "matching.json",
        "startup.json",
        "history.jsonl",
        "wandb.json",
        "calibration.pt",
        "calibration.json",
        "validation.pt",
        "validation.json",
    }
    if not required <= completed["hashes"].keys():
        raise ValueError("Completion does not authenticate every required training artifact")
    record = read_json(directory / "config.json")
    scientific, candidate = record["scientific_config"], record["candidate_settings"]
    profile = "smoke" if allow_smoke else "exploratory"
    if (
        scientific["condition"] != "dense"
        or completed["condition"] != "dense"
        or scientific["seed"] != 0
        or completed["seed"] != 0
        or completed["profile"] != profile
        or scientific["profile"] != profile
        or candidate["profile"] != profile
        or candidate["seed"] != 0
        or (not allow_smoke and scientific["steps"] != 100_000)
    ):
        raise ValueError("Require fixed 100000-update seed-zero dense exploratory endpoints")
    if not allow_smoke and tuple(
        candidate[f"{split}_groups"] for split in ("train", "calibration", "validation", "test")
    ) != (8192, 512, 256, 512):
        raise ValueError("Scientific comparison requires the original fixed cohort sizes")
    if (
        record["identity"] != completed["identity"]
        or record["code_identity"] != completed["code_identity"]
        or record["data_identity"] != completed["data_identity"]
    ):
        raise ValueError("Configuration and completion identities disagree")
    expected_candidate = (
        "isotropic_innovation" if encoding == "isotropic" else "balanced_action_innovation"
    )
    if (
        candidate["candidate"] != expected_candidate
        or candidate["transport_chart"] != "isotropic"
        or (encoding == "balanced" and candidate.get("action_mix") != "paley24")
    ):
        raise ValueError("Candidate action encoding differs from the requested pairing")
    checkpoint = cast(
        dict[str, Any], torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    )
    steps = int(scientific["steps"])
    counters = (completed["steps"], completed["successful_updates"], checkpoint["step"])
    if (
        any(value != steps for value in counters)
        or completed["skipped_updates"] != 0
        or checkpoint["identity"] != completed["identity"]
        or not checkpoint["optimizer"]["state"]
        or any(int(value["step"]) != steps for value in checkpoint["optimizer"]["state"].values())
    ):
        raise ValueError("Checkpoint or optimizer update counts differ from the fixed endpoint")
    if checkpoint["initial_checkpoint_sha256"] != sha(directory / "initial.pt"):
        raise ValueError("Final checkpoint initial-source link differs")
    if (directory / "test.pt").exists() or (directory / "test.json").exists():
        raise ValueError("Paired candidate screen must not contain test inference artifacts")
    initial = cast(
        dict[str, Any], torch.load(directory / "initial.pt", map_location="cpu", weights_only=True)
    )
    for prefix, name in (
        ("online_encoder", "initial_encoder_identity"),
        ("predictor", "initial_predictor_identity"),
    ):
        if not (
            _identity(initial["model"], prefix) == initial[name] == completed[name] == record[name]
        ):
            raise ValueError("Initial tensor identity differs from its authenticated record")
    matching = read_json(directory / "matching.json")
    payloads = {
        split: cast(
            dict[str, Any],
            torch.load(directory / f"{split}.pt", map_location="cpu", weights_only=True),
        )
        for split in ("calibration", "validation")
    }
    for split, payload in payloads.items():
        sample = audit_samples(payload, int(candidate[f"{split}_groups"]))
        if payload["metrics"] != read_json(directory / f"{split}.json")["metrics"]:
            raise ValueError("Tensor and JSON metric records differ")
        if "graph_action" not in sample or sample["graph_action"].shape != (
            int(candidate[f"{split}_groups"]),
            6,
            24,
        ):
            raise ValueError("Malformed local routing: dense endpoint requires 24-action samples")
        routing_metrics(sample, matching)  # Includes the unchanged 24-action interface guard.
    calibration, validation = (payloads[name]["samples"] for name in ("calibration", "validation"))
    if set(calibration["group_ids"].tolist()) & set(validation["group_ids"].tolist()):
        raise ValueError("Calibration and validation parameter groups overlap")
    measured: dict[str, float] = {}
    for encoder, mcc_key in (("online", "state_mcc"), ("target", "target_state_mcc")):
        features, truth = (
            calibration[f"{encoder}_states"].flatten(0, 2),
            calibration["states"].flatten(0, 2),
        )
        columns = tuple(matching[f"{encoder}_columns"])
        if fit_coordinate_match(features, truth, method="pearson").columns != columns:
            raise ValueError("Saved matching was not fitted on calibration")
        probe = fit_state_probe(features, truth, ridge=1e-3)
        for split, payload in payloads.items():
            samples, reported = payload["samples"], payload["metrics"]
            scored, reference = (
                samples[f"{encoder}_states"].flatten(0, 2),
                samples["states"].flatten(0, 2),
            )
            mcc = mean_correlation_coefficient(CoordinateMatch(columns), scored, reference)
            readout = state_probe_metrics(probe.predict(scored), reference)
            _equal(mcc, reported[mcc_key], "MCC")
            _equal(
                readout["r2"], reported[f"{encoder}_state_probe/r2"], "calibration-only state R2"
            )
            if split == "validation":
                measured[mcc_key] = mcc
                measured[f"{encoder}_state_probe/r2"] = readout["r2"]
                health = _health(samples[f"{encoder}_states"])
                for name, value in health.items():
                    _equal(value, reported[f"{encoder}/{name}"], "state-code health")
                    measured[f"{encoder}/{name}"] = value
                if encoder == "target":
                    forecast = state_probe_metrics(
                        probe.predict(samples["predictions"].flatten(0, 2)),
                        samples["states"][:, :, 10:12].flatten(0, 2),
                    )
                    _equal(
                        forecast["r2"], reported["forecast_state_probe/r2"], "forecast readout R2"
                    )
                    measured["forecast_state_probe/r2"] = forecast["r2"]
    target = validation["targets"].double()
    variance = float(target.flatten(0, 2).var(0, correction=0).mean())
    if variance <= 0:
        raise ValueError("Forecast normalization requires nonconstant targets")
    mse = float((validation["predictions"].double() - target).square().mean())
    measured.update(prediction=mse, normalized_prediction_mse=mse / variance)
    _equal(
        measured["normalized_prediction_mse"],
        payloads["validation"]["metrics"]["normalized_prediction_mse"],
        "normalized forecast error",
        rounding=True,
    )
    return {
        "directory": str(directory),
        "completion": completed,
        "config": record,
        "matching": matching,
        "splits": payloads,
        "metrics": measured,
        "completion_sha256": sha(directory / "completion.json"),
    }


def compare(
    isotropic_dir: Path,
    balanced_dir: Path,
    output: Path,
    *,
    draws: int = 1000,
    allow_smoke: bool = False,
) -> dict[str, Any]:
    """Authenticate the paired dense control and bootstrap validation groups jointly."""
    original = _load(isotropic_dir.resolve(), "isotropic", allow_smoke=allow_smoke)
    balanced = _load(balanced_dir.resolve(), "balanced", allow_smoke=allow_smoke)
    if original["config"]["scientific_config"] != balanced["config"]["scientific_config"]:
        raise ValueError("Scientific training settings differ between the paired encodings")
    original_settings, balanced_settings = (
        item["config"]["candidate_settings"] for item in (original, balanced)
    )
    if (
        original_settings | {"candidate": "balanced_action_innovation", "action_mix": "paley24"}
        != balanced_settings
    ):
        raise ValueError("Physical DGP or cohort settings differ between the paired encodings")
    for name in ("initial_encoder_identity", "initial_predictor_identity"):
        if original["completion"][name] != balanced["completion"][name]:
            raise ValueError("Initial encoder/predictor tensors are not paired")
    for split in ("calibration", "validation"):
        for name in ("states", "group_ids"):
            left, right = (item["splits"][split]["samples"][name] for item in (original, balanced))
            if tensor_fingerprint(left) != tensor_fingerprint(right):
                raise ValueError(f"Physical {split} {name} are not bitwise paired")
    comparisons = [
        {
            "encoder": encoder,
            "left": "balanced",
            "right": "isotropic",
            **paired_mcc_interval(
                balanced["splits"]["validation"]["samples"][f"{encoder}_states"],
                original["splits"]["validation"]["samples"][f"{encoder}_states"],
                original["splits"]["validation"]["samples"]["states"],
                balanced["matching"][f"{encoder}_columns"],
                original["matching"][f"{encoder}_columns"],
                draws=draws,
            ),
        }
        for encoder in ("online", "target")
    ]
    arms: dict[str, Any] = {}
    for name, arm in (("isotropic", original), ("balanced", balanced)):
        metrics = arm["metrics"]
        arms[name] = {
            "directory": arm["directory"],
            "run_identity": arm["completion"]["identity"],
            "source_identity": arm["completion"]["code_identity"],
            "data_identity": arm["completion"]["data_identity"],
            "completion_sha256": arm["completion_sha256"],
            "artifact_sha256": arm["completion"]["hashes"],
            "runtime": arm["config"].get("runtime", {}),
            "metrics": metrics,
            "quality_and_headroom_without_parameter_use": (
                metrics["state_mcc"] < 0.95
                and metrics["normalized_prediction_mse"] <= 0.15
                and metrics["online_state_probe/r2"] >= 0.90
                and all(math.isfinite(value) for value in metrics.values())
                and all(
                    metrics[f"{encoder}/variance_min"] > 1e-12 for encoder in ("online", "target")
                )
            ),
        }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "profile": "smoke" if allow_smoke else "exploratory",
        "paired_physical_samples": True,
        "paired_initial_encoder_predictor": True,
        "different_training_gpu": original["config"].get("runtime", {}).get("gpu_name")
        != balanced["config"].get("runtime", {}).get("gpu_name"),
        "primary_split": "validation",
        "arms": arms,
        "comparisons": comparisons,
        "analysis_source_sha256": sha(Path(__file__)),
        "analysis_dependency_sha256": {
            name: sha(Path(__file__).resolve().parents[2] / name)
            for name in (
                "diagnostics/isotropic_oracle/analysis.py",
                "diagnostics/oracle_state_learning/analysis.py",
                "src/scjepa/eval/recovery.py",
                "src/scjepa/eval/state_probes.py",
            )
        },
        "limitations": [
            "The paired group intervals condition on these fixed one-seed models; they do not "
            "measure training-seed variability.",
            "The runs may use different GPUs. Equal initial tensors and settings do not guarantee "
            "identical floating-point optimization trajectories across hardware.",
            "This comparison directly authenticates saved calibration/validation physical samples "
            "and initial weights. Cache transformation lineage is enforced by the frozen "
            "preparation/loading code, not independently reopened here.",
            "Parameter predictive use requires the separate frozen parameter-use audit; "
            "quality/headroom alone does not authorize extension.",
            "This adaptive validation control neither establishes full > baseline > dense nor "
            "opens a scientific test split.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    """Compare two completed dense artifact directories without model inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isotropic-dir", type=Path, required=True)
    parser.add_argument("--balanced-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = compare(args.isotropic_dir, args.balanced_dir, args.output_dir, draws=args.draws)
    print(json.dumps(result["comparisons"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
