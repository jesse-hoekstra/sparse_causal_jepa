"""Evaluate paired parameter signs without replacing the original predictive-use criterion."""

import argparse
import copy
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.joint_acquisition.baseline import (
    DiagnosticSelection,
    MatchedBatch,
    _interval,  # pyright: ignore[reportPrivateUsage]
    build_matched_batch,
    predict_matched,
    response_report,
)
from diagnostics.joint_acquisition.corrections import CASES, CorrectionCase
from diagnostics.joint_acquisition.run import Config, code_identity
from diagnostics.joint_acquisition.source import load_comparison_sources, load_primary_source
from diagnostics.multiquery_control.data import EvaluationData, module_fingerprint
from diagnostics.multiquery_control.run import sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.rng import preserve_rng


def verify_completed(output: Path) -> dict[str, Any]:
    """Authenticate completed artifacts before loading models or measurements."""
    completion = cast(dict[str, Any], read_json(output / "completed.json"))
    if "results.json" not in completion["outputs"]:
        raise ValueError("Completion manifest must authenticate results")
    for name, expected in completion["outputs"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or sha(path) != expected:
            raise ValueError(f"Completed artifact changed: {name}")
    result = cast(dict[str, Any], read_json(output / "results.json"))
    if result["identity"] != completion["identity"]:
        raise ValueError("Completed result identity changed")
    return result


def paired_sign_report(
    base: Tensor,
    positive: Tensor,
    negative: Tensor,
    plus_batch: MatchedBatch,
    minus_batch: MatchedBatch,
    *,
    bootstrap_samples: int = 1000,
    seed: int = 202610261301,
) -> dict[str, Any]:
    """Retain one-sided results and pair signs within groups before resampling."""
    if base.shape[-2:] != (2, 6):
        raise ValueError("Sensitivity report requires the two-step six-state rollout")
    if (
        plus_batch.group_indices != minus_batch.group_indices
        or plus_batch.split != minus_batch.split
        or plus_batch.donors != minus_batch.donors
    ):
        raise ValueError("Both signs must use the same groups and split")
    if not torch.equal(plus_batch.finite_steps, -minus_batch.finite_steps):
        raise ValueError("Both signs must use exactly opposite physical steps")
    for name in (
        "targets",
        "history",
        "query_actions",
        "donor_states",
        "donor_actions",
        "query_indices",
    ):
        if not torch.equal(getattr(plus_batch, name), getattr(minus_batch, name)):
            raise ValueError(f"Both signs must share original {name}")
    reports: dict[str, Any] = {}
    for label, selection in (
        ("both_horizons", slice(None)),
        ("horizon_1", slice(0, 1)),
        ("horizon_2", slice(1, 2)),
    ):
        batches = [
            replace(
                batch,
                targets=batch.targets[..., selection, :],
                changed_targets=batch.changed_targets[..., selection, :],
            )
            for batch in (plus_batch, minus_batch)
        ]
        unchanged = base[..., selection, :]
        forecasts = (positive[..., selection, :], negative[..., selection, :])
        summaries: list[dict[str, Any]] = []
        raws: list[dict[str, Tensor]] = []
        for forecast, batch in zip(forecasts, batches, strict=True):
            summary, raw = response_report(
                unchanged, forecast, batch, bootstrap_samples=bootstrap_samples, seed=seed
            )
            summaries.append(summary)
            raws.append(raw)
        plus, minus = raws
        reference = plus["exact_response_energy"] + minus["exact_response_energy"]
        error = plus["response_error_energy"] + minus["response_error_energy"]
        benefit = (plus["predictive_benefit"] + minus["predictive_benefit"]) / 2
        count = unchanged[0].numel()
        fidelity = (reference - error) / (2 * count)
        cross = benefit - fidelity
        positive_fidelity = (plus["exact_response_energy"] - plus["response_error_energy"]) / count
        negative_fidelity = (
            minus["exact_response_energy"] - minus["response_error_energy"]
        ) / count
        cross_odd = (
            (plus["predictive_benefit"] - positive_fidelity)
            - (minus["predictive_benefit"] - negative_fidelity)
        ) / 2
        generator = torch.Generator().manual_seed(seed)
        samples = torch.randint(len(base), (bootstrap_samples, len(base)), generator=generator)
        ones = torch.ones(len(base), dtype=torch.float64)
        rows: list[dict[str, Any]] = []
        for factor in range(6):
            rows.append(
                {
                    "parameter": factor + 1,
                    "normalized_response_error": _interval(
                        error[:, factor], reference[:, factor], samples
                    ),
                    "predictive_benefit": _interval(benefit[:, factor], ones, samples),
                    "fidelity_contribution": _interval(fidelity[:, factor], ones, samples),
                    "residual_cross_even": _interval(cross[:, factor], ones, samples),
                    "residual_cross_odd": _interval(cross_odd[:, factor], ones, samples),
                }
            )
        reports[label] = {
            "positive_original_criterion": summaries[0],
            "negative_sensitivity": summaries[1],
            "paired_sensitivity": {
                "per_parameter": rows,
                "ordinary_mse_by_horizon": summaries[0]["ordinary_mse_by_horizon"],
            },
        }
    return {
        "reports": reports,
        "finite_steps_positive": [
            [float(value) for value in row] for row in plus_batch.finite_steps.cpu()
        ],
        "groups": list(plus_batch.group_indices),
        "bootstrap": {
            "unit": "independent parameter group; signs paired before resampling",
            "samples": bootstrap_samples,
            "seed": seed,
        },
        "interpretation": (
            "Paired signs are an evaluation sensitivity analysis, not a replacement for "
            "the original positive-perturbation success criterion. No fitting or final-test access."
        ),
    }


@torch.no_grad()
def evaluate_model(
    model: SCJEPA,
    data: EvaluationData,
    selection: DiagnosticSelection,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Use the same held-out recipients, exact two-step targets and opposite signs."""
    plus = build_matched_batch(
        model, data, selection, "validation", displacement=0.01
    ).select_queries("heldout")
    minus = build_matched_batch(
        model, data, selection, "validation", displacement=0.01, perturbation_sign=-1
    ).select_queries("heldout")
    positive, negative = predict_matched(model, plus), predict_matched(model, minus)
    torch.testing.assert_close(positive.predictions, negative.predictions, rtol=0, atol=0)
    return paired_sign_report(
        positive.predictions,
        positive.changed_predictions,
        negative.changed_predictions,
        plus,
        minus,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def verify_positive_reproduction(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Require the retained original measurement to reproduce its recorded values."""
    values: list[tuple[float, float]] = list(
        zip(actual["ordinary_mse_by_horizon"], expected["ordinary_mse_by_horizon"], strict=True)
    )
    if (
        actual["group_indices"] != expected["group_indices"]
        or actual["query_indices"] != expected["query_indices"]
    ):
        raise ValueError("Original positive measurement uses different groups or queries")
    for current, original in zip(actual["per_parameter"], expected["per_parameter"], strict=True):
        if current["parameter"] != original["parameter"]:
            raise ValueError("Original positive measurement parameter order changed")
        for key in ("normalized_response_error", "predictive_benefit"):
            for field in ("estimate", "interval_95"):
                new, old = current[key][field], original[key][field]
                if new is None or old is None:
                    if new != old:
                        raise ValueError("Original positive measurement definedness changed")
                elif field == "interval_95":
                    values.extend(zip(new, old, strict=True))
                else:
                    values.append((new, old))
    if any(not math.isclose(new, old, rel_tol=1e-9, abs_tol=1e-12) for new, old in values):
        raise ValueError("Original positive measurement did not reproduce")


def _metrics(label: str, audit: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for horizon, reports in audit["reports"].items():
        for sign, report in reports.items():
            for index, value in enumerate(report["ordinary_mse_by_horizon"], 1):
                metrics[f"{label}/{horizon}/{sign}/ordinary_mse_{index}"] = value
            for row in report["per_parameter"]:
                for name in ("normalized_response_error", "predictive_benefit"):
                    estimate = row[name]
                    prefix = f"{label}/{horizon}/{sign}/theta_{row['parameter']}/{name}"
                    if estimate["estimate"] is not None:
                        metrics[prefix] = estimate["estimate"]
                    if estimate["interval_95"] is not None:
                        metrics[prefix + "/lower_95"], metrics[prefix + "/upper_95"] = estimate[
                            "interval_95"
                        ]
    return metrics


def load_repaired_model(
    primary: SCJEPA,
    payload: dict[str, Any],
    *,
    case: CorrectionCase,
    parent_identity: str,
    trajectory_record: dict[str, Any],
) -> SCJEPA:
    """Require a fixed-budget endpoint with the original frozen and inactive modules."""
    original_modules = {
        name: module_fingerprint(module)
        for name, module in (
            ("online", primary.online_encoder),
            ("target", primary.target_encoder),
            ("encoder", primary.parameter_encoder),
            ("predictor", primary.predictor),
        )
    }
    if (
        payload["status"] != "complete"
        or payload["step"] != 1000
        or payload["protocol"]["parent_identity"] != parent_identity
        or payload["protocol"]["source_model"] != original_modules
    ):
        raise ValueError("Require authenticated fixed-budget trajectories from the same source")
    if (
        payload["protocol"]["case"] != case
        or payload["identity"] != trajectory_record["identity"]
        or trajectory_record["status"] != "complete"
        or trajectory_record["updates"] != 1000
    ):
        raise ValueError("Trajectory case or identity differs from the completed result")
    candidate = copy.deepcopy(primary)
    candidate.load_state_dict(payload["model"], strict=True)
    for name, module in (
        ("online", candidate.online_encoder),
        ("target", candidate.target_encoder),
    ):
        if module_fingerprint(module) != original_modules[name]:
            raise ValueError("Diagnostic repair changed frozen state maps")
    inactive = {
        "encoder": ("predictor", candidate.predictor),
        "predictor": ("encoder", candidate.parameter_encoder),
    }.get(case)
    if inactive is not None and module_fingerprint(inactive[1]) != original_modules[inactive[0]]:
        raise ValueError("Diagnostic repair changed an inactive parameter block")
    if candidate.get_extra_state() != primary.get_extra_state():
        raise ValueError("Diagnostic repair changed model prediction settings")
    return candidate


def audit(
    previous_output: Path,
    output: Path,
    *,
    trajectory_output: Path | None = None,
    device: str = "cpu",
    wandb_mode: str = "disabled",
) -> dict[str, Any]:
    """Evaluate verified primary, compatible and optional diagnostic-repair endpoints."""
    previous_output, output = previous_output.resolve(), output.resolve()
    previous = verify_completed(previous_output)
    protocol = cast(dict[str, Any], read_json(previous_output / "protocol.json"))
    keys = (
        "settings",
        "source_hashes",
        "training_code_identity",
        "diagnostic_code_identity",
        "selection",
        "torch",
        "comparison",
        "numerical_policy",
    )
    if canonical_fingerprint({key: protocol[key] for key in keys}) != previous["identity"]:
        raise ValueError("Original protocol identity changed")
    config = Config(**protocol["config"])
    settings = asdict(config)
    for key in (
        "source_dir",
        "output_dir",
        "training_source_root",
        "wandb_mode",
        "wandb_entity",
        "wandb_project",
    ):
        settings.pop(key)
    if settings != protocol["settings"]:
        raise ValueError("Original config differs from hashed scientific settings")
    if config.displacement != 0.01 or config.groups != 64 or config.verification:
        raise ValueError("Sensitivity audit requires the fixed 64-group scientific diagnostic")
    for source in (previous_output, Path(config.source_dir), trajectory_output):
        if source is not None and (
            output == source.resolve() or output.is_relative_to(source.resolve())
        ):
            raise ValueError("Sensitivity output must be outside original source/results")
    selection = DiagnosticSelection(
        tuple(protocol["selection"]["calibration_indices"]),
        tuple(protocol["selection"]["validation_indices"]),
        tuple(protocol["selection"]["donors"]),
        protocol["selection"]["fit_queries"],
        protocol["selection"]["heldout_queries"],
    )
    if selection != DiagnosticSelection():
        raise ValueError("Sensitivity audit requires original donor 0 and query episodes 5-9")
    target_device = torch.device(device)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with preserve_rng(target_device):
            bundle = load_primary_source(
                Path(config.source_dir),
                config.seed,
                training_source_root=Path(config.training_source_root)
                if config.training_source_root
                else None,
            )
            if bundle.hashes != protocol["source_hashes"]:
                raise ValueError("Primary source hashes changed")
            primary, _ = build_source_model(bundle, target_device)
            primary.double().eval().requires_grad_(False)
            if not (
                config.comparison_bundle and config.all_six_checkpoint and config.current_checkpoint
            ):
                raise ValueError("Actual compatible source/current checkpoints are required")
            compatible = load_comparison_sources(
                Path(config.comparison_bundle),
                Path(config.all_six_checkpoint),
                Path(config.current_checkpoint),
                device=target_device,
            )
            if compatible.identity != protocol["comparison"]:
                raise ValueError("Compatible checkpoint identity changed")
            models = {
                "primary_joint": (primary, bundle.evaluation),
                "compatible_source": (compatible.source_model, compatible.evaluation),
                "compatible_current": (compatible.current_model, compatible.evaluation),
            }
            expected_positive = {
                "primary_joint": previous["baseline"],
                "compatible_source": previous["compatible_control"]["source"],
                "compatible_current": previous["compatible_control"]["current"],
            }
            trajectory_hashes: dict[str, str] = {}
            if trajectory_output is not None:
                repaired = verify_completed(trajectory_output)
                if repaired["parent_identity"] != previous["identity"]:
                    raise ValueError("Repair trajectories belong to a different primary diagnostic")
                repaired_completion = cast(
                    dict[str, Any], read_json(trajectory_output / "completed.json")
                )
                for case in CASES:
                    path = trajectory_output / case / "last.pt"
                    if f"{case}/last.pt" not in repaired_completion["outputs"]:
                        raise ValueError(
                            "Completion manifest must authenticate trajectory checkpoint"
                        )
                    payload = cast(
                        dict[str, Any],
                        torch.load(path, map_location=target_device, weights_only=False),
                    )
                    candidate = load_repaired_model(
                        primary,
                        payload,
                        case=case,
                        parent_identity=repaired["identity"],
                        trajectory_record=repaired["trajectories"][case],
                    )
                    models[f"diagnostic_repaired_{case}"] = (candidate, bundle.evaluation)
                    trajectory_hashes[case] = sha(path)
                    expected_positive[f"diagnostic_repaired_{case}"] = repaired["trajectories"][
                        case
                    ]["records"][-1]["assessment"]
            evidence: dict[str, Any] = {
                "parent_identity": previous["identity"],
                "parent_results_sha256": sha(previous_output / "results.json"),
                "source_hashes": bundle.hashes,
                "comparison": compatible.identity,
                "trajectory_hashes": trajectory_hashes,
                "selection": asdict(selection),
                "displacement": 0.01,
                "boundary_policy": (
                    "h=min(0.01,0.5*(0.75-abs(theta_i))); use identical magnitude for both signs"
                ),
                "criterion": (
                    "Original positive criterion retained; paired signs are sensitivity only"
                ),
                "bootstrap_samples": config.bootstrap_samples,
                "seed": config.seed,
                "diagnostic_code_identity": code_identity(),
            }
            identity = canonical_fingerprint(evidence)
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "protocol.json", {"identity": identity, **evidence})
            logger = WandbLogger(
                WandbConfig(
                    mode=wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=output.parent.name,
                    job_type="diagnostic",
                ),
                resolved_config=evidence,
                dataset_identity=bundle.protocol["identity"],
                out_dir=output,
            )
            write_json(output / "tracking.json", logger.metadata)
            try:
                measured: dict[str, Any] = {}
                for label, (model, data) in models.items():
                    measured[label] = evaluate_model(
                        model,
                        data,
                        selection,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                    )
                    verify_positive_reproduction(
                        measured[label]["reports"]["both_horizons"]["positive_original_criterion"],
                        expected_positive[label],
                    )
                    measured[label]["original_positive_reproduced"] = True
                    logger.log(0, _metrics(label, measured[label]))
                result = {
                    "identity": identity,
                    "parent_identity": previous["identity"],
                    "models": measured,
                    "tracking": logger.metadata,
                    "interpretation": evidence["criterion"],
                }
                write_json(output / "results.json", result)
                logger.log_artifact(
                    output / "results.json", name=f"joint-symmetric-{identity[:12]}-results"
                )
                verify_completed(previous_output)
                if trajectory_output is not None:
                    verify_completed(trajectory_output)
                for name, expected in bundle.hashes.items():
                    if sha(Path(config.source_dir) / name) != expected:
                        raise RuntimeError("Primary source changed during sensitivity audit")
                logger.finish()
                write_json(
                    output / "completed.json",
                    {
                        "identity": identity,
                        "outputs": {
                            name: sha(output / name) for name in ("results.json", "protocol.json")
                        },
                    },
                )
                return result
            except BaseException:
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Run an evaluation-only sensitivity audit in a fresh output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    args = parser.parse_args()
    result = audit(
        args.previous_output,
        args.output_dir,
        trajectory_output=args.trajectory_output,
        device=args.device,
        wandb_mode=args.wandb_mode,
    )
    print(result["identity"])


if __name__ == "__main__":
    main()
