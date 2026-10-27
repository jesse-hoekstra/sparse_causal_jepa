"""Check smaller two-sided steps along saved diagnostic directions, without refitting."""

import argparse
import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.joint_acquisition.baseline import (
    DiagnosticSelection,
    build_matched_batch,
    evaluate_responses,
    predict_matched,
    response_loss,
    validation_retention,
)
from diagnostics.joint_acquisition.corrections import CASES, loss_gradients
from diagnostics.joint_acquisition.run import Config, code_identity
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.joint_acquisition.tracking import result_metrics
from diagnostics.joint_acquisition.trajectories import TrajectorySettings, run_trajectory
from diagnostics.multiquery_control.run import sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.rng import preserve_rng

FRACTIONS = (0.01, 0.001, 0.0001, 0.00001)
FINITE_RTOL = 0.05
FINITE_ATOL = 1e-8


def verified_previous(output: Path) -> dict[str, Any]:
    """Load only intact completed diagnostic artifacts, independent of tracking mode."""
    completion = cast(dict[str, Any], read_json(output / "completed.json"))
    if not {"results.json", "measurements.pt"}.issubset(completion["outputs"]):
        raise ValueError("Original completion must authenticate results and measurements")
    for name, expected in completion["outputs"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or sha(path) != expected:
            raise ValueError(f"Original diagnostic artifact changed: {name}")
    result = cast(dict[str, Any], read_json(output / "results.json"))
    if result["identity"] != completion["identity"]:
        raise ValueError("Original result identity changed")
    return result


def check_direction(
    model: nn.Module,
    loss: Callable[[nn.Module], Tensor],
    direction: Mapping[str, Tensor],
    saved_slope: float,
) -> dict[str, Any]:
    """Test a fixed direction against autograd and both signed finite secants.

    Passing requires agreement with the original saved slope and both secants
    at the smallest predeclared fraction. No fraction is selected
    using validation outcomes, and neither direction nor tolerance is refitted.
    """
    parameters = dict(model.named_parameters())
    if not direction or any(
        key not in parameters
        or value.shape != parameters[key].shape
        or not bool(torch.isfinite(value).all())
        for key, value in direction.items()
    ):
        raise ValueError("Saved direction has invalid names, shapes or values")
    point = loss(model)
    gradients = loss_gradients(point, {key: parameters[key] for key in direction})
    slope = sum(
        float((gradients[key] * value.to(gradients[key])).sum()) for key, value in direction.items()
    )
    initial = float(point.detach())
    if not math.isfinite(slope) or not math.isfinite(saved_slope):
        raise FloatingPointError("Nonfinite directional slope")
    saved_matches = math.isclose(slope, saved_slope, rel_tol=1e-8, abs_tol=1e-10)
    checks: list[dict[str, Any]] = []
    for fraction in FRACTIONS:
        row: dict[str, Any] = {"fraction": fraction}
        for name, sign in (("positive", 1.0), ("negative", -1.0)):
            candidate = copy.deepcopy(model)
            with torch.no_grad():
                for key, delta in direction.items():
                    dict(candidate.named_parameters())[key].add_(
                        delta.to(parameters[key]), alpha=sign * fraction
                    )
                changed = float(loss(candidate))
            actual, predicted = changed - initial, sign * fraction * slope
            error = abs(actual - predicted)
            row[name] = {
                "actual_change": actual,
                "predicted_change": predicted,
                "absolute_error": error,
                "relative_error": error / abs(predicted) if predicted else None,
                "secant_slope": actual / (sign * fraction),
                "valid": math.isfinite(error)
                and error <= FINITE_ATOL + FINITE_RTOL * abs(predicted),
            }
        row["central_slope"] = (
            row["positive"]["actual_change"] - row["negative"]["actual_change"]
        ) / (2 * fraction)
        checks.append(row)
    return {
        "initial_loss": initial,
        "saved_slope": saved_slope,
        "autograd_slope": slope,
        "saved_slope_matches": saved_matches,
        "finite_steps": checks,
        "numerically_valid": saved_matches
        and all(checks[-1][side]["valid"] for side in ("positive", "negative")),
    }


def followup(
    previous_output: Path,
    output: Path,
    *,
    device: str = "cpu",
    wandb_mode: str = "disabled",
    run_trajectories: bool = False,
) -> dict[str, Any]:
    """Recheck immutable directions and optionally run the original bounded trajectories."""
    previous_output, output = previous_output.resolve(), output.resolve()
    previous = verified_previous(previous_output)
    protocol = cast(dict[str, Any], read_json(previous_output / "protocol.json"))
    evidence_keys = (
        "settings",
        "source_hashes",
        "training_code_identity",
        "diagnostic_code_identity",
        "selection",
        "torch",
        "comparison",
        "numerical_policy",
    )
    if canonical_fingerprint({key: protocol[key] for key in evidence_keys}) != previous["identity"]:
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
        raise ValueError("Original config differs from hashed settings")
    if (
        output == previous_output
        or output.is_relative_to(previous_output)
        or output.is_relative_to(Path(config.source_dir).resolve())
    ):
        raise ValueError("Follow-up output must be outside original results and source")
    target_device = torch.device(device)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with preserve_rng(target_device):
            bundle = load_primary_source(
                Path(config.source_dir),
                config.seed,
                expected_step=config.expected_source_step,
                expected_checkpoint_sha256=config.expected_checkpoint_sha256,
                training_source_root=Path(config.training_source_root)
                if config.training_source_root
                else None,
            )
            if bundle.hashes != protocol["source_hashes"]:
                raise ValueError("Original source artifacts changed")
            model, _ = build_source_model(bundle, target_device)
            model.double().eval()
            values = protocol["selection"]
            selection = DiagnosticSelection(
                calibration_indices=tuple(values["calibration_indices"]),
                validation_indices=tuple(values["validation_indices"]),
                donors=tuple(values["donors"]),
                fit_queries=values["fit_queries"],
                heldout_queries=values["heldout_queries"],
            )
            measurements = cast(
                dict[str, Any],
                torch.load(
                    previous_output / "measurements.pt",
                    map_location=target_device,
                    weights_only=False,
                ),
            )
            directions = cast(dict[str, dict[str, Tensor]], measurements["shared_directions"])
            if set(directions) != set(CASES):
                raise ValueError("All original shared directions are required")
            calibration = build_matched_batch(
                model, bundle.evaluation, selection, "calibration", displacement=config.displacement
            )
            fitting = calibration.select_queries("fit")

            def fit_loss(candidate: nn.Module) -> Tensor:
                return response_loss(cast(SCJEPA, candidate), fitting)

            followup_protocol: dict[str, Any] = {
                "parent_identity": previous["identity"],
                "parent_results_sha256": sha(previous_output / "results.json"),
                "parent_measurements_sha256": sha(previous_output / "measurements.pt"),
                "source_hashes": bundle.hashes,
                "diagnostic_code_identity": code_identity(),
                "fractions": FRACTIONS,
                "finite_rtol": FINITE_RTOL,
                "finite_atol": FINITE_ATOL,
                "gate": (
                    "Saved/autograd slopes match; both signs pass at smallest fraction; "
                    "original physical derivative checks passed"
                ),
                "run_trajectories": run_trajectories,
                "trajectory_settings": asdict(
                    TrajectorySettings(
                        updates=config.steps, learning_rate=config.learning_rate, seed=config.seed
                    )
                ),
                "device": device,
            }
            identity = canonical_fingerprint(followup_protocol)
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "protocol.json", {"identity": identity, **followup_protocol})
            logger = WandbLogger(
                WandbConfig(
                    mode=wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=output.parent.name,
                    job_type="diagnostic",
                ),
                resolved_config=followup_protocol,
                dataset_identity=bundle.protocol["identity"],
                out_dir=output,
            )
            write_json(output / "tracking.json", logger.metadata)
            try:
                checks = {
                    case: check_direction(
                        model,
                        fit_loss,
                        directions[case],
                        previous["shared_corrections"]["cases"][case]["fit_directional_derivative"],
                    )
                    for case in CASES
                }
                valid = previous["numerical_checks"]["passed"] and all(
                    row["numerically_valid"] for row in checks.values()
                )
                scalars = {"followup/numerically_valid": float(valid)}
                for case, check in checks.items():
                    scalars[f"followup/{case}/valid"] = float(check["numerically_valid"])
                    for row in check["finite_steps"]:
                        for side in ("positive", "negative"):
                            value = row[side]["relative_error"]
                            if value is not None:
                                scalars[
                                    f"followup/{case}/{row['fraction']:g}/{side}_relative_error"
                                ] = value
                logger.log(0, scalars)
                write_json(
                    output / "finite_checks.json", {"numerically_valid": valid, "cases": checks}
                )
                trajectories: dict[str, Any] = {}
                if valid and run_trajectories:
                    validation = build_matched_batch(
                        model,
                        bundle.evaluation,
                        selection,
                        "validation",
                        displacement=config.displacement,
                    ).select_queries("heldout")

                    def assess(candidate: SCJEPA) -> dict[str, Any]:
                        summary, raw = evaluate_responses(
                            candidate,
                            validation,
                            bootstrap_samples=config.bootstrap_samples,
                            seed=config.seed,
                        )
                        return {
                            **summary,
                            "raw_groups": {key: value.tolist() for key, value in raw.items()},  # pyright: ignore[reportUnknownMemberType]
                            "retention": validation_retention(
                                candidate, bundle.evaluation, selection
                            ),
                        }

                    for case in CASES:

                        def progress(row: dict[str, Any], case: str = case) -> None:
                            snapshot = {
                                **previous,
                                "trajectories": {case: {"status": "running", "records": [row]}},
                            }
                            metrics = {
                                key.replace("/final/", "/milestone/").replace(
                                    "/final_retention/", "/retention/"
                                ): value
                                for key, value in result_metrics(snapshot).items()
                                if key.startswith(f"trajectory/{case}/")
                                and not key.endswith(("/complete", "/skipped"))
                            }
                            logger.log(int(row["step"]), metrics)

                        trajectories[case] = run_trajectory(
                            model,
                            fit_loss,
                            assess,
                            lambda candidate: predict_matched(candidate, calibration).codes,
                            case=case,
                            checkpoint_path=output / case / "last.pt",
                            identity=identity,
                            local_checks={"numerically_valid": valid},
                            settings=TrajectorySettings(
                                updates=config.steps,
                                learning_rate=config.learning_rate,
                                seed=config.seed,
                            ),
                            progress=progress,
                        )
                else:
                    trajectories = {
                        case: {
                            "status": "not_run_invalid_local_checks"
                            if not valid
                            else "not_requested"
                        }
                        for case in CASES
                    }
                result = {
                    "identity": identity,
                    "parent_identity": previous["identity"],
                    "numerically_valid": valid,
                    "finite_checks": checks,
                    "trajectories": trajectories,
                    "tracking": logger.metadata,
                }
                write_json(output / "results.json", result)
                # Publish final trajectory assessments alongside the original baseline metrics.
                metrics_source = {**previous, "trajectories": trajectories}
                logger.log(
                    config.steps if valid and run_trajectories else 0,
                    {
                        key if key.startswith("trajectory/") else f"parent/{key}": value
                        for key, value in result_metrics(metrics_source).items()
                    },
                )
                logger.log_artifact(
                    output / "results.json", name=f"joint-followup-{identity[:12]}-results"
                )
                logger.log_artifact(
                    output / "finite_checks.json",
                    name=f"joint-followup-{identity[:12]}-finite-checks",
                )
                verified_previous(previous_output)
                for name, expected in bundle.hashes.items():
                    if sha(Path(config.source_dir) / name) != expected:
                        raise RuntimeError("Source changed during follow-up")
                logger.finish()
                artifacts = ["results.json", "finite_checks.json", "protocol.json"]
                artifacts.extend(
                    f"{case}/last.pt" for case in CASES if (output / case / "last.pt").exists()
                )
                write_json(
                    output / "completed.json",
                    {
                        "identity": identity,
                        "outputs": {name: sha(output / name) for name in artifacts},
                    },
                )
                return result
            except BaseException:
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Run an explicitly separate follow-up, preserving all original artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-output", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    parser.add_argument("--run-trajectories", action="store_true")
    args = vars(parser.parse_args())
    print(followup(args.pop("previous_output"), args.pop("output_dir"), **args)["identity"])


if __name__ == "__main__":
    main()
