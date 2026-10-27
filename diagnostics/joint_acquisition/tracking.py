"""Publish saved diagnostic measurements even when numerical gates skip fitting."""

import math
from pathlib import Path
from typing import Any, Protocol


class ResultsLogger(Protocol):
    """Small logging interface shared by the runner and artifact backfills."""

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Record one scalar event."""
        ...

    def log_artifact(self, path: Path, *, name: str, artifact_type: str = "evaluation") -> None:
        """Publish one explicitly selected artifact."""
        ...


def result_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Select bounded scalar summaries; keep raw groups and full detail in artifacts."""
    metrics: dict[str, float] = {}

    def scalar(key: str, value: object) -> None:
        if isinstance(value, int | float) and math.isfinite(value):
            metrics[key] = float(value)

    def estimate(key: str, value: dict[str, Any]) -> None:
        scalar(key, value.get("estimate"))
        interval = value.get("interval_95")
        if interval is not None:
            scalar(key + "_lower_95", interval[0])
            scalar(key + "_upper_95", interval[1])

    def assessment(prefix: str, value: dict[str, Any]) -> None:
        for name in ("ordinary_mse_by_horizon", "ordinary_variance_normalized_mse_by_horizon"):
            for horizon, number in enumerate(value.get(name, []), 1):
                scalar(f"{prefix}/{name}/horizon_{horizon}", number)
        for row in value.get("per_parameter", []):
            for name in ("normalized_response_error", "predictive_benefit"):
                estimate(f"{prefix}/theta_{row['parameter']}/{name}", row[name])

    def retention(prefix: str, value: dict[str, Any]) -> None:
        for parameter, number in enumerate(value.get("selected_validation_r2", []), 1):
            scalar(f"{prefix}/theta_{parameter}/selected_validation_r2", number)

    scalar("diagnostic/derivative_checks_passed", result["numerical_checks"]["passed"])
    scalar("diagnostic/shared_checks_passed", result["shared_corrections"]["numerically_valid"])
    for key, value in result["numerical_checks"].get("geometry", {}).get("checks", {}).items():
        scalar(f"numerical/{key}", value)
    assessment("baseline", result["baseline"])
    retention("retention", result["retention"])
    for case, row in result["shared_corrections"]["cases"].items():
        prefix = f"shared/{case}"
        scalar(prefix + "/defined", row.get("defined", False))
        scalar(prefix + "/checks_passed", row.get("numerically_valid", False))
        for finite in row.get("finite_steps", []):
            step_prefix = f"{prefix}/fraction_{finite['fraction']:g}"
            for key in (
                "fit_loss",
                "predicted_fit_change",
                "actual_fit_change",
                "absolute_linearization_error",
                "linearization_valid",
            ):
                scalar(f"{step_prefix}/{key}", finite[key])
            predicted = abs(finite["predicted_fit_change"])
            if predicted > 0:
                scalar(
                    step_prefix + "/relative_linearization_error",
                    finite["absolute_linearization_error"] / predicted,
                )
    repair = result["code_repair"]
    if repair["status"] == "measured":
        assessment("code_repair/before", repair["before"])
        assessment("code_repair/after", repair["after"])
    comparison = result["compatible_control"]
    scalar("compatible/measured", comparison["status"] == "measured")
    if comparison["status"] == "measured":
        for kind in ("source", "current"):
            assessment(f"compatible/{kind}", comparison[kind])
        for horizon, value in enumerate(
            comparison["paired_source_minus_current_ordinary_mse_by_horizon"], 1
        ):
            scalar(f"compatible/source_minus_current_mse/horizon_{horizon}", value)
        for row in comparison["paired_per_parameter"]:
            for key in (
                "source_minus_current_normalized_response_error",
                "source_minus_current_predictive_benefit",
            ):
                estimate(f"compatible/theta_{row['parameter']}/{key}", row[key])
    for case, trajectory in result["trajectories"].items():
        prefix = f"trajectory/{case}"
        scalar(prefix + "/complete", trajectory["status"] == "complete")
        scalar(prefix + "/skipped", trajectory["status"] == "not_run_invalid_local_checks")
        records = trajectory.get("records", [])
        if records:
            final = records[-1]
            for key in ("step", "fit_loss", "code_displacement_rms", "total_weight_displacement"):
                scalar(f"{prefix}/final/{key}", final[key])
            assessment(prefix + "/final", final["assessment"])
            retention(prefix + "/final_retention", final["assessment"]["retention"])
    return metrics


def publish_results(logger: ResultsLogger, result: dict[str, Any], out: Path) -> None:
    """Log measured outcomes and upload only the saved JSON and readable report.

    The event counter is the largest actual recorded update, zero when all
    trajectories were skipped. Publishing never recomputes or rewrites results.
    """
    step = max(
        (
            int(row["step"])
            for trajectory in result["trajectories"].values()
            for row in trajectory.get("records", [])
        ),
        default=0,
    )
    logger.log(step, result_metrics(result))
    for filename, suffix in (("results.json", "results"), ("report.md", "report")):
        logger.log_artifact(
            out / filename, name=f"joint-acquisition-{result['identity'][:12]}-{suffix}"
        )
