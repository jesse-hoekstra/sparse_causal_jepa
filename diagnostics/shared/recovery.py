"""Calibration-fitted coordinate matching and whole-state recovery."""

from diagnostics.shared.oracle import EvaluationResult
from scjepa.eval.recovery import (
    CoordinateMatch,
    fit_coordinate_match,
    mean_correlation_coefficient,
    pearson_matrix,
)
from scjepa.eval.state_probes import fit_state_probe, state_probe_metrics

NORMALIZED_MSE_CAP = 0.05


def recovery_metrics(
    calibration: EvaluationResult, result: EvaluationResult
) -> tuple[dict[str, float], dict[str, list[int]]]:
    """Fit assignment and a fixed ridge readout on calibration rows alone."""
    matches: dict[str, CoordinateMatch] = {}
    metrics: dict[str, float] = {}
    truth = calibration.samples["states"].reshape(-1, 6)
    scored_truth = result.samples["states"].reshape(-1, 6)
    for prefix in ("online", "target"):
        features = calibration.samples[f"{prefix}_states"].reshape(-1, 6)
        scored = result.samples[f"{prefix}_states"].reshape(-1, 6)
        match = fit_coordinate_match(features, truth, method="pearson")
        matches[prefix] = match
        key = "state_mcc" if prefix == "online" else "target_state_mcc"
        metrics[key] = mean_correlation_coefficient(match, scored, scored_truth)
        correlations = pearson_matrix(scored, scored_truth)
        metrics.update(
            {
                f"{prefix}_state_correlation/coordinate_{index + 1}": float(
                    correlations[index, column]
                )
                for index, column in enumerate(match.columns)
            }
        )
        probe = fit_state_probe(features, truth, ridge=1e-3)
        scores = state_probe_metrics(probe.predict(scored), scored_truth)
        metrics.update({f"{prefix}_state_probe/{name}": value for name, value in scores.items()})
        if prefix == "target":
            predicted = result.samples["predictions"].reshape(-1, 6)
            future_truth = result.samples["states"][:, :, 10:12].reshape(-1, 6)
            scores = state_probe_metrics(probe.predict(predicted), future_truth)
            metrics.update(
                {f"forecast_state_probe/{name}": value for name, value in scores.items()}
            )
    metrics["normalized_prediction_mse"] = result.metrics["forecast/normalized_mse"]
    metrics["forecast_quality_pass"] = float(
        metrics["normalized_prediction_mse"] <= NORMALIZED_MSE_CAP
    )
    return metrics, {f"{prefix}_columns": list(match.columns) for prefix, match in matches.items()}
