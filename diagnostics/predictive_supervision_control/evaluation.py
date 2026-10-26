"""Keep raw recovery measurements and score every model at common prediction weights."""

import copy
from typing import Any, cast

from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.parameter_sigreg_control.evaluation import evaluate as base_evaluate
from scjepa.models.model import SCJEPA
from scjepa.training.loop import ObjectiveConfig

WEIGHTS = (1.0, 4.0, 16.0)


def weight_name(weight: float) -> str:
    """Give supported prediction coefficients stable JSON and W&B keys."""
    if weight not in WEIGHTS:
        raise ValueError("Prediction weight must be 1, 4 or 16")
    return "prediction_" + format(weight, "g")


def objective_scores(
    reference: dict[str, Any],
    training_weight: float,
    weights: tuple[float, ...] = WEIGHTS,
) -> dict[str, Any]:
    """Rescore the unchanged full SIGReg-0.4 objective using raw predictive MSE."""
    weight_name(training_weight)
    if not weights or len(set(weights)) != len(weights) or training_weight not in weights:
        raise ValueError("Distinct comparison weights must include the training weight")
    comparisons = tuple(dict.fromkeys((1.0, *weights)))
    for weight in comparisons:
        weight_name(weight)

    def score(row: dict[str, float]) -> dict[str, float]:
        reference_total = row["total_at_sigreg_0p4"]
        result = {key: value for key, value in row.items() if not key.startswith("total_at_")}
        for weight in comparisons:
            result["total_at_" + weight_name(weight)] = (
                reference_total + (weight - 1.0) * row["prediction"]
            )
        result["weighted_prediction"] = training_weight * row["prediction"]
        result["total_at_training_prediction_weight"] = result[
            "total_at_" + weight_name(training_weight)
        ]
        return result

    result = copy.deepcopy(reference)
    result["mean"] = score(cast(dict[str, float], reference["mean"]))
    result["by_projection_seed"] = [
        score(row) for row in cast(list[dict[str, float]], reference["by_projection_seed"])
    ]
    result.pop("training_parameter_sigreg_weight", None)
    result["parameter_sigreg_weight"] = 0.4
    result["consistency_weight"] = 1.0
    result["training_prediction_weight"] = training_weight
    result["comparison_prediction_weights"] = comparisons
    result["interpretation"] = (
        "Every model is scored at prediction weight one and every selected coefficient, "
        "including the same frozen-state SIGReg term. Parameter SIGReg is 0.4 and consistency "
        "is one. Compare raw predictive MSE and common-coefficient totals across models; "
        "different training-objective totals are not the same objective."
    )
    return result


def evaluate(
    model: SCJEPA,
    evaluation_data: EvaluationData,
    objective_config: ObjectiveConfig,
    *,
    prediction_weight: float = 1.0,
    prediction_weights: tuple[float, ...] = WEIGHTS,
    batch_size: int = 16,
    final: bool = False,
) -> dict[str, Any]:
    """Reuse calibrated whole-code probes, shuffling and all six interventions unchanged."""
    if (
        objective_config.parameter_regularizer != "sigreg"
        or objective_config.parameter_regularizer_weight != 0.4
        or objective_config.parameter_consistency_weight != 1.0
    ):
        raise ValueError("This comparison fixes parameter SIGReg at 0.4 and consistency at one")
    weight_name(prediction_weight)
    result = base_evaluate(
        model, evaluation_data, objective_config, batch_size=batch_size, final=final
    )
    objective = objective_scores(result["objective"], prediction_weight, prediction_weights)
    mean = cast(dict[str, float], objective["mean"])
    result.pop("total_at_training_parameter_sigreg")
    result.pop("totals_at_parameter_sigreg")
    result["prediction_weight"] = prediction_weight
    result["weighted_prediction"] = mean["weighted_prediction"]
    result["total_at_training_prediction_weight"] = mean["total_at_training_prediction_weight"]
    result["totals_at_prediction_weights"] = {
        weight_name(weight): mean["total_at_" + weight_name(weight)]
        for weight in cast(tuple[float, ...], objective["comparison_prediction_weights"])
    }
    result["objective"] = objective
    if final:
        result["limits"] = [
            "Whole-code probes assess recovery without requiring disentangled coordinates; "
            "finite probes do not certify injectivity or absence of information.",
            "Raw gradient measurements precede clipping and Adam preconditioning; large "
            "norm ratios alone do not establish the cause of information loss.",
            "State encoders remain frozen. This control tests acquisition and use of parameter "
            "information, not changes in state representations or sparse graphs.",
        ]
    return result
