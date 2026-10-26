"""Compare information and predictive use while monitoring code rescaling.

Every checkpoint is scored at all five parameter SIGReg weights with consistency
fixed at one. Training with weight zero never uses the diagnostic SIGReg value.
Single-context probes use calibration-only centering and a common scalar rescale
to avoid mistaking uniform code shrinkage for lost information.
"""

import copy
import math
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint

# Reuse the existing evaluator without editing code bound to earlier run hashes.
from diagnostics.multiquery_control.evaluation import (
    _common_objective,  # pyright: ignore[reportPrivateUsage]
    _linear_progress,  # pyright: ignore[reportPrivateUsage]
    _parameter_interventions,  # pyright: ignore[reportPrivateUsage]
    _shuffle_metrics,  # pyright: ignore[reportPrivateUsage]
    _variance,  # pyright: ignore[reportPrivateUsage]
    encode_codes,
    parameter_readouts,
)
from scjepa.models.model import SCJEPA
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng

WEIGHTS = (0.4, 0.1, 0.0, 0.8, 1.6)


def weight_name(weight: float) -> str:
    """Give the supported coefficients stable JSON and W&B keys."""
    if weight not in WEIGHTS:
        raise ValueError("Parameter SIGReg weight must be 0.4, 0.1, 0, 0.8 or 1.6")
    return "sigreg_" + format(weight, "g").replace(".", "p")


@torch.no_grad()
def code_scale_diagnostics(model: SCJEPA, codes: Tensor) -> dict[str, float]:
    """Measure code scale and the predictor gain that can compensate for it.

    SPARTAN applies the same scalar-to-token Linear(1, embed_dim) to each code
    coordinate. Thus std_rms * ||W|| / sqrt(embed_dim) is exactly the RMS of
    projected-token standard deviations, centered separately for each coordinate.
    """
    if codes.ndim != 3 or codes.shape[-1] != 6:
        raise ValueError("Codes must have shape (groups, views, 6)")
    values = codes.detach().double().flatten(0, 1)
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Nonfinite parameter codes")
    standard_deviation = float(values.var(0, correction=0).mean().sqrt())
    projection_norm = float(
        model.predictor.param_project.weight.detach().double().square().sum().sqrt()
    )
    projection_gain = projection_norm / math.sqrt(model.predictor.param_project.out_features)
    return {
        "code_rms": float(values.square().mean().sqrt()),
        "code_centered_std_rms": standard_deviation,
        "code_coordinate_mean_rms": float(values.mean(0).square().mean().sqrt()),
        "parameter_projection_weight_norm": projection_norm,
        "effective_parameter_token_std_rms": standard_deviation * projection_gain,
    }


def normalize_probe_codes(codes: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Remove a global scale using calibration codes only; preserve mixed coordinates."""
    calibration = codes["calibration"].flatten(0, 1).double()
    center = calibration.mean(0)
    scale = calibration.var(0, correction=0).mean().sqrt()
    if not bool(torch.isfinite(scale)):
        raise FloatingPointError("Nonfinite calibration code scale")
    # Zero-variance codes remain zero. Nonzero scales are not floored at a fixed
    # absolute value, which would turn mere rescaling into artificial information loss.
    divisor = scale if float(scale) > 0 else scale.new_ones(())
    normalized = {name: (value.double() - center) / divisor for name, value in codes.items()}
    return normalized, {
        "calibration_global_std_rms": float(scale),
        "calibration_has_zero_variance": float(scale) == 0,
        "protocol": (
            "Calibration-only per-coordinate centering and one shared scalar rescale. "
            "No theta labels enter normalization; no averaging of contexts before probing. "
            "Prediction, interventions, consistency and SIGReg use the original raw codes."
        ),
    }


def objective_scores(reference: dict[str, Any], training_weight: float) -> dict[str, Any]:
    """Derive comparable full objectives from a common 0.4-SIGReg, consistency-1 score."""
    weight_name(training_weight)

    def score(row: dict[str, float]) -> dict[str, float]:
        raw_sigreg = row["parameter_regularizer"]
        reference_total = row["total_at_lambda1"]
        result = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "base",
                "total",
                "total_at_lambda0",
                "total_at_lambda1",
                "total_at_training_lambda",
            }
        }
        for weight in WEIGHTS:
            result["total_at_" + weight_name(weight)] = (
                reference_total + (weight - 0.4) * raw_sigreg
            )
        result["total_at_training_parameter_sigreg"] = result[
            "total_at_" + weight_name(training_weight)
        ]
        return result

    result = copy.deepcopy(reference)
    result["mean"] = score(cast(dict[str, float], reference["mean"]))
    result["by_projection_seed"] = [
        score(row) for row in cast(list[dict[str, float]], reference["by_projection_seed"])
    ]
    result["consistency_weight"] = 1.0
    result["training_parameter_sigreg_weight"] = training_weight
    result["interpretation"] = (
        "Each checkpoint is evaluated under all five parameter SIGReg coefficients, "
        "including the same frozen-state SIGReg constant and consistency weight one. "
        "Different training-objective totals must not be compared as the same objective."
    )
    return result


@torch.no_grad()
def evaluate(
    model: SCJEPA,
    evaluation_data: EvaluationData,
    objective_config: ObjectiveConfig,
    *,
    batch_size: int = 16,
    final: bool = False,
) -> dict[str, Any]:
    """Evaluate a CPU model copy without training, selecting checkpoints or changing RNG."""
    if batch_size < 2 or next(model.parameters()).device.type != "cpu":
        raise ValueError("Use a CPU evaluation copy and at least two independent groups")
    if objective_config.parameter_consistency_weight != 1.0:
        raise ValueError("This comparison fixes parameter consistency at one")
    weight = objective_config.parameter_regularizer_weight
    weight_name(weight)
    metadata = cast(dict[str, Any], evaluation_data["metadata"])
    for key, module in (
        ("state_online_sha256", model.online_encoder),
        ("state_target_sha256", model.target_encoder),
    ):
        if metadata[key] != module_fingerprint(module):
            raise ValueError(f"Frozen state cache mismatch: {key}")
    reference = copy.deepcopy(objective_config)
    reference.parameter_regularizer = "sigreg"
    reference.parameter_regularizer_weight = 0.4
    was_training = model.training
    model.eval()
    try:
        with preserve_rng(torch.device("cpu")):
            names = (
                ("calibration", "validation", "test") if final else ("calibration", "validation")
            )
            splits = {key: cast(TensorSplit, evaluation_data[key]) for key in names}
            codes = {
                key: encode_codes(model, split, max(128, batch_size))
                for key, split in splits.items()
            }
            probe_codes, normalization = normalize_probe_codes(codes)
            split_name = "test" if final else "validation"
            common = _common_objective(model, splits[split_name], reference, batch_size, final)
            objective = objective_scores(common, weight)
            mean = cast(dict[str, float], objective["mean"])
            linear = _linear_progress(
                probe_codes["calibration"],
                probe_codes[split_name],
                splits["calibration"]["theta"],
                splits[split_name]["theta"],
            )
            result: dict[str, Any] = {
                "split": split_name,
                "groups": len(codes[split_name]),
                "queries_per_code": 9 if final else 1,
                "parameter_sigreg_weight": weight,
                "consistency_weight": 1.0,
                "prediction_mse": mean["prediction"],
                "raw_parameter_sigreg": mean["parameter_regularizer"],
                "raw_consistency": mean["parameter_consistency"],
                "total_at_training_parameter_sigreg": mean["total_at_training_parameter_sigreg"],
                "totals_at_parameter_sigreg": {
                    weight_name(w): mean["total_at_" + weight_name(w)] for w in WEIGHTS
                },
                "parameter_code_variance": _variance(codes[split_name]),
                "code_scale": code_scale_diagnostics(model, codes[split_name]),
                "probe_normalization": normalization,
                "linear_r2": linear,
                "linear_r2_mean": sum(linear) / 6,
                "linear_r2_minimum": min(linear),
                "objective": objective,
            }
            if final:
                result["parameter_readouts"] = parameter_readouts(probe_codes, evaluation_data)
                shuffle = _shuffle_metrics(
                    model, splits["test"], codes["test"], max(128, batch_size)
                )
                if abs(float(shuffle["two_step_mse"]) - mean["prediction"]) > 2e-6:
                    raise RuntimeError("All-query objective and shuffle predictor disagree")
                result["parameter_use_shuffle"] = shuffle
                result["true_parameter_interventions"] = _parameter_interventions(
                    model, evaluation_data, codes["test"], max(128, batch_size)
                )
                result["limits"] = [
                    "Small raw consistency can result from code shrinkage; inspect normalized "
                    "within/total variance, parameter projection gain and predictive use.",
                    "Finite joint probes do not certify injectivity or absence of information.",
                    "State encoders stay frozen; no state-MCC or sparse-graph improvement "
                    "is tested.",
                    "Raw parameter SIGReg is measured at zero training weight for evaluation only; "
                    "it contributes no model-training gradient in that arm.",
                ]
            return result
    finally:
        model.train(was_training)
        model.online_encoder.eval()
        model.target_encoder.eval()
