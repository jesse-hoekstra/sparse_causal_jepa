"""Attached forecasts and exact coordinate conjugation for the existing global MLP.

The state coordinates and raw physical parameters are explicit oracle inputs.
Only the ordinary MLP learns; these helpers do not infer representations or
invent attention graphs for a model that has none.
"""

import copy
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from diagnostics.chart_sparsity.core import Chart, chart_matrix
from diagnostics.mlp_oracle_code_control.model import MLPPredictor
from diagnostics.oracle_state_predictor.data import OracleStateData
from diagnostics.oracle_state_predictor.run import forecast_metrics
from scjepa.training.rng import preserve_rng


def rollout(
    model: MLPPredictor, initial_states: Tensor, parameters: Tensor, actions: Tensor
) -> Tensor:
    """Reuse the same attached parameters and first prediction through both steps."""
    batch = len(initial_states)
    if initial_states.shape != (batch, 1, 6) or parameters.shape != (batch, 6):
        raise ValueError("require initial states (B,1,6) and raw parameters (B,6)")
    if actions.shape != (batch, 2, 16):
        raise ValueError("require observed actions (B,2,16)")
    state = initial_states
    predictions: list[Tensor] = []
    for horizon in range(2):
        prediction = model(state, parameters, actions[:, horizon : horizon + 1]).prediction
        predictions.append(prediction)
        state = prediction.unsqueeze(1)
    return torch.stack(predictions, dim=1)


@dataclass(frozen=True)
class EvaluationResult:
    """Prediction metrics and CPU samples retaining the independent group axis."""

    metrics: dict[str, float]
    samples: dict[str, Tensor]


@torch.no_grad()
def evaluate(
    model: MLPPredictor, data: OracleStateData, batch_size: int, device: torch.device
) -> EvaluationResult:
    """Evaluate a non-test split without changing model modes or caller RNG streams."""
    if batch_size < 1 or len(data) < 1:
        raise ValueError("batch size and number of groups must be positive")
    if data.config.split == "test":
        raise ValueError("this fixed-chart feasibility control does not perform test inference")
    reference = next(model.parameters())
    if reference.device != device:
        raise ValueError("model and requested evaluation device differ")
    flags = [(module, module.training) for module in model.modules()]
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    try:
        with preserve_rng(device):
            model.eval()
            for start in range(0, len(data), batch_size):
                indices = torch.arange(start, min(start + batch_size, len(data)))
                batch = data.batch(indices, device)
                predicted = rollout(
                    model,
                    batch.initial_states.to(dtype=reference.dtype),
                    batch.parameters.to(dtype=reference.dtype),
                    batch.actions.to(dtype=reference.dtype),
                ).cpu()
                predictions.append(predicted.reshape(batch.groups, batch.views, 2, 6))
                targets.append(batch.targets.cpu().reshape(batch.groups, batch.views, 2, 6))
    finally:
        for module, training in flags:
            module.training = training
    predicted, target = torch.cat(predictions), torch.cat(targets)
    samples = {
        "predictions": predicted,
        "targets": target,
        "group_ids": data.episode_ids.clone(),
        "group_horizon_mse": (predicted.double() - target.double()).square().mean(dim=(1, 3)),
    }
    if not all(bool(torch.isfinite(value).all()) for value in samples.values()):
        raise FloatingPointError("nonfinite MLP chart evaluation measurements")
    metrics = forecast_metrics(predicted.flatten(0, 1), target.flatten(0, 1))
    metrics["base_loss"] = metrics["prediction_mse"]
    return EvaluationResult(metrics, samples)


def _orthogonal_matrix(matrix: Tensor) -> Tensor:
    if matrix.shape != (6, 6) or not matrix.is_floating_point():
        raise ValueError("coordinate change must be a floating-point 6x6 matrix")
    orthogonal = matrix.detach().double()
    if not bool(torch.isfinite(orthogonal).all()):
        raise ValueError("coordinate change must be finite")
    tolerance = 1e-6 if matrix.dtype != torch.float64 else 1e-10
    if not torch.allclose(
        orthogonal.T @ orthogonal,
        torch.eye(6, dtype=torch.float64, device=orthogonal.device),
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError("coordinate change must be orthogonal")
    return orthogonal


def chart_transport(source: Chart, target: Chart) -> Tensor:
    """Return R with z_target=R z_source, using the declared physical state charts."""
    return chart_matrix(target) @ chart_matrix(source).T


@torch.no_grad()
def conjugate(model: MLPPredictor, matrix: Tensor) -> MLPPredictor:
    """Copy weights to represent R f(R.T z,a,theta) within the same MLP architecture.

    With input order [state, action, parameter], only the first linear layer's
    state columns and the last linear layer's output rows/bias change. The copy
    has no runtime inverse-map wrapper, new layers or optimization. The original
    model, its modes and its gradient flags are preserved, as are RNG streams.
    """
    matrix = _orthogonal_matrix(matrix)
    transformed = copy.deepcopy(model)
    first, last = transformed.network[0], transformed.network[-1]
    if not isinstance(first, nn.Linear) or not isinstance(last, nn.Linear):
        raise TypeError("MLP conjugation requires linear input and output layers")
    if (first.in_features, last.out_features) != (28, 6):
        raise ValueError("MLP conjugation requires the declared 28-input, 6-output architecture")
    if last.bias is None:  # pyright: ignore[reportUnnecessaryComparison]
        raise ValueError("MLP conjugation requires the declared biased output layer")
    matrix = matrix.to(device=first.weight.device)
    first.weight[:, :6].copy_((first.weight[:, :6].double() @ matrix.T).to(first.weight.dtype))
    last.weight.copy_((matrix @ last.weight.double()).to(last.weight.dtype))
    last.bias.copy_((matrix @ last.bias.double()).to(last.bias.dtype))
    return transformed


def closure_audit(
    source: EvaluationResult,
    transported: EvaluationResult,
    matrix: Tensor,
    *,
    max_prediction_error: float = 1e-4,
    max_mse_error: float = 1e-6,
) -> dict[str, float]:
    """Verify the analytic copy's forecasts and invariant MSE on paired groups.

    This numerical check is distinct from the scientific forecast-accuracy
    requirement. Its tolerances accommodate float32 arithmetic, including the
    stored chart conversion. It never selects checkpoints or changes weights.
    """
    if any(
        not math.isfinite(value) or value <= 0 for value in (max_prediction_error, max_mse_error)
    ):
        raise ValueError("closure tolerances must be finite and positive")
    matrix = _orthogonal_matrix(matrix).cpu()
    original, copied = source.samples, transported.samples
    if not torch.equal(original["group_ids"], copied["group_ids"]):
        raise ValueError("coordinate closure requires the same ordered parameter groups")
    for name in ("predictions", "targets"):
        if original[name].shape != copied[name].shape or original[name].shape[-2:] != (2, 6):
            raise ValueError("coordinate closure requires paired two-horizon forecast shapes")
        if not bool(torch.isfinite(original[name]).all() & torch.isfinite(copied[name]).all()):
            raise FloatingPointError("nonfinite coordinate-closure samples")
    expected_predictions = original["predictions"].double() @ matrix.T
    expected_targets = original["targets"].double() @ matrix.T
    prediction_error = float((expected_predictions - copied["predictions"].double()).abs().max())
    target_error = float((expected_targets - copied["targets"].double()).abs().max())
    if not torch.allclose(expected_targets, copied["targets"].double(), atol=1e-6, rtol=1e-6):
        raise ValueError("transported evaluation targets are not the declared coordinate change")
    actual_mses: list[float] = []
    for evaluation in (source, transported):
        sample = evaluation.samples
        actual = float(
            (sample["predictions"].double() - sample["targets"].double()).square().mean()
        )
        if not math.isclose(
            actual, evaluation.metrics["prediction_mse"], rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError("coordinate-closure MSE differs from raw saved forecasts")
        actual_mses.append(actual)
    mse_error = abs(actual_mses[1] - actual_mses[0])
    if prediction_error > max_prediction_error or mse_error > max_mse_error:
        raise ValueError("analytic MLP coordinate conjugation exceeded numerical tolerances")
    return {
        "max_prediction_absolute_error": prediction_error,
        "max_target_absolute_error": target_error,
        "pooled_mse_absolute_difference": mse_error,
        "prediction_tolerance": max_prediction_error,
        "mse_tolerance": max_mse_error,
        "passed": 1.0,
    }
