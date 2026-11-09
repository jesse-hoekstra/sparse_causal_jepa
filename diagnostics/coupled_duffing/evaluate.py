"""Calibration-fitted validation measurements for the mechanical baseline screen."""

from typing import cast

import torch
from torch import Tensor

from diagnostics.masked_mlp_learned.run import (
    _design,  # pyright: ignore[reportPrivateUsage]
    _r2,  # pyright: ignore[reportPrivateUsage]
    _ridge,  # pyright: ignore[reportPrivateUsage]
    shuffled_codes,
)
from scjepa.eval.recovery import (
    fit_coordinate_match,
    mean_correlation_coefficient,
    pearson_matrix,
    spearman_matrix,
)
from scjepa.eval.state_probes import fit_state_probe
from scjepa.models.model import SCJEPA, ModelOutput


@torch.no_grad()
def collect(
    model: SCJEPA,
    inputs: dict[str, Tensor],
    batch_size: int,
    *,
    limit: int | None = None,
    shuffle: bool = False,
) -> dict[str, Tensor]:
    """Collect predictions and single-context codes without reading any reference labels."""
    model.eval()
    device = next(model.parameters()).device
    count = (
        len(inputs["observations"]) if limit is None else min(limit, len(inputs["observations"]))
    )
    saved: dict[str, list[Tensor]] = {
        name: [] for name in ("states", "codes", "targets", "predictions")
    }
    for start in range(0, count, batch_size):
        x = inputs["observations"][start : start + batch_size].to(device)
        a = inputs["actions"][start : start + batch_size].to(device)
        if shuffle:
            with shuffled_codes(model, x.shape[1]):
                output = cast(ModelOutput, model(x, a))
        else:
            output = cast(ModelOutput, model(x, a))
        for name, value in (
            ("states", output.online_states),
            ("codes", output.parameters),
            ("targets", output.targets),
            ("predictions", output.predictions),
        ):
            saved[name].append(value.detach().double().cpu())
    return {name: torch.cat(values) for name, values in saved.items()}


def prediction_mse(samples: dict[str, Tensor]) -> float:
    """Return the deterministic mean forecast error used for GECO calibration."""
    return float((samples["predictions"] - samples["targets"]).square().mean())


def selected_retention(
    calibration: Tensor, truth: Tensor, values: Tensor, scored_truth: Tensor
) -> Tensor:
    """Fit linear/cubic probes on half the calibration groups; select on the other half."""
    half = len(calibration) // 2
    train_x = calibration[:half].flatten(0, 1)
    select_x = calibration[half:].flatten(0, 1)
    train_y = truth[:half, None].expand(-1, calibration.shape[1], -1).flatten(0, 1)
    select_y = truth[half:, None].expand(-1, calibration.shape[1], -1).flatten(0, 1)
    test_y = scored_truth[:, None].expand(-1, values.shape[1], -1).flatten(0, 1)
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-8)
    train_x, select_x = (train_x - mean) / std, (select_x - mean) / std
    test_x = (values.flatten(0, 1) - mean) / std
    selection: list[Tensor] = []
    scored: list[Tensor] = []
    for family in ("linear", "cubic"):
        weights = _ridge(_design(train_x, family), train_y)
        selection.append(_r2(_design(select_x, family) @ weights, select_y))
        scored.append(_r2(_design(test_x, family) @ weights, test_y))
    choice = torch.stack(selection).argmax(0)
    return torch.stack(scored)[choice, torch.arange(truth.shape[-1])]


def metrics(
    calibration: dict[str, Tensor],
    validation: dict[str, Tensor],
    calibration_labels: dict[str, Tensor],
    validation_labels: dict[str, Tensor],
    context: int,
) -> dict[str, float]:
    """Score fixed assignments, state/code retention and nontrivial forecasting on validation."""
    cal_n, val_n = len(calibration["states"]), len(validation["states"])
    cal_reference = calibration_labels["states"][:cal_n].double()
    val_reference = validation_labels["states"][:val_n].double()
    cal_z = cal_reference.reshape(-1, 2)
    val_z = val_reference.reshape(-1, 2)
    cal_s, val_s = calibration["states"].reshape(-1, 2), validation["states"].reshape(-1, 2)
    output: dict[str, float] = {}
    for method in ("pearson", "spearman"):
        match = fit_coordinate_match(cal_s, cal_z, method=method)
        scores = (
            pearson_matrix(val_s, val_z) if method == "pearson" else spearman_matrix(val_s, val_z)
        )
        output[f"state_mcc/{method}"] = float(scores[torch.arange(2), list(match.columns)].mean())
        for i, column in enumerate(match.columns):
            output[f"state_correlation/{method}_{i + 1}"] = float(scores[i, column])
            output[f"calibration/{method}_column_{i + 1}"] = float(column)
    state_probe = fit_state_probe(cal_s, cal_z)
    retained = _r2(state_probe.predict(val_s), val_z)
    output["state_retention/linear_mean_r2"] = float(retained.mean())
    mean, std = cal_s.mean(0), cal_s.std(0).clamp_min(1e-8)
    weights = _ridge(_design((cal_s - mean) / std, "cubic"), cal_z)
    decoded = _design((val_s - mean) / std, "cubic") @ weights
    retained = _r2(decoded, val_z)
    output["state_retention/cubic_mean_r2"] = float(retained.mean())
    output["state_retention/cubic_min_r2"] = float(retained.min())
    coordinate_match = fit_coordinate_match(cal_s, cal_z, method="spearman")
    coordinate_scores: list[float] = []
    for i, column in enumerate(coordinate_match.columns):
        coordinate_weights = _ridge(
            _design(((cal_s - mean) / std)[:, column : column + 1], "cubic"),
            cal_z[:, i : i + 1],
        )
        estimate = (
            _design(((val_s - mean) / std)[:, column : column + 1], "cubic") @ coordinate_weights
        )
        score = float(_r2(estimate, val_z[:, i : i + 1])[0])
        coordinate_scores.append(score)
        output[f"coordinate_retention/cubic_r2_{i + 1}"] = score
    output["coordinate_retention/cubic_mean_r2"] = sum(coordinate_scores) / 2
    code_scores = selected_retention(
        calibration["codes"],
        calibration_labels["theta"][:cal_n],
        validation["codes"],
        validation_labels["theta"][:val_n],
    )
    output.update(
        {f"parameter_retention/r2_{i + 1}": float(value) for i, value in enumerate(code_scores)}
    )
    output["parameter_retention/min_r2"] = float(code_scores.min())
    output["parameter_retention/mean_r2"] = float(code_scores.mean())
    # The overcomplete acquisition code has no square coordinate assignment.
    # Complete-code retention is measured at both widths; MCC only after projection.
    if calibration["codes"].shape[-1] == 4:
        cal_theta = (
            calibration_labels["theta"][:cal_n, None].expand_as(calibration["codes"]).reshape(-1, 4)
        )
        val_theta = (
            validation_labels["theta"][:val_n, None].expand_as(validation["codes"]).reshape(-1, 4)
        )
        code_match = fit_coordinate_match(
            calibration["codes"].reshape(-1, 4), cal_theta, method="pearson"
        )
        output["parameter_mcc"] = mean_correlation_coefficient(
            code_match, validation["codes"].reshape(-1, 4), val_theta
        )
    predictions, targets = validation["predictions"], validation["targets"]
    variance = float(targets.reshape(-1, 2).var(0, correction=0).mean())
    output["forecast/mse"] = prediction_mse(validation)
    output["forecast/nmse"] = output["forecast/mse"] / max(variance, 1e-12)
    for time in (1, predictions.shape[2] - 1):
        output[f"forecast/nmse_step_{time + 1}"] = float(
            (predictions[:, :, time] - targets[:, :, time]).square().mean()
        ) / max(variance, 1e-12)
    raw = val_reference
    future = raw[:, :, context:]
    prediction_rows = predictions.reshape(-1, 2)
    cal_targets = calibration["targets"].reshape(-1, 2)
    target_mean = cal_targets.mean(0)
    target_std = cal_targets.std(0).clamp_min(1e-8)
    forecast_weights = _ridge(
        _design((cal_targets - target_mean) / target_std, "cubic"),
        cal_reference[:, :, context:].reshape(-1, 2),
    )
    decoded_forecast = (
        _design((prediction_rows - target_mean) / target_std, "cubic") @ forecast_weights
    )
    output["forecast/physical_cubic_r2"] = float(
        _r2(decoded_forecast, future.reshape(-1, 2)).mean()
    )
    horizon = torch.arange(1, future.shape[2] + 1, dtype=torch.float64)[None, None, :, None]
    extrapolated = raw[:, :, context - 1, None] + horizon * (
        raw[:, :, context - 1, None] - raw[:, :, context - 2, None]
    )
    output["forecast/constant_velocity_r2"] = float(
        _r2(extrapolated.reshape(-1, 2), future.reshape(-1, 2)).mean()
    )
    output["state_std/min"] = float(val_s.std(0).min())
    eigenvalues = cast(Tensor, torch.linalg.eigvalsh(torch.cov(val_s.T)))  # pyright: ignore[reportUnknownMemberType]
    output["state_covariance/min_eigenvalue"] = float(eigenvalues.min())
    output["code_std/min"] = float(
        validation["codes"].reshape(-1, validation["codes"].shape[-1]).std(0).min()
    )
    return output
