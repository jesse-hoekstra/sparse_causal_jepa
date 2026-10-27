"""Bound full-code diagnostic repairs and test their transfer to separate queries."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from diagnostics.joint_acquisition.baseline import MatchedBatch, predict_matched, response_report
from diagnostics.signal_path.geometry import (
    fit_code_directions,
    jacobian_columns,
    predict_rollout,
)
from scjepa.models.model import SCJEPA


@dataclass(frozen=True)
class CodeRepair:
    """Actual bounded displacement with finite held-out response evidence."""

    delta: Tensor
    fit_response: Tensor
    heldout_response: Tensor
    summary: dict[str, Any]


def _relative_error(response: Tensor, reference: Tensor) -> float | None:
    energy = float(reference.square().sum())
    if energy <= 0 or not math.isfinite(energy):
        return None
    return float((response - reference).square().sum()) / energy


def fit_bounded_code(
    fit_predict: Callable[[Tensor], Tensor],
    heldout_predict: Callable[[Tensor], Tensor],
    code: Tensor,
    fit_reference: Tensor,
    heldout_reference: Tensor,
    *,
    code_std: float,
    max_displacement_std: float = 0.1,
    rcond: float = 1e-6,
    fractions: tuple[float, ...] = (1.0, 0.1, 0.01),
) -> CodeRepair:
    """Fit finite response with a truncated-SVD tangent, then bound actual RMS.

    Both references are finite target changes from the same unperturbed state.
    Only fit_reference enters the fit. Fractions check actual nonlinear changes
    against the predicted tangent; no fraction is selected by held-out outcomes.
    """
    if not math.isfinite(code_std) or code_std <= 0:
        raise ValueError("Undefined correction: degenerate calibration code scale")
    if (
        not 0 < max_displacement_std <= 0.1
        or not fractions
        or any(not math.isfinite(f) or not 0 < f <= 1 for f in fractions)
    ):
        raise ValueError("Require displacement <=0.1 and finite fractions in (0,1]")
    if not all(bool(torch.isfinite(t).all()) for t in (code, fit_reference, heldout_reference)):
        raise FloatingPointError("Nonfinite code repair input")
    point = code.detach().clone()
    jacobian = jacobian_columns(fit_predict, point)
    if jacobian.shape[:-1] != fit_reference.shape:
        raise ValueError("Fit reference must match prediction output shape")
    directions, singular = fit_code_directions(
        jacobian.reshape(-1, point.numel()), fit_reference.reshape(-1, 1), rcond
    )
    delta = directions[:, 0]
    rms = float(delta.square().mean().sqrt())
    bound = max_displacement_std * code_std
    if rms > bound:
        delta = delta * (bound / rms)
    if not bool(torch.isfinite(delta).all()):
        raise FloatingPointError("Nonfinite fitted code correction")
    with torch.no_grad():
        original_fit, original_heldout = fit_predict(point), heldout_predict(point)
        fit_response = fit_predict(point + delta) - original_fit
        heldout_response = heldout_predict(point + delta) - original_heldout
        if heldout_response.shape != heldout_reference.shape:
            raise ValueError("Held-out reference must match prediction output shape")
        checks: list[dict[str, Any]] = []
        for fraction in fractions:
            expected = (jacobian @ delta) * fraction
            actual = fit_predict(point + delta * fraction) - original_fit
            checks.append(
                {
                    "fraction": fraction,
                    "finite_minus_tangent_rms": float((actual - expected).square().mean().sqrt()),
                    "relative_tangent_error": _relative_error(actual, expected),
                }
            )
    return CodeRepair(
        delta.detach(),
        fit_response.detach(),
        heldout_response.detach(),
        {
            "method": "truncated SVD finite-response tangent, radial RMS projection",
            "rcond": rcond,
            "code_std": code_std,
            "max_displacement_std": max_displacement_std,
            "displacement_rms_in_calibration_std": float(delta.square().mean().sqrt()) / code_std,
            "fit_relative_response_error": _relative_error(fit_response, fit_reference),
            "heldout_relative_response_error": _relative_error(heldout_response, heldout_reference),
            "singular_values": [float(value) for value in singular.detach().cpu()],
            "finite_checks": checks,
            "interpretation": "Local code accessibility within this bound; not shared acquisition.",
        },
    )


def bounded_code_report(
    model: SCJEPA,
    batch: MatchedBatch,
    *,
    code_std: float,
    rcond: float = 1e-6,
    bootstrap_samples: int = 1000,
    seed: int = 202610271,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Repair calibration donors on support queries and score disjoint recipients.

    Each physical counterfactual receives a full-code shift from the original
    donor code, not an extra unbounded shift from the changed encoder code.
    This is per-context accessibility, never shared or validation-group fitting.
    """
    if batch.split != "calibration":
        raise ValueError("Code fitting uses calibration groups only")
    if not math.isfinite(code_std) or code_std <= 0:
        return {"status": "undefined", "reason": "degenerate calibration code scale"}, {}
    fit, heldout = batch.select_queries("fit"), batch.select_queries("heldout")
    with torch.no_grad():
        original = predict_matched(model, batch)
    corrected = original.changed_predictions.detach().clone()
    shifts = torch.empty_like(original.changed_codes)
    details: list[dict[str, Any]] = []
    for group in range(len(batch.group_indices)):
        for donor in range(len(batch.donors)):
            code = original.codes[group, donor].detach()

            def predictor(
                value: Tensor, subset: MatchedBatch, group: int = group, donor: int = donor
            ) -> Tensor:
                history, actions = subset.history[group, donor], subset.query_actions[group, donor]
                return predict_rollout(model, history, value.expand(len(history), -1), actions)

            for factor in range(6):
                repair = fit_bounded_code(
                    lambda value: predictor(value, fit),
                    lambda value: predictor(value, heldout),
                    code,
                    fit.changed_targets[group, donor, factor] - fit.targets[group, donor],
                    heldout.changed_targets[group, donor, factor] - heldout.targets[group, donor],
                    code_std=code_std,
                    rcond=rcond,
                )
                with torch.no_grad():
                    corrected[group, donor, factor] = predictor(code + repair.delta, batch)
                shifts[group, donor, factor] = repair.delta
                details.append(
                    {
                        "group": batch.group_indices[group],
                        "donor": batch.donors[donor],
                        "parameter": factor + 1,
                        **repair.summary,
                    }
                )
    before, before_raw = response_report(
        original.predictions[:, :, batch.fit_queries :],
        original.changed_predictions[:, :, :, batch.fit_queries :],
        heldout,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    after, after_raw = response_report(
        original.predictions[:, :, batch.fit_queries :],
        corrected[:, :, :, batch.fit_queries :],
        heldout,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    summary = {
        "status": "measured",
        "before": before,
        "after": after,
        "per_donor_fits": details,
        "code_std": code_std,
        "max_displacement_std": 0.1,
        "group_split": "calibration",
        "fit_query_indices": [[int(value) for value in row] for row in fit.query_indices],
        "heldout_query_indices": [[int(value) for value in row] for row in heldout.query_indices],
        "scope": "Per-donor support-query fit and held-out-query transfer, not shared weights.",
    }
    raw = {
        "code_displacements": shifts.detach().cpu(),
        "finite_steps": batch.finite_steps.detach().cpu(),
        **{f"before/{key}": value for key, value in before_raw.items()},
        **{f"after/{key}": value for key, value in after_raw.items()},
    }
    return summary, raw
