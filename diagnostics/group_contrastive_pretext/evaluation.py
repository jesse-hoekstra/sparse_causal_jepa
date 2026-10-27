"""Retention-only assessment; full-validation gate remains independent of final test."""

import copy
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.group_contrastive_pretext.forward import code_statistics, contrastive_loss
from diagnostics.joint_acquisition.baseline import DiagnosticSelection, selected_split
from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.multiquery_control.evaluation import encode_codes, parameter_readouts
from scjepa.models.model import SCJEPA
from scjepa.training.rng import preserve_rng


def assess_model(
    model: SCJEPA,
    data: EvaluationData,
    selection: DiagnosticSelection,
    *,
    full: bool = False,
    final: bool = False,
) -> dict[str, Any]:
    """Fit calibration probes/select validation; never evaluate the untrained predictor."""
    with preserve_rng(next(model.parameters()).device), torch.no_grad():
        candidate = copy.deepcopy(model).double().eval()
        names = ("calibration", "validation", "test") if final else ("calibration", "validation")
        splits = {
            name: data[name] if full or final else selected_split(data, selection, name)
            for name in ("calibration", "validation")
        }
        if final:
            splits["test"] = data["test"]
        if final:
            for first in ("calibration", "validation"):
                if bool(
                    (splits[first]["theta"][:, None] == splits["test"]["theta"][None]).all(-1).any()
                ):
                    raise ValueError("Test groups overlap fitting/selection groups")
        codes = {name: encode_codes(candidate, splits[name]) for name in names}
        if not final:
            splits["test"] = splits["validation"]
            codes["test"] = codes["validation"]
        retention = parameter_readouts(codes, splits)
        validation_scores = [
            float(retention["families"][method]["validation_r2"][j])
            for j, method in enumerate(retention["selected_methods_per_parameter"])
        ]
        if not final:
            retention = {
                "selected_validation_r2": validation_scores,
                "selected_methods_per_parameter": retention["selected_methods_per_parameter"],
                "families": {
                    name: {
                        "validation_r2": family["validation_r2"],
                        "selected_hyperparameters_per_parameter": family[
                            "selected_hyperparameters_per_parameter"
                        ],
                    }
                    for name, family in retention["families"].items()
                },
                "validation_group_mse_by_parameter": retention[
                    "selected_test_group_mse_by_parameter"
                ],
            }
        else:
            retention["selected_validation_r2"] = validation_scores
        evaluation_codes = codes["test" if final else "validation"]
        batches = [
            evaluation_codes[start : start + 16]
            for start in range(0, len(evaluation_codes), 16)
            if len(evaluation_codes[start : start + 16]) >= 2
        ]
        statistics = {
            key: float(torch.stack([code_statistics(batch)[key] for batch in batches]).mean())
            for key in code_statistics(batches[0])
        }
        statistics["contrastive"] = float(
            torch.stack([contrastive_loss(batch) for batch in batches]).mean()
        )
        return {
            "retention": retention,
            "pretext": statistics,
            "groups": {name: len(cast(Tensor, splits[name]["theta"])) for name in names},
            "full_validation_predictor_fit_gate": bool(min(validation_scores) >= 0.8)
            if full or final
            else None,
            "evaluation_scope": (
                "Retention only. P remains frozen/unfitted and predictive use is not assessed. "
                "Full512cal/256val all-six R2>=.8 alone gates any later P-fitting diagnostic; "
                "Test cannot set the gate. Nonfinal scores are selection-optimistic validation. "
                "Pretext metrics use sequential batches of16 groups; include final remainder if>=2."
            ),
        }


def measurement_metrics(report: dict[str, Any], *, final: bool = False) -> dict[str, float]:
    """Log each retained factor and held-out pretext behavior, without forecasting claims."""
    prefix = "test" if final else "validation"
    scores = report["retention"]["selected_test_r2" if final else "selected_validation_r2"]
    result = {f"{prefix}/theta_{j}/retention_r2": float(value) for j, value in enumerate(scores, 1)}
    result.update({f"{prefix}/{key}": float(value) for key, value in report["pretext"].items()})
    if report["full_validation_predictor_fit_gate"] is not None:
        result["validation/full_predictor_fit_gate"] = float(
            report["full_validation_predictor_fit_gate"]
        )
    return result
