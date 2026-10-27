"""Re-encoded matched validation/use and calibration-only state assignments."""

import copy
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.joint_state_survival.data import refresh_evaluation
from diagnostics.joint_state_survival.probes import fit_probes, predict, score
from diagnostics.joint_state_survival.source import response_gate
from diagnostics.mlp_acquisition_control.evaluation import assess_model
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit
from diagnostics.multiquery_control.evaluation import encode_codes
from diagnostics.predictor_transfer_precision.evaluation import evaluate
from scjepa.eval.recovery import fit_coordinate_match, mean_correlation_coefficient
from scjepa.models.model import SCJEPA
from scjepa.training.rng import preserve_rng


def state_metrics(data: EvaluationData, split: str) -> dict[str, Any]:
    """Fit the Pearson/Hungarian assignment on calibration only and score unchanged groups."""
    calibration, scored = (cast(TensorSplit, data[name]) for name in ("calibration", split))
    fit_x, fit_y = (
        calibration["states_online"].reshape(-1, 6),
        calibration["true_states"].reshape(-1, 6),
    )
    match = fit_coordinate_match(fit_x, fit_y, method="pearson")
    x, y = scored["states_online"].reshape(-1, 6), scored["true_states"].reshape(-1, 6)
    target = scored["states_target"].reshape(-1, 6)
    return {
        "mcc": mean_correlation_coefficient(match, x, y),
        "calibration_assignment": list(match.columns),
        "online_variance_per_coordinate": [float(value) for value in x.double().var(0)],
        "online_target_mse": float((x.double() - target.double()).square().mean()),
        "groups": len(scored["theta"]),
        "split": split,
        "scope": (
            "Current online chart; calibration-only Pearson/Hungarian assignment; no sparsity claim"
        ),
    }


def monitor(
    model: SCJEPA,
    data: EvaluationData,
    selection: DiagnosticSelection,
    *,
    bootstrap_samples: int,
    seed: int,
    final: bool = False,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Re-encode factual inputs before every original64 monitor or fixed endpoint test."""
    with preserve_rng(next(model.parameters()).device):
        refreshed = refresh_evaluation(
            model,
            data,
            ("calibration", "validation", "test") if final else ("calibration", "validation"),
        )
        report, raw = assess_model(
            model, refreshed, selection, bootstrap_samples=bootstrap_samples, seed=seed, final=final
        )
        report.pop("mlp_architecture_diagnostic")
        report["state"] = state_metrics(refreshed, "test" if final else "validation")
        return report, raw


def precision_assessment(
    model: SCJEPA,
    old_data: EvaluationData,
    precision_data: EvaluationData,
    *,
    bootstrap_samples: int,
    seed: int,
    chunk_size: int,
    progress: Callable[[int], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit on original512cal/select256val; score existing8192 without any fitting on it."""
    with preserve_rng(next(model.parameters()).device), torch.no_grad():
        old = refresh_evaluation(model, old_data, ("calibration", "validation"))
        precision = refresh_evaluation(model, precision_data, ("validation",))
        candidate = copy.deepcopy(model).double().eval()
        codes = {name: encode_codes(candidate, old[name]) for name in ("calibration", "validation")}
        probes = fit_probes(
            codes["calibration"],
            codes["validation"],
            old["calibration"]["theta"],
            old["validation"]["theta"],
        )
        old_retention, _ = score(predict(probes, codes["validation"]), old["validation"]["theta"])
        extra_codes = encode_codes(candidate, precision["validation"])
        retention, errors = score(predict(probes, extra_codes), precision["validation"]["theta"])
        response, raw, vectors = evaluate(
            candidate,
            precision,
            chunk_size=chunk_size,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            progress=progress,
        )
        state = state_metrics(old, "validation")
        return {
            "split": "existing_precision_validation",
            "original_full_validation_retention": old_retention,
            "precision_retention": retention,
            "response": response,
            "state": state,
            "all_six_source_gate": old_retention["all_six_above_0_80"] and response_gate(response),
            "scope": (
                "Same fixed physical validation groups, re-encoded current charts; no test "
                "or fresh32768 data; probes selected only on original "
                "calibration/validation"
            ),
        }, {
            "probes": probes,
            "precision_group_retention_mse": errors,
            "response": raw,
            "vectors": vectors,
        }


def paired_changes(
    source: dict[str, Any], final: dict[str, Any], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    """Report paired benefit changes, distinguishing lost significance from loss."""
    before, after = (
        cast(Tensor, source["response"]["predictive_benefit"]),
        cast(Tensor, final["response"]["predictive_benefit"]),
    )
    if before.shape != after.shape:
        raise ValueError("Paired use comparison requires identical group/factor axes")
    difference = after - before
    samples = torch.randint(
        len(before), (bootstrap_samples, len(before)), generator=torch.Generator().manual_seed(seed)
    )
    interval = torch.quantile(
        difference[samples].mean(1), torch.tensor([0.025, 0.975], dtype=difference.dtype), dim=0
    )
    return {
        "final_minus_source_predictive_benefit": [
            {
                "parameter": index + 1,
                "estimate": float(difference[:, index].mean()),
                "interval_95": [float(interval[0, index]), float(interval[1, index])],
            }
            for index in range(6)
        ],
        "scope": (
            "Paired existing8192 validation groups; a gate becoming inconclusive is "
            "distinct from a negative change confidence interval; target coordinates may "
            "move in live_state"
        ),
    }
