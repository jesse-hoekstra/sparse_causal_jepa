"""Individual-context retention and inferred-code responses in learned coordinates."""

import copy
from typing import Any

import torch
from torch import Tensor

from diagnostics.joint_acquisition.baseline import (
    DiagnosticSelection,
    evaluate_responses,
    validation_retention,
)
from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.multiquery_control.evaluation import encode_codes, parameter_readouts
from diagnostics.oracle_code_control.evaluation import matched_batch
from scjepa.models.model import SCJEPA
from scjepa.training.rng import preserve_rng


def assess_model(
    model: SCJEPA,
    data: EvaluationData,
    selection: DiagnosticSelection,
    *,
    bootstrap_samples: int,
    seed: int,
    final: bool = False,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Evaluate inferred codes only; exact theta serves solely to generate counterfactuals."""
    with preserve_rng(next(model.parameters()).device), torch.no_grad():
        candidate = copy.deepcopy(model).double().eval()
        split = "test" if final else "validation"
        # The shared builder explicitly handles test; its supplied-code outputs are discarded.
        batch, _, _ = matched_batch(
            candidate, data, split, selection.validation_indices, final=final
        )
        response, raw = evaluate_responses(
            candidate, batch, bootstrap_samples=bootstrap_samples, seed=seed
        )
        if final:
            names = ("calibration", "validation", "test")
            for first, second in (("calibration", "test"), ("validation", "test")):
                if bool(
                    (data[first]["theta"][:, None] == data[second]["theta"][None]).all(-1).any()
                ):
                    raise ValueError("Final-test groups overlap probe-fitting groups")
            retention = parameter_readouts(
                {name: encode_codes(candidate, data[name]) for name in names},
                {name: data[name] for name in names},
            )
            response["interpretation"] = "Fixed-budget independent-test inferred-code response"
        else:
            retention = validation_retention(candidate, data, selection)
            retention.pop("families")
        return {
            "response": response,
            "retention": retention,
            "mlp_architecture_diagnostic": True,
        }, raw


def measurement_metrics(report: dict[str, Any], *, final: bool = False) -> dict[str, float]:
    """Log whole-code retention and each factor's response/error confidence interval."""
    prefix = "test" if final else "validation"
    result: dict[str, float] = {}
    for index, value in enumerate(
        report["retention"]["selected_test_r2" if final else "selected_validation_r2"], 1
    ):
        result[f"{prefix}/theta_{index}/retention_r2"] = float(value)
    for horizon, value in enumerate(report["response"]["ordinary_mse_by_horizon"], 1):
        result[f"{prefix}/ordinary_mse/horizon_{horizon}"] = float(value)
    for row in report["response"]["per_parameter"]:
        for name in ("normalized_response_error", "predictive_benefit"):
            entry = row[name]
            key = f"{prefix}/theta_{row['parameter']}/{name}"
            if entry["estimate"] is not None:
                result[key] = float(entry["estimate"])
            if entry["interval_95"] is not None:
                result[key + "/lower95"], result[key + "/upper95"] = entry["interval_95"]
    return result
