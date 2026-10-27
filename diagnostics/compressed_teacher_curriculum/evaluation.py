"""Keep direct-code evaluation and separately report predeclared fixed-head readouts."""

import copy
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.joint_acquisition.baseline import DiagnosticSelection, selected_split
from diagnostics.mlp_acquisition_control.evaluation import assess_model as direct_assess_model
from diagnostics.mlp_acquisition_control.evaluation import measurement_metrics as direct_metrics
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiquery_control.evaluation import encode_codes, parameter_readouts
from diagnostics.multiteacher_curriculum.forward import Heads
from scjepa.models.model import SCJEPA
from scjepa.training.rng import preserve_rng

FEATURE_MAPS = ("direct_q", "fixed_head_1", "fixed_head_2", "fixed_head_3")


def _validation_score(result: dict[str, Any], parameter: int) -> float:
    method = result["selected_methods_per_parameter"][parameter]
    return float(result["families"][method]["validation_r2"][parameter])


@torch.no_grad()
def expanded_readouts(
    codes: dict[str, Tensor], splits: EvaluationData, heads: Heads, *, final: bool = False
) -> dict[str, Any]:
    """Choose feature map, probe family and tuning on validation, never held-out test scores.

    Feature maps use only the student's complete six-vector and fixed label-free
    compressor heads. No original teacher, context, averaged code or theta enters
    the feature computation. Theta is passed only to calibration-fitted probes.
    """
    if any(parameter.requires_grad for parameter in heads.parameters()):
        raise ValueError("Expanded readout heads must be frozen before physical-label evaluation")
    names = ("calibration", "validation", "test") if final else ("calibration", "validation")
    fixed_heads = copy.deepcopy(heads).cpu().double().eval()
    features: dict[str, dict[str, Tensor]] = {name: {} for name in FEATURE_MAPS}
    labels: EvaluationData = {}
    for name in names:
        values = codes[name].detach().cpu().double()
        if (
            values.ndim != 3
            or values.shape[1:] != (10, 6)
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError("Expanded recovery requires finite individual-context six-codes")
        features["direct_q"][name] = values
        mapped = fixed_heads(values)
        for index in range(3):
            features[f"fixed_head_{index + 1}"][name] = mapped[:, :, index]
        labels[name] = {"theta": cast(TensorSplit, splits[name])["theta"]}
    if not final:
        labels["test"] = labels["validation"]
        for values in features.values():
            values["test"] = values["validation"]
    results = {name: parameter_readouts(values, labels) for name, values in features.items()}
    # Each helper already chooses its family/hyperparameters by validation.
    chosen = [
        max(FEATURE_MAPS, key=lambda name, j=j: _validation_score(results[name], j))
        for j in range(6)
    ]
    score_name = "test" if final else "validation"
    per_map: dict[str, Any] = {}
    for name, result in results.items():
        per_map[name] = {
            "selected_validation_r2": [_validation_score(result, j) for j in range(6)],
            f"selected_{score_name}_r2": result["selected_test_r2"],
            "selected_methods_per_parameter": result["selected_methods_per_parameter"],
            "families": {
                family_name: {
                    "validation_r2": family["validation_r2"],
                    f"{score_name}_scored_r2": family["test_r2"],
                    "selected_hyperparameters_per_parameter": family[
                        "selected_hyperparameters_per_parameter"
                    ],
                }
                for family_name, family in result["families"].items()
            },
        }
    return {
        f"selected_{score_name}_r2": [
            results[name]["selected_test_r2"][j] for j, name in enumerate(chosen)
        ],
        "selection_validation_r2": [
            _validation_score(results[name], j) for j, name in enumerate(chosen)
        ],
        "selected_feature_maps_per_parameter": chosen,
        "selected_methods_per_parameter": [
            results[name]["selected_methods_per_parameter"][j] for j, name in enumerate(chosen)
        ],
        "selected_hyperparameters_per_parameter": [
            results[name]["families"][results[name]["selected_methods_per_parameter"][j]][
                "selected_hyperparameters_per_parameter"
            ][j]
            for j, name in enumerate(chosen)
        ],
        f"{score_name}_group_mse_by_parameter": [
            [
                results[name]["selected_test_group_mse_by_parameter"][group][j]
                for j, name in enumerate(chosen)
            ]
            for group in range(
                len(results[FEATURE_MAPS[0]]["selected_test_group_mse_by_parameter"])
            )
        ],
        "feature_map_results": per_map,
        "fixed_heads_sha256": module_fingerprint(heads),
        "contexts_per_group": 10,
        "protocol": (
            "NEW predeclared expanded family: direct6D q or one of three fixed6D heads(q), "
            "then existing linear/cubic/RFF readouts. Physical-label fitting on calibration "
            "only; feature map, family and tuning selected per parameter using validation "
            "only. Test scores cannot select feature maps. No teacher/context input to heads."
        ),
        "scope": (
            "Fixed-budget test reporting with validation-selected expanded probes; adaptive "
            "teacher-selection ancestry remains. Original direct-code result reported separately."
            if final
            else (
                "Selection-optimistic validation monitoring; no test accessed. This expanded "
                "family does not retroactively change any historical direct-code gate."
            )
        ),
    }


def assess_model(
    model: SCJEPA,
    data: EvaluationData,
    selection: DiagnosticSelection,
    *,
    heads: Heads,
    bootstrap_samples: int,
    seed: int,
    final: bool = False,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Preserve original retention/physical responses and add the fixed-head feature family."""
    report, raw = direct_assess_model(
        model, data, selection, bootstrap_samples=bootstrap_samples, seed=seed, final=final
    )
    with preserve_rng(next(model.parameters()).device), torch.no_grad():
        candidate = copy.deepcopy(model).double().eval()
        if final:
            names = ("calibration", "validation", "test")
            selected: EvaluationData = {name: data[name] for name in names}
        else:
            names = ("calibration", "validation")
            selected = {name: selected_split(data, selection, name) for name in names}
        codes = {name: encode_codes(candidate, selected[name]) for name in names}
        report["expanded_retention"] = expanded_readouts(codes, selected, heads, final=final)
    report["evaluation_scope"] = (
        "Original direct-code retention and response criterion unchanged; expanded fixed-head "
        "whole-code readouts are separately declared for this new curriculum."
    )
    return report, raw


def measurement_metrics(report: dict[str, Any], *, final: bool = False) -> dict[str, float]:
    """Log original direct metrics and explicitly named expanded whole-code retention."""
    values = direct_metrics(report, final=final)
    split = "test" if final else "validation"
    scores = report["expanded_retention"][f"selected_{split}_r2"]
    for parameter, score in enumerate(scores, 1):
        values[f"{split}/theta_{parameter}/expanded_retention_r2"] = float(score)
    values[f"{split}/expanded_all_six_ge_0_8"] = float(min(scores) >= 0.8)
    return values
