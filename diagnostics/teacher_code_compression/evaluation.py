"""Calibration-fitted single-context probes scored on validation only, never test."""

import copy
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.multiquery_control.data import TensorSplit
from diagnostics.multiquery_control.evaluation import parameter_readouts
from diagnostics.multiteacher_curriculum.forward import Heads
from diagnostics.multiteacher_curriculum.source import leave_one_out_targets
from diagnostics.teacher_code_compression.data import CompressionInputs
from diagnostics.teacher_code_compression.model import CompressionEncoder
from scjepa.training.rng import preserve_rng

Evaluation = dict[str, dict[str, Tensor]]


@torch.no_grad()
def prepare_evaluation(inputs: CompressionInputs, device: torch.device) -> Evaluation:
    """Infer teacher unions separately on calibration/validation; fit no transformations."""
    teachers = [copy.deepcopy(teacher).to(device).eval() for teacher in inputs.parent.teachers]
    reference = next(teachers[0].parameters())
    result: Evaluation = {}
    for name in ("calibration", "validation"):
        split = cast(TensorSplit, inputs.parent.bundle.evaluation[name])
        groups = len(split["states_online"])
        chunks: list[Tensor] = []
        for start in range(0, groups, 64):
            end = min(start + 64, groups)
            states = split["states_online"][start:end, :, :10].flatten(0, 1).to(reference)
            actions = split["actions"][start:end, :, :9].flatten(0, 1).to(reference)
            chunks.append(
                torch.stack(
                    [teacher(states, actions).reshape(end - start, 10, 6) for teacher in teachers],
                    dim=2,
                ).cpu()
            )
        raw = torch.cat(chunks)
        result[name] = {
            "features": ((raw - inputs.mean) / inputs.scale).flatten(-2),
            "targets": leave_one_out_targets(raw, inputs.mean, inputs.scale),
            "theta": split["theta"].detach().cpu().clone(),
        }
    return result


@torch.no_grad()
def assess(
    encoder: CompressionEncoder, heads: Heads, data: Evaluation, *, groups: int | None
) -> dict[str, Any]:
    """Use validation for both selection and scoring, with explicit selection optimism."""
    device = next(encoder.parameters()).device
    with preserve_rng(device):
        candidate, decoders = (
            copy.deepcopy(encoder).double().eval(),
            copy.deepcopy(heads).double().eval(),
        )
        reference = next(candidate.parameters())
        codes: dict[str, Tensor] = {}
        split_labels: dict[str, TensorSplit] = {}
        reconstruction: dict[str, list[float]] = {}
        for name in ("calibration", "validation"):
            split = data[name]
            count = len(split["features"]) if groups is None else groups
            if count < 2 or count > len(split["features"]):
                raise ValueError("Invalid calibration/validation group count")
            features = split["features"][:count]
            output = torch.cat(
                [
                    candidate(features[start : start + 64].to(reference)).cpu()
                    for start in range(0, count, 64)
                ]
            )
            codes[name] = output
            split_labels[name] = {"theta": split["theta"][:count]}
            prediction = decoders(output.to(reference)).cpu()
            reconstruction[name] = (
                (prediction - split["targets"][:count].double()).square().mean((0, 1, 3)).tolist()
            )
        # Legacy probe helper's third slot is the SAME validation, never actual test.
        result = parameter_readouts(
            {**codes, "test": codes["validation"]},
            {**split_labels, "test": split_labels["validation"]},
        )
        families = {
            name: {
                "validation_r2": family["validation_r2"],
                "selected_hyperparameters_per_parameter": family[
                    "selected_hyperparameters_per_parameter"
                ],
                "validation_group_mse_by_parameter": family["test_group_mse_by_parameter"],
            }
            for name, family in result["families"].items()
        }
        return {
            "selected_validation_r2": result["selected_test_r2"],
            "selected_methods_per_parameter": result["selected_methods_per_parameter"],
            "validation_group_mse_by_parameter": result["selected_test_group_mse_by_parameter"],
            "families": families,
            "calibration_code_covariance_eigenvalues": result[
                "calibration_code_covariance_eigenvalues"
            ],
            "teacher_mse": reconstruction,
            "calibration_groups": len(codes["calibration"]),
            "validation_groups": len(codes["validation"]),
            "contexts_per_group": 10,
            "validation_raw_consistency": float(
                2 * codes["validation"].var(1, correction=1).mean()
            ),
            "scope": (
                "Individual-context compression; calibration fits, validation selects and scores. "
                "Selection-optimistic exploratory evidence. No test data or forecasting evaluated."
            ),
        }


def metrics(report: dict[str, Any], prefix: str = "validation") -> dict[str, float]:
    """Expose six-factor retention and balanced teacher reconstruction in W&B."""
    values = {
        f"{prefix}/theta_{i + 1}_r2": float(value)
        for i, value in enumerate(report["selected_validation_r2"])
    }
    values[f"{prefix}/all_six_retention_ge_0_8"] = float(
        min(report["selected_validation_r2"]) >= 0.8
    )
    values[f"{prefix}/raw_consistency"] = report["validation_raw_consistency"]
    for split, errors in report["teacher_mse"].items():
        values.update(
            {
                f"{prefix}/{split}/teacher_{i + 1}_mse": float(value)
                for i, value in enumerate(errors)
            }
        )
    return values
