"""Read-only code-distribution and gradient checks for a fixed dynamics predictor."""

import copy
import math
from collections.abc import Mapping
from itertools import combinations
from typing import Any, cast

import torch
from torch import Tensor

from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng


def _vector(value: Tensor) -> list[float]:
    return cast(list[float], value.detach().cpu().tolist())  # pyright: ignore[reportUnknownMemberType]


def code_reference_stats(codes: Tensor) -> dict[str, Any]:
    """Summarize complete codes without storing contexts, targets or parameter labels.

    The same summary is used for the known informative source codes and fresh
    codes. Coordinate statistics describe this fixed predictor's input space;
    their axes are not assigned to physical parameters.
    """
    if codes.ndim < 2 or codes.shape[-1] != 6 or codes.numel() == 0:
        raise ValueError("Expected nonempty codes with final dimension six")
    values = codes.detach().to(device="cpu", dtype=torch.float64).reshape(-1, 6)
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Nonfinite codes in distribution diagnostic")
    mean = values.mean(0)
    centered = values - mean
    covariance = centered.T @ centered / len(values)
    eigenvalues = cast(Tensor, torch.linalg.eigvalsh(covariance)).clamp_min(0)  # pyright: ignore[reportUnknownMemberType]
    return {
        "count": len(values),
        "mean": _vector(mean),
        "std": _vector(centered.square().mean(0).sqrt()),
        "rms": float(values.square().mean().sqrt()),
        "centered_rms": float(centered.square().mean().sqrt()),
        "covariance_eigenvalues": _vector(eigenvalues),
    }


def encoder_gradient_stats(
    model: SCJEPA,
    output: ModelOutput,
    terms: dict[str, Tensor],
    objective_config: ObjectiveConfig,
) -> dict[str, float]:
    """Measure original-objective gradients through the frozen predictor.

    This uses the existing forward graph and retains it for ordinary backward.
    Frozen weights remain frozen, but predictions and both autoregressive steps
    must remain attached to the parameter code. It does not populate or clear
    any .grad buffer, change model modes, update weights, or draw random numbers.
    """
    frozen = (model.online_encoder, model.target_encoder, model.predictor)
    if any(parameter.requires_grad for module in frozen for parameter in module.parameters()):
        raise ValueError("State encoders and predictor must be frozen")
    parameters = tuple(p for p in model.parameter_encoder.parameters() if p.requires_grad)
    if not parameters or not output.parameters.requires_grad:
        raise ValueError("Parameter encoder and inferred codes must remain trainable and attached")
    if {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in parameters}:
        raise ValueError("Only the parameter encoder may be trainable")
    inputs = (output.parameters, *parameters)
    losses = {
        "prediction": terms["prediction"],
        "sigreg": objective_config.parameter_regularizer_weight * terms["parameter_regularizer"],
        "consistency": objective_config.parameter_consistency_weight
        * terms["parameter_consistency"],
        # Logit penalties may depend on code even when predictor weights are fixed.
        "logit": objective_config.logit_weight * terms["logit"],
    }
    blocks: dict[str, dict[str, Tensor]] = {"code": {}, "encoder": {}}
    for name, loss in losses.items():
        gradients = (
            cast(
                tuple[Tensor | None, ...],
                torch.autograd.grad(loss, inputs, retain_graph=True, allow_unused=True),
            )
            if loss.requires_grad
            else (None,) * len(inputs)
        )
        pieces = [
            (torch.zeros_like(value) if gradient is None else gradient.detach()).flatten().double()
            for value, gradient in zip(inputs, gradients, strict=True)
        ]
        blocks["code"][name] = pieces[0]
        blocks["encoder"][name] = torch.cat(pieces[1:])
    metrics: dict[str, float] = {}
    for block, components in blocks.items():
        components["auxiliary"] = (
            components["sigreg"] + components["consistency"] + components["logit"]
        )
        components["total"] = components["prediction"] + components["auxiliary"]
        for name, values in components.items():
            metrics[f"{block}/{name}_norm"] = float(values.square().sum().sqrt())
        for first, second in [
            *combinations(("prediction", "sigreg", "consistency", "logit"), 2),
            ("prediction", "auxiliary"),
        ]:
            dot = float(torch.dot(components[first], components[second]))
            denominator = metrics[f"{block}/{first}_norm"] * metrics[f"{block}/{second}_norm"]
            key = f"{block}/{first}_{second}"
            metrics[key + "_dot"] = dot
            metrics[key + "_cosine"] = dot / denominator if denominator > 0 else 0.0
            metrics[key + "_cosine_defined"] = float(denominator > 0)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Nonfinite encoder gradient diagnostic")
    return metrics


def _rollout(model: SCJEPA, history: Tensor, codes: Tensor, actions: Tensor) -> Tensor:
    predictions: list[Tensor] = []
    for step in range(2):
        result = model.predictor(history, codes, actions[:, step : step + 1]).prediction
        predictions.append(result)
        history = result[:, None]
    return torch.stack(predictions, dim=1)


def predictor_code_sensitivity(
    model: SCJEPA,
    batch: Mapping[str, Tensor],
    *,
    reference_stats: Mapping[str, Any] | None = None,
    max_groups: int = 16,
) -> dict[str, Any]:
    """Inspect the fixed predictor at current code values on a float64 CPU copy.

    Use a fixed calibration batch with only online/target states and actions.
    All ten contexts contribute, with shift-one donor pairing. Six JVPs give
    each query's complete two-step output-by-code Jacobian. Reported sensitivity
    is local and coordinate dependent; it does not measure parameter recovery.
    Neither code scale nor source reference statistics alter training.
    """
    if max_groups < 1 or set(batch) != {"online", "target", "actions"}:
        raise ValueError("Use a positive group limit and label-free cached inputs")
    online, target, actions = (batch[key] for key in ("online", "target", "actions"))
    if (
        online.ndim != 4
        or online.shape[1:] != (10, 12, 6)
        or target.shape != online.shape
        or actions.shape != (len(online), 10, 11, 16)
        or len(online) < 1
    ):
        raise ValueError("Expected R10/K10/two-step six-state cached trajectories")
    if (
        model.context_len != 10
        or model.predictor.history_len != 1
        or not model.autoregressive
        or model.parameter_source != "learned"
    ):
        raise ValueError("Use the original learned-parameter K10 autoregressive model")
    device = next(model.parameters()).device
    with preserve_rng(device), torch.enable_grad():
        diagnostic_model = copy.deepcopy(model).to(device="cpu", dtype=torch.float64).eval()
        diagnostic_model.requires_grad_(False)
        groups = min(max_groups, len(online))
        states = online[:groups].detach().to(device="cpu", dtype=torch.float64)
        targets = target[:groups, :, 10:12].detach().to(device="cpu", dtype=torch.float64)
        controls = actions[:groups].detach().to(device="cpu", dtype=torch.float64)
        with torch.no_grad():
            codes = diagnostic_model.parameter_encoder(
                states[:, :, :10].flatten(0, 1), controls[:, :, :9].flatten(0, 1)
            ).reshape(groups, 10, 6)
        donor_codes = codes.roll(1, 1).flatten(0, 1)
        history = states[:, :, 9:10].flatten(0, 1)
        future_actions = controls[:, :, 9:11].flatten(0, 1)

        def predict(values: Tensor) -> Tensor:
            return _rollout(diagnostic_model, history, values, future_actions)

        columns: list[Tensor] = []
        for coordinate in range(6):
            direction = torch.zeros_like(donor_codes)
            direction[:, coordinate] = 1
            _, derivative = cast(
                tuple[Tensor, Tensor],
                torch.autograd.functional.jvp(predict, donor_codes, direction),  # pyright: ignore[reportUnknownMemberType]
            )
            columns.append(derivative.detach())
        jacobian = torch.stack(columns, -1).reshape(groups * 10, 12, 6)
        singular = cast(Tensor, torch.linalg.svdvals(jacobian))  # pyright: ignore[reportUnknownMemberType]
        independent_codes = donor_codes.detach().requires_grad_(True)
        prediction = predict(independent_codes)
        loss = (prediction - targets.flatten(0, 1)).square().mean()
        code_gradient = cast(tuple[Tensor, ...], torch.autograd.grad(loss, independent_codes))[0]
        if not all(
            bool(torch.isfinite(value).all()) for value in (jacobian, singular, loss, code_gradient)
        ):
            raise FloatingPointError("Nonfinite fixed-predictor sensitivity diagnostic")
        current = code_reference_stats(codes)
        summary: dict[str, Any] = {
            "groups": groups,
            "queries": groups * 10,
            "code_stats": current,
            "prediction_mse": float(loss.detach()),
            "prediction_code_gradient_norm": float(code_gradient.square().sum().sqrt()),
            "predictor_jacobian_frobenius_rms": float(
                jacobian.square().sum((-2, -1)).mean().sqrt()
            ),
            "predictor_jacobian_singular_values_median": _vector(singular.median(0).values),
            "predictor_jacobian_singular_values_minimum": _vector(singular.amin(0)),
            "nonzero_predictor_jacobian_fraction": float((singular[:, 0] > 0).double().mean()),
            "protocol": (
                "Float64 CPU copy; fixed calibration inputs, shift-one cross-episode pairing, "
                "attached two-step autoregression."
            ),
            "limits": (
                "Local sensitivity and code-scale comparisons do not certify recovery, "
                "information absence, or membership in the source code distribution. "
                "Code axes have no assigned physical meaning."
            ),
        }
        if reference_stats is not None:
            source_rms = float(reference_stats["rms"])
            source_centered = float(reference_stats["centered_rms"])
            source_mean = torch.as_tensor(reference_stats["mean"], dtype=torch.float64)
            if (
                source_mean.shape != (6,)
                or not bool(torch.isfinite(source_mean).all())
                or not math.isfinite(source_rms)
                or not math.isfinite(source_centered)
                or source_rms < 0
                or source_centered < 0
            ):
                raise ValueError("Invalid source code reference statistics")
            mean_shift = float((codes.mean((0, 1)) - source_mean).square().mean().sqrt())
            summary["source_code_comparison"] = {
                "rms_ratio": current["rms"] / source_rms if source_rms > 0 else None,
                "centered_rms_ratio": current["centered_rms"] / source_centered
                if source_centered > 0
                else None,
                "mean_shift_in_source_centered_rms": mean_shift / source_centered
                if source_centered > 0
                else None,
                "source_rms": source_rms,
                "source_centered_rms": source_centered,
                "used_for_training": False,
            }
        return summary
