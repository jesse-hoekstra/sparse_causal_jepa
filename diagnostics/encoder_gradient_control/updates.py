"""Apply paired encoder/predictor clipping and encoder-only prediction controls."""

import math
from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor

from scjepa.models.model import SCJEPA

CONDITIONS = ("joint_clip", "separate_clip", "separate_clip_encoder4")


def _norm(values: Sequence[Tensor | None], reference: Tensor) -> Tensor:
    """Use the same blockwise Euclidean norm convention as torch clipping."""
    norms = [
        cast(Tensor, torch.linalg.vector_norm(value.detach()))  # pyright: ignore[reportUnknownMemberType]
        for value in values
        if value is not None
    ]
    return (
        cast(Tensor, torch.linalg.vector_norm(torch.stack(norms)))  # pyright: ignore[reportUnknownMemberType]
        if norms
        else reference.new_zeros(())
    )


def _coefficient(norm: Tensor) -> float:
    return float((1.0 / (norm + 1e-6)).clamp(max=1.0))


def backward_and_clip(
    model: SCJEPA,
    total: Tensor,
    prediction_loss: Tensor,
    *,
    condition: str,
    encoder_prediction_scale: float = 4.0,
    measure_prediction: bool = False,
) -> dict[str, float]:
    """Populate clipped gradients without taking an optimizer step or drawing RNG.

    ``total`` is the ordinary prediction-weight-one objective, including unchanged
    auxiliary losses. ``prediction_loss`` is its raw MSE term. The boosted arm
    adds three extra copies of only that term's encoder gradient. Predictor
    gradients remain those of ``total``. This block-dependent update is not the
    gradient of a newly scaled joint scalar objective.

    Call ``optimizer.zero_grad(set_to_none=True)`` before this function and
    ``optimizer.step()`` afterward. Both state encoders must be frozen and only
    the parameter encoder and predictor may be trainable. Clipping is norm one
    jointly or independently per trainable block, matching the selected arm.

    Measuring prediction components requires an extra encoder backward pass.
    Baseline/separate arms do this only when ``measure_prediction`` is true;
    the boosted arm necessarily computes the same gradient on every update.
    Reported gradient components precede Adam's moment update/preconditioning.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown gradient control: {condition}")
    if not math.isfinite(encoder_prediction_scale) or encoder_prediction_scale != 4.0:
        raise ValueError("The separate_clip_encoder4 arm requires encoder_prediction_scale=4")
    if total.numel() != 1 or prediction_loss.numel() != 1:
        raise ValueError("Expected scalar total and prediction losses")
    if not bool(torch.isfinite(total).all()) or not bool(torch.isfinite(prediction_loss).all()):
        raise FloatingPointError("Nonfinite training loss")
    if not total.requires_grad:
        raise ValueError("The total objective must have an attached training graph")
    all_parameters = tuple(model.parameters())
    if any(parameter.grad is not None for parameter in all_parameters):
        raise ValueError("Clear all gradient buffers with zero_grad(set_to_none=True) first")
    if any(parameter.requires_grad for parameter in model.online_encoder.parameters()) or any(
        parameter.requires_grad for parameter in model.target_encoder.parameters()
    ):
        raise ValueError("Both state encoders must remain frozen")
    encoder = tuple(p for p in model.parameter_encoder.parameters() if p.requires_grad)
    predictor = tuple(p for p in model.predictor.parameters() if p.requires_grad)
    trainable = tuple(p for p in all_parameters if p.requires_grad)
    encoder_ids, predictor_ids = ({id(p) for p in block} for block in (encoder, predictor))
    if (
        not encoder
        or not predictor
        or encoder_ids & predictor_ids
        or {id(p) for p in trainable} != encoder_ids | predictor_ids
    ):
        raise ValueError("Only distinct parameter-encoder and predictor blocks may be trainable")
    scale = encoder_prediction_scale if condition == "separate_clip_encoder4" else 1.0
    measured = measure_prediction or scale != 1.0
    prediction_gradients: tuple[Tensor | None, ...] | None = None
    if measured:
        prediction_gradients = (
            cast(
                tuple[Tensor | None, ...],
                torch.autograd.grad(prediction_loss, encoder, retain_graph=True, allow_unused=True),
            )
            if prediction_loss.requires_grad
            else (None,) * len(encoder)
        )
    total.backward()  # pyright: ignore[reportUnknownMemberType]
    metrics: dict[str, float] = {
        "encoder/prediction_scale": scale,
        "prediction_components_measured": float(measured),
    }
    if prediction_gradients is not None:
        auxiliary = [
            (torch.zeros_like(p) if p.grad is None else p.grad.detach())
            - (torch.zeros_like(p) if predictive is None else predictive.detach())
            for p, predictive in zip(encoder, prediction_gradients, strict=True)
        ]
        prediction_norm = float(_norm(prediction_gradients, total))
        metrics.update(
            {
                "encoder/raw_prediction_gradient_norm": prediction_norm,
                "encoder/prediction_gradient_norm_before_clip": scale * prediction_norm,
                "encoder/prediction_correction_norm_pre_clip": (scale - 1.0) * prediction_norm,
                "encoder/auxiliary_gradient_norm_pre_clip": float(_norm(auxiliary, total)),
            }
        )
        if scale != 1.0:
            for parameter, predictive in zip(encoder, prediction_gradients, strict=True):
                if predictive is not None:
                    correction = (scale - 1.0) * predictive.detach()
                    if parameter.grad is None:
                        parameter.grad = correction.clone()
                    else:
                        parameter.grad.add_(correction)
    norms = {
        name: _norm([p.grad for p in block], total)
        for name, block in (("encoder", encoder), ("predictor", predictor), ("joint", trainable))
    }
    if not all(bool(torch.isfinite(norm)) for norm in norms.values()) or not all(
        math.isfinite(value) for value in metrics.values()
    ):
        raise FloatingPointError("Nonfinite gradients before clipping")
    for name, norm in norms.items():
        metrics[f"{name}/pre_clip_norm"] = float(norm)
    if condition == "joint_clip":
        # Preserve exactly the baseline's parameter ordering and clipping operation.
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        encoder_coefficient = predictor_coefficient = _coefficient(norm)
    else:
        encoder_norm = torch.nn.utils.clip_grad_norm_(encoder, 1.0, error_if_nonfinite=True)
        predictor_norm = torch.nn.utils.clip_grad_norm_(predictor, 1.0, error_if_nonfinite=True)
        encoder_coefficient, predictor_coefficient = (
            _coefficient(encoder_norm),
            _coefficient(predictor_norm),
        )
    metrics["encoder/clip_coefficient"] = encoder_coefficient
    metrics["predictor/clip_coefficient"] = predictor_coefficient
    for name, block in (("encoder", encoder), ("predictor", predictor), ("joint", trainable)):
        metrics[f"{name}/post_clip_norm"] = float(_norm([p.grad for p in block], total))
    if prediction_gradients is not None:
        metrics["encoder/prediction_gradient_norm_after_clip"] = (
            metrics["encoder/prediction_gradient_norm_before_clip"] * encoder_coefficient
        )
    return metrics
