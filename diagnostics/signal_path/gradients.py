"""Local target-tangent residuals and their transmission through learned codes.

The six projected losses are diagnostic, overlapping local surrogates. They are
not an additive decomposition of prediction loss or a causal attribution of its
error. True theta is used only to construct frozen diagnostic directions.
"""

import math
from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.multiquery_control.data import EvaluationData, TensorSplit
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective
from diagnostics.signal_path.geometry import exact_rollout
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.models.model import SCJEPA
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng


def independent_factor_directions(
    jacobian: Tensor, rcond: float = 1e-6
) -> tuple[Tensor, Tensor, Tensor]:
    """Residualize each target tangent against the other five, without fitting labels.

    Input is ``(..., output_dimension, 6)``. Return unit residual directions,
    validity flags and residual/original tangent norm ratios. Near-zero unique
    effects are left zero, rather than amplified into arbitrary directions.
    SVD truncation is relative to the largest singular value of the full matrix.
    """
    if jacobian.ndim < 2 or jacobian.shape[-1] != 6 or jacobian.shape[-2] < 6:
        raise ValueError("Expected target Jacobians (..., output_dimension>=6, 6)")
    if not math.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError("rcond must lie strictly between zero and one")
    if not bool(torch.isfinite(jacobian).all()):
        raise FloatingPointError("Nonfinite target Jacobian")
    values = jacobian.detach()
    scale = cast(Tensor, torch.linalg.svdvals(values))[..., 0]  # pyright: ignore[reportUnknownMemberType]
    threshold = rcond * scale
    unit: list[Tensor] = []
    valid: list[Tensor] = []
    fraction: list[Tensor] = []
    for factor in range(6):
        others = values[..., [j for j in range(6) if j != factor]]
        left, singular, _ = cast(
            tuple[Tensor, Tensor, Tensor],
            torch.linalg.svd(others, full_matrices=False),  # pyright: ignore[reportUnknownMemberType]
        )
        original = values[..., factor]
        coefficients = torch.einsum("...ok,...o->...k", left, original)
        coefficients = coefficients * (singular > threshold[..., None])
        unique = original - torch.einsum("...ok,...k->...o", left, coefficients)
        norm = unique.square().sum(-1).sqrt()
        original_norm = original.square().sum(-1).sqrt()
        usable = (norm > threshold) & (original_norm > threshold) & (scale > 0)
        denominator = torch.where(usable, norm, torch.ones_like(norm))
        unit.append(unique / denominator[..., None] * usable[..., None])
        valid.append(usable)
        fraction.append(torch.where(original_norm > 0, norm / original_norm.clamp_min(1e-300), 0.0))
    return torch.stack(unit, -1), torch.stack(valid, -1), torch.stack(fraction, -1)


def projected_losses(residual: Tensor, directions: Tensor, weight: float) -> tuple[Tensor, Tensor]:
    """Return six frozen-projector losses, normalized like full predictive MSE.

    Residual shape is ``(groups, views, 2, 6)`` and directions shape is
    ``(groups, views, 12, 6)``. Per-group losses average views and all twelve
    output elements, not merely the nonzero projected direction.
    """
    if residual.ndim != 4 or residual.shape[-2:] != (2, 6):
        raise ValueError("Residuals must have shape (groups, views, 2, 6)")
    if directions.shape != (*residual.shape[:2], 12, 6):
        raise ValueError("Directions must have shape (groups, views, 12, 6)")
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("weight must be finite and positive")
    frozen = directions.detach()
    coefficients = torch.einsum("bro,broj->brj", residual.flatten(-2), frozen)
    # Valid directions have unit norm; explicit norm also supports zero directions.
    per_group = weight * (coefficients.square() * frozen.square().sum(-2)).mean(1) / 12
    return per_group.mean(0), per_group


def adamw_next_direction(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor | None],
    optimizer_state: dict[str, Any],
    *,
    clip_norm: float = 1.0,
) -> tuple[tuple[Tensor, ...], dict[str, float]]:
    """Compute, without mutation, a hypothetical next clipped AdamW update.

    The checkpoint's serialized parameter IDs are interpreted in exactly the
    current trainable-parameter order, as in the diagnostic training runner.
    Each call starts from the same moments. Missing gradients skip decay and
    moment updates as PyTorch does. AMSGrad and maximize are supported.
    """
    if not parameters or len(parameters) != len(gradients):
        raise ValueError("Expected matching nonempty parameters and gradients")
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")
    groups = cast(list[dict[str, Any]], optimizer_state["param_groups"])
    states = cast(dict[int, dict[str, Any]], optimizer_state["state"])
    ids = [int(pid) for group in groups for pid in group["params"]]
    if len(ids) != len(parameters) or len(set(ids)) != len(ids):
        raise ValueError("Optimizer parameter IDs do not match trainable parameter order")
    reference = parameters[0]
    total_square = reference.new_zeros(())
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if gradient is not None:
            if gradient.shape != parameter.shape or gradient.is_sparse:
                raise ValueError("Expected dense gradients matching parameter shapes")
            total_square = total_square + gradient.detach().square().sum()
    norm = float(total_square.sqrt())
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite total gradient norm")
    clip = min(1.0, clip_norm / (norm + 1e-6))
    updates: list[Tensor] = []
    cursor = 0
    for group in groups:
        beta1, beta2 = (float(x) for x in group["betas"])
        lr, eps, decay = float(group["lr"]), float(group["eps"]), float(group["weight_decay"])
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or lr < 0 or eps <= 0 or decay < 0:
            raise ValueError("Unsupported invalid AdamW hyperparameters")
        if not all(math.isfinite(x) for x in (beta1, beta2, lr, eps, decay)):
            raise ValueError("AdamW hyperparameters must be finite")
        for pid in group["params"]:
            parameter, gradient = parameters[cursor], gradients[cursor]
            cursor += 1
            if gradient is None:
                updates.append(torch.zeros_like(parameter))
                continue
            state = states.get(int(pid), {})
            old_step = state.get("step", 0)
            step_value = float(old_step.item()) if isinstance(old_step, Tensor) else float(old_step)
            if not math.isfinite(step_value) or step_value < 0 or not step_value.is_integer():
                raise ValueError("AdamW step counter must be a nonnegative integer")
            step = int(step_value) + 1
            previous_m = cast(Tensor, state.get("exp_avg", torch.zeros_like(parameter))).to(
                parameter
            )
            previous_v = cast(Tensor, state.get("exp_avg_sq", torch.zeros_like(parameter))).to(
                parameter
            )
            if previous_m.shape != parameter.shape or previous_v.shape != parameter.shape:
                raise ValueError("AdamW moment shapes do not match the current parameters")
            g = gradient.detach() * (-clip if group.get("maximize", False) else clip)
            moment = beta1 * previous_m + (1 - beta1) * g
            variance = beta2 * previous_v + (1 - beta2) * g.square()
            if group.get("amsgrad", False):
                previous_max = cast(
                    Tensor, state.get("max_exp_avg_sq", torch.zeros_like(parameter))
                ).to(parameter)
                if previous_max.shape != parameter.shape:
                    raise ValueError("AdamW maximum moment shape does not match")
                variance = torch.maximum(variance, previous_max)
            denominator = variance.sqrt() / math.sqrt(1 - beta2**step) + eps
            update = (
                -lr * decay * parameter.detach() - lr / (1 - beta1**step) * moment / denominator
            )
            if not bool(torch.isfinite(update).all()):
                raise FloatingPointError("Nonfinite hypothetical AdamW update")
            updates.append(update.detach())
    return tuple(updates), {"pre_clip_gradient_norm": norm, "clip_coefficient": clip}


def _gradients(loss: Tensor, inputs: tuple[Tensor, ...]) -> tuple[Tensor | None, ...]:
    if not loss.requires_grad:
        return (None,) * len(inputs)
    return cast(
        tuple[Tensor | None, ...],
        torch.autograd.grad(loss, inputs, retain_graph=True, allow_unused=True),
    )


def _flatten(values: Sequence[Tensor | None], inputs: Sequence[Tensor]) -> Tensor:
    return torch.cat(
        [
            torch.zeros_like(tensor).flatten() if value is None else value.detach().flatten()
            for value, tensor in zip(values, inputs, strict=True)
        ]
    )


def _blocks(
    values: Sequence[Tensor | None],
    inputs: Sequence[Tensor],
    encoder_indices: Sequence[int],
    predictor_indices: Sequence[int],
) -> dict[str, Tensor]:
    return {
        "code": _flatten(values[:1], inputs[:1]),
        "encoder": _flatten(
            [values[i] for i in encoder_indices], [inputs[i] for i in encoder_indices]
        ),
        "predictor": _flatten(
            [values[i] for i in predictor_indices], [inputs[i] for i in predictor_indices]
        ),
    }


def _target_jacobian(
    model: SCJEPA, observation_map: VectorObservationMap, split: TensorSplit
) -> Tensor:
    theta = split["theta"]
    views = split["true_states"].shape[1]
    initial = split["true_states"][:, :, 9].flatten(0, 1)
    actions = split["true_actions"][:, :, 9:11].flatten(0, 1)
    expanded = theta.repeat_interleave(views, 0)

    def target(parameters: Tensor) -> Tensor:
        return exact_rollout(model, observation_map, initial, parameters, actions)

    columns: list[Tensor] = []
    for factor in range(6):
        direction = torch.zeros_like(expanded)
        direction[:, factor] = 1
        _, derivative = cast(
            tuple[Tensor, Tensor],
            torch.autograd.functional.jvp(target, expanded, direction),  # pyright: ignore[reportUnknownMemberType]
        )
        columns.append(derivative.detach())
    return torch.stack(columns, -1).reshape(len(theta), views, 2, 6, 6)


def gradient_path_diagnostic(
    model: SCJEPA,
    data: EvaluationData,
    objective_config: ObjectiveConfig,
    prediction_weight: float,
    *,
    groups: int,
    batch_size: int,
    optimizer_state: dict[str, Any] | None = None,
    rcond: float = 1e-6,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Measure transmission and local descent directions without updating any weights.

    A deterministic shift-one donor assignment matches the Q1 training protocol.
    Cached frozen states define ordinary prediction residuals; exact true states
    and actions define only the target-tangent diagnostic. Double precision is
    required. Projector SVDs and theta derivatives are detached before comparing
    ordinary parameter gradients. Module modes, RNGs, weights and .grad buffers
    are left untouched; model must already be in eval mode.
    """
    reference = next(model.parameters())
    if model.training or reference.dtype != torch.float64:
        raise ValueError("Use a double-precision model in evaluation mode")
    if any(p.requires_grad for p in model.online_encoder.parameters()) or any(
        p.requires_grad for p in model.target_encoder.parameters()
    ):
        raise ValueError("Both state encoders must remain frozen")
    if groups < 1 or batch_size < 2 or groups % batch_size:
        raise ValueError("groups must be positive and divisible by batch_size>=2")
    source = cast(TensorSplit, data["test"])
    if groups > len(source["theta"]):
        raise ValueError("Requested more groups than the held-out test cache contains")
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    encoder_ids = {id(p) for p in model.parameter_encoder.parameters() if p.requires_grad}
    predictor_ids = {id(p) for p in model.predictor.parameters() if p.requires_grad}
    if (
        not encoder_ids
        or not predictor_ids
        or {id(p) for p in parameters} != encoder_ids | predictor_ids
    ):
        raise ValueError("Only the parameter encoder and predictor may be trainable")
    encoder_indices = tuple(i + 1 for i, p in enumerate(parameters) if id(p) in encoder_ids)
    predictor_indices = tuple(i + 1 for i, p in enumerate(parameters) if id(p) in predictor_ids)
    observation_map = VectorObservationMap(
        matrix=cast(Tensor, data["metadata"]["observation_matrix"]).to(reference), dimension=6
    )
    objective = PredictiveSupervisionObjective(objective_config, prediction_weight).to(
        reference.device
    )
    batches: list[dict[str, Any]] = []
    raw: dict[str, list[Tensor]] = {
        name: []
        for name in (
            "target_jacobian",
            "unique_directions",
            "unique_direction_valid",
            "unique_direction_fraction",
            "factor_projected_loss_per_group",
            "factor_code_gradients",
            "predictions",
            "targets",
            "codes",
        )
    }
    with preserve_rng(reference.device), torch.enable_grad():
        for batch_index, start in enumerate(range(0, groups, batch_size)):
            split = {
                key: value[start : start + batch_size].to(reference)
                for key, value in source.items()
            }
            targets_jacobian = _target_jacobian(model, observation_map, split)
            directions, valid, fractions = independent_factor_directions(
                targets_jacobian.flatten(-3, -2), rcond
            )
            cached = {
                "online": split["states_online"].detach(),
                "target": split["states_target"].detach(),
                "actions": split["actions"].detach(),
            }
            projection_seed = 20261027001 + batch_index
            torch.default_generator.manual_seed(projection_seed)
            if reference.device.type == "cuda":
                with torch.cuda.device(reference.device):
                    torch.cuda.manual_seed(projection_seed)
            output = cached_multiquery_forward(model, cached, queries=1, shift=1)
            total, terms = objective(output)
            residual = output.predictions - output.targets.detach()
            factor_losses, per_group = projected_losses(residual, directions, prediction_weight)
            inputs = (output.parameters, *parameters)
            component_losses = {
                "prediction": prediction_weight * terms["prediction"],
                "sigreg": objective_config.parameter_regularizer_weight
                * terms["parameter_regularizer"],
                "consistency": objective_config.parameter_consistency_weight
                * terms["parameter_consistency"],
                "logit": objective_config.logit_weight * terms["logit"],
                "total": total,
            }
            full_values = {
                name: _gradients(loss, inputs) for name, loss in component_losses.items()
            }
            full_blocks = {
                name: _blocks(values, inputs, encoder_indices, predictor_indices)
                for name, values in full_values.items()
            }
            adam_blocks: dict[str, Tensor] = {}
            clipping: dict[str, float] = {}
            if optimizer_state is not None:
                update, clipping = adamw_next_direction(
                    parameters, full_values["total"][1:], optimizer_state
                )
                update_inputs = (output.parameters.new_zeros(output.parameters.shape), *update)
                adam_blocks = _blocks(update_inputs, inputs, encoder_indices, predictor_indices)
            factor_records: list[dict[str, Any]] = []
            code_gradients: list[Tensor] = []
            for factor in range(6):
                values = _gradients(factor_losses[factor], inputs)
                block_values = _blocks(values, inputs, encoder_indices, predictor_indices)
                code_gradients.append(
                    torch.zeros_like(output.parameters) if values[0] is None else values[0].detach()
                )
                row: dict[str, Any] = {
                    "factor": factor + 1,
                    "weighted_projected_loss": float(factor_losses[factor].detach()),
                    "unweighted_projected_loss": float(factor_losses[factor].detach())
                    / prediction_weight,
                    "valid_query_fraction": float(valid[..., factor].double().mean()),
                    "unique_tangent_fraction_mean": float(fractions[..., factor].mean()),
                    "blocks": {},
                }
                for block, factor_gradient in block_values.items():
                    factor_norm = float(factor_gradient.square().sum().sqrt())
                    measured: dict[str, Any] = {
                        "projected_gradient_norm": factor_norm,
                        "unweighted_projected_gradient_norm": factor_norm / prediction_weight,
                        "components": {},
                    }
                    for component, parts in full_blocks.items():
                        other = parts[block]
                        other_norm = float(other.square().sum().sqrt())
                        dot = float(torch.dot(factor_gradient, other))
                        denominator = factor_norm * other_norm
                        measured["components"][component] = {
                            "gradient_norm": other_norm,
                            "dot": dot,
                            "cosine": dot / denominator if denominator > 0 else None,
                            "unit_sgd_predicted_loss_change": -dot,
                        }
                    if block in ("encoder", "predictor") and adam_blocks:
                        measured["next_adam_predicted_loss_change"] = float(
                            torch.dot(factor_gradient, adam_blocks[block])
                        )
                    row["blocks"][block] = measured
                if adam_blocks:
                    row["next_adam_all_trainable_predicted_loss_change"] = sum(
                        row["blocks"][block]["next_adam_predicted_loss_change"]
                        for block in ("encoder", "predictor")
                    )
                factor_records.append(row)
            batches.append(
                {
                    "batch_index": batch_index,
                    "first_test_group": start,
                    "groups": batch_size,
                    "prediction_weight": prediction_weight,
                    "component_losses": {
                        name: float(value.detach()) for name, value in component_losses.items()
                    },
                    "clipping": clipping,
                    "by_parameter": factor_records,
                }
            )
            raw["target_jacobian"].append(targets_jacobian.detach().cpu())
            raw["unique_directions"].append(directions.cpu())
            raw["unique_direction_valid"].append(valid.cpu())
            raw["unique_direction_fraction"].append(fractions.cpu())
            raw["factor_projected_loss_per_group"].append(per_group.detach().cpu())
            raw["factor_code_gradients"].append(torch.stack(code_gradients, -2).cpu())
            raw["predictions"].append(output.predictions.detach().cpu())
            raw["targets"].append(output.targets.detach().cpu())
            raw["codes"].append(output.parameters.detach().cpu())
    tensors = {key: torch.cat(values) for key, values in raw.items()}
    summaries: dict[str, Any] = {}
    for factor in range(6):
        records = [batch["by_parameter"][factor] for batch in batches]
        aggregate: dict[str, Any] = {
            "weighted_projected_loss_mean": sum(r["weighted_projected_loss"] for r in records)
            / len(records),
            "unweighted_projected_loss_mean": sum(r["unweighted_projected_loss"] for r in records)
            / len(records),
            "valid_query_fraction": float(
                tensors["unique_direction_valid"][..., factor].double().mean()
            ),
            "unique_tangent_fraction_mean": float(
                tensors["unique_direction_fraction"][..., factor].mean()
            ),
            "blocks": {},
        }
        for block in ("code", "encoder", "predictor"):
            block_summary: dict[str, Any] = {
                "projected_gradient_norm_mean": sum(
                    r["blocks"][block]["projected_gradient_norm"] for r in records
                )
                / len(records),
                "unweighted_projected_gradient_norm_mean": sum(
                    r["blocks"][block]["unweighted_projected_gradient_norm"] for r in records
                )
                / len(records),
                "total_unit_sgd_loss_change_mean": sum(
                    r["blocks"][block]["components"]["total"]["unit_sgd_predicted_loss_change"]
                    for r in records
                )
                / len(records),
            }
            if block in ("encoder", "predictor") and optimizer_state is not None:
                deltas = [r["blocks"][block]["next_adam_predicted_loss_change"] for r in records]
                block_summary["next_adam_predicted_loss_change_mean"] = sum(deltas) / len(deltas)
                block_summary["next_adam_descent_batch_fraction"] = sum(
                    x < 0 for x in deltas
                ) / len(deltas)
            aggregate["blocks"][block] = block_summary
        if optimizer_state is not None:
            deltas = [r["next_adam_all_trainable_predicted_loss_change"] for r in records]
            aggregate["next_adam_all_trainable_loss_change_mean"] = sum(deltas) / len(deltas)
            aggregate["next_adam_all_trainable_descent_batch_fraction"] = sum(
                x < 0 for x in deltas
            ) / len(deltas)
        summaries[f"theta_{factor + 1}"] = aggregate
    result = {
        "groups": groups,
        "batch_size": batch_size,
        "batches": batches,
        "by_parameter": summaries,
        "optimizer_direction_available": optimizer_state is not None,
        "protocol": (
            "Q1 shift-one cross-episode two-step autoregression. Each exact target theta tangent "
            "is residualized against the other five within each query. The detached unit direction "
            "projects the actual residual; each projected squared loss uses the ordinary MSE "
            "normalization and trained prediction coefficient. All gradient cosines use complete "
            "code/module vectors, without aligning code coordinates to true parameters."
        ),
        "limits": [
            "These six overlapping local target-tangent losses do not sum to MSE "
            "and are not a causal attribution of error.",
            "A zero unique direction can reflect locally overlapping or weak target tangents; "
            "inspect validity and separability first.",
            "Positive gradient dot means the corresponding vanilla descent direction reduces "
            "this fixed-projector loss to first order.",
            "Adam directions are hypothetical held-out steps from the same checkpoint moments "
            "independently for every batch, not an observed training update.",
            "Negative predicted Adam loss change is only a local first-order decrease; it does "
            "not establish acquisition, retained information, or finite-step improvement.",
            "Code-space SGD is hypothetical direct code optimization; actual encoder and "
            "predictor parameter directions are reported separately.",
            "Euclidean code/module gradient norms depend on parameterization and are not "
            "direct measures of information.",
            "Weighted projected gradients include the trained prediction coefficient; "
            "unweighted loss and norm fields remove its trivial scalar multiplier.",
        ],
    }
    return result, tensors
