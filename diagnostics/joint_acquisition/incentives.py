"""Compare useful diagnostic corrections with the recorded label-free objective."""

import copy
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from diagnostics.joint_acquisition.corrections import (
    ComponentFunction,
    CorrectionSettings,
    LossFunction,
    SharedCorrections,
    apply_direction,
    block_parameters,
    loss_gradients,
    vector_norm,
)
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from scjepa.models.model import SCJEPA
from scjepa.training.loop import ObjectiveConfig, TrainingObjective
from scjepa.training.rng import preserve_rng


def recorded_objective_components(
    model: SCJEPA,
    batch: Mapping[str, Tensor],
    objective_config: ObjectiveConfig,
    *,
    seed: int,
    shift: int = 1,
) -> dict[str, Tensor]:
    """Evaluate recorded weights with fixed SIGReg projections and donor pairing.

    Return weighted scalar components and unweighted prediction/consistency
    observations per parameter group. SIGReg couples samples, so no fictitious
    additive group-level SIGReg attribution is supplied. Frozen state SIGReg is
    included in the objective value, with zero derivative in trainable blocks.
    """
    reference = next(model.parameters())
    if any(
        (
            objective_config.state_sparsity_weight,
            objective_config.action_sparsity_weight,
            objective_config.parameter_sparsity_weight,
        )
    ):
        raise ValueError("Joint acquisition diagnostics require the original dense objective")
    with preserve_rng(reference.device):
        torch.default_generator.manual_seed(seed)
        if reference.device.type == "cuda":
            with torch.cuda.device(reference.device):
                torch.cuda.manual_seed(seed)
        output = cached_multiquery_forward(
            model,
            {key: value.to(reference) for key, value in batch.items()},
            queries=1,
            shift=shift,
        )
        objective = TrainingObjective(copy.deepcopy(objective_config)).to(reference)
        total, terms = objective(output)
        zero = total.new_zeros(())
        return {
            "prediction": terms["prediction"],
            "raw_sigreg": terms.get("parameter_regularizer", zero),
            "raw_consistency": terms.get("parameter_consistency", zero),
            "raw_logit": terms["logit"],
            "sigreg": objective_config.parameter_regularizer_weight
            * terms.get("parameter_regularizer", zero),
            "consistency": objective_config.parameter_consistency_weight
            * terms.get("parameter_consistency", zero),
            "logit": objective_config.logit_weight * terms["logit"],
            "frozen_state": objective_config.state_regularizer_weight
            * terms.get("state_regularizer", zero),
            "total": total,
            "_group_prediction": (output.predictions - output.targets).square().mean((1, 2, 3)),
            "_group_consistency": 2 * output.parameters.var(dim=1, correction=1).mean(-1),
        }


def _classification(cosine: float | None, *, descent: bool) -> str:
    if cosine is None:
        return "undefined"
    if abs(cosine) <= 1e-6:
        return "weak"
    favorable = cosine < 0 if descent else cosine > 0
    return "supportive" if favorable else "opposing"


def _direction_components(
    candidate: SCJEPA,
    components: ComponentFunction,
    direction: Mapping[str, Tensor],
    initial_values: Mapping[str, float],
    gradients: Mapping[str, Mapping[str, Tensor]],
    settings: CorrectionSettings,
) -> dict[str, dict[str, Any]]:
    movement = vector_norm(direction)
    actual_values: list[tuple[float, dict[str, float]]] = []
    for fraction in settings.fractions:
        finite = copy.deepcopy(candidate)
        apply_direction(finite, direction, fraction)
        measured = components(finite)
        actual_values.append(
            (fraction, {name: float(measured[name].detach()) for name in initial_values})
        )
    rows: dict[str, dict[str, Any]] = {}
    for name, gradient in gradients.items():
        norm = vector_norm({key: gradient[key] for key in direction})
        slope = sum(
            float((gradient[key] * delta.to(gradient[key])).sum())
            for key, delta in direction.items()
        )
        cosine = slope / (norm * movement) if norm > 0 and movement > 0 else None
        finite_rows: list[dict[str, Any]] = []
        for fraction, measured in actual_values:
            predicted = fraction * slope
            actual = measured[name] - initial_values[name]
            finite_rows.append(
                {
                    "fraction": fraction,
                    "predicted_change": predicted,
                    "actual_change": actual,
                    "linearization_valid": abs(actual - predicted)
                    <= settings.finite_atol + settings.finite_rtol * abs(predicted),
                }
            )
        rows[name] = {
            "initial_value": initial_values[name],
            "gradient_norm": norm,
            "directional_derivative": slope,
            "gradient_direction_cosine": cosine,
            "classification": _classification(cosine, descent=True),
            "finite_steps": finite_rows,
        }
    return rows


def analyze_incentives(
    model: SCJEPA,
    components: ComponentFunction,
    corrections: SharedCorrections,
    *,
    fit_loss: LossFunction | None = None,
    settings: CorrectionSettings | None = None,
) -> dict[str, Any]:
    """Check component slopes and actual finite steps along useful and saved directions.

    The actual saved update is the immutable one computed during local correction
    fitting. No comparison advances or reuses modified source moments. Negative
    loss slope means that component rewards the proposed correction locally.
    """
    settings = settings or CorrectionSettings()
    with preserve_rng(next(model.parameters()).device):
        candidate = copy.deepcopy(model).double().eval()
        blocks = block_parameters(candidate)
        parameters = {name: p for block in blocks.values() for name, p in block.items()}
        values = components(candidate)
        scalars = {name: value for name, value in values.items() if not name.startswith("_")}
        if any(value.numel() != 1 for value in scalars.values()):
            raise ValueError("Objective components must be scalar")
        gradients = {name: loss_gradients(value, parameters) for name, value in scalars.items()}
        initial_values = {name: float(value.detach()) for name, value in scalars.items()}
        rows: dict[str, Any] = {}
        for case, direction in corrections.directions.items():
            movement = vector_norm(direction)
            if not math.isfinite(movement) or movement <= 0:
                rows[case] = {"defined": False}
                continue
            component_rows = _direction_components(
                candidate, components, direction, initial_values, gradients, settings
            )
            saved = {key: corrections.saved_update[key] for key in direction}
            saved_norm = vector_norm(saved)
            dot = sum(
                float((saved[key].to(delta) * delta).sum()) for key, delta in direction.items()
            )
            cosine = dot / (saved_norm * movement) if saved_norm > 0 else None
            rows[case] = {
                "defined": True,
                "components": component_rows,
                "saved_optimizer_alignment": {
                    "dot": dot,
                    "cosine": cosine,
                    "norm": saved_norm,
                    "classification": _classification(cosine, descent=False),
                },
            }
        saved_components = {
            name: {
                key: value
                for key, value in row.items()
                if key not in {"initial_value", "gradient_norm"}
            }
            for name, row in _direction_components(
                candidate, components, corrections.saved_update, initial_values, gradients, settings
            ).items()
        }
        saved_response: dict[str, Any] | None = None
        if fit_loss is not None:
            loss = fit_loss(candidate)
            gradient = loss_gradients(loss, parameters)
            slope = sum(
                float((gradient[name] * delta.to(gradient[name])).sum())
                for name, delta in corrections.saved_update.items()
            )
            finite = copy.deepcopy(candidate)
            apply_direction(finite, corrections.saved_update, 1.0)
            saved_response = {
                "initial_loss": float(loss.detach()),
                "predicted_change": slope,
                "actual_change": float(fit_loss(finite).detach()) - float(loss.detach()),
            }
        return {
            "cases": rows,
            "saved_optimizer_response": saved_response,
            "saved_optimizer_components": saved_components,
            "raw_group_components": {
                name.removeprefix("_group_"): value.detach().cpu().tolist()  # pyright: ignore[reportUnknownMemberType]
                for name, value in values.items()
                if name.startswith("_group_")
            },
            "component_gradient_norms_by_block": {
                block: {
                    component: vector_norm({name: gradient[name] for name in names})
                    for component, gradient in gradients.items()
                }
                for block, names in blocks.items()
            },
            "frozen_state_regularization": (
                "Constant with respect to parameter-encoder and predictor weights"
            ),
            "classification_policy": (
                "Undefined for zero norms; weak for |cosine|<=1e-6; "
                "otherwise sign of local slope/alignment"
            ),
            "limits": (
                "Local incentives are not eventual acquisition; finite component checks "
                "and held-out response/ordinary prediction accompany each useful repair. "
                "SIGReg has no additive per-group decomposition."
            ),
        }
