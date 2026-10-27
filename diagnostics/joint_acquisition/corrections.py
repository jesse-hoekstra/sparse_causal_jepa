"""Shared diagnostic repairs on disposable encoder/predictor copies.

The caller supplies a calibration-only response loss and a separate assessment
function. These targets never replace the ordinary label-free objective.
"""

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from scjepa.models.model import SCJEPA
from scjepa.training.rng import preserve_rng

CorrectionCase = Literal["encoder", "predictor", "joint"]
LossFunction = Callable[[SCJEPA], Tensor]
AssessmentFunction = Callable[[SCJEPA], dict[str, Any]]
ComponentFunction = Callable[[SCJEPA], dict[str, Tensor]]
CASES: tuple[CorrectionCase, ...] = ("encoder", "predictor", "joint")


@dataclass(frozen=True)
class CorrectionSettings:
    """Numerical policy fixed before any assessment outcomes are inspected."""

    fractions: tuple[float, ...] = (1.0, 0.1, 0.01)
    clip_norm: float = 1.0
    finite_rtol: float = 0.05
    finite_atol: float = 1e-8

    def __post_init__(self) -> None:
        """Reject invalid bounds and nonshrinking finite-check schedules."""
        if (
            len(self.fractions) < 2
            or any(not math.isfinite(x) or not 0 < x <= 1 for x in self.fractions)
            or any(a <= b for a, b in zip(self.fractions, self.fractions[1:], strict=False))
        ):
            raise ValueError("Finite-check fractions must decrease within (0,1]")
        if any(
            not math.isfinite(x) or x <= 0
            for x in (self.clip_norm, self.finite_rtol, self.finite_atol)
        ):
            raise ValueError("Clipping and finite-check tolerances must be finite and positive")


@dataclass(frozen=True)
class SharedCorrections:
    """Auditable finite-step report and named directions for incentive analysis."""

    report: dict[str, Any]
    directions: dict[str, dict[str, Tensor]]
    saved_update: dict[str, Tensor]


def block_parameters(model: SCJEPA) -> dict[str, dict[str, nn.Parameter]]:
    """Return disjoint trainable modules in their serialized model order."""
    if any(p.requires_grad for p in model.online_encoder.parameters()) or any(
        p.requires_grad for p in model.target_encoder.parameters()
    ):
        raise ValueError("Both state encoders must remain frozen")
    blocks = {
        "encoder": {
            f"parameter_encoder.{name}": p for name, p in model.parameter_encoder.named_parameters()
        },
        "predictor": {f"predictor.{name}": p for name, p in model.predictor.named_parameters()},
    }
    if not all(blocks.values()):
        raise ValueError("Both parameter encoder and predictor must contain parameters")
    expected = {id(p) for block in blocks.values() for p in block.values()}
    if {id(p) for p in model.parameters() if p.requires_grad} != expected:
        raise ValueError("Only complete parameter encoder and predictor blocks may be trainable")
    return blocks


def vector_norm(values: Mapping[str, Tensor]) -> float:
    """Compute a Euclidean block norm in float64 without changing tensors."""
    return math.sqrt(
        sum(float(value.detach().double().square().sum()) for value in values.values())
    )


def loss_gradients(loss: Tensor, parameters: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Read gradients without touching source .grad buffers; unused entries are zero."""
    if loss.numel() != 1 or not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("Diagnostic loss must be a finite scalar")
    values = (
        cast(
            tuple[Tensor | None, ...],
            torch.autograd.grad(
                loss, tuple(parameters.values()), retain_graph=True, allow_unused=True
            ),
        )
        if loss.requires_grad
        else (None,) * len(parameters)
    )
    result = {
        name: torch.zeros_like(p) if value is None else value.detach()
        for (name, p), value in zip(parameters.items(), values, strict=True)
    }
    if any(not bool(torch.isfinite(value).all()) for value in result.values()):
        raise FloatingPointError("Nonfinite diagnostic gradient")
    return result


def apply_direction(model: SCJEPA, direction: Mapping[str, Tensor], fraction: float) -> None:
    """Apply a named finite movement to an explicitly disposable model."""
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, delta in direction.items():
            parameters[name].add_(delta.to(parameters[name]), alpha=fraction)


def validate_saved_optimizer(model: SCJEPA, optimizer: torch.optim.Optimizer) -> None:
    """Require coherent saved Adam history and its exact trainable parameter order."""
    blocks = block_parameters(model)
    if type(optimizer) not in (torch.optim.Adam, torch.optim.AdamW):
        raise ValueError("Saved optimizer must be Adam or AdamW")
    expected = [p for p in model.parameters() if p.requires_grad]
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    if [id(p) for p in actual] != [id(p) for p in expected]:
        raise ValueError("Saved optimizer parameter order differs from model order")
    counts: set[float] = set()
    for block in blocks.values():
        if not any(bool(optimizer.state.get(p)) for p in block.values()):
            raise ValueError("Saved optimizer history is required for both trainable blocks")
        for parameter in block.values():
            state = optimizer.state.get(parameter)
            if not state:
                continue  # Dense, unused gates legitimately have no moments.
            count = float(state["step"])
            if not math.isfinite(count) or count < 1 or not count.is_integer():
                raise ValueError("Invalid saved optimizer step count")
            counts.add(count)
            for key in ("exp_avg", "exp_avg_sq"):
                moment = state.get(key)
                if (
                    not isinstance(moment, Tensor)
                    or moment.shape != parameter.shape
                    or not bool(torch.isfinite(moment).all())
                ):
                    raise ValueError("Invalid saved optimizer moment")
                if key == "exp_avg_sq" and bool((moment < 0).any()):
                    raise ValueError("Saved optimizer variance must be nonnegative")
    if len(counts) != 1:
        raise ValueError("Saved optimizer steps must have coherent checkpoint history")


def saved_optimizer_update(
    model: SCJEPA,
    optimizer: torch.optim.Optimizer,
    components: ComponentFunction,
    clip_norm: float = 1.0,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Take the actual next clipped optimizer step on a model/moment deep copy."""
    validate_saved_optimizer(model, optimizer)
    with preserve_rng(next(model.parameters()).device):
        candidate, candidate_optimizer = copy.deepcopy((model, optimizer))
        candidate_optimizer.zero_grad(set_to_none=True)
        terms = components(candidate)
        total = terms["total"]
        if total.numel() != 1 or not bool(torch.isfinite(total).all()):
            raise FloatingPointError("Ordinary objective must be a finite scalar")
        total.backward()  # pyright: ignore[reportUnknownMemberType]
        parameters = [p for p in candidate.parameters() if p.requires_grad]
        if any(p.grad is not None and not candidate_optimizer.state.get(p) for p in parameters):
            raise ValueError(
                "Missing saved optimizer history for a parameter used by the objective"
            )
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=True))
        candidate_optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        original = dict(model.named_parameters())
        updates = {
            name: p.detach().clone() - original[name].detach()
            for name, p in candidate.named_parameters()
            if p.requires_grad
        }
        if any(not bool(torch.isfinite(delta).all()) for delta in updates.values()):
            raise FloatingPointError("Nonfinite saved optimizer update")
        return updates, {
            "optimizer_type": type(optimizer).__name__,
            "pre_clip_gradient_norm": norm,
            "clip_coefficient": min(1.0, clip_norm / (norm + 1e-6)),
            "clip_norm": clip_norm,
        }


def run_shared_corrections(
    model: SCJEPA,
    optimizer: torch.optim.Optimizer,
    fit_loss: LossFunction,
    assess: AssessmentFunction,
    components: ComponentFunction,
    settings: CorrectionSettings | None = None,
) -> SharedCorrections:
    """Fit one shared steepest-response correction per case and test finite copies.

    Each module's movement has the norm of its next saved-Adam update, measured
    using the supplied ordinary training batch. The joint case moves two blocks;
    its total search budget is therefore larger than either isolated case.
    """
    settings = settings or CorrectionSettings()
    with preserve_rng(next(model.parameters()).device):
        blocks = block_parameters(model)
        saved, saved_metadata = saved_optimizer_update(
            model, optimizer, components, settings.clip_norm
        )
        bounds = {
            name: vector_norm({key: saved[key] for key in block}) for name, block in blocks.items()
        }
        base = copy.deepcopy(model).double().eval()
        base_blocks = block_parameters(base)
        parameters = {key: p for block in base_blocks.values() for key, p in block.items()}
        loss = fit_loss(base)
        gradients = loss_gradients(loss, parameters)
        initial_loss = float(loss.detach())
        initial_assessment = assess(base)
        directions: dict[str, dict[str, Tensor]] = {}
        rows: dict[str, Any] = {}
        for case in CASES:
            active = ("encoder", "predictor") if case == "joint" else (case,)
            direction: dict[str, Tensor] = {}
            undefined: list[str] = []
            for block in active:
                g = {key: gradients[key] for key in blocks[block]}
                norm = vector_norm(g)
                if not math.isfinite(bounds[block]) or bounds[block] <= 0 or norm <= 0:
                    undefined.append(block)
                else:
                    direction.update(
                        {key: -value * (bounds[block] / norm) for key, value in g.items()}
                    )
            if undefined:
                rows[case] = {
                    "defined": False,
                    "undefined_blocks": undefined,
                    "reason": "Zero/invalid saved-update norm or zero diagnostic gradient",
                    "finite_steps": [],
                }
                continue
            directions[case] = direction
            slope = sum(float((gradients[key] * value).sum()) for key, value in direction.items())
            finite: list[dict[str, Any]] = []
            for fraction in settings.fractions:
                candidate = copy.deepcopy(base)
                apply_direction(candidate, direction, fraction)
                actual = float(fit_loss(candidate).detach())
                change, predicted = actual - initial_loss, fraction * slope
                error = abs(change - predicted)
                finite.append(
                    {
                        "fraction": fraction,
                        "fit_loss": actual,
                        "predicted_fit_change": predicted,
                        "actual_fit_change": change,
                        "absolute_linearization_error": error,
                        "linearization_valid": error
                        <= settings.finite_atol + settings.finite_rtol * abs(predicted),
                        "assessment": assess(candidate),
                        "movement": {
                            block: vector_norm(
                                {
                                    key: value * fraction
                                    for key, value in direction.items()
                                    if key in blocks[block]
                                }
                            )
                            for block in blocks
                        },
                    }
                )
            rows[case] = {
                "defined": True,
                "movement": {
                    block: vector_norm(
                        {key: value for key, value in direction.items() if key in blocks[block]}
                    )
                    for block in blocks
                },
                "total_movement": vector_norm(direction),
                "fit_directional_derivative": slope,
                "finite_steps": finite,
                "numerically_valid": finite[-1]["linearization_valid"],
            }
        report: dict[str, Any] = {
            "settings": asdict(settings),
            "solver": (
                "One full-calibration negative response-loss gradient, "
                "normalized separately per module"
            ),
            "supervision": (
                "Diagnostic exact parameter responses only; ordinary training objective unchanged"
            ),
            "reference_optimizer": saved_metadata,
            "reference_update_norms": bounds,
            "initial_fit_loss": initial_loss,
            "initial_assessment": initial_assessment,
            "cases": rows,
            "numerically_valid": all(row.get("numerically_valid", False) for row in rows.values()),
            "limits": [
                "Joint first-order effects add; this alone does not establish "
                "a nonlinear coordination barrier.",
                "Failure is limited to this direction and local budget, not global incapacity.",
                "Predictor-only repair cannot improve retention in unchanged encoder codes.",
            ],
        }
        return SharedCorrections(report, directions, saved)
