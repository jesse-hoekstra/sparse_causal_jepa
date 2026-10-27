"""Temporary nonlinear distillation heads and isolated curriculum phases."""

from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective, cached_forward
from diagnostics.mlp_oracle_code_control.model import MLPPredictor
from diagnostics.multiquery_control.data import TensorSplit, module_fingerprint
from scjepa.models.model import SCJEPA, cross_episode_parameters
from scjepa.training.rng import preserve_rng

Stage = Literal["distill", "predictor", "washout"]


class Heads(nn.Module):
    """Three independently parameterized 6→64→64→6 GELU decoders."""

    def __init__(self) -> None:
        """Use no projection or averaging on the student's complete six-vector."""
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 6)
                )
                for _ in range(3)
            ]
        )

    def forward(self, codes: Tensor) -> Tensor:
        """Return one six-vector per teacher, shape (...,3,6)."""
        return torch.stack([head(codes) for head in self.heads], dim=-2)


def initialize_heads(model: SCJEPA, seed: int) -> tuple[Heads, dict[str, Any]]:
    """Seed temporary modules independently without changing E/P or caller RNG."""
    with preserve_rng(next(model.parameters()).device):
        torch.manual_seed(seed + 1)  # pyright: ignore[reportUnknownMemberType]
        heads = Heads().to(next(model.parameters()))
    return heads, {
        "seed": seed + 1,
        "module_sha256": module_fingerprint(heads),
        "parameter_count": sum(parameter.numel() for parameter in heads.parameters()),
        "architecture": "Three separate 6-64-64-6 biased Linear/GELU heads",
    }


def infer_codes(model: SCJEPA, batch: TensorSplit) -> Tensor:
    """Encode each context independently; optimization accepts no physical labels."""
    if set(batch) != {"online", "target", "actions"}:
        raise ValueError("Training accepts only frozen online/target states and observed actions")
    online, target, actions = batch["online"], batch["target"], batch["actions"]
    groups = len(online)
    if online.shape != (groups, 10, 12, 6) or target.shape != online.shape:
        raise ValueError("Require original R10/T12 six-state caches")
    if actions.shape != (groups, 10, 11, 16):
        raise ValueError("Require original sixteen-action cache")
    if any(value.requires_grad for value in batch.values()):
        raise ValueError("Cached states and actions must remain frozen")
    if not isinstance(model.predictor, MLPPredictor) or (
        model.context_len != 10 or not model.autoregressive or model.parameter_source != "learned"
    ):
        raise ValueError("Require original K10 single-context E with the diagnostic MLP")
    return cast(
        Tensor,
        model.parameter_encoder(online[:, :, :10].flatten(0, 1), actions[:, :, :9].flatten(0, 1)),
    ).reshape(groups, 10, 6)


def configure_phase(stage: Stage, model: SCJEPA, heads: Heads | None) -> list[nn.Parameter]:
    """Use fresh optimizers over exactly the parameters authorized in this phase."""
    if stage not in ("distill", "predictor", "washout"):
        raise ValueError("Unknown curriculum phase")
    if stage == "distill" and heads is None:
        raise ValueError("Distillation requires temporary teacher heads")
    if stage == "washout" and heads is not None:
        raise ValueError("Discard all temporary heads before ordinary washout")
    model.requires_grad_(False)
    model.parameter_encoder.requires_grad_(stage != "predictor")
    model.predictor.requires_grad_(stage != "distill")
    if heads is not None:
        heads.requires_grad_(stage == "distill")
        heads.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    return [
        p
        for module in (model, heads)
        if module is not None
        for p in module.parameters()
        if p.requires_grad
    ]


def pretext_loss(
    model: SCJEPA,
    heads: Heads,
    batch: TensorSplit,
    targets: Tensor,
    objective: AcquisitionObjective,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Fit normalized detached leave-one-context-out teacher targets, without P."""
    codes = infer_codes(model, batch)
    if targets.shape != (*codes.shape[:2], 3, 6) or targets.requires_grad:
        raise ValueError("Teacher targets must be detached (B,R,3,6) values")
    if (
        targets.device != codes.device
        or targets.dtype != codes.dtype
        or not bool(torch.isfinite(targets).all())
    ):
        raise ValueError("Teacher targets must be finite and match student codes")
    # Preserve the original pairing draw although this phase does not forecast.
    cross_episode_parameters(codes, training=model.training)
    online = batch["online"]
    torch.randn(online.shape[-1], objective.projections, device=online.device, dtype=online.dtype)
    regularizer = objective.regularizer(codes.transpose(0, 1))
    consistency = 2 * codes.var(dim=1, correction=1).mean()
    teacher_errors = (heads(codes) - targets).square().mean((0, 1, 3))
    distillation = teacher_errors.mean()
    loss = distillation + 0.4 * regularizer + consistency
    return loss, {
        "distillation": distillation,
        **{f"teacher_{index + 1}_mse": error for index, error in enumerate(teacher_errors)},
        "parameter_regularizer": regularizer,
        "parameter_consistency": consistency,
        "optimized_total": loss,
    }


def phase_training_loss(
    stage: Stage,
    model: SCJEPA,
    heads: Heads | None,
    batch: TensorSplit,
    targets: Tensor | None,
    objective: AcquisitionObjective,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Separate teacher amortization, predictor preparation and ordinary joint washout."""
    if stage == "distill":
        if heads is None or targets is None:
            raise ValueError("Distillation requires heads and detached teacher targets")
        return pretext_loss(model, heads, batch, targets, objective)
    if targets is not None:
        raise ValueError("Teacher targets must not enter predictor fitting or washout")
    if stage not in ("predictor", "washout"):
        raise ValueError("Unknown curriculum phase")
    if stage == "washout" and heads is not None:
        raise ValueError("Washout must discard every auxiliary head")
    output = cached_forward(model, batch)
    full, terms = objective(output)
    terms["ordinary_total_without_constant_state_sigreg"] = full
    if stage == "predictor":
        if output.parameters.requires_grad:
            raise ValueError("Predictor preparation requires a frozen parameter encoder")
        # Auxiliary terms have no P gradient; preserve draws and report them.
        terms["optimized_total"] = terms["prediction"]
        return terms["prediction"], terms
    return full, terms
