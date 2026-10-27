"""Learn shared group information from single-context codes without a predictor loss."""

import math

import torch
from torch import Tensor

from scjepa.models.model import SCJEPA


def cached_codes(model: SCJEPA, batch: dict[str, Tensor]) -> Tensor:
    """Infer one original six-code per context; group IDs and physical labels are forbidden."""
    if set(batch) != {"online", "target", "actions"}:
        raise ValueError("Contrastive training excludes physical labels and group IDs")
    online, target, actions = batch["online"], batch["target"], batch["actions"]
    groups = len(online)
    if groups < 2 or online.shape != (groups, 10, 12, 6) or target.shape != online.shape:
        raise ValueError("Require at least two groups and original R10/T12 six-state caches")
    if actions.shape != (groups, 10, 11, 16):
        raise ValueError("Require original sixteen-action cache")
    if any(value.requires_grad for value in batch.values()):
        raise ValueError("Cached state and action tensors must stay frozen")
    if model.context_len != 10 or model.parameter_source != "learned":
        raise ValueError("Require the original learned-code K10 context encoder")
    return model.parameter_encoder(
        online[:, :, :10].flatten(0, 1), actions[:, :, :9].flatten(0, 1)
    ).reshape(groups, 10, 6)


def contrastive_loss(codes: Tensor) -> Tensor:
    """Mean nine-positive NLL minus log9; raw Euclidean T1, self excluded from all sums."""
    if codes.ndim != 3 or codes.shape[1:] != (10, 6) or len(codes) < 2:
        raise ValueError("Require Bx10x6 single-context codes with B>=2")
    if not codes.is_floating_point() or not bool(torch.isfinite(codes).all()):
        raise ValueError("Contrastive codes must be finite floating-point values")
    flat = codes.flatten(0, 1)
    logits = -0.5 * (flat[:, None] - flat[None]).square().sum(-1)
    diagonal = torch.eye(len(flat), device=flat.device, dtype=torch.bool)
    groups = torch.arange(len(codes), device=flat.device).repeat_interleave(10)
    positive = (groups[:, None] == groups[None]) & ~diagonal
    logits = logits.masked_fill(diagonal, -torch.inf)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -log_probability[positive].reshape(len(flat), 9).mean() - math.log(9)


@torch.no_grad()
def code_statistics(codes: Tensor) -> dict[str, Tensor]:
    """Report scale and within/between distances without imposing an extra penalty."""
    flat = codes.detach().flatten(0, 1)
    groups = torch.arange(len(codes), device=flat.device).repeat_interleave(10)
    same = groups[:, None] == groups[None]
    diagonal = torch.eye(len(flat), device=flat.device, dtype=torch.bool)
    distances = (flat[:, None] - flat[None]).square().sum(-1)
    return {
        "code_rms": flat.square().mean().sqrt(),
        "within_group_squared_distance": distances[same & ~diagonal].mean(),
        "between_group_squared_distance": distances[~same].mean(),
    }


def training_loss(model: SCJEPA, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
    """Only contrastive gradients train E; no forecasts, SIGReg or separate consistency."""
    codes = cached_codes(model, batch)
    loss = contrastive_loss(codes)
    return loss, {"contrastive": loss, "optimized_total": loss, **code_statistics(codes)}
