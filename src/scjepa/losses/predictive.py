"""Configurable single-step predictive loss.

StateJepa's rows descend from known simulator object tracks (whether raw or
embedded), so they use ordinary aligned MSE. Visual learned slots remain an unordered set: the
context/target branches may assign objects to slot indices arbitrarily, so that
regime retains permutation-invariant Hungarian MSE. Per sample, Hungarian MSE
builds the (N, N) squared-distance cost between predicted and target slots,
solves the optimal one-to-one assignment exactly (scipy), and takes the MSE
over the matched pairs. The assignment is computed on a detached CPU copy
(DETR-style): gradients flow through the matched distances while the discrete
assignment is treated as a constant (correct almost everywhere).

Symbol table (paper ↔ code):
    Ŝ_{t+1}  predicted next state   (B, N, d)   ``pred``
    S_{t+1}  target slots           (B, N, d)   ``target`` (raw target-encoder slots, D9)
    π        per-sample assignment  (B, N)      ``match_slots`` output; target index π(i)
"""

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int64
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def _validate_slots(pred: Tensor, target: Tensor) -> None:
    """Require two equally shaped ``(batch, slots, dim)`` tensors."""
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(
            f"expected matching (B, N, d), got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )


def match_slots(
    pred: Float[Tensor, "b n d"], target: Float[Tensor, "b n d"]
) -> Int64[Tensor, "b n"]:
    """Optimal slot assignment π per sample: pred slot i ↔ target slot π(i).

    Non-differentiable (runs on a detached CPU copy). Also used by eval code to
    align slots with ground-truth objects.
    """
    _validate_slots(pred, target)
    cost = torch.cdist(pred, target).square()  # (B, N, N)
    cost_np = cost.detach().cpu().numpy()
    cols = [
        linear_sum_assignment(sample_cost)[1]  # rows come back sorted 0..N-1
        for sample_cost in cost_np
    ]
    return torch.as_tensor(np.stack(cols), dtype=torch.int64, device=pred.device)


def hungarian_mse(
    pred: Float[Tensor, "b n d"], target: Float[Tensor, "b n d"]
) -> Float[Tensor, ""]:
    """Permutation-invariant predictive loss: MSE over Hungarian-matched slots.

    Equals ``F.mse_loss(pred, target[:, π])`` for the optimal per-sample π;
    exactly ``F.mse_loss(pred, target)`` when slots are already aligned.
    """
    assignment = match_slots(pred, target)
    matched = target.gather(1, assignment.unsqueeze(-1).expand_as(target))
    return F.mse_loss(pred, matched)


def aligned_mse(
    pred: Float[Tensor, "b n d"], target: Float[Tensor, "b n d"]
) -> Float[Tensor, ""]:
    """Object-aligned MSE for tracked states whose row identity is known."""
    _validate_slots(pred, target)
    return F.mse_loss(pred, target)


def resolve_prediction_matching(requested: str, *, object_aligned: bool) -> str:
    """Resolve ``auto`` from whether input rows are persistent object tracks."""
    if requested == "auto":
        return "aligned" if object_aligned else "hungarian"
    if requested not in ("aligned", "hungarian"):
        raise ValueError(
            f"unknown prediction matching {requested!r}; expected auto, aligned, or hungarian"
        )
    return requested


def resolve_constraint_normalization(requested: str, *, gt_states: bool) -> str:
    """Choose the loss scale used by the sparsity constraint.

    Raw GT states already provide the fixed observation-space ruler assumed by
    Baumgartner/SPARTAN, so their constraint uses raw MSE. Learned target
    embeddings can rescale during training; that regime retains D17's detached
    target-variance normalization as a JEPA-specific stabilization.
    """
    if requested == "auto":
        return "raw" if gt_states else "target_variance"
    if requested not in ("raw", "target_variance"):
        raise ValueError(
            f"unknown constraint normalization {requested!r}; "
            "expected auto, raw, or target_variance"
        )
    return requested


def prediction_constraint(
    pred_loss: Float[Tensor, ""],
    target_var: Float[Tensor, ""],
    normalization: str,
) -> Float[Tensor, ""]:
    """Return prediction error on the configured dual-constraint scale."""
    if normalization == "raw":
        return pred_loss
    if normalization == "target_variance":
        return pred_loss / target_var
    raise ValueError(
        f"constraint normalization must be raw or target_variance, got {normalization!r}"
    )


def prediction_mse(
    pred: Float[Tensor, "b n d"],
    target: Float[Tensor, "b n d"],
    matching: str,
) -> Float[Tensor, ""]:
    """Apply the selected object correspondence policy."""
    if matching == "aligned":
        return aligned_mse(pred, target)
    if matching == "hungarian":
        return hungarian_mse(pred, target)
    raise ValueError(f"prediction matching must be aligned or hungarian, got {matching!r}")


__all__ = [
    "aligned_mse",
    "hungarian_mse",
    "match_slots",
    "prediction_constraint",
    "prediction_mse",
    "resolve_constraint_normalization",
    "resolve_prediction_matching",
]
