"""Hungarian-matched single-step predictive loss (D6).

Slots are a set: the context/target branches assign objects to slot indices
arbitrarily, so the predictive loss must be permutation-invariant. Per sample,
we build the (N, N) squared-distance cost between predicted and target slots,
solve the optimal one-to-one assignment exactly (Hungarian algorithm, scipy),
and take the MSE over the matched pairs. The assignment is computed on a
detached CPU copy (DETR-style): gradients flow through the matched distances —
into the predictor AND the target encoder (joint training, D7) — while the
discrete assignment is treated as a constant (correct a.e.: the argmin is
piecewise constant in the inputs).

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


def match_slots(
    pred: Float[Tensor, "b n d"], target: Float[Tensor, "b n d"]
) -> Int64[Tensor, "b n"]:
    """Optimal slot assignment π per sample: pred slot i ↔ target slot π(i).

    Non-differentiable (runs on a detached CPU copy). Also used by eval code to
    align slots with ground-truth objects.
    """
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(
            f"expected matching (B, N, d), got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
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


__all__ = ["hungarian_mse", "match_slots"]
