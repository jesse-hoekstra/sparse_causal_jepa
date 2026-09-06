"""Shared teacher-forcing and endpoint-only T=2 losses for both experiments.

Rows reach these functions already aligned: Experiment 1 uses simulator row
order; Experiment 2 reorders EMA targets using one detached assignment derived
from the observed context, fixed throughout each episode. No rematching takes
place inside either loss.

The T=2 loss supervises only the second generated state, averaging over all
windows, objects and coordinates. The first transition is already covered by
teacher forcing. Neither experiment optimizes a long-horizon rollout loss.
"""

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


def aligned_mse(pred: Float[Tensor, "b n k"], target: Float[Tensor, "b n k"]) -> Float[Tensor, ""]:
    """Object-aligned MSE for tracked rows whose identity is known (Eq. 32/39)."""
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(
            f"expected matching (B, N, k), got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    return F.mse_loss(pred, target)


def rollout_t2_endpoint_mse(
    pred: Float[Tensor, "b w n d"],
    target: Float[Tensor, "b w n d"],
) -> Float[Tensor, ""]:
    """Mean endpoint error for independently anchored two-step rollouts.

    Both tensors contain only ``S_hat_(t+2)`` / ``S_(t+2)``. A plain
    elementwise mean therefore implements the required average over episodes,
    windows, objects, and coordinates without an intermediate-step loss.
    """
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError(
            f"expected matching (B, W, N, d), got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    return F.mse_loss(pred, target)


__all__ = ["aligned_mse", "rollout_t2_endpoint_mse"]
