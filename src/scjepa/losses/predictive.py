"""Prediction losses for the SCJEPA regimes.

Teacher forcing (experiments.pdf Eq. 39 == hybrid Eq. 32)

    L_TF = (1 / (|I_TF| N d_s)) Σ_{t∈I_TF} ||Ŝ^TF_{t+1} - sg(S̄_{t+1})||²_F

with the K transitions flattened into the batch axis, which is exactly the
element-wise MSE over aligned rows. Row identity is known (tracked simulator
objects), so no assignment of any kind is performed here; the visual
experiments use a single detached trajectory-level assignment on the LOSS side
(§6.5), never per-step rematching.

State-to-state local composition auxiliary

    L_AR2 = mean_(b,w,n,d) ||S_hat_(t+2) - S_(t+2)||²

supervises only the endpoint after one generated state has been fed back. The
first transition is already covered by ``L_TF``. The mean includes the window
axis, so changing or duplicating the number of anchors cannot multiply the
coefficient's scale.

``weighted_rollout_mse`` and ``rollout_weights`` remain only for the separately
defined visual-to-visual experiment, whose latent-space objective has not been
changed by the state-to-state simplification.
"""

import torch
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


def weighted_rollout_mse(
    pred: Float[Tensor, "b j n k"],
    target: Float[Tensor, "b j n k"],
    weights: Float[Tensor, "j"],
) -> Float[Tensor, ""]:
    """Experiment 3's fixed latent K-step rollout loss.

    ``pred[:, k-1]`` is Ŝ^[k]_{t+k} and ``target[:, k-1]`` is S̄_{t+k}; the
    target must already carry the stop-gradient from the EMA target branch.

    The 1/(N d_s) normalisation is the mean over the trailing (N, k) axes and
    the outer 1/K is the mean over the horizon axis. With ``rollout_weights``
    this equals the MEAN per-step error over k = 2..K.
    """
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError(
            f"expected matching (B, K, N, k), got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    if weights.ndim != 1 or weights.shape[0] != pred.shape[1]:
        raise ValueError(
            f"expected one weight per horizon step (K={pred.shape[1]}), got {tuple(weights.shape)}"
        )
    per_step = ((pred - target) ** 2).mean(dim=(0, 2, 3))  # the 1/(N d_s) factor
    return (weights.to(per_step) * per_step).mean()  # the (1/K) Σ_k factor


def rollout_weights(horizon: int, device: torch.device | None = None) -> Float[Tensor, "j"]:
    """Experiment 3 weights: w_1 = 0, uniform over k = 2..K, mean one.

    w_1 = 0 always — see the module docstring: at k=1 the rollout recomputes the
    teacher-forced term, and the rollout start is structurally inside I_TF, so
    the write-up's coverage condition holds without it. The remaining weights
    are K/(K-1) so that K⁻¹ Σ_k w_k = 1, which keeps L_roll a MEAN per-step
    error and stops λ_roll from silently rescaling when K changes.
    """
    if horizon < 2:
        raise ValueError(f"horizon must be >= 2 once w_1 = 0 is dropped, got {horizon}")
    weights = torch.full((horizon,), horizon / (horizon - 1), device=device)
    weights[0] = 0.0
    return weights


__all__ = [
    "aligned_mse",
    "rollout_t2_endpoint_mse",
    "rollout_weights",
    "weighted_rollout_mse",
]
