"""Prediction losses — the two branches of the hybrid objective.

Teacher forcing (experiments.pdf Eq. 39 == hybrid Eq. 32)

    L_TF = (1 / (|I_TF| N d_s)) Σ_{t∈I_TF} ||Ŝ^TF_{t+1} - sg(S̄_{t+1})||²_F

with the K transitions flattened into the batch axis, which is exactly the
element-wise MSE over aligned rows. Row identity is known (tracked simulator
objects), so no assignment of any kind is performed here; the visual
experiments use a single detached trajectory-level assignment on the LOSS side
(§6.5), never per-step rematching.

Dense K-step autoregressive rollout (hybrid Eq. 35)

    L_roll^(K) = (1/K) Σ_{k=1..K} w_k (1 / (N d_s)) ||Ŝ^[k]_{t+k} - sg(S̄_{t+k})||²_F

supervises EVERY autoregressive prefix, not just the terminal state — that is
the stated difference from V-JEPA 2-AC (hybrid Remark 4).

CHOICE OF w_k (Eq. 35 leaves them free; this is our decision, not the paper's):

* w_1 = 0, ALWAYS. At k=1 the rollout computes f_gamma(S^on_t, θ̂) against
  S̄_{t+1} — bit-for-bit the teacher-forced term at the same t, differing only
  by the gate draw. Since the rollout starts at t = Tpar-1 and I_TF =
  {Tpar-1, …, T-2}, the start is ALWAYS inside I_TF, so the write-up's coverage
  condition ("every state in the rollout prefix constrained by either term") is
  satisfied by L_TF structurally, not just for the current numbers. Keeping
  w_1 > 0 would only double-count one transition inside a bound that has no
  slack to spare.
* Uniform over the remaining k, normalised so K⁻¹ Σ_k w_k = 1 — the third
  option the write-up offers. Uniform because §4.4(ii) requires only that every
  prefix be constrained, so any positive profile satisfies the theory and a
  non-uniform one would be an unmotivated hyperparameter. Normalising means
  L_roll is the MEAN per-step error, so λ_roll keeps its meaning if K changes
  (it would otherwise silently rescale the constraint).

The alternative worth knowing about is a geometric discount w_k ~ gamma^(k-1),
which counteracts late steps dominating as error compounds. In the 1500-step
CPU smoke (paper geometry, K=10) L_roll stayed within ~1.4x of L_TF rather than
running away, so there is nothing to counteract yet; revisit only if the ratio
of L_roll to L_TF grows large during a real run.
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


def weighted_rollout_mse(
    pred: Float[Tensor, "b j n k"],
    target: Float[Tensor, "b j n k"],
    weights: Float[Tensor, "j"],
) -> Float[Tensor, ""]:
    """Hybrid Eq. 35: dense prefix supervision of a K-step rollout.

    ``pred[:, k-1]`` is Ŝ^[k]_{t+k} and ``target[:, k-1]`` is S̄_{t+k}; the
    target must already carry the stop-gradient (in Experiment 1 it is fixed
    observed data, so sg is a no-op, but the EMA-target regime needs it).

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
    """Eq. 35 weights: w_1 = 0, uniform over k = 2..K, normalised to mean 1.

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


__all__ = ["aligned_mse", "rollout_weights", "weighted_rollout_mse"]
