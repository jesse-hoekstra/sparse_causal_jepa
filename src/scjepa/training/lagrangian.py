"""GECO-style dual controller for the path-objective weight (experiments.pdf §6.1.3).

The sparse regime solves  min L_path  s.t.  c = L_pred + λ_logit·L_logit ≤ τ
via the objective  L = L_pred + λ_logit·L_logit + λ⁻¹·L_path  (Eq. 40), with

    log λ  ←  log λ + alpha · MA[c - τ],

exactly as declared: "The dual variable starts at λ = 10⁶, making the initial
path weight 10⁻⁶, and increases or decreases according to a moving average of
c - τ." No clamp exists in the write-up and none is applied here; the step
size alpha is the one numerical knob the papers leave open.

λ and the moving average live in buffers so checkpoints resume exactly.
"""

import math

import torch
from torch import Tensor, nn


class SparsityLagrangian(nn.Module):
    """Dual controller: path weight is 1/λ; λ moves on a moving average of c - τ."""

    ma_error: Tensor
    log_lambda: Tensor

    def __init__(
        self,
        tau: float,
        step_size: float = 2e-2,
        lambda_init: float = 1e6,
        momentum: float = 0.99,
    ) -> None:
        """Build the controller.

        Args:
            tau: τ, the held-out constraint of the converged dense reference
                (recalibrated for every architecture and seed, §6.1.3).
            step_size: alpha, dual step on log λ (unspecified upstream).
            lambda_init: λ₀ = 10⁶ — initial path weight 10⁻⁶ (dynamics first).
            momentum: Moving-average momentum for the c - τ estimate.
        """
        super().__init__()
        if lambda_init <= 0:
            raise ValueError("lambda_init must be positive")
        self.tau = tau
        self.step_size = step_size
        self.momentum = momentum
        self.register_buffer("log_lambda", torch.tensor(math.log(lambda_init)))
        self.register_buffer("ma_error", torch.tensor(0.0))

    @property
    def penalty_weight(self) -> Tensor:
        """Current path-objective weight λ⁻¹ (Eq. 40)."""
        return torch.exp(-self.log_lambda)

    @torch.no_grad()
    def update(self, constraint: Tensor) -> None:
        """Dual step: log λ += alpha · MA[c - τ]."""
        error = constraint.detach() - self.tau
        self.ma_error.mul_(self.momentum).add_(error, alpha=1.0 - self.momentum)
        self.log_lambda.add_(self.step_size * self.ma_error)


__all__ = ["SparsityLagrangian"]
