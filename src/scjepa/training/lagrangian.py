"""Lagrangian relaxation schedule for the SPARTAN sparsity weight (App. A.2).

The SPARTAN paper solves ``min |Ā| s.t. MSE <= τ`` via the GECO-style rewrite

    loss = (pred_loss - τ) + |Ā| / λ,      λ ← alpha^{sign} scaling: λ ← exp(alpha·(MSE-τ))·λ

with a moving-average estimator of (MSE - τ) for stability, λ initialised HIGH
(training first focuses on dynamics; sparsity pressure grows once the error
drops below the target τ). τ is ideally set to the loss a fully connected model
achieves. Note ``exp(MSE-τ) > 1`` when the error exceeds the target, so λ grows
and the sparsity term |Ā|/λ weakens — error first, pruning later.

Implemented as an ``nn.Module`` so λ and the moving average live in buffers and
ride along in checkpoints for exact resume.
"""

import math

import torch
from torch import Tensor, nn


class SparsityLagrangian(nn.Module):
    """GECO-style dual controller for the sparsity penalty weight 1/λ."""

    ma_error: Tensor
    log_lambda: Tensor

    def __init__(
        self,
        tau: float,
        step_size: float = 1e-3,
        lambda_init: float = 1e3,
        momentum: float = 0.99,
        lambda_max: float = 1e6,
        lambda_min: float = 1e-3,
    ) -> None:
        """Build the controller.

        Args:
            tau: Target constraint value (SPARTAN sets it to the loss of a fully
                connected model — calibrate under the IDENTICAL config with only
                the sparsity toggle off; τ is config-dependent, D12). Since D17
                the caller feeds ``update`` either raw prediction MSE (literal
                GT-state ruler) or D17's variance-normalized prediction error
                (trainable target space), plus the logit term. Tau must be
                calibrated with the same configured normalization.
            step_size: alpha, dual ascent step size on log λ.
            momentum: Moving-average momentum for the (loss - τ) estimate.
            lambda_init: Initial λ (paper: high, so sparsity starts switched off).
            lambda_max: Clamp so the dual stays responsive if the loss sits above
                τ for a long stretch (unclamped, λ can run to 1e13+ and then
                take just as long to come back once the error drops).
            lambda_min: Symmetric lower clamp.
        """
        super().__init__()
        if lambda_init <= 0:
            raise ValueError("lambda_init must be positive")
        if not lambda_min <= lambda_init <= lambda_max:
            raise ValueError("need lambda_min <= lambda_init <= lambda_max")
        self.tau = tau
        self.step_size = step_size
        self.momentum = momentum
        self.log_lambda_max = math.log(lambda_max)
        self.log_lambda_min = math.log(lambda_min)
        self.register_buffer("log_lambda", torch.tensor(math.log(lambda_init)))
        self.register_buffer("ma_error", torch.tensor(0.0))

    @property
    def penalty_weight(self) -> Tensor:
        """Current sparsity weight 1/λ (multiplies |Ā| in the objective)."""
        return torch.exp(-self.log_lambda)

    @torch.no_grad()
    def update(self, pred_loss: Tensor) -> None:
        """Dual step: λ ← exp(alpha·MA[loss - τ])·λ, i.e. log λ += alpha·MA[loss - τ]."""
        error = pred_loss.detach() - self.tau
        self.ma_error.mul_(self.momentum).add_(error, alpha=1.0 - self.momentum)
        self.log_lambda.add_(self.step_size * self.ma_error).clamp_(
            self.log_lambda_min, self.log_lambda_max
        )


__all__ = ["SparsityLagrangian"]
