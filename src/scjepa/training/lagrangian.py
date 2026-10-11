"""Optional generic dual controller for externally specified constraints.

The default Trainer uses explicit independent graph weights. Future experiments
may compose this utility with a constraint they define; no calibration rule,
threshold, schedule, or initial weight is imposed here.
"""

import math

import torch
from torch import Tensor, nn


class SparsityLagrangian(nn.Module):
    """Dual controller with inverse-exponential penalty weight and buffered state."""

    ma_error: Tensor
    log_lambda: Tensor

    def __init__(self, tau: float, step_size: float, lambda_init: float, momentum: float) -> None:
        """Set an explicit constraint threshold and dual update parameters."""
        super().__init__()
        if not all(math.isfinite(value) for value in (tau, step_size, lambda_init, momentum)):
            raise ValueError("dual controller settings must be finite")
        if lambda_init <= 0 or step_size < 0 or not 0 <= momentum < 1:
            raise ValueError("require lambda_init > 0, step_size >= 0, and 0 <= momentum < 1")
        self.tau = tau
        self.step_size = step_size
        self.momentum = momentum
        self.register_buffer("log_lambda", torch.tensor(math.log(lambda_init)))
        self.register_buffer("ma_error", torch.tensor(0.0))

    @property
    def penalty_weight(self) -> Tensor:
        """Return the current inverse dual weight."""
        return torch.exp(-self.log_lambda)

    @torch.no_grad()
    def update(self, constraint: Tensor) -> None:
        """Move the log dual using the detached moving-average constraint error."""
        error = constraint.detach() - self.tau
        self.ma_error.mul_(self.momentum).add_(error, alpha=1.0 - self.momentum)
        self.log_lambda.add_(self.step_size * self.ma_error)


__all__ = ["SparsityLagrangian"]
