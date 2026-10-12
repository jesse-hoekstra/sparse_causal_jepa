"""GECO-style sparse optimization with an explicit noncausal-loss constraint.

The primal objective is ``base_loss + path_loss / lambda``: this has the
same parameter gradient as the rescaled Lagrangian after dropping ``-tau``.
``base_loss`` includes prediction and all enabled representation/logit terms.
A positive constraint violation increases lambda and reduces sparsity pressure.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class GECOConfig:
    """Settings for a calibrated loss constraint; disabled by default.

    ``tau`` bounds prediction MSE plus weighted SIGReg and attention-logit
    regularization, not prediction MSE alone. The experiment supplies it from
    a fully connected reference trained with the same noncausal objective.
    Graph-block weights retain their role as independent relative coefficients.
    """

    enabled: bool = False
    tau: float | None = None
    step_size: float = 0.1
    momentum: float = 0.99
    lambda_init: float = 1000.0
    log_lambda_min: float = -12.0
    log_lambda_max: float = 12.0

    def __post_init__(self) -> None:
        """Reject undefined constraints and unsafe numerical settings."""
        if self.enabled and self.tau is None:
            raise ValueError("enabled GECO requires an explicit calibrated tau")
        if self.tau is not None and (not math.isfinite(self.tau) or self.tau < 0):
            raise ValueError("GECO tau must be finite and non-negative")
        if not all(
            math.isfinite(value)
            for value in (
                self.step_size,
                self.momentum,
                self.lambda_init,
                self.log_lambda_min,
                self.log_lambda_max,
            )
        ):
            raise ValueError("GECO controller settings must be finite")
        if self.lambda_init <= 0 or self.step_size < 0 or not 0 <= self.momentum < 1:
            raise ValueError("require lambda_init > 0, step_size >= 0, and 0 <= momentum < 1")
        if not -60 <= self.log_lambda_min < self.log_lambda_max <= 60:
            raise ValueError("GECO log bounds must satisfy -60 <= min < max <= 60")
        if not self.log_lambda_min <= math.log(self.lambda_init) <= self.log_lambda_max:
            raise ValueError("GECO lambda_init must lie within the configured log bounds")


class SparsityLagrangian(nn.Module):
    """Bounded inverse-exponential dual with checkpointed moving-average state.

    Only the trainer updates this controller, once per accepted optimizer step.
    Its input is already averaged over that accumulation group and all ranks.
    The first accepted value initializes the moving average without startup bias.
    """

    ma_error: Tensor
    log_lambda: Tensor
    num_updates: Tensor

    def __init__(
        self,
        tau: float,
        step_size: float,
        lambda_init: float,
        momentum: float,
        log_lambda_min: float = -12.0,
        log_lambda_max: float = 12.0,
    ) -> None:
        """Set the calibrated threshold and explicit dual update parameters."""
        super().__init__()
        config = GECOConfig(
            enabled=True,
            tau=tau,
            step_size=step_size,
            lambda_init=lambda_init,
            momentum=momentum,
            log_lambda_min=log_lambda_min,
            log_lambda_max=log_lambda_max,
        )
        self.tau = tau
        self.step_size = config.step_size
        self.momentum = config.momentum
        self.log_lambda_min = config.log_lambda_min
        self.log_lambda_max = config.log_lambda_max
        self.register_buffer("log_lambda", torch.tensor(math.log(lambda_init)))
        self.register_buffer("ma_error", torch.tensor(0.0))
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

    @property
    def penalty_weight(self) -> Tensor:
        """Return the current inverse multiplier for the weighted path loss."""
        return torch.exp(-self.log_lambda)

    def validate_state(self) -> None:
        """Reject nonfinite or out-of-bounds controller state on checkpoint restore."""
        log_value = float(self.log_lambda)
        if (
            not math.isfinite(log_value)
            or not bool(torch.isfinite(self.ma_error))
            or not self.log_lambda_min - 1e-5 <= log_value <= self.log_lambda_max + 1e-5
            or int(self.num_updates) < 0
        ):
            raise ValueError("invalid checkpoint GECO controller state")

    @torch.no_grad()
    def update(self, constraint: Tensor) -> None:
        """Apply one detached accepted-step loss and clamp the log multiplier."""
        if constraint.numel() != 1 or not bool(torch.isfinite(constraint).all()):
            raise ValueError("GECO update requires one finite scalar base loss")
        error = constraint.detach().reshape(()) - self.tau
        if int(self.num_updates) == 0:
            self.ma_error.copy_(error)
        else:
            self.ma_error.mul_(self.momentum).add_(error, alpha=1.0 - self.momentum)
        self.log_lambda.add_(self.step_size * self.ma_error).clamp_(
            self.log_lambda_min, self.log_lambda_max
        )
        self.num_updates.add_(1)


__all__ = ["GECOConfig", "SparsityLagrangian"]
