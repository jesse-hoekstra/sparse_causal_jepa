"""Direct coordinatewise latent prediction against stopped-gradient targets."""

import torch.nn.functional as F
from torch import Tensor


def prediction_mse(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean squared error over episodes, prediction times, and coordinates.

    Inputs are ``(B, W, d_z)``. No assignment, permutation, or alignment is
    performed. The target is detached here as a safeguard for external callers.
    """
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("prediction and target must have matching (B, W, d_z) shapes")
    if prediction.numel() == 0:
        raise ValueError("prediction and target must be nonempty")
    return F.mse_loss(prediction, target.detach())


__all__ = ["prediction_mse"]
