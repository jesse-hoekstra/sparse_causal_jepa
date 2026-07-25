"""Prediction loss — experiments.pdf Eq. 39.

L_pred = (1 / (|I| N k)) Σ_{t∈I} Σ_i ||ẑⁱ_{t+1} - zⁱ_{t+1}||²: with the K
transitions flattened into the batch axis this is exactly the element-wise MSE
over aligned rows. Row identity is known (tracked simulator objects), so no
assignment of any kind is performed here; the visual experiments use a single
detached trajectory-level assignment on the LOSS side (§6.5), never per-step
rematching.
"""

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


def aligned_mse(pred: Float[Tensor, "b n k"], target: Float[Tensor, "b n k"]) -> Float[Tensor, ""]:
    """Object-aligned MSE for tracked rows whose identity is known."""
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(
            f"expected matching (B, N, k), got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    return F.mse_loss(pred, target)


__all__ = ["aligned_mse"]
