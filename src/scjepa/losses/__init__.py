"""Predictive loss terms for the three SCJEPA observation regimes.

The state-to-state regime uses aligned teacher-forced MSE plus a mean endpoint
loss over sampled T=2 windows. Object rows are tracked, so neither needs
matching. The fixed-horizon helpers are retained for the distinct
visual-to-visual latent objective.
"""

from scjepa.losses.predictive import (
    aligned_mse,
    rollout_t2_endpoint_mse,
    rollout_weights,
    weighted_rollout_mse,
)

__all__ = [
    "aligned_mse",
    "rollout_t2_endpoint_mse",
    "rollout_weights",
    "weighted_rollout_mse",
]
