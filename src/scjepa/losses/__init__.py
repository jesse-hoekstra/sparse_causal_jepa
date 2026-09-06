"""Shared prediction losses for the two SCJEPA observation regimes."""

from scjepa.losses.predictive import aligned_mse, rollout_t2_endpoint_mse

__all__ = ["aligned_mse", "rollout_t2_endpoint_mse"]
