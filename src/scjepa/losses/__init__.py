"""Loss terms (experiments.pdf §6.1.3 / §6.2, hybrid write-up §4).

The state-to-state regime optimizes the hybrid objective Eq. 36: teacher-forced
aligned raw-state MSE (Eq. 32/39) plus the dense K-step autoregressive rollout
term (Eq. 35), plus the attention-logit penalty and the dual-weighted path
objective — the latter two are computed inside the SPARTAN module; only the
prediction terms live here. Object rows are tracked, so both are plain aligned
MSE: no matching.
"""

from scjepa.losses.predictive import aligned_mse, rollout_weights, weighted_rollout_mse

__all__ = ["aligned_mse", "rollout_weights", "weighted_rollout_mse"]
