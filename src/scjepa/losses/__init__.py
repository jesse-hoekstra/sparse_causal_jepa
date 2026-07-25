"""Loss terms (experiments.pdf §6.1.3 / §6.2).

The state-to-state regime optimizes Eq. 40: aligned raw-state MSE (Eq. 39) plus the
attention-logit penalty and the dual-weighted path objective — the latter two
are computed inside the SPARTAN module; only the prediction term lives here.
Object rows are tracked, so the loss is plain aligned MSE: no matching.
"""

from scjepa.losses.predictive import aligned_mse

__all__ = ["aligned_mse"]
