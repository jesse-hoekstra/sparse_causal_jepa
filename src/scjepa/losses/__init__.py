"""Loss terms: predictive loss, anti-collapse regularizer, sparsity penalty.

Will contain: the single-step Hungarian-matched predictive loss (D6 — one
assignment per sample over N slots via scipy linear_sum_assignment on a detached
cost; gradients through matched pairs only), the config-selectable anti-collapse
regularizer (D3 — VISReg default, SIGReg as ablation; wrappers around code
vendored in third_party/), and the SPARTAN sparsity penalty.

Owner: paper-to-code-translator (definitions traceable to equations);
experiment-infra-engineer assembles the terms into the training objective.
"""

from scjepa.losses.predictive import hungarian_mse, match_slots
from scjepa.losses.regularizer import RegularizerKind, SlotRegularizer

__all__ = ["RegularizerKind", "SlotRegularizer", "hungarian_mse", "match_slots"]
