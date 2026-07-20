"""Loss terms: predictive loss, anti-collapse regularizer, sparsity penalty.

Prediction matching is aligned for tracked object rows and Hungarian-matched for
unordered learned slots. The package also contains the config-selectable
anti-collapse regularizer and the SPARTAN sparsity penalty.

Owner: paper-to-code-translator (definitions traceable to equations);
experiment-infra-engineer assembles the terms into the training objective.
"""

from scjepa.losses.predictive import (
    aligned_mse,
    hungarian_mse,
    match_slots,
    prediction_constraint,
    prediction_mse,
    resolve_constraint_normalization,
    resolve_prediction_matching,
)
from scjepa.losses.regularizer import RegularizerKind, SlotRegularizer

__all__ = [
    "RegularizerKind",
    "SlotRegularizer",
    "aligned_mse",
    "hungarian_mse",
    "match_slots",
    "prediction_constraint",
    "prediction_mse",
    "resolve_constraint_normalization",
    "resolve_prediction_matching",
]
