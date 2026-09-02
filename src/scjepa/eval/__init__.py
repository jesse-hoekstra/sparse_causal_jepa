"""Evaluation: causal-graph metrics, probes, autoregressive rollouts.

Will contain: SHD/MCC diagnostics against ground-truth causal structure (reading
SPARTAN's exposed interaction graph), linear probes on the learned channels, and
multi-step autoregressive rollout evaluation (eval-time only per D6 — e.g.
CLEVRER frames 128 to 160; the rollout horizon Tp is an eval knob, not a training
objective).

Owner: experiment-infra-engineer.
"""

from scjepa.eval.graph import (
    gt_causal_graph_from_contacts,
    read_learned_graph,
    structural_hamming_distance,
)
from scjepa.eval.harness import IdentifiabilityReport, evaluate_identifiability
from scjepa.eval.observational_equivalence import (
    OeSummary,
    oe_worst_step_nrmse,
    summarize_oe,
    training_coordinate_std,
)
from scjepa.eval.parameters import MccReport, nonlinear_mcc

__all__ = [
    "IdentifiabilityReport",
    "MccReport",
    "OeSummary",
    "evaluate_identifiability",
    "gt_causal_graph_from_contacts",
    "nonlinear_mcc",
    "oe_worst_step_nrmse",
    "read_learned_graph",
    "structural_hamming_distance",
    "summarize_oe",
    "training_coordinate_std",
]
