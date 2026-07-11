"""Evaluation: causal-graph metrics, probes, autoregressive rollouts.

Will contain: SHD/MCC diagnostics against ground-truth causal structure (reading
SPARTAN's exposed interaction graph), linear probes on the learned channels, and
multi-step autoregressive rollout evaluation (eval-time only per D6 — e.g.
CLEVRER frames 128 to 160; the rollout horizon Tp is an eval knob, not a training
objective).

Owner: experiment-infra-engineer.
"""

from scjepa.eval.graph import (
    gt_graphs_from_contacts,
    read_learned_graphs,
    structural_hamming_distance,
)
from scjepa.eval.harness import IdentifiabilityReport, evaluate_identifiability
from scjepa.eval.parameters import (
    correlation_matrix,
    marginal_recovery,
    mean_max_correlation,
    nonlinear_mcc,
)

__all__ = [
    "IdentifiabilityReport",
    "correlation_matrix",
    "evaluate_identifiability",
    "gt_graphs_from_contacts",
    "marginal_recovery",
    "mean_max_correlation",
    "nonlinear_mcc",
    "read_learned_graphs",
    "structural_hamming_distance",
]
