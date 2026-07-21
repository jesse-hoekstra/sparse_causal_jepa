"""Evaluation: causal-graph metrics, probes, autoregressive rollouts.

Will contain: SHD/MCC diagnostics against ground-truth causal structure (reading
SPARTAN's exposed interaction graph), linear probes on the learned channels, and
multi-step autoregressive rollout evaluation (eval-time only per D6 — e.g.
CLEVRER frames 128 to 160; the rollout horizon Tp is an eval knob, not a training
objective).

Owner: experiment-infra-engineer.
"""

from scjepa.eval.graph import (
    align_parameter_columns,
    gt_graphs_from_contacts,
    read_learned_graphs,
    structural_hamming_distance,
)
from scjepa.eval.harness import IdentifiabilityReport, evaluate_identifiability
from scjepa.eval.parameters import (
    OneToOneRecovery,
    correlation_matrix,
    marginal_recovery,
    mean_max_correlation,
    nonlinear_mcc,
    one_to_one_recovery,
    optimal_one_to_one_assignment,
)

__all__ = [
    "IdentifiabilityReport",
    "OneToOneRecovery",
    "align_parameter_columns",
    "correlation_matrix",
    "evaluate_identifiability",
    "gt_graphs_from_contacts",
    "marginal_recovery",
    "mean_max_correlation",
    "nonlinear_mcc",
    "one_to_one_recovery",
    "optimal_one_to_one_assignment",
    "read_learned_graphs",
    "structural_hamming_distance",
]
