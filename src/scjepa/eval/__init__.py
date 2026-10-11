"""Reusable evaluation algorithms; experiments supply splits, labels, and tolerances."""

from scjepa.eval.graph import structural_hamming_distance
from scjepa.eval.observational_equivalence import (
    OeSummary,
    oe_worst_step_nrmse,
    summarize_oe,
    training_coordinate_std,
)
from scjepa.eval.parameters import MccReport, nonlinear_mcc
from scjepa.eval.state_probes import StateProbe, fit_state_probe, state_probe_metrics

__all__ = [
    "MccReport",
    "OeSummary",
    "StateProbe",
    "fit_state_probe",
    "nonlinear_mcc",
    "oe_worst_step_nrmse",
    "state_probe_metrics",
    "structural_hamming_distance",
    "summarize_oe",
    "training_coordinate_std",
]
