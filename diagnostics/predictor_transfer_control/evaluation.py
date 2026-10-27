"""Original six-code assessment shared by both frozen-encoder predictor arms."""

from typing import Any

from torch import Tensor

from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.mlp_acquisition_control.evaluation import assess_model as assess_six
from diagnostics.mlp_acquisition_control.evaluation import (
    measurement_metrics as measurement_metrics,
)
from diagnostics.multiquery_control.data import EvaluationData
from scjepa.models.model import SCJEPA


def assess_model(
    model: SCJEPA,
    data: EvaluationData,
    selection: DiagnosticSelection,
    *,
    bootstrap_samples: int,
    seed: int,
    final: bool = False,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Keep original probes and physical interventions, with correct diagnostic scope."""
    report, raw = assess_six(
        model, data, selection, bootstrap_samples=bootstrap_samples, seed=seed, final=final
    )
    report.pop("mlp_architecture_diagnostic")
    report["frozen_encoder_predictor_transfer_diagnostic"] = True
    report["predictor_kind"] = type(model.predictor).__name__
    return report, raw
