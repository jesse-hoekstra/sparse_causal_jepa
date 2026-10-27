"""Change only optimized parameter-auxiliary weights, preserving forecast and RNG."""

from typing import Literal

from torch import Tensor

from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective, cached_forward
from diagnostics.multiquery_control.data import TensorSplit
from scjepa.models.model import SCJEPA

Arm = Literal["forecast", "parameter_objective"]


def training_loss(
    model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective, arm: Arm
) -> tuple[Tensor, dict[str, Tensor]]:
    """Use the same auxiliary draws in both arms; retain attached E/P prediction gradients."""
    if arm not in ("forecast", "parameter_objective"):
        raise ValueError("Unknown parameter-objective survival arm")
    if any(
        p.requires_grad
        for block in (model.online_encoder, model.target_encoder)
        for p in block.parameters()
    ):
        raise ValueError("Both state charts must stay frozen")
    if not all(
        p.requires_grad
        for block in (model.parameter_encoder, model.predictor)
        for p in block.parameters()
    ):
        raise ValueError("Both parameter encoder and predictor must train")
    total, terms = objective(cached_forward(model, batch))
    loss = terms["prediction"] if arm == "forecast" else total
    restored = arm == "parameter_objective"
    return loss, {
        **terms,
        "optimized_total": loss,
        "restored_parameter_objective": total,
        "effective_parameter_sigreg_weight": loss.new_tensor(0.4 if restored else 0),
        "effective_parameter_consistency_weight": loss.new_tensor(1 if restored else 0),
        "state_ema_disabled": loss.new_ones(()),
    }
