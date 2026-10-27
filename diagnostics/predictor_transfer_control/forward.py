"""Fresh dense predictors fitted to one frozen learned PCA6 code per context."""

from typing import cast

import torch
from torch import Tensor, nn

from diagnostics.mlp_acquisition_control.forward import (
    AcquisitionObjective,
    AcquisitionOutput,
    cached_forward,
)
from diagnostics.mlp_oracle_code_control.model import MLPPredictor
from diagnostics.multiquery_control.data import TensorSplit
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import Spartan


def training_loss(
    model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
) -> tuple[Tensor, dict[str, Tensor]]:
    """Optimize original SPARTAN forecast/logit or MLP forecast; keep auxiliary RNG paired."""
    if set(batch) != {"online", "target", "actions"} or any(
        value.requires_grad for value in batch.values()
    ):
        raise ValueError("Predictor transfer accepts only frozen label-free cached inputs")
    if any(
        parameter.requires_grad
        for module in (model.parameter_encoder, model.online_encoder, model.target_encoder)
        for parameter in module.parameters()
    ):
        raise ValueError("Parameter and state encoders must remain frozen")
    predictor = cast(nn.Module, model.predictor)
    if isinstance(predictor, MLPPredictor):
        output = cached_forward(model, batch)
        logit = output.predictions.new_zeros(())
        logit_weight = 0.0
    elif isinstance(predictor, Spartan):
        if any(not layer.dense or layer.identity for layer in model.predictor.layers):
            raise ValueError("Original SPARTAN transfer requires all dense gates held open")
        original = cached_multiquery_forward(model, batch, queries=1)
        output = AcquisitionOutput(
            original.predictions, original.targets, original.online_states, original.parameters
        )
        logit = torch.stack([value.logit_penalty for value in original.dynamics]).mean()
        logit_weight = 0.0001
    else:
        raise TypeError("Only original dense SPARTAN or the matched MLP211 is supported")
    total, terms = objective(output)
    loss = terms["prediction"] + logit_weight * logit
    return loss, {
        **terms,
        "ordinary_total_without_constant_state_sigreg": total + logit_weight * logit,
        "optimized_total": loss,
        "logit": logit,
        "effective_logit_weight": loss.new_tensor(logit_weight),
        "frozen_code_auxiliary_terms_constant": loss.new_ones(()),
        "attention_logit_absent": loss.new_tensor(float(logit_weight == 0)),
    }
