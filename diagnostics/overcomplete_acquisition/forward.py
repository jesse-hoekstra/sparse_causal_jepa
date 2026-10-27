"""Original single-context inference, cross-donor rollout and label-free losses."""

import torch
from torch import Tensor

from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective, AcquisitionOutput
from diagnostics.multiquery_control.data import TensorSplit
from diagnostics.overcomplete_acquisition.model import OvercompletePredictor
from scjepa.models.model import SCJEPA, cross_episode_parameters


def cached_forward(model: SCJEPA, batch: TensorSplit) -> AcquisitionOutput:
    """Infer each individual context and forecast a nonself episode in its group."""
    if set(batch) != {"online", "target", "actions"}:
        raise ValueError("Acquisition training must exclude physical labels and group IDs")
    online, target, actions = batch["online"], batch["target"], batch["actions"]
    groups = len(online)
    if online.shape != (groups, 10, 12, 6) or target.shape != online.shape:
        raise ValueError("Expected original R10/T12 six-state caches")
    if actions.shape != (groups, 10, 11, 16):
        raise ValueError("Expected original R10/T12 sixteen-action cache")
    if any(value.requires_grad for value in batch.values()):
        raise ValueError("All cached states and actions must remain frozen")
    if not isinstance(model.predictor, OvercompletePredictor) or (
        model.context_len != 10 or not model.autoregressive or model.parameter_source != "learned"
    ):
        raise ValueError("Require the dedicated MLP with original learned-code K10 autoregression")
    codes = model.parameter_encoder(
        online[:, :, :10].flatten(0, 1), actions[:, :, :9].flatten(0, 1)
    ).reshape(groups, 10, 12)
    donor = cross_episode_parameters(codes, training=model.training).flatten(0, 1)
    history = online[:, :, 9:10].flatten(0, 1)
    predictions: list[Tensor] = []
    for step in (9, 10):
        predicted = model.predictor(
            history, donor, actions[:, :, step : step + 1].flatten(0, 1)
        ).prediction
        predictions.append(predicted)
        history = predicted[:, None]
    return AcquisitionOutput(
        torch.stack(predictions, 1).reshape(groups, 10, 2, 6),
        target[:, :, 10:12],
        online,
        codes,
    )


def training_loss(
    model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
) -> tuple[Tensor, dict[str, Tensor]]:
    """Optimize prediction only, retaining original auxiliary diagnostics and RNG draws."""
    original, terms = objective(cached_forward(model, batch))
    loss = terms["prediction"]
    return loss, {
        **terms,
        "ordinary_total_without_constant_state_sigreg": original,
        "optimized_total": loss,
        "effective_parameter_regularizer_weight": loss.new_zeros(()),
        "effective_parameter_consistency_weight": loss.new_zeros(()),
    }
