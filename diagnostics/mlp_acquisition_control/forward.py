"""Original single-context inference, cross-donor rollout and label-free losses."""

from typing import NamedTuple

import torch
from torch import Tensor, nn

from diagnostics.mlp_oracle_code_control.model import MLPPredictor
from diagnostics.multiquery_control.data import TensorSplit
from scjepa.losses import RepresentationRegularizer
from scjepa.models.model import SCJEPA, cross_episode_parameters
from scjepa.training.loop import ObjectiveConfig


class AcquisitionOutput(NamedTuple):
    """One code per context, reused across both attached forecast steps."""

    predictions: Tensor
    targets: Tensor
    online_states: Tensor
    parameters: Tensor


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
    if not isinstance(model.predictor, MLPPredictor) or (
        model.context_len != 10 or not model.autoregressive or model.parameter_source != "learned"
    ):
        raise ValueError("Require the dedicated MLP with original learned-code K10 autoregression")
    codes = model.parameter_encoder(
        online[:, :, :10].flatten(0, 1), actions[:, :, :9].flatten(0, 1)
    ).reshape(groups, 10, 6)
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


class AcquisitionObjective(nn.Module):
    """Original prediction1/SIGReg0.4/consistency1 without nonexistent MLP logits."""

    def __init__(self, config: ObjectiveConfig) -> None:
        """Require the authenticated ordinary objective; omit only constant state loss."""
        super().__init__()
        if (
            config.state_regularizer != "sigreg"
            or config.state_regularizer_weight != 0.4
            or config.parameter_regularizer != "sigreg"
            or config.parameter_regularizer_weight != 0.4
            or config.parameter_consistency_weight != 1.0
            or config.logit_weight != 0.0001
            or any(
                (
                    config.state_sparsity_weight,
                    config.action_sparsity_weight,
                    config.parameter_sparsity_weight,
                )
            )
        ):
            raise ValueError("Require original prediction1, SIGReg0.4, consistency1 configuration")
        self.projections = config.regularizer_num_projections
        self.regularizer = RepresentationRegularizer("sigreg", self.projections)

    def forward(self, output: AcquisitionOutput) -> tuple[Tensor, dict[str, Tensor]]:
        """Retain original SIGReg projection order and grouped consistency scaling."""
        if output.online_states.requires_grad:
            raise ValueError("Cannot omit state SIGReg when online states are trainable")
        prediction = (output.predictions - output.targets).square().mean()
        torch.randn(
            output.online_states.shape[-1],
            self.projections,
            device=output.online_states.device,
            dtype=output.online_states.dtype,
        )
        regularizer = self.regularizer(output.parameters.transpose(0, 1))
        consistency = 2 * output.parameters.var(dim=1, correction=1).mean()
        total = prediction + 0.4 * regularizer + consistency
        return total, {
            "prediction": prediction,
            "parameter_regularizer": regularizer,
            "parameter_consistency": consistency,
            "optimized_total": total,
            "frozen_state_sigreg_omitted": total.new_ones(()),
            "attention_logit_absent": total.new_ones(()),
        }


def training_loss(
    model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
) -> tuple[Tensor, dict[str, Tensor]]:
    """Accept no parameters, inverse maps or evaluation objects in optimization."""
    return objective(cached_forward(model, batch))
