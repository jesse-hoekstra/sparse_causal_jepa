"""Scale predictive supervision and inspect its path to the parameter encoder."""

import math
from itertools import combinations

import torch
from torch import Tensor

from diagnostics.multiquery_control.objective import FrozenStateFastObjective
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.loop import ObjectiveConfig


class PredictiveSupervisionObjective(FrozenStateFastObjective):
    """Change only the predictive coefficient in the frozen-state dense objective."""

    def __init__(self, config: ObjectiveConfig, prediction_weight: float) -> None:
        """Require the fixed SIGReg/consistency control and a positive finite weight."""
        if not math.isfinite(prediction_weight) or prediction_weight <= 0:
            raise ValueError("prediction_weight must be finite and positive")
        if (
            config.parameter_regularizer != "sigreg"
            or config.parameter_regularizer_weight != 0.4
            or config.state_regularizer_weight != 0.4
            or config.parameter_consistency_weight != 1.0
            or any(
                (
                    config.state_sparsity_weight,
                    config.action_sparsity_weight,
                    config.parameter_sparsity_weight,
                )
            )
        ):
            raise ValueError("Use dense SIGReg 0.4 and consistency 1 for this control")
        super().__init__(config)
        self.prediction_weight = prediction_weight

    def forward(self, output: ModelOutput) -> tuple[Tensor, dict[str, Tensor]]:
        """Retain raw MSE for comparison and preserve the original projection RNG."""
        total, terms = super().forward(output)
        # Preserve the baseline's arithmetic exactly when alpha=1.
        if self.prediction_weight != 1.0:
            total = total + (self.prediction_weight - 1.0) * terms["prediction"]
        terms["weighted_prediction"] = self.prediction_weight * terms["prediction"]
        terms["base_without_frozen_state_sigreg"] = total
        terms["total_without_frozen_state_sigreg"] = total
        return total, terms


def gradient_diagnostics(
    model: SCJEPA,
    output: ModelOutput,
    terms: dict[str, Tensor],
    objective_config: ObjectiveConfig,
    prediction_weight: float,
) -> dict[str, float]:
    """Measure weighted component gradients on this training forward, without mutation.

    These are Euclidean gradients before clipping and AdamW. They neither measure
    factor-specific information nor predict effective optimizer update magnitudes.
    No labels, extra stochastic forwards, RNG draws, or optimizer steps are used.
    The graph is retained for the ordinary backward call. A cosine involving a
    zero gradient is reported as zero with a separate ``cosine_defined`` flag.
    """
    if not output.parameters.requires_grad:
        raise ValueError("Gradient diagnostics require attached trainable parameter codes")
    blocks: dict[str, tuple[Tensor, ...]] = {
        "code": (output.parameters,),
        "encoder": tuple(p for p in model.parameter_encoder.parameters() if p.requires_grad),
        "predictor": tuple(p for p in model.predictor.parameters() if p.requires_grad),
    }
    if not all(blocks.values()):
        raise ValueError("Both the parameter encoder and predictor must be trainable")
    inputs = tuple(tensor for block in blocks.values() for tensor in block)
    losses = {
        "prediction": prediction_weight * terms["prediction"],
        "sigreg": objective_config.parameter_regularizer_weight * terms["parameter_regularizer"],
        "consistency": objective_config.parameter_consistency_weight
        * terms["parameter_consistency"],
        "logit": objective_config.logit_weight * terms["logit"],
    }
    gradients: dict[str, dict[str, Tensor]] = {name: {} for name in blocks}
    for component, loss in losses.items():
        values = (
            torch.autograd.grad(loss, inputs, retain_graph=True, allow_unused=True)
            if loss.requires_grad
            else (None,) * len(inputs)
        )
        offset = 0
        for block, tensors in blocks.items():
            pieces: list[Tensor] = []
            for tensor, value in zip(tensors, values[offset : offset + len(tensors)], strict=True):
                pieces.append(
                    torch.zeros_like(tensor).flatten()
                    if value is None
                    else value.detach().flatten()
                )
            gradients[block][component] = torch.cat(pieces).double()
            offset += len(tensors)
    metrics: dict[str, float] = {}
    for block, components in gradients.items():
        for component, value in components.items():
            metrics[f"{block}/{component}_norm"] = float(value.square().sum().sqrt())
        regularizer = components["sigreg"] + components["consistency"] + components["logit"]
        metrics[f"{block}/auxiliary_norm"] = float(regularizer.square().sum().sqrt())
        metrics[f"{block}/total_norm"] = float(
            (components["prediction"] + regularizer).square().sum().sqrt()
        )
        for first, second in combinations(components, 2):
            dot = float(torch.dot(components[first], components[second]))
            denominator = metrics[f"{block}/{first}_norm"] * metrics[f"{block}/{second}_norm"]
            key = f"{block}/{first}_{second}"
            metrics[key + "_dot"] = dot
            metrics[key + "_cosine"] = dot / denominator if denominator > 0 else 0.0
            metrics[key + "_cosine_defined"] = float(denominator > 0)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Nonfinite gradient diagnostic")
    return metrics
