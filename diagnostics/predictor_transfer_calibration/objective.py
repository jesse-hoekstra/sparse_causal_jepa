"""Exact six-code factual forecast expectation for frozen dense SPARTAN or MLP."""

import copy
from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn

from diagnostics.mlp_oracle_code_control.model import MLPPredictor
from diagnostics.multiquery_control.data import TensorSplit
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.models.spartan import Spartan


def check_train(train: TensorSplit) -> None:
    """Keep all physical labels and evaluation objects outside calibration."""
    if set(train) != {"online", "target", "actions"}:
        raise ValueError("Calibration accepts only original label-free TRAIN inputs")
    groups = len(train["online"])
    if (
        train["online"].shape != (groups, 10, 12, 6)
        or train["target"].shape != train["online"].shape
        or train["actions"].shape != (groups, 10, 11, 16)
    ):
        raise ValueError("Require original R10/T12/six-state/sixteen-action cache")
    if any(value.requires_grad for value in train.values()):
        raise ValueError("Frozen cache must not require gradients")


@torch.no_grad()
def encode_train(
    model: SCJEPA,
    train: TensorSplit,
    chunk_size: int,
    progress: Callable[[int], None] | None = None,
) -> Tensor:
    """Cache individual TRAIN context codes once; do not average across contexts."""
    check_train(train)
    if chunk_size < 1:
        raise ValueError("Positive chunk size required")
    reference = next(model.parameters())
    parts: list[Tensor] = []
    for start in range(0, len(train["online"]), chunk_size):
        states, actions = (
            train[key][start : start + chunk_size].to(reference) for key in ("online", "actions")
        )
        parts.append(
            model.parameter_encoder(
                states[:, :, :10].flatten(0, 1), actions[:, :, :9].flatten(0, 1)
            )
            .reshape(-1, 10, 6)
            .detach()
            .cpu()
        )
        if progress:
            progress(min(start + chunk_size, len(train["online"])))
    return torch.cat(parts)


def prediction_loss(
    model: SCJEPA, train: TensorSplit, codes: Tensor, translation: Tensor
) -> Tensor:
    """Mean over all nine nonself shifts and two attached autoregressive horizons."""
    check_train(train)
    predictor = cast(nn.Module, model.predictor)
    if model.predictor.param_dim != 6 or not isinstance(predictor, MLPPredictor | Spartan):
        raise TypeError("Require frozen six-code MLP or SPARTAN")
    if isinstance(predictor, Spartan) and not all(
        layer.dense and not layer.identity for layer in predictor.layers
    ):
        raise ValueError("Calibration requires original all-open dense SPARTAN")
    groups = len(codes)
    if codes.shape != (groups, 10, 6) or translation.shape != (6,) or codes.requires_grad:
        raise ValueError("Require detached individual codes and one global six-vector")
    donor = torch.stack(
        [(codes + translation).roll(shift, 1) for shift in range(1, 10)], 2
    ).flatten(0, 2)
    history = train["online"][:, :, 9:10, :].unsqueeze(2).expand(-1, -1, 9, -1, -1).flatten(0, 2)
    actions = train["actions"][:, :, 9:11, :].unsqueeze(2).expand(-1, -1, 9, -1, -1).flatten(0, 2)
    predictions: list[Tensor] = []
    for step in range(2):
        predicted = model.predictor(history, donor, actions[:, step : step + 1]).prediction
        predictions.append(predicted)
        history = predicted[:, None]
    predicted = torch.stack(predictions, 1).reshape(groups, 10, 9, 2, 6)
    target = train["target"][:, :, 10:12].unsqueeze(2)
    return (predicted - target).square().mean()


class TrainObjective:
    """Accumulate exact full-TRAIN scalar/gradient without retaining a full-dataset graph."""

    def __init__(self, model: SCJEPA, train: TensorSplit, codes: Tensor, chunk_size: int) -> None:
        """Accept no held-out data or labels and require completely frozen source modules."""
        check_train(train)
        if (
            model.training
            or any(parameter.requires_grad for parameter in model.parameters())
            or next(model.parameters()).dtype != torch.float64
        ):
            raise ValueError("Require eval-mode, frozen float64 source")
        if codes.shape != (len(train["online"]), 10, 6) or codes.requires_grad or chunk_size < 1:
            raise ValueError("Invalid detached TRAIN-code cache")
        self.model, self.train, self.codes, self.chunk_size = model, train, codes, chunk_size

    def __call__(self, translation: Tensor) -> tuple[float, Tensor]:
        """Every call covers every group and nonself donor with identical normalization."""
        if translation.dtype != torch.float64 or not translation.requires_grad:
            raise ValueError("Require differentiable float64 translation")
        groups = len(self.codes)
        total = translation.new_zeros(())
        gradient = torch.zeros_like(translation)
        with torch.enable_grad():
            for start in range(0, groups, self.chunk_size):
                end = min(start + self.chunk_size, groups)
                batch = {key: value[start:end].to(translation) for key, value in self.train.items()}
                codes = self.codes[start:end].to(translation)
                loss = prediction_loss(self.model, batch, codes, translation) * (
                    (end - start) / groups
                )
                derivative = torch.autograd.grad(loss, translation)[0]
                total += loss.detach()
                gradient += derivative.detach()
        if not bool(torch.isfinite(total)) or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Nonfinite factual calibration objective/gradient")
        return float(total), gradient


@torch.no_grad()
def compose_translation(model: SCJEPA, translation: Tensor) -> SCJEPA:
    """Fold the single invertible translation into a separate encoder bias copy."""
    result = copy.deepcopy(model)
    if not isinstance(
        result.parameter_encoder, TransitionParameterEncoder
    ) or translation.shape != (6,):
        raise ValueError("Require original six-output transition encoder")
    output = cast(nn.Linear, result.parameter_encoder.head[4])
    output.bias.add_(translation.to(output.bias))
    return result.requires_grad_(False).eval()
