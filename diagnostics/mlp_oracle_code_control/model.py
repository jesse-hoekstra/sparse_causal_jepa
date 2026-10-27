"""A prediction-only adapter for the existing exact-code evaluation protocol."""

from typing import NamedTuple, cast

import torch
from torch import Tensor, nn

from diagnostics.multiquery_control.data import TensorSplit, module_fingerprint
from diagnostics.oracle_code_control.evaluation import supplied_codes
from scjepa.models.model import SCJEPA, cross_episode_parameters
from scjepa.models.spartan import Spartan
from scjepa.training.rng import preserve_rng


class Prediction(NamedTuple):
    """Only predicted states are meaningful; this model has no attention graphs."""

    prediction: Tensor


class MLPPredictor(nn.Module):
    """Unnormalized 28→211→211→6 ReLU network with 52,123 parameters."""

    history_len = 1
    state_dim = 6
    action_dim = 16
    param_dim = 6

    def __init__(self) -> None:
        """Initialize ordinary biased linear layers using PyTorch defaults."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(28, 211), nn.ReLU(), nn.Linear(211, 211), nn.ReLU(), nn.Linear(211, 6)
        )

    def forward(self, states: Tensor, parameters: Tensor, actions: Tensor) -> Prediction:
        """Consume the same state/action history and complete supplied code as SPARTAN."""
        batch = len(states)
        if states.shape != (batch, 1, 6) or parameters.shape != (batch, 6):
            raise ValueError("MLP requires (B,1,6) states and (B,6) parameter codes")
        if actions.shape != (batch, 1, 16):
            raise ValueError("MLP requires (B,1,16) observed actions")
        return Prediction(self.network(torch.cat((states[:, 0], actions[:, 0], parameters), -1)))


def initialize_mlp(model: SCJEPA, seed: int) -> dict[str, str | int]:
    """Replace only P with an independently seeded fresh MLP, preserving all encoders.

    The cast is confined to the adapter boundary: shared oracle evaluators access
    predictor dimension attributes and `.prediction` only. This is not a Spartan
    and must never be passed to graph extraction or the production objective.
    """
    reference = next(model.parameters())
    with preserve_rng(torch.device("cpu")):
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
        predictor = MLPPredictor().to(reference)
    count = sum(parameter.numel() for parameter in predictor.parameters())
    if count != 52123:
        raise RuntimeError("Declared MLP architecture changed")
    model.predictor = cast(Spartan, predictor)
    return {
        "seed": seed,
        "parameter_count": count,
        "module_sha256": module_fingerprint(predictor),
        "policy": "CPU manual_seed(seed), native Linear initialization, then source dtype/device",
    }


def training_loss(
    model: SCJEPA, batch: TensorSplit, theta: Tensor
) -> tuple[Tensor, dict[str, Tensor]]:
    """Optimize attached two-step MSE with exact codes and no attention regularizer."""
    if not isinstance(model.predictor, MLPPredictor):
        raise TypeError("This diagnostic trains only its dedicated MLP predictor")
    if set(batch) != {"online", "target", "actions"}:
        raise ValueError("Keep physical labels separate from cached state/action inputs")
    online, target, actions = batch["online"], batch["target"], batch["actions"]
    groups = len(online)
    if online.shape != (groups, 10, 12, 6) or target.shape != online.shape:
        raise ValueError("Expected original R10/T12 six-state cache")
    if actions.shape != (groups, 10, 11, 16):
        raise ValueError("Expected original R10/T12 sixteen-action cache")
    if any(value.requires_grad for value in batch.values()):
        raise ValueError("Cached inputs must remain frozen")
    codes = supplied_codes(theta)[:, None].expand(-1, 10, -1)
    codes = cross_episode_parameters(codes, training=model.training).flatten(0, 1)
    history = online[:, :, 9:10].flatten(0, 1)
    predictions: list[Tensor] = []
    for step in (9, 10):
        prediction = model.predictor(history, codes, actions[:, :, step : step + 1].flatten(0, 1))
        predictions.append(prediction.prediction)
        history = prediction.prediction[:, None]
    predicted = torch.stack(predictions, 1).reshape(groups, 10, 2, 6)
    loss = (predicted - target[:, :, 10:12]).square().mean()
    return loss, {"prediction": loss, "optimized_total": loss}
