"""Reuse the native state-learning objective with 22 observed action coordinates."""

from typing import cast

import torch
from torch import Tensor, nn

from diagnostics.mlp_oracle_code_control.model import MLPPredictor, Prediction
from diagnostics.oracle_state_learning.core import Condition
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import Spartan
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.rng import preserve_rng

ACTION_DIMENSION = 22


class InnovationMLP(MLPPredictor):
    """Ordinary global MLP; inherited type identifies its graph-free objective."""

    action_dim = ACTION_DIMENSION

    def __init__(self, hidden_dim: int) -> None:
        """Initialize the width selected by nearest SPARTAN parameter count."""
        nn.Module.__init__(self)
        self.network = nn.Sequential(
            nn.Linear(12 + self.action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, states: Tensor, parameters: Tensor, actions: Tensor) -> Prediction:
        """Consume attached state histories, raw theta and all mixed actions."""
        batch = len(states)
        if states.shape != (batch, 1, 6) or parameters.shape != (batch, 6):
            raise ValueError("MLP requires (B,1,6) states and (B,6) parameters")
        if actions.shape != (batch, 1, self.action_dim):
            raise ValueError("MLP requires (B,1,22) observed actions")
        return Prediction(self.network(torch.cat((states[:, 0], actions[:, 0], parameters), -1)))


def build_control(condition: Condition, seed: int) -> SCJEPA:
    """Preserve paired fresh encoder/SPARTAN tensors across all four conditions."""
    if condition not in {"dense", "baseline", "full", "mlp"}:
        raise ValueError("Unknown harder oracle condition")
    config = ModelConfig(
        observation_dim=6,
        state_dim=6,
        param_dim=6,
        action_dim=ACTION_DIMENSION,
        context_len=10,
        history_len=1,
        encoder_hidden_dims=(64, 64),
        parameter_encoder_kind="transition",
        parameter_source="oracle",
        spartan_layers=2,
        spartan_embed_dim=64,
        spartan_mlp_hidden=64,
        spartan_mlp_layers=3,
        spartan_dense=condition == "dense",
        spartan_parameter_gates_open=condition == "baseline",
        ema_decay=0.996,
        autoregressive=True,
    )
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(seed)
        model = build_model(config)
        if condition == "mlp":
            count = sum(parameter.numel() for parameter in model.predictor.parameters())
            width = min(range(1, 1025), key=lambda h: abs(h * h + 42 * h + 6 - count))
            torch.default_generator.manual_seed(seed)
            model.predictor = cast(Spartan, InnovationMLP(width))
    return model
