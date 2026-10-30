"""Original construction order retained for exact masked-MLP initialization.

The temporary native predictor preserves historical parameter counting and RNG
consumption. Both experiment builders replace it with MaskedTrunkPredictor.
It is not a training or evaluation condition in this repository.
"""

import torch

from diagnostics.shared.oracle import Condition
from scjepa.models.model import SCJEPA
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.rng import preserve_rng

ACTION_DIMENSION = 24


def build_control(condition: Condition, seed: int) -> SCJEPA:
    """Retain paired fresh encoder/SPARTAN tensors across the three conditions."""
    if condition not in {"dense", "baseline", "full"}:
        raise ValueError("Unknown isotropic oracle condition")
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
        return build_model(config)
