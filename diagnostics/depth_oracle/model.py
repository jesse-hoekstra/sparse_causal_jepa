"""Add one native communication layer while preserving every compatible initial tensor."""

import hashlib
from pathlib import Path
from typing import cast

import torch

from diagnostics.isotropic_oracle.model import IsotropicMLP
from diagnostics.oracle_state_learning.core import Condition
from diagnostics.rotating_pair_oracle.data import code_identity as parent_code_identity
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import Spartan
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.rng import preserve_rng

NATIVE_LAYERS = 3
NATIVE_WIDTH = 64
NATIVE_PREDICTOR_PARAMETERS = 77_697
MLP_HIDDEN_DIM = 258
MLP_PREDICTOR_PARAMETERS = 77_922


def code_identity() -> str:
    """Bind the frozen rotating-pair parent and only these new numerical training files."""
    digest = hashlib.sha256(parent_code_identity().encode())
    for name in ("__init__.py", "model.py", "run.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def build_control(condition: Condition, seed: int) -> SCJEPA:
    """Use the native factory at fixed depth three; initialize all arms freshly."""
    if condition not in {"dense", "baseline", "full", "mlp"}:
        raise ValueError("Unknown depth-control condition")
    config = ModelConfig(
        observation_dim=6,
        state_dim=6,
        param_dim=6,
        action_dim=24,
        context_len=10,
        history_len=1,
        encoder_hidden_dims=(64, 64),
        parameter_encoder_kind="transition",
        parameter_source="oracle",
        spartan_layers=NATIVE_LAYERS,
        spartan_embed_dim=NATIVE_WIDTH,
        spartan_mlp_hidden=NATIVE_WIDTH,
        spartan_mlp_layers=3,
        spartan_dense=condition == "dense",
        spartan_parameter_gates_open=condition == "baseline",
        ema_decay=0.996,
        autoregressive=True,
    )
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(seed)
        model = build_model(config)
        count = sum(parameter.numel() for parameter in model.predictor.parameters())
        if count != NATIVE_PREDICTOR_PARAMETERS:
            raise RuntimeError("Native depth-control parameter count changed")
        if condition == "mlp":
            width = min(range(1, 1025), key=lambda h: abs(h * h + 44 * h + 6 - count))
            if width != MLP_HIDDEN_DIM:
                raise RuntimeError("Nearest parameter-count-matched MLP width changed")
            torch.default_generator.manual_seed(seed)
            model.predictor = cast(Spartan, IsotropicMLP(width))
            if sum(p.numel() for p in model.predictor.parameters()) != MLP_PREDICTOR_PARAMETERS:
                raise RuntimeError("Depth-control MLP parameter count changed")
    return model
