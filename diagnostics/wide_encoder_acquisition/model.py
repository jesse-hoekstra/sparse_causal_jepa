"""Diagnostic width-only subclass; reuse the production transition pooling forward exactly."""

from typing import Any

import torch
from torch import nn

from diagnostics.multiquery_control.data import module_fingerprint
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.training.rng import preserve_rng


class WideTransitionParameterEncoder(TransitionParameterEncoder):
    """Double every hidden layer to256; preserve GELU, mean pooling and raw six-code semantics."""

    def __init__(
        self, state_dim: int, param_dim: int, action_dim: int = 0, max_history: int = 64
    ) -> None:
        """Construct only wide layers; avoid consuming RNG for discarded128-wide layers."""
        nn.Module.__init__(self)
        if min(state_dim, param_dim) < 1 or action_dim < 0 or max_history < 2:
            raise ValueError("positive dimensions and max_history>=2 are required")
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.action_dim = action_dim
        self.max_history = max_history
        self.transition = nn.Sequential(
            nn.Linear(2 * state_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, param_dim),
        )


def initialize_wide_encoder(model: SCJEPA, seed: int) -> dict[str, Any]:
    """Replace only E using an isolated CPU seed; preserve actual predictor/state tensors."""
    original = model.parameter_encoder
    if type(original) is not TransitionParameterEncoder:
        raise ValueError("Width control requires the original plain transition encoder")
    if (original.state_dim, original.param_dim, original.action_dim) != (6, 6, 16) or seed < 0:
        raise ValueError("Width control requires original6state/6code/16action dimensions")
    reference = next(original.parameters())
    old_hash = module_fingerprint(original)
    old_count = sum(parameter.numel() for parameter in original.parameters())
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(seed + 1)
        encoder = WideTransitionParameterEncoder(6, 6, 16, original.max_history).to(reference)
    count = sum(parameter.numel() for parameter in encoder.parameters())
    if old_count != 70534 or count != 272134:
        raise RuntimeError("Declared original or widened encoder parameter count changed")
    model.parameter_encoder = encoder
    return {
        "scientific_seed": seed,
        "encoder_initialization_seed": seed + 1,
        "hidden_width": 256,
        "parameter_count": count,
        "module_sha256": module_fingerprint(encoder),
        "replaced_encoder_sha256": old_hash,
        "replaced_encoder_parameter_count": old_count,
        "policy": (
            "Isolated CPU default_generator.manual_seed(scientific_seed+1), native Linear init of "
            "all wide E layers, then source dtype/device. No archived E tensors copied; "
            "P uses unchanged scientific_seed; states untouched. No discarded128-wide draws. "
            "Distinct fixed E seed avoids sharing the same random prefix with P."
        ),
        "architecture": (
            "transition28-256-256-256 GELU; mean over9transitions; head256-256-256-6 GELU"
        ),
        "forward": "Inherited production TransitionParameterEncoder.forward unchanged",
    }
