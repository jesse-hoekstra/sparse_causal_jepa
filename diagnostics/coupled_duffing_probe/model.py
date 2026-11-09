"""Smooth temporal feature extraction within one physical transition context."""

import math

import torch
from torch import Tensor, nn

from diagnostics.coupled_duffing.model import LaggedParameterEncoder
from diagnostics.masked_mlp_oracle.model import (
    _gaussian_,  # pyright: ignore[reportPrivateUsage]
)
from scjepa.training.rng import preserve_rng


class TemporalParameterEncoder(LaggedParameterEncoder):
    """Combine neighbouring transition tuples before producing one context's code."""

    def __init__(self, context: int, code_dim: int = 8) -> None:
        """Keep the MLP output head and downstream initialization RNG unchanged."""
        super().__init__(context, code_dim)
        # Isolate new weights so the predictor and existing head match the MLP control.
        with preserve_rng(torch.device("cpu")):
            temporal_seed = int(torch.randint(0, 2**63 - 1, ()).item())
            torch.default_generator.manual_seed(temporal_seed)
            self.transition = nn.Sequential(
                nn.Conv1d(8, 64, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(64, 64, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(64, 128, kernel_size=1),
                nn.GELU(),
            )
            for layer in self.transition:
                if isinstance(layer, nn.Conv1d):
                    fan_in = layer.in_channels * layer.kernel_size[0]
                    _gaussian_(layer.weight, fan_in)
                    if layer.bias is not None:
                        _gaussian_(layer.bias, fan_in)
                    with torch.no_grad():
                        layer.weight.mul_(math.sqrt(6))

    def features(self, states: Tensor, actions: Tensor | None = None) -> Tensor:
        """Encode ordered tuples independently for each episode, then pool over time."""
        if (
            states.ndim != 3
            or states.shape[-1] != 2
            or not 3 <= states.shape[1] <= self.max_history
        ):
            raise ValueError("expected states (B,K,2) with 3<=K<=context")
        if actions is None or actions.shape != (len(states), states.shape[1] - 1, 1):
            raise ValueError("expected context actions (B,K-1,1)")
        inputs = torch.cat(
            (states[:, :-2], states[:, 1:-1], actions[:, :-1], actions[:, 1:], states[:, 2:]),
            -1,
        )
        features = self.transition(inputs.transpose(1, 2)).mean(2)
        return self.head[:-1](features)
