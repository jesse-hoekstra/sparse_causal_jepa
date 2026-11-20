"""Framewise CNN state encoder, lagged context-code encoder and masked two-lag predictor.

Conditions follow Experiment 1: dense opens every gate; baseline learns and
penalizes state and action gates (parameter gates fixed open); full additionally
learns and penalizes parameter gates. State/action gates are shared by both lags
and are lag-aggregated, matching the theory's graph definition.
"""

import math
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from diagnostics.masked_mlp_oracle.model import (
    MaskedTrunkPredictor,
    _gaussian_,  # pyright: ignore[reportPrivateUsage]
)
from diagnostics.vision_dimer import dgp
from scjepa.models.encoders import FramewiseEncoder
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import GraphBlocks, GraphPenalties, Spartan, SpartanOutput
from scjepa.training.rng import preserve_rng

type Phase = Literal["dense", "baseline", "full"]
HIDDEN = 209


class ConvStateEncoder(FramewiseEncoder):
    """Strided convolutions with per-frame GroupNorm, flattened into an MLP head.

    Flattening keeps spatial layout, so positions remain recoverable; no
    coordinate channels or keypoint heads are added (they would bias the chart
    toward the physical axes and confound the sparsity comparison).
    """

    observation_ndim = 3

    def __init__(
        self, shape: tuple[int, int, int] = (3, 64, 64), channels: tuple[int, ...] = (32, 64, 64)
    ) -> None:
        """Build the backbone for ``shape = (C, H, W)``."""
        super().__init__()
        self.state_dim = dgp.STATE_DIM
        self.shape = shape
        layers: list[nn.Module] = []
        previous = shape[0]
        for width in channels:
            layers += [
                nn.Conv2d(previous, width, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(1, width),
                nn.GELU(),
            ]
            previous = width
        self.backbone = nn.Sequential(*layers, nn.Flatten(1))
        scale = 2 ** len(channels)
        features = previous * (shape[1] // scale) * (shape[2] // scale)
        self.mlp = nn.Sequential(nn.Linear(features, 256), nn.GELU())
        self.head = nn.Linear(256, self.state_dim)

    def forward(self, observations: Tensor) -> Tensor:
        """Encode ``(B, C, H, W)`` frames independently."""
        if observations.ndim != 4 or tuple(observations.shape[1:]) != self.shape:
            raise ValueError(f"expected frames (B, {self.shape})")
        return self.head(self.mlp(self.backbone(observations)))


class LaggedCodeEncoder(nn.Module):
    """Pool MLP features of ``(z^{t-1}, z^t, a^{t-1}, a^t, z^{t+1})`` over one context."""

    def __init__(self, context: int, code_dim: int) -> None:
        """Use Experiment 1's 128-wide GELU transition/head MLP with its weight gains."""
        super().__init__()
        if context < 3 or code_dim < 1:
            raise ValueError("need context >= 3 and a positive code dimension")
        self.state_dim, self.action_dim = dgp.STATE_DIM, dgp.ACTION_DIM
        self.param_dim, self.max_history = code_dim, context
        width = 3 * self.state_dim + 2 * self.action_dim
        self.transition = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
        )  # fmt: skip
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, code_dim),
        )  # fmt: skip
        linears = [m for m in self.modules() if isinstance(m, nn.Linear)]
        with torch.no_grad():
            for layer in linears:
                _gaussian_(layer.weight, layer.in_features)
                _gaussian_(layer.bias, layer.in_features)
            for layer in linears[:-1]:
                layer.weight.mul_(math.sqrt(6))
            linears[-1].weight.mul_(math.sqrt(3))

    def tuples(self, states: Tensor, actions: Tensor) -> Tensor:
        """Ordered second-order transition tuples ``(B, K-2, 3 d_z + 2 d_a)``."""
        if states.ndim != 3 or states.shape[-1] != self.state_dim or states.shape[1] < 3:
            raise ValueError(f"expected context states (B, K>=3, {self.state_dim})")
        if actions.shape != (len(states), states.shape[1] - 1, self.action_dim):
            raise ValueError(f"expected context actions (B, K-1, {self.action_dim})")
        return torch.cat(
            (states[:, :-2], states[:, 1:-1], actions[:, :-1], actions[:, 1:], states[:, 2:]), -1
        )

    def features(self, states: Tensor, actions: Tensor) -> Tensor:
        """Penultimate pooled features of one context."""
        return self.head[:-1](self.transition(self.tuples(states, actions)).mean(1))

    def forward(self, states: Tensor, actions: Tensor | None = None) -> Tensor:
        """Return the complete context code ``(B, code_dim)``."""
        if actions is None:
            raise ValueError("the code encoder requires context actions")
        return self.head[-1](self.features(states, actions))


class LaggedMaskedPredictor(MaskedTrunkPredictor):
    """Shared GELU trunk with output-specific gates shared across both input lags."""

    history_len = 2

    def __init__(self, *, phase: Phase, code_dim: int, initial_logit: float = 3.0) -> None:
        """Gate learning by condition; logits start open and consume no RNG draws."""
        sparse = phase != "dense"
        super().__init__(
            state_dim=dgp.STATE_DIM,
            action_dim=dgp.ACTION_DIM,
            param_dim=code_dim,
            hidden_dim=HIDDEN,
            learn_state=sparse,
            learn_action=sparse,
            learn_parameter=phase == "full",
            initial_logit=initial_logit,
            activation="gelu",
        )
        self.first = nn.Linear(2 * dgp.STATE_DIM + 2 * dgp.ACTION_DIM + code_dim, HIDDEN)
        _gaussian_(self.first.weight, self.first.in_features)
        _gaussian_(self.first.bias, self.first.in_features)

    def forward(
        self, states: Tensor, parameters: Tensor, actions: Tensor | None = None
    ) -> SpartanOutput:
        """Predict ``z^{t+1}`` from ``(z^{t-1}, z^t)``, ``(a^{t-1}, a^t)`` and the code."""
        batch = len(states)
        if states.shape != (batch, 2, self.state_dim):
            raise ValueError("expected two state lags")
        if parameters.shape != (batch, self.param_dim):
            raise ValueError("unexpected code shape")
        if actions is None or actions.shape != (batch, 2, self.action_dim):
            raise ValueError("expected two action lags")
        sg, so = self._block(self.state_logits, self.state_dim, states)
        ag, ao = self._block(self.action_logits, self.action_dim, states)
        pg, po = self._block(self.parameter_logits, self.param_dim, states)
        state_gates = sg[:, :, None].expand(-1, -1, 2, -1)
        action_gates = ag[:, :, None].expand(-1, -1, 2, -1)
        inputs = torch.cat(
            (
                (states[:, None] * state_gates).flatten(2),
                (actions[:, None] * action_gates).flatten(2),
                parameters[:, None] * pg,
            ),
            -1,
        )
        features = F.gelu(self.second(F.gelu(self.first(inputs))))
        prediction = (features * self.readout_weight).sum(-1) + self.readout_bias
        tokens = 2 * self.state_dim + 2 * self.action_dim + self.param_dim
        paths = torch.eye(tokens, device=states.device, dtype=states.dtype).repeat(batch, 1, 1)
        paths[:, : self.state_dim] = torch.cat(
            (state_gates.flatten(2), action_gates.flatten(2), pg), -1
        )
        logits = list(self.learned_logits().values())
        zero = states.new_zeros(())
        flat = torch.cat([v.flatten() for v in logits]).detach() if logits else zero[None]
        return SpartanOutput(
            prediction,
            paths,
            GraphBlocks(state_gates, action_gates, pg),
            GraphPenalties(so, ao, po),
            zero,
            flat.abs().mean(),
            torch.sigmoid(flat).mean(),
            (F.softplus(flat) - flat * torch.sigmoid(flat)).mean(),
        )


class CachedEncoder(FramewiseEncoder):
    """Pass precomputed frozen encodings through the unchanged rollout implementation."""

    observation_ndim = 1

    def __init__(self) -> None:
        """Identity on ``(B, d_z)`` cached states."""
        super().__init__()
        self.state_dim = dgp.STATE_DIM

    def forward(self, observations: Tensor) -> Tensor:
        """Return cached states unchanged."""
        if observations.ndim != 2 or observations.shape[-1] != self.state_dim:
            raise ValueError("cached states have the wrong shape")
        return observations


def assemble(
    encoder: FramewiseEncoder, code_encoder: LaggedCodeEncoder, predictor: LaggedMaskedPredictor,
    context: int,
) -> SCJEPA:  # fmt: skip
    """Wrap modules in the shared SCJEPA rollout (attached donors, autoregressive steps)."""
    return SCJEPA(
        encoder,
        cast(object, code_encoder),  # pyright: ignore[reportArgumentType]
        cast(Spartan, predictor),
        context,
        ema_decay=0.996,
        autoregressive=True,
    )


def build_model(
    seed: int, *, phase: Phase, code_dim: int, context: int,
    shape: tuple[int, int, int] = (3, 64, 64), initial_logit: float = 3.0,
) -> SCJEPA:  # fmt: skip
    """Seeded model; predictor weights use identical RNG draws in every condition."""
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(seed)
        encoder = ConvStateEncoder(shape)
        code_encoder = LaggedCodeEncoder(context, code_dim)
        predictor = LaggedMaskedPredictor(
            phase=phase, code_dim=code_dim, initial_logit=initial_logit
        )
        return assemble(encoder, code_encoder, predictor, context)


def fresh_code_modules(
    seed: int, code_dim: int, context: int
) -> tuple[LaggedCodeEncoder, LaggedMaskedPredictor]:
    """Freshly seeded dense code encoder and predictor (acquisition restarts)."""
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(seed)
        return LaggedCodeEncoder(context, code_dim), LaggedMaskedPredictor(
            phase="dense", code_dim=code_dim
        )


__all__ = [
    "CachedEncoder",
    "ConvStateEncoder",
    "LaggedCodeEncoder",
    "LaggedMaskedPredictor",
    "Phase",
    "assemble",
    "build_model",
    "fresh_code_modules",
]
