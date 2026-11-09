"""Controlled SSL geometry changes that leave the graph-state channel unchanged."""

from typing import Literal

import torch
from torch import Tensor, nn

from scjepa.models.model import ModelOutput
from scjepa.training.lagrangian import GECOConfig
from scjepa.training.loop import ObjectiveConfig, TrainingObjective
from scjepa.training.rng import preserve_rng

StateRegularization = Literal[
    "aligned", "mixture", "mixture-variance", "projected-mixture-variance"
]
STATE_REGULARIZATIONS = ("aligned", "mixture", "mixture-variance", "projected-mixture-variance")


def mix_state_samples(states: Tensor) -> Tensor:
    """Shuffle view/time within each system without consuming prediction RNG draws.

    Input and output are (independent systems, view-times, state coordinates).
    Every regularizer row retains exactly one sample per independent system.
    The runner seeds the global generator per update; this separate stream is
    consequently reproducible across checkpoint resume and across conditions.
    """
    if states.ndim != 3 or min(states.shape) < 1:
        raise ValueError("expected nonempty states (B,RT,D)")
    generator = torch.Generator(device=states.device)
    generator.manual_seed((torch.initial_seed() + 104729) % (1 << 63))
    order = torch.rand(states.shape[:2], generator=generator, device=states.device).argsort(1)
    return states.gather(1, order[..., None].expand_as(states))


def state_variance_floor(states: Tensor) -> Tensor:
    """Anchor each raw coordinate's scale without penalizing cross-coordinate dependence.

    This is the variance-only hinge from VICReg, with unit target standard
    deviation and epsilon 1e-4. Statistics span independent systems, then average
    over view-times and coordinates. It is a loss, not batch normalization of E.
    """
    if states.ndim != 3 or states.shape[0] < 2:
        raise ValueError("variance floor requires states (B>=2,RT,D)")
    standard_deviation = (states.var(dim=0, correction=1) + 1e-4).sqrt()
    return (1 - standard_deviation).clamp_min(0).mean()


class StateRegularizationObjective(TrainingObjective):
    """Apply SIGReg to a time mixture, optionally through an auxiliary smooth head.

    The predictor, context encoder, autoregressive feedback, stopped EMA targets
    and prediction MSE always use the original two-dimensional state. The head
    serves only SIGReg and can be discarded at inference. It has two outputs:
    a smooth map of a two-dimensional state cannot fill a higher-dimensional
    isotropic Gaussian. A raw-state variance floor prevents an arbitrarily small
    state scale from being hidden by amplification in this auxiliary head.
    """

    def __init__(
        self,
        config: ObjectiveConfig,
        geco: GECOConfig | None,
        *,
        mode: str,
        seed: int = 0,
    ) -> None:
        """Keep optional-head initialization independent of scientific model weights."""
        if mode not in STATE_REGULARIZATIONS or mode == "aligned":
            raise ValueError("unknown mixture state regularization")
        super().__init__(config, geco)
        self.mode = mode
        self.projector: nn.Module = nn.Identity()
        if mode == "projected-mixture-variance":
            with preserve_rng(torch.device("cpu")):
                torch.default_generator.manual_seed(seed + 104729)
                self.projector = nn.Sequential(
                    nn.Linear(2, 128),
                    nn.GELU(),
                    nn.Linear(128, 128),
                    nn.GELU(),
                    nn.Linear(128, 2),
                )

    def forward(self, output: ModelOutput) -> tuple[Tensor, dict[str, Tensor]]:
        """Change regularization only; preserve raw prediction and all attached paths."""
        states = output.online_states
        if states.ndim == 4:
            states = states.flatten(1, 2)
        if states.shape[-1] != 2:
            raise ValueError("this physical-state control requires two state coordinates")
        mixed = mix_state_samples(states)
        projected: Tensor = self.projector(mixed)
        total, terms = super().forward(output._replace(online_states=projected))
        terms["state_std_min"] = mixed.var(0, correction=1).mean(0).sqrt().min().detach()
        if self.mode.endswith("-variance"):
            terms["state_variance_floor"] = state_variance_floor(mixed)
            penalty = self.config.state_regularizer_weight * terms["state_variance_floor"]
            total = total + penalty
            terms["base"] = terms["base"] + penalty
        return total, terms
