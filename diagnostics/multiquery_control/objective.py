"""Skip only constant frozen-state SIGReg while preserving its projection draw."""

import copy

import torch
from torch import Tensor, nn

from scjepa.models.model import ModelOutput
from scjepa.training.loop import ObjectiveConfig, TrainingObjective


class _ProjectionDrawOnly(nn.Module):
    """Consume upstream SIGReg's RNG draw on the selected tensor device."""

    def __init__(self, projections: int) -> None:
        super().__init__()
        self.projections = projections

    def forward(self, samples: Tensor) -> Tensor:
        if samples.requires_grad:
            raise ValueError("Cannot omit state SIGReg for trainable online states")
        if samples.ndim != 3 or samples.shape[1] < 2 or min(samples.shape[::2]) < 1:
            raise ValueError("Expected valid (views*time, batch>=2, state_dim) samples")
        torch.randn(samples.shape[-1], self.projections, device=samples.device, dtype=samples.dtype)
        return samples.new_zeros(())


class FrozenStateFastObjective(TrainingObjective):
    """Keep the full gradient and RNG, excluding one constant from training value.

    Only frozen online states and a non-GECO objective are supported. Reporting
    full objective values requires the ordinary TrainingObjective. The returned
    reduced value must never drive loss-dependent controllers or model selection.
    """

    def __init__(self, config: ObjectiveConfig) -> None:
        """Copy the configuration and replace only frozen-state SIGReg evaluation."""
        if config.state_regularizer != "sigreg" or config.state_regularizer_weight <= 0:
            raise ValueError("This optimization requires active frozen-state SIGReg")
        super().__init__(copy.deepcopy(config))
        self.state_regularizer = _ProjectionDrawOnly(config.regularizer_num_projections)

    def forward(self, output: ModelOutput) -> tuple[Tensor, dict[str, Tensor]]:
        """Return gradient-equivalent terms with the omitted constant identified."""
        if self.geco is not None:
            raise ValueError("A reduced objective cannot drive GECO")
        if output.online_states.requires_grad:
            raise ValueError("The online state encoder must remain frozen")
        total, terms = super().forward(output)
        terms["base_without_frozen_state_sigreg"] = terms.pop("base")
        terms.pop("state_regularizer")
        terms["frozen_state_sigreg_omitted_from_training_value"] = total.new_ones(())
        return total, terms


__all__ = ["FrozenStateFastObjective"]
