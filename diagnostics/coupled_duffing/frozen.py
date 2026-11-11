"""Replay fixed image encodings while retaining the ordinary learned-code rollout."""

from typing import cast

import torch
from torch import Tensor

from scjepa.models.encoders import FramewiseEncoder
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.loop import TrainingObjective


class _CachedEncoder(FramewiseEncoder):
    """Pass already-computed learned states to the unchanged rollout implementation."""

    observation_ndim = 1

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim

    def forward(self, observations: Tensor) -> Tensor:
        """Preserve each cached state independently."""
        if observations.ndim != 2 or observations.shape[-1] != self.state_dim:
            raise ValueError("cached states have the wrong shape")
        return observations


class FrozenStateReplay:
    """Cache only frozen encoders; recompute context codes and attached forecasts each step.

    The original model owns every trained parameter and remains the checkpointed
    model. The replay shares its parameter encoder and predictor. Online and EMA
    encodings are cached separately, so synchronization is not assumed. This
    optimization is valid only while both image encoders remain fixed.
    """

    @torch.no_grad()
    def __init__(self, model: SCJEPA, inputs: dict[str, Tensor], batch_size: int) -> None:
        """Encode observations/actions only, keeping compact tensors on the model device."""
        if set(inputs) != {"observations", "actions"}:
            raise ValueError("frozen replay accepts only observations and actions")
        if batch_size < 1 or inputs["observations"].ndim != 6:
            raise ValueError("frozen replay requires grouped images and a positive batch size")
        if model.parameter_source != "learned" or any(
            p.requires_grad
            for encoder in (model.online_encoder, model.target_encoder)
            for p in encoder.parameters()
        ):
            raise ValueError("frozen replay requires fixed state encoders and learned parameters")
        self.original = model
        device = next(model.parameters()).device
        model.online_encoder.eval()
        model.target_encoder.eval()
        online: list[Tensor] = []
        targets: list[Tensor] = []
        first_target = max(model.context_len, model.predictor.history_len)
        dimension = model.online_encoder.state_dim
        for start in range(0, len(inputs["observations"]), batch_size):
            x = inputs["observations"][start : start + batch_size].to(device)
            groups, views, length = x.shape[:3]
            flat = x.flatten(0, 1)
            online.append(
                model.online_encoder.encode_sequence(flat).reshape(groups, views, length, dimension)
            )
            targets.append(
                model.target_encoder.encode_sequence(flat[:, first_target:]).reshape(
                    groups, views, length - first_target, dimension
                )
            )
        self.inputs = {
            "online_states": torch.cat(online),
            "target_states": torch.cat(targets),
            "actions": inputs["actions"].to(device),
        }
        self.replay = SCJEPA(
            _CachedEncoder(dimension),
            model.parameter_encoder,
            model.predictor,
            model.context_len,
            ema_decay=model.ema_decay,
            autoregressive=model.autoregressive,
        )

    def forward_loss(
        self, batch: dict[str, Tensor], objective: TrainingObjective
    ) -> tuple[Tensor, dict[str, Tensor], ModelOutput]:
        """Use the same donor RNG, parameter gradients, two-step feedback and loss."""
        if set(batch) != {"online_states", "target_states", "actions"}:
            raise ValueError("unexpected frozen replay inputs")
        self.replay.train(self.original.training)
        result = cast(ModelOutput, self.replay(batch["online_states"], batch["actions"]))
        result = result._replace(targets=batch["target_states"])
        loss, terms = objective.forward(result)
        return loss, terms, result
