"""Single-context parameter inference reused across independent query episodes.

This optional frozen-state diagnostic leaves production training unchanged.
Prediction pairs flatten into R*Q; the original B*R codes enter SIGReg once.
"""

from collections.abc import Mapping

import torch
from torch import Tensor

from scjepa.models.model import SCJEPA, ModelOutput, cross_episode_parameters
from scjepa.models.spartan import SpartanOutput


def query_shifts(queries: int, shift: int) -> tuple[int, ...]:
    """Return equally sampled, distinct non-self shifts for the R=10 protocol."""
    if queries not in (1, 3, 9):
        raise ValueError("This R=10 diagnostic permits Q=1,3,9")
    if not 1 <= shift <= 9:
        raise ValueError("The first cyclic shift must lie in 1..9")
    return tuple(1 + ((shift - 1 + offset) % 9) for offset in range(0, 9, 9 // queries))


def cached_multiquery_forward(
    model: SCJEPA,
    batch: Mapping[str, Tensor],
    queries: int = 3,
    shift: int | None = None,
) -> ModelOutput:
    """Infer each code once from K10 and predict Q other episodes for two steps.

    Training draws the same single global nonzero cyclic shift as production.
    Evaluation starts at shift one. An explicit shift consumes no pairing RNG.
    The code and first predicted state remain attached throughout the rollout.
    States, targets and actions must be cached tensors on the model's device.
    """
    if queries not in (1, 3, 9):
        raise ValueError("This R=10 diagnostic permits Q=1,3,9")
    if set(batch) != {"online", "target", "actions"}:
        raise ValueError("Training batches contain only frozen states, targets and actions")
    online, actions, target = batch["online"], batch["actions"], batch["target"]
    if online.ndim != 4 or online.shape[1:] != (10, 12, 6):
        raise ValueError("This diagnostic requires R=10, T=12 and six state coordinates")
    groups, views, _, state_dim = online.shape
    if target.shape != online.shape or actions.shape != (groups, views, 11, 16):
        raise ValueError("Expected matching frozen targets and actions (B,10,11,16)")
    if model.context_len != 10 or model.predictor.history_len != 1:
        raise ValueError("This diagnostic requires K=10 and one-state prediction history")
    if model.parameter_source != "learned" or not model.autoregressive:
        raise ValueError("This diagnostic requires learned parameters and autoregression")
    if model.predictor.param_dim != 6 or not all(layer.dense for layer in model.predictor.layers):
        raise ValueError("This diagnostic requires a dense predictor with six parameters")
    if online.requires_grad or target.requires_grad:
        raise ValueError("The cached state encoders must remain frozen")

    q = model.parameter_encoder(
        online[:, :, :10].flatten(0, 1), actions[:, :, :9].flatten(0, 1)
    ).reshape(groups, views, 6)
    if queries == 1 and shift is None:
        # Preserve the original single-query operations and random draw exactly.
        donor = cross_episode_parameters(q, training=model.training).flatten(0, 1)
        history = online[:, :, 9:10].flatten(0, 1)
        expanded_actions = actions.flatten(0, 1)
        targets = target[:, :, 10:12]
    else:
        if shift is None:
            shift = (
                int(torch.randint(1, views, (), device=q.device).item()) if model.training else 1
            )
        shifts = query_shifts(queries, shift)
        # At recipient r, shift s supplies q[(r-s) mod R]. Every source code
        # therefore predicts Q distinct other episodes from the same theta group.
        donor = torch.stack([q.roll(s, dims=1) for s in shifts], dim=2).flatten(0, 2)
        history = (
            online[:, :, 9:10]
            .unsqueeze(2)
            .expand(groups, views, queries, 1, state_dim)
            .flatten(0, 2)
        )
        expanded_actions = (
            actions.unsqueeze(2).expand(groups, views, queries, *actions.shape[2:]).flatten(0, 2)
        )
        targets = (
            target[:, :, 10:12]
            .unsqueeze(2)
            .expand(groups, views, queries, 2, state_dim)
            .reshape(groups, views * queries, 2, state_dim)
        )
    dynamics: list[SpartanOutput] = []
    for source in (9, 10):
        result = model.predictor(history, donor, expanded_actions[:, source : source + 1])
        dynamics.append(result)
        history = result.prediction[:, None]
    predictions = torch.stack([item.prediction for item in dynamics], dim=1).reshape(
        groups, views * queries, 2, state_dim
    )
    return ModelOutput(predictions, targets, online, q, tuple(dynamics), 9)


__all__ = ["cached_multiquery_forward", "query_shifts"]
