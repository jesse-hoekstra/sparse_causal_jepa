"""Train-group latent controls with attached rollouts and sparse global clipping."""

import math
from collections.abc import Iterable, Mapping

import torch
from torch import Tensor, nn

from scjepa.models.model import SCJEPA, ModelOutput, cross_episode_parameters
from scjepa.models.spartan import SpartanOutput


def group_teacher_forward(model: SCJEPA, batch: Mapping[str, Tensor], codes: Tensor) -> ModelOutput:
    """Use one learned training-group code for every episode's attached two-step rollout.

    The caller alone indexes the train-only table. This function accepts neither
    parameter labels nor group IDs. Repeating codes across R preserves the
    original per-view SIGReg objective and gives zero within-group consistency.
    The ordinary pairing draw is retained although every donor shares one code.
    """
    if set(batch) != {"online", "target", "actions"}:
        raise ValueError("Teacher batches contain only frozen states, targets and actions")
    online, target, actions = batch["online"], batch["target"], batch["actions"]
    if online.ndim != 4 or online.shape[1:] != (10, 12, 6):
        raise ValueError("Teacher requires R10/T12 and six state coordinates")
    groups = online.shape[0]
    if target.shape != online.shape or actions.shape != (groups, 10, 11, 16):
        raise ValueError("Teacher cache shapes must match the original grouped protocol")
    if codes.shape != (groups, 6) or codes.device != online.device or codes.dtype != online.dtype:
        raise ValueError("Teacher codes must have shape (groups, 6), cache device and dtype")
    if any(value.requires_grad for value in (online, target, actions)):
        raise ValueError("Teacher cached inputs must remain frozen")
    if any(
        value.device != online.device or value.dtype != online.dtype for value in (target, actions)
    ):
        raise ValueError("Teacher cached inputs must share device and dtype")
    if (
        model.context_len != 10
        or not model.autoregressive
        or model.parameter_source != "learned"
        or model.predictor.history_len != 1
        or (model.predictor.state_dim, model.predictor.param_dim, model.predictor.action_dim)
        != (6, 6, 16)
        or not all(layer.dense for layer in model.predictor.layers)
    ):
        raise ValueError("Teacher requires the original dense learned-code K10 two-step model")
    grouped_codes = codes[:, None, :].expand(-1, 10, -1)
    donor = cross_episode_parameters(grouped_codes, training=model.training).flatten(0, 1)
    history = online[:, :, 9:10].flatten(0, 1)
    flat_actions = actions.flatten(0, 1)
    dynamics: list[SpartanOutput] = []
    for source in (9, 10):
        result = model.predictor(history, donor, flat_actions[:, source : source + 1])
        dynamics.append(result)
        history = result.prediction[:, None]
    predictions = torch.stack([row.prediction for row in dynamics], dim=1).reshape(groups, 10, 2, 6)
    return ModelOutput(predictions, target[:, :, 10:12], online, grouped_codes, tuple(dynamics), 9)


def clip_teacher_gradients(
    predictor_parameters: Iterable[nn.Parameter], embedding: nn.Embedding, max_norm: float = 1.0
) -> float:
    """Apply one norm budget to predictor gradients and coalesced sparse table rows.

    Neither table weights nor optimizer moments are touched here. SparseAdam
    subsequently updates only the sampled rows; callers step both optimizers.
    """
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("Teacher gradient bound must be finite and positive")
    if not embedding.sparse or embedding.weight.grad is None:
        raise ValueError("Teacher embedding must have a sparse gradient")
    sparse = embedding.weight.grad
    if not sparse.is_sparse:
        raise ValueError("Teacher embedding requires sparse COO gradients")
    sparse = sparse.coalesce()
    embedding.weight.grad = sparse
    gradients: list[Tensor] = [sparse.values()]
    seen: set[int] = set()
    for parameter in predictor_parameters:
        if id(parameter) in seen or parameter is embedding.weight:
            raise ValueError("Predictor parameters must be distinct from the embedding")
        seen.add(id(parameter))
        if parameter.grad is not None:
            if parameter.grad.is_sparse:
                raise ValueError("Predictor gradients must be dense")
            gradients.append(parameter.grad)
    norm = math.sqrt(
        sum(float(gradient.detach().double().square().sum()) for gradient in gradients)
    )
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite teacher gradient norm")
    coefficient = min(1.0, max_norm / (norm + 1e-6))
    with torch.no_grad():
        for gradient in gradients:
            gradient.mul_(coefficient)
    return norm
