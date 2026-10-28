"""Replay actual factual binary gates while preserving content-dependent attention."""

import copy
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import Tensor, nn

import scjepa.models.spartan as spartan_module
from diagnostics.joint_acquisition.baseline import MatchedBatch, MatchedPredictions, predict_matched
from diagnostics.signal_path.geometry import predict_rollout
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import Spartan, SpartanLayer
from scjepa.training.rng import preserve_rng


@dataclass(frozen=True)
class GateReplayResult:
    """Natural and intervened vectors plus effective mask changes on explicit batch axes.

    Predictions retain their computation graph. Masks are CPU bool; flip counts
    are CPU int64. Mask axes are G,D,[F],Q,H,L,M,M; counts drop the last two axes.
    """

    natural: MatchedPredictions
    replayed_predictions: Tensor
    factual_replay: Tensor
    factual_masks: Tensor
    natural_changed_masks: Tensor
    flip_counts: Tensor


@contextmanager
def _capture(predictor: Spartan) -> Generator[list[tuple[int, Tensor]]]:
    recorded: list[tuple[int, Tensor]] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def capture(index: int) -> Callable[[nn.Module, tuple[object, ...], object], None]:
        def hook(_module: nn.Module, _args: tuple[object, ...], output: object) -> None:
            adjacency = cast(tuple[Tensor, ...], output)[1]
            if not bool(((adjacency == 0) | (adjacency == 1)).all()):
                raise ValueError("Gate replay requires effective binary evaluation masks")
            recorded.append((index, adjacency.detach().bool().clone()))

        return hook

    try:
        for index, layer in enumerate(predictor.layers):
            handles.append(layer.register_forward_hook(capture(index)))
        yield recorded
    finally:
        for handle in handles:
            handle.remove()


def _masks(records: list[tuple[int, Tensor]], layers: int, batch: int) -> list[Tensor]:
    if len(records) != 2 * layers or [index for index, _ in records] != list(range(layers)) * 2:
        raise RuntimeError("Expected every layer exactly once at each attached prediction horizon")
    masks = [mask for _, mask in records]
    if any(mask.ndim != 3 or mask.shape[0] != batch for mask in masks):
        raise RuntimeError("Recorded masks do not match the factual/counterfactual batch")
    return masks


@contextmanager
def _replay(predictor: Spartan, masks: list[Tensor]) -> Generator[None]:
    """Replace the private gate sampler before masked attention, then always restore it."""
    index = 0
    dense = all(cast(SpartanLayer, layer).dense for layer in predictor.layers)

    def adjacency(logits: Tensor, _temperature: float, training: bool) -> Tensor:
        nonlocal index
        if training or index >= len(masks):
            raise RuntimeError("Unexpected training sampler or replay call count")
        selected = masks[index]
        index += 1
        if selected.shape != logits.shape:
            raise RuntimeError("Replay mask and live attention logits have different shapes")
        return selected.to(logits)

    with patch.object(spartan_module, "_sample_hard_adjacency", adjacency):
        yield
        if index != (0 if dense else len(masks)):
            raise RuntimeError("Not every recorded horizon/layer mask was consumed")


def _state_equal(model: SCJEPA, expected: dict[str, Any]) -> None:
    actual = model.state_dict()
    if set(actual) != set(expected):
        raise RuntimeError("Gate replay mutated model state keys")
    for name, value in actual.items():
        if isinstance(value, Tensor):
            if not torch.equal(value, expected[name]):
                raise RuntimeError(f"Gate replay mutated model tensor {name}")
        elif value != expected[name]:
            raise RuntimeError(f"Gate replay mutated model flags {name}")


def replay_matched(model: SCJEPA, batch: MatchedBatch) -> GateReplayResult:
    """Compare native counterfactuals with actual two-step factual-mask interventions.

    Only binary masks are frozen. All logits, masked normalizers, values and token
    MLPs are recomputed, and the replay's own first prediction enters its second
    step. The entire code is reused without detach. No model/probe fitting occurs.
    """
    predictor = model.predictor
    if (
        not isinstance(cast(nn.Module, predictor), Spartan)
        or any(module.training for module in model.modules())
        or not model.autoregressive
        or predictor.history_len != 1
        or predictor.param_dim != 6
        or predictor.state_dim != 6
        or batch.history.shape[-2:] != (1, 6)
        or batch.query_actions.shape[-2] != 2
        or batch.changed_donor_states.shape[2] != 6
        or batch.split != "validation"
    ):
        raise ValueError("Require an eval-mode attached six-code SPARTAN validation assay")
    layers = len(predictor.layers)
    layer_modes = {
        (cast(SpartanLayer, layer).dense, cast(SpartanLayer, layer).identity)
        for layer in predictor.layers
    }
    if not layers or layer_modes not in ({(False, False)}, {(True, False)}):
        raise ValueError("Require uniformly dense or sparse layers without identity overrides")
    reference = next(model.parameters())
    batch = batch.to(reference)
    groups, donors, queries = batch.history.shape[:3]
    factual_size = groups * donors * queries
    state = copy.deepcopy(model.state_dict())
    try:
        with preserve_rng(reference.device):
            with _capture(predictor) as records:
                natural = predict_matched(model, batch)
            factual = _masks(records[: 2 * layers], layers, factual_size)
            changed = _masks(records[2 * layers :], layers, factual_size * 6)
            expanded = [
                mask.reshape(groups, donors, queries, predictor.num_tokens, predictor.num_tokens)
                .unsqueeze(2)
                .expand(-1, -1, 6, -1, -1, -1)
                .flatten(0, 3)
                for mask in factual
            ]
            with _replay(predictor, factual), _capture(predictor) as replay_records:
                factual_replay = predict_rollout(
                    model,
                    batch.history.flatten(0, 2),
                    natural.codes[:, :, None].expand(-1, -1, queries, -1).flatten(0, 2),
                    batch.query_actions.flatten(0, 2),
                ).reshape(groups, donors, queries, 2, 6)
            if not torch.equal(factual_replay, natural.predictions):
                raise RuntimeError(
                    "Factual mask replay did not exactly reproduce factual predictions"
                )
            for expected, actual in zip(
                factual, _masks(replay_records, layers, factual_size), strict=True
            ):
                if not torch.equal(actual, expected):
                    raise RuntimeError("Factual effective mask replay differs")
            with _replay(predictor, expanded), _capture(predictor) as replay_records:
                replayed = predict_rollout(
                    model,
                    batch.history[:, :, None].expand(-1, -1, 6, -1, -1, -1).flatten(0, 3),
                    natural.changed_codes[:, :, :, None]
                    .expand(-1, -1, -1, queries, -1)
                    .flatten(0, 3),
                    batch.query_actions[:, :, None].expand(-1, -1, 6, -1, -1, -1).flatten(0, 3),
                ).reshape(groups, donors, 6, queries, 2, 6)
            for expected, actual in zip(
                expanded, _masks(replay_records, layers, factual_size * 6), strict=True
            ):
                if not torch.equal(actual, expected):
                    raise RuntimeError("Counterfactual effective masks differ from factual replay")
            tokens = predictor.num_tokens
            factual_tensor = (
                torch.stack(factual, dim=1)
                .reshape(groups, donors, queries, 2, layers, tokens, tokens)
                .cpu()
            )
            changed_tensor = (
                torch.stack(changed, dim=1)
                .reshape(groups, donors, 6, queries, 2, layers, tokens, tokens)
                .cpu()
            )
            flips = (changed_tensor != factual_tensor[:, :, None]).sum((-1, -2))
            return GateReplayResult(
                natural, replayed, factual_replay, factual_tensor, changed_tensor, flips
            )
    finally:
        _state_equal(model, state)
