"""Authenticate original observations and regenerate all moving state representations."""

from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
from omegaconf import OmegaConf

from diagnostics.multiquery_control.data import (
    EvaluationData,
    TensorSplit,
    encode_states,
    module_fingerprint,
)
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.factory import build_dgp_dataset
from scjepa.models.model import SCJEPA
from scjepa.training.rng import preserve_rng


@torch.no_grad()
def regenerate_train(
    model: SCJEPA,
    cached: TensorSplit,
    metadata: dict[str, Any],
    data_config: dict[str, Any],
    *,
    progress: Callable[[int], None] | None = None,
) -> tuple[TensorSplit, dict[str, Any]]:
    """Keep only observed inputs after checking every regenerated action/state against TRAIN."""
    if set(cached) != {"online", "target", "actions"}:
        raise ValueError("Require the original label-free TRAIN cache")
    observation_map = VectorObservationMap(matrix=metadata["observation_matrix"], dimension=6)
    if observation_map.fingerprint != metadata["observation_map_fingerprint"]:
        raise ValueError("Observation map identity differs")
    count = len(cached["online"])
    dataset = build_dgp_dataset(
        OmegaConf.create(dict(data_config, split="train", num_episodes=count)),
        observation_matrix=observation_map.matrix,
    )
    if dataset.fingerprint != metadata["training_dataset_fingerprint"]:
        raise ValueError("Regenerated TRAIN dataset fingerprint differs")
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    errors = {name: 0.0 for name in ("online", "target")}
    with preserve_rng(next(model.parameters()).device):
        model.eval()
        for start in range(0, count, 32):
            episodes = [
                dataset.process.generate(index) for index in range(start, min(start + 32, count))
            ]
            observed = torch.stack([episode.observations for episode in episodes])
            action = torch.stack([episode.actions for episode in episodes])
            if not torch.equal(action, cached["actions"][start : start + len(episodes)]):
                raise ValueError("Regenerated observed actions differ from original TRAIN")
            online, target = encode_states(model, observed, 32)
            for name, realized in (("online", online), ("target", target)):
                expected = cached[name][start : start + len(episodes)]
                torch.testing.assert_close(realized, expected, atol=1e-6, rtol=1e-5)
                errors[name] = max(errors[name], float((realized - expected).abs().max()))
            observations.append(observed.detach().cpu())
            actions.append(action.detach().cpu())
            if progress:
                progress(min(start + 32, count))
    train = {"observations": torch.cat(observations), "actions": torch.cat(actions)}
    check_train(train)
    return train, {
        "dataset_fingerprint": dataset.fingerprint,
        "observations_only": True,
        "actions_exact": True,
        "all_initial_encodings_checked": True,
        "encoding_atol": 1e-6,
        "encoding_rtol": 1e-5,
        "maximum_encoding_absolute_error": errors,
    }


def check_train(train: TensorSplit) -> None:
    """Reject labels, stale encodings and changed grouped geometry at the learner boundary."""
    if set(train) != {"observations", "actions"}:
        raise ValueError("Training accepts only observations and actions")
    count = len(train["observations"])
    if train["observations"].shape != (count, 10, 12, 6) or train["actions"].shape != (
        count,
        10,
        11,
        16,
    ):
        raise ValueError("Require original R10/T12/observation6/action16 geometry")
    if any(
        value.dtype != torch.float32 or value.requires_grad or not bool(torch.isfinite(value).all())
        for value in train.values()
    ):
        raise ValueError("Observed TRAIN tensors must be detached finite float32")


@torch.no_grad()
def refresh_evaluation(
    model: SCJEPA, data: EvaluationData, splits: Sequence[str]
) -> EvaluationData:
    """Re-observe fixed physical trajectories and encode both current float32 state charts."""
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("Preserve original float32 factual state encoding policy")
    observation_map = VectorObservationMap(
        matrix=data["metadata"]["observation_matrix"], dimension=6
    )
    result: EvaluationData = {"metadata": dict(data["metadata"])}
    with preserve_rng(next(model.parameters()).device):
        model.eval()
        for name in splits:
            split = cast(TensorSplit, data[name])
            online, target = encode_states(model, observation_map.forward(split["true_states"]), 32)
            result[name] = {**split, "states_online": online, "states_target": target}
    for name in ("online", "target"):
        result["metadata"][f"state_{name}_sha256"] = module_fingerprint(
            getattr(model, f"{name}_encoder")
        )
    result["metadata"]["state_encodings_refreshed"] = True
    return result
