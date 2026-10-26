"""CPU caches for the isolated single-context multiple-query diagnostic."""

import hashlib
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.factory import build_dgp_dataset
from scjepa.models.model import SCJEPA

type TensorSplit = dict[str, Tensor]
type EvaluationData = dict[str, Any]


def module_fingerprint(module: nn.Module) -> str:
    """Hash the realized state-encoder tensors independently of checkpoint bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def encode_states(model: SCJEPA, observations: Tensor, batch_size: int) -> tuple[Tensor, Tensor]:
    """Encode grouped trajectories on the model device and return CPU tensors."""
    shape = observations.shape[:-2]
    flat = observations.flatten(0, 1)
    reference = next(model.parameters())
    online: list[Tensor] = []
    target: list[Tensor] = []
    for start in range(0, len(flat), batch_size):
        chunk = flat[start : start + batch_size].to(reference)
        online.append(model.online_encoder.encode_sequence(chunk).detach().cpu())
        target.append(model.target_encoder.encode_sequence(chunk).detach().cpu())
    return (
        torch.cat(online).reshape(*shape, observations.shape[-2], -1),
        torch.cat(target).reshape(*shape, observations.shape[-2], -1),
    )


@torch.no_grad()
def build_cache(
    model: SCJEPA,
    observation_map: VectorObservationMap,
    data_config: dict[str, Any],
    *,
    train_groups: int = 8192,
    calibration_groups: int = 512,
    validation_groups: int = 256,
    test_groups: int = 512,
    eval_seed: int = 202610261401,
    batch_size: int = 32,
) -> tuple[TensorSplit, EvaluationData]:
    """Cache source-law training inputs and disjoint exact-law evaluation draws.

    Training has only frozen online/target states and observed actions. True theta,
    true states and simulation-precision actions appear only in evaluation splits.
    Calibration fits probes, validation selects them and final test scores them.
    """
    if data_config.get("family") != "simplified_gaussian":
        raise ValueError("This diagnostic requires simplified_gaussian")
    if min(train_groups, calibration_groups, validation_groups, test_groups) < 2:
        raise ValueError("Every split needs at least two independent theta groups")
    if batch_size < 1:
        raise ValueError("Cache encoding batch size must be positive")
    views = int(data_config.get("episodes_per_parameter", 2))
    length = int(data_config.get("trajectory_length", 0))
    if views != 10 or length != 12 or model.context_len != 10:
        raise ValueError("This isolated comparison requires R10/K10/T12")
    frozen = (module_fingerprint(model.online_encoder), module_fingerprint(model.target_encoder))
    was_training = model.training
    model.eval()
    train_parts: dict[str, list[Tensor]] = {key: [] for key in ("online", "target", "actions")}
    evaluation: EvaluationData = {
        "metadata": {
            "seed": eval_seed,
            "views_per_group": views,
            "context_length": model.context_len,
            "trajectory_length": length,
            "state_online_sha256": frozen[0],
            "state_target_sha256": frozen[1],
            "observation_map_fingerprint": observation_map.fingerprint,
            "observation_matrix": observation_map.matrix.detach().cpu(),
            "generator_version": dgp.GENERATOR_VERSION,
            "distribution": "Exact scientific draws; distinct seed per evaluation split",
            "training_labels_absent": True,
        }
    }
    try:
        train_config = dict(data_config, split="train", num_episodes=train_groups)
        dataset = build_dgp_dataset(
            OmegaConf.create(train_config), observation_matrix=observation_map.matrix
        )
        for start in range(0, train_groups, batch_size):
            episodes = [
                dataset.process.generate(index)
                for index in range(start, min(start + batch_size, train_groups))
            ]
            observations = torch.stack([episode.observations for episode in episodes])
            online, target = encode_states(model, observations, batch_size)
            train_parts["online"].append(online)
            train_parts["target"].append(target)
            train_parts["actions"].append(torch.stack([episode.actions for episode in episodes]))
        evaluation["metadata"]["training_dataset_fingerprint"] = dataset.fingerprint
        for index, (name, groups) in enumerate(
            (
                ("calibration", calibration_groups),
                ("validation", validation_groups),
                ("test", test_groups),
            )
        ):
            generator = torch.Generator(device="cpu").manual_seed(eval_seed + 1009 * index)
            theta = dgp.PARAMETER_BOUND * (
                2 * torch.rand(groups, 6, dtype=torch.float64, generator=generator) - 1
            )
            initial = torch.randn(groups, views, 6, dtype=torch.float64, generator=generator)
            gaussian = torch.randn(
                groups, views, length - 1, 10, dtype=torch.float64, generator=generator
            )
            coupling = (
                2
                * torch.rand(groups, views, length - 1, 6, dtype=torch.float64, generator=generator)
                - 1
            )
            actions = dgp.encode_controls(torch.cat((gaussian, coupling), -1))
            states = dgp.simulate(initial, theta[:, None], actions)
            online, target = encode_states(model, observation_map.forward(states), batch_size)
            evaluation[name] = {
                "states_online": online,
                "states_target": target,
                "actions": actions.float(),
                "true_actions": actions,
                "true_states": states,
                "theta": theta,
            }
    finally:
        model.train(was_training)
        model.online_encoder.eval()
        model.target_encoder.eval()
    if frozen != (
        module_fingerprint(model.online_encoder),
        module_fingerprint(model.target_encoder),
    ):
        raise RuntimeError("Caching changed a frozen state encoder")
    train = {key: torch.cat(parts).detach().cpu() for key, parts in train_parts.items()}
    return train, evaluation
