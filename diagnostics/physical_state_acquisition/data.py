"""Authenticate the original addressed trajectories before exposing physical states."""

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from diagnostics.encoder_gradient_control.source import SourceBundle
from diagnostics.multiquery_control.data import (
    EvaluationData,
    TensorSplit,
    encode_states,
    module_fingerprint,
)
from diagnostics.multiquery_control.run import sha
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.datasets import tensor_fingerprint
from scjepa.data.factory import build_dgp_dataset
from scjepa.experiments.storage import read_json
from scjepa.models.encoders import FramewiseEncoder
from scjepa.models.model import SCJEPA


class PhysicalIdentityEncoder(FramewiseEncoder):
    """Accept explicitly supplied physical states; never invert observed sensor values."""

    state_dim = 6
    observation_ndim = 1

    def __init__(self) -> None:
        """Persist a coordinate marker so frozen-state hashes are nonempty."""
        super().__init__()
        self.register_buffer("physical_coordinate_identity", torch.eye(6))

    def forward(self, observations: Tensor) -> Tensor:
        """Return already-physical state coordinates unchanged."""
        if observations.ndim != 2 or observations.shape[-1] != 6:
            raise ValueError("Physical identity expects (batch, 6) raw physical states")
        return observations


def source_data_config(
    bundle: SourceBundle, path: Path | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require the original dataset definition, not a guessed seed or inverse map."""
    provenance = bundle.protocol["source"]
    path = path or Path(provenance["checkpoint"]).parent / "run_config.json"
    record = cast(dict[str, Any], read_json(path))
    if (
        record["run_identity"] != provenance["run_identity"]
        or record["dataset_identity"] != provenance["dataset_identity"]
        or record["model"] != bundle.protocol["model_config"]
    ):
        raise ValueError("Source run configuration differs from authenticated provenance")
    dataset = build_dgp_dataset(
        cast(DictConfig, OmegaConf.create(record["data"])),
        observation_matrix=bundle.evaluation["metadata"]["observation_matrix"],
    )
    if dataset.fingerprint != provenance["dataset_identity"]:
        raise ValueError("Source data settings do not reproduce the original dataset identity")
    return record["data"], {"path": str(path.resolve()), "sha256": sha(path)}


@torch.no_grad()
def build_physical_cache(
    model: SCJEPA,
    train: TensorSplit,
    evaluation: EvaluationData,
    data_config: dict[str, Any],
    *,
    batch_size: int = 32,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[TensorSplit, EvaluationData, dict[str, Any]]:
    """Replace state coordinates only after all addressed TRAIN trajectories authenticate.

    Physical parameter labels remain confined to reconstruction and evaluation.
    The returned training mapping contains only online/target physical states
    and the original mixed observed actions. No inverse sensor/action map enters
    training. Original source/cache objects are never modified.
    """
    if set(train) != {"online", "target", "actions"} or batch_size < 1:
        raise ValueError("Require label-free source cache and positive verification batch size")
    config = {**data_config, "split": "train", "num_episodes": len(train["online"])}
    if (
        config["family"] != "simplified_gaussian"
        or config["episodes_per_parameter"] != 10
        or config["trajectory_length"] != 12
    ):
        raise ValueError("Physical diagnostic preserves the original DGP/R10/T12")
    metadata = evaluation["metadata"]
    dataset = build_dgp_dataset(
        OmegaConf.create(config), observation_matrix=metadata["observation_matrix"]
    )
    if dataset.fingerprint != metadata["training_dataset_fingerprint"]:
        raise ValueError("Training dataset fingerprint does not reproduce the recorded cache")
    observation = VectorObservationMap(matrix=metadata["observation_matrix"], dimension=6)
    pieces: list[Tensor] = []
    maximum = {"online": 0.0, "target": 0.0}
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            episodes = [dataset.process.generate(index) for index in range(start, end)]
            actions = torch.stack([episode.actions for episode in episodes])
            if not torch.equal(actions, train["actions"][start:end]):
                raise ValueError(
                    f"Addressed training actions differ from source cache: {start}:{end}"
                )
            for episode in episodes:
                truth = episode.ground_truth
                states = dgp.simulate(truth["initial_state"], truth["parameters"], truth["actions"])
                if not torch.equal(states, truth["states"]) or not torch.equal(
                    observation.forward(states).to(episode.observations), episode.observations
                ):
                    raise ValueError(
                        "Generated physical states do not reproduce their observations"
                    )
            online, target = encode_states(
                model, torch.stack([episode.observations for episode in episodes]), batch_size
            )
            for name, regenerated in (("online", online), ("target", target)):
                cached = train[name][start:end]
                if regenerated.shape != cached.shape or not bool(torch.isfinite(cached).all()):
                    raise ValueError("Invalid source learned-state cache")
                error = float((regenerated.double() - cached.double()).abs().max())
                maximum[name] = max(maximum[name], error)
                if not bool(torch.isfinite(regenerated).all()) or error > 2e-5:
                    raise ValueError(f"Regenerated learned {name} states differ from source cache")
            pieces.append(torch.stack([episode.ground_truth["states"] for episode in episodes]))
            if progress is not None:
                progress(end, len(dataset))
    finally:
        model.train(was_training)
        model.online_encoder.eval()
        model.target_encoder.eval()
    physical = torch.cat(pieces).to(train["online"]).detach()
    output_train = {
        "online": physical.clone(),
        "target": physical.clone(),
        "actions": train["actions"].clone(),
    }
    output_evaluation = copy.deepcopy(evaluation)
    for split in ("calibration", "validation", "test"):
        values = cast(TensorSplit, output_evaluation[split])
        reconstructed = dgp.simulate(
            values["true_states"][:, :, 0], values["theta"][:, None], values["true_actions"]
        )
        if not torch.equal(reconstructed, values["true_states"]) or not torch.equal(
            values["true_actions"].to(values["actions"]), values["actions"]
        ):
            raise ValueError(f"Evaluation {split} physical trajectories do not reproduce")
        states = values["true_states"].to(values["states_online"])
        values["states_online"] = states.clone()
        values["states_target"] = states.clone()
    output_evaluation["metadata"]["state_coordinates"] = "physical_identity"
    identity_hash = module_fingerprint(PhysicalIdentityEncoder().to(train["online"]))
    for name in ("online", "target"):
        key = f"state_{name}_sha256"
        output_evaluation["metadata"]["source_" + key] = metadata[key]
        output_evaluation["metadata"][key] = identity_hash
    receipt = {
        "training_dataset_fingerprint": dataset.fingerprint,
        "training_groups_verified": len(dataset),
        "all_addressed_actions_equal": True,
        "source_state_max_absolute_error": maximum,
        "source_state_tolerance": 2e-5,
        "physical_train_state_hash": tensor_fingerprint(physical),
        "original_mixed_action_hash": tensor_fingerprint(output_train["actions"]),
        "training_keys": sorted(output_train),
        "training_parameter_labels_absent": True,
        "physical_coordinates_privileged": True,
    }
    return output_train, output_evaluation, receipt
