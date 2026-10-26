"""Read and validate a finished weight-one checkpoint for matched continuations."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiquery_control.run import frozen_hashes, sha
from diagnostics.predictive_supervision_control.run import code_identity as training_code_identity
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.training.factory import build_model
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng


@dataclass(frozen=True)
class SourceBundle:
    """Validated immutable source references; callers must not modify their payloads."""

    source_dir: Path
    seed: int
    protocol: dict[str, Any]
    checkpoint: dict[str, Any]
    train: TensorSplit
    evaluation: EvaluationData
    hashes: dict[str, str]

    @property
    def source_step(self) -> int:
        """Return the completed source update count, used to continue its RNG stream."""
        return int(self.checkpoint["step"])

    @property
    def batch_size(self) -> int:
        """Retain the original number of independent parameter groups per batch."""
        return int(self.protocol["config"]["batch_size"])

    @property
    def train_groups(self) -> int:
        """Return the original cached dataset size for the deterministic sampler."""
        return int(self.protocol["config"]["train_groups"])


def validate_output_destination(source_dir: Path, output_dir: Path) -> None:
    """Keep all continuation artifacts outside the source experiment, including symlinks."""
    source, output = source_dir.resolve(), output_dir.resolve()
    if source == output or source in output.parents:
        raise ValueError("Write continuation outputs outside the source experiment directory")


def _validate_model(model: SCJEPA, bundle: SourceBundle) -> None:
    if (
        not isinstance(model.parameter_encoder, TransitionParameterEncoder)
        or model.parameter_source != "learned"
        or model.context_len != 10
        or not model.autoregressive
        or model.predictor.history_len != 1
        or (model.predictor.state_dim, model.predictor.param_dim, model.predictor.action_dim)
        != (6, 6, 16)
        or not all(layer.dense for layer in model.predictor.layers)
        or frozen_hashes(model) != bundle.protocol["frozen_hashes"]
        or bundle.checkpoint["frozen_hashes"] != bundle.protocol["frozen_hashes"]
    ):
        raise ValueError("Source must be the original MLP dense K10 frozen-state comparison")
    metadata = bundle.evaluation["metadata"]
    if metadata["state_online_sha256"] != module_fingerprint(model.online_encoder) or metadata[
        "state_target_sha256"
    ] != module_fingerprint(model.target_encoder):
        raise ValueError("Evaluation cache state-encoder metadata differ from the checkpoint")


def build_source_model(
    bundle: SourceBundle, device: torch.device
) -> tuple[SCJEPA, torch.optim.AdamW]:
    """Restore a separate model and AdamW state for one arm without resetting moments."""
    with preserve_rng(torch.device("cpu")):
        model = build_model(cast(DictConfig, OmegaConf.create(bundle.protocol["model_config"])))
    model.load_state_dict(bundle.checkpoint["model"], strict=True)
    _validate_model(model, bundle)
    model.online_encoder.requires_grad_(False).eval()
    model.target_encoder.requires_grad_(False).eval()
    model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    state = cast(dict[str, Any], bundle.checkpoint["optimizer"])
    groups = cast(list[dict[str, Any]], state["param_groups"])
    if len(groups) != 1 or len(groups[0]["params"]) != len(parameters):
        raise ValueError("Source optimizer parameter groups do not match trainable parameters")
    ids = cast(list[int], groups[0]["params"])
    if len(set(ids)) != len(ids):
        raise ValueError("Source optimizer parameter IDs are not distinct")
    if groups[0]["lr"] != bundle.protocol["config"]["lr"] or groups[0]["weight_decay"] != 0:
        raise ValueError("Source AdamW learning rate or weight decay differs from its protocol")
    states = cast(dict[int, dict[str, Any]], state["state"])
    if not set(states).issubset(ids) or not states:
        raise ValueError("Finished source must contain valid Adam moments")
    for pid, parameter in zip(ids, parameters, strict=True):
        if pid not in states:
            continue
        entry = states[pid]
        step = entry["step"]
        count = float(step.item()) if isinstance(step, Tensor) else float(step)
        if count != bundle.source_step:
            raise ValueError("Source Adam moment step differs from the completed source step")
        for key in ("exp_avg", "exp_avg_sq"):
            moment = entry[key]
            if (
                not isinstance(moment, Tensor)
                or moment.shape != parameter.shape
                or not bool(torch.isfinite(moment).all())
                or (key == "exp_avg_sq" and bool((moment < 0).any()))
            ):
                raise ValueError("Invalid source Adam moment shape or values")
    optimizer = torch.optim.AdamW(parameters, lr=float(groups[0]["lr"]), weight_decay=0.0)
    # load_state_dict may reuse CPU tensor storage from the supplied state. Clone
    # moments so a CPU continuation cannot mutate the shared starting checkpoint.
    optimizer.load_state_dict(copy.deepcopy(state))
    return model, optimizer


def _validate_caches(bundle: SourceBundle) -> None:
    groups, batch_size = bundle.train_groups, bundle.batch_size
    if groups < batch_size or batch_size < 2 or groups % batch_size:
        raise ValueError("Source training groups must be divisible by batch_size>=2")
    if set(bundle.train) != {"online", "target", "actions"}:
        raise ValueError("Training cache must contain only frozen states and observed actions")
    expected_shapes = {
        "online": (groups, 10, 12, 6),
        "target": (groups, 10, 12, 6),
        "actions": (groups, 10, 11, 16),
    }
    for key, shape in expected_shapes.items():
        value = bundle.train[key]
        if (
            not isinstance(value, Tensor)  # pyright: ignore[reportUnnecessaryIsInstance]
            or tuple(value.shape) != shape
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"Invalid cached training tensor: {key}")
    metadata = bundle.evaluation["metadata"]
    observation = VectorObservationMap(matrix=metadata["observation_matrix"], dimension=6)
    if (
        metadata["generator_version"] != dgp.GENERATOR_VERSION
        or metadata["observation_map_fingerprint"] != observation.fingerprint
        or observation.fingerprint != bundle.protocol["source"]["observation_map_fingerprint"]
        or metadata["views_per_group"] != 10
        or metadata["context_length"] != 10
        or metadata["trajectory_length"] != 12
        or metadata["training_labels_absent"] is not True
    ):
        raise ValueError("Evaluation metadata differ from the approved frozen-state source")
    for split_name in ("calibration", "validation", "test"):
        split = cast(TensorSplit, bundle.evaluation[split_name])
        count = int(bundle.protocol["config"][f"{split_name}_groups"])
        shapes = {
            "theta": (count, 6),
            "states_online": (count, 10, 12, 6),
            "states_target": (count, 10, 12, 6),
            "true_states": (count, 10, 12, 6),
            "actions": (count, 10, 11, 16),
            "true_actions": (count, 10, 11, 16),
        }
        if set(split) != set(shapes) or count < batch_size or count % batch_size:
            raise ValueError(f"Invalid evaluation split: {split_name}")
        for key, shape in shapes.items():
            value = split[key]
            if (
                not isinstance(value, Tensor)  # pyright: ignore[reportUnnecessaryIsInstance]
                or tuple(value.shape) != shape
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"Invalid evaluation tensor: {split_name}/{key}")


def load_source(source_dir: Path, seed: int) -> SourceBundle:
    """Load only validated final weight-one artifacts; never write to the source directory."""
    root = source_dir.resolve()
    protocol_path = root / "protocol.json"
    protocol = cast(dict[str, Any], read_json(protocol_path))
    if protocol["code_identity"] != training_code_identity():
        raise ValueError(
            "Source training code changed; restore its original helper/source versions"
        )
    if (
        seed not in protocol["config"]["seeds"]
        or 1.0 not in protocol["config"]["prediction_weights"]
    ):
        raise ValueError("Requested seed and prediction-weight-one arm must exist in the source")
    objective = ObjectiveConfig(**protocol["objective_config"])
    if (
        objective.state_regularizer != "sigreg"
        or objective.parameter_regularizer != "sigreg"
        or objective.state_regularizer_weight != 0.4
        or objective.parameter_regularizer_weight != 0.4
        or objective.parameter_consistency_weight != 1.0
        or any(
            (
                objective.state_sparsity_weight,
                objective.action_sparsity_weight,
                objective.parameter_sparsity_weight,
            )
        )
    ):
        raise ValueError("Source must retain SIGReg0.4, consistency1 and zero sparsity weights")
    hashes = {"protocol.json": sha(protocol_path)}
    for filename in ("train_inputs.pt", "evaluation_data.pt"):
        hashes[filename] = sha(root / filename)
        if hashes[filename] != protocol["artifacts"][filename]:
            raise ValueError(f"Source cache hash mismatch: {filename}")
    arm = root / f"seed-{seed}" / "prediction_1"
    completed = cast(dict[str, Any], read_json(arm / "completed.json"))
    if completed["identity"] != protocol["identity"]:
        raise ValueError("Completed source arm belongs to another protocol")
    hashes[str((arm / "completed.json").relative_to(root))] = sha(arm / "completed.json")
    for filename, key in (
        ("last.pt", "checkpoint_sha256"),
        ("results.json", "results_sha256"),
        ("gradient_history.json", "gradient_history_sha256"),
    ):
        value = sha(arm / filename)
        hashes[str((arm / filename).relative_to(root))] = value
        if value != completed[key]:
            raise ValueError(f"Completed source artifact changed: {filename}")
    result = cast(dict[str, Any], read_json(arm / "results.json"))
    payload = cast(
        dict[str, Any], torch.load(arm / "last.pt", map_location="cpu", weights_only=False)
    )
    expected = {
        "identity": protocol["identity"],
        "condition": "prediction_1",
        "seed": seed,
        "prediction_weight": 1.0,
    }
    if any(payload[key] != value or result[key] != value for key, value in expected.items()):
        raise ValueError("Source checkpoint and result do not identify the selected weight-one arm")
    source_step = int(protocol["config"]["steps"])
    if (
        source_step < 1
        or payload["step"] != source_step
        or result["steps"] != source_step
        or result["checkpoint_sha256"] != completed["checkpoint_sha256"]
        or payload["source_sha256"] != protocol["source"]["sha256"]
        or payload["initial_sha256"] != protocol["initializations"][str(seed)]["sha256"]
        or payload["queries"] != 1
        or payload["parameter_regularizer_weight"] != 0.4
        or payload["consistency_weight"] != 1.0
    ):
        raise ValueError("Source checkpoint is not the completed matched prediction-one run")
    bundle = SourceBundle(
        root,
        seed,
        protocol,
        payload,
        cast(
            TensorSplit, torch.load(root / "train_inputs.pt", map_location="cpu", weights_only=True)
        ),
        cast(
            EvaluationData,
            torch.load(root / "evaluation_data.pt", map_location="cpu", weights_only=True),
        ),
        hashes,
    )
    _validate_caches(bundle)
    # Validate actual state tensors, extra-state prediction flags and saved Adam
    # moments before any continuation creates checkpoints or W&B runs.
    build_source_model(bundle, torch.device("cpu"))
    return bundle
