"""Independent calibration data and strict read-only reuse of a dense diagnostic."""

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.oracle_state_predictor.data import (
    ACTION_DIMENSION,
    CONTEXT_LENGTH,
    FORECAST_STEPS,
    STATE_DIMENSION,
    OracleStateData,
)
from diagnostics.oracle_state_predictor.run import Config, build_predictor, code_identity
from scjepa.data.datasets import canonical_fingerprint
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.types import DGPConfig
from scjepa.training.rng import preserve_rng

_OPERATIONAL_FIELDS = (
    "output_dir",
    "steps",
    "device",
    "threads",
    "log_every",
    "evaluate_every",
    "checkpoint_every",
    "wandb_mode",
    "wandb_project",
    "wandb_entity",
    "wandb_group",
)


def _scientific_config(config: Config) -> dict[str, object]:
    result = cast(dict[str, object], asdict(config))
    for name in _OPERATIONAL_FIELDS:
        result.pop(name)
    return result


def _data_config(split: str, groups: int, seed: int, views: int) -> DGPConfig:
    return DGPConfig(
        family="simplified_gaussian",
        parameter_graph="complementary",
        root_seed=seed,
        split=split,
        num_episodes=groups,
        episodes_per_parameter=views,
        trajectory_length=CONTEXT_LENGTH + FORECAST_STEPS,
        dtype="float32",
        observation_seed=1729,
    )


def _diagnostic_fingerprint(source_fingerprint: str) -> str:
    """Reproduce the unchanged data.py interface identity without generating episodes."""
    start = CONTEXT_LENGTH - 1
    return canonical_fingerprint(
        {
            "schema": "scjepa.diagnostics.oracle-state-predictor.v1",
            "source_dataset_fingerprint": source_fingerprint,
            "source_state_index": start,
            "action_indices": list(range(start, start + FORECAST_STEPS)),
            "target_state_indices": list(range(start + 1, start + 1 + FORECAST_STEPS)),
            "state_source": "raw_ground_truth",
            "parameter_source": "raw_ground_truth",
            "action_source": "observed_mixed_coordinates",
            "dtype": "float32",
        }
    )


def build_calibration(
    groups: int,
    *,
    seed: int = 0,
    views: int = 10,
    progress: bool = False,
) -> OracleStateData:
    """Build the reserved evaluation_calibration split without changing old data.py.

    This is distinct from the original diagnostic's train, validation and test
    streams. The float32 input and target slices exactly match its fixed task.
    """
    if type(groups) is not int or groups < 1:
        raise ValueError("groups must be a positive integer")
    config = _data_config("evaluation_calibration", groups, seed, views)
    dataset = build_dgp_dataset(config)
    initial_states = torch.empty(groups, views, 1, STATE_DIMENSION, dtype=torch.float32)
    actions = torch.empty(groups, views, FORECAST_STEPS, ACTION_DIMENSION, dtype=torch.float32)
    targets = torch.empty(groups, views, FORECAST_STEPS, STATE_DIMENSION, dtype=torch.float32)
    parameters = torch.empty(groups, STATE_DIMENSION, dtype=torch.float32)
    episode_ids = torch.empty(groups, dtype=torch.int64)
    start = CONTEXT_LENGTH - 1
    for index in range(groups):
        episode = dataset.process.generate(index)
        states = episode.ground_truth["states"].reshape(views, config.trajectory_length, -1)
        observed_actions = episode.actions.reshape(views, config.trajectory_length - 1, -1)
        initial_states[index] = states[:, start : start + 1]
        actions[index] = observed_actions[:, start : start + FORECAST_STEPS]
        targets[index] = states[:, start + 1 : start + 1 + FORECAST_STEPS]
        parameters[index] = episode.ground_truth["parameters"]
        episode_ids[index] = episode.episode_id
        if progress and ((index + 1) % 1024 == 0 or index + 1 == groups):
            print(f"Prepared calibration oracle-state groups: {index + 1}/{groups}", flush=True)
    source = dataset.fingerprint
    return OracleStateData(
        initial_states,
        actions,
        targets,
        parameters,
        episode_ids,
        config,
        source,
        _diagnostic_fingerprint(source),
    )


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"dense reference {label} must be a mapping")
    return cast(dict[str, Any], value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_identities(record: dict[str, Any], config: Config) -> dict[str, str]:
    identities: dict[str, str] = {}
    recorded_data = _mapping(record.get("data"), "data metadata")
    for split, groups in (
        ("train", config.train_groups),
        ("validation", config.validation_groups),
        ("test", config.test_groups),
    ):
        data_config = _data_config(split, groups, config.data_seed, config.views)
        dataset = build_dgp_dataset(data_config)
        identities[split] = _diagnostic_fingerprint(dataset.fingerprint)
        metadata = _mapping(recorded_data.get(split), f"{split} data metadata")
        if (
            canonical_fingerprint(metadata.get("config"))
            != canonical_fingerprint(asdict(data_config))
            or metadata.get("source_fingerprint") != dataset.fingerprint
            or metadata.get("fingerprint") != identities[split]
            or metadata.get("source_state_index") != 9
            or metadata.get("action_indices") != [9, 10]
            or metadata.get("target_state_indices") != [10, 11]
            or metadata.get("dtype") != "float32"
        ):
            raise ValueError(f"dense reference {split} data metadata does not match current data")
    if record.get("dataset_identities") != identities:
        raise ValueError("dense reference dataset identities do not match current addressed data")
    return identities


def _load_checkpoint(
    path: Path, identity: str, config: Config, expected_state: dict[str, Tensor]
) -> dict[str, Any]:
    # Only use files explicitly chosen by the user as trusted local references.
    payload = _mapping(torch.load(path, map_location="cpu", weights_only=False), path.name)
    if payload.get("schema") != "oracle-state-predictor-v1" or payload.get("identity") != identity:
        raise ValueError(f"dense reference {path.name} schema or identity does not match")
    try:
        checkpoint_config = Config(**_mapping(payload.get("config"), f"{path.name} config"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"dense reference {path.name} has invalid configuration") from error
    if _scientific_config(checkpoint_config) != _scientific_config(config):
        raise ValueError(f"dense reference {path.name} scientific configuration does not match")
    step = payload.get("step")
    if type(step) is not int or step < 1:
        raise ValueError(f"dense reference {path.name} must contain a positive completed step")
    score = payload.get("best_validation_mse")
    if not isinstance(score, int | float) or not math.isfinite(score) or score < 0:
        raise ValueError(f"dense reference {path.name} validation score must be finite")
    state = _mapping(payload.get("model"), f"{path.name} model")
    if set(state) != set(expected_state):
        raise ValueError(f"dense reference {path.name} model keys do not match the architecture")
    for name, expected in expected_state.items():
        value: object = state[name]
        if (
            not isinstance(value, Tensor)
            or value.shape != expected.shape
            or value.dtype != expected.dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"dense reference {path.name} model tensor {name} is invalid")
    _mapping(payload.get("wandb"), f"{path.name} W&B metadata")
    return payload


def load_dense_reference(
    directory: Path, requested: Config
) -> tuple[dict[str, Any], dict[str, object]]:
    """Validate a completed trusted dense run and return its selected best checkpoint.

    The source directory is read-only. Scientific configuration, declared v1
    protocol, generated dataset descriptors and both model states must match.
    Compatible historical source identities are recorded explicitly. Operational
    output/device/logging settings and the requested sparse-stage duration may
    differ. Test observations or test scores never select the returned weights.
    """
    directory = directory.expanduser().resolve()
    record = _mapping(json.loads((directory / "config.json").read_text()), "config.json")
    if record.get("schema") != "oracle-state-predictor-v1":
        raise ValueError("dense reference is not the oracle-state predictor diagnostic")
    try:
        recorded_config = Config(**_mapping(record.get("config"), "configuration"))
    except (TypeError, ValueError) as error:
        raise ValueError("dense reference configuration is invalid") from error
    if _scientific_config(recorded_config) != _scientific_config(requested):
        raise ValueError("requested scientific configuration differs from the dense reference")
    current_source = code_identity()
    source = record.get("source_fingerprint")
    if (
        not isinstance(source, str)
        or len(source) != 64
        or any(character not in "0123456789abcdef" for character in source)
    ):
        raise ValueError("dense reference source fingerprint is invalid")
    if record.get("protocol") != (
        "true z9 and raw theta; observed actions9,10; attached two-step forecast; "
        "fixed true z10,z11 targets"
    ) or record.get("objective") != (
        "mean two-step MSE + logit_weight * mean SPARTAN logit penalty"
    ):
        raise ValueError("dense reference protocol or objective does not match the v1 diagnostic")
    expected_architecture = {
        "predictor": "Spartan",
        "dense": True,
        "layers": 2,
        "history_len": 1,
        "embed_dim": recorded_config.embed_dim,
        "mlp_hidden": recorded_config.mlp_hidden,
        "mlp_layers": 3,
    }
    if record.get("architecture") != expected_architecture:
        raise ValueError("dense reference architecture metadata does not match")
    identities = _validated_identities(record, recorded_config)
    identity = canonical_fingerprint(
        {"config": _scientific_config(recorded_config), "data": identities, "code": source}
    )
    if record.get("identity") != identity:
        raise ValueError("dense reference identity does not match its configuration/data/source")
    with preserve_rng(torch.device("cpu")):
        model = build_predictor(recorded_config)
        expected_state = cast(dict[str, Tensor], model.state_dict())
    best_path, last_path = directory / "best.pt", directory / "last.pt"
    best = _load_checkpoint(best_path, identity, recorded_config, expected_state)
    last = _load_checkpoint(last_path, identity, recorded_config, expected_state)
    if last["step"] != recorded_config.steps or best["step"] > last["step"]:
        raise ValueError("dense reference is incomplete or its selected step exceeds its last step")
    if best["best_validation_mse"] != last["best_validation_mse"]:
        raise ValueError("dense reference selected and last checkpoints disagree on the best score")
    metadata: dict[str, object] = {
        "directory": str(directory),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_checkpoint_sha256": _sha256(best_path),
        "last_checkpoint_sha256": _sha256(last_path),
        "selected_step": best["step"],
        "completed_step": last["step"],
        "identity": identity,
        "source_fingerprint": source,
        "current_source_fingerprint": current_source,
        "historical_source_differs": source != current_source,
        "dataset_identities": identities,
        "config": asdict(recorded_config),
        "wandb": last["wandb"],
    }
    return best, metadata
