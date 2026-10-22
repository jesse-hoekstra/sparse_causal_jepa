"""Read compatible dense references without changing their original provenance."""

from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.datasets import canonical_fingerprint
from scjepa.data.factory import build_dgp_dataset
from scjepa.experiments.config import ExperimentConfig
from scjepa.experiments.storage import json_safe, read_json


def dense_directory(config: ExperimentConfig, graph: str, seed: int) -> Path:
    """Locate the selected graph/seed's dense stage in the configured reference root."""
    return Path(config.dense_reference_dir or config.output_dir) / graph / f"seed-{seed}" / "dense"


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"external dense reference lacks a valid {name}")
    return cast(dict[str, object], value)


def external_dense_manifest(
    config: ExperimentConfig, current_manifest: dict[str, object], graph: str, seed: int
) -> dict[str, object]:
    """Check compatibility and return the old manifest for normal tau/hash validation.

    Source versions and optimizer schedules may differ. The observation map,
    architecture and calibrated base-loss statistic must retain their meaning.
    Formerly absent options receive their legacy defaults only for semantic
    compatibility checks; stored identity/hash verification remains unchanged.
    """
    if config.dense_reference_dir is None:
        raise ValueError("external dense reference directory is not configured")
    root = Path(config.dense_reference_dir)
    manifest = read_json(root / "experiment.json")
    body = {key: value for key, value in manifest.items() if key != "identity"}
    if (
        manifest.get("format_version") != 1
        or manifest.get("identity") != canonical_fingerprint(json_safe(body))
        or manifest.get("config_fingerprint") != canonical_fingerprint(manifest.get("definition"))
    ):
        raise ValueError("external dense reference manifest integrity check failed")

    directory = dense_directory(config, graph, seed)
    record = read_json(directory / "run_config.json")
    if _mapping(record.get("data"), "data configuration").get("family") != config.data_family:
        raise ValueError(
            "external dense reference data family differs; train a new dense reference"
        )
    tau = read_json(directory / "tau.json")
    checkpoint = _mapping(
        torch.load(directory / "best.pt", map_location="cpu", weights_only=False), "checkpoint"
    )
    identity = canonical_fingerprint([manifest["identity"], graph, seed, "dense"])
    if any(
        record.get(key) != value
        for key, value in {
            "experiment_identity": manifest["identity"],
            "run_identity": identity,
            "graph": graph,
            "seed": seed,
            "condition": "dense",
        }.items()
    ):
        raise ValueError("external dense reference run identity differs from its manifest")
    metadata = _mapping(checkpoint.get("run_metadata"), "checkpoint metadata")
    dataset_identity = record.get("dataset_identity")
    step = checkpoint.get("step")
    if (
        checkpoint.get("format_version") != 1
        or metadata.get("experiment_run_identity") != identity
        or not isinstance(dataset_identity, str)
        or not dataset_identity
        or checkpoint.get("dataset_identity") != dataset_identity
        or type(step) is not int
        or step < 0
        or step != tau.get("selected_step")
    ):
        raise ValueError("external dense reference checkpoint identity or selected step differs")

    expected_model = canonical_fingerprint(asdict(config.condition_model("dense")))
    for value in (record.get("model"), checkpoint.get("model_config")):
        model = dict(_mapping(value, "model configuration"))
        model.setdefault("spartan_parameter_gates_open", False)
        model.setdefault("spartan_action_gates_open", False)
        model.setdefault("parameter_encoder_kind", "attention")
        if canonical_fingerprint(model) != expected_model:
            raise ValueError("external dense reference architecture differs")
    if canonical_fingerprint(record.get("data")) != canonical_fingerprint(
        asdict(config.data_config(graph))
    ):
        raise ValueError("external dense reference data configuration differs")

    expected_objective = {
        key: value
        for key, value in asdict(config.training_config("dense", seed, directory).objective).items()
        if not key.endswith("_sparsity_weight")
    }
    for value in (record.get("train"), checkpoint.get("train_config")):
        training = _mapping(value, "training configuration")
        objective = _mapping(training.get("objective"), "base objective")
        base_objective = {
            key: value for key, value in objective.items() if not key.endswith("_sparsity_weight")
        }
        base_objective.setdefault("parameter_consistency_weight", 0.0)
        if base_objective != expected_objective:
            raise ValueError("external dense reference base objective differs")
    evaluation = _mapping(record.get("evaluation"), "calibration configuration")
    if any(
        evaluation.get(key) != getattr(config.evaluation, key)
        for key in ("seed", "batch_size", "calibration_episodes")
    ):
        raise ValueError("external dense reference calibration configuration differs")

    observation = _mapping(
        torch.load(root / "observation_map.pt", map_location="cpu", weights_only=True),
        "observation map",
    )
    matrix = observation.get("matrix")
    if not isinstance(matrix, Tensor):
        raise ValueError("external dense reference observation map lacks its matrix")
    fingerprint = VectorObservationMap(
        matrix=matrix, dimension=config.model.observation_dim
    ).fingerprint
    if any(
        item.get("observation_fingerprint") != fingerprint for item in (manifest, current_manifest)
    ):
        raise ValueError("external dense reference observation map differs")
    expected_dataset = build_dgp_dataset(config.data_config(graph), observation_matrix=matrix)
    if dataset_identity != expected_dataset.fingerprint:
        raise ValueError("external dense reference dataset generator differs")
    return manifest
