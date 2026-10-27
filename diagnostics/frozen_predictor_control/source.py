"""Export a portable frozen predictor source and initialize an independent fresh encoder."""

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from diagnostics.frozen_predictor_control.diagnostics import code_reference_stats
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiquery_control.evaluation import encode_codes
from diagnostics.multiquery_control.run import sha
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.training.factory import build_model
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng

FORMAT_VERSION = "fresh-encoder-frozen-six-factor-predictor-v1"
FROZEN_MODULES = ("online_encoder", "target_encoder", "predictor")
ARTIFACTS = ("frozen_modules.pt", "train_inputs.pt", "evaluation_data.pt")


@dataclass(frozen=True)
class SourceBundle:
    """Validated frozen modules and independent train/evaluation caches."""

    source_dir: Path
    manifest: dict[str, Any]
    frozen_payload: dict[str, Any]
    train: TensorSplit
    evaluation: EvaluationData

    @property
    def model_config(self) -> dict[str, Any]:
        """Return the architecture configuration bound to the frozen weights."""
        return cast(dict[str, Any], self.manifest["model_config"])

    @property
    def objective_config(self) -> dict[str, Any]:
        """Return the unchanged original dense objective configuration."""
        return cast(dict[str, Any], self.manifest["objective_config"])

    @property
    def batch_size(self) -> int:
        """Return the original number of independent groups in a minibatch."""
        return int(self.manifest["batch_size"])

    @property
    def train_groups(self) -> int:
        """Return the number of cached independent training groups."""
        return int(self.manifest["train_groups"])


def manifest_identity(manifest: dict[str, Any]) -> str:
    """Hash portable semantic metadata and artifact hashes, excluding this digest."""
    payload = {key: value for key, value in manifest.items() if key != "identity"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def frozen_module_hashes(model: SCJEPA) -> dict[str, str]:
    """Fingerprint all three fixed modules using sorted tensor names, shapes and bytes."""
    return {name: module_fingerprint(getattr(model, name)) for name in FROZEN_MODULES}


def legacy_fingerprint(module: nn.Module) -> str:
    """Reproduce the archived positive control's insertion-order, no-shape digest."""
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _validate_model(model: SCJEPA) -> None:
    if (
        not isinstance(model.parameter_encoder, TransitionParameterEncoder)
        or model.parameter_source != "learned"
        or model.context_len != 10
        or not model.autoregressive
        or model.predictor.history_len != 1
        or (model.predictor.state_dim, model.predictor.param_dim, model.predictor.action_dim)
        != (6, 6, 16)
        or not all(layer.dense for layer in model.predictor.layers)
    ):
        raise ValueError("Source requires the original dense transition MLP and R10/K10/T12")


def _validate_objective(config: dict[str, Any]) -> None:
    objective = ObjectiveConfig(**config)
    if (
        objective.state_regularizer != "sigreg"
        or objective.parameter_regularizer != "sigreg"
        or objective.state_regularizer_weight != 0.4
        or objective.parameter_regularizer_weight != 0.4
        or objective.parameter_consistency_weight != 1.0
        or objective.logit_weight != 0.0001
        or any(
            (
                objective.state_sparsity_weight,
                objective.action_sparsity_weight,
                objective.parameter_sparsity_weight,
            )
        )
    ):
        raise ValueError("Source objective must retain SIGReg0.4, consistency1 and dense weights")


def _validate_evidence(evidence: dict[str, Any]) -> None:
    scores = cast(list[float], evidence["test_r2"])
    responses = cast(list[dict[str, Any]], evidence["per_parameter"])
    if len(scores) != 6 or len(responses) != 6:
        raise ValueError("Source evaluation must report all six parameters")
    for index, (score, response) in enumerate(zip(scores, responses, strict=True), start=1):
        interval = response["counterfactual_mse_improvement"][
            "paired_theta_group_bootstrap_95percent_interval"
        ]
        values = [score, response["response_cosine"], response["response_energy_ratio"], *interval]
        if (
            response["parameter"] != index
            or len(interval) != 2
            or not all(math.isfinite(value) and value > 0 for value in values)
            or interval[0] > interval[1]
        ):
            raise ValueError(
                "Source lacks positive recovery and predictive-use evidence for all six"
            )


def _validate_caches(bundle: SourceBundle) -> None:
    groups, batch_size = bundle.train_groups, bundle.batch_size
    if groups < batch_size or batch_size < 2 or groups % batch_size:
        raise ValueError("Training groups must be divisible by batch_size>=2")
    shapes = {
        "online": (groups, 10, 12, 6),
        "target": (groups, 10, 12, 6),
        "actions": (groups, 10, 11, 16),
    }
    if set(bundle.train) != set(shapes):
        raise ValueError("Training inputs must contain only online, target and actions")
    for name, shape in shapes.items():
        value = bundle.train[name]
        if (
            not isinstance(value, Tensor)  # pyright: ignore[reportUnnecessaryIsInstance]
            or tuple(value.shape) != shape
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"Invalid label-free training tensor: {name}")
    if set(bundle.evaluation) != {"metadata", "calibration", "validation", "test"}:
        raise ValueError("Evaluation must have separate calibration, validation and test splits")
    metadata = bundle.evaluation["metadata"]
    observation = VectorObservationMap(matrix=metadata["observation_matrix"], dimension=6)
    if (
        metadata["generator_version"] != dgp.GENERATOR_VERSION
        or metadata["observation_map_fingerprint"] != observation.fingerprint
        or observation.fingerprint != bundle.manifest["observation_map_fingerprint"]
        or metadata["views_per_group"] != 10
        or metadata["context_length"] != 10
        or metadata["trajectory_length"] != 12
        or metadata["training_labels_absent"] is not True
    ):
        raise ValueError("Evaluation DGP/map/shape metadata differ from the frozen source")
    for split_name in ("calibration", "validation", "test"):
        split = cast(TensorSplit, bundle.evaluation[split_name])
        count = int(bundle.manifest["evaluation_groups"][split_name])
        expected = {
            "theta": (count, 6),
            "states_online": (count, 10, 12, 6),
            "states_target": (count, 10, 12, 6),
            "true_states": (count, 10, 12, 6),
            "actions": (count, 10, 11, 16),
            "true_actions": (count, 10, 11, 16),
        }
        if set(split) != set(expected) or count < 2:
            raise ValueError(f"Invalid held-out split: {split_name}")
        for name, shape in expected.items():
            value = split[name]
            if (
                not isinstance(value, Tensor)  # pyright: ignore[reportUnnecessaryIsInstance]
                or tuple(value.shape) != shape
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
                or (
                    name in {"states_online", "states_target", "actions"}
                    and value.dtype != torch.float32
                )
            ):
                raise ValueError(f"Invalid held-out tensor: {split_name}/{name}")


def build_fresh_model(bundle: SourceBundle, seed: int, device: torch.device) -> SCJEPA:
    """Create a new random q encoder, copying only frozen modules and preserving caller RNG."""
    if set(bundle.frozen_payload) != {*FROZEN_MODULES, "extra_state"}:
        raise ValueError("Frozen payload must not contain a parameter encoder or optimizer")
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(seed)
        model = build_model(OmegaConf.create(bundle.model_config))
    if model.get_extra_state() != bundle.frozen_payload["extra_state"]:
        raise ValueError("Model extra state differs from source configuration")
    for name in FROZEN_MODULES:
        module = cast(nn.Module, getattr(model, name))
        module.load_state_dict(bundle.frozen_payload[name], strict=True)
    _validate_model(model)
    if frozen_module_hashes(model) != bundle.manifest["frozen_hashes"]:
        raise ValueError("Frozen module fingerprint differs from manifest")
    metadata = bundle.evaluation["metadata"]
    if (
        module_fingerprint(model.online_encoder) != metadata["state_online_sha256"]
        or module_fingerprint(model.target_encoder) != metadata["state_target_sha256"]
    ):
        raise ValueError("Cached states were encoded by different frozen encoders")
    model.requires_grad_(False)
    model.parameter_encoder.requires_grad_(True).train()
    for name in FROZEN_MODULES:
        cast(nn.Module, getattr(model, name)).eval()
    return model.to(device)


def load_source_bundle(source_dir: Path) -> SourceBundle:
    """Verify a self-contained source bundle without consulting original machine paths."""
    root = source_dir.resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest["format_version"] != FORMAT_VERSION or manifest["identity"] != manifest_identity(
        manifest
    ):
        raise ValueError("Source manifest version or identity mismatch")
    if set(manifest["artifacts"]) != set(ARTIFACTS):
        raise ValueError("Source bundle has unexpected artifact entries")
    for filename in ARTIFACTS:
        if sha(root / filename) != manifest["artifacts"][filename]:
            raise ValueError(f"Source artifact hash mismatch: {filename}")
    _validate_objective(manifest["objective_config"])
    _validate_evidence(manifest["all_six_evidence"])
    bundle = SourceBundle(
        root,
        manifest,
        cast(
            dict[str, Any],
            torch.load(root / "frozen_modules.pt", map_location="cpu", weights_only=True),
        ),
        cast(
            TensorSplit, torch.load(root / "train_inputs.pt", map_location="cpu", weights_only=True)
        ),
        cast(
            EvaluationData,
            torch.load(root / "evaluation_data.pt", map_location="cpu", weights_only=True),
        ),
    )
    _validate_caches(bundle)
    build_fresh_model(bundle, seed=0, device=torch.device("cpu"))
    return bundle


def export_source_bundle(
    *,
    checkpoint: Path,
    source_protocol: Path,
    source_evaluation: Path,
    run_config: Path,
    training_cache: Path,
    evaluation_cache: Path,
    output_dir: Path,
    completion: Path | None = None,
    observation_map: Path | None = None,
) -> dict[str, Any]:
    """Export the selected label-assisted positive control without its q encoder or Adam state."""
    if output_dir.exists():
        raise FileExistsError("Export requires a new output directory")
    protocol, evaluation_record = _read_json(source_protocol), _read_json(source_evaluation)
    config = _read_json(run_config)
    completion = completion or checkpoint.parent / "completed.json"
    completed = _read_json(completion)
    observation_map = observation_map or Path(protocol["source"]["observation_map"])
    paths = {
        "checkpoint": checkpoint,
        "protocol": source_protocol,
        "evaluation": source_evaluation,
        "run_config": run_config,
        "training_cache": training_cache,
        "evaluation_cache": evaluation_cache,
        "completion": completion,
        "observation_map": observation_map,
    }
    output = output_dir.resolve()
    if any(
        path.resolve().parent == output or path.resolve().parent in output.parents
        for path in paths.values()
    ):
        raise ValueError("Export outside original source artifact directories")
    hashes = {name: sha(path) for name, path in paths.items()}
    source_hash = protocol["source_sha256"]
    payload = cast(dict[str, Any], torch.load(checkpoint, map_location="cpu", weights_only=False))
    if (
        payload["arm"] != "mlp"
        or payload["stage"] != "washout"
        or payload["source_sha256"] != source_hash
        or protocol["source"]["sha256"] != source_hash
        or evaluation_record["source_sha256"] != source_hash
        or evaluation_record["checkpoint_sha256"] != hashes["checkpoint"]
        or evaluation_record["checkpoint_stage"] != payload["stage"]
        or evaluation_record["checkpoint_step"] != payload["step"]
        or completed["stage"] != payload["stage"]
        or completed["completed_steps"] != payload["step"]
        or completed["last_sha256"] != hashes["checkpoint"]
        or completed["frozen_unchanged"] is not True
        or protocol["input_cache_sha256"] != hashes["training_cache"]
        or protocol["evaluation_cache_sha256"] != hashes["evaluation_cache"]
        or evaluation_record["evaluation_data_sha256"] != hashes["evaluation_cache"]
        or config["run_identity"] != protocol["source"]["run_identity"]
        or config["dataset_identity"] != protocol["source"]["dataset_identity"]
    ):
        raise ValueError(
            "Source checkpoint, completion, evaluation, configuration or cache provenance mismatch"
        )
    _validate_objective(config["train"]["objective"])
    with preserve_rng(torch.device("cpu")):
        model = build_model(cast(DictConfig, OmegaConf.create(config["model"])))
    model.load_state_dict(payload["model"], strict=True)
    _validate_model(model)
    model.eval()
    if payload["freeze_hashes"] != {
        name: legacy_fingerprint(getattr(model, name)) for name in FROZEN_MODULES[:2]
    }:
        raise ValueError("Source frozen encoders do not match original recorded fingerprints")
    frozen = {name: copy.deepcopy(getattr(model, name).state_dict()) for name in FROZEN_MODULES}
    frozen["extra_state"] = model.get_extra_state()
    train_cache = cast(
        dict[str, Any], torch.load(training_cache, map_location="cpu", weights_only=False)
    )
    train = cast(TensorSplit, train_cache["train"])
    evaluation = cast(
        EvaluationData, torch.load(evaluation_cache, map_location="cpu", weights_only=False)
    )
    map_payload = cast(
        dict[str, Tensor], torch.load(observation_map, map_location="cpu", weights_only=True)
    )
    observation = VectorObservationMap(matrix=map_payload["matrix"], dimension=6)
    if (
        observation.fingerprint != protocol["source"]["observation_map_fingerprint"]
        or observation.fingerprint != evaluation["metadata"]["observation_map_fingerprint"]
        or protocol["source"]["generator_version"] != dgp.GENERATOR_VERSION
        or config["data"]["family"] != "simplified_gaussian"
        or config["data"]["episodes_per_parameter"] != 10
        or config["data"]["trajectory_length"] != 12
    ):
        raise ValueError("Source DGP or observation map differs from cache/configuration")
    evaluation["metadata"]["observation_matrix"] = observation.matrix.detach().cpu()
    evaluation["metadata"]["training_labels_absent"] = True
    evidence = {
        "test_r2": evaluation_record["validation_selected_joint_readout"]["test_r2"],
        "per_parameter": evaluation_record["intervention_summary"]["per_parameter"],
        "interpretation": (
            "Existing fixed source: positive held-out recovery and positive counterfactual "
            "benefit for every parameter; not a new checkpoint selection"
        ),
    }
    _validate_evidence(evidence)
    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_config": config["model"],
        "objective_config": config["train"]["objective"],
        "data_config": config["data"],
        "batch_size": config["train"]["batch_size"],
        "train_groups": len(train["online"]),
        "evaluation_groups": {
            name: len(evaluation[name]["theta"]) for name in ("calibration", "validation", "test")
        },
        "observation_map_fingerprint": observation.fingerprint,
        "generator_version": dgp.GENERATOR_VERSION,
        "frozen_hashes": frozen_module_hashes(model),
        "all_six_evidence": evidence,
        "provenance": {
            "original_paths": {name: str(path.resolve()) for name, path in paths.items()},
            "sha256": hashes,
            "source_state_checkpoint_sha256": source_hash,
            "source_stage": payload["stage"],
            "source_step": payload["step"],
            "source_q_sha256": module_fingerprint(model.parameter_encoder),
            "label_assisted_predictor_preparation": True,
        },
        "reference_code_stats": {},
        "reference_code_stats_provenance": {
            "checkpoint_sha256": hashes["checkpoint"],
            "evaluation_cache_sha256": hashes["evaluation_cache"],
            "split": "calibration",
            "use": "Reporting only; no targets or initialization matching",
        },
        "training_labels_absent": True,
        "trained_parameter_encoder_exported": False,
        "optimizer_exported": False,
        "artifacts": {},
    }
    provisional = SourceBundle(output, manifest, frozen, train, evaluation)
    _validate_caches(provisional)
    # Check cache encoder identity before calculating any reference statistics.
    build_fresh_model(provisional, seed=0, device=torch.device("cpu"))
    manifest["reference_code_stats"] = code_reference_stats(
        encode_codes(model, evaluation["calibration"])
    )
    output.mkdir(parents=True)
    torch.save(frozen, output / "frozen_modules.pt")
    torch.save(train, output / "train_inputs.pt")
    torch.save(evaluation, output / "evaluation_data.pt")
    manifest["artifacts"] = {name: sha(output / name) for name in ARTIFACTS}
    manifest["identity"] = manifest_identity(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    load_source_bundle(output)
    return manifest


def main() -> None:
    """Export a self-contained server source from the documented positive-control artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint",
        "source-protocol",
        "source-evaluation",
        "run-config",
        "training-cache",
        "evaluation-cache",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--completion", type=Path)
    parser.add_argument("--observation-map", type=Path)
    arguments = vars(parser.parse_args())
    manifest = export_source_bundle(**arguments)
    print(json.dumps({"exported": str(arguments["output_dir"]), "identity": manifest["identity"]}))


if __name__ == "__main__":
    main()
