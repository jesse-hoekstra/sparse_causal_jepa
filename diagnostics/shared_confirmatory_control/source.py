"""Authenticate both frozen source/calibrated pairs without loading TRAIN tensors."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from diagnostics.global_code_calibration.objective import compose_translation
from diagnostics.global_code_calibration.source import verify_files
from diagnostics.multiquery_control.data import EvaluationData, module_fingerprint
from diagnostics.multiquery_control.run import sha
from diagnostics.overcomplete_acquisition.model import OvercompletePredictor
from diagnostics.shared_code_calibration.run import code_identity as calibration_code_identity
from diagnostics.shared_code_calibration.source import PINNED as ENDPOINTS
from diagnostics.shared_feature_acquisition.run import code_identity as shared_code_identity
from diagnostics.table_settling_control.source import authenticated_directory, model_fingerprint
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.factory import build_dgp_dataset
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import Spartan
from scjepa.training.factory import build_model

SEEDS = (202610261301, 202610262301)
COMPOSED = (
    "4191898716a4863d31ac584174e48c996f69f19210f0dc3a7457ff0bd1fc3e91",
    "36312bcb965a7fc6109427d5e89f231888b3456d47b14b621d17f52e2a08cdc1",
)


@dataclass(frozen=True)
class Inputs:
    """Four frozen models and prior data solely for probe fitting or identity checks."""

    models: dict[str, SCJEPA]
    old_data: EvaluationData
    precision_data: EvaluationData
    hashes: dict[str, str]
    translations: dict[str, Tensor]
    train_config: dict[str, Any]
    endpoint_results: dict[str, dict[str, Any]]


def load_inputs(
    calibration_dirs: tuple[Path, Path],
    source: Path,
    precision: Path,
    *,
    verification: bool = False,
) -> Inputs:
    """Bind finalized accepted models, exact translations and their common data law."""
    models: dict[str, SCJEPA] = {}
    translations: dict[str, Tensor] = {}
    hashes: dict[str, str] = {}
    endpoints: dict[str, dict[str, Any]] = {}
    common_frozen: dict[str, str] | None = None
    for index, (root, seed) in enumerate(zip(calibration_dirs, SEEDS, strict=True)):
        completed = authenticated_directory(
            root,
            {
                "protocol.json",
                "results.json",
                "composed_model.pt",
                "translation.pt",
                "fit_state.pt",
            },
        )
        protocol = cast(dict[str, Any], read_json(root / "protocol.json"))
        if (
            protocol["identity"] != completed["identity"]
            or protocol["code_identity"] != calibration_code_identity()
        ):
            raise ValueError("Completed calibration identity or implementation changed")
        if not verification and (
            protocol["config"]["seed"] != seed or sha(root / "composed_model.pt") != COMPOSED[index]
        ):
            raise ValueError(
                "Require both predeclared accepted calibration endpoints in seed order"
            )
        if protocol["config"]["verification"] != verification:
            raise ValueError("Scientific and verification calibration sources cannot be mixed")
        for name, expected in completed["hashes"].items():
            hashes[str((root / name).resolve())] = expected
        hashes[str((root / "completed.json").resolve())] = sha(root / "completed.json")
        verify_files(protocol["source_hashes"])
        hashes.update(protocol["source_hashes"])
        for name in ("protocol.json", "evaluation_data.pt"):
            declared = Path(protocol["config"]["source_dir"]) / name
            if protocol["source_hashes"].get(str(declared.resolve())) != sha(source / name):
                raise ValueError("Original data/source differs from the calibrated inputs")
            hashes[str((source / name).resolve())] = sha(source / name)
        declared_precision = Path(protocol["config"]["precision_dir"]) / "validation_data.pt"
        if protocol["source_hashes"].get(str(declared_precision.resolve())) != sha(
            precision / "validation_data.pt"
        ):
            raise ValueError("Prior precision data changed")
        endpoint = Path(protocol["config"]["endpoint_dir"])
        endpoint_protocol = cast(dict[str, Any], read_json(endpoint / "protocol.json"))
        if (
            endpoint_protocol["code_identity"] != shared_code_identity()
            or endpoint_protocol["identity"] != protocol["source_identity"]
        ):
            raise ValueError("Acquisition source identity or implementation differs")
        if not verification and (
            sha(endpoint / "last.pt") != ENDPOINTS[seed] or protocol["evaluation_seed"] != SEEDS[0]
        ):
            raise ValueError("Wrong fixed-final acquisition endpoint or original evaluation seed")
        frozen = endpoint_protocol["frozen_hashes"]
        if common_frozen is not None and frozen != common_frozen:
            raise ValueError("Both sources must use identical frozen state charts")
        common_frozen = frozen
        source_protocol = cast(dict[str, Any], read_json(source / "protocol.json"))
        config = copy.deepcopy(source_protocol["model_config"])
        config["param_dim"] = 12
        with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
            model = build_model(cast(DictConfig, OmegaConf.create(config)))
            model.predictor = cast(Spartan, OvercompletePredictor())
        payload = cast(
            dict[str, Any], torch.load(endpoint / "last.pt", map_location="cpu", weights_only=False)
        )
        if (
            payload["identity"] != endpoint_protocol["identity"]
            or payload["step"] != endpoint_protocol["config"]["steps"]
        ):
            raise ValueError("Source final checkpoint identity or budget differs")
        if not verification and payload["step"] != 100000:
            raise ValueError("Scientific confirmation requires fixed100k final checkpoints")
        model.load_state_dict(payload["model"], strict=True)
        for name, expected in frozen.items():
            if module_fingerprint(getattr(model, name)) != expected:
                raise ValueError("Source frozen chart bytes changed")
        model = model.requires_grad_(False).double().eval()
        composed = cast(
            dict[str, Any],
            torch.load(root / "composed_model.pt", map_location="cpu", weights_only=False),
        )
        delta = cast(
            dict[str, Any],
            torch.load(root / "translation.pt", map_location="cpu", weights_only=False),
        )
        if (
            composed["identity"] != protocol["identity"]
            or delta["identity"] != protocol["identity"]
        ):
            raise ValueError("Composed model or translation belongs to another calibration")
        if composed["source_model_sha256"] != model_fingerprint(model) or delta[
            "source_model_sha256"
        ] != model_fingerprint(model):
            raise ValueError("Calibrated source tensor identity differs")
        candidate = compose_translation(model, delta["translation"])
        restored = copy.deepcopy(model)
        restored.load_state_dict(composed["model"], strict=True)
        if model_fingerprint(candidate) != model_fingerprint(restored) or composed[
            "composed_model_sha256"
        ] != model_fingerprint(restored):
            raise ValueError("Accepted composition is not exactly the declared global translation")
        prefix = f"seed{index + 1}"
        models[prefix + "/source"], models[prefix + "/calibrated"] = model, restored
        translations[prefix] = delta["translation"].detach().cpu()
        endpoints[prefix] = cast(dict[str, Any], read_json(endpoint / "final_test.json"))
    old = cast(
        EvaluationData,
        torch.load(source / "evaluation_data.pt", map_location="cpu", weights_only=False),
    )
    prior = cast(
        EvaluationData,
        torch.load(precision / "validation_data.pt", map_location="cpu", weights_only=False),
    )
    for name in (
        "observation_map_fingerprint",
        "views_per_group",
        "context_length",
        "trajectory_length",
        "state_online_sha256",
        "state_target_sha256",
    ):
        if old["metadata"][name] != prior["metadata"][name]:
            raise ValueError("Prior precision law or state charts differ")
    if (
        not torch.equal(
            old["metadata"]["observation_matrix"], prior["metadata"]["observation_matrix"]
        )
        or prior["metadata"]["generator_version"] != dgp.GENERATOR_VERSION
    ):
        raise ValueError("Prior precision observation map or generator differs")
    if not verification and (
        len(old["calibration"]["theta"]),
        len(old["validation"]["theta"]),
        len(prior["validation"]["theta"]),
        prior["metadata"]["seed"],
    ) != (512, 256, 8192, 202610271901):
        raise ValueError("Require original full calibration/validation and fixed precision sample")
    for name in ("online", "target"):
        if (
            common_frozen is None
            or old["metadata"][f"state_{name}_sha256"] != common_frozen[f"{name}_encoder"]
        ):
            raise ValueError("Original cache and completed models have different state charts")
    source_protocol = cast(dict[str, Any], read_json(source / "protocol.json"))
    if verification and "checkpoint" not in source_protocol.get("source", {}):
        # Historical genuine model fixtures predate the persisted TRAIN identity record.
        # Scientific mode never takes this path; dedicated data tests cover TRAIN overlaps.
        config = {
            "family": "simplified_gaussian",
            "split": "train",
            "root_seed": 0,
            "num_episodes": 0,
            "verification_fixture_missing_train_identity": True,
        }
    else:
        record_path = Path(source_protocol["source"]["checkpoint"]).parent / "run_config.json"
        record = cast(dict[str, Any], read_json(record_path))
        config = dict(
            record["data"], split="train", num_episodes=source_protocol["config"]["train_groups"]
        )
        dataset = build_dgp_dataset(
            OmegaConf.create(config), observation_matrix=old["metadata"]["observation_matrix"]
        )
        if dataset.fingerprint != old["metadata"]["training_dataset_fingerprint"]:
            raise ValueError("TRAIN identity-check configuration differs from archived cache")
        hashes[str(record_path.resolve())] = sha(record_path)
    return Inputs(models, old, prior, hashes, translations, config, endpoints)
