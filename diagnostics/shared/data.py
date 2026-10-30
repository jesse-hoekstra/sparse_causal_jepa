"""Pair an authenticated isotropic cache while changing only observed action coordinates."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from diagnostics.shared import geometry as dgp
from diagnostics.shared import source_data
from diagnostics.shared import source_geometry as source_dgp
from diagnostics.shared.io import read_json, save_torch, sha, source_identity, verify_completion
from diagnostics.shared.source_data import StateData as StateData
from scjepa.data.datasets import canonical_fingerprint
from scjepa.data.identity import tensor_digest, training_tensors_fingerprint
from scjepa.data.random import episode_id, generator
from scjepa.experiments.storage import file_lock, write_json
from scjepa.training.logging import WandbConfig, WandbLogger

SPLITS = source_data.SPLITS
SCHEMA = "balanced-action-oracle-cache-v1"
type Profile = Literal["exploratory", "smoke"]


@dataclass(frozen=True)
class PrepareConfig:
    """Inherit every physical setting and cohort from one completed source cache."""

    output_dir: str
    source_data_dir: str
    profile: Profile = "exploratory"
    threads: int = 1
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None
    wandb_group: str = "balanced-action-oracle-screen"

    def __post_init__(self) -> None:
        """Reject undeclared profiles or invalid logging settings before touching a cache."""
        if self.profile not in {"exploratory", "smoke"} or self.threads < 1:
            raise ValueError("Require a declared profile and positive threads")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def code_identity() -> str:
    """Bind the publication data path and its complete shared numerical dependencies."""
    return source_identity()


def _paired_inputs(directory: Path, split: str, completion: dict[str, Any]) -> StateData:
    payload = cast(dict[str, Any], torch.load(directory / f"{split}.pt", weights_only=True))
    if set(payload) != {"tensors", "metadata"}:
        raise ValueError("Source input cache contains unauthorized evaluation fields")
    original = StateData(split, payload["tensors"], None, payload["metadata"])
    if (
        original.metadata != completion["splits"][split]
        or training_tensors_fingerprint(original.tensors)
        != original.metadata["training_tensor_fingerprint"]
    ):
        raise ValueError("Source cached tensor identity differs")
    seed = int(completion["settings"]["seed"])
    address = "evaluation_calibration" if split == "calibration" else split
    actions: list[Tensor] = []
    for index in range(len(original)):
        controls = source_dgp.sample_controls(
            (10, 11), generator(seed, address, index, "harder_oracle/actions")
        )
        if not torch.equal(
            source_dgp.encode_controls(controls).float(), original.tensors["actions"][index]
        ):
            raise ValueError(f"Regenerated original actions differ in {split} group {index}")
        actions.append(dgp.encode_controls(controls).float())
    tensors = original.tensors | {"actions": torch.stack(actions)}
    metadata = original.metadata | {
        "candidate": "balanced_action_innovation",
        "action_mix": "paley24",
        "source_data_identity": completion["identity"],
        "source_training_tensor_fingerprint": original.metadata["training_tensor_fingerprint"],
        "preserved_tensor_sha256": {
            name: tensor_digest(value)
            for name, value in original.tensors.items()
            if name != "actions"
        },
        "training_tensor_fingerprint": training_tensors_fingerprint(tensors),
        "original_actions_regenerated_bitwise": True,
    }
    return StateData(split, tensors, None, metadata)


def prepare(config: PrepareConfig, *, resume: bool = False) -> dict[str, Any]:
    """Copy physical data exactly and remix authenticated float64 addressed controls."""
    output = Path(config.output_dir).resolve()
    source = Path(config.source_data_dir).resolve()
    source_protocol = read_json(source / "protocol.json")
    if source_protocol.get("schema") != "isotropic-oracle-cache-v1":
        raise ValueError("Source must be a completed isotropic oracle cache")
    if source_protocol.get("torch") != str(torch.__version__):
        raise ValueError("Raw control regeneration requires the source PyTorch version")
    previous_threads = torch.get_num_threads()
    logger: WandbLogger | None = None
    torch.set_num_threads(config.threads)
    try:
        _, source_completion = source_data.load_data(source)
        if source_completion["settings"]["profile"] != config.profile:
            raise ValueError("Source and paired cache profiles differ")
        settings = source_completion["settings"] | {
            "candidate": "balanced_action_innovation",
            "action_mix": "paley24",
        }
        protocol: dict[str, Any] = {
            "schema": SCHEMA,
            "settings": settings,
            "code_identity": code_identity(),
            "torch": str(torch.__version__),
            "dimensions": source_protocol["dimensions"],
            "source_data_identity": source_completion["identity"],
            "source_code_identity": source_data.code_identity(),
            "source_completion_sha256": sha(source / "completion.json"),
            "source_protocol_sha256": sha(source / "protocol.json"),
            "pairing": "physical tensors copied; addressed controls remixed after bitwise check",
        }
        identity = canonical_fingerprint(protocol)
        protocol["identity"] = identity
        with file_lock(output.parent / f".{output.name}.lock"):
            if output.exists() and any(output.iterdir()):
                if resume and read_json(output / "protocol.json") == protocol:
                    return load_data(output)[1]
                raise FileExistsError("Use a fresh output or matching completed cache resume")
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "protocol.json", protocol)
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    group=config.wandb_group,
                    name="balanced-action-data-paley24",
                    job_type="data-preparation",
                ),
                resolved_config=protocol,
                dataset_identity=identity,
                out_dir=output,
            )
            write_json(output / "wandb.json", logger.metadata)
            for original, copied in (
                ("observation_map.pt", "observation_map.pt"),
                ("protocol.json", "source_protocol.json"),
                ("completion.json", "source_completion.json"),
            ):
                shutil.copyfile(source / original, output / copied)
            save_torch(output / "action_map.pt", {"matrix": dgp.action_rotation()})
            names = [
                "protocol.json",
                "wandb.json",
                "observation_map.pt",
                "action_map.pt",
                "source_protocol.json",
                "source_completion.json",
            ]
            split_records: dict[str, Any] = {}
            for split in SPLITS:
                data = _paired_inputs(source, split, source_completion)
                save_torch(
                    output / f"{split}.pt", {"tensors": data.tensors, "metadata": data.metadata}
                )
                names.append(f"{split}.pt")
                if split != "train":
                    name = f"{split}_states.pt"
                    shutil.copyfile(source / name, output / name)
                    names.append(name)
                split_records[split] = data.metadata
                logger.log(len(split_records), {f"data/{split}/groups": float(len(data))})
            if verify_completion(source) != source_completion:
                raise ValueError("Source cache changed during paired preparation")
            logger.log(5, {"data/complete": 1.0, "data/paired_physical_tensors": 1.0})
            logger.finish()
            completion: dict[str, Any] = {
                "identity": identity,
                "settings": settings,
                "splits": split_records,
                "source_data_identity": source_completion["identity"],
                "hashes": {name: sha(output / name) for name in names},
            }
            completion["completion_fingerprint"] = canonical_fingerprint(completion)
            write_json(output / "completion.json", completion)
            return completion
    except BaseException:
        if logger is not None:
            logger.finish(exit_code=1)
        raise
    finally:
        torch.set_num_threads(previous_threads)


def load_data(
    directory: Path, *, include_test: bool = False
) -> tuple[dict[str, StateData], dict[str, Any]]:
    """Authenticate this paired cache; leave test tensors closed until explicit evaluation."""
    protocol = read_json(directory / "protocol.json")
    if protocol.get("schema") != SCHEMA:
        raise ValueError("Require the balanced action cache schema")
    required = (
        {
            "protocol.json",
            "observation_map.pt",
            "action_map.pt",
            "wandb.json",
            "source_protocol.json",
            "source_completion.json",
        }
        | {f"{split}.pt" for split in SPLITS}
        | {f"{split}_states.pt" for split in SPLITS if split != "train"}
    )
    completion = verify_completion(directory, required)
    identity = canonical_fingerprint(
        {key: value for key, value in protocol.items() if key != "identity"}
    )
    if completion["identity"] != identity or protocol["identity"] != identity:
        raise ValueError("Cache protocol identity differs")
    if (
        protocol["code_identity"] != code_identity()
        or protocol["source_code_identity"] != source_data.code_identity()
        or protocol["settings"] != completion["settings"]
    ):
        raise ValueError("Cache numerical source or settings differ")
    if "completion_fingerprint" not in completion:
        raise ValueError("Cache metadata fingerprint missing")
    original = read_json(directory / "source_completion.json")
    source_settings = original["settings"] | {
        "candidate": "balanced_action_innovation",
        "action_mix": "paley24",
    }
    if (
        sha(directory / "source_completion.json") != protocol["source_completion_sha256"]
        or sha(directory / "source_protocol.json") != protocol["source_protocol_sha256"]
        or original["identity"] != protocol["source_data_identity"]
        or completion["source_data_identity"] != original["identity"]
        or completion["settings"] != source_settings
    ):
        raise ValueError("Paired source provenance differs")
    for name in ("observation_map.pt", *(f"{split}_states.pt" for split in SPLITS[1:])):
        if completion["hashes"][name] != original["hashes"][name]:
            raise ValueError("Paired physical reference differs from source")
    action_map = cast(dict[str, Tensor], torch.load(directory / "action_map.pt", weights_only=True))
    if set(action_map) != {"matrix"} or not torch.equal(
        action_map["matrix"], dgp.action_rotation()
    ):
        raise ValueError("Cached Paley action map differs")
    data: dict[str, StateData] = {}
    seen: set[int] = set()
    for split in SPLITS if include_test else SPLITS[:-1]:
        payload = cast(dict[str, Any], torch.load(directory / f"{split}.pt", weights_only=True))
        if set(payload) != {"tensors", "metadata"}:
            raise ValueError("Input cache contains unauthorized evaluation fields")
        states: Tensor | None = None
        if split != "train":
            reference = cast(
                dict[str, Tensor], torch.load(directory / f"{split}_states.pt", weights_only=True)
            )
            if set(reference) != {"states"}:
                raise ValueError("Evaluation reference must contain only states")
            states = reference["states"]
        sample = StateData(split, payload["tensors"], states, payload["metadata"])
        if (
            sample.metadata != completion["splits"][split]
            or training_tensors_fingerprint(sample.tensors)
            != sample.metadata["training_tensor_fingerprint"]
            or sample.metadata["source_data_identity"] != original["identity"]
            or sample.metadata["source_training_tensor_fingerprint"]
            != original["splits"][split]["training_tensor_fingerprint"]
            or any(
                tensor_digest(value) != sample.metadata["preserved_tensor_sha256"][name]
                for name, value in sample.tensors.items()
                if name != "actions"
            )
        ):
            raise ValueError("Cached tensor identity differs")
        address = "evaluation_calibration" if split == "calibration" else split
        expected = torch.tensor(
            [
                episode_id(int(completion["settings"]["seed"]), address, i)
                for i in range(len(sample))
            ]
        )
        expected_ids = {int(value) for value in expected}
        if not torch.equal(sample.tensors["episode_ids"], expected) or seen & expected_ids:
            raise ValueError("Parameter-group addresses differ or overlap")
        seen.update(expected_ids)
        data[split] = sample
    return data, completion
