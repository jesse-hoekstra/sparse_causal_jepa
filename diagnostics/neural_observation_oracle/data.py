"""Change observations alone, authenticate the old map, and reserve fresh test groups."""

import hashlib
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from diagnostics.neural_observation_oracle import map as observation
from diagnostics.oracle_state_learning import data as original
from diagnostics.oracle_state_learning.core import TRAINING_KEYS
from diagnostics.oracle_state_learning.core import LiveStateData as StateData
from diagnostics.oracle_state_learning.data import read_json, save_torch, sha, verify_completion
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.datasets import canonical_fingerprint
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.identity import tensor_digest, training_tensors_fingerprint
from scjepa.data.random import episode_id
from scjepa.data.types import DGPConfig
from scjepa.experiments.oracle import OracleParameterEpisodes
from scjepa.experiments.storage import file_lock, json_safe, write_json
from scjepa.training.logging import WandbConfig, WandbLogger

SPLITS = ("train", "calibration", "validation", "test")
PAIRED_SPLITS = SPLITS[:3]
SCHEMA = "neural-observation-oracle-cache-v1"
CANDIDATE = "neural_observation_original_oracle"
TEST_ROOT_SEED = 20261028
type Profile = Literal["exploratory", "smoke"]


@dataclass(frozen=True)
class PrepareConfig:
    """Use one authenticated original cache and its independently persisted base matrix."""

    output_dir: str
    source_data_dir: str
    base_observation_map_path: str
    profile: Profile = "exploratory"
    threads: int = 1
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None
    wandb_group: str = "neural-observation-oracle-screen"

    def __post_init__(self) -> None:
        """Reject undeclared profiles and missing provenance paths before writing."""
        if self.profile not in {"exploratory", "smoke"} or self.threads < 1:
            raise ValueError("Require a declared profile and positive threads")
        if not self.output_dir or not self.source_data_dir or not self.base_observation_map_path:
            raise ValueError("Output, original cache and persisted base matrix are required")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def code_identity() -> str:
    """Bind the frozen original helpers and only this diagnostic's numerical files."""
    digest = hashlib.sha256(original.code_identity().encode())
    for name in ("__init__.py", "map.py", "data.py", "run.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def _record(data: StateData) -> dict[str, Any]:
    return {
        "config": json_safe(asdict(data.config)),
        "fingerprint": data.fingerprint,
        "metadata": data.metadata,
        "tensor_hashes": {name: tensor_digest(value) for name, value in data.tensors.items()},
        "states_hash": tensor_digest(data.states) if data.states is not None else None,
    }


def _input_payload(path: Path) -> StateData:
    payload = cast(dict[str, Any], torch.load(path, weights_only=True))
    if (
        set(payload) != {"config", "tensors", "states", "fingerprint", "metadata"}
        or payload["states"] is not None
    ):
        raise ValueError("Input artifacts must have the strict state-free oracle payload")
    sample = StateData.from_payload(payload)
    if sample.metadata["training_tensor_fingerprint"] != training_tensors_fingerprint(
        sample.tensors
    ) or sample.fingerprint != canonical_fingerprint(sample.metadata):
        raise ValueError("Cached input tensors or metadata changed")
    return sample


def _state_labels(directory: Path, split: str, sample: StateData) -> StateData:
    payload = cast(
        dict[str, Tensor], torch.load(directory / f"{split}_states.pt", weights_only=True)
    )
    if set(payload) != {"states"}:
        raise ValueError("Evaluation reference must contain only physical states")
    return StateData(
        sample.config, sample.tensors, payload["states"], sample.fingerprint, sample.metadata
    )


def _source_cache(
    directory: Path, profile: Profile
) -> tuple[dict[str, StateData], dict[str, Any], dict[str, Any]]:
    """Authenticate all old bytes but deserialize only train/calibration/validation."""
    required = {"protocol.json", "wandb.json", *(f"{split}.pt" for split in SPLITS)} | {
        f"{split}_states.pt" for split in SPLITS[1:]
    }
    complete = verify_completion(directory, required)
    protocol = read_json(directory / "protocol.json")
    if (
        complete.get("schema") != "oracle-state-learning-cache-v1"
        or protocol.get("schema") != "oracle-state-learning-cache-v1"
        or "completion_fingerprint" not in complete
        or complete["identity"] != protocol["identity"]
        or protocol["identity"]
        != canonical_fingerprint(
            {key: value for key, value in protocol.items() if key != "identity"}
        )
        or protocol["code_identity"] != original.code_identity()
        or protocol["settings"] != complete["settings"]
        or set(complete["splits"]) != set(SPLITS)
    ):
        raise ValueError("Original cache definition, source or integrity identity changed")
    settings = complete["settings"]
    original.PrepareConfig(output_dir=str(directory), **settings)
    if settings["profile"] != ("scientific" if profile == "exploratory" else "smoke"):
        raise ValueError("Original and observation-control cache profiles differ")
    data: dict[str, StateData] = {}
    for split in PAIRED_SPLITS:
        sample = _input_payload(directory / f"{split}.pt")
        if split != "train":
            sample = _state_labels(directory, split, sample)
        if _record(sample) != complete["splits"][split]:
            raise ValueError("Original cache tensor or label record changed")
        address = "evaluation_calibration" if split == "calibration" else split
        expected = DGPConfig(
            family="simplified_gaussian",
            root_seed=0,
            split=address,
            num_episodes=int(settings[f"{split}_groups"]),
            episodes_per_parameter=10,
            trajectory_length=12,
            observation_seed=1729,
        )
        if json_safe(asdict(sample.config)) != json_safe(asdict(expected)):
            raise ValueError("Original data no longer follows the fixed oracle process")
        data[split] = sample
    _check_addresses(data)
    return data, complete, protocol


def _authenticate_base_matrix(path: Path, source: dict[str, StateData]) -> VectorObservationMap:
    """Recover authoritative W from a persisted artifact, never from an unchecked seed."""
    payload = cast(dict[str, Tensor], torch.load(path, weights_only=True))
    if set(payload) != {"matrix"}:
        raise ValueError("Base observation artifact must contain exactly its matrix")
    base = VectorObservationMap(seed=1729, matrix=payload["matrix"], dimension=6)
    for split, sample in source.items():
        restored = build_dgp_dataset(sample.config, observation_matrix=base.matrix)
        if OracleParameterEpisodes(restored).fingerprint != sample.metadata["dataset_fingerprint"]:
            raise ValueError("Persisted base matrix differs from the original dataset identity")
        if split != "train":
            if sample.states is None:
                raise ValueError("Calibration/validation replay requires separate state labels")
            for index in range(len(sample)):
                replay = base.forward(sample.states[index]).float()
                if not torch.equal(replay, sample.tensors["observations"][index]):
                    raise ValueError(
                        "Base-map replay differs from original evaluation observations"
                    )
    return base


def _transform(
    source: StateData,
    network: observation.FrozenObservationMap,
    base: VectorObservationMap,
    source_identity: str,
    *,
    fresh_test: bool = False,
) -> StateData:
    tensors = source.tensors | {
        "observations": network.forward(source.tensors["observations"]).float()
    }
    metadata = source.metadata | {
        "schema": "neural-observation-oracle-raw-data-v1",
        "candidate": CANDIDATE,
        "observation_map_kind": observation.MAP_KIND,
        "observation_map_fingerprint": network.fingerprint,
        "base_observation_matrix_fingerprint": base.fingerprint,
        "source_data_identity": source_identity,
        "source_dataset_fingerprint": source.metadata["dataset_fingerprint"],
        "source_training_tensor_fingerprint": source.metadata["training_tensor_fingerprint"],
        "preserved_tensor_sha256": {
            name: tensor_digest(value)
            for name, value in source.tensors.items()
            if name != "observations"
        },
        "dataset_fingerprint": canonical_fingerprint(
            {"original_dataset": source.metadata["dataset_fingerprint"], "map": network.fingerprint}
        ),
        "training_tensor_fingerprint": training_tensors_fingerprint(tensors),
        "observation_composition": (
            "fixed network applied in float64 to original float32 observations"
        ),
        "fresh_test": fresh_test,
    }
    return StateData(
        source.config, tensors, source.states, canonical_fingerprint(metadata), metadata
    )


def _fresh_test(groups: int, base: VectorObservationMap) -> StateData:
    """Generate the newly reserved address using restored W and original physical equations."""
    config = DGPConfig(
        family="simplified_gaussian",
        root_seed=TEST_ROOT_SEED,
        split="test",
        num_episodes=groups,
        episodes_per_parameter=10,
        trajectory_length=12,
        observation_seed=1729,
    )
    dataset = build_dgp_dataset(config, observation_matrix=base.matrix)
    parts: dict[str, list[Tensor]] = {key: [] for key in TRAINING_KEYS}
    states: list[Tensor] = []
    for index in range(groups):
        episode = dataset.process.generate(index)
        parts["observations"].append(episode.observations)
        parts["actions"].append(episode.actions)
        parts["oracle_parameters"].append(
            episode.ground_truth["parameters"].to(episode.observations)
        )
        parts["episode_ids"].append(torch.tensor(episode.episode_id, dtype=torch.int64))
        states.append(episode.ground_truth["states"].detach().cpu())
    tensors = {key: torch.stack(values) for key, values in parts.items()}
    metadata = {
        "schema": "oracle-state-learning-raw-data-v1",
        "dataset_fingerprint": OracleParameterEpisodes(dataset).fingerprint,
        "training_tensor_fingerprint": training_tensors_fingerprint(tensors),
        "parameter_source": "supplied_raw_ground_truth",
        "physical_states_training": False,
        "views_per_group": 10,
        "context_length": 10,
        "trajectory_length": 12,
    }
    return StateData(
        config, tensors, torch.stack(states), canonical_fingerprint(metadata), metadata
    )


def _check_addresses(data: dict[str, StateData]) -> None:
    seen: set[int] = set()
    for split, sample in data.items():
        address = "evaluation_calibration" if split == "calibration" else split
        seed = TEST_ROOT_SEED if split == "test" else 0
        expected = torch.tensor(
            [episode_id(seed, address, index) for index in range(len(sample))], dtype=torch.int64
        )
        if (
            sample.config.split != address
            or sample.config.root_seed != seed
            or not torch.equal(expected, sample.tensors["episode_ids"])
        ):
            raise ValueError(
                "Cached parameter groups differ from their prescribed stream addresses"
            )
        ids = {int(value) for value in expected}
        if ids & seen:
            raise ValueError("Cache splits share parameter-group addresses")
        seen.update(ids)


def prepare(config: PrepareConfig, *, resume: bool = False) -> dict[str, Any]:
    """Transform paired inputs only, save exact maps, and generate the unopened fresh test."""
    output = Path(config.output_dir).resolve()
    source_dir = Path(config.source_data_dir).resolve()
    base_path = Path(config.base_observation_map_path).resolve()
    if output == source_dir:
        raise ValueError("Observation-control output must be separate from its source")
    old_threads = torch.get_num_threads()
    logger: WandbLogger | None = None
    torch.set_num_threads(config.threads)
    try:
        with file_lock(output.parent / f".{output.name}.lock"):
            if output.exists() and any(output.iterdir()):
                if not resume:
                    raise FileExistsError("Use a fresh output or matching completed cache resume")
                _, complete = load_data(output)
                if (
                    complete["settings"]["profile"] != config.profile
                    or sha(source_dir / "completion.json") != complete["source_completion_sha256"]
                    or sha(source_dir / "protocol.json") != complete["source_protocol_sha256"]
                    or sha(base_path) != complete["base_observation_artifact_sha256"]
                ):
                    raise ValueError("Resume source cache or persisted base map changed")
                # Authenticate source bytes without generating any new neural matrices.
                verify_completion(source_dir)
                return complete
            source, source_complete, source_protocol = _source_cache(source_dir, config.profile)
            base = _authenticate_base_matrix(base_path, source)
            network = observation.FrozenObservationMap()
            settings: dict[str, Any] = {
                "candidate": CANDIDATE,
                "profile": config.profile,
                "seed": 0,
                **{
                    f"{split}_groups": source_complete["settings"][f"{split}_groups"]
                    for split in SPLITS
                },
                "test_root_seed": TEST_ROOT_SEED,
                "observation_map": observation.settings(),
            }
            protocol: dict[str, Any] = {
                "schema": SCHEMA,
                "settings": settings,
                "code_identity": code_identity(),
                "torch": str(torch.__version__),
                "dimensions": {"state": 6, "parameter": 6, "action": 16, "observation": 6},
                "source_data_identity": source_complete["identity"],
                "source_code_identity": source_protocol["code_identity"],
                "source_completion_sha256": sha(source_dir / "completion.json"),
                "source_protocol_sha256": sha(source_dir / "protocol.json"),
                "base_observation_artifact_sha256": sha(base_path),
                "base_observation_matrix_fingerprint": base.fingerprint,
                "observation_map_fingerprint": network.fingerprint,
                "pairing": (
                    "train/calibration/validation copied except observations; "
                    "original test replaced by fresh address"
                ),
                "physical_states_training": False,
                "original_test_deserialized": False,
            }
            identity = canonical_fingerprint(protocol)
            protocol["identity"] = identity
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "protocol.json", protocol)
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    group=config.wandb_group,
                    name="neural-observation-data",
                    job_type="data-preparation",
                ),
                resolved_config=protocol,
                dataset_identity=identity,
                out_dir=output,
            )
            write_json(output / "wandb.json", logger.metadata)
            shutil.copyfile(base_path, output / "observation_map.pt")
            shutil.copyfile(source_dir / "protocol.json", output / "source_protocol.json")
            shutil.copyfile(source_dir / "completion.json", output / "source_completion.json")
            save_torch(output / "neural_observation_map.pt", network.payload())
            names = [
                "protocol.json",
                "wandb.json",
                "observation_map.pt",
                "neural_observation_map.pt",
                "source_protocol.json",
                "source_completion.json",
            ]
            records: dict[str, Any] = {}
            transformed: dict[str, StateData] = {}
            for split in SPLITS:
                sample = (
                    source[split]
                    if split in source
                    else _fresh_test(int(settings["test_groups"]), base)
                )
                data = _transform(
                    sample, network, base, source_complete["identity"], fresh_test=split == "test"
                )
                transformed[split] = data
                payload = data.payload()
                payload["states"] = None
                name = f"{split}.pt"
                save_torch(output / name, payload)
                names.append(name)
                if split != "train":
                    name = f"{split}_states.pt"
                    if split in PAIRED_SPLITS:
                        shutil.copyfile(source_dir / name, output / name)
                    else:
                        save_torch(output / name, {"states": data.states})
                    names.append(name)
                records[split] = _record(data)
                logger.log(len(records), {f"data/{split}/groups": float(len(data))})
            _check_addresses(transformed)
            if (
                verify_completion(source_dir) != source_complete
                or sha(base_path) != protocol["base_observation_artifact_sha256"]
            ):
                raise ValueError("Original cache or persisted base map changed during preparation")
            logger.log(
                5,
                {
                    "data/complete": 1.0,
                    "data/paired_non_observation_inputs": 1.0,
                    "data/fresh_test": 1.0,
                },
            )
            logger.finish()
            complete = protocol | {
                "splits": records,
                "hashes": {name: sha(output / name) for name in names},
                "disjoint_parameter_group_addresses": True,
            }
            complete["completion_fingerprint"] = canonical_fingerprint(complete)
            write_json(output / "completion.json", complete)
            return complete
    except BaseException:
        if logger is not None:
            logger.finish(exit_code=1)
        raise
    finally:
        torch.set_num_threads(old_threads)


def load_data(
    directory: Path, *, include_test: bool = False
) -> tuple[dict[str, StateData], dict[str, Any]]:
    """Authenticate all bytes while keeping fresh test inputs and labels closed by default."""
    required = {
        "protocol.json",
        "wandb.json",
        "observation_map.pt",
        "neural_observation_map.pt",
        "source_protocol.json",
        "source_completion.json",
        *(f"{split}.pt" for split in SPLITS),
        *(f"{split}_states.pt" for split in SPLITS[1:]),
    }
    complete = verify_completion(directory, required)
    protocol = read_json(directory / "protocol.json")
    if (
        complete.get("schema") != SCHEMA
        or protocol.get("schema") != SCHEMA
        or "completion_fingerprint" not in complete
        or complete["identity"] != protocol["identity"]
        or protocol["identity"]
        != canonical_fingerprint(
            {key: value for key, value in protocol.items() if key != "identity"}
        )
        or protocol["code_identity"] != code_identity()
        or any(complete.get(key) != value for key, value in protocol.items())
        or set(complete["splits"]) != set(SPLITS)
    ):
        raise ValueError("Observation cache definition, source or completion changed")
    settings = complete["settings"]
    counts = tuple(settings[f"{split}_groups"] for split in SPLITS)
    if (
        settings.get("candidate") != CANDIDATE
        or settings.get("seed") != 0
        or settings.get("test_root_seed") != TEST_ROOT_SEED
        or settings.get("observation_map") != observation.settings()
        or settings.get("profile") not in {"exploratory", "smoke"}
        or (settings["profile"] == "exploratory" and counts != (8192, 512, 256, 512))
        or (settings["profile"] == "smoke" and any(not 2 <= count <= 16 for count in counts))
    ):
        raise ValueError("Observation cache scientific settings changed")
    previous = read_json(directory / "source_completion.json")
    previous_protocol = read_json(directory / "source_protocol.json")
    if (
        sha(directory / "source_completion.json") != complete["source_completion_sha256"]
        or sha(directory / "source_protocol.json") != complete["source_protocol_sha256"]
        or sha(directory / "observation_map.pt") != complete["base_observation_artifact_sha256"]
        or previous["identity"] != complete["source_data_identity"]
        or previous_protocol["code_identity"] != original.code_identity()
        or previous_protocol["code_identity"] != complete["source_code_identity"]
    ):
        raise ValueError("Copied source provenance or base matrix changed")
    base_payload = cast(
        dict[str, Tensor], torch.load(directory / "observation_map.pt", weights_only=True)
    )
    if set(base_payload) != {"matrix"}:
        raise ValueError("Base observation artifact must contain exactly its matrix")
    base = VectorObservationMap(seed=1729, matrix=base_payload["matrix"], dimension=6)
    network = observation.FrozenObservationMap.from_payload(
        cast(dict[str, Any], torch.load(directory / "neural_observation_map.pt", weights_only=True))
    )
    if (
        network.fingerprint != complete["observation_map_fingerprint"]
        or base.fingerprint != complete["base_observation_matrix_fingerprint"]
    ):
        raise ValueError("Persisted observation map identity changed")
    data: dict[str, StateData] = {}
    for split in SPLITS if include_test else PAIRED_SPLITS:
        sample = _input_payload(directory / f"{split}.pt")
        if split != "train":
            sample = _state_labels(directory, split, sample)
        if (
            _record(sample) != complete["splits"][split]
            or len(sample) != settings[f"{split}_groups"]
        ):
            raise ValueError("Cached tensor, label or group count changed")
        metadata = sample.metadata
        if (
            metadata.get("candidate") != CANDIDATE
            or metadata.get("observation_map_kind") != observation.MAP_KIND
            or metadata.get("observation_map_fingerprint") != network.fingerprint
            or metadata.get("base_observation_matrix_fingerprint") != base.fingerprint
            or metadata.get("source_data_identity") != previous["identity"]
            or metadata.get("fresh_test") != (split == "test")
        ):
            raise ValueError("Cached split observation identity changed")
        restored = build_dgp_dataset(sample.config, observation_matrix=base.matrix)
        source_fingerprint = OracleParameterEpisodes(restored).fingerprint
        if metadata["source_dataset_fingerprint"] != source_fingerprint or metadata[
            "dataset_fingerprint"
        ] != canonical_fingerprint(
            {"original_dataset": source_fingerprint, "map": network.fingerprint}
        ):
            raise ValueError("Cached physical generator or composed observation definition changed")
        preserved = {
            name: tensor_digest(value)
            for name, value in sample.tensors.items()
            if name != "observations"
        }
        if metadata["preserved_tensor_sha256"] != preserved:
            raise ValueError("Cached non-observation inputs changed")
        if split in PAIRED_SPLITS:
            source_record = previous["splits"][split]
            if (
                metadata["source_training_tensor_fingerprint"]
                != source_record["metadata"]["training_tensor_fingerprint"]
                or source_fingerprint != source_record["metadata"]["dataset_fingerprint"]
                or any(
                    preserved[name] != source_record["tensor_hashes"][name] for name in preserved
                )
                or (
                    split != "train"
                    and sha(directory / f"{split}_states.pt")
                    != previous["hashes"][f"{split}_states.pt"]
                )
            ):
                raise ValueError("Paired original inputs or evaluation-state bytes changed")
        data[split] = sample
    _check_addresses(data)
    return data, complete
