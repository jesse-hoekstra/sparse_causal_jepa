"""Pair an authenticated balanced-action cache while changing only the innovation phase."""

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from diagnostics.balanced_action_oracle import data as source_data
from diagnostics.balanced_action_oracle import dgp as source_dgp
from diagnostics.isotropic_oracle.data import StateData as StateData
from diagnostics.oracle_state_learning.data import read_json, save_torch, sha, verify_completion
from diagnostics.rotating_pair_oracle import dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.datasets import canonical_fingerprint
from scjepa.data.identity import tensor_digest, training_tensors_fingerprint
from scjepa.data.random import episode_id, generator
from scjepa.experiments.storage import file_lock, write_json
from scjepa.training.logging import WandbConfig, WandbLogger

SPLITS = source_data.SPLITS
SCHEMA = "rotating-pair-oracle-cache-v1"
type Profile = Literal["exploratory", "smoke"]


@dataclass(frozen=True)
class PrepareConfig:
    """Inherit paired inputs and cohorts; replace only the fixed parameter-pair phase."""

    output_dir: str
    source_data_dir: str
    profile: Profile = "exploratory"
    threads: int = 1
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None
    wandb_group: str = "rotating-pair-oracle-screen"

    def __post_init__(self) -> None:
        """Reject undeclared profiles or invalid logging settings before touching a cache."""
        if self.profile not in {"exploratory", "smoke"} or self.threads < 1:
            raise ValueError("Require a declared profile and positive threads")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def code_identity() -> str:
    """Bind the frozen balanced implementation plus these four numerical adapter files."""
    digest = hashlib.sha256(source_data.code_identity().encode())
    for name in ("__init__.py", "dgp.py", "data.py", "run.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def _settings(original: dict[str, Any]) -> dict[str, Any]:
    return original | {
        "candidate": "rotating_pair_innovation",
        "action_mix": "paley24",
        "phase_formula": "rotating_pair",
        "phase_gain": dgp.PHASE_GAIN,
    }


def _paired_inputs(
    directory: Path,
    split: str,
    completion: dict[str, Any],
    observation: VectorObservationMap,
) -> StateData:
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
    settings = completion["settings"]
    observations: list[Tensor] = []
    initial_states: list[Tensor] = []
    evaluation_states: list[Tensor] = []
    for index in range(len(original)):
        theta = dgp.PARAMETER_BOUND * (
            2
            * torch.rand(
                6,
                dtype=torch.float64,
                generator=generator(seed, address, index, "harder_oracle/parameters"),
            )
            - 1
        )
        initial = torch.randn(
            10,
            6,
            dtype=torch.float64,
            generator=generator(seed, address, index, "harder_oracle/initial"),
        )
        controls = source_dgp.sample_controls(
            (10, 11), generator(seed, address, index, "harder_oracle/actions")
        )
        actions = dgp.encode_controls(controls)
        # Match the source renderer's R10/T12 matrix-multiplication geometry.
        initial_observations = observation.forward(
            initial[:, None].expand(-1, 12, -1).contiguous()
        )[:, 0].float()
        if (
            not torch.equal(theta.float(), original.tensors["oracle_parameters"][index])
            or int(original.tensors["episode_ids"][index]) != episode_id(seed, address, index)
            or not torch.equal(actions.float(), original.tensors["actions"][index])
            or not torch.equal(initial_observations, original.tensors["observations"][index, :, 0])
        ):
            raise ValueError(f"Regenerated paired inputs differ in {split} group {index}")
        physical = dgp.simulate(
            initial,
            theta,
            actions,
            rho=float(settings["rho"]),
            transport_chart=settings["transport_chart"],
            rotation_scale=float(settings["rotation_scale"]),
        )
        observations.append(observation.forward(physical).float())
        initial_states.append(initial)
        if split != "train":
            evaluation_states.append(physical)
    tensors = original.tensors | {"observations": torch.stack(observations)}
    metadata = original.metadata | {
        "candidate": "rotating_pair_innovation",
        "action_mix": "paley24",
        "phase_formula": "rotating_pair",
        "phase_gain": dgp.PHASE_GAIN,
        "source_data_identity": completion["identity"],
        "source_training_tensor_fingerprint": original.metadata["training_tensor_fingerprint"],
        "preserved_tensor_sha256": {
            name: tensor_digest(value)
            for name, value in original.tensors.items()
            if name != "observations"
        },
        "paired_initial_observation_sha256": tensor_digest(
            original.tensors["observations"][:, :, 0]
        ),
        "paired_initial_state_sha256": tensor_digest(torch.stack(initial_states)),
        "training_tensor_fingerprint": training_tensors_fingerprint(tensors),
        "addressed_theta_actions_ids_initial_observations_verified": True,
    }
    return StateData(
        split, tensors, torch.stack(evaluation_states) if evaluation_states else None, metadata
    )


def prepare(config: PrepareConfig, *, resume: bool = False) -> dict[str, Any]:
    """Regenerate trajectories from a fixed new phase after authenticating paired inputs."""
    output = Path(config.output_dir).resolve()
    source = Path(config.source_data_dir).resolve()
    source_protocol = read_json(source / "protocol.json")
    if source_protocol.get("schema") != "balanced-action-oracle-cache-v1":
        raise ValueError("Source must be a completed balanced action oracle cache")
    if source_protocol.get("torch") != str(torch.__version__):
        raise ValueError("Raw control regeneration requires the source PyTorch version")
    previous_threads = torch.get_num_threads()
    logger: WandbLogger | None = None
    torch.set_num_threads(config.threads)
    try:
        _, source_completion = source_data.load_data(source)
        if source_completion["settings"]["profile"] != config.profile:
            raise ValueError("Source and paired cache profiles differ")
        settings = _settings(source_completion["settings"])
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
            "pairing": (
                "same addressed theta, controls, actions, initial states, IDs and observation map; "
                "later trajectories regenerated from the fixed rotating-pair phase"
            ),
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
                    name="rotating-pair-data-paley24",
                    job_type="data-preparation",
                ),
                resolved_config=protocol,
                dataset_identity=identity,
                out_dir=output,
            )
            write_json(output / "wandb.json", logger.metadata)
            for original, copied in (
                ("observation_map.pt", "observation_map.pt"),
                ("action_map.pt", "action_map.pt"),
                ("protocol.json", "source_protocol.json"),
                ("completion.json", "source_completion.json"),
            ):
                shutil.copyfile(source / original, output / copied)
            observation = VectorObservationMap(seed=1729, dimension=6)
            source_map = cast(
                dict[str, Tensor], torch.load(output / "observation_map.pt", weights_only=True)
            )
            if set(source_map) != {"matrix"} or not torch.equal(
                source_map["matrix"], observation.matrix
            ):
                raise ValueError("Paired observation map differs from the fixed renderer")
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
                data = _paired_inputs(source, split, source_completion, observation)
                save_torch(
                    output / f"{split}.pt", {"tensors": data.tensors, "metadata": data.metadata}
                )
                names.append(f"{split}.pt")
                if data.states is not None:
                    name = f"{split}_states.pt"
                    save_torch(output / name, {"states": data.states})
                    names.append(name)
                split_records[split] = data.metadata
                logger.log(len(split_records), {f"data/{split}/groups": float(len(data))})
            if verify_completion(source) != source_completion:
                raise ValueError("Source cache changed during paired preparation")
            logger.log(5, {"data/complete": 1.0, "data/paired_inputs_verified": 1.0})
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
        raise ValueError("Require the rotating pair cache schema")
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
    source_settings = _settings(original["settings"])
    if (
        sha(directory / "source_completion.json") != protocol["source_completion_sha256"]
        or sha(directory / "source_protocol.json") != protocol["source_protocol_sha256"]
        or original["identity"] != protocol["source_data_identity"]
        or completion["source_data_identity"] != original["identity"]
        or completion["settings"] != source_settings
    ):
        raise ValueError("Paired source provenance differs")
    for name in ("observation_map.pt", "action_map.pt"):
        if completion["hashes"][name] != original["hashes"][name]:
            raise ValueError("Paired observation or action map differs from source")
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
                if name != "observations"
            )
            or tensor_digest(sample.tensors["observations"][:, :, 0])
            != sample.metadata["paired_initial_observation_sha256"]
        ):
            raise ValueError("Cached tensor identity differs")
        if states is not None and (
            states.dtype != torch.float64
            or states.device.type != "cpu"
            or states.requires_grad
            or not bool(torch.isfinite(states).all())
            or tensor_digest(states[:, :, 0]) != sample.metadata["paired_initial_state_sha256"]
        ):
            raise ValueError("Evaluation state references or paired initial states differ")
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
