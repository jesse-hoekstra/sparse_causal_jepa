"""Addressed harder-DGP caches with isolated evaluation states and provenance."""

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from diagnostics.harder_oracle import dgp
from diagnostics.oracle_state_learning.core import TRAINING_KEYS
from diagnostics.oracle_state_learning.data import (
    code_identity as original_code_identity,
)
from diagnostics.oracle_state_learning.data import read_json, save_torch, sha, verify_completion
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.datasets import canonical_fingerprint
from scjepa.data.identity import tensor_digest, training_tensors_fingerprint
from scjepa.data.random import episode_id, generator
from scjepa.experiments.storage import file_lock, write_json
from scjepa.training.logging import WandbConfig, WandbLogger

SPLITS = ("train", "calibration", "validation", "test")
type Profile = Literal["exploratory", "smoke"]


@dataclass(frozen=True)
class PrepareConfig:
    """Freeze the candidate, its coefficient and all addressed cohort sizes."""

    output_dir: str
    candidate: str = "innovation"
    transport_chart: Literal["mixed", "physical"] = "mixed"
    rho: float = 0.7
    seed: int = 0
    train_groups: int = 8192
    calibration_groups: int = 512
    validation_groups: int = 256
    test_groups: int = 512
    profile: Profile = "exploratory"
    threads: int = 1
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None
    wandb_group: str = "harder-oracle-screen"

    def __post_init__(self) -> None:
        """Separate exploratory scientific runs from tiny local software fixtures."""
        if self.candidate != "innovation" or not math.isfinite(self.rho) or not 0 < self.rho < 1:
            raise ValueError("Require innovation candidate and finite 0 < rho < 1")
        if self.transport_chart not in {"mixed", "physical"}:
            raise ValueError("Transport chart must be mixed or physical")
        if self.seed != 0 or self.threads < 1 or self.profile not in {"exploratory", "smoke"}:
            raise ValueError("Require data seed zero, positive threads and a declared profile")
        counts = tuple(getattr(self, f"{split}_groups") for split in SPLITS)
        if self.profile == "exploratory" and counts != (8192, 512, 256, 512):
            raise ValueError("Exploratory cohorts are fixed at 8192/512/256/512")
        if self.profile == "smoke" and any(not 2 <= count <= 16 for count in counts):
            raise ValueError("Smoke caches require two to sixteen groups per split")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


@dataclass(frozen=True)
class StateData:
    """Six-state R10/T12 observations, 22 actions and raw supplied theta only."""

    split: str
    tensors: dict[str, Tensor]
    states: Tensor | None
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Reject state labels in training inputs and changed data dimensions."""
        if self.split not in SPLITS or frozenset(self.tensors) != TRAINING_KEYS:
            raise ValueError("Invalid split or unauthorized training input keys")
        groups = self.tensors["observations"].shape[0]
        shapes = {
            "observations": (groups, 10, 12, 6),
            "actions": (groups, 10, 11, 22),
            "oracle_parameters": (groups, 6),
            "episode_ids": (groups,),
        }
        for name, shape in shapes.items():
            value = self.tensors[name]
            dtype = torch.int64 if name == "episode_ids" else torch.float32
            if (
                value.shape != shape
                or value.dtype != dtype
                or value.device.type != "cpu"
                or value.requires_grad
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"Invalid cached tensor {name}")
        if self.split == "train" and self.states is not None:
            raise ValueError("Training must never store physical state labels")
        if self.states is not None and self.states.shape != shapes["observations"]:
            raise ValueError("Evaluation states must align with the observations")

    def __len__(self) -> int:
        """Count independent parameter groups, rather than their trajectories."""
        return len(self.tensors["observations"])

    def batch(self, indices: Tensor, device: torch.device | str) -> dict[str, Tensor]:
        """Expose authorized training inputs without the evaluation reference."""
        return {name: value[indices.cpu()].to(device) for name, value in self.tensors.items()}


def code_identity() -> str:
    """Bind original numerical helpers plus this candidate's complete implementation."""
    digest = hashlib.sha256(original_code_identity().encode())
    for name in ("__init__.py", "dgp.py", "model.py", "data.py", "run.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def build_data(config: PrepareConfig, split: str, observation: VectorObservationMap) -> StateData:
    """Sample independent parameter groups while pairing addresses across rho choices."""
    address = "evaluation_calibration" if split == "calibration" else split
    groups = int(getattr(config, f"{split}_groups"))
    parts: dict[str, list[Tensor]] = {key: [] for key in TRAINING_KEYS}
    states: list[Tensor] = []
    for index in range(groups):
        theta = 0.75 * (
            2
            * torch.rand(
                6,
                dtype=torch.float64,
                generator=generator(config.seed, address, index, "harder_oracle/parameters"),
            )
            - 1
        )
        initial = torch.randn(
            10,
            6,
            dtype=torch.float64,
            generator=generator(config.seed, address, index, "harder_oracle/initial"),
        )
        controls = dgp.sample_controls(
            (10, 11), generator(config.seed, address, index, "harder_oracle/actions")
        )
        actions = dgp.encode_controls(controls)
        physical = dgp.simulate(
            initial, theta, actions, rho=config.rho, transport_chart=config.transport_chart
        )
        parts["observations"].append(observation.forward(physical).float())
        parts["actions"].append(actions.float())
        parts["oracle_parameters"].append(theta.float())
        parts["episode_ids"].append(torch.tensor(episode_id(config.seed, address, index)))
        if split != "train":
            states.append(physical)
    tensors = {name: torch.stack(values) for name, values in parts.items()}
    metadata: dict[str, Any] = {
        "candidate": config.candidate,
        "transport_chart": config.transport_chart,
        "rho": config.rho,
        "split": split,
        "data_seed": config.seed,
        "observation_seed": 1729,
        "training_tensor_fingerprint": training_tensors_fingerprint(tensors),
        "observation_matrix_sha256": tensor_digest(observation.matrix),
        "state_labels": "evaluation only",
    }
    return StateData(split, tensors, torch.stack(states) if states else None, metadata)


def prepare(config: PrepareConfig, *, resume: bool = False) -> dict[str, Any]:
    """Save authenticated input caches; physical states are separate nontraining files."""
    output = Path(config.output_dir).resolve()
    settings = asdict(config)
    for name in (
        "output_dir",
        "threads",
        "wandb_mode",
        "wandb_project",
        "wandb_entity",
        "wandb_group",
    ):
        settings.pop(name)
    protocol: dict[str, Any] = {
        "schema": "harder-oracle-cache-v1",
        "settings": settings,
        "code_identity": code_identity(),
        "torch": str(torch.__version__),
        "dimensions": {"state": 6, "parameter": 6, "action": 22, "R": 10, "T": 12, "K": 10},
    }
    identity = canonical_fingerprint(protocol)
    protocol["identity"] = identity
    previous_threads = torch.get_num_threads()
    logger: WandbLogger | None = None
    torch.set_num_threads(config.threads)
    try:
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
                    name=f"harder-data-{config.transport_chart}-rho{config.rho:g}",
                    job_type="data-preparation",
                ),
                resolved_config=protocol,
                dataset_identity=identity,
                out_dir=output,
            )
            write_json(output / "wandb.json", logger.metadata)
            observation = VectorObservationMap(seed=1729, dimension=6)
            save_torch(output / "observation_map.pt", {"matrix": observation.matrix})
            names = ["protocol.json", "wandb.json", "observation_map.pt"]
            split_records: dict[str, Any] = {}
            for split in SPLITS:
                data = build_data(config, split, observation)
                save_torch(
                    output / f"{split}.pt", {"tensors": data.tensors, "metadata": data.metadata}
                )
                names.append(f"{split}.pt")
                if data.states is not None:
                    save_torch(output / f"{split}_states.pt", {"states": data.states})
                    names.append(f"{split}_states.pt")
                split_records[split] = data.metadata
                logger.log(len(split_records), {f"data/{split}/groups": float(len(data))})
            logger.log(5, {"data/complete": 1.0})
            logger.finish()
            completion: dict[str, Any] = {
                "identity": identity,
                "settings": settings,
                "splits": split_records,
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
    """Authenticate the cache and load test tensors only for explicit final evaluation."""
    required = (
        {"protocol.json", "observation_map.pt", "wandb.json"}
        | {f"{split}.pt" for split in SPLITS}
        | {f"{split}_states.pt" for split in SPLITS if split != "train"}
    )
    completion = verify_completion(directory, required)
    protocol = read_json(directory / "protocol.json")
    identity = canonical_fingerprint(
        {key: value for key, value in protocol.items() if key != "identity"}
    )
    if completion["identity"] != identity or protocol["identity"] != identity:
        raise ValueError("Cache protocol identity differs")
    if (
        protocol["code_identity"] != code_identity()
        or protocol["settings"] != completion["settings"]
    ):
        raise ValueError("Cache numerical source or settings differ")
    if "completion_fingerprint" not in completion:
        raise ValueError("Cache metadata fingerprint missing")
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
