"""Authenticate immutable raw-observation caches with separate evaluation labels."""

import hashlib
import os
import sys
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from diagnostics.oracle_state_learning.core import LiveStateData, build_data
from scjepa.data.datasets import canonical_fingerprint
from scjepa.data.identity import tensor_digest, training_tensors_fingerprint
from scjepa.data.random import episode_id
from scjepa.experiments.storage import file_lock, json_safe, source_fingerprint, write_json
from scjepa.experiments.storage import read_json as native_read_json
from scjepa.training.logging import WandbConfig, WandbLogger

SPLITS = ("train", "calibration", "validation", "test")


@dataclass(frozen=True)
class PrepareConfig:
    """Fix the original addressed process; permit only tiny explicit software fixtures."""

    output_dir: str
    train_groups: int = 8192
    calibration_groups: int = 512
    validation_groups: int = 256
    test_groups: int = 512
    seed: int = 0
    threads: int = 1
    profile: Literal["scientific", "smoke"] = "scientific"
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None
    wandb_group: str | None = None

    def __post_init__(self) -> None:
        """Reject scientific cohort changes and accidental large local smoke caches."""
        counts = (
            self.train_groups,
            self.calibration_groups,
            self.validation_groups,
            self.test_groups,
        )
        if self.seed != 0 or self.threads < 1:
            raise ValueError("Require original data seed 0 and positive threads")
        if self.profile not in {"scientific", "smoke"}:
            raise ValueError("Unknown preparation profile")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)
        if self.profile == "smoke":
            if any(not 2 <= count <= 16 for count in counts):
                raise ValueError("Verification permits two to sixteen groups per split")
        elif counts != (8192, 512, 256, 512):
            raise ValueError("Scientific cache requires original 8192/512/256/512 groups")


def read_json(path: Path) -> dict[str, Any]:
    """Read an authenticated diagnostic mapping using the repository JSON reader."""
    return cast(dict[str, Any], native_read_json(path))


def sha(path: Path) -> str:
    """Hash file bytes without loading a potentially large cache into working memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_torch(path: Path, payload: object) -> None:
    """Atomically write one CPU cache or checkpoint, preserving complete prior files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def code_identity() -> str:
    """Bind the active diagnostic, model adapter, sampler and production dependencies."""
    root = Path(__file__).resolve().parents[2]
    paths = [
        *(
            Path(__file__).with_name(name)
            for name in ("__init__.py", "core.py", "data.py", "run.py")
        ),
        root / "diagnostics/mlp_oracle_code_control/model.py",
        root / "diagnostics/oracle_state_predictor/run.py",
    ]
    digest = hashlib.sha256(source_fingerprint().encode())
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_completion(directory: Path, required: set[str] | None = None) -> dict[str, Any]:
    """Authenticate listed artifacts and reject missing required files or path escapes."""
    completion = read_json(directory / "completion.json")
    hashes: object = completion.get("hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Completion requires authenticated artifact hashes")
    records = cast(dict[str, object], hashes)
    if required is not None and not required <= records.keys():
        raise ValueError("Completion is missing required artifact hashes")
    for name, expected in records.items():
        path = directory / name
        if path.resolve().parent != directory.resolve():
            raise ValueError("Completion artifact paths must be direct children")
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Completed artifact changed: {path}")
    if "completion_fingerprint" in completion and completion["completion_fingerprint"] != (
        canonical_fingerprint(
            {key: value for key, value in completion.items() if key != "completion_fingerprint"}
        )
    ):
        raise ValueError("Completion metadata changed")
    return completion


def _split_record(data: LiveStateData) -> dict[str, Any]:
    return {
        "config": json_safe(asdict(data.config)),
        "fingerprint": data.fingerprint,
        "metadata": data.metadata,
        "tensor_hashes": {key: tensor_digest(value) for key, value in data.tensors.items()},
        "states_hash": tensor_digest(data.states) if data.states is not None else None,
    }


def _check_addresses(data: dict[str, LiveStateData]) -> None:
    seen: set[int] = set()
    for split, sample in data.items():
        expected_split = "evaluation_calibration" if split == "calibration" else split
        if sample.config.split != expected_split:
            raise ValueError("Cached split differs from its original process address")
        expected = torch.tensor(
            [
                episode_id(sample.config.root_seed, expected_split, index)
                for index in range(len(sample))
            ],
            dtype=torch.int64,
        )
        if not torch.equal(sample.tensors["episode_ids"], expected):
            raise ValueError("Cache episode IDs differ from original process addresses")
        ids = {int(value) for value in expected}
        if ids & seen:
            raise ValueError("Cache splits share parameter-group addresses")
        seen.update(ids)


def prepare(
    config: PrepareConfig,
    *,
    resume: bool = False,
    progress: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    """Generate original raw observations once; train artifacts never contain true states."""
    directory = Path(config.output_dir)
    settings = asdict(config)
    settings.pop("output_dir")
    settings.pop("threads")
    for name in ("wandb_mode", "wandb_project", "wandb_entity", "wandb_group"):
        settings.pop(name)
    evidence: dict[str, Any] = {
        "schema": "oracle-state-learning-cache-v1",
        "settings": settings,
        "code_identity": code_identity(),
        "torch": str(torch.__version__),
        "python": sys.version.split()[0],
        "state_labels": "Separate evaluation reference files only; never stored for train",
        "protocol": "Original addressed simplified_gaussian, R10/K10/T12, observation seed 1729",
    }
    identity = canonical_fingerprint(evidence)
    protocol: dict[str, Any] = {"identity": identity, **evidence}
    logger: WandbLogger | None = None
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        with file_lock(directory / ".lock"):
            if (directory / "protocol.json").exists():
                if not resume or read_json(directory / "protocol.json") != protocol:
                    raise ValueError(
                        "Matching explicit cache resume required; settings/code changed"
                    )
                if (directory / "completion.json").exists():
                    return load_data(directory)[1]
                raise ValueError(
                    "Incomplete preparation is immutable; choose a fresh output directory"
                )
            if any(path.name != ".lock" for path in directory.iterdir()):
                raise FileExistsError("Use a fresh cache output directory")
            write_json(directory / "protocol.json", protocol)
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    group=config.wandb_group,
                    name="oracle-state-data",
                    job_type="data-preparation",
                ),
                resolved_config=protocol,
                dataset_identity=identity,
                out_dir=directory,
            )
            write_json(directory / "wandb.json", logger.metadata)
            hashes = {
                "protocol.json": sha(directory / "protocol.json"),
                "wandb.json": sha(directory / "wandb.json"),
            }
            records: dict[str, Any] = {}
            data: dict[str, LiveStateData] = {}
            for split in SPLITS:
                sample = build_data(split, getattr(config, f"{split}_groups"), seed=config.seed)
                data[split] = sample
                payload = sample.payload()
                payload["states"] = None
                name = f"{split}.pt"
                save_torch(directory / name, payload)
                hashes[name] = sha(directory / name)
                if sample.states is not None:
                    name = f"{split}_states.pt"
                    save_torch(directory / name, {"states": sample.states})
                    hashes[name] = sha(directory / name)
                records[split] = _split_record(sample)
                logger.log(len(records), {f"data/{split}/groups": float(len(sample))})
                if progress is not None:
                    progress(split, len(sample))
            _check_addresses(data)
            completion: dict[str, Any] = {
                "identity": identity,
                "schema": evidence["schema"],
                "settings": settings,
                "splits": records,
                "hashes": hashes,
                "disjoint_parameter_group_addresses": True,
            }
            completion["completion_fingerprint"] = canonical_fingerprint(completion)
            logger.log(len(records) + 1, {"data/complete": 1.0})
            logger.finish()
            write_json(directory / "completion.json", completion)
            return completion
    except BaseException:
        if logger is not None:
            logger.finish(exit_code=1)
        raise
    finally:
        torch.set_num_threads(old_threads)


def load_data(directory: Path) -> tuple[dict[str, LiveStateData], dict[str, Any]]:
    """Verify every byte and tensor identity before exposing strict training batches."""
    required = {"protocol.json", "wandb.json", *(f"{split}.pt" for split in SPLITS)} | {
        f"{split}_states.pt" for split in SPLITS if split != "train"
    }
    completion = verify_completion(directory, required)
    if "completion_fingerprint" not in completion:
        raise ValueError("Cache completion metadata requires an integrity fingerprint")
    protocol = read_json(directory / "protocol.json")
    evidence = {key: value for key, value in protocol.items() if key != "identity"}
    if completion["identity"] != protocol["identity"] or protocol[
        "identity"
    ] != canonical_fingerprint(evidence):
        raise ValueError("Cache protocol identity changed")
    if protocol["code_identity"] != code_identity():
        raise ValueError("Cache preparation source changed")
    if completion["settings"] != protocol["settings"]:
        raise ValueError("Cache settings changed")
    if set(completion["splits"]) != set(SPLITS):
        raise ValueError("Cache must have exactly four isolated splits")
    data: dict[str, LiveStateData] = {}
    for split in SPLITS:
        payload = cast(dict[str, Any], torch.load(directory / f"{split}.pt", weights_only=True))
        if payload.get("states") is not None:
            raise ValueError("Input cache must not contain evaluation state labels")
        if split != "train":
            reference = cast(
                dict[str, Tensor], torch.load(directory / f"{split}_states.pt", weights_only=True)
            )
            if set(reference) != {"states"}:
                raise ValueError("Evaluation reference must contain only physical states")
            payload["states"] = reference["states"]
        sample = LiveStateData.from_payload(payload)
        if _split_record(sample) != completion["splits"][split]:
            raise ValueError("Cached tensor or metadata identity changed")
        if (
            training_tensors_fingerprint(sample.tensors)
            != sample.metadata["training_tensor_fingerprint"]
        ):
            raise ValueError("Cached training fingerprint changed")
        if canonical_fingerprint(sample.metadata) != sample.fingerprint:
            raise ValueError("Cached data fingerprint changed")
        data[split] = sample
    _check_addresses(data)
    return data, completion
