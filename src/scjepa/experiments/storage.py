"""Atomic experiment records, integrity checks, and process ownership."""

import fcntl
import hashlib
import json
import math
import os
import sys
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.config import ExperimentConfig


def json_safe(value: object) -> object:
    """Represent config infinities explicitly; never emit nonstandard JSON numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): json_safe(item) for key, item in cast(dict[object, object], value).items()
        }
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in cast(list[object] | tuple[object, ...], value)]
    return value


def write_json(path: Path, value: object) -> None:
    """Atomically replace a complete local record after flushing its bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, object]:
    """Read a structured record, rejecting non-mapping roots."""
    value: object = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping at {path}")
    return cast(dict[str, object], value)


@contextmanager
def file_lock(path: Path, *, blocking: bool = False) -> Generator[None, None, None]:
    """Hold an OS-released lock; stale lock files never block a restarted process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError(f"another process owns {path.parent}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def source_fingerprint() -> str:
    """Identify actual source bytes, including uncommitted and vendored code."""
    repository = Path(__file__).resolve().parents[3]
    files = sorted((repository / "src" / "scjepa").rglob("*.py"))
    files += sorted((repository / "scripts").glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(repository)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def claim_execution(config: ExperimentConfig, identity: str, process_count: int) -> None:
    """Fix SIGReg's rank-local sampling topology for every compared run."""
    if process_count < 1:
        raise ValueError("world_size must be positive")
    root = Path(config.output_dir)
    expected = {"experiment_identity": identity, "world_size": process_count}
    with file_lock(root / ".experiment.lock", blocking=True):
        path = root / "execution.json"
        if path.exists():
            if read_json(path) != expected:
                raise ValueError("all experiment runs require the same world_size and identity")
        else:
            for record_path in root.glob("*/seed-*/*/run_config.json"):
                record = read_json(record_path)
                if any(record.get(key) != value for key, value in expected.items()):
                    raise ValueError(
                        "existing experiment run has a different world_size or identity"
                    )
            write_json(path, expected)


def prepare_experiment(config: ExperimentConfig) -> tuple[dict[str, object], Tensor]:
    """Create or verify the fixed scientific definition and authoritative W."""
    root = Path(config.output_dir)
    config_identity = canonical_fingerprint(json_safe(config.definition()))
    code_identity = source_fingerprint()
    manifest_path, matrix_path = root / "experiment.json", root / "observation_map.pt"
    with file_lock(root / ".experiment.lock", blocking=True):
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if (
                manifest.get("format_version") != 1
                or manifest.get("config_fingerprint") != config_identity
                or manifest.get("source_fingerprint") != code_identity
            ):
                raise ValueError("experiment definition/source changed; choose a new output_dir")
            claimed_identity = manifest.get("identity")
            stored_body = {key: value for key, value in manifest.items() if key != "identity"}
            if (
                claimed_identity != canonical_fingerprint(json_safe(stored_body))
                or canonical_fingerprint(manifest.get("definition")) != config_identity
            ):
                raise ValueError("experiment manifest integrity check failed")
            if (
                manifest.get("torch_version") != str(torch.__version__)
                or manifest.get("python_version") != sys.version.split()[0]
            ):
                raise ValueError(
                    "exact experiment resume requires the same Python/PyTorch versions"
                )
            payload: object = torch.load(matrix_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise ValueError("persisted observation map must contain a matrix")
            matrix = cast(dict[str, object], payload).get("matrix")
            if not isinstance(matrix, Tensor):
                raise ValueError("persisted observation matrix is missing")
            observer = VectorObservationMap(matrix=matrix, dimension=config.model.observation_dim)
            if manifest.get("observation_fingerprint") != observer.fingerprint:
                raise ValueError("persisted observation map changed")
            return manifest, observer.matrix
        if matrix_path.exists():
            raise ValueError("observation map exists without its experiment manifest")
        observer = VectorObservationMap(
            seed=config.observation_seed, dimension=config.model.observation_dim
        )
        manifest: dict[str, object] = {
            "format_version": 1,
            "config_fingerprint": config_identity,
            "source_fingerprint": code_identity,
            "observation_fingerprint": observer.fingerprint,
            "definition": config.definition(),
            "torch_version": str(torch.__version__),
            "python_version": sys.version.split()[0],
        }
        manifest["identity"] = canonical_fingerprint(json_safe(manifest))
        temporary = matrix_path.with_suffix(".pt.tmp")
        torch.save({"matrix": observer.matrix}, temporary)
        temporary.replace(matrix_path)
        write_json(manifest_path, manifest)
        return manifest, observer.matrix
