"""Atomic artifacts and strict authentication of completed experiments."""

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any, cast

import torch

from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import read_json as native_read_json


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


def source_identity() -> str:
    """Hash all shared and production numerical code, including initialization dependencies."""
    root = Path(__file__).resolve().parents[2]
    paths = sorted((*Path(__file__).parent.glob("*.py"), *(root / "src").rglob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
