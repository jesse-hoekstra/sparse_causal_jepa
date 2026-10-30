"""Atomic experiment records, integrity checks, and process ownership."""

import fcntl
import json
import math
import os
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast


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
