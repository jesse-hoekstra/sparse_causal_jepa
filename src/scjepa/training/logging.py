"""Optional scalar W&B logging without changing training's random streams."""

import importlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from scjepa.training.rng import preserve_rng as _preserve_rng


@dataclass
class WandbConfig:
    """W&B run selection; credentials remain in the SDK's normal login environment."""

    mode: str = "online"
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list[str])
    job_type: str = "train"

    def __post_init__(self) -> None:
        """Reject unsupported modes and empty explicitly selected identities."""
        if self.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")
        for key in ("project", "entity"):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"wandb.{key} must be a nonempty string or null")


class JsonLogger:
    """Print scalar training metrics as one JSON record per logging event."""

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Write an event; rank selection is managed by the trainer."""
        print(json.dumps({"step": step, **metrics}), flush=True)


class _Artifact(Protocol):
    def add_file(self, local_path: str) -> object: ...


class _Run(Protocol):
    id: str
    project: str
    entity: str

    def define_metric(self, name: str, *, step_metric: str | None = None) -> object: ...

    def log(self, data: dict[str, float | int]) -> None: ...

    def finish(self, *, exit_code: int) -> None: ...

    def log_artifact(self, artifact: _Artifact) -> object: ...


class _SDK(Protocol):
    def init(self, **kwargs: object) -> _Run: ...

    def Artifact(self, name: str, *, type: str) -> _Artifact: ...


def _resume_identity(
    config: WandbConfig, resume_state: dict[str, object] | None
) -> tuple[str, str, str | None]:
    """Resolve a checkpoint's identity before contacting the external service."""
    if not resume_state:
        return uuid.uuid4().hex, config.project or "sparse_causal_jepa", config.entity
    saved_id = resume_state.get("id")
    saved_project = resume_state.get("project")
    saved_entity = resume_state.get("entity")
    saved_mode = resume_state.get("mode")
    if (
        not isinstance(saved_id, str)
        or not saved_id
        or not isinstance(saved_project, str)
        or not saved_project
        or (saved_entity is not None and not isinstance(saved_entity, str))
        or saved_mode not in ("online", "offline")
    ):
        raise ValueError("checkpoint contains an invalid W&B run identity")
    for key, requested, saved in (
        ("project", config.project, saved_project),
        ("entity", config.entity, saved_entity),
    ):
        if requested is not None and requested != saved:
            raise ValueError(f"wandb.{key} conflicts with the checkpoint's W&B run identity")
    return saved_id, saved_project, saved_entity


class WandbLogger:
    """Log JSON and W&B scalars; only rank zero should construct this adapter.

    W&B owns its internal append-only step. ``train/microbatch`` is the plot axis,
    allowing a restored older checkpoint to log its actual training counter.
    Offline continuations create separate local segments with the same logical ID;
    W&B does not support appending to an offline run through ``resume``.
    """

    def __init__(
        self,
        config: WandbConfig,
        *,
        resolved_config: dict[str, object],
        dataset_identity: str | None,
        out_dir: Path,
        resume_state: dict[str, object] | None = None,
    ) -> None:
        """Open a run lazily, respecting checkpoint identity and optional installation."""
        self._json = JsonLogger()
        self._run: _Run | None = None
        self._finished = False
        self._metadata = dict(resume_state or {})
        run_id, project, entity = _resume_identity(config, resume_state)
        if config.mode == "disabled":
            return
        with _preserve_rng():
            try:
                sdk = cast(_SDK, importlib.import_module("wandb"))
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "W&B logging requires the optional dependency: "
                    "pip install -e '.[logging]'; or set wandb.mode=disabled"
                ) from error
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = dict(resolved_config)
            payload["dataset_identity"] = dataset_identity
            kwargs: dict[str, object] = {
                "id": run_id,
                "project": project,
                "entity": entity,
                "mode": config.mode,
                "name": config.name,
                "group": config.group,
                "tags": list(config.tags),
                "job_type": config.job_type,
                "dir": str(out_dir),
                "config": payload,
                "allow_val_change": True,
                "save_code": False,
            }
            if config.mode == "online" and resume_state:
                kwargs["resume"] = "must" if resume_state["mode"] == "online" else "allow"
            run = sdk.init(**kwargs)
            self._run = run
            try:
                run.define_metric("train/microbatch")
                run.define_metric("*", step_metric="train/microbatch")
            except BaseException:
                run.finish(exit_code=1)
                self._finished = True
                raise
            self._metadata = {
                "id": run.id,
                "project": run.project,
                "entity": run.entity or entity,
                "mode": config.mode,
            }

    @property
    def metadata(self) -> dict[str, object]:
        """Return a defensive, credential-free run identity for checkpoint restoration."""
        return dict(self._metadata)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Log scalar objectives and health metrics at a consumed-microbatch counter."""
        if self._finished:
            raise RuntimeError("cannot log to a finished W&B logger")
        self._json.log(step, metrics)
        if self._run is not None:
            with _preserve_rng():
                self._run.log({**metrics, "train/microbatch": step})

    def finish(self, exit_code: int = 0) -> None:
        """Flush once on success or failure; closing an already closed run is harmless."""
        if self._finished:
            return
        self._finished = True
        if self._run is not None:
            with _preserve_rng():
                self._run.finish(exit_code=exit_code)

    def log_artifact(self, path: Path, *, name: str, artifact_type: str = "evaluation") -> None:
        """Publish an explicitly selected result file; never upload weights implicitly."""
        if self._finished:
            raise RuntimeError("cannot log to a finished W&B logger")
        if self._run is None:
            return
        if not path.is_file():
            raise ValueError(f"artifact file does not exist: {path}")
        with _preserve_rng():
            sdk = cast(_SDK, importlib.import_module("wandb"))
            artifact = sdk.Artifact(name, type=artifact_type)
            artifact.add_file(str(path.resolve()))
            self._run.log_artifact(artifact)
