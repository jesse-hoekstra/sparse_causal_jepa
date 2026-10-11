"""W&B integration uses a fake SDK: no login, network, or scientific training."""

import importlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from scjepa.training.logging import JsonLogger, WandbConfig, WandbLogger


def consume_random_streams() -> None:
    random.random()
    np.random.random()  # noqa: NPY002
    torch.rand(1)


class FakeArtifact:
    """Record a named result artifact while exercising logger RNG isolation."""

    def __init__(self, name: str, artifact_type: str) -> None:
        """Create a fake artifact without contacting the service."""
        consume_random_streams()
        self.name = name
        self.artifact_type = artifact_type
        self.files: list[str] = []

    def add_file(self, path: str) -> None:
        """Record the explicitly selected file."""
        consume_random_streams()
        self.files.append(path)


class FakeRun:
    """Minimal run that deliberately consumes the training random streams."""

    def __init__(self, options: dict[str, object]) -> None:
        """Record the service-resolved run identity."""
        self.id = str(options["id"])
        self.project = str(options["project"])
        self.entity = str(options["entity"] or "authenticated-user")
        self.metric_definitions: list[tuple[str, str | None]] = []
        self.history: list[dict[str, float | int]] = []
        self.exits: list[int] = []
        self.artifacts: list[FakeArtifact] = []

    def define_metric(self, name: str, *, step_metric: str | None = None) -> None:
        """Record axis definitions."""
        consume_random_streams()
        self.metric_definitions.append((name, step_metric))

    def log(self, data: dict[str, float | int]) -> None:
        """Record scalar events, with no explicit service step argument."""
        consume_random_streams()
        self.history.append(dict(data))

    def log_artifact(self, artifact: FakeArtifact) -> None:
        """Record a published artifact."""
        consume_random_streams()
        self.artifacts.append(artifact)

    def finish(self, *, exit_code: int) -> None:
        """Record successful/error closure."""
        consume_random_streams()
        self.exits.append(exit_code)


class FakeSDK:
    """Collect init calls without importing the real optional SDK."""

    def __init__(self) -> None:
        """Start with no external runs."""
        self.calls: list[dict[str, object]] = []
        self.runs: list[FakeRun] = []

    def Artifact(self, name: str, *, type: str) -> FakeArtifact:
        """Construct the same minimal interface as the optional SDK."""
        return FakeArtifact(name, type)

    def init(self, **kwargs: object) -> FakeRun:
        """Open one fake run."""
        consume_random_streams()
        self.calls.append(dict(kwargs))
        run = FakeRun(kwargs)
        self.runs.append(run)
        return run


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> FakeSDK:
    fake = FakeSDK()

    def fake_import(name: str) -> FakeSDK:
        assert name == "wandb"
        consume_random_streams()
        return fake

    monkeypatch.setattr(importlib, "import_module", fake_import)
    return fake


def make_logger(
    tmp_path: Path,
    *,
    config: WandbConfig | None = None,
    resume_state: dict[str, object] | None = None,
) -> WandbLogger:
    return WandbLogger(
        config or WandbConfig(),
        resolved_config={"model": {"state_dim": 3}, "train": {"steps": 4}},
        dataset_identity="dataset-fingerprint",
        out_dir=tmp_path,
        resume_state=resume_state,
    )


def test_json_logger_outputs_scalar_record(capsys: pytest.CaptureFixture[str]) -> None:
    JsonLogger().log(3, {"loss/total": 0.25})
    assert json.loads(capsys.readouterr().out) == {"step": 3, "loss/total": 0.25}


def test_new_run_configuration_metrics_and_finish(
    tmp_path: Path, sdk: FakeSDK, capsys: pytest.CaptureFixture[str]
) -> None:
    config = WandbConfig(
        project="research", entity="team", name="pilot", group="comparison", tags=["cpu"]
    )
    logger = make_logger(tmp_path, config=config)
    call = sdk.calls[0]
    assert call["mode"] == "online"
    assert call["project"] == "research"
    assert call["entity"] == "team"
    assert call["name"] == "pilot"
    assert call["group"] == "comparison"
    assert call["tags"] == ["cpu"]
    assert call["job_type"] == "train"
    assert call["config"] == {
        "model": {"state_dim": 3},
        "train": {"steps": 4},
        "dataset_identity": "dataset-fingerprint",
    }
    assert call["allow_val_change"] is True
    assert call["save_code"] is False
    assert call["dir"] == str(tmp_path)
    assert "resume" not in call
    assert sdk.runs[0].metric_definitions == [
        ("train/microbatch", None),
        ("*", "train/microbatch"),
    ]
    metrics = {"loss/total": 1.25, "health/optimizer_steps": 2.0, "health/skipped": 0.0}
    logger.log(4, metrics)
    assert sdk.runs[0].history == [{**metrics, "train/microbatch": 4}]
    assert json.loads(capsys.readouterr().out) == {"step": 4, **metrics}
    assert "train/microbatch" not in metrics
    metadata = logger.metadata
    assert metadata == {"id": call["id"], "project": "research", "entity": "team", "mode": "online"}
    metadata["id"] = "unrelated"
    assert logger.metadata["id"] == call["id"]
    logger.finish(exit_code=1)
    logger.finish()
    assert sdk.runs[0].exits == [1]
    with pytest.raises(RuntimeError, match="finished"):
        logger.log(5, metrics)


@pytest.mark.parametrize(
    ("saved_mode", "requested_mode", "resume"),
    [("online", "online", "must"), ("offline", "online", "allow"), ("offline", "offline", None)],
)
def test_resume_reuses_identity_and_custom_axis(
    tmp_path: Path, sdk: FakeSDK, saved_mode: str, requested_mode: str, resume: str | None
) -> None:
    saved: dict[str, object] = {
        "id": "known-run",
        "project": "existing",
        "entity": "team",
        "mode": saved_mode,
    }
    logger = make_logger(tmp_path, config=WandbConfig(mode=requested_mode), resume_state=saved)
    call = sdk.calls[0]
    assert call["id"] == "known-run"
    assert call["project"] == "existing"
    assert call["entity"] == "team"
    assert call.get("resume") == resume
    logger.log(20, {"health/optimizer_steps": 10.0})
    logger.log(15, {"health/optimizer_steps": 7.0})
    assert [event["train/microbatch"] for event in sdk.runs[0].history] == [20, 15]
    logger.finish()


@pytest.mark.parametrize("field", ["project", "entity"])
def test_resume_rejects_destination_conflict_before_sdk_import(
    tmp_path: Path, sdk: FakeSDK, field: str
) -> None:
    saved: dict[str, object] = {
        "id": "run",
        "project": "existing",
        "entity": "team",
        "mode": "online",
    }
    config = WandbConfig()
    setattr(config, field, "conflict")
    with pytest.raises(ValueError, match=f"wandb.{field} conflicts"):
        make_logger(tmp_path, config=config, resume_state=saved)
    assert not sdk.calls


@pytest.mark.parametrize(
    "saved", [{"id": "x"}, {"id": 12}, {"id": "x", "project": "p", "mode": {}}]
)
def test_resume_rejects_malformed_identity(
    tmp_path: Path, sdk: FakeSDK, saved: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="invalid W&B run identity"):
        make_logger(tmp_path, resume_state=saved)
    assert not sdk.calls


def test_disabled_does_not_import_or_create_run_and_retains_saved_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden_import(name: str) -> None:
        pytest.fail(f"disabled logger tried importing {name}")

    monkeypatch.setattr(importlib, "import_module", forbidden_import)
    saved: dict[str, object] = {
        "id": "run",
        "project": "existing",
        "entity": "team",
        "mode": "online",
    }
    logger = make_logger(
        tmp_path / "absent", config=WandbConfig(mode="disabled"), resume_state=saved
    )
    logger.log(2, {"loss/total": 0.5})
    logger.log_artifact(tmp_path / "absent-result.json", name="disabled-result")
    logger.finish()
    assert logger.metadata == saved
    assert not (tmp_path / "absent").exists()
    assert json.loads(capsys.readouterr().out) == {"step": 2, "loss/total": 0.5}


def test_missing_optional_sdk_has_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing_import(name: str) -> None:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing_import)
    with pytest.raises(RuntimeError, match=r"pip install -e '\.\[logging\]'"):
        make_logger(tmp_path)


def test_sdk_calls_preserve_python_numpy_torch_random_streams(tmp_path: Path, sdk: FakeSDK) -> None:
    python_state = random.getstate()
    numpy_state = pickle.dumps(np.random.get_state())  # noqa: NPY002
    torch_state = torch.get_rng_state().clone()
    logger = make_logger(tmp_path)
    logger.log(1, {"loss/total": 2.0})
    logger.finish()
    assert len(sdk.calls) == 1
    assert random.getstate() == python_state
    assert pickle.dumps(np.random.get_state()) == numpy_state  # noqa: NPY002
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)


def test_config_rejects_invalid_modes_and_explicit_blank_identity() -> None:
    with pytest.raises(ValueError, match="mode must"):
        WandbConfig(mode="typo")
    with pytest.raises(ValueError, match="project must"):
        WandbConfig(project="")
    with pytest.raises(ValueError, match="entity must"):
        WandbConfig(entity=" ")


def test_import_failure_restores_rng_even_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python_state = random.getstate()
    numpy_state = pickle.dumps(np.random.get_state())  # noqa: NPY002
    torch_state = torch.get_rng_state().clone()

    def failing_import(name: str) -> None:
        consume_random_streams()
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", failing_import)
    with pytest.raises(RuntimeError, match="optional dependency"):
        make_logger(tmp_path)
    assert random.getstate() == python_state
    assert pickle.dumps(np.random.get_state()) == numpy_state  # noqa: NPY002
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)


def test_initialized_cuda_rng_is_preserved_without_requiring_hardware(
    tmp_path: Path, sdk: FakeSDK, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = torch.tensor([1, 2], dtype=torch.uint8)
    restored: list[tuple[torch.Tensor, int]] = []

    def read_state(device: int) -> torch.Tensor:
        assert device == 3
        return state

    def restore_state(value: torch.Tensor, device: int) -> None:
        restored.append((value, device))

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "get_rng_state", read_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", restore_state)
    logger = make_logger(tmp_path)
    logger.log(1, {"loss/total": 2.0})
    logger.finish()
    assert len(sdk.calls) == 1
    assert len(restored) == 3
    for actual, device in restored:
        assert device == 3
        torch.testing.assert_close(actual, state, rtol=0, atol=0)


def test_log_artifact_publishes_only_selected_file_and_preserves_rng(
    tmp_path: Path, sdk: FakeSDK
) -> None:
    logger = make_logger(tmp_path)
    chosen = tmp_path / "metrics.json"
    chosen.write_text('{"score": 0.5}')
    (tmp_path / "private-checkpoint.pt").write_bytes(b"must not be uploaded")
    python_state = random.getstate()
    numpy_state = pickle.dumps(np.random.get_state())  # noqa: NPY002
    torch_state = torch.get_rng_state().clone()
    logger.log_artifact(chosen, name="calibrated-results")
    assert len(sdk.runs[0].artifacts) == 1
    artifact = sdk.runs[0].artifacts[0]
    assert artifact.name == "calibrated-results"
    assert artifact.artifact_type == "evaluation"
    assert artifact.files == [str(chosen.resolve())]
    assert random.getstate() == python_state
    assert pickle.dumps(np.random.get_state()) == numpy_state  # noqa: NPY002
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    logger.log_artifact(chosen, name="run-summary", artifact_type="summary")
    assert sdk.runs[0].artifacts[1].artifact_type == "summary"
    logger.finish()


def test_log_artifact_rejects_missing_files_directories_and_finished_run(
    tmp_path: Path, sdk: FakeSDK
) -> None:
    logger = make_logger(tmp_path)
    for invalid in (tmp_path / "missing.json", tmp_path):
        with pytest.raises(ValueError, match="artifact file does not exist"):
            logger.log_artifact(invalid, name="invalid")
    assert not sdk.runs[0].artifacts
    logger.finish()
    with pytest.raises(RuntimeError, match="finished"):
        logger.log_artifact(tmp_path / "missing.json", name="too-late")


def test_artifact_failure_preserves_rng_and_leaves_logger_closable(
    tmp_path: Path, sdk: FakeSDK, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = make_logger(tmp_path)
    chosen = tmp_path / "metrics.json"
    chosen.write_text("{}")

    def fail_upload(artifact: FakeArtifact) -> None:
        assert artifact.files == [str(chosen.resolve())]
        consume_random_streams()
        raise OSError("simulated artifact upload failure")

    monkeypatch.setattr(sdk.runs[0], "log_artifact", fail_upload)
    python_state = random.getstate()
    numpy_state = pickle.dumps(np.random.get_state())  # noqa: NPY002
    torch_state = torch.get_rng_state().clone()
    with pytest.raises(OSError, match="artifact upload failure"):
        logger.log_artifact(chosen, name="interrupted")
    assert random.getstate() == python_state
    assert pickle.dumps(np.random.get_state()) == numpy_state  # noqa: NPY002
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    logger.finish(exit_code=1)
    assert sdk.runs[0].exits == [1]
