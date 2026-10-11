"""Exercise CLI logging lifecycle and checkpoint resume without contacting W&B."""

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from scjepa.data import TensorEpisodes
from scjepa.training import TrainConfig, Trainer
from scjepa.training.factory import ModelConfig


class _TrainingEntrypoint(Protocol):
    def run_training(self, cfg: DictConfig) -> None: ...


def _load_entrypoint() -> _TrainingEntrypoint:
    path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("scjepa_train_entrypoint_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(_TrainingEntrypoint, module)


train = _load_entrypoint()


class FakeLogger:
    """Record the public logger contract without importing the external SDK."""

    instances: ClassVar[list["FakeLogger"]] = []

    def __init__(
        self,
        config: object,
        *,
        resolved_config: Mapping[str, object],
        dataset_identity: str | None,
        out_dir: Path,
        resume_state: Mapping[str, object] | None = None,
    ) -> None:
        """Capture construction data and reuse any checkpoint run metadata."""
        self.config = config
        self.resolved_config = dict(resolved_config)
        self.dataset_identity = dataset_identity
        self.out_dir = out_dir
        self.resume_state = None if resume_state is None else dict(resume_state)
        self.metadata = (
            dict(resume_state)
            if resume_state is not None
            else {"id": "test-run", "entity": "test-entity", "project": "test", "mode": "offline"}
        )
        self.events: list[tuple[int, dict[str, float]]] = []
        self.exits: list[int] = []
        self.instances.append(self)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Capture each emitted scalar event."""
        self.events.append((step, dict(metrics)))

    def finish(self, exit_code: int = 0) -> None:
        """Capture the process result passed to W&B."""
        self.exits.append(exit_code)


@pytest.fixture
def fake_logger(monkeypatch: pytest.MonkeyPatch) -> type[FakeLogger]:
    FakeLogger.instances = []
    monkeypatch.setattr(train, "WandbLogger", FakeLogger)
    return FakeLogger


def tiny_config(tmp_path: Path) -> DictConfig:
    """Use tensor-only data, independently sized latents, and one CPU update."""
    observations = torch.randn(4, 4, 5, generator=torch.Generator().manual_seed(321))
    data_path = tmp_path / "episodes.pt"
    torch.save({"observations": observations, "episode_ids": torch.arange(4)}, data_path)
    return OmegaConf.create(
        {
            "data": None,
            "data_path": str(data_path),
            "resume": None,
            "model": OmegaConf.structured(
                ModelConfig(
                    observation_dim=5,
                    state_dim=3,
                    param_dim=2,
                    context_len=2,
                    history_len=1,
                    encoder_hidden_dims=(8,),
                    parameter_embed_dim=8,
                    parameter_heads=2,
                    max_context_len=4,
                    spartan_layers=1,
                    spartan_embed_dim=8,
                    spartan_mlp_hidden=8,
                    spartan_mlp_layers=2,
                )
            ),
            "train": OmegaConf.structured(
                TrainConfig(steps=1, batch_size=2, out_dir=str(tmp_path / "first"))
            ),
            "wandb": {"mode": "disabled", "project": "test", "entity": "test-entity"},
        }
    )


def read_checkpoint(path: Path) -> dict[str, Any]:
    """Read this test's trusted training checkpoint, including Python RNG state."""
    return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))


def test_entrypoint_logs_metrics_configuration_and_checkpoint_identity(
    tmp_path: Path, fake_logger: type[FakeLogger]
) -> None:
    cfg = tiny_config(tmp_path)
    train.run_training(cfg)

    assert len(fake_logger.instances) == 1
    logger = fake_logger.instances[0]
    assert logger.resume_state is None
    assert logger.dataset_identity == TensorEpisodes.load(str(cfg.data_path)).fingerprint
    assert logger.resolved_config["model"] == OmegaConf.to_container(cfg.model, resolve=True)
    assert logger.resolved_config["train"] == OmegaConf.to_container(cfg.train, resolve=True)
    assert logger.out_dir == tmp_path / "first"
    assert [step for step, _ in logger.events] == [1]
    assert logger.events[0][1]["health/optimizer_steps"] == 1
    assert "loss/prediction" in logger.events[0][1]
    assert logger.exits == [0]

    checkpoint = read_checkpoint(logger.out_dir / "last.pt")
    assert checkpoint["run_metadata"]["wandb"] == logger.metadata
    assert checkpoint["dataset_identity"] == logger.dataset_identity
    assert checkpoint["optimizer_steps"] == 1
    assert (logger.out_dir / "resolved_config.yaml").exists()


def test_resume_passes_checkpoint_run_identity_and_continues_steps_in_new_directory(
    tmp_path: Path, fake_logger: type[FakeLogger]
) -> None:
    cfg = tiny_config(tmp_path)
    train.run_training(cfg)
    initial = fake_logger.instances[0]
    cfg.resume = str(initial.out_dir / "last.pt")
    cfg.train.steps = 2
    cfg.train.out_dir = str(tmp_path / "resumed")
    train.run_training(cfg)

    assert len(fake_logger.instances) == 2
    resumed = fake_logger.instances[1]
    assert resumed.resume_state == initial.metadata
    assert resumed.metadata["id"] == initial.metadata["id"]
    assert resumed.dataset_identity == initial.dataset_identity
    assert [step for step, _ in resumed.events] == [2]
    assert resumed.events[0][1]["health/optimizer_steps"] == 2
    assert resumed.exits == [0]
    checkpoint = read_checkpoint(resumed.out_dir / "last.pt")
    assert checkpoint["run_metadata"]["wandb"] == initial.metadata
    assert checkpoint["step"] == 2


def test_training_failure_finishes_logger_and_releases_distributed_state(
    tmp_path: Path, fake_logger: type[FakeLogger], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tiny_config(tmp_path)
    cleanup: list[bool] = []

    def fail_training(_self: Trainer) -> dict[str, float]:
        raise RuntimeError("test training failure")

    monkeypatch.setattr(Trainer, "train", fail_training)
    monkeypatch.setattr(train, "destroy_distributed", lambda: cleanup.append(True))
    with pytest.raises(RuntimeError, match="test training failure"):
        train.run_training(cfg)
    assert len(fake_logger.instances) == 1
    assert fake_logger.instances[0].exits == [1]
    assert cleanup == [True]


def test_checkpoint_validation_precedes_external_logger_creation(
    tmp_path: Path, fake_logger: type[FakeLogger]
) -> None:
    cfg = tiny_config(tmp_path)
    train.run_training(cfg)
    cfg.resume = str(tmp_path / "first" / "last.pt")
    cfg.train.steps = 2
    torch.save({"observations": torch.zeros(4, 4, 5)}, str(cfg.data_path))
    fake_logger.instances.clear()
    with pytest.raises(ValueError, match="dataset_identity"):
        train.run_training(cfg)
    assert fake_logger.instances == []


def test_non_main_rank_does_not_create_external_run(
    tmp_path: Path, fake_logger: type[FakeLogger], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tiny_config(tmp_path)
    monkeypatch.setattr(train, "is_main_process", lambda: False)
    monkeypatch.setattr("scjepa.training.loop.is_main_process", lambda: False)
    train.run_training(cfg)
    assert fake_logger.instances == []
    assert not (tmp_path / "first" / "last.pt").exists()


def test_non_main_rank_observes_rank_zero_logging_initialization_failure(
    tmp_path: Path, fake_logger: type[FakeLogger], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tiny_config(tmp_path)
    cleanup: list[bool] = []
    local_failure_flags: list[bool] = []

    def remote_failure(value: bool, _device: object) -> bool:
        local_failure_flags.append(value)
        return True

    monkeypatch.setattr(train, "is_main_process", lambda: False)
    monkeypatch.setattr(train, "any_rank", remote_failure)
    monkeypatch.setattr(train, "destroy_distributed", lambda: cleanup.append(True))
    with pytest.raises(RuntimeError, match="logging initialization failed on rank zero"):
        train.run_training(cfg)
    assert local_failure_flags == [False]
    assert fake_logger.instances == []
    assert cleanup == [True]
    assert not (tmp_path / "first" / "last.pt").exists()


def test_metric_logging_failure_finishes_run_as_failed(
    tmp_path: Path, fake_logger: type[FakeLogger], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tiny_config(tmp_path)

    def fail_logging(_self: FakeLogger, _step: int, _metrics: dict[str, float]) -> None:
        raise RuntimeError("test logger failure")

    monkeypatch.setattr(fake_logger, "log", fail_logging)
    with pytest.raises(RuntimeError, match="metric logging failed on rank zero") as caught:
        train.run_training(cfg)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "test logger failure"
    assert fake_logger.instances[0].exits == [1]
