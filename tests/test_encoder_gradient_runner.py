import copy
import sys
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import torch
from test_encoder_gradient_source import write_source_fixture

from diagnostics.encoder_gradient_control import run as runner
from diagnostics.encoder_gradient_control.source import (
    SourceBundle,
    build_source_model,
    load_source,
)
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.multiquery_control.run import frozen_hashes, sha
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig
from scjepa.training.loop import ObjectiveConfig


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def settings(source: Path, output: Path, condition: str = "joint_clip") -> runner.Config:
    return runner.Config(
        source_dir=str(source),
        output_dir=str(output),
        seed=7,
        steps=3,
        conditions=(condition,),
        log_every=1,
        checkpoint_every=1,
        evaluate_every=1,
        gradient_every=1,
    )


def checkpoint(config: runner.Config, condition: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        torch.load(
            Path(config.output_dir) / f"seed-{config.seed}" / condition / "last.pt",
            weights_only=False,
        ),
    )


def replace_evaluation(monkeypatch: pytest.MonkeyPatch, bundle: SourceBundle) -> None:
    expected_frozen = bundle.protocol["frozen_hashes"]

    def evaluate(
        candidate: SCJEPA,
        _data: EvaluationData,
        objective: ObjectiveConfig,
        *,
        prediction_weight: float,
        prediction_weights: tuple[float, ...],
        batch_size: int,
        final: bool,
    ) -> dict[str, Any]:
        assert frozen_hashes(candidate) == expected_frozen
        assert objective.parameter_regularizer_weight == 0.4
        assert objective.parameter_consistency_weight == 1.0
        assert prediction_weight == 1.0
        assert prediction_weights == (1.0,)
        assert batch_size == bundle.batch_size
        # This must not change the next training batch's pairing/projection RNG.
        torch.randn(13)
        return {"prediction_mse": 0.25, "final": final}

    monkeypatch.setattr(runner, "evaluate", evaluate)


def reference_continuation(
    bundle: SourceBundle, steps: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    model, optimizer = build_source_model(bundle, torch.device("cpu"))
    objective = PredictiveSupervisionObjective(
        ObjectiveConfig(**bundle.protocol["objective_config"]), 1.0
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    for step in range(bundle.source_step + 1, bundle.source_step + steps + 1):
        epoch, position = divmod(step - 1, bundle.train_groups // bundle.batch_size)
        indices = list(
            GlobalBatchSampler(bundle.train_groups, bundle.batch_size, bundle.seed, epoch)
        )[position]
        batch = {name: values[indices] for name, values in bundle.train.items()}
        torch.manual_seed(bundle.seed * 100000 + step)  # pyright: ignore[reportUnknownMemberType]
        model.train()
        model.online_encoder.eval()
        model.target_encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        output = cached_multiquery_forward(model, batch, queries=1)
        loss, _ = objective(output)
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())


def test_cli_defaults_to_three_additional_30k_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    invocations: list[tuple[runner.Config, bool]] = []

    def capture(config: runner.Config, *, resume: bool = False) -> dict[str, Any]:
        invocations.append((config, resume))
        return {}

    monkeypatch.setattr(runner, "run", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--source-dir",
            "finished",
            "--output-dir",
            "continuation",
            "--wandb-mode",
            "online",
        ],
    )
    runner.main()
    assert len(invocations) == 1
    config, resume = invocations[0]
    assert not resume
    assert config.steps == 30000
    assert config.conditions == ("joint_clip", "separate_clip", "separate_clip_encoder4")
    assert config.encoder_prediction_scale == 4.0
    assert config.seed == 202610261301
    assert config.wandb_mode == "online"


def test_baseline_exactly_continues_source_model_adam_and_future_random_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = write_source_fixture(tmp_path / "source", steps=2)
    bundle = load_source(source, 7)
    before = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    expected_model, expected_optimizer = reference_continuation(bundle, 3)
    replace_evaluation(monkeypatch, bundle)
    config = settings(source, tmp_path / "continued")
    runner.run(config)
    saved = checkpoint(config, "joint_clip")
    assert saved["step"] == 5
    assert saved["source_step"] == 2
    assert saved["continuation_step"] == 3
    torch.testing.assert_close(saved["model"], expected_model, rtol=0, atol=0)
    torch.testing.assert_close(saved["optimizer"], expected_optimizer, rtol=0, atol=0)
    assert before == {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}


@pytest.mark.parametrize("condition", ["joint_clip", "separate_clip", "separate_clip_encoder4"])
def test_each_arm_resume_matches_uninterrupted_updates_and_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, condition: str
) -> None:
    source = write_source_fixture(tmp_path / "source", steps=2)
    bundle = load_source(source, 7)
    replace_evaluation(monkeypatch, bundle)
    full = settings(source, tmp_path / "full", condition)
    runner.run(full)
    interrupted = replace(full, output_dir=str(tmp_path / "interrupted"))
    calls = 0

    def interrupt(model: SCJEPA, batch: TensorSplit, *, queries: int) -> ModelOutput:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return cached_multiquery_forward(model, batch, queries=queries)

    monkeypatch.setattr(runner, "cached_multiquery_forward", interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        runner.run(interrupted)
    saved = checkpoint(interrupted, condition)
    assert saved["continuation_step"] == 1
    assert saved["step"] == 3
    directory = Path(interrupted.output_dir) / "seed-7" / condition
    (directory / "gradient_history.json").write_text('{"history":"stale after crash"}')
    monkeypatch.setattr(runner, "cached_multiquery_forward", cached_multiquery_forward)
    result = runner.run(interrupted, resume=True)
    recovered = checkpoint(interrupted, condition)
    reference = checkpoint(full, condition)
    assert recovered["continuation_step"] == 3
    assert recovered["step"] == 5
    assert recovered["gradient_history"] == reference["gradient_history"]
    torch.testing.assert_close(recovered["model"], reference["model"], rtol=0, atol=0)
    torch.testing.assert_close(recovered["optimizer"], reference["optimizer"], rtol=0, atol=0)
    assert runner.run(interrupted, resume=True) == result
    (directory / "gradient_history.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact"):
        runner.run(interrupted, resume=True)


class RecordingLogger:
    """Capture W&B identities and scalar events without contacting the service."""

    instances: ClassVar[list["RecordingLogger"]] = []

    def __init__(
        self,
        config: WandbConfig,
        *,
        resolved_config: dict[str, Any],
        dataset_identity: str | None,
        out_dir: Path,
        resume_state: dict[str, object] | None = None,
    ) -> None:
        """Record the run constructor's provenance and resume identity."""
        self.config = config
        self.resolved_config = resolved_config
        self.dataset_identity = dataset_identity
        self.out_dir = out_dir
        self.resume_state = copy.deepcopy(resume_state)
        self.logs: list[tuple[int, dict[str, float]]] = []
        self.finished: list[int] = []
        self.artifacts: list[Path] = []
        self._metadata = dict(resume_state or {}) or {
            "id": f"continuation-{len(self.instances)}",
            "project": config.project or "sparse_causal_jepa",
            "entity": "test-account",
            "mode": config.mode,
        }
        self.instances.append(self)

    @property
    def metadata(self) -> dict[str, object]:
        """Return the run identity that the continuation checkpoint must preserve."""
        return dict(self._metadata)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Record a scalar event on the continuation-step axis."""
        self.logs.append((step, dict(metrics)))

    def finish(self, exit_code: int = 0) -> None:
        """Record success or failure without SDK side effects."""
        self.finished.append(exit_code)

    def log_artifact(self, path: Path, *, name: str, artifact_type: str = "evaluation") -> None:
        """Check that only existing declared result artifacts are published."""
        assert name
        assert artifact_type
        assert path.is_file()
        self.artifacts.append(path)


def test_three_arms_have_fresh_distinct_wandb_runs_and_same_first_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = write_source_fixture(tmp_path / "source", steps=2)
    bundle = load_source(source, 7)
    replace_evaluation(monkeypatch, bundle)
    RecordingLogger.instances.clear()
    monkeypatch.setattr(runner, "WandbLogger", RecordingLogger)
    first_batches: list[TensorSplit] = []
    first_rngs: list[torch.Tensor] = []

    def record(model: SCJEPA, batch: TensorSplit, *, queries: int) -> ModelOutput:
        first_batches.append({name: value.clone() for name, value in batch.items()})
        first_rngs.append(torch.get_rng_state().clone())
        return cached_multiquery_forward(model, batch, queries=queries)

    monkeypatch.setattr(runner, "cached_multiquery_forward", record)
    config = replace(
        settings(source, tmp_path / "continued"),
        steps=1,
        conditions=("joint_clip", "separate_clip", "separate_clip_encoder4"),
        wandb_mode="online",
    )
    result = runner.run(config)
    assert len(result["results"]) == 3
    assert len(RecordingLogger.instances) == 3
    assert all(not logger.resume_state for logger in RecordingLogger.instances)
    assert len({logger.metadata["id"] for logger in RecordingLogger.instances}) == 3
    assert all(logger.finished == [0] for logger in RecordingLogger.instances)
    assert all(any(step == 1 for step, _ in logger.logs) for logger in RecordingLogger.instances)
    assert len(first_batches) == 3
    for batch in first_batches[1:]:
        torch.testing.assert_close(batch, first_batches[0], rtol=0, atol=0)
    assert all(torch.equal(state, first_rngs[0]) for state in first_rngs)
    epoch, position = divmod(bundle.source_step, bundle.train_groups // bundle.batch_size)
    indices = list(GlobalBatchSampler(bundle.train_groups, bundle.batch_size, bundle.seed, epoch))[
        position
    ]
    torch.testing.assert_close(
        first_batches[0],
        {name: values[indices] for name, values in bundle.train.items()},
        rtol=0,
        atol=0,
    )


def test_interruption_resumes_only_the_continuation_wandb_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = write_source_fixture(tmp_path / "source", steps=2)
    bundle = load_source(source, 7)
    replace_evaluation(monkeypatch, bundle)
    RecordingLogger.instances.clear()
    monkeypatch.setattr(runner, "WandbLogger", RecordingLogger)
    calls = 0

    def interrupt(model: SCJEPA, batch: TensorSplit, *, queries: int) -> ModelOutput:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return cached_multiquery_forward(model, batch, queries=queries)

    monkeypatch.setattr(runner, "cached_multiquery_forward", interrupt)
    config = replace(settings(source, tmp_path / "continued"), wandb_mode="online")
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        runner.run(config)
    first = RecordingLogger.instances[0]
    assert not first.resume_state
    assert first.metadata["id"] != bundle.checkpoint["wandb"]["id"]
    assert first.finished == [1]
    saved = checkpoint(config, "joint_clip")
    assert saved["wandb"] == first.metadata
    monkeypatch.setattr(runner, "cached_multiquery_forward", cached_multiquery_forward)
    runner.run(config, resume=True)
    assert len(RecordingLogger.instances) == 2
    resumed = RecordingLogger.instances[1]
    assert resumed.resume_state == first.metadata
    assert resumed.metadata == first.metadata
    assert resumed.finished == [0]
    assert [path.name for path in resumed.artifacts] == [
        "results.json",
        "gradient_history.json",
        "protocol.json",
    ]


def test_resume_rejects_changed_continuation_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = write_source_fixture(tmp_path / "source", steps=2)
    replace_evaluation(monkeypatch, load_source(source, 7))
    config = replace(settings(source, tmp_path / "continued"), steps=1)
    runner.run(config)
    with pytest.raises(ValueError, match="settings|identity|changed|differ"):
        runner.run(replace(config, steps=2), resume=True)


@pytest.mark.parametrize("extra_state", [{"autoregressive": False}, {"context_len": 9}])
def test_resume_rejects_prediction_semantics_changed_in_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra_state: dict[str, Any]
) -> None:
    source = write_source_fixture(tmp_path / "source", steps=2)
    replace_evaluation(monkeypatch, load_source(source, 7))
    calls = 0

    def interrupt(model: SCJEPA, batch: TensorSplit, *, queries: int) -> ModelOutput:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return cached_multiquery_forward(model, batch, queries=queries)

    monkeypatch.setattr(runner, "cached_multiquery_forward", interrupt)
    config = settings(source, tmp_path / "continued")
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        runner.run(config)
    saved = checkpoint(config, "joint_clip")
    saved["model"]["_extra_state"].update(extra_state)
    torch.save(saved, Path(config.output_dir) / "seed-7/joint_clip/last.pt")
    monkeypatch.setattr(runner, "cached_multiquery_forward", cached_multiquery_forward)
    with pytest.raises(ValueError, match="prediction mode changed"):
        runner.run(config, resume=True)
