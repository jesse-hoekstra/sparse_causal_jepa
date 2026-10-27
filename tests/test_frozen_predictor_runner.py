"""Scientific and restart invariants for fresh inference through a fixed predictor."""

import copy
import sys
from collections.abc import Generator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import torch
from test_frozen_predictor_source import write_source_fixture

from diagnostics.frozen_predictor_control import run as runner
from diagnostics.frozen_predictor_control.source import (
    SourceBundle,
    build_fresh_model,
    frozen_module_hashes,
    load_source_bundle,
)
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.multiquery_control.run import sha
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig
from scjepa.training.loop import ObjectiveConfig, TrainingObjective


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def settings(source: Path, output: Path) -> runner.Config:
    return runner.Config(
        source_dir=str(source),
        output_dir=str(output),
        seed=7,
        steps=3,
        batch_size=2,
        log_every=1,
        checkpoint_every=1,
        evaluate_every=1,
        gradient_every=1,
    )


def checkpoint(config: runner.Config) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        torch.load(Path(config.output_dir) / f"seed-{config.seed}" / "last.pt", weights_only=False),
    )


def replace_evaluation(monkeypatch: pytest.MonkeyPatch, bundle: SourceBundle) -> None:
    expected_frozen = frozen_module_hashes(build_fresh_model(bundle, 7, torch.device("cpu")))

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
        assert frozen_module_hashes(candidate) == expected_frozen
        assert objective.parameter_regularizer_weight == 0.4
        assert objective.parameter_consistency_weight == 1.0
        assert prediction_weight == 1.0
        assert prediction_weights == (1.0,)
        assert batch_size == 2
        # Evaluation is allowed labels, but its random draws must not affect updates.
        torch.randn(13)
        return {"prediction_mse": 0.25, "final": final}

    def sensitivity(
        candidate: SCJEPA,
        batch: Mapping[str, torch.Tensor],
        *,
        reference_stats: Mapping[str, Any] | None = None,
        max_groups: int = 16,
    ) -> dict[str, Any]:
        assert frozen_module_hashes(candidate) == expected_frozen
        assert set(batch) == {"online", "target", "actions"}
        assert max_groups > 0
        del reference_stats
        return {"groups": len(batch["online"]), "prediction_mse": 0.25}

    monkeypatch.setattr(runner, "evaluate", evaluate)
    monkeypatch.setattr(runner, "predictor_code_sensitivity", sensitivity)


def reference_training(
    bundle: SourceBundle, config: runner.Config
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Use the full original objective to independently check the reduced training loss."""
    model = build_fresh_model(bundle, config.seed, torch.device("cpu"))
    initial_hash = module_fingerprint(model.parameter_encoder)
    parameters = list(model.parameter_encoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.lr, weight_decay=0)
    assert not optimizer.state
    objective = TrainingObjective(ObjectiveConfig(**bundle.objective_config))
    for step in range(1, config.steps + 1):
        epoch, position = divmod(step - 1, bundle.train_groups // config.batch_size)
        indices = list(
            GlobalBatchSampler(bundle.train_groups, config.batch_size, config.seed, epoch)
        )[position]
        batch = {name: values[indices] for name, values in bundle.train.items()}
        torch.manual_seed(config.seed * 100000 + step)  # pyright: ignore[reportUnknownMemberType]
        model.train()
        model.online_encoder.eval()
        model.target_encoder.eval()
        model.predictor.eval()
        optimizer.zero_grad(set_to_none=True)
        output = cached_multiquery_forward(model, batch, queries=1)
        loss, _ = objective(output)
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), initial_hash


def test_cli_defaults_to_one_fresh_100k_run(monkeypatch: pytest.MonkeyPatch) -> None:
    invocations: list[tuple[runner.Config, bool]] = []

    def capture(config: runner.Config, *, resume: bool = False) -> dict[str, Any]:
        invocations.append((config, resume))
        return {}

    monkeypatch.setattr(runner, "run", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run", "--source-dir", "source", "--output-dir", "fresh", "--wandb-mode", "online"],
    )
    runner.main()
    assert len(invocations) == 1
    config, resume = invocations[0]
    assert not resume
    assert config.steps == 100000
    assert config.seed == 202610261301
    assert config.lr == 0.001
    assert config.wandb_mode == "online"


def test_only_fresh_encoder_changes_with_original_objective_and_empty_adam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = write_source_fixture(tmp_path / "source")
    bundle = load_source_bundle(source)
    before = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    config = settings(source, tmp_path / "fresh")
    expected_model, expected_optimizer, initial_hash = reference_training(bundle, config)
    replace_evaluation(monkeypatch, bundle)
    calls: list[TensorSplit] = []
    optimizer_states: list[int] = []
    encoder_ids: set[int] = set()

    def check_forward(model: SCJEPA, batch: TensorSplit, *, queries: int) -> ModelOutput:
        assert queries == 1
        assert set(batch) == {"online", "target", "actions"}
        assert not any(value.requires_grad for value in batch.values())
        assert not model.online_encoder.training
        assert not model.target_encoder.training
        assert not model.predictor.training
        assert model.parameter_encoder.training
        encoder_ids.clear()
        encoder_ids.update(id(p) for p in model.parameter_encoder.parameters())
        assert {id(p) for p in model.parameters() if p.requires_grad} == encoder_ids
        calls.append({key: value.clone() for key, value in batch.items()})
        return cached_multiquery_forward(model, batch, queries=queries)

    original_step = torch.optim.Adam.step  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]

    def check_step(optimizer: torch.optim.Adam) -> None:
        optimizer_states.append(len(optimizer.state))
        assert {id(p) for group in optimizer.param_groups for p in group["params"]} == encoder_ids
        original_step(optimizer)

    monkeypatch.setattr(runner, "cached_multiquery_forward", check_forward)
    monkeypatch.setattr(torch.optim.Adam, "step", check_step)
    runner.run(config)
    saved = checkpoint(config)
    assert saved["step"] == 3
    assert saved["initial_encoder_sha256"] == initial_hash
    assert optimizer_states[0] == 0
    assert all(count > 0 for count in optimizer_states[1:])
    torch.testing.assert_close(saved["model"], expected_model, rtol=0, atol=0)
    torch.testing.assert_close(saved["optimizer"], expected_optimizer, rtol=0, atol=0)
    initial = build_fresh_model(bundle, config.seed, torch.device("cpu"))
    final = build_fresh_model(bundle, config.seed, torch.device("cpu"))
    final.load_state_dict(saved["model"])
    assert frozen_module_hashes(final) == frozen_module_hashes(initial)
    assert module_fingerprint(final.parameter_encoder) != initial_hash
    assert len(calls) == 3
    indices = next(iter(GlobalBatchSampler(bundle.train_groups, 2, config.seed, 0)))
    torch.testing.assert_close(
        calls[0], {key: value[indices] for key, value in bundle.train.items()}, rtol=0, atol=0
    )
    assert before == {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}


def test_frozen_predictor_keeps_logit_and_second_step_gradients_attached(tmp_path: Path) -> None:
    source = write_source_fixture(tmp_path / "source")
    bundle = load_source_bundle(source)
    model = build_fresh_model(bundle, 7, torch.device("cpu"))
    output = cached_multiquery_forward(model, bundle.train, queries=1)
    first_prediction = output.dynamics[0].prediction
    second_prediction = output.dynamics[1].prediction
    rollout_gradient = torch.autograd.grad(
        second_prediction.square().mean(), first_prediction, retain_graph=True
    )[0]
    logit_penalty = torch.stack([item.logit_penalty for item in output.dynamics]).mean()
    logit_gradient = torch.autograd.grad(logit_penalty, output.parameters, retain_graph=True)[0]
    code_gradient = torch.autograd.grad(second_prediction.square().mean(), output.parameters)[0]
    assert float(rollout_gradient.abs().sum()) > 0
    assert float(logit_gradient.abs().sum()) > 0
    assert float(code_gradient.abs().sum()) > 0
    assert all(p.grad is None for p in model.predictor.parameters())


class RecordingLogger:
    """Capture W&B identities and events without contacting the service."""

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
        """Record fresh-run provenance and the resume identity when supplied."""
        self.config = config
        self.resolved_config = resolved_config
        self.dataset_identity = dataset_identity
        self.out_dir = out_dir
        self.resume_state = copy.deepcopy(resume_state)
        self.logs: list[tuple[int, dict[str, float]]] = []
        self.finished: list[int] = []
        self.artifacts: list[Path] = []
        self._metadata = dict(resume_state or {}) or {
            "id": f"fresh-encoder-{len(self.instances)}",
            "project": config.project or "sparse_causal_jepa",
            "entity": "test-account",
            "mode": config.mode,
        }
        self.instances.append(self)

    @property
    def metadata(self) -> dict[str, object]:
        """Return a defensive copy of the current W&B run identity."""
        return dict(self._metadata)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Record scalar logging on the fresh training-step axis."""
        self.logs.append((step, dict(metrics)))

    def finish(self, exit_code: int = 0) -> None:
        """Record success or failure without SDK side effects."""
        self.finished.append(exit_code)

    def log_artifact(self, path: Path, *, name: str, artifact_type: str = "evaluation") -> None:
        """Require every published artifact to exist."""
        assert name
        assert artifact_type
        assert path.is_file()
        self.artifacts.append(path)


def test_resume_exactly_restores_encoder_adam_random_stream_and_own_wandb_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = write_source_fixture(tmp_path / "source")
    bundle = load_source_bundle(source)
    replace_evaluation(monkeypatch, bundle)
    RecordingLogger.instances.clear()
    monkeypatch.setattr(runner, "WandbLogger", RecordingLogger)
    full = replace(settings(source, tmp_path / "full"), wandb_mode="online")
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
    saved = checkpoint(interrupted)
    assert saved["step"] == 1
    first = RecordingLogger.instances[1]
    assert not first.resume_state
    assert first.finished == [1]
    assert saved["wandb"] == first.metadata
    assert first.metadata["id"] != RecordingLogger.instances[0].metadata["id"]
    directory = Path(interrupted.output_dir) / "seed-7"
    (directory / "gradient_history.json").write_text('{"history":"stale after crash"}')
    monkeypatch.setattr(runner, "cached_multiquery_forward", cached_multiquery_forward)
    result = runner.run(interrupted, resume=True)
    recovered = checkpoint(interrupted)
    reference = checkpoint(full)
    assert recovered["step"] == 3
    assert recovered["gradient_history"] == reference["gradient_history"]
    torch.testing.assert_close(recovered["model"], reference["model"], rtol=0, atol=0)
    torch.testing.assert_close(recovered["optimizer"], reference["optimizer"], rtol=0, atol=0)
    resumed = RecordingLogger.instances[2]
    assert resumed.resume_state == first.metadata
    assert resumed.metadata == first.metadata
    assert resumed.finished == [0]
    assert any(step == 3 for step, _ in resumed.logs)
    assert runner.run(interrupted, resume=True) == result
    (directory / "gradient_history.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact"):
        runner.run(interrupted, resume=True)


def test_resume_rejects_changed_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = write_source_fixture(tmp_path / "source")
    replace_evaluation(monkeypatch, load_source_bundle(source))
    config = replace(settings(source, tmp_path / "fresh"), steps=1)
    runner.run(config)
    with pytest.raises(ValueError, match="settings|identity|changed|differ"):
        runner.run(replace(config, steps=2), resume=True)


@pytest.mark.parametrize("extra_state", [{"autoregressive": False}, {"context_len": 9}])
def test_resume_rejects_changed_rollout_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra_state: dict[str, Any]
) -> None:
    source = write_source_fixture(tmp_path / "source")
    replace_evaluation(monkeypatch, load_source_bundle(source))
    calls = 0

    def interrupt(model: SCJEPA, batch: TensorSplit, *, queries: int) -> ModelOutput:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return cached_multiquery_forward(model, batch, queries=queries)

    monkeypatch.setattr(runner, "cached_multiquery_forward", interrupt)
    config = settings(source, tmp_path / "fresh")
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        runner.run(config)
    saved = checkpoint(config)
    saved["model"]["_extra_state"].update(extra_state)
    torch.save(saved, Path(config.output_dir) / "seed-7/last.pt")
    monkeypatch.setattr(runner, "cached_multiquery_forward", cached_multiquery_forward)
    with pytest.raises(ValueError, match="prediction mode changed|semantics"):
        runner.run(config, resume=True)
