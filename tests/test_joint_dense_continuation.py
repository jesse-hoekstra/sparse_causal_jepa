import copy
import shutil
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_joint_state_calibration import Arm
from test_joint_state_calibration import calibration_sources as calibration_sources
from test_joint_state_survival import joint_sources as joint_sources
from torch import Tensor

from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.joint_dense_continuation import calibration as final_calibration
from diagnostics.joint_dense_continuation import run as runner
from diagnostics.joint_dense_continuation.source import Inputs, load_inputs
from diagnostics.joint_state_survival.forward import after_update, training_loss
from diagnostics.multiquery_control.data import (
    EvaluationData,
    TensorSplit,
    encode_states,
    module_fingerprint,
)
from diagnostics.multiquery_control.run import sha
from diagnostics.predictor_transfer_calibration.objective import TrainObjective
from diagnostics.predictor_transfer_calibration.optimization import EmbeddedObjective
from scjepa.experiments.storage import read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.loop import TrainingObjective


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def assert_optimizer_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    torch.testing.assert_close(actual["state"], expected["state"], rtol=0, atol=0)
    assert actual["param_groups"] == expected["param_groups"]


def direct_updates(
    inputs: Inputs, steps: int, *, reset_optimizer: bool = False
) -> list[dict[str, Any]]:
    """Run the original model/objective directly, independently of the continuation loop."""
    model = copy.deepcopy(inputs.model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0)
    if not reset_optimizer:
        optimizer.load_state_dict(copy.deepcopy(inputs.optimizer.state_dict()))
    objective = TrainingObjective(inputs.objective.config)
    records: list[dict[str, Any]] = []
    for update in range(1, steps + 1):
        step = inputs.source_step + update
        epoch, position = divmod(step - 1, 2)
        indices = list(GlobalBatchSampler(4, 2, 7, epoch))[position]
        batch = {key: value[indices] for key, value in inputs.observations.items()}
        torch.default_generator.manual_seed(700000 + step)
        model.train()
        model.target_encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective(model(batch["observations"], batch["actions"]))
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        model.update_target()
        records.append(
            {
                "model": copy.deepcopy(model.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "torch_rng": torch.get_rng_state().clone(),
                "loss": float(loss.detach()),
            }
        )
    return records


def test_source_restores_live_weights_moments_rng_and_rejects_other_or_reset_endpoints(
    calibration_sources: tuple[dict[Arm, Path], Path, Path],  # noqa: F811
    tmp_path: Path,
) -> None:
    endpoints, source, precision = calibration_sources
    endpoint = endpoints["live_state"]
    inputs = load_inputs(
        endpoint,
        source,
        precision,
        checkpoint_sha256=sha(endpoint / "last.pt"),
        device=torch.device("cpu"),
        verification=True,
    )
    saved = cast(dict[str, Any], torch.load(endpoint / "last.pt", weights_only=False))
    assert inputs.source_step == saved["step"] == 4
    assert inputs.optimizer_step == saved["updates"] == 2
    assert set(inputs.observations) == {"observations", "actions"}
    assert next(inputs.model.parameters()).dtype == torch.float32
    torch.testing.assert_close(inputs.model.state_dict(), saved["model"], rtol=0, atol=0)
    assert_optimizer_equal(inputs.optimizer.state_dict(), saved["optimizer"])
    assert torch.equal(inputs.rng["torch"], saved["rng"]["torch"])
    assert module_fingerprint(inputs.model.online_encoder) != module_fingerprint(
        inputs.model.target_encoder
    )
    assert all(parameter.requires_grad for parameter in inputs.model.online_encoder.parameters())
    assert all(
        not parameter.requires_grad for parameter in inputs.model.target_encoder.parameters()
    )
    assert all(float(row["step"]) == 2 for row in inputs.optimizer.state_dict()["state"].values())
    before_hashes = {path: sha(Path(path)) for path in inputs.hashes}
    with pytest.raises(ValueError, match="pin|checkpoint|source"):
        load_inputs(
            endpoint,
            source,
            precision,
            checkpoint_sha256="f" * 64,
            device=torch.device("cpu"),
            verification=True,
        )
    frozen = endpoints["frozen_state"]
    with pytest.raises(ValueError, match="arm|live"):
        load_inputs(
            frozen,
            source,
            precision,
            checkpoint_sha256=sha(frozen / "last.pt"),
            device=torch.device("cpu"),
            verification=True,
        )
    damaged = tmp_path / "reset-moment-source"
    shutil.copytree(endpoint, damaged)
    payload = copy.deepcopy(saved)
    first = next(iter(payload["optimizer"]["state"].values()))
    first["step"].zero_()
    torch.save(payload, damaged / "last.pt")
    checkpoint_hash = sha(damaged / "last.pt")
    assessment = cast(dict[str, Any], read_json(damaged / "endpoint_assessment.json"))
    assessment["checkpoint_sha256"] = checkpoint_hash
    write_json(damaged / "endpoint_assessment.json", assessment)
    complete = cast(dict[str, Any], read_json(damaged / "completed.json"))
    for name in ("last.pt", "endpoint_assessment.json"):
        complete["hashes"][name] = sha(damaged / name)
    write_json(damaged / "completed.json", complete)
    with pytest.raises(ValueError, match="moment|optimizer|step"):
        load_inputs(
            damaged,
            source,
            precision,
            checkpoint_sha256=checkpoint_hash,
            device=torch.device("cpu"),
            verification=True,
        )
    assert all(sha(Path(path)) == value for path, value in before_hashes.items())


def test_exact_next_updates_live_ema_final_only_evaluation_and_interrupted_resume(
    calibration_sources: tuple[dict[Arm, Path], Path, Path],  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoints, source, precision = calibration_sources
    endpoint = endpoints["live_state"]
    inputs = load_inputs(
        endpoint,
        source,
        precision,
        checkpoint_sha256=sha(endpoint / "last.pt"),
        device=torch.device("cpu"),
        verification=True,
    )
    expected = direct_updates(inputs, 3)
    fresh = direct_updates(inputs, 1, reset_optimizer=True)
    assert any(
        not torch.equal(value, fresh[0]["model"][key])
        for key, value in expected[0]["model"].items()
        if isinstance(value, Tensor)
    )
    active = tmp_path / "full"
    config = runner.Config(
        str(endpoint),
        str(source),
        str(precision),
        str(active),
        sha(endpoint / "last.pt"),
        steps=3,
        seed=7,
        groups=2,
        bootstrap_samples=8,
        chunk_size=3,
        max_iterations=2,
        max_evaluations=6,
        verification=True,
    )
    original_load = cast(Callable[..., Any], torch.load)
    original_monitor, original_precision = runner.monitor, runner.precision_assessment
    original_evaluate = runner.evaluate
    original_fit = final_calibration.fit
    original_fit_final = final_calibration.fit_final
    assert runner.training_loss is training_loss
    assert runner.after_update is after_update
    calls, ema_calls, tests, precisions = 0, 0, 0, 0
    fits = 0
    responses = 0
    fitted = False
    interrupt = False
    interrupt_after_fit = False

    def guarded_load(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            assert "joint-calibration-" not in str(file)
            assert "shared-confirmatory" not in str(file)
            assert Path(file).name not in {"fresh_test.pt", "train_inputs.pt"}
        return original_load(file, *args, **kwargs)

    def checked_loss(
        model: SCJEPA, batch: TensorSplit, objective: TrainingObjective, arm: Arm
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal calls
        calls += 1
        assert arm == "live_state"
        step = inputs.source_step + calls
        epoch, position = divmod(step - 1, 2)
        indices = list(GlobalBatchSampler(4, 2, 7, epoch))[position]
        torch.testing.assert_close(
            batch,
            {key: value[indices] for key, value in inputs.observations.items()},
            rtol=0,
            atol=0,
        )
        assert torch.equal(
            torch.get_rng_state(), torch.Generator().manual_seed(700000 + step).get_state()
        )
        if calls == 1:
            torch.testing.assert_close(
                model.state_dict(), inputs.model.state_dict(), rtol=0, atol=0
            )
            initial = cast(dict[str, Any], original_load(active / "last.pt", weights_only=False))
            assert_optimizer_equal(initial["optimizer"], inputs.optimizer.state_dict())
            assert initial["step"] == 4
        if interrupt and calls == 2:
            raise RuntimeError("continuation interruption")
        loss, terms = training_loss(model, batch, objective, arm)
        assert float(loss.detach()) == expected[calls - 1]["loss"]
        return loss, terms

    def checked_after(model: SCJEPA, arm: Arm) -> None:
        nonlocal ema_calls
        ema_calls += 1
        before = copy.deepcopy(model.target_encoder.state_dict())
        after_update(model, arm)
        for key, value in model.target_encoder.state_dict().items():
            torch.testing.assert_close(
                value,
                before[key] * 0.996 + model.online_encoder.state_dict()[key] * 0.004,
                rtol=1e-6,
                atol=1e-7,
            )
        torch.testing.assert_close(model.state_dict(), expected[calls - 1]["model"], rtol=0, atol=0)
        assert all(parameter.grad is None for parameter in model.target_encoder.parameters())

    def checked_monitor(
        model: SCJEPA,
        data: EvaluationData,
        selection: DiagnosticSelection,
        *,
        bootstrap_samples: int,
        seed: int,
        final: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Tensor]]:
        nonlocal tests
        if final:
            assert calls == 3
            assert fitted
            tests += 1
        return original_monitor(
            model, data, selection, bootstrap_samples=bootstrap_samples, seed=seed, final=final
        )

    def checked_precision(
        model: SCJEPA,
        old_data: EvaluationData,
        precision_data: EvaluationData,
        *,
        bootstrap_samples: int,
        seed: int,
        chunk_size: int,
        progress: Callable[[int], None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal precisions
        assert calls == 3
        assert fitted
        precisions += 1
        torch.testing.assert_close(
            precision_data["validation"]["theta"],
            inputs.precision_data["validation"]["theta"],
            rtol=0,
            atol=0,
        )
        return original_precision(
            model,
            old_data,
            precision_data,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            chunk_size=chunk_size,
            progress=progress,
        )

    def checked_fit(
        evaluator: EmbeddedObjective,
        *,
        device: torch.device,
        max_iterations: int = 25,
        max_evaluations: int = 50,
        state: dict[str, Any] | None = None,
        on_state: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        nonlocal fits, fitted
        assert calls == 3
        assert not fitted
        assert (max_iterations, max_evaluations) == (2, 6)
        assert isinstance(evaluator.evaluator, TrainObjective)
        objective = evaluator.evaluator
        assert set(objective.train) == {"online", "target", "actions"}
        assert all(not parameter.requires_grad for parameter in objective.model.parameters())
        for key, value in objective.model.state_dict().items():
            expected_value = expected[-1]["model"][key]
            if isinstance(value, Tensor):
                torch.testing.assert_close(value, expected_value.double(), rtol=0, atol=0)
            else:
                assert value == expected_value
        online, target = encode_states(
            copy.deepcopy(objective.model).float(), inputs.observations["observations"], 32
        )
        torch.testing.assert_close(objective.train["online"], online, rtol=0, atol=0)
        torch.testing.assert_close(objective.train["target"], target, rtol=0, atol=0)
        torch.testing.assert_close(
            objective.train["actions"], inputs.observations["actions"], rtol=0, atol=0
        )
        fits += 1
        result = original_fit(
            evaluator,
            device=device,
            max_iterations=max_iterations,
            max_evaluations=max_evaluations,
            state=state,
            on_state=on_state,
        )
        assert (active / "calibration/fit_state.pt").exists()
        assert result["stop_reason"] != "running"
        fitted = True
        return result

    def checked_evaluate(
        model: SCJEPA,
        data: EvaluationData,
        *,
        chunk_size: int,
        bootstrap_samples: int,
        seed: int,
        progress: Callable[[int], None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Tensor], dict[str, Tensor]]:
        nonlocal responses
        assert calls == 3
        assert fitted
        assert set(data) == {"metadata", "validation"}
        assert all(not parameter.requires_grad for parameter in model.parameters())
        responses += 1
        return original_evaluate(
            model,
            data,
            chunk_size=chunk_size,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            progress=progress,
        )

    def checked_fit_final(
        model: SCJEPA,
        observations: TensorSplit,
        output: Path,
        *,
        identity: str,
        checkpoint_sha256: str,
        chunk_size: int,
        max_iterations: int,
        max_evaluations: int,
        log: Callable[[dict[str, float]], None],
    ) -> tuple[SCJEPA, SCJEPA, dict[str, Any], list[str]]:
        assert calls == 3
        assert checkpoint_sha256 == sha(active / "last.pt")
        result = original_fit_final(
            model,
            observations,
            output,
            identity=identity,
            checkpoint_sha256=checkpoint_sha256,
            chunk_size=chunk_size,
            max_iterations=max_iterations,
            max_evaluations=max_evaluations,
            log=log,
        )
        assert (output / "composed_model.pt").exists()
        assert (output / "completed.json").exists()
        if interrupt_after_fit:
            raise RuntimeError("interrupted after fixed TRAIN calibration")
        return result

    monkeypatch.setattr(torch, "load", guarded_load)
    monkeypatch.setattr(runner, "training_loss", checked_loss)
    monkeypatch.setattr(runner, "after_update", checked_after)
    monkeypatch.setattr(runner, "monitor", checked_monitor)
    monkeypatch.setattr(runner, "precision_assessment", checked_precision)
    monkeypatch.setattr(runner, "evaluate", checked_evaluate)
    monkeypatch.setattr(final_calibration, "fit", checked_fit)
    monkeypatch.setattr(runner, "fit_final", checked_fit_final)
    full = runner.run(config)
    full_checkpoint = cast(dict[str, Any], original_load(active / "last.pt", weights_only=False))
    assert (calls, ema_calls, tests, precisions) == (3, 3, 1, 1)
    assert fits == 1
    assert responses == 3
    assert full_checkpoint["step"] == 7
    assert full_checkpoint["updates"] == 3
    torch.testing.assert_close(full_checkpoint["model"], expected[-1]["model"], rtol=0, atol=0)
    assert_optimizer_equal(full_checkpoint["optimizer"], expected[-1]["optimizer"])
    assert all(float(row["step"]) == 5 for row in full_checkpoint["optimizer"]["state"].values())
    completed = cast(dict[str, Any], read_json(active / "completed.json"))
    assert all(sha(active / name) == value for name, value in completed["hashes"].items())

    calls, ema_calls, tests, precisions = 0, 0, 0, 0
    responses = 0
    fitted = False
    active = tmp_path / "resumed"
    resumed_config = replace(config, output_dir=str(active))
    interrupt = True
    with pytest.raises(RuntimeError, match="continuation interruption"):
        runner.run(resumed_config)
    partial = cast(dict[str, Any], original_load(active / "last.pt", weights_only=False))
    assert partial["updates"] == 1
    assert partial["step"] == 5
    assert (ema_calls, tests, precisions) == (1, 0, 0)
    saved_checkpoint = (active / "last.pt").read_bytes()
    saved_receipt = (active / "last.receipt.json").read_bytes()
    corrupted = copy.deepcopy(partial)
    corrupted["optimizer"]["state"] = {}
    torch.save(corrupted, active / "last.pt")
    receipt = cast(dict[str, Any], read_json(active / "last.receipt.json"))
    receipt["sha256"] = sha(active / "last.pt")
    write_json(active / "last.receipt.json", receipt)
    with pytest.raises(ValueError, match="moment|optimizer|AdamW|state"):
        runner.run(resumed_config, resume=True)
    assert calls == 2  # Reject missing moments before another training forward.
    (active / "last.pt").write_bytes(saved_checkpoint)
    (active / "last.receipt.json").write_bytes(saved_receipt)
    calls, interrupt = 1, False
    interrupt_after_fit = True
    with pytest.raises(RuntimeError, match="after fixed TRAIN calibration"):
        runner.run(resumed_config, resume=True)
    assert (calls, ema_calls, tests, precisions, fits) == (3, 3, 0, 0, 2)
    fit_hash = sha(active / "calibration/fit_state.pt")

    def no_fit_reentry(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Completed calibration budget must not reopen")

    monkeypatch.setattr(final_calibration, "fit", no_fit_reentry)
    interrupt_after_fit = False
    resumed = runner.run(resumed_config, resume=True)
    assert sha(active / "calibration/fit_state.pt") == fit_hash
    replay = cast(dict[str, Any], original_load(active / "last.pt", weights_only=False))
    torch.testing.assert_close(full_checkpoint["model"], replay["model"], rtol=0, atol=0)
    assert_optimizer_equal(full_checkpoint["optimizer"], replay["optimizer"])
    assert torch.equal(full_checkpoint["rng"]["torch"], replay["rng"]["torch"])
    assert full["records"] == resumed["records"]
    assert full["endpoint"] == resumed["endpoint"]
    assert (calls, ema_calls, tests, precisions) == (3, 3, 1, 1)
    assert fits == 2
    assert responses == 3
    runner.run(resumed_config, resume=True)
    assert (calls, ema_calls, tests, precisions) == (3, 3, 1, 1)
    assert fits == 2
    assert responses == 3
    with pytest.raises(ValueError, match="resume|changed|identity"):
        runner.run(replace(resumed_config, steps=2), resume=True)
    assert all(sha(Path(path)) == value for path, value in inputs.hashes.items())


def test_scientific_source_and_fixed_continuation_budget_cannot_be_selected(tmp_path: Path) -> None:
    checkpoint = "9e551d96cfaf292d6353f6ab1268619bca686f29ac5fc7b819bcafc5f7f56643"
    config = runner.Config(
        str(tmp_path / "endpoint"),
        str(tmp_path / "source"),
        str(tmp_path / "precision"),
        str(tmp_path / "output"),
        checkpoint,
    )
    assert config.steps == 90000
    assert (config.seed, config.groups, config.bootstrap_samples) == (202610261301, 64, 1000)
    with pytest.raises(ValueError, match="pin|source|checkpoint"):
        replace(config, checkpoint_sha256="f" * 64)
    with pytest.raises(ValueError, match="fixed|fixes|90"):
        replace(config, steps=1000)
    with pytest.raises(ValueError, match="disjoint"):
        replace(config, output_dir=str(tmp_path / "endpoint/nested"))
