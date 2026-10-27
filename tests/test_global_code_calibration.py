import copy
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_overcomplete_use_precision import configuration
from test_overcomplete_use_precision import source_pair as source_pair
from torch import Tensor

from diagnostics.global_code_calibration import run as runner
from diagnostics.global_code_calibration.objective import TrainObjective
from diagnostics.global_code_calibration.source import load_inputs
from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.multiquery_control.run import sha
from diagnostics.overcomplete_use_precision.run import run as run_precision
from diagnostics.table_settling_control.source import model_fingerprint
from scjepa.models.model import SCJEPA
from scjepa.training.logging import WandbLogger


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def calibration_sources(
    source_pair: tuple[Path, Path],  # noqa: F811
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path]:
    precision = tmp_path_factory.mktemp("calibration-precision") / "results"
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        run_precision(configuration(source_pair, precision))
        return *source_pair, precision
    finally:
        torch.set_num_threads(previous)


def test_source_reads_only_train_and_authenticates_endpoint_and_cache(
    calibration_sources: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, endpoint, precision = calibration_sources
    original_load = cast(Callable[..., Any], torch.load)
    seen: list[str] = []

    def guarded(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            name = Path(file).name
            seen.append(name)
            assert name not in {"evaluation_data.pt", "validation_data.pt", "audit_inputs.pt"}
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded)
    inputs = load_inputs(endpoint, source, precision, verification=True)
    assert set(seen) == {"last.pt", "train_inputs.pt"}
    assert set(inputs.train) == {"online", "target", "actions"}
    assert all(not p.requires_grad and p.grad is None for p in inputs.model.parameters())
    assert next(inputs.model.parameters()).dtype == torch.float64
    assert not inputs.model.training
    assert all(sha(Path(name)) == value for name, value in inputs.hashes.items())


def test_fit_has_no_heldout_access_preserves_source_and_resumes_exactly(
    calibration_sources: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, endpoint, precision = calibration_sources
    inputs = load_inputs(endpoint, source, precision, verification=True)
    frozen = model_fingerprint(inputs.model)
    config = runner.Config(
        str(endpoint),
        str(source),
        str(precision),
        str(tmp_path / "full"),
        chunk_size=2,
        max_iterations=2,
        max_evaluations=6,
        verification=True,
    )
    original_load = cast(Callable[..., Any], torch.load)
    finished = False
    heldout_reads = 0

    def guarded(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        nonlocal heldout_reads
        if isinstance(file, str | Path):
            name = Path(file).name
            assert name != "audit_inputs.pt"
            if name in {"evaluation_data.pt", "validation_data.pt"}:
                assert finished, "Held-out tensors entered fitting"
                heldout_reads += 1
        return original_load(file, *args, **kwargs)

    original_fit = runner.fit

    def fit(
        evaluator: Callable[[Tensor], tuple[float, Tensor]],
        *,
        device: torch.device,
        max_iterations: int,
        max_evaluations: int,
        state: dict[str, Any] | None,
        on_state: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        nonlocal finished
        result = original_fit(
            evaluator,
            device=device,
            max_iterations=max_iterations,
            max_evaluations=max_evaluations,
            state=state,
            on_state=on_state,
        )
        finished = True
        return result

    assessment = runner.evaluate
    assessment_calls = 0

    def evaluate(
        model: SCJEPA,
        data: EvaluationData,
        *,
        chunk_size: int,
        bootstrap_samples: int,
        seed: int,
        progress: Callable[[int], None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Tensor], dict[str, Tensor]]:
        nonlocal assessment_calls
        assessment_calls += 1
        assert finished
        assert set(data) == {"metadata", "validation"}
        assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
        return assessment(
            model,
            data,
            chunk_size=chunk_size,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            progress=progress,
        )

    actual_objective = TrainObjective.__call__
    evaluations = 0
    logged_steps: list[int] = []
    original_log = WandbLogger.log

    def log(self: WandbLogger, step: int, metrics: dict[str, float]) -> None:
        logged_steps.append(step)
        original_log(self, step, metrics)

    def objective(self: TrainObjective, translation: Tensor) -> tuple[float, Tensor]:
        nonlocal evaluations
        assert not finished
        assert model_fingerprint(self.model) == frozen
        assert set(self.train) == {"online", "target", "actions"}
        evaluations += 1
        return actual_objective(self, translation)

    monkeypatch.setattr(torch, "load", guarded)
    monkeypatch.setattr(runner, "fit", fit)
    monkeypatch.setattr(runner, "evaluate", evaluate)
    monkeypatch.setattr(TrainObjective, "__call__", objective)
    monkeypatch.setattr(WandbLogger, "log", log)
    full = runner.run(config)
    assert heldout_reads == 2
    assert assessment_calls == 4
    assert evaluations == full["fit"]["full_evaluations"] <= 6
    assert full["fit"]["accepted_loss"] <= full["fit"]["initial_loss"]
    final = cast(dict[str, Any], torch.load(tmp_path / "full/fit_state.pt", weights_only=False))
    candidate = cast(
        dict[str, Any], torch.load(tmp_path / "full/composed_model.pt", weights_only=False)
    )
    expected = copy.deepcopy(inputs.model.state_dict())
    expected["parameter_encoder.head.4.bias"] += final["state"]["translation"]
    torch.testing.assert_close(candidate["model"], expected, rtol=0, atol=0)
    finished = False
    logged_steps.clear()
    interrupted_calls = 0

    def interrupt(self: TrainObjective, translation: Tensor) -> tuple[float, Tensor]:
        nonlocal interrupted_calls
        interrupted_calls += 1
        if interrupted_calls == 2:
            raise RuntimeError("interrupt full objective")
        return objective(self, translation)

    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    monkeypatch.setattr(TrainObjective, "__call__", interrupt)
    with pytest.raises(RuntimeError, match="interrupt full objective"):
        runner.run(resumed_config)
    assert heldout_reads == 2
    assert not (tmp_path / "resumed/translation.pt").exists()
    monkeypatch.setattr(TrainObjective, "__call__", objective)
    resumed = runner.run(resumed_config, resume=True)
    assert logged_steps == sorted(set(logged_steps))
    replay = cast(dict[str, Any], torch.load(tmp_path / "resumed/fit_state.pt", weights_only=False))
    for key in ("translation", "cache"):
        torch.testing.assert_close(final["state"][key], replay["state"][key], rtol=0, atol=0)
    assert (
        final["state"]["optimizer"]["param_groups"] == replay["state"]["optimizer"]["param_groups"]
    )
    torch.testing.assert_close(
        final["state"]["optimizer"]["state"],
        replay["state"]["optimizer"]["state"],
        rtol=0,
        atol=0,
    )
    assert full["fit"]["full_evaluations"] == resumed["fit"]["full_evaluations"]
    assert full["reports"] == resumed["reports"]
    assert all(sha(Path(name)) == value for name, value in inputs.hashes.items())
    assert model_fingerprint(inputs.model) == frozen
    read_count, eval_count = heldout_reads, assessment_calls
    runner.run(resumed_config, resume=True)
    assert (heldout_reads, assessment_calls) == (read_count, eval_count)
    with pytest.raises(ValueError, match="disjoint"):
        runner.Config(str(endpoint), str(source), str(precision), str(source / "bad"))
    assert not (source / "bad").exists()
    cache = cast(
        dict[str, Any], torch.load(tmp_path / "resumed/train_codes.pt", weights_only=False)
    )
    cache["codes"][0, 0, 0] += 1
    torch.save(cache, tmp_path / "resumed/train_codes.pt")
    with pytest.raises(ValueError, match="changed|mismatch|hash"):
        runner.run(resumed_config, resume=True)
