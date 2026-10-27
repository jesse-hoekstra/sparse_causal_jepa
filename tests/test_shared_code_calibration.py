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

from diagnostics.global_code_calibration import objective as original_objective
from diagnostics.global_code_calibration import optimization as original_optimization
from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.multiquery_control.run import sha
from diagnostics.overcomplete_replication.run import Config as ReplicaConfig
from diagnostics.overcomplete_replication.run import run as run_replica
from diagnostics.overcomplete_use_precision.run import run as run_precision
from diagnostics.shared_code_calibration import run as runner
from diagnostics.shared_code_calibration.source import load_inputs, validate_precision_data
from diagnostics.shared_feature_acquisition.run import Config as SharedConfig
from diagnostics.shared_feature_acquisition.run import run as run_shared
from diagnostics.table_settling_control.source import model_fingerprint
from scjepa.models.model import SCJEPA


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def shared_sources(
    source_pair: tuple[Path, Path],  # noqa: F811
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, dict[int, Path]]:
    source, first = source_pair
    root = tmp_path_factory.mktemp("shared-calibration")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        precision = root / "precision"
        run_precision(configuration(source_pair, precision))
        second = root / "replica"
        run_replica(
            ReplicaConfig(
                str(source),
                str(second),
                source_seed=7,
                seed=17,
                evaluation_seed=7,
                steps=2,
                groups=2,
                bootstrap_samples=16,
                verification=True,
            )
        )
        endpoints = {7: root / "shared-first", 17: root / "shared-second"}
        for seed, parent in ((7, first), (17, second)):
            run_shared(
                SharedConfig(
                    str(source),
                    str(parent),
                    str(endpoints[seed]),
                    source_seed=7,
                    seed=seed,
                    evaluation_seed=7,
                    steps=2,
                    groups=2,
                    bootstrap_samples=16,
                    verification=True,
                )
            )
        return source, precision, endpoints
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("seed", [7, 17])
def test_source_restores_shared_endpoint_not_precision_predictor_and_never_loads_heldout(
    shared_sources: tuple[Path, Path, dict[int, Path]],
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
) -> None:
    source, precision, endpoints = shared_sources
    checkpoint = cast(dict[str, Any], torch.load(endpoints[seed] / "last.pt", weights_only=False))
    original_load = cast(Callable[..., Any], torch.load)
    seen: list[str] = []

    def guarded(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            name = Path(file).name
            seen.append(name)
            assert name not in {"evaluation_data.pt", "validation_data.pt", "audit_inputs.pt"}
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded)
    inputs = load_inputs(endpoints[seed], source, precision, seed=seed, verification=True)
    expected = {
        name: value.double() if isinstance(value, Tensor) else value
        for name, value in checkpoint["model"].items()
    }
    torch.testing.assert_close(inputs.model.state_dict(), expected, rtol=0, atol=0)
    assert inputs.evaluation_seed == 7
    assert set(inputs.train) == {"online", "target", "actions"}
    assert "train_inputs.pt" in seen
    assert not inputs.model.training
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in inputs.model.parameters()
    )
    assert all(sha(Path(name)) == value for name, value in inputs.hashes.items())
    with pytest.raises(ValueError, match="seed|identity|endpoint"):
        load_inputs(
            endpoints[seed], source, precision, seed=17 if seed == 7 else 7, verification=True
        )


def test_precision_cache_compatibility_is_state_chart_and_dgp_not_parent_parameter_encoder(
    shared_sources: tuple[Path, Path, dict[int, Path]],
) -> None:
    source, precision, endpoints = shared_sources
    inputs = load_inputs(endpoints[17], source, precision, seed=17, verification=True)
    old = cast(EvaluationData, torch.load(source / "evaluation_data.pt", weights_only=False))
    new = cast(EvaluationData, torch.load(precision / "validation_data.pt", weights_only=False))
    validate_precision_data(new, old, inputs)
    changed = copy.deepcopy(new)
    changed["metadata"]["state_online_sha256"] = "wrong-chart"
    with pytest.raises(ValueError, match="chart|state|metadata"):
        validate_precision_data(changed, old, inputs)
    changed = copy.deepcopy(new)
    changed["metadata"]["observation_matrix"][0, 0] += 0.1
    with pytest.raises(ValueError, match="observation|DGP|metadata"):
        validate_precision_data(changed, old, inputs)


@pytest.mark.parametrize("seed", [7, 17])
def test_fitting_and_resume_use_frozen_global_optimizer_and_original_evaluation_seed(
    shared_sources: tuple[Path, Path, dict[int, Path]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
) -> None:
    source, precision, endpoints = shared_sources
    assert runner.fit is original_optimization.fit
    assert runner.TrainObjective is original_objective.TrainObjective
    assert runner.encode_train is original_objective.encode_train
    inputs = load_inputs(endpoints[seed], source, precision, seed=seed, verification=True)
    fixed = model_fingerprint(inputs.model)
    config = runner.Config(
        str(endpoints[seed]),
        str(source),
        str(precision),
        str(tmp_path / "full"),
        seed=seed,
        chunk_size=2,
        max_iterations=2,
        max_evaluations=6,
        verification=True,
    )
    original_load = cast(Callable[..., Any], torch.load)
    finished = False

    def guarded(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            name = Path(file).name
            assert name != "audit_inputs.pt"
            if name in {"evaluation_data.pt", "validation_data.pt"}:
                assert finished, "Held-out data was deserialized before fitting ended"
        return original_load(file, *args, **kwargs)

    fit = runner.fit

    def finish_fit(
        evaluator: Callable[[Tensor], tuple[float, Tensor]],
        *,
        device: torch.device,
        max_iterations: int,
        max_evaluations: int,
        state: dict[str, Any] | None,
        on_state: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        nonlocal finished
        result = fit(
            evaluator,
            device=device,
            max_iterations=max_iterations,
            max_evaluations=max_evaluations,
            state=state,
            on_state=on_state,
        )
        finished = True
        return result

    evaluate = runner.evaluate
    evaluated_seeds: list[int] = []

    def assess(
        model: SCJEPA,
        data: EvaluationData,
        *,
        chunk_size: int,
        bootstrap_samples: int,
        seed: int,
        progress: Callable[[int], None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Tensor], dict[str, Tensor]]:
        assert finished
        assert seed == 7
        assert set(data) == {"metadata", "validation"}
        evaluated_seeds.append(seed)
        return evaluate(
            model,
            data,
            chunk_size=chunk_size,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            progress=progress,
        )

    objective = original_objective.TrainObjective.__call__
    calls = 0

    def checked(
        self: original_objective.TrainObjective, translation: Tensor
    ) -> tuple[float, Tensor]:
        nonlocal calls
        assert not finished
        assert model_fingerprint(self.model) == fixed
        assert set(self.train) == {"online", "target", "actions"}
        calls += 1
        return objective(self, translation)

    monkeypatch.setattr(torch, "load", guarded)
    monkeypatch.setattr(runner, "fit", finish_fit)
    monkeypatch.setattr(runner, "evaluate", assess)
    monkeypatch.setattr(original_objective.TrainObjective, "__call__", checked)
    full = runner.run(config)
    assert evaluated_seeds == [7] * 4
    assert calls == full["fit"]["full_evaluations"]
    checkpoint = cast(
        dict[str, Any], torch.load(tmp_path / "full/fit_state.pt", weights_only=False)
    )
    finished = False
    calls = 0

    def interrupt(
        self: original_objective.TrainObjective, translation: Tensor
    ) -> tuple[float, Tensor]:
        if calls == 1:
            raise RuntimeError("interrupt shared calibration")
        return checked(self, translation)

    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    monkeypatch.setattr(original_objective.TrainObjective, "__call__", interrupt)
    with pytest.raises(RuntimeError, match="interrupt shared calibration"):
        runner.run(resumed_config)
    assert evaluated_seeds == [7] * 4
    monkeypatch.setattr(original_objective.TrainObjective, "__call__", checked)
    resumed = runner.run(resumed_config, resume=True)
    replay = cast(dict[str, Any], torch.load(tmp_path / "resumed/fit_state.pt", weights_only=False))
    for key in ("translation", "cache"):
        torch.testing.assert_close(checkpoint["state"][key], replay["state"][key], rtol=0, atol=0)
    torch.testing.assert_close(
        checkpoint["state"]["optimizer"]["state"],
        replay["state"]["optimizer"]["state"],
        rtol=0,
        atol=0,
    )
    assert full["reports"] == resumed["reports"]
    assert all(sha(Path(name)) == value for name, value in inputs.hashes.items())
    with pytest.raises(ValueError, match="disjoint"):
        replace(config, output_dir=str(endpoints[seed] / "bad"))
