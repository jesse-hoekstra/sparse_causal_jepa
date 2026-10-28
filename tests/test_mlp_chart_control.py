import pickle
import random
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import replace
from math import isclose
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from diagnostics.chart_sparsity import run as chart_runner
from diagnostics.chart_sparsity.core import Chart, transform_data
from diagnostics.mlp_chart_control import run as runner
from diagnostics.mlp_chart_control.core import (
    EvaluationResult,
    chart_transport,
    closure_audit,
    conjugate,
    evaluate,
    rollout,
)
from diagnostics.mlp_oracle_code_control.model import MLPPredictor, Prediction
from diagnostics.oracle_state_predictor.data import OracleStateData, build_data
from diagnostics.oracle_state_predictor.run import forecast_metrics


@pytest.fixture(autouse=True)
def one_torch_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def source() -> OracleStateData:
    return build_data("validation", 3, views=2, seed=23)


def test_native_mlp_weights_absorb_arbitrary_orthogonal_charts_for_both_steps() -> None:
    generator = torch.Generator().manual_seed(314)
    matrix, _ = cast(
        tuple[Tensor, Tensor],
        torch.linalg.qr(  # pyright: ignore[reportUnknownMemberType]
            torch.randn(6, 6, dtype=torch.float64, generator=generator)
        ),
    )
    assert not torch.allclose(matrix, matrix.T)
    model = MLPPredictor().double().eval()
    original = deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    transported = conjugate(model, matrix)
    assert type(transported) is type(model)
    assert sum(parameter.numel() for parameter in transported.parameters()) == 52_123
    assert list(transported.state_dict()) == list(model.state_dict())
    torch.testing.assert_close(model.state_dict(), original, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    states = torch.randn(5, 1, 6, dtype=torch.float64, generator=generator)
    theta = torch.randn(5, 6, dtype=torch.float64, generator=generator)
    actions = torch.randn(5, 2, 16, dtype=torch.float64, generator=generator)
    physical = rollout(model, states, theta, actions)
    mixed = rollout(transported, states @ matrix.T, theta, actions)
    torch.testing.assert_close(mixed, physical @ matrix.T, rtol=1e-11, atol=1e-12)
    recovered = conjugate(transported, matrix.T)
    torch.testing.assert_close(
        rollout(recovered, states, theta, actions), physical, rtol=1e-11, atol=1e-12
    )
    # The transported model is an independent native MLP, not a wrapper around source weights.
    assert all(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(model.parameters(), transported.parameters(), strict=True)
    )


def test_two_step_rollout_keeps_exact_raw_theta_and_first_prediction_attached(
    source: OracleStateData,
) -> None:
    model = MLPPredictor().requires_grad_(False)
    batch = source.batch(torch.arange(len(source)), "cpu")
    states = batch.initial_states.clone().requires_grad_(True)
    theta = batch.parameters.clone().requires_grad_(True)
    observed: list[tuple[tuple[Tensor, ...], Tensor]] = []

    def retain(_module: nn.Module, inputs: tuple[Tensor, ...], output: Prediction) -> None:
        output.prediction.retain_grad()
        observed.append((inputs, output.prediction))

    handle = model.register_forward_hook(retain)
    try:
        predicted = rollout(model, states, theta, batch.actions)
        predicted[:, 1].square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
    finally:
        handle.remove()
    assert len(observed) == 2
    for index, (inputs, _) in enumerate(observed):
        assert inputs[1] is theta
        torch.testing.assert_close(inputs[2], batch.actions[:, index : index + 1])
    torch.testing.assert_close(observed[1][0][0][:, 0], observed[0][1], rtol=0, atol=0)
    assert observed[0][1].grad is not None
    assert bool(observed[0][1].grad.abs().sum() > 0)
    assert theta.grad is not None
    assert bool(theta.grad.abs().sum() > 0)
    assert states.grad is not None
    assert bool(states.grad.abs().sum() > 0)
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_preserves_modes_rng_weights_and_cached_inputs(
    source: OracleStateData, fail: bool
) -> None:
    model = MLPPredictor().train()
    model.network[0].eval()
    flags = tuple(module.training for module in model.modules())
    weights = deepcopy(model.state_dict())
    cached = deepcopy(source)
    python_rng = random.getstate()
    numpy_rng = pickle.dumps(np.random.get_state())  # noqa: NPY002
    torch_rng = torch.get_rng_state().clone()

    def consume_or_fail(_module: nn.Module, _inputs: object, _output: object) -> None:
        random.random()
        np.random.random()  # noqa: NPY002
        torch.rand(2)
        if fail:
            raise RuntimeError("intentional MLP evaluation failure")

    handle = model.register_forward_hook(consume_or_fail)
    try:
        if fail:
            with pytest.raises(RuntimeError, match="intentional MLP evaluation failure"):
                evaluate(model, source, 2, torch.device("cpu"))
        else:
            evaluate(model, source, 2, torch.device("cpu"))
    finally:
        handle.remove()
    assert tuple(module.training for module in model.modules()) == flags
    torch.testing.assert_close(model.state_dict(), weights, rtol=0, atol=0)
    assert random.getstate() == python_rng
    assert pickle.dumps(np.random.get_state()) == numpy_rng  # noqa: NPY002
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
    for name in ("initial_states", "targets", "actions", "parameters", "episode_ids"):
        torch.testing.assert_close(getattr(source, name), getattr(cached, name), rtol=0, atol=0)


def test_raw_evaluation_reproduces_metrics_and_conjugacy_witness(
    source: OracleStateData,
) -> None:
    model = MLPPredictor()
    physical = evaluate(model, source, 2, torch.device("cpu"))
    whole_batch = evaluate(model, source, 3, torch.device("cpu"))
    samples = physical.samples
    assert samples["predictions"].shape == samples["targets"].shape == (3, 2, 2, 6)
    torch.testing.assert_close(samples["group_ids"], source.episode_ids, rtol=0, atol=0)
    predictions, targets = samples["predictions"], samples["targets"]
    expected_metrics = forecast_metrics(predictions.flatten(0, 1), targets.flatten(0, 1))
    for key, expected in expected_metrics.items():
        assert isclose(physical.metrics[key], expected, rel_tol=1e-12, abs_tol=1e-12)
        assert isclose(physical.metrics[key], whole_batch.metrics[key], rel_tol=1e-6, abs_tol=1e-8)
    torch.testing.assert_close(
        samples["group_horizon_mse"],
        (predictions.double() - targets.double()).square().mean(dim=(1, 3)),
        rtol=0,
        atol=0,
    )
    matrix = chart_transport("physical", "mixed")
    mixed_data = transform_data(source, "mixed")
    mixed = evaluate(conjugate(model, matrix), mixed_data, 2, torch.device("cpu"))
    closure_audit(physical, mixed, matrix)
    wrong = evaluate(model, mixed_data, 2, torch.device("cpu"))
    with pytest.raises(ValueError, match="numerical tolerances"):
        closure_audit(physical, wrong, matrix)


def _smoke_config(root: Path, chart: Chart, *, steps: int = 4) -> runner.TrainConfig:
    return runner.TrainConfig(
        output_dir=str(root / chart),
        data_dir=str(root / "data"),
        chart=chart,
        profile="smoke",
        steps=steps,
        batch_size=2,
        log_every=1,
        evaluate_every=2,
        checkpoint_every=2,
    )


def _read_checkpoint(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))


def test_fixed_budget_smoke_authenticates_pairing_and_resumes_exactly_without_test_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chart_runner.prepare(
        chart_runner.PrepareConfig(
            output_dir=str(tmp_path / "data"),
            profile="smoke",
            train_groups=4,
            calibration_groups=3,
            validation_groups=3,
            test_groups=3,
            views=2,
        )
    )
    actual_evaluate = runner.evaluate
    splits: list[str] = []

    def evaluate_without_test(
        model: MLPPredictor, data: OracleStateData, batch_size: int, device: torch.device
    ) -> EvaluationResult:
        splits.append(data.config.split)
        assert data.config.split != "test"
        return actual_evaluate(model, data, batch_size, device)

    monkeypatch.setattr(runner, "evaluate", evaluate_without_test)
    physical_config = _smoke_config(tmp_path, "physical")
    physical = runner.train(physical_config)
    mixed_config = _smoke_config(tmp_path, "mixed")
    mixed = runner.train(mixed_config)
    assert physical["initial_state_identity"] == mixed["initial_state_identity"]
    assert set(splits) == {"validation", "evaluation_calibration"}
    for config, completed in ((physical_config, physical), (mixed_config, mixed)):
        directory = Path(config.output_dir)
        assert completed["successful_updates"] == 4
        assert completed["skipped_updates"] == 0
        assert runner.verify_completion(directory, require_current_source=True) == completed
        assert runner.train(config, resume=True) == completed
        assert not (directory / "test.pt").exists()
        assert not (directory / "test.json").exists()
        assert not (directory / "best.pt").exists()
        with pytest.raises(FileExistsError):
            runner.train(config)
        with pytest.raises(ValueError, match="provenance|identity|configuration|source"):
            runner.train(replace(config, steps=5), resume=True)

    interrupted = replace(physical_config, output_dir=str(tmp_path / "interrupted"))
    actual_save = runner.save_torch

    def save_then_interrupt(path: Path, value: object) -> None:
        actual_save(path, value)
        if path.name == "last.pt" and cast(dict[str, Any], value).get("step") == 2:
            raise RuntimeError("simulated interruption after checkpoint two")

    with monkeypatch.context() as context:
        context.setattr(runner, "save_torch", save_then_interrupt)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            runner.train(interrupted)
    assert _read_checkpoint(Path(interrupted.output_dir) / "last.pt")["step"] == 2
    resumed = runner.train(interrupted, resume=True)
    expected = _read_checkpoint(Path(physical_config.output_dir) / "last.pt")
    actual = _read_checkpoint(Path(interrupted.output_dir) / "last.pt")
    for name in ("model", "optimizer", "sampler", "torch_rng"):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert actual["step"] == expected["step"] == 4
    assert resumed["successful_updates"] == 4
    with monkeypatch.context() as context:
        context.setattr(runner, "code_identity", lambda: "changed-source")
        with pytest.raises(ValueError, match="source|provenance|identity"):
            runner.train(interrupted, resume=True)
    checkpoint = Path(interrupted.output_dir) / "last.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash|artifact"):
        runner.verify_completion(Path(interrupted.output_dir))


def test_scientific_profile_rejects_changed_budget_or_optimization() -> None:
    with pytest.raises(ValueError, match="scientific"):
        runner.TrainConfig("unused", "unused", "physical", steps=4)
    with pytest.raises(ValueError, match="scientific"):
        runner.TrainConfig("unused", "unused", "physical", lr=0.002)
