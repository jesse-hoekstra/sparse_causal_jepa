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
from torch import Tensor

from diagnostics.chart_sparsity import run as runner
from diagnostics.chart_sparsity.core import (
    Chart,
    chart_matrix,
    evaluate,
    transform_data,
    true_graphs,
)
from diagnostics.oracle_state_predictor.data import OracleStateData, build_data
from diagnostics.oracle_state_predictor.graphs import Condition, build_predictor, rollout_graphs
from diagnostics.oracle_state_predictor.run import Config, forecast_metrics
from scjepa.data.simplified_gaussian import transition
from scjepa.models.spartan import SpartanLayer, SpartanOutput
from scjepa.training.logging import WandbLogger


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


@pytest.mark.parametrize("chart", ["physical", "mixed"])
def test_chart_transform_preserves_information_and_supplied_context(
    source: OracleStateData, chart: Chart
) -> None:
    before = deepcopy(source)
    q = chart_matrix(chart)
    torch.testing.assert_close(q @ q.T, torch.eye(6, dtype=torch.float64), rtol=0, atol=3e-16)
    result = transform_data(source, chart)
    expected_states = (source.initial_states.double() @ q.T).to(source.initial_states.dtype)
    expected_targets = (source.targets.double() @ q.T).to(source.targets.dtype)
    torch.testing.assert_close(result.initial_states, expected_states, rtol=0, atol=0)
    torch.testing.assert_close(result.targets, expected_targets, rtol=0, atol=0)
    for name in ("actions", "parameters", "episode_ids"):
        torch.testing.assert_close(getattr(result, name), getattr(before, name), rtol=0, atol=0)
    for name in ("initial_states", "targets", "actions", "parameters", "episode_ids"):
        torch.testing.assert_close(getattr(source, name), getattr(before, name), rtol=0, atol=0)
    torch.testing.assert_close(
        result.initial_states.double() @ q, source.initial_states.double(), rtol=2e-7, atol=2e-7
    )
    assert result.config == source.config
    assert result.source_fingerprint == source.source_fingerprint
    assert result.fingerprint != source.fingerprint
    assert result.fingerprint == transform_data(source, chart).fingerprint
    assert (
        transform_data(source, "physical").fingerprint
        != transform_data(source, "mixed").fingerprint
    )


def test_fixed_mixing_and_reference_graphs_match_the_analytic_control() -> None:
    h = torch.eye(3, dtype=torch.float64) - torch.full((3, 3), 2 / 3, dtype=torch.float64)
    expected = torch.zeros(6, 6, dtype=torch.float64)
    expected[:3, :3], expected[3:, 3:] = h, h
    torch.testing.assert_close(chart_matrix("mixed"), expected, rtol=0, atol=0)
    torch.testing.assert_close(chart_matrix("physical"), torch.eye(6, dtype=torch.float64))
    for chart, counts in (("physical", (18, 96, 12)), ("mixed", (18, 96, 36))):
        graphs = true_graphs(cast(Chart, chart))
        assert tuple(int(graphs[name].sum()) for name in ("state", "action", "parameter")) == counts
        assert all(graph.dtype == torch.bool for graph in graphs.values())
        assert graphs["state"].shape == graphs["parameter"].shape == (6, 6)
        assert graphs["action"].shape == (6, 16)
    assert not bool(true_graphs("mixed")["state"][:3, 3:].any())
    assert not bool(true_graphs("mixed")["state"][3:, :3].any())


def test_two_step_dgp_conjugacy_and_forecast_error_are_orthogonally_invariant(
    source: OracleStateData,
) -> None:
    batch = source.batch(torch.arange(len(source)), "cpu")
    q = chart_matrix("mixed")
    physical = batch.initial_states[:, 0].double()
    mixed = physical @ q.T
    theta, actions = batch.parameters.double(), batch.actions.double()
    expected: list[Tensor] = []
    rotated: list[Tensor] = []
    for horizon in range(2):
        physical = transition(physical, theta, actions[:, horizon])
        mixed = transition(mixed @ q, theta, actions[:, horizon]) @ q.T
        expected.append(physical)
        rotated.append(mixed)
        torch.testing.assert_close(mixed, physical @ q.T, rtol=1e-13, atol=1e-13)
    predictions = torch.stack(expected, dim=1)
    rotated_predictions = torch.stack(rotated, dim=1)
    torch.testing.assert_close(predictions, batch.targets.double(), rtol=2e-6, atol=2e-6)
    residual = torch.arange(predictions.numel(), dtype=torch.float64).reshape_as(predictions) / 100
    original_mse = (predictions + residual - predictions).square().mean()
    rotated_mse = ((predictions + residual) @ q.T - rotated_predictions).square().mean()
    torch.testing.assert_close(original_mse, rotated_mse, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("chart", ["physical", "mixed"])
def test_evaluation_raw_groups_reproduce_metrics_and_ignore_batch_partition(
    source: OracleStateData, chart: Chart
) -> None:
    data = transform_data(source, chart)
    model = build_predictor(Config(embed_dim=8, mlp_hidden=8), "full")
    result = evaluate(model, data, 2, torch.device("cpu"), 1e-4, chart)
    whole = evaluate(model, data, 3, torch.device("cpu"), 1e-4, chart)
    assert result.metrics.keys() == whole.metrics.keys()
    for key, value in result.metrics.items():
        assert isclose(value, whole.metrics[key], rel_tol=2e-5, abs_tol=1e-7), key
    for key, values in result.samples.items():
        torch.testing.assert_close(values, whole.samples[key], rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(result.samples["group_ids"], data.episode_ids, rtol=0, atol=0)
    samples = result.samples
    predictions, targets = samples["predictions"], samples["targets"]
    assert predictions.shape == targets.shape == (3, 2, 2, 6)
    expected = forecast_metrics(predictions.flatten(0, 1), targets.flatten(0, 1))
    for key, value in expected.items():
        assert isclose(result.metrics[key], value, rel_tol=1e-12, abs_tol=1e-12)
    errors = (predictions.double() - targets.double()).square().mean(dim=(1, 3))
    torch.testing.assert_close(samples["group_horizon_mse"], errors, rtol=1e-12, atol=1e-12)
    assert isclose(
        result.metrics["logit"], float(samples["group_horizon_logit"].mean()), rel_tol=1e-6
    )
    for block in ("state", "action", "parameter"):
        means = samples[f"group_{block}_path_means"]
        assert means.shape == (3, 2)
        assert isclose(result.metrics[f"{block}_sparsity"], float(means.mean()), rel_tol=1e-9)
        reachability = samples[f"group_{block}_reachability"]
        assert reachability.shape[:3] == (3, 2, 6)
        assert bool(((reachability >= 0) & (reachability <= 1)).all())
        frequency = reachability.mean(dim=(0, 1))
        assert result.metrics[f"{block}_edges"] == int((frequency > 0).sum())
        assert isclose(
            result.metrics[f"mean_local_{block}_edges"], float(frequency.sum()), rel_tol=1e-9
        )


@pytest.mark.parametrize("fail", [False, True])
def test_real_spartan_evaluation_preserves_model_modes_weights_and_rng(
    source: OracleStateData, fail: bool
) -> None:
    model = build_predictor(Config(embed_dim=8, mlp_hidden=8), "full").train()
    model.layers[0].eval()
    flags = tuple(module.training for module in model.modules())
    weights = deepcopy(model.state_dict())
    python_state = random.getstate()
    numpy_state = pickle.dumps(np.random.get_state())  # noqa: NPY002
    torch_state = torch.get_rng_state().clone()

    def consume_or_fail(_module: object, _inputs: object, _output: object) -> None:
        random.random()
        np.random.random()  # noqa: NPY002
        torch.rand(2)
        if fail:
            raise RuntimeError("intentional chart evaluation failure")

    handle = model.register_forward_hook(consume_or_fail)
    try:
        if fail:
            with pytest.raises(RuntimeError, match="intentional chart evaluation failure"):
                evaluate(
                    model, transform_data(source, "mixed"), 2, torch.device("cpu"), 1e-4, "mixed"
                )
        else:
            evaluate(model, transform_data(source, "mixed"), 2, torch.device("cpu"), 1e-4, "mixed")
    finally:
        handle.remove()
    assert tuple(module.training for module in model.modules()) == flags
    torch.testing.assert_close(model.state_dict(), weights, rtol=0, atol=0)
    assert random.getstate() == python_state
    assert pickle.dumps(np.random.get_state()) == numpy_state  # noqa: NPY002
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)


@pytest.mark.parametrize("condition", ["baseline", "full"])
def test_real_gate_overrides_preserve_baseline_parameter_access(
    source: OracleStateData, condition: Condition
) -> None:
    model = build_predictor(Config(embed_dim=8, mlp_hidden=8), condition)
    with torch.no_grad():
        for layer in model.layers:
            assert isinstance(layer, SpartanLayer)
            layer.project_q.weight.zero_()
            layer.project_k.weight.zero_()
    result = evaluate(model, transform_data(source, "mixed"), 2, torch.device("cpu"), 0, "mixed")
    assert result.metrics["state_edges"] == 6
    assert result.metrics["action_edges"] == 0
    assert result.metrics["parameter_edges"] == (36 if condition == "baseline" else 0)
    assert result.metrics["parameter_sparsity"] == (2 if condition == "baseline" else 0)


def test_training_rollout_uses_attached_first_prediction_and_supplied_parameter(
    source: OracleStateData,
) -> None:
    model = build_predictor(Config(embed_dim=8, mlp_hidden=8), "dense")
    data = transform_data(source, "mixed")
    batch = data.batch(torch.arange(len(data)), "cpu")
    parameters = batch.parameters.clone().requires_grad_(True)
    intermediates: list[Tensor] = []

    def retain_prediction(_module: object, _inputs: object, output: SpartanOutput) -> None:
        output.prediction.retain_grad()
        intermediates.append(output.prediction)

    handle = model.register_forward_hook(retain_prediction)
    try:
        predictions, _, _, _ = rollout_graphs(
            model, batch.initial_states, parameters, batch.actions
        )
        predictions[:, 1].square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
    finally:
        handle.remove()
    assert len(intermediates) == 2
    assert intermediates[0].grad is not None
    assert bool(intermediates[0].grad.abs().sum() > 0)
    assert parameters.grad is not None
    assert bool(parameters.grad.abs().sum() > 0)


def test_scientific_profile_rejects_smoke_budgets_and_relaxed_quality() -> None:
    with pytest.raises(ValueError, match="scientific"):
        runner.PrepareConfig(output_dir="unused", train_groups=4)
    with pytest.raises(ValueError, match="scientific"):
        runner.TrainConfig("unused", "unused", "physical", "dense", steps=2)
    with pytest.raises(ValueError, match="scientific"):
        runner.CalibrateConfig("unused", (), quality_cap=0.03)


def test_shared_rule_uses_calibration_only_for_tau_and_checks_each_validation_mse() -> None:
    rows: list[dict[str, Any]] = []
    for chart in ("physical", "mixed"):
        for seed in (0, 1):
            rows.append(
                {
                    "chart": chart,
                    "seed": seed,
                    "calibration_metrics": {
                        "prediction_mse": 0.01,
                        "base_loss": 0.012 + seed * 0.001,
                    },
                    "validation_metrics": {"prediction_mse": 0.015, "base_loss": 123.0},
                    "test_metrics": {"prediction_mse": 999.0, "base_loss": 999.0},
                }
            )
    passed = runner.calibration_rule(rows)
    assert passed["admitted"]
    assert isclose(passed["tau"], 0.013)
    assert passed["prediction_mse_bound"] == 0.02
    rows[2]["validation_metrics"]["prediction_mse"] = 0.02001
    failed = runner.calibration_rule(rows)
    assert not failed["admitted"]
    assert failed["tau"] == passed["tau"]
    assert failed["failures"] == [
        {"chart": "mixed", "seed": 0, "split": "validation", "prediction_mse": 0.02001}
    ]
    rows[2]["validation_metrics"]["prediction_mse"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        runner.calibration_rule(rows)


def _smoke_config(
    root: Path, chart: Chart, condition: Condition, *, seed: int = 0, steps: int = 2
) -> runner.TrainConfig:
    return runner.TrainConfig(
        output_dir=str(root / f"{chart}-{condition}-{seed}"),
        data_dir=str(root / "data"),
        chart=chart,
        condition=condition,
        profile="smoke",
        seed=seed,
        steps=steps,
        batch_size=2,
        embed_dim=8,
        mlp_hidden=8,
        log_every=1,
        evaluate_every=1,
        checkpoint_every=1,
        dense_dir=None if condition == "dense" else str(root / f"{chart}-dense-{seed}"),
        calibration_dir=None if condition == "dense" else str(root / "calibration"),
    )


def _read_checkpoint(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))


def test_tiny_complete_control_authenticates_sources_pairs_bounds_and_resumes_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = runner.prepare(
        runner.PrepareConfig(
            output_dir=str(tmp_path / "data"),
            profile="smoke",
            train_groups=4,
            calibration_groups=3,
            validation_groups=3,
            test_groups=3,
            views=2,
        )
    )
    assert prepared["profile"] == "smoke"
    data, _ = runner.load_data(tmp_path / "data")
    assert {data[split].config.split for split in data} == {
        "train",
        "evaluation_calibration",
        "validation",
        "test",
    }
    assert len({item.source_fingerprint for item in data.values()}) == 4
    sources: list[str] = []
    dense_results: dict[tuple[str, int], dict[str, Any]] = {}
    for chart in ("physical", "mixed"):
        for seed in (0, 1):
            config = _smoke_config(tmp_path, chart, "dense", seed=seed)
            dense_results[chart, seed] = runner.train(config)
            directory = Path(config.output_dir)
            sources.append(str(directory))
            assert not (directory / "test.json").exists()
            assert not (directory / "test.pt").exists()
            assert not (directory / "best.pt").exists()
            assert dense_results[chart, seed]["successful_updates"] == 2
            assert runner.train(config, resume=True) == dense_results[chart, seed]
            with pytest.raises(FileExistsError, match="not empty"):
                runner.train(config)
    for seed in (0, 1):
        assert (
            dense_results["physical", seed]["initial_state_identity"]
            == dense_results["mixed", seed]["initial_state_identity"]
        )
    calibration_config = runner.CalibrateConfig(
        str(tmp_path / "calibration"), tuple(sources), profile="smoke", quality_cap=1e3
    )
    calibrated = runner.calibrate(calibration_config)
    assert calibrated["admitted"]
    calibration = runner.read_json(tmp_path / "calibration" / "calibration.json")
    assert calibration["tau"] == max(
        runner.read_json(Path(source) / "calibration.json")["metrics"]["base_loss"]
        for source in sources
    )
    with pytest.raises(ValueError, match="exactly four"):
        runner.calibrate(
            replace(
                calibration_config,
                output_dir=str(tmp_path / "missing"),
                dense_dirs=tuple(sources[:3]),
            )
        )
    with pytest.raises(ValueError, match="duplicate"):
        runner.calibrate(
            replace(
                calibration_config,
                output_dir=str(tmp_path / "duplicate"),
                dense_dirs=tuple(sources[:3] + sources[:1]),
            )
        )
    rejected = runner.calibrate(
        replace(calibration_config, output_dir=str(tmp_path / "rejected"), quality_cap=1e-12)
    )
    assert not rejected["admitted"]
    with pytest.raises(ValueError, match="admission failed"):
        runner.train(
            replace(
                _smoke_config(tmp_path, "physical", "full"),
                output_dir=str(tmp_path / "prohibited"),
                calibration_dir=str(tmp_path / "rejected"),
            )
        )
    for chart in ("physical", "mixed"):
        configs = [_smoke_config(tmp_path, chart, condition) for condition in ("baseline", "full")]
        results = [runner.train(config) for config in configs]
        assert results[0]["initial_state_identity"] == results[1]["initial_state_identity"]
        for config, result in zip(configs, results, strict=True):
            directory = Path(config.output_dir)
            record = runner.read_json(directory / "config.json")
            assert record["calibration"]["tau"] == calibration["tau"]
            assert record["dense_source"]["checkpoint_sha256"] == runner.sha(
                Path(cast(str, config.dense_dir)) / "last.pt"
            )
            assert (directory / "test.json").exists()
            assert (directory / "test.pt").exists()
            checkpoint = _read_checkpoint(directory / "last.pt")
            assert checkpoint["step"] == int(checkpoint["controller"]["num_updates"]) == 2
            assert result["skipped_updates"] == 0
            assert runner.train(config, resume=True) == result
            with pytest.raises(ValueError, match="provenance differs"):
                runner.train(replace(config, steps=3), resume=True)
        baseline_metrics = runner.read_json(Path(configs[0].output_dir) / "test.json")["metrics"]
        assert baseline_metrics["parameter_edges"] == 36

    uninterrupted = replace(
        _smoke_config(tmp_path, "physical", "full", steps=3),
        output_dir=str(tmp_path / "uninterrupted"),
    )
    interrupted = replace(uninterrupted, output_dir=str(tmp_path / "interrupted"))
    runner.train(uninterrupted)
    original_log = runner._log  # pyright: ignore[reportPrivateUsage]

    def interrupt_log(
        logger: WandbLogger, output: Path, step: int, metrics: dict[str, float]
    ) -> None:
        if step == 2:
            raise RuntimeError("simulated interruption after saved update one")
        original_log(logger, output, step, metrics)

    with monkeypatch.context() as context:
        context.setattr(runner, "_log", interrupt_log)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            runner.train(interrupted)
    partial = _read_checkpoint(Path(interrupted.output_dir) / "last.pt")
    assert partial["step"] == 1
    runner.train(interrupted, resume=True)
    expected = _read_checkpoint(Path(uninterrupted.output_dir) / "last.pt")
    actual = _read_checkpoint(Path(interrupted.output_dir) / "last.pt")
    for key in ("model", "optimizer", "controller", "sampler", "torch_rng"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    with monkeypatch.context() as context:
        context.setattr(runner, "code_identity", lambda: "changed-source")
        with pytest.raises(ValueError, match="source identity differs"):
            runner.train(interrupted, resume=True)
    dense_checkpoint = Path(sources[0]) / "last.pt"
    dense_checkpoint.write_bytes(dense_checkpoint.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="artifact hash differs"):
        runner.train(
            replace(
                _smoke_config(tmp_path, "physical", "full"),
                output_dir=str(tmp_path / "tampered-source"),
            )
        )
