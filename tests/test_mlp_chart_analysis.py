import json
import math
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from diagnostics.chart_sparsity.core import Chart, chart_matrix
from diagnostics.mlp_chart_control.analysis import (
    Arm,
    audit_conjugated_weights,
    audit_samples,
    load_arms,
    summarize,
)
from diagnostics.mlp_chart_control.core import EvaluationResult, conjugate
from diagnostics.mlp_chart_control.run import SCHEMA
from diagnostics.mlp_oracle_code_control.model import MLPPredictor
from diagnostics.oracle_state_predictor.run import forecast_metrics

CHARTS: tuple[Chart, Chart] = ("physical", "mixed")


def _evaluation(mse: float, groups: int = 256) -> EvaluationResult:
    targets = torch.linspace(-2, 2, groups * 10 * 2 * 6, dtype=torch.float64).reshape(
        groups, 10, 2, 6
    )
    predictions = targets + math.sqrt(mse)
    samples = {
        "predictions": predictions,
        "targets": targets,
        "group_ids": torch.arange(groups),
        "group_horizon_mse": (predictions - targets).square().mean(dim=(1, 3)),
    }
    metrics = forecast_metrics(predictions.flatten(0, 1), targets.flatten(0, 1))
    metrics["base_loss"] = metrics["prediction_mse"]
    return EvaluationResult(metrics, samples)


def _arm(seed: int, chart: Chart, mse: float, calibration: float | None = None) -> Arm:
    evaluations = {
        "validation": _evaluation(mse),
        "calibration": _evaluation(mse if calibration is None else calibration, 512),
    }
    return Arm(Path("."), seed, chart, {}, {}, evaluations, {})


@pytest.mark.parametrize(
    ("physical", "mixed", "expected"),
    [
        (0.01, 0.011, "comparable_fresh_fits"),
        (0.005, 0.015, "residual_chart_dependence"),
        (0.01, 0.03, "attainable_but_fresh_optimization_gap"),
        (0.03, 0.01, "attainable_but_fresh_optimization_gap"),
        (0.03, 0.04, "fit_limitation_inconclusive"),
    ],
)
def test_predeclared_decisions(physical: float, mixed: float, expected: str) -> None:
    chart_errors: tuple[tuple[Chart, float], ...] = (("physical", physical), ("mixed", mixed))
    arms: dict[tuple[int, Chart], Arm] = {
        (seed, chart): _arm(seed, chart, value) for seed in (0, 1) for chart, value in chart_errors
    }
    result = summarize(arms)
    assert result["decision"] == expected
    assert all(row["decision"] == expected for row in result["seeds"])
    assert result["bootstrap_draws"] == 1000
    assert result["bootstrap_seed"] == 2026102802


def test_calibration_failure_is_not_hidden_by_good_validation() -> None:
    arms: dict[tuple[int, Chart], Arm] = {
        (seed, chart): _arm(seed, chart, 0.01) for seed in (0, 1) for chart in CHARTS
    }
    arms[1, "mixed"] = _arm(1, "mixed", 0.01, calibration=0.03)
    result = summarize(arms)
    assert result["decision"] == "mixed_outcomes"
    assert result["seeds"][0]["decision"] == "comparable_fresh_fits"
    assert result["seeds"][1]["decision"] == "attainable_but_fresh_optimization_gap"


def test_bootstrap_is_shared_across_seeds_and_reproducible() -> None:
    arms: dict[tuple[int, Chart], Arm] = {
        (seed, chart): _arm(seed, chart, 0.01) for seed in (0, 1) for chart in CHARTS
    }
    for seed in (0, 1):
        samples = arms[seed, "mixed"].evaluations["validation"].samples
        samples["group_horizon_mse"] += torch.linspace(-0.004, 0.006, 256)[:, None]
    first, second = summarize(arms), summarize(arms)
    intervals = [row["paired_group_bootstrap_95_interval"] for row in first["seeds"]]
    assert intervals[0] == intervals[1]
    assert intervals[0][1] > intervals[0][0]
    assert first == second


def test_saved_group_errors_and_metrics_are_recomputed() -> None:
    evaluation = _evaluation(0.01)
    audit_samples(evaluation, 256)
    evaluation.samples["predictions"][0, 0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="group errors disagree"):
        audit_samples(evaluation, 256)
    evaluation = _evaluation(0.01)
    evaluation.metrics["prediction_mse"] = 0.001
    with pytest.raises(ValueError, match="metric prediction_mse disagrees"):
        audit_samples(evaluation, 256)


def test_derived_weights_must_be_the_analytic_copy() -> None:
    model = MLPPredictor()
    matrix = chart_matrix("mixed")
    copied = conjugate(model, matrix)
    source = cast(dict[str, Tensor], model.state_dict())
    transported = cast(dict[str, Tensor], copied.state_dict())
    audit_conjugated_weights(source, transported, matrix)
    transported["network.2.weight"][0, 0] += 0.01
    with pytest.raises(ValueError, match="not the declared conjugation"):
        audit_conjugated_weights(source, transported, matrix)


def test_analysis_rejects_tampered_files_before_loading_them(tmp_path: Path) -> None:
    directories: list[Path] = []
    for index in range(4):
        directory = tmp_path / str(index)
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        (directory / "completion.json").write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "kind": "train",
                    "hashes": {"config.json": "wrong-hash"},
                }
            )
        )
        directories.append(directory)
    with pytest.raises(ValueError, match="artifact hash differs"):
        load_arms(directories)
