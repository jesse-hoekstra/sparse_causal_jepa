"""Independent endpoint checks and prespecified effects for the harder-DGP screen."""

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import pytest
import torch

from diagnostics.harder_oracle.analysis import (
    analyze,
    ordering_evidence,
    routing_metrics,
    screen_decision,
)
from diagnostics.harder_oracle.data import PrepareConfig, code_identity, prepare
from diagnostics.harder_oracle.run import CalibrateConfig, TrainConfig, calibrate, train
from diagnostics.oracle_state_learning.analysis import paired_mcc_interval
from scjepa.eval.recovery import CoordinateMatch, mean_correlation_coefficient


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _metrics() -> dict[str, dict[str, float]]:
    return {
        name: {
            "state_mcc": mcc,
            "normalized_prediction_mse": 0.1,
            "online_state_probe/r2": 0.98,
            "online/variance_min": 0.8,
            "target/variance_min": 0.8,
        }
        for name, mcc in (("dense", 0.7), ("mlp", 0.6), ("baseline", 0.85), ("full", 0.98))
    }


def test_prespecified_effect_and_health_decision() -> None:
    metrics = _metrics()
    result = screen_decision(metrics)
    assert result["screen_passed"]
    assert "requires_frozen_test_confirmation" in result["decision"]
    metrics["baseline"]["state_mcc"] = 0.96
    result = screen_decision(metrics)
    assert not result["screen_passed"]
    assert not result["mcc_effect_criteria_met"]
    metrics = _metrics()
    metrics["mlp"]["online_state_probe/r2"] = 0.3
    result = screen_decision(metrics)
    assert result["mcc_effect_criteria_met"]
    assert not result["state_linear_r2_all_conditions"]
    assert not result["screen_passed"]
    metrics = _metrics()
    metrics["dense"]["normalized_prediction_mse"] = 0.2
    assert not screen_decision(metrics)["screen_passed"]


def _ordering_case(
    *, full_angle: float = 0.1, baseline_angle: float = 0.27, split: str = "validation"
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    generator = torch.Generator().manual_seed(73)
    truth = torch.randn(32, 10, 12, 6, generator=generator, dtype=torch.float64)
    features: dict[str, torch.Tensor] = {}
    metrics = _metrics()
    columns = tuple(range(6))
    for name, angle in (("full", full_angle), ("baseline", baseline_angle), ("dense", 0.72)):
        rotation = torch.zeros(6, 6, dtype=torch.float64)
        block = torch.tensor(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
            dtype=torch.float64,
        )
        for start in (0, 2, 4):
            rotation[start : start + 2, start : start + 2] = block
        features[name] = truth @ rotation.T
        metrics[name]["state_mcc"] = mean_correlation_coefficient(
            CoordinateMatch(columns), features[name].flatten(0, 2), truth.flatten(0, 2)
        )
    comparisons: list[dict[str, Any]] = []
    for left, right in (("full", "baseline"), ("baseline", "dense")):
        comparisons.append(
            {
                "left": left,
                "right": right,
                "split": split,
                **paired_mcc_interval(features[left], features[right], truth, columns, columns),
            }
        )
    return metrics, comparisons


@pytest.mark.parametrize("split", ["validation", "test"])
def test_adaptive_positive_ordering_does_not_relabel_original_effect_target(split: str) -> None:
    metrics, comparisons = _ordering_case(split=split)
    strict = screen_decision(metrics)
    assert 0.02 < strict["full_minus_baseline"] < 0.04
    assert not strict["screen_passed"]
    assert strict["criteria"]["full_baseline_gap_minimum"] == 0.05
    result = ordering_evidence(metrics, comparisons, split=split)
    assert result["ordering_supported"]
    assert result["confirmation_eligible"]
    assert "adaptively declared" in result["adaptive_ordering_rule"]["status"]
    assert result["adaptive_ordering_rule"]["original_strict_effect_targets_unchanged"]
    assert screen_decision(metrics) == strict


@pytest.mark.parametrize("lower", [-0.01, 0.0])
def test_adaptive_ordering_requires_strictly_positive_interval_lower_bounds(lower: float) -> None:
    metrics, comparisons = _ordering_case()
    comparisons[0]["lower_95"] = lower
    result = ordering_evidence(metrics, comparisons, split="validation")
    assert not result["ordering_supported"]
    assert not result["confirmation_eligible"]


def test_adaptive_ordering_rejects_negative_gap_and_retains_quality_requirements() -> None:
    metrics, comparisons = _ordering_case(full_angle=0.27, baseline_angle=0.1)
    assert comparisons[0]["difference"] < 0
    assert not ordering_evidence(metrics, comparisons, split="validation")["confirmation_eligible"]
    metrics, comparisons = _ordering_case()
    metrics["dense"]["normalized_prediction_mse"] = 0.2
    result = ordering_evidence(metrics, comparisons, split="validation")
    assert result["ordering_supported"]
    assert not result["confirmation_eligible"]


def test_true22action_support_is_aligned_with_separate_encoder_assignments() -> None:
    rows, columns = [2, 4, 0, 5, 1, 3], [5, 2, 1, 0, 3, 4]
    state = torch.zeros(6, 6)
    state[:3, :3] = state[3:, 3:] = 1
    parameter = torch.eye(6)
    parameter[torch.arange(6), [3, 4, 5, 0, 1, 2]] = 1
    learned_state, learned_parameter = torch.empty_like(state), torch.empty_like(parameter)
    learned_state[torch.tensor(rows)[:, None], torch.tensor(columns)[None, :]] = state
    learned_parameter[rows] = parameter
    samples = {
        "graph_state": learned_state[None].repeat(2, 1, 1),
        "graph_parameter": learned_parameter[None].repeat(2, 1, 1),
        "graph_action": torch.ones(2, 6, 22),
    }
    result = routing_metrics(samples, {"online_columns": columns, "target_columns": rows})
    for name, edges in (("state", 18), ("action", 132), ("parameter", 12)):
        assert result[f"routing/{name}_observed_union_edges"] == edges
        assert result[f"routing/{name}_aligned_union_shd"] == 0


@pytest.mark.parametrize("transport_chart", ["mixed", "physical"])
def test_analysis_audits_real_tiny_endpoints_but_cannot_call_them_scientific(
    tmp_path: Path,
    transport_chart: Literal["mixed", "physical"],
) -> None:
    before = code_identity()
    prepare(
        PrepareConfig(
            str(tmp_path / "data"),
            profile="smoke",
            transport_chart=transport_chart,
            train_groups=4,
            calibration_groups=2,
            validation_groups=2,
            test_groups=2,
        )
    )
    for condition in ("dense", "mlp", "baseline", "full"):
        if condition == "baseline":
            calibrate(
                CalibrateConfig(
                    str(tmp_path / "bound"), (str(tmp_path / "dense"),), profile="smoke"
                )
            )
        train(
            TrainConfig(
                str(tmp_path / condition),
                str(tmp_path / "data"),
                condition,
                profile="smoke",
                steps=2,
                batch_size=2,
                log_every=1,
                evaluate_every=2,
                checkpoint_every=1,
                calibration_dir=str(tmp_path / "bound")
                if condition in {"baseline", "full"}
                else None,
            )
        )
    directories = [tmp_path / name for name in ("dense", "mlp", "baseline", "full")]
    with pytest.raises(ValueError, match="Scientific analysis"):
        analyze(directories, tmp_path / "forbidden")
    result = analyze(
        directories,
        tmp_path / "analysis",
        primary_chart=transport_chart,
        draws=10,
        allow_smoke=True,
    )
    assert result["profile"] == "smoke"
    assert result["primary_chart"] == transport_chart
    assert "ordering_supported" in result
    assert "confirmation_eligible" in result
    assert len(result["arms"]) == 4
    assert len(result["comparisons"]) == 4
    assert all(row["split"] == "validation" for row in result["comparisons"])
    assert before == code_identity()
    assert not any((directory / "test.pt").exists() for directory in directories)
    wrong_chart = "mixed" if transport_chart == "physical" else "physical"
    with pytest.raises(ValueError, match="must use only the declared"):
        analyze(directories, tmp_path / "wrong-chart", primary_chart=wrong_chart, allow_smoke=True)


def test_physical_primary_rejects_original_companion_before_reading_artifacts(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="only to the mixed primary"):
        analyze([], tmp_path / "analysis", primary_chart="physical", companion=tmp_path / "unused")
