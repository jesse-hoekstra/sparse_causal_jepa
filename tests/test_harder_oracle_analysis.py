"""Independent endpoint checks and prespecified effects for the harder-DGP screen."""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest
import torch

from diagnostics.harder_oracle.analysis import analyze, routing_metrics, screen_decision
from diagnostics.harder_oracle.data import PrepareConfig, code_identity, prepare
from diagnostics.harder_oracle.run import CalibrateConfig, TrainConfig, calibrate, train


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
