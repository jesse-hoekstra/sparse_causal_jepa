from typing import Any

from diagnostics.joint_acquisition.report import decision_report


def comparison_result() -> dict[str, Any]:
    rows: list[dict[str, Any]] = [
        {
            "parameter": parameter,
            "normalized_response_error": {"estimate": 1.2, "interval_95": [1.1, 1.3]},
            "predictive_benefit": {"estimate": 0.0, "interval_95": [-0.1, 0.1]},
        }
        for parameter in range(1, 7)
    ]
    baseline: dict[str, Any] = {"ordinary_mse_by_horizon": [0.3, 0.4], "per_parameter": rows}
    source: dict[str, Any] = {**baseline, "per_parameter": [dict(row) for row in rows]}
    source["per_parameter"][0] = {
        "parameter": 1,
        "normalized_response_error": {"estimate": 0.5, "interval_95": [0.4, 0.6]},
        "predictive_benefit": {"estimate": 0.2, "interval_95": [0.1, 0.3]},
    }
    paired = [
        {
            "parameter": parameter,
            "source_minus_current_normalized_response_error": {
                "estimate": -0.3 if parameter == 1 else 0.0,
                "interval_95": [-0.4, -0.2] if parameter == 1 else [-0.1, 0.1],
            },
            "source_minus_current_predictive_benefit": {
                "estimate": 0.2 if parameter == 1 else 0.0,
                "interval_95": [0.1, 0.3] if parameter == 1 else [-0.1, 0.1],
            },
        }
        for parameter in range(1, 7)
    ]
    return {
        "identity": "report-fixture",
        "scientific_status": "software verification only",
        "baseline": baseline,
        "retention": {"selected_validation_r2": [0.1] * 6},
        "shared_corrections": {"numerically_valid": True, "cases": {}},
        "objective_incentives": {"cases": {}},
        "code_repair": {"status": "undefined"},
        "trajectories": {},
        "compatible_control": {
            "status": "measured",
            "source": source,
            "current": baseline,
            "paired_source_minus_current_ordinary_mse_by_horizon": [-0.02, -0.03],
            "paired_per_parameter": paired,
        },
    }


def test_report_connects_paired_source_advantage_to_a_bounded_next_decision() -> None:
    report = decision_report(comparison_result())
    assert "[-0.02, -0.03]" in report
    assert "-0.3; 95% CI [-0.4, -0.2]" in report
    assert "0.2; 95% CI [0.1, 0.3]" in report
    assert "Paired source-code advantage for parameters [1]" in report
    assert "code-dependent limitation" in report
    assert "predictor adequacy remains open for parameters [2, 3, 4, 5, 6]" in report
    assert "point differences alone have no reported confidence interval" in report


def test_report_does_not_turn_failed_or_undefined_bounds_into_predictor_incapacity() -> None:
    result = comparison_result()
    comparison = result["compatible_control"]
    comparison["source"] = comparison["current"]
    for row in comparison["paired_per_parameter"]:
        row["source_minus_current_normalized_response_error"]["interval_95"] = None
    report = decision_report(result)
    assert "compatible comparison is inconclusive" in report
    assert "predictor adequacy remains open for parameters [1, 2, 3, 4, 5, 6]" in report
    assert "not proof of predictor incapacity" in report
    assert "Paired source-code advantage for parameters" not in report
