import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from diagnostics.oracle_state_learning.analysis import (
    audit_samples,
    controller_history,
    grouped_moments,
    moment_mcc,
    paired_mcc_interval,
    routing_metrics,
)
from scjepa.eval.recovery import CoordinateMatch, mean_correlation_coefficient


def _rounding_payload() -> dict[str, Any]:
    states = torch.linspace(-2, 2, 2 * 10 * 12 * 6).reshape(2, 10, 12, 6)
    targets = states[:, :, 10:12].clone()
    # Captured CUDA float32 encodings of one identical observation, using
    # 320-frame versus 1920-frame GEMMs. Float64 replay agrees exactly.
    states[0, 0, 11, 0] = 0.1287655234336853
    targets[0, 0, 1, 0] = 0.1287667453289032
    predictions = targets + 0.1
    return {
        "samples": {
            "online_states": states.clone(),
            "target_states": states,
            "states": states.clone(),
            "targets": targets,
            "predictions": predictions,
            "group_ids": torch.arange(2),
        },
        "metrics": {"prediction": float((predictions.double() - targets.double()).square().mean())},
    }


def test_sample_audit_accepts_captured_batch_shape_rounding_without_changing_targets() -> None:
    payload = _rounding_payload()
    original = payload["samples"]["targets"].clone()
    audited = audit_samples(payload, 2)
    assert torch.equal(audited["targets"], original)


@pytest.mark.parametrize("corruption", ["wrong_frame", "changed_encoder", "wrong_metric"])
def test_sample_audit_still_rejects_semantic_or_metric_mismatch(corruption: str) -> None:
    payload = _rounding_payload()
    if corruption == "wrong_frame":
        payload["samples"]["target_states"] = payload["samples"]["target_states"].roll(1, 2)
    elif corruption == "changed_encoder":
        payload["samples"]["target_states"] += 1e-4
    else:
        payload["metrics"]["prediction"] += 0.01
    message = "Reported forecast MSE" if corruption == "wrong_metric" else "Forecast targets differ"
    with pytest.raises(ValueError, match=message):
        audit_samples(payload, 2)


def test_group_moments_match_pearson_and_keep_calibration_assignment() -> None:
    generator = torch.Generator().manual_seed(6)
    truth = torch.randn(8, 10, 12, 6, generator=generator)
    columns = [2, 4, 0, 5, 3, 1]
    features = torch.empty_like(truth)
    features[..., columns] = 2 * truth + 1
    features[..., columns[0]] = 0
    actual = float(moment_mcc(grouped_moments(features, truth, columns).sum(0)))
    expected = mean_correlation_coefficient(
        CoordinateMatch(tuple(columns)), features.flatten(0, 2), truth.flatten(0, 2)
    )
    assert abs(actual - expected) < 1e-12
    assert abs(actual - 5 / 6) < 1e-12


def test_paired_bootstrap_identical_arms_zero_and_mixing_positive() -> None:
    generator = torch.Generator().manual_seed(9)
    truth = torch.randn(24, 10, 12, 6, generator=generator)
    columns = list(range(6))
    identical = paired_mcc_interval(truth, truth, truth, columns, columns, draws=40)
    assert identical["difference"] == identical["lower_95"] == identical["upper_95"] == 0
    mixed = (truth + truth.roll(1, -1)) / 2**0.5
    improved = paired_mcc_interval(truth, mixed, truth, columns, columns, draws=100)
    assert improved["lower_95"] > 0.25
    assert improved["upper_95"] < 0.35


def test_routing_uses_distinct_online_and_target_calibration_permutations() -> None:
    rows, columns = [2, 4, 0, 5, 1, 3], [5, 2, 1, 0, 3, 4]
    state = torch.zeros(6, 6)
    state[:3, :3] = 1
    state[3:, 3:] = 1
    parameter = torch.eye(6)
    parameter[torch.arange(6), [3, 4, 5, 0, 1, 2]] = 1
    learned_state, learned_parameter = torch.empty_like(state), torch.empty_like(parameter)
    learned_state[torch.tensor(rows)[:, None], torch.tensor(columns)[None, :]] = state
    learned_parameter[rows] = parameter
    samples = {
        "graph_state": learned_state[None].repeat(2, 1, 1),
        "graph_parameter": learned_parameter[None].repeat(2, 1, 1),
        "graph_action": torch.ones(2, 6, 16),
    }
    result = routing_metrics(samples, {"online_columns": columns, "target_columns": rows})
    for name, edges in (("state", 18), ("action", 96), ("parameter", 12)):
        assert result[f"routing/{name}_observed_union_edges"] == edges
        assert result[f"routing/{name}_aligned_union_shd"] == 0


def test_controller_history_deduplicates_resumed_steps_and_labels_sampling(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    rows = [
        {"step": 100, "geco/penalty_weight": math.exp(-12)},
        {"step": 200, "geco/penalty_weight": math.exp(-12)},
        {"step": 200, "geco/penalty_weight": 0.1},
        {"step": 200, "validation/state_mcc": 0.99},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    result = controller_history(path)
    assert result["logged_updates"] == 2
    assert result["fraction_logged_at_penalty_floor"] == 0.5
    assert result["maximum_penalty_weight"] == result["final_penalty_weight"] == 0.1
