import json
import math
from pathlib import Path

import torch

from diagnostics.oracle_state_learning.analysis import (
    controller_history,
    grouped_moments,
    moment_mcc,
    paired_mcc_interval,
    routing_metrics,
)
from scjepa.eval.recovery import CoordinateMatch, mean_correlation_coefficient


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
