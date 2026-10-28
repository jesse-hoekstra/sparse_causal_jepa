import math
from collections.abc import Generator
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from diagnostics.chart_sparsity.analysis import (
    REQUIRED_ARTIFACTS,
    Arm,
    ArmKey,
    analyze,
    compare,
    interval,
    load_arms,
)
from diagnostics.chart_sparsity.core import BLOCKS, chart_matrix
from diagnostics.chart_sparsity.run import SCHEMA, calibration_rule, save_torch, sha
from diagnostics.oracle_state_predictor.run import forecast_metrics, write_json
from scjepa.data.datasets import canonical_fingerprint


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def make_arms(
    root: Path, *, parameter_signal: bool = True, mse_shift: float = 0
) -> dict[ArmKey, Arm]:
    """Make fixed raw measurements with known paired chart and policy contrasts."""
    physical_targets = torch.randn(
        (256, 10, 2, 6), generator=torch.Generator().manual_seed(83)
    ).double()
    group_noise = torch.linspace(-0.01, 0.01, 256, dtype=torch.float64)[:, None].expand(-1, 2)
    sources: list[dict[str, Any]] = [
        {
            "identity": f"dense-{chart}-{seed}",
            "checkpoint_sha256": f"checkpoint-{chart}-{seed}",
            "completion_sha256": f"completion-{chart}-{seed}",
            "chart": chart,
            "seed": seed,
            "steps": 100_000,
            "calibration_metrics": {"prediction_mse": 0.01, "base_loss": 0.015},
            "validation_metrics": {"prediction_mse": 0.01, "base_loss": 0.015},
        }
        for chart in ("physical", "mixed")
        for seed in (0, 1)
    ]
    calibration = calibration_rule(sources) | {
        "dense_sources": sources,
        "artifact_sha256": "shared",
    }
    arms: dict[ArmKey, Arm] = {}
    for seed in (0, 1):
        for chart in ("physical", "mixed"):
            targets = physical_targets @ chart_matrix(chart).T
            for policy in ("baseline", "full"):
                error = (0.005 + (mse_shift if chart == "mixed" else 0) + group_noise / 100).sqrt()
                predictions = targets + error[:, None, :, None]
                samples = {
                    "predictions": predictions,
                    "targets": targets,
                    "group_ids": torch.arange(256),
                    "group_horizon_mse": (predictions - targets).square().mean(dim=(1, 3)),
                    "group_horizon_logit": torch.full((256, 2), 2.0, dtype=torch.float64),
                }
                for block in BLOCKS:
                    value = 2.0
                    if block == "parameter":
                        value = 3.0 if policy == "baseline" else 1.5
                        if chart == "mixed":
                            value += 0.2 if policy == "baseline" else 2.0 if parameter_signal else 0
                    elif (
                        block == "state"
                        and not parameter_signal
                        and chart == "mixed"
                        and policy == "full"
                    ):
                        value += 2
                    means = torch.full((256, 2), value, dtype=torch.float64) + group_noise
                    width = 16 if block == "action" else 6
                    samples[f"group_{block}_path_means"] = means
                    samples[f"group_{block}_path_sums"] = means * (6 * width)
                    samples[f"group_{block}_reachability"] = torch.ones(
                        (256, 2, 6, width), dtype=torch.float64
                    )
                metrics = forecast_metrics(predictions.flatten(0, 1), targets.flatten(0, 1))
                metrics |= {"logit": 2.0, "base_loss": metrics["prediction_mse"] + 0.0002}
                for block in BLOCKS:
                    metrics[f"{block}_sparsity"] = float(
                        samples[f"group_{block}_path_means"].mean()
                    )
                    metrics[f"{block}_edges"] = 96.0 if block == "action" else 36.0
                scientific = {
                    "profile": "scientific",
                    "seed": seed,
                    "chart": chart,
                    "condition": policy,
                    "steps": 100_000,
                    "logit_weight": 0.0001,
                }
                dense_source = next(
                    row for row in sources if row["chart"] == chart and row["seed"] == seed
                )
                specification = {
                    "scientific_config": scientific,
                    "data_identity": "same-physical-cache",
                    "data_sha256": "same-physical-cache-hash",
                    "code_identity": "same-historical-numerical-source",
                    "dense_source": dense_source,
                    "calibration": calibration,
                    "initial_state_identity": f"dense-state-{chart}-{seed}",
                }
                identity = canonical_fingerprint(specification)
                record = {"schema": SCHEMA, "identity": identity, **specification}
                completion = {
                    "schema": SCHEMA,
                    "identity": identity,
                    "kind": "train",
                    "successful_updates": 100_000,
                    "skipped_updates": 0,
                    **scientific,
                    **{
                        name: specification[name]
                        for name in ("code_identity", "data_identity", "initial_state_identity")
                    },
                }
                directory = root / f"s{seed}-{chart}-{policy}"
                arms[seed, chart, policy] = Arm(
                    directory, seed, chart, policy, completion, record, metrics, samples
                )
    return arms


def write_arms(arms: dict[ArmKey, Arm]) -> list[Path]:
    """Serialize authenticated minimal endpoint fixtures with genuine raw tensors."""
    for arm in arms.values():
        directory = arm.directory
        directory.mkdir(parents=True)
        write_json(directory / "config.json", arm.record)
        write_json(directory / "validation.json", {"metrics": arm.metrics})
        write_json(directory / "test.json", {"metrics": {"prediction_mse": 999.0}})
        for name in ("wandb.json", "startup.json"):
            write_json(directory / name, {})
        (directory / "history.jsonl").write_text("{}\n")
        save_torch(directory / "validation.pt", {"samples": arm.samples})
        save_torch(directory / "test.pt", {"not_read_for_selection": torch.tensor(999)})
        save_torch(
            directory / "last.pt",
            {
                "schema": SCHEMA,
                "identity": arm.record["identity"],
                "step": 100_000,
                "optimizer": {"state": {0: {"step": torch.tensor(100_000)}}},
                "controller": {"num_updates": torch.tensor(100_000)},
            },
        )
        write_json(
            directory / "completion.json",
            arm.completion
            | {"hashes": {name: sha(directory / name) for name in REQUIRED_ARTIFACTS}},
        )
    return [arm.directory for arm in arms.values()]


def bootstrap_indices() -> Tensor:
    return torch.randint(256, (1000, 256), generator=torch.Generator().manual_seed(493))


def test_paired_common_score_and_difference_in_differences_match_known_signals(
    tmp_path: Path,
) -> None:
    result = compare(make_arms(tmp_path), bootstrap_indices())
    assert result["decision"] == "positive_exploratory_preference"
    for seed in result["seeds"]:
        assert math.isclose(seed["common_score_difference_in_differences"]["mean"], 1.8)
        rows = seed["policies"]["full"]["mixed_minus_physical"]
        assert all(math.isclose(value, 2) for value in rows["common_score"]["interval_95"])
        assert all(math.isclose(value, 2) for value in rows["parameter_paths"]["interval_95"])
        assert abs(rows["state_paths"]["mean"]) < 1e-12
        assert seed["policies"]["baseline"]["optimized_path_score"] == "baseline_objective_paths"


def test_parameter_guard_rejects_a_score_gap_entirely_from_other_blocks(tmp_path: Path) -> None:
    result = compare(make_arms(tmp_path, parameter_signal=False), bootstrap_indices())
    assert result["all_quality_pass"]
    assert result["decision"] == "bounded_preference_not_established"
    for seed in result["seeds"]:
        assert math.isclose(seed["common_score_difference_in_differences"]["mean"], 1.8)
        assert not seed["preference_pass"]


def test_common_quality_requires_matched_mse_even_when_every_endpoint_is_accurate(
    tmp_path: Path,
) -> None:
    result = compare(make_arms(tmp_path, mse_shift=0.005), bootstrap_indices())
    assert result["decision"] == "inconclusive_prediction_quality"
    for seed in result["seeds"]:
        for policy in seed["policies"].values():
            assert policy["both_endpoints_meet_quality_bounds"]
            assert not policy["prediction_difference_within_margin"]


def test_bootstrap_resamples_groups_and_keeps_paired_shared_noise_cancelled() -> None:
    noise = torch.arange(256, dtype=torch.float64)
    values = (noise + 0.5) - noise
    assert interval(values, bootstrap_indices()) == {"mean": 0.5, "interval_95": [0.5, 0.5]}


def test_artifact_analysis_authenticates_and_ignores_test_results_for_decision(
    tmp_path: Path,
) -> None:
    directories = write_arms(make_arms(tmp_path / "runs"))
    output = tmp_path / "analysis"
    result = analyze(directories, output)
    assert result["decision"] == "positive_exploratory_preference"
    assert len(result["sources"]) == 8
    assert all(row["verified_artifacts"] == 9 for row in result["sources"])
    assert all(row["metrics"]["prediction_mse"] == 999 for row in result["test_descriptive_only"])
    indices = torch.load(output / "bootstrap_indices.pt", weights_only=True)
    assert indices.shape == (1000, 256)
    with pytest.raises(FileExistsError, match="not empty"):
        analyze(directories, output)
    with (directories[0] / "validation.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="artifact hash"):
        load_arms(directories)


def test_pairing_rejects_a_different_dense_initializer(tmp_path: Path) -> None:
    arms = make_arms(tmp_path / "runs")
    arm = arms[0, "physical", "full"]
    record = deepcopy(arm.record)
    record["initial_state_identity"] = "different-initializer"
    specification = {
        key: value for key, value in record.items() if key not in ("schema", "identity")
    }
    identity = canonical_fingerprint(specification)
    record["identity"] = identity
    completion = arm.completion | {
        "identity": identity,
        "initial_state_identity": "different-initializer",
    }
    arms[0, "physical", "full"] = replace(arm, record=record, completion=completion)
    with pytest.raises(ValueError, match="paired initial_state_identity"):
        load_arms(write_arms(arms))


def test_unpaired_chart_targets_are_rejected_even_with_internally_consistent_metrics(
    tmp_path: Path,
) -> None:
    arms = make_arms(tmp_path / "runs")
    physical = arms[0, "physical", "baseline"]
    for policy in ("baseline", "full"):
        arm = arms[0, "mixed", policy]
        samples = dict(arm.samples)
        samples["targets"] = physical.samples["targets"].clone()
        samples["predictions"] = physical.samples["predictions"].clone()
        samples["group_horizon_mse"] = physical.samples["group_horizon_mse"].clone()
        metrics = arm.metrics | {
            key: physical.metrics[key] for key in ("prediction_mse", "target_variance", "base_loss")
        }
        arms[0, "mixed", policy] = replace(arm, metrics=metrics, samples=samples)
    with pytest.raises(ValueError, match="not the declared fixed chart"):
        load_arms(write_arms(arms))
