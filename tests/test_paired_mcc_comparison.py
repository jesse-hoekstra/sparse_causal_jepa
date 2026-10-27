"""Paired independent-group MCC reporting at the evaluation artifact boundary."""

import hashlib
import math
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from scjepa.experiments.comparison import paired_mcc_comparison
from scjepa.experiments.storage import write_json


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _conditions(tmp_path: Path, features: tuple[Tensor, Tensor, Tensor]) -> dict[str, Path]:
    generator = torch.Generator().manual_seed(19)
    truth = torch.randn(8, 2, 4, 2, generator=generator)
    directories: dict[str, Path] = {}
    for name, values in zip(("dense", "baseline", "full"), features, strict=True):
        directory = tmp_path / name
        evaluation = directory / "evaluation"
        evaluation.mkdir(parents=True)
        checkpoint = directory / "last.pt"
        checkpoint.write_bytes(name.encode())
        samples = evaluation / "mcc_samples.pt"
        torch.save(
            {
                "format_version": 1,
                "group_ids": torch.arange(8),
                "calibration_group_ids": torch.arange(100, 108),
                "states": truth,
                "online": values,
                "state_match": torch.tensor([0, 1]),
            },
            samples,
        )
        write_json(directory / "status.json", {"status": "completed"})
        write_json(
            directory / "run_config.json",
            {
                "run_identity": name,
                "dataset_identity": "dataset",
                "experiment_identity": "experiment",
                "seed": 4,
                "train": {"steps": 100},
            },
        )
        write_json(
            evaluation / "evaluated_checkpoint.json",
            {
                "run_identity": name,
                "checkpoint_path": "last.pt",
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "mcc_samples_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
                "evaluated_step": 100,
            },
        )
        write_json(
            evaluation / "metrics.json",
            {
                "graph/state_shd": 0.0,
                "graph/parameter_shd": 24.0 if name == "baseline" else 0.0,
                "test/prediction_mse": 0.1,
                "test/base_loss": 0.2,
                **({"geco/tau": 0.3, "test/feasible": 1.0} if name != "dense" else {}),
                "test/shuffled_parameter_prediction_mse_delta": 0.01,
                "recovery/parameter_mcc": 0.9,
            },
        )
        directories[name] = directory
    return directories


def _truth() -> Tensor:
    return torch.randn(8, 2, 4, 2, generator=torch.Generator().manual_seed(19))


def test_identical_conditions_have_exact_zero_paired_contrasts(tmp_path: Path) -> None:
    truth = _truth()
    directories = _conditions(tmp_path, (truth, truth, truth))
    result = paired_mcc_comparison(directories, fixed_budget=100, resamples=32, seed=2)
    for value in _mapping(result["contrasts"]).values():
        contrast = _mapping(value)
        assert contrast["estimate"] == 0.0
        assert contrast["interval_95"] == [0.0, 0.0]
        assert contrast["interval_above_zero"] is False
        assert contrast["interpretation"] == "inconclusive"
    assert result["state_shd_guardrail"] is True
    assert result["uncertainty_unit"] == "independent parameter group"
    assert "conditional on training seed" in str(result["interpretation"])
    conditions = _mapping(result["conditions"])
    baseline = _mapping(conditions["baseline"])
    assert baseline["parameter_routing_shd_descriptive"] == 24.0
    assert baseline["geco_feasible"] is True


def test_positive_contrast_preserves_frozen_match_and_group_draws(tmp_path: Path) -> None:
    truth = _truth()
    dense = truth.flip(-1)
    baseline = truth + 0.4 * torch.randn(truth.shape, generator=torch.Generator().manual_seed(7))
    directories = _conditions(tmp_path, (dense, baseline, truth))
    result = paired_mcc_comparison(directories, fixed_budget=100, resamples=64, seed=2)
    contrasts = _mapping(result["contrasts"])
    for value in contrasts.values():
        contrast = _mapping(value)
        assert contrast["interval_above_zero"] is True
        assert contrast["interpretation"] == "positive effect"
    conditions = _mapping(result["conditions"])
    # A test-time rematch would score the swapped dense features as perfect.
    assert cast(float, _mapping(conditions["dense"])["state_mcc"]) < 0.5
    assert math.isclose(cast(float, _mapping(conditions["full"])["state_mcc"]), 1.0)
    draws = torch.tensor(cast(list[list[int]], result["resampled_group_indices"]))
    assert draws.shape == (64, 8)  # eight independent groups, not sixteen episodes
    assert torch.equal(draws, torch.randint(8, (64, 8), generator=torch.Generator().manual_seed(2)))
    assert result == paired_mcc_comparison(directories, fixed_budget=100, resamples=64, seed=2)


def test_mcc_evaluation_writes_grouped_samples_and_calibration_assignment(tmp_path: Path) -> None:
    from scjepa.data.controlled_vector import VectorObservationMap
    from scjepa.data.types import DGPConfig
    from scjepa.experiments.evaluation import EvaluationConfig, evaluate_model
    from scjepa.training.factory import ModelConfig, build_model

    model = build_model(
        ModelConfig(
            observation_dim=10,
            state_dim=10,
            param_dim=10,
            action_dim=10,
            context_len=2,
            history_len=1,
            parameter_embed_dim=8,
            parameter_heads=2,
            spartan_embed_dim=8,
            spartan_layers=1,
        )
    )
    data = DGPConfig(trajectory_length=4, episodes_per_parameter=2)
    config = EvaluationConfig(
        calibration_episodes=4, validation_episodes=4, test_episodes=4, batch_size=2
    )
    evaluate_model(model, data, VectorObservationMap().matrix, config, tmp_path)
    payload = cast(dict[str, Tensor], torch.load(tmp_path / "mcc_samples.pt", weights_only=True))
    assert payload["states"].shape == (4, 2, 4, 10)
    assert payload["online"].shape == (4, 2, 4, 10)
    assert payload["group_ids"].shape == (4,)
    assert not bool(torch.isin(payload["group_ids"], payload["calibration_group_ids"]).any())
    references = cast(
        dict[str, Tensor], torch.load(tmp_path / "evaluation_references.pt", weights_only=True)
    )
    assert torch.equal(payload["state_match"], references["state_match"])


@pytest.mark.parametrize("mutation", ["budget", "checkpoint", "sample", "group", "overlap"])
def test_rejects_unmatched_groups_or_unverified_final_artifacts(
    tmp_path: Path, mutation: str
) -> None:
    from scjepa.experiments.storage import read_json

    truth = _truth()
    directories = _conditions(tmp_path, (truth, truth, truth))
    directory = directories["full"]
    evaluation = directory / "evaluation"
    provenance = read_json(evaluation / "evaluated_checkpoint.json")
    if mutation == "budget":
        provenance["evaluated_step"] = 90
    elif mutation == "checkpoint":
        (directory / "last.pt").write_bytes(b"changed")
    elif mutation == "sample":
        (evaluation / "mcc_samples.pt").write_bytes(b"changed")
    else:
        path = evaluation / "mcc_samples.pt"
        payload = cast(dict[str, Tensor], torch.load(path, weights_only=True))
        payload["group_ids"] = payload["group_ids"] + (1 if mutation == "group" else 100)
        torch.save(payload, path)
        provenance["mcc_samples_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(evaluation / "evaluated_checkpoint.json", provenance)
    with pytest.raises(ValueError, match="fixed-budget|identity|unmatched|overlap"):
        paired_mcc_comparison(directories, fixed_budget=100, resamples=8)


def test_repeated_episodes_do_not_increase_bootstrap_independent_units(tmp_path: Path) -> None:
    from scjepa.experiments.storage import read_json

    truth = _truth()
    features = truth + 0.4 * torch.randn(truth.shape, generator=torch.Generator().manual_seed(7))
    directories = _conditions(tmp_path, (features, truth, truth))
    before = paired_mcc_comparison(directories, fixed_budget=100, resamples=32, seed=2)
    for directory in directories.values():
        path = directory / "evaluation/mcc_samples.pt"
        payload = cast(dict[str, Tensor], torch.load(path, weights_only=True))
        for key in ("states", "online"):
            payload[key] = payload[key].repeat_interleave(5, dim=1)
        torch.save(payload, path)
        provenance_path = directory / "evaluation/evaluated_checkpoint.json"
        provenance = read_json(provenance_path)
        provenance["mcc_samples_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(provenance_path, provenance)
    after = paired_mcc_comparison(directories, fixed_budget=100, resamples=32, seed=2)
    assert after["resampled_group_indices"] == before["resampled_group_indices"]
    old = _mapping(_mapping(before["contrasts"])["baseline_minus_dense"])
    new = _mapping(_mapping(after["contrasts"])["baseline_minus_dense"])
    torch.testing.assert_close(
        torch.tensor(old["bootstrap_differences"]), torch.tensor(new["bootstrap_differences"])
    )


def test_existing_aggregation_includes_new_group_comparisons(tmp_path: Path) -> None:
    from scjepa.experiments.config import ExperimentConfig
    from scjepa.experiments.reporting import aggregate_results
    from scjepa.experiments.storage import read_json

    truth = _truth()
    root = tmp_path / "complementary/seed-4"
    directories = _conditions(root, (truth, truth, truth))
    for name, directory in directories.items():
        record_path = directory / "run_config.json"
        record = read_json(record_path)
        record.update({"initial_state_fingerprint": "initial", "world_size": 1})
        record["tau_calibration"] = (
            None if name == "dense" else {"tau": 0.3, "dense_run_identity": "dense"}
        )
        write_json(record_path, record)
    config = ExperimentConfig(output_dir=str(tmp_path), graphs=("complementary",), seeds=(4,))
    config.train.steps = 100
    summary = aggregate_results(config)
    report = _mapping(_mapping(summary["paired_group_mcc"])["complementary/seed-4"])
    assert report["state_shd_guardrail"] is True
    assert summary["missing_group_mcc"] == []
    assert read_json(tmp_path / "paired_mcc/complementary/seed-4.json") == report


def test_group_bootstrap_matches_existing_pearson_metric_on_resampled_rows(tmp_path: Path) -> None:
    from scjepa.eval.recovery import CoordinateMatch, mean_correlation_coefficient

    truth = _truth()
    noise = torch.randn(truth.shape, generator=torch.Generator().manual_seed(7))
    directories = _conditions(tmp_path, (truth, truth + noise, truth + 2 * noise))
    result = paired_mcc_comparison(directories, fixed_budget=100, resamples=16, seed=3)
    contrast = _mapping(_mapping(result["contrasts"])["baseline_minus_dense"])
    values = cast(list[float], contrast["bootstrap_differences"])
    draws = cast(list[list[int]], result["resampled_group_indices"])
    for indices, value in zip(draws, values, strict=True):
        expected = (
            mean_correlation_coefficient(
                CoordinateMatch((0, 1)),
                (truth + noise)[indices].flatten(0, 2),
                truth[indices].flatten(0, 2),
            )
            - 1
        )
        assert math.isclose(value, expected, abs_tol=1e-12)
    assert contrast["interpretation"] == "observed negative effect"
    assert contrast["interval_above_zero"] is False
