"""Exercise the constituent lock, combined recovery decision, and a real toy warm fork."""

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from diagnostics.masked_mlp_learned import run as runner
from diagnostics.masked_mlp_learned.budget_control import BudgetConfig
from diagnostics.masked_mlp_learned.continuation import continue_training
from diagnostics.shared import data, source_data
from diagnostics.shared.io import read_json, sha
from scjepa.experiments.storage import write_json

from . import synthetic_combined as combo
from .synthetic_combined import FAMILIES, combined_checks, eligibility, numerical_replay
from .synthetic_consensus_runtime import train_child
from .synthetic_warm import train_warm


def test_cohort_gate_preserves_each_denominator_and_both_pilots() -> None:
    """Four observed passes cannot authorize an incomplete family or override a failed pilot."""
    rows: dict[str, list[dict[str, Any]]] = {
        family: [{"seed": n, "passed": n != 3, "commit": "original"} for n in range(5)]
        for family in FAMILIES
    }
    assert eligibility(rows)["status"] == "eligible"
    rows[FAMILIES[0]][0]["passed"] = False
    rows[FAMILIES[0]][3]["passed"] = True
    assert eligibility(rows)["status"] == "ineligible"
    rows[FAMILIES[0]][0]["passed"] = True
    rows[FAMILIES[1]][2]["passed"] = False
    assert eligibility(rows)["status"] == "ineligible"
    rows[FAMILIES[1]][2]["passed"] = True
    rows[FAMILIES[1]].pop(3)
    assert eligibility(rows)["status"] == "pending"
    rows[FAMILIES[1]].append(dict(rows[FAMILIES[1]][0]))
    with pytest.raises(ValueError, match=FAMILIES[1]):
        eligibility(rows)


def test_interaction_requires_warm_baseline_and_both_heldout_forecasts() -> None:
    """A favorable full score cannot erase failed calibration or a small paired warm MCC gain."""
    warm = {
        "worst_within_fraction": 0.40,
        "sixth_share": 0.20,
        "prediction": 0.015,
        "total_numerical_rank": 6,
        "code_r2_min": 0.93,
    }
    improved = warm | {"worst_within_fraction": 0.30}
    records = {
        "native": {"calibration": warm, "validation": warm},
        "other": {"calibration": dict(improved), "validation": dict(improved)},
    }
    full = {
        "state_mcc": 0.95,
        "masks/state_aligned_shd": 0.0,
        "masks/action_aligned_shd": 1.0,
        "masks/parameter_code_aligned_shd": 0.0,
        "normalized_prediction_mse": 0.02,
        "code_r2_min": 0.93,
    }
    reference = {"full": full, "baseline": {"state_mcc": 0.70}}
    measured = {"full": full, "baseline": {"state_mcc": 0.73}}
    assert all(combined_checks(records, measured, reference).values())
    measured["baseline"]["state_mcc"] = 0.80
    records["other"]["calibration"]["prediction"] = 0.018
    records["other"]["validation"]["code_r2_min"] = 0.90
    failed = {k for k, p in combined_checks(records, measured, reference).items() if not p}
    assert failed == {"paired_mcc_gain", "calibration_forecast", "warm_parameter_retention"}


def test_genuine_warm_native_replay_and_sibling_cpu(tmp_path: Path) -> None:
    """Fork a warm acquisition with its native optimizer, sampler and GECO state."""
    torch.set_num_threads(1)
    source, cohort = tmp_path / "physical", tmp_path / "data"
    source_data.prepare(
        source_data.PrepareConfig(
            str(source),
            train_groups=10,
            calibration_groups=8,
            validation_groups=6,
            test_groups=2,
            profile="smoke",
        )
    )
    data.prepare(data.PrepareConfig(str(cohort), str(source), profile="smoke"))
    joint = runner.TrainConfig(
        str(tmp_path / "joint"),
        str(cohort),
        "dense",
        profile="smoke",
        steps=2,
        batch_size=2,
        code_dim=12,
        predictor_activation="gelu",
        log_every=1,
        evaluate_every=20,
        checkpoint_every=1,
    )
    runner.train(joint)
    acquisition = replace(
        joint,
        output_dir=str(tmp_path / "acquire"),
        states_from=joint.output_dir,
        acquisition_seed=1000,
        initializer="shared_feature",
    )
    train_warm(acquisition)
    projected = tmp_path / "project"
    runner.project(
        runner.ProjectConfig(
            str(projected), acquisition.output_dir, str(cohort), profile="smoke", batch_size=2
        )
    )
    bound = tmp_path / "bound"
    runner.calibrate(
        runner.CalibrateConfig(str(bound), str(projected), tau_factor=1.0, profile="smoke")
    )
    adapt = replace(
        joint, output_dir=str(tmp_path / "adapt"), code_dim=None, adapt_from=str(projected)
    )
    runner.train(adapt)
    parent = tmp_path / "full-parent"
    runner.train(
        BudgetConfig(
            str(parent),
            str(cohort),
            "full",
            profile="smoke",
            steps=2,
            batch_size=2,
            predictor_activation="gelu",
            calibration_dir=str(bound),
            init_dir=adapt.output_dir,
            log_every=1,
            evaluate_every=20,
            checkpoint_every=1,
        )
    )
    native, child = tmp_path / "native", tmp_path / "sibling"
    original = sha(parent / "last.pt")
    continue_training(parent, native, additional_steps=3, device="cpu", threads=1)
    replay = numerical_replay(parent, native, tmp_path / "replay", steps=2, budget=3, device="cpu")
    assert replay["matched"]
    assert set(replay["measured"]) == {"3", "4"}
    kid = train_child(parent, child, "other", steps=3, device="cpu", resume=False)
    assert kid["steps"] == kid["successful_updates"] == 5
    assert kid["hashes"]["last.pt"] != sha(native / "last.pt")
    assert (
        sha(parent / "last.pt")
        == original
        == read_json(child / "continuation.json")["parent_checkpoint_sha256"]
    )
    assert (
        read_json(child / "config.json")["scientific_config"]["frozen_teacher"]["target"] == "other"
    )


def test_missing_native_parent_remains_an_explicit_combined_failure(tmp_path: Path) -> None:
    """Count a licensed seed without a warm 300k parent and refuse an unlicensed expansion."""
    root, dataset = tmp_path / "runs", tmp_path / "cohort"
    source = root / "synthetic-warm-s1"
    ledger = root / "synthetic-combined-cohort/metrics.json"
    dataset.mkdir()
    source.mkdir(parents=True)
    ledger.parent.mkdir()
    write_json(dataset / "completion.json", {"identity": "toy"})
    write_json(source / "metrics.json", {"reason": "train_admission"})
    evidence = [{"seed": 0}, {"seed": 1, "sha256": sha(source / "metrics.json")}]
    cohort = {"status": "eligible", "commit": "toy", "constituents": {FAMILIES[0]: evidence}}
    write_json(ledger, cohort)
    pilot = root / "synthetic-combined-s0/metrics.json"
    pilot.parent.mkdir()
    pilot_row = {
        "seed": 0,
        "commit": "toy",
        "cohort_sha256": sha(ledger),
        "checks": {"criterion": False},
        "passed": False,
    }
    write_json(pilot, pilot_row)
    with (
        patch.object(combo.schedule, "PRIVATE", root),
        patch.object(combo.schedule, "DATA", dataset),
        patch.object(combo.consensus, "check_head"),
        patch.object(combo, "_cohort", return_value=cohort),
        patch.object(
            combo.consensus,
            "train_phase",
            side_effect=AssertionError("missing parent reached training"),
        ),
    ):
        with pytest.raises(ValueError, match="passing combined"):
            combo.run("toy", 1, "run")
        write_json(pilot, pilot_row | {"passed": True, "checks": {"criterion": True}})
        report = combo.run("toy", 1, "run")
        assert report["reason"] == "unavailable_warm_parent"
        assert report["passed"] is False
        assert report == combo.run("toy", 1, "run")
