"""Exercise both loss schedules through every synthetic stage on tiny CPU data."""

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from diagnostics.masked_mlp_learned import run as runner
from diagnostics.masked_mlp_learned.budget_control import BudgetConfig
from diagnostics.masked_mlp_learned.continuation import continue_training
from diagnostics.shared import data, source_data
from diagnostics.shared.io import read_json, sha

from .timing_synthetic import guidance, original_start


@pytest.mark.parametrize("schedule", ["entire", "sparse"])
def test_training_schedule_through_native_continuation(tmp_path: Path, schedule: str) -> None:
    """Check native initialization, one cold candidate, fixed TRAIN teacher and actual step logs."""
    torch.set_num_threads(1)
    raw, cohort = tmp_path / "raw", tmp_path / "data"
    source_data.prepare(
        source_data.PrepareConfig(
            str(raw),
            train_groups=16,
            calibration_groups=12,
            validation_groups=10,
            test_groups=2,
            profile="smoke",
        )
    )
    data.prepare(data.PrepareConfig(str(cohort), str(raw), profile="smoke"))
    joint = runner.TrainConfig(
        str(tmp_path / "control"),
        str(cohort),
        "dense",
        profile="smoke",
        seed=0,
        steps=2,
        batch_size=4,
        code_dim=12,
        initializer="shared_feature",
        predictor_activation="gelu",
        evaluate_every=2,
        log_every=1,
        checkpoint_every=1,
    )
    original = runner.train(joint)
    joint_path = Path(joint.output_dir)
    if schedule == "entire":
        joint_path = tmp_path / "joint"
        with original_start(Path(joint.output_dir)), guidance(None, "cpu"):
            fitted = runner.train(replace(joint, output_dir=str(joint_path)))
        assert fitted["initial_model_identity"] == original["initial_model_identity"]
        assert fitted["identity"] != original["identity"]
    acquire, project, adapt = (tmp_path / s for s in ("acquire", "project", "adapt"))
    with guidance(None, "cpu") if schedule == "entire" else nullcontext():
        receipt = runner.train(
            replace(
                joint, output_dir=str(acquire), states_from=str(joint_path), acquisition_seed=1000
            )
        )
    assert receipt["state_source"]["checkpoint_sha256"] == sha(joint_path / "last.pt")
    runner.project(
        runner.ProjectConfig(str(project), str(acquire), str(cohort), profile="smoke", batch_size=4)
    )
    with guidance(None, "cpu") if schedule == "entire" else nullcontext():
        runner.train(
            replace(
                joint,
                output_dir=str(adapt),
                code_dim=None,
                initializer=None,
                adapt_from=str(project),
            )
        )
    tau = tmp_path / "tau"
    runner.calibrate(runner.CalibrateConfig(str(tau), str(adapt), profile="smoke"))
    parent, final = tmp_path / "sparse", tmp_path / "end"
    config = BudgetConfig(
        str(parent),
        str(cohort),
        "full",
        profile="smoke",
        steps=2,
        batch_size=4,
        init_dir=str(adapt),
        calibration_dir=str(tau),
        predictor_activation="gelu",
        log_every=1,
        evaluate_every=2,
        checkpoint_every=1,
    )
    with guidance(adapt, "cpu"):
        runner.train(config)
    with guidance(adapt, "cpu"):
        terminal = continue_training(parent, final, additional_steps=2)
    assert terminal["steps"] == 4
    for path in (joint_path, acquire, adapt, parent, final):
        history = [json.loads(row) for row in (path / "history.jsonl").read_text().splitlines()]
        rows = [r for r in history if "loss/prediction" in r]
        assert len(rows) == (4 if path == final else 2)
        for row in rows:
            assert ("loss/consensus_loss" in row) == (
                path in (parent, final) or schedule == "entire"
            )
    for path in (parent, final):
        recipe = read_json(path / "config.json")["scientific_config"]["sibling_guidance"]
        assert recipe["teacher"] == "frozen_adapted"
        assert recipe["checkpoint_sha256"] == sha(adapt / "last.pt")
    assert read_json(final / "continuation-completion.json")["parent_checkpoint_sha256"] == sha(
        parent / "last.pt"
    )
    assert not (final / "test.json").exists()
