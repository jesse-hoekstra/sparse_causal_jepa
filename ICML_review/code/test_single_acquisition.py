"""Gate checks and a tiny CPU rehearsal of first-acquisition training and continuation."""

from dataclasses import replace
from pathlib import Path

from diagnostics.masked_mlp_learned import run as runner
from diagnostics.masked_mlp_learned.budget_control import BudgetConfig
from diagnostics.masked_mlp_learned.continuation import continue_training
from diagnostics.shared import data, source_data

from .single_acquisition import decision


def test_pilot_requires_retention_and_paired_gain() -> None:
    """A strong full endpoint can still lose the paired contrast or a physical parameter."""
    full = {
        "state_mcc": 0.91,
        "normalized_prediction_mse": 0.02,
        "code_r2_min": 0.95,
        "masks/state_aligned_shd": 1.0,
        "masks/action_aligned_shd": 0.0,
        "masks/parameter_code_aligned_shd": 0.0,
    }
    original = {"full": full, "baseline": {"state_mcc": 0.65}}
    assert all(decision(original, original).values())
    control = {"full": full | {"code_r2_min": 0.7}, "baseline": {"state_mcc": 0.85}}
    checks = decision(control, original)
    assert not checks["parameter_retention"]
    assert not checks["paired_mcc_gain"]


def test_tiny_cpu_stages(tmp_path: Path) -> None:
    """Run GELU projection, adaptation, both sparse arms and exact continuations on toy groups."""
    physical, cohort = tmp_path / "physical", tmp_path / "cohort"
    source_data.prepare(
        source_data.PrepareConfig(
            str(physical),
            train_groups=4,
            calibration_groups=4,
            validation_groups=2,
            test_groups=2,
            profile="smoke",
        )
    )
    data.prepare(data.PrepareConfig(str(cohort), str(physical), profile="smoke"))
    joint = runner.TrainConfig(
        str(tmp_path / "joint"),
        str(cohort),
        "dense",
        profile="smoke",
        steps=2,
        batch_size=2,
        predictor_activation="gelu",
        code_dim=12,
        log_every=1,
        evaluate_every=2,
        checkpoint_every=1,
    )
    runner.train(joint)
    acquisition = replace(
        joint, output_dir=str(tmp_path / "a0"), states_from=joint.output_dir, acquisition_seed=1000
    )
    runner.train(acquisition)
    projected = str(tmp_path / "projected")
    runner.project(
        runner.ProjectConfig(
            projected, acquisition.output_dir, str(cohort), profile="smoke", batch_size=2
        )
    )
    bound = str(tmp_path / "fixed-bound")
    runner.calibrate(runner.CalibrateConfig(bound, projected, tau_factor=1.0, profile="smoke"))
    dense = replace(joint, output_dir=str(tmp_path / "dense"), code_dim=None, adapt_from=projected)
    runner.train(dense)
    for arm in ("baseline", "full"):
        parent = tmp_path / (arm + "-parent")
        cfg = BudgetConfig(
            str(parent),
            str(cohort),
            arm,
            profile="smoke",
            steps=2,
            batch_size=2,
            predictor_activation="gelu",
            init_dir=dense.output_dir,
            calibration_dir=bound,
            log_every=1,
            evaluate_every=2,
            checkpoint_every=1,
        )
        runner.train(cfg)
        result = continue_training(parent, tmp_path / (arm + "-end"), additional_steps=2)
        assert result["steps"] == 4
        assert result["predictor_activation"] == "gelu"
