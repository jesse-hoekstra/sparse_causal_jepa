"""Gradient, collapse and toy CPU transfer tests for the acquisition-only intervention."""

from dataclasses import asdict, replace
from math import isclose
from pathlib import Path

import torch

from diagnostics.masked_mlp_learned import run as runner
from diagnostics.masked_mlp_learned.budget_control import BudgetConfig
from diagnostics.masked_mlp_learned.continuation import continue_training
from diagnostics.shared import data, source_data
from diagnostics.shared.io import read_json, sha

from .synthetic_loss import _measures  # pyright: ignore[reportPrivateUsage]
from .synthetic_loss_math import (
    AcquisitionConfig,
    acquisition_runtime,
    code_quality,
    paired_intervals,
    within_shares,
)


def test_ratio_scales_and_constant_penalty() -> None:
    """The ratio cancels translations/scales, punishes constants, and permits finite gradients."""
    torch.default_generator.manual_seed(7)
    x = torch.randn(16, 10, 12, dtype=torch.float64).requires_grad_()
    r = within_shares(x)
    torch.testing.assert_close(r, within_shares(-1e-7 * x + 0.012))
    y = torch.cat((x, torch.full_like(x[:, :, :1], 0.1)), dim=-1)
    assert float(within_shares(y)[-1].detach()) == 1
    r.mean().backward()  # pyright: ignore[reportUnknownMemberType]
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())
    assert bool(x.grad.abs().max() > 0)
    group = torch.randn(16, 1, 12).expand(-1, 10, -1)
    assert float(within_shares(group).max()) < 1e-10
    assert code_quality(group)["sixth_share"] > 0


def test_directional_spread_survives_coordinate_shear() -> None:
    """A nearly shared linear mixture can hide episodic variation in per-axis summaries."""
    gen = torch.Generator().manual_seed(17)
    stable = torch.randn(24, 1, 5, dtype=torch.float64, generator=gen).expand(-1, 10, -1)
    noise = torch.randn(24, 10, 1, dtype=torch.float64, generator=gen)
    original = code_quality(torch.cat((stable, noise), dim=-1))
    sheared = code_quality(torch.cat((stable, stable[:, :, :1] + 0.01 * noise), dim=-1))
    assert sheared["within_share"] < original["within_share"] / 100
    assert isclose(
        original["worst_within_fraction"], sheared["worst_within_fraction"], abs_tol=1e-9
    )
    collapsed = code_quality(torch.cat((stable, stable[:, :, :1]), dim=-1))
    assert collapsed["total_numerical_rank"] == 5
    assert collapsed["worst_within_fraction"] < 1e-12


def test_loss_only_and_transfer_on_toy_cpu(tmp_path: Path) -> None:
    """Replay identical a0 tensors with a zero coefficient, then train and transfer treatment."""
    torch.set_num_threads(1)
    raw, cohort = tmp_path / "raw", tmp_path / "cohort"
    source_data.prepare(
        source_data.PrepareConfig(
            str(raw),
            train_groups=8,
            calibration_groups=8,
            validation_groups=6,
            test_groups=2,
            profile="smoke",
        )
    )
    data.prepare(data.PrepareConfig(str(cohort), str(raw), profile="smoke"))
    joint = runner.TrainConfig(
        str(tmp_path / "joint"),
        str(cohort),
        "dense",
        profile="smoke",
        steps=2,
        batch_size=2,
        predictor_activation="gelu",
        code_dim=12,
        initializer="shared_feature",
        log_every=1,
        evaluate_every=2,
        checkpoint_every=1,
    )
    runner.train(joint)
    old = tmp_path / "a0"
    config = replace(
        joint, output_dir=str(old), states_from=joint.output_dir, acquisition_seed=1000
    )
    baseline = runner.train(config)
    orig_identity = runner.code_identity()
    projected = tmp_path / "projected"
    coefficient = read_json(Path(joint.output_dir) / "calibration.json")["metrics"]["prediction"]
    for weight in (0.0, coefficient):
        out = tmp_path / f"acquisition-{weight}"
        args = asdict(replace(config, output_dir=str(out)))
        new_cfg = AcquisitionConfig(
            **args,
            initialization_sha256=sha(old / "initial.pt"),
            joint_calibration_prediction=weight,
        )
        with acquisition_runtime(old, weight, restore_initial=True):
            record = runner.train(new_cfg)
            assert record["code_identity"] != orig_identity
            assert record["initial_model_identity"] == baseline["initial_model_identity"]
            spec = read_json(out / "config.json")["scientific_config"]
            assert spec["objective"]["within_total_weight"] == weight
            actual = read_json(out / "validation.json")["metrics"]["prediction"]
            previous = read_json(old / "validation.json")["metrics"]["prediction"]
            if weight == 0:
                assert isclose(actual, previous, abs_tol=1e-7)
                _, current = _measures(out)
                _, control = _measures(old)
                ci = paired_intervals(current["validation"], control["validation"])
                assert ci["within_share_95"] == [0.0, 0.0]
                assert ci["prediction_mse_95"] == [0.0, 0.0]
            else:
                assert actual != previous
                runner.project(
                    runner.ProjectConfig(
                        str(projected), str(out), str(cohort), profile="smoke", batch_size=2
                    )
                )
        assert runner.code_identity() == orig_identity
    adapted = replace(
        joint,
        output_dir=str(tmp_path / "adapt"),
        adapt_from=str(projected),
        initializer=None,
        code_dim=None,
    )
    runner.train(adapted)
    bound = tmp_path / "tau"
    runner.calibrate(runner.CalibrateConfig(str(bound), adapted.output_dir, profile="smoke"))
    for arm in ("full", "baseline"):
        first = tmp_path / f"{arm}-parent"
        sparse = BudgetConfig(
            str(first),
            str(cohort),
            arm,
            profile="smoke",
            steps=2,
            batch_size=2,
            predictor_activation="gelu",
            init_dir=adapted.output_dir,
            calibration_dir=str(bound),
            log_every=1,
            evaluate_every=2,
            checkpoint_every=1,
        )
        runner.train(sparse)
        continued = continue_training(first, tmp_path / f"{arm}-end", additional_steps=2)
        assert continued["steps"] == 4
        assert continued["code_identity"] == orig_identity
