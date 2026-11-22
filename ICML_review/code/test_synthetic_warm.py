"""Exercise the joint-head synthetic acquisition and its native downstream training on CPU."""

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.masked_mlp_learned import run as runner
from diagnostics.masked_mlp_learned.budget_control import BudgetConfig
from diagnostics.masked_mlp_learned.continuation import continue_training
from diagnostics.masked_mlp_learned.selection import select
from diagnostics.shared import data, source_data
from diagnostics.shared.io import read_json, sha

from .synthetic_warm import train_warm


def test_joint_head_and_cold_control_reach_native_forks(tmp_path: Path) -> None:
    """Preserve frozen joint tensors, original cold data stream and both genuine sparse forks."""
    cohort, source = tmp_path / "data", tmp_path / "physical"
    source_data.prepare(
        source_data.PrepareConfig(
            str(source),
            train_groups=6,
            calibration_groups=6,
            validation_groups=4,
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
        predictor_activation="gelu",
        log_every=1,
        checkpoint_every=1,
        evaluate_every=3,
        code_dim=12,
    )
    runner.train(joint)
    ctrl = replace(
        joint,
        states_from=joint.output_dir,
        acquisition_seed=1000,
        output_dir=str(tmp_path / "cold"),
        initializer="shared_feature",
    )
    runner.train(ctrl)
    cfg = replace(ctrl, output_dir=str(tmp_path / "warm"))
    warm = train_warm(cfg)
    assert train_warm(cfg, resume=True)["identity"] == warm["identity"]

    def ck(name: str) -> dict[str, Any]:
        return cast(
            dict[str, Any], torch.load(tmp_path / name, map_location="cpu", weights_only=False)
        )

    s0, cold, hot, start = (
        ck("joint/last.pt"),
        ck("cold/last.pt"),
        ck("warm/last.pt"),
        ck("warm/initial.pt"),
    )
    torch.testing.assert_close(cold["sampler"], hot["sampler"], rtol=0, atol=0)
    torch.testing.assert_close(start["model"], s0["model"], rtol=0, atol=0)
    for k, v in s0["model"].items():
        if k.startswith(runner.STATE_ENCODERS):
            assert torch.equal(hot["model"][k], v)
    assert any(
        not torch.equal(hot["model"][k], v)
        for k, v in start["model"].items()
        if k.startswith("parameter_encoder.")
    )
    assert any(
        not torch.equal(cold["model"][k], v)
        for k, v in hot["model"].items()
        if k.startswith("predictor.")
    )
    warm_selection = select([Path(cfg.output_dir)], window=2, maximum_loss=100.0)
    cold_selection = select([Path(ctrl.output_dir)], window=2, maximum_loss=100.0)
    assert warm_selection["common"] == cold_selection["common"]
    assert warm_selection["admitted"]
    assert not select([Path(cfg.output_dir)], window=2, maximum_loss=1e-15)["admitted"]
    proj = runner.project(
        runner.ProjectConfig(
            str(tmp_path / "project"), cfg.output_dir, str(cohort), profile="smoke", batch_size=2
        )
    )
    assert proj["source"]["checkpoint_sha256"] == sha(tmp_path / "warm/last.pt")
    cap = tmp_path / "bound"
    runner.calibrate(
        runner.CalibrateConfig(str(cap), str(tmp_path / "project"), tau_factor=1.0, profile="smoke")
    )
    adapt = replace(
        joint,
        output_dir=str(tmp_path / "adapt"),
        code_dim=None,
        adapt_from=str(tmp_path / "project"),
    )
    runner.train(adapt)
    for arm in ("full", "baseline"):
        folder = tmp_path / arm
        runner.train(
            BudgetConfig(
                str(folder),
                str(cohort),
                arm,
                profile="smoke",
                seed=0,
                steps=2,
                batch_size=2,
                predictor_activation="gelu",
                calibration_dir=str(cap),
                init_dir=adapt.output_dir,
                log_every=1,
                evaluate_every=3,
                checkpoint_every=1,
            )
        )
        end = tmp_path / f"{arm}-end"
        final = continue_training(folder, end, additional_steps=1)
        assert final["steps"] == 3
        assert read_json(end / "continuation-completion.json")["parent_checkpoint_sha256"] == sha(
            folder / "last.pt"
        )
