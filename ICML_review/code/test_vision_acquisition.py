"""Check direct projection of a joint RGB model through both sparse endpoints on CPU."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from diagnostics.vision_dimer import data, train

from .vision_acquisition import TRAIN, qualification, sparse_checks, stage_args


def test_direct_joint_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the child's release CLI on tiny data with no independent acquisition."""
    torch.set_num_threads(1)
    cohort = tmp_path / "data"
    data.prepare(
        cohort,
        data.DataConfig(
            train_groups=8,
            calibration_groups=8,
            validation_groups=4,
            test_groups=2,
            episode_frames=12,
        ),
    )
    joint = train.StageConfig(
        "joint",
        str(cohort),
        str(tmp_path / "joint"),
        device="cpu",
        steps=2,
        batch_groups=2,
        log_every=1,
        eval_every=2,
        checkpoint_every=2,
    )

    def invoke(args: list[str]) -> None:
        monkeypatch.setattr(sys, "argv", [TRAIN[-1], *args])
        train.main()

    invoke(["train", *stage_args(joint)])
    projected = tmp_path / "projected"
    invoke(
        [
            "project",
            "--source",
            joint.output_dir,
            "--data-dir",
            str(cohort),
            "--output",
            str(projected),
            "--device",
            "cpu",
        ]
    )
    adapt = replace(
        joint,
        stage="adapt",
        code_dim=4,
        parent=str(projected / "final.pt"),
        output_dir=str(tmp_path / "adapt"),
    )
    invoke(["train", *stage_args(adapt)])
    for arm in ("full", "baseline"):
        out = tmp_path / arm
        sparse = replace(
            adapt, stage=arm, tau=0.02, parent=str(tmp_path / "adapt/final.pt"), output_dir=str(out)
        )
        invoke(["train", *stage_args(sparse)])
        end = tmp_path / f"{arm}-continued"
        invoke(["fork", "--source", str(out / "last.pt"), "--output-dir", str(end), "--steps", "3"])
        assert train.load_checkpoint(end / "last.pt")["step"] == 3


def test_decisions_are_paired_and_protect_forecast() -> None:
    """A favorable MCC cannot excuse poor forecasts."""
    reference = {
        "prediction_mse": 1.0,
        "nmse": 1.0,
        "mcc": 0.9,
        "state_shd": 1.0,
        "action_shd": 0.0,
        "parameter_shd": 0.0,
        "parameter_r2_min": 0.97,
    }
    base = {"full": reference, "baseline": reference | {"mcc": 0.6}}
    assert all(qualification(reference, reference).values())
    assert all(sparse_checks(base, base).values())
    assert not sparse_checks(base | {"full": reference | {"nmse": 1.2}}, base)["nmse"]
