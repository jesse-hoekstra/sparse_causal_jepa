"""Run both RGB timing schedules on real rendered inputs and inspect their actual updates."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from diagnostics.vision_dimer import data as vdata
from diagnostics.vision_dimer import train as release

from .timing_vision_train import frozen_teacher, restore_log, train_stage
from .vision_invariant_eval import endpoint


@pytest.mark.parametrize("schedule", ["entire", "sparse"])
def test_timing_on_real_frames(tmp_path: Path, schedule: str) -> None:
    """Exercise all four native optimizers, including cached EMA targets and frozen lookup."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _train_and_check(tmp_path, schedule)
    finally:
        torch.set_num_threads(threads)


def _train_and_check(out: Path, schedule: str) -> None:
    data = out / "data"
    vdata.prepare(
        data,
        vdata.DataConfig(
            episode_frames=14, train_groups=9, calibration_groups=6, validation_groups=4
        ),
    )
    common = release.StageConfig(
        "joint",
        str(data),
        str(out / "joint"),
        seed=3,
        device="cpu",
        batch_groups=2,
        steps=2,
        log_every=1,
        checkpoint_every=1,
        eval_every=1,
    )
    parent = None
    for stage in ("joint", "acquire", "adapt", "full"):
        if stage == "adapt":
            assert parent is not None
            release.project(parent.parent, data, out / "projected", device="cpu")
            parent = out / "projected/final.pt"
        directory = out / stage
        cfg = replace(
            common,
            stage=stage,
            output_dir=str(directory),
            parent=str(parent) if parent else None,
            acquisition_seed=1003 if stage == "acquire" else None,
            code_dim=8 if stage in ("joint", "acquire") else 4,
            tau=0.05 if stage == "full" else None,
        )
        teacher = (
            frozen_teacher(parent, data, torch.device("cpu"), directory)
            if stage == "full" and parent
            else None
        )
        train_stage(cfg, schedule, teacher)
        parent = directory / "final.pt"
        aux_path = directory / "sibling-history.jsonl"
        if schedule == "entire" or stage == "full":
            native = [
                json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()
            ]
            sibling = [json.loads(line) for line in aux_path.read_text().splitlines()]
            assert len(sibling) == len(native) == cfg.steps
            for row, reference in zip(sibling, native, strict=True):
                assert row["step"] == reference["step"]
                assert row["updates"] == 1
                assert row["forecast_mse"] == reference["prediction"]
                assert row["consensus_loss"] > 0
                assert 0 < row["consensus_gradient_scale"] <= 1
        else:
            assert not aux_path.exists()
    assert parent is not None
    result = endpoint(parent, data, torch.device("cpu"))
    assert all(result[s]["mse_matched"] for s in ("calibration", "validation"))


def test_interrupted_history_preserves_evidence(tmp_path: Path) -> None:
    """Roll back to the checkpoint and retain the discarded and incomplete rows."""
    history = tmp_path / "history.jsonl"
    before = b'{"step":1}\n{"step":2}\n{"ste'
    history.write_bytes(before)
    restore_log(history, 1)
    assert history.read_bytes() == b'{"step":1}\n'
    (backup,) = (tmp_path / "interrupted-history").iterdir()
    assert backup.read_bytes() == before
    history.write_bytes(b'{junk}\n{"step":2}\n')
    with pytest.raises(json.JSONDecodeError):
        restore_log(history, 1)
