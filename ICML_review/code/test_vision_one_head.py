"""Verify one-head archive corroboration, scientific triggers and real CPU training stages."""

import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from diagnostics.vision_dimer import data, train
from scjepa.experiments.storage import write_json

from . import vision_acquisition as stages
from .vision_one_head import (
    admission_checks,
    corroborate_model,
    replay_forecasts,
    scientific_trigger,
    verify_streams,
)


def test_one_head_projection_adaptation_and_sparse_forks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Project and adapt frozen a0 weights, then exercise both sparse continuations."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _cpu_path(tmp_path, monkeypatch)
    finally:
        torch.set_num_threads(threads)


def _cpu_path(tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base, output, cohort = tmp / "archive", tmp / "review", tmp / "data"
    geometry = data.DataConfig(
        train_groups=8, calibration_groups=8, validation_groups=4, episode_frames=16
    )
    data.prepare(cohort, geometry)
    cfg = train.StageConfig(
        "joint",
        str(cohort),
        str(base / "joint"),
        device="cpu",
        seed=1,
        steps=2,
        batch_groups=2,
        eval_every=2,
        log_every=1,
        checkpoint_every=1,
    )
    train.train(cfg)
    joint = base / "joint/final.pt"
    acquire = replace(
        cfg,
        stage="acquire",
        parent=str(joint),
        acquisition_seed=1001,
        output_dir=str(base / "acquire/a0"),
    )
    acquisition = train.train(acquire)
    source = base / "acquire/a0/final.pt"
    original, head, saved = (
        train.load_checkpoint(p) for p in (joint, source, source.parent / "last.pt")
    )
    counts = corroborate_model(original, head, saved)
    assert counts["online_encoder"] > 0
    assert counts["target_encoder"] > 0
    assert verify_streams(saved, acquire, geometry, torch.device("cpu"))[
        "sampler_stream_reproduced"
    ]
    with pytest.raises(ValueError, match="time-window stream"):
        verify_streams(
            saved | {"windows": torch.Generator().manual_seed(42).get_state()},
            acquire,
            geometry,
            torch.device("cpu"),
        )
    corrupted = dict(head["model"])
    key = next(
        k
        for k in corrupted
        if k.startswith("predictor.") and isinstance(corrupted[k], torch.Tensor)
    )
    corrupted[key] = corrupted[key] + 1
    with pytest.raises(ValueError, match="tensor differs"):
        corroborate_model(original, head, saved | {"model": corrupted})
    inputs = {
        s: data.RenderedInputs(cohort, s, torch.device("cpu"))
        for s in ("calibration", "validation")
    }
    reproduced = replay_forecasts(head, inputs, torch.device("cpu"))
    assert abs(reproduced["calibration"] - acquisition["calibration_prediction_mse"]) < 1e-7
    assert abs(reproduced["validation"] - acquisition["prediction_mse"]) < 1e-7

    # The archived config supplies every optimizer setting. The review functions invoke the
    # original CLI, with the CPU fixture's much shorter budgets in both sparse conditions.
    adapt = replace(cfg, stage="adapt", parent=str(source), code_dim=4)
    write_json(base / "adapt/config.json", {"config": asdict(adapt)})
    write_json(base / "adapt/final.json", acquisition)
    for arm in ("full", "baseline"):
        for label, steps in (("300k", 2), ("500k", 4)):
            sparse = replace(adapt, stage=arm, tau=0.05, steps=steps)  # type: ignore[arg-type]
            write_json(base / f"sparse-{label}/{arm}/config.json", {"config": asdict(sparse)})
    monkeypatch.setattr(stages, "_idle_gpu", lambda: None)
    root = Path.cwd()
    monkeypatch.setenv("PYTHONPATH", f"{root}:{root / 'src'}")
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    initial = stages.project_adapt(
        1, rev, output, "smoke", source, base=base, data_dir=cohort, group="smoke"
    )
    assert stages.sha(output / "projected/final.pt") == initial["checkpoint_sha256"]["projected"]
    assert not stages.read(output / "projected/final.json")["labels_used"]
    for arm in ("full", "baseline"):
        hashes = stages.sparse_arm(
            1, rev, output, "smoke", arm, base=base, data_dir=cohort, group="smoke"
        )
        assert hashes[f"500k/{arm}"] == stages.sha(output / f"sparse-500k/{arm}/final.pt")
        assert stages.read(output / f"sparse-500k/{arm}/fork.json")["source_sha256"] == stages.sha(
            output / f"sparse-300k/{arm}/last.pt"
        )
        endpoint = train.load_checkpoint(output / f"sparse-500k/{arm}/last.pt")
        assert endpoint["step"] == 4
        assert endpoint["config"]["tau"] == 0.05
        assert hashes == stages.sparse_arm(
            1, rev, output, "smoke", arm, base=base, data_dir=cohort, group="smoke"
        )


def test_scientific_trigger_checks_validation_decisions_and_closed_queues(tmp_path: Path) -> None:
    """Reject unfinished queues and operational success without a failed direct scientific gate."""
    queue = tmp_path / "loss-queues"
    queue.mkdir()
    for suite in ("synthetic-loss", "vision-loss", "vision-direct"):
        (queue / f"{suite}.exit_code").write_text("0\n")
    measured = {"prediction_mse": 1.0, "calibration_prediction_mse": 1.0, "parameter_r2_min": 0.90}
    direct = tmp_path / "vision-direct-s0"
    direct.mkdir()
    (direct / "NOTE.md").write_text("revision-1")
    decision = {
        "seed": 0,
        "commit": "revision-1",
        "adaptation": measured,
        "reference_adaptation": measured,
        "adaptation_checks": stages.adaptation_checks(measured, measured),
        "passed": False,
        "reason": "adaptation",
    }
    (direct / "metrics.json").write_text(json.dumps(decision))
    assert scientific_trigger(tmp_path)["failed_direct"][0]["seed"] == 0
    write_json(direct / "metrics.json", decision | {"passed": True})
    with pytest.raises(ValueError, match="inconsistent direct"):
        scientific_trigger(tmp_path)
    (queue / "vision-direct.exit_code").write_text("1\n")
    with pytest.raises(ValueError, match="did not finish"):
        scientific_trigger(tmp_path)


def test_fixed_head_admission_cannot_select_by_validation() -> None:
    """A worse CAL forecast ends this candidate even if its validation forecast improves."""
    candidate = {
        "verified": True,
        "first_window": {
            "a0": {"measured": {"calibration": 1.1, "validation": 0.1}},
            "joint": {"measured": {"calibration": 1.0, "validation": 1.0}},
        },
    }
    assert not admission_checks(candidate)["a0_calibration_at_most_joint"]
    with pytest.raises(RuntimeError, match="replay differed"):
        admission_checks(candidate | {"verified": False})
