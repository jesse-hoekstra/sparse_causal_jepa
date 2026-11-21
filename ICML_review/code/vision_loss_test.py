"""Check the paired training objective, release replay and preregistered decision."""

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.vision_dimer import data as vdata
from diagnostics.vision_dimer import train as release
from diagnostics.vision_dimer.model import LaggedMaskedPredictor, build_model
from ICML_review.code.vision_loss import (
    assess,
    compare_replay,
    continue_training,
    hard_forecast,
    paired_calibration,
)
from scjepa.models.model import ModelOutput, cross_episode_parameters


def test_deployed_forecast_is_exact_with_trainable_gates() -> None:
    """Recover released eval predictions for arbitrary mixed graphs and attached logits."""
    model = build_model(4, phase="full", context=10, code_dim=4)
    p = cast(LaggedMaskedPredictor, model.predictor)
    with torch.no_grad():
        for logits in p.learned_logits().values():
            logits.copy_(torch.linspace(-2, 2, logits.numel()).reshape_as(logits))
    model.eval()
    obs = torch.randn(2, 3, 12, 3, 64, 64)
    act = torch.randn(2, 3, 11, p.action_dim)
    out = cast(ModelOutput, model(obs, act))
    donor = cross_episode_parameters(out.parameters).flatten(0, 1)
    pred = hard_forecast(p, out, act, donor)
    torch.testing.assert_close(pred, out.predictions, rtol=0, atol=0)
    (pred - out.targets).square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
    for logits in p.learned_logits().values():
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0


def test_paired_resume_matches_original_control_and_runs_variant(tmp_path: Path) -> None:
    """Exercise real optimizer/EMA/controller/RNG replay and finite variant outputs on CPU."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _run_smoke(tmp_path)
    finally:
        torch.set_num_threads(threads)


def _run_smoke(root: Path) -> None:
    data = root / "data"
    vdata.prepare(
        data,
        vdata.DataConfig(
            episode_frames=16, train_groups=8, calibration_groups=8, validation_groups=4
        ),
    )
    cfg = release.StageConfig(
        "joint",
        str(data),
        str(root / "joint"),
        device="cpu",
        batch_groups=2,
        code_dim=4,
        steps=1,
        log_every=1,
        checkpoint_every=1,
        eval_every=1000,
    )
    release.train(cfg)
    cfg = replace(
        cfg,
        stage="full",
        parent=str(root / "joint/final.pt"),
        output_dir=str(root / "parent"),
        tau=0.05,
    )
    release.train(cfg)
    source = root / "parent/last.pt"
    cfg = replace(cfg, steps=4)
    for arm in ("original", "control", "paired"):
        settings = replace(cfg, output_dir=str(root / arm))
        release.fork(source, settings)
        if arm == "original":
            release.train(settings, resume=True)
        else:
            continue_training(settings, use_max=arm == "paired")
    native = release.load_checkpoint(root / "original/last.pt")
    control = release.load_checkpoint(root / "control/last.pt")
    for key in ("model", "optimizer", "geco", "sampler", "windows", "cpu_rng", "cuda_rng"):
        torch.testing.assert_close(native[key], control[key], rtol=0, atol=0)
    logs = [json.loads(line) for line in (root / "paired/branches.jsonl").read_text().splitlines()]
    history = [
        json.loads(line) for line in (root / "paired/history.jsonl").read_text().splitlines()
    ]
    assert len(logs) == 3
    for branch, logged in zip(logs, history[-3:], strict=True):
        assert branch["maximum_mse"] == max(branch["stochastic_mse"], branch["hard_mse"])
        assert branch["maximum_mse"] == logged["prediction"]
    probe = paired_calibration(source, cfg, draws=2)
    truth = json.loads((root / "parent/final.json").read_text())
    assert math.isclose(
        probe["paired_batch_means"]["hard_mse"], truth["calibration_prediction_mse"], abs_tol=1e-7
    )
    assert probe["batches_per_draw"] == 4
    assert probe["gate_seeds"] == [0, 1]
    assert [row["hard_mse"] for row in probe["paired_batches_by_draw"][:4]] == [
        row["hard_mse"] for row in probe["paired_batches_by_draw"][4:]
    ]


def test_decision_checks_both_splits_and_preserves_exact_graph() -> None:
    """Reject residual hard infeasibility and expansion that worsens a good graph."""
    row: dict[str, Any] = {
        "mcc": 0.8,
        "prediction_mse": 0.1,
        "calibration_prediction_mse": 0.1,
        "state_r2_min": 0.99,
        "parameter_r2_min": 0.96,
        "state_shd": 0,
        "action_shd": 0,
        "parameter_shd": 0,
    }
    assert assess(0, 0.1, row, row)["passed"]
    assert not assess(0, 0.1, row | {"prediction_mse": 0.11}, row)["passed"]
    assert not assess(1, 0.1, row | {"action_shd": 1}, row)["passed"]


def test_control_checks_every_scientific_log_field() -> None:
    """A drift in any reported scientific field flags the archived control as unresolved."""
    row = {"step": 10, "prediction": 0.01, "log_lambda": 0.01, "seconds_per_step": 1.0}
    assert compare_replay(row | {"seconds_per_step": 2.0}, row)["identical_logged_values"]
    changed = compare_replay(row | {"log_lambda": 0.011}, row)
    assert not changed["numerically_matched"]
    assert not changed["fields"]["log_lambda"]["within_tolerance"]
