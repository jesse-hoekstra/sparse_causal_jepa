"""Compare the analytic optimum, its invariances and the real CPU RGB training path."""

import inspect
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.vision_dimer import data, train

from . import rgb_predictive as rgb
from . import vision_acquisition, vision_one_head
from .predictive_projection import reduced_rank_map


def test_rank_optimum_and_invertible_input_units() -> None:
    """The induced predictions are invariant to a nonsingular linear code reparameterization."""
    g = torch.Generator().manual_seed(19)
    z = torch.randn(128, 6, dtype=torch.float64, generator=g)
    z = z - z.mean(0)
    w = torch.randn(11, 6, dtype=torch.float64, generator=g)
    t = z.T @ z / len(z)
    s = torch.diag(torch.arange(6, dtype=torch.float64))
    p, r = reduced_rank_map(t, s, w, 3)
    y = z @ w.T
    approx = z @ p.T @ r.T @ w.T
    spectrum = cast(Tensor, torch.linalg.svdvals(y))  # pyright: ignore[reportUnknownMemberType]
    assert torch.allclose((y - approx).square().sum(), spectrum[3:].square().sum(), rtol=1e-9)
    transform = torch.randn(6, 6, dtype=torch.float64, generator=g) + torch.eye(6) * 5
    transformed_w = cast(Tensor, torch.linalg.solve(transform.T, w.T)).T  # pyright: ignore[reportUnknownMemberType]
    p2, r2 = reduced_rank_map(
        transform @ t @ transform.T, transform @ s @ transform.T, transformed_w, 3
    )
    assert torch.allclose(z @ transform.T @ p2.T @ r2.T @ transformed_w.T, approx, atol=1e-9)
    assert torch.allclose(p @ p.T, torch.eye(3, dtype=torch.float64), atol=1e-12)


def test_low_rank_and_ambiguous_cutoffs_are_explicit() -> None:
    """Reject undefined predictive subspaces rather than silently selecting a coordinate."""
    t = torch.eye(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="tie"):
        reduced_rank_map(t, t, t, 2)
    with pytest.raises(ValueError, match="rank below"):
        reduced_rank_map(t, t, torch.diag(torch.tensor([1.0, 0.0, 0.0])), 2)
    with pytest.raises(ValueError, match="indefinite"):
        reduced_rank_map(-t, t, t, 2)


def test_defaults_and_wrapper_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the requested CLI variant and preserve the released defaults."""
    standard = inspect.signature(vision_one_head.run_phase).parameters["trial"].default
    assert standard.folder == "vision-one-head"
    assert standard.protocol == vision_one_head.PROTOCOL
    assert standard.project_command is None
    assert (
        inspect.signature(vision_acquisition.project_adapt).parameters["project_command"].default
        is None
    )
    seen: dict[str, Any] = {}

    def capture(commit: str, seed: int, phase: str, *, trial: vision_one_head.OneHeadTrial) -> None:
        seen.update(commit=commit, seed=seed, phase=phase, trial=trial)

    monkeypatch.setattr(rgb, "run_phase", capture)
    monkeypatch.setattr("sys.argv", ["rgb_predictive", "fixture-commit", "2", "prepare"])
    rgb.main()
    trial = cast(vision_one_head.OneHeadTrial, seen["trial"])
    assert trial.folder == "rgb-predictive"
    assert trial.module == rgb.MODULE
    assert trial.project_command == (
        ".venv/bin/python",
        "-m",
        rgb.MODULE,
        "project",
        "--commit",
        "fixture-commit",
    )


def test_cpu_native_projection_and_adaptation(tmp_path: Path) -> None:
    """Exercise the fitted map, native adaptation and both native sparse forks."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _smoke(tmp_path)
    finally:
        torch.set_num_threads(threads)


def _smoke(root: Path) -> None:
    cohort = root / "data"
    data.prepare(
        cohort,
        data.DataConfig(
            train_groups=8, calibration_groups=8, validation_groups=4, episode_frames=16
        ),
    )
    cfg = train.StageConfig(
        "joint",
        str(cohort),
        str(root / "joint"),
        device="cpu",
        seed=1,
        steps=2,
        batch_groups=2,
        eval_every=2,
        log_every=1,
        checkpoint_every=1,
    )
    train.train(cfg)
    source = root / "a0"
    acquisition = train.train(
        replace(
            cfg,
            stage="acquire",
            parent=str(root / "joint/final.pt"),
            acquisition_seed=1001,
            output_dir=str(source),
        )
    )
    original = root / "native"
    train.project(source, cohort, original, device="cpu")
    artifact = rgb.project(source, cohort, root / "predictive", device="cpu")
    assert not artifact["labels_used"]
    assert (
        artifact["training"]["predictive"]["preactivation_relative_squared_error"]
        <= artifact["training"]["native"]["preactivation_relative_squared_error"] + 1e-10
    )
    native = train.model_from_checkpoint(train.load_checkpoint(original / "final.pt")).eval()
    native_errors = rgb.group_errors(
        native, data.RenderedInputs(cohort, "calibration", torch.device("cpu"))
    )
    means = artifact["immediate"]["calibration"]["means"]
    assert math.isclose(means["native"], float(native_errors.mean()), rel_tol=1e-5, abs_tol=1e-8)
    assert math.isclose(means["upstream"], acquisition["calibration_prediction_mse"], rel_tol=1e-5)
    assert artifact["immediate"]["validation"]["systems"] == 4
    adapted = root / "adapt"
    dense = replace(
        cfg,
        stage="adapt",
        code_dim=4,
        parent=str(root / "predictive/final.pt"),
        output_dir=str(adapted),
    )
    assert train.train(dense)["step"] == 2
    for arm in ("baseline", "full"):
        first = replace(
            dense, stage=arm, tau=0.05, parent=str(adapted / "final.pt"), output_dir=str(root / arm)
        )  # pyright: ignore[reportArgumentType]
        assert train.train(first)["step"] == 2
        end = replace(first, output_dir=str(root / (arm + "-end")), steps=4)
        train.fork(root / arm / "last.pt", end)
        assert train.train(end, resume=True)["step"] == 4
