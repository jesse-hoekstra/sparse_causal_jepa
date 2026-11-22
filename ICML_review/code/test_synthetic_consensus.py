"""Frozen-teacher matching, norm cap, system bootstrap and native toy continuations."""

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from diagnostics.masked_mlp_learned import run as runner
from diagnostics.masked_mlp_learned.continuation import continue_training
from diagnostics.shared import data, source_data
from diagnostics.shared.io import read_json, sha

from .consensus_gradient import add_capped_gradient, anchor, context_intervals, whitening
from .synthetic_consensus import endpoint
from .synthetic_consensus_runtime import train_child


def test_whitened_disjoint_teacher_and_capped_encoder_only() -> None:
    """Ground the loss in whitened coordinates and protect the actual native parameter gradient."""
    g = torch.Generator().manual_seed(13)
    x = torch.randn(11, 4, 3, generator=g, dtype=torch.float64)
    w = whitening(x)
    flatten = x.flatten(0, 1)
    y = (flatten - flatten.mean(0)) @ w
    torch.testing.assert_close(y.T @ y / len(y), torch.eye(3, dtype=y.dtype))
    student = nn.Parameter(x.clone())
    teacher = x.clone().requires_grad_()
    auxiliary = anchor(student, teacher, w, "other")
    mean_other = torch.stack(
        [x[:, [other for other in range(4) if other != index]].mean(1) for index in range(4)], 1
    )
    torch.testing.assert_close(auxiliary, ((x - mean_other) @ w).square().mean())
    assert anchor(student, teacher, w, "same").item() == 0
    predictor = nn.Parameter(torch.ones((), dtype=x.dtype))
    loss = (student * predictor).square().mean() * 0.01
    raw, prediction_grad = torch.autograd.grad(loss, (student, predictor), retain_graph=True)
    vals = add_capped_gradient(loss, auxiliary, (student,))
    assert 0 < vals["consensus_gradient_scale"] < 1
    assert predictor.grad is None
    assert teacher.grad is None
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert isinstance(student.grad, Tensor)
    assert (raw * student.grad).sum() >= -1e-12
    torch.testing.assert_close(
        (student.grad - raw).square().sum().sqrt(), raw.square().sum().sqrt()
    )
    torch.testing.assert_close(predictor.grad, prediction_grad)
    with pytest.raises(ValueError, match="support"):
        whitening(torch.cat((x, x[..., 0:1]), dim=-1))


def test_system_bootstrap_keeps_both_arms_paired() -> None:
    """Identical policies yield zero paired differences, even with heterogeneous groups."""
    g = torch.Generator().manual_seed(31)
    codes = torch.randn(20, 4, 3, generator=g)
    errors = torch.rand(20, generator=g)
    vals = context_intervals(
        dict.fromkeys(("native", "same", "other"), codes),
        dict.fromkeys(("native", "same", "other"), errors),
        42,
        draws=20,
    )
    assert vals["other_minus_native"] == {
        f"{q}_{key}": {"estimate": v, "ci95": [v, v]}
        for q in ("worst", "mse")
        for key, v in (("difference", 0.0), ("ratio", 1.0))
    }


def test_frozen_teacher_native_cpu_forks(tmp_path: Path) -> None:
    """Take actual optimizer steps from a toy completed model and evaluate both matched forks."""
    torch.set_num_threads(1)
    source, cohort = tmp_path / "source", tmp_path / "data"
    source_data.prepare(
        source_data.PrepareConfig(
            str(source),
            train_groups=12,
            calibration_groups=12,
            validation_groups=10,
            test_groups=2,
            profile="smoke",
        )
    )
    data.prepare(data.PrepareConfig(str(cohort), str(source), profile="smoke"))
    cfg = runner.TrainConfig(
        str(tmp_path / "dense"),
        str(cohort),
        "dense",
        profile="smoke",
        steps=2,
        batch_size=2,
        code_dim=6,
        initializer="gain",
        predictor_activation="gelu",
        log_every=1,
        evaluate_every=3,
        checkpoint_every=1,
    )
    runner.train(cfg)
    bound = tmp_path / "tau"
    runner.calibrate(runner.CalibrateConfig(str(bound), cfg.output_dir, profile="smoke"))
    parent = tmp_path / "original"
    sparse = replace(
        cfg,
        output_dir=str(parent),
        condition="full",
        init_dir=cfg.output_dir,
        calibration_dir=str(bound),
        code_dim=None,
        initializer=None,
    )
    source_receipt = runner.train(sparse)
    raw = tmp_path / "native"
    continue_training(parent, raw, additional_steps=3, device="cpu")
    forecasts: list[float] = []
    for mode in ("same", "other"):
        path = tmp_path / mode
        result = train_child(parent, path, mode, steps=3, device="cpu", resume=False)
        assert result["steps"] == 5
        assert result["code_identity"] == source_receipt["code_identity"]
        assert (
            read_json(path / "config.json")["scientific_config"]["frozen_teacher"]["target"] == mode
        )
        records, _, _ = endpoint(path, 0)
        forecasts.append(records["validation"]["prediction"])
        assert math.isfinite(forecasts[-1])
        assert read_json(path / "continuation.json")["parent_checkpoint_sha256"] == sha(
            parent / "last.pt"
        )
    assert forecasts[0] != forecasts[1]
