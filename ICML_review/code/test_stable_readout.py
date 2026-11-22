"""Check generalized-eigenvalue fitting, intrinsic chart alignment and native forecasts."""

import math
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor, nn

from diagnostics.masked_mlp_learned import acquisition, run
from diagnostics.shared.data import StateData
from diagnostics.vision_dimer import data, train
from diagnostics.vision_dimer.model import LaggedCodeEncoder, build_model
from scjepa.models.model import ModelOutput

from .stable_readout import SingularMeanMap, StableReadoutError, fit_readout


def _example(fractions: list[float]) -> Tensor:
    """Build centered features with identity total covariance and chosen diagonal within shares."""
    g, r, p = 43, 7, len(fractions)
    rng = torch.Generator().manual_seed(24)
    mean = torch.randn(g, p, generator=rng, dtype=torch.float64)
    mean = cast(tuple[Tensor, Tensor], torch.linalg.qr(mean - mean.mean(0)))[0] * g**0.5  # pyright: ignore[reportUnknownMemberType]
    resid = torch.randn(g, r, p, generator=rng, dtype=torch.float64)
    left = (resid - resid.mean(1, keepdim=True)).flatten(0, 1)
    q = cast(tuple[Tensor, Tensor], torch.linalg.qr(left))[0].reshape(g, r, p) * (g * r) ** 0.5  # pyright: ignore[reportUnknownMemberType]
    lam = torch.tensor(fractions, dtype=torch.float64)
    return mean[:, None, :] * (1 - lam).sqrt() + q * lam.sqrt()


def _layer(w: Tensor, b: Tensor) -> nn.Linear:
    layer = nn.Linear(w.shape[1], w.shape[0], dtype=w.dtype)
    with torch.no_grad():
        layer.weight.copy_(w)
        layer.bias.copy_(b)
    return layer


def test_tied_interior_and_affine_feature_and_teacher_equivariance() -> None:
    """Test the analytic optimum and intrinsic alignment even for a repeated selected eigenvalue."""
    x = _example([0.2, 0.2, 0.7, 0.9])
    w = x.new_tensor([[2.0, -1.0, 0.7, 0.2], [0.3, 1.5, -0.4, 0.6]])
    b = x.new_tensor([1.0, -2.0])
    fits, info = fit_readout(x, _layer(w, b))
    torch.testing.assert_close(fits["own"][0], w, atol=0, rtol=0)
    trial = x @ fits["sibling"][0].T + fits["sibling"][1]
    torch.testing.assert_close(trial, x[:, :, :2] @ w[:, :2].T + b, atol=5e-13, rtol=5e-13)
    assert math.isclose(info["optimum_trace"], 0.4)
    assert math.isclose(info["boundary_gap"], 0.5)
    assert math.isclose(info["optimum_worst"], 0.2)
    torch.testing.assert_close(
        x.new_tensor(info["aligned_train_spectrum"]), x.new_tensor([0.2, 0.2])
    )

    change = x.new_tensor(
        [[0.5, 1.0, 0.0, 1.0], [1.0, 2.0, 0.2, 0.0], [2.0, 0.6, -1.0, 1.0], [-1.0, 0.0, 1.0, 2.0]]
    )
    offset = x.new_tensor([2.0, -1.0, 0.5, 3.0])
    layer2 = cast(Tensor, torch.linalg.solve(change, w.T)).T  # pyright: ignore[reportUnknownMemberType]
    x2 = x @ change + offset
    new, _ = fit_readout(x2.flip(0, 1), _layer(layer2, b - offset @ layer2.T))
    torch.testing.assert_close(
        x2 @ new["sibling"][0].T + new["sibling"][1], trial, atol=1e-12, rtol=1e-12
    )

    chart = x.new_tensor([[2.0, 1.0], [-0.5, 0.7]])
    shift = x.new_tensor([0.7, 1.3])
    new, _ = fit_readout(x, _layer(chart.T @ w, b @ chart + shift))
    torch.testing.assert_close(x @ new["sibling"][0].T + new["sibling"][1], trial @ chart + shift)


def test_support_gap_and_singular_mean_map_have_distinct_receipts() -> None:
    """Distinguish insufficient support, a tied boundary and a singular alignment."""
    x = _example([0.2, 0.4, 0.4, 0.8])
    b = x.new_zeros(2)
    w = x.new_tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    with pytest.raises(StableReadoutError, match="gap") as gap:
        fit_readout(x, _layer(w, b))
    assert gap.value.reason == "unresolved_boundary"
    assert gap.value.diagnostics["boundary_gap"] < gap.value.diagnostics["gap_cutoff"]

    x = _example([0.1, 0.2, 0.7, 0.8])
    good = x[:, :, :2]
    dependent = torch.cat(
        (good, good[:, :, :1] - 2 * good[:, :, 1:], torch.ones_like(good[:, :, :1])), dim=-1
    )
    both, receipt = fit_readout(dependent, _layer(w, b))
    assert receipt["entire_resolved_support"]
    assert receipt["design_rank"] == 2
    assert receipt["boundary_gap"] is None
    torch.testing.assert_close(
        dependent @ both["sibling"][0].T + both["sibling"][1], dependent @ w.T + b
    )
    with pytest.raises(StableReadoutError, match="rank below") as small:
        fit_readout(dependent[:, :, :1].expand_as(dependent), _layer(w, b))
    assert small.value.reason == "insufficient_feature_rank"
    with pytest.raises(SingularMeanMap) as singular:
        fit_readout(x, _layer(x.new_tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]), b))
    assert singular.value.reason == "singular_alignment"
    with pytest.raises(SingularMeanMap) as missing_teacher:
        fit_readout(x, _layer(w[:1].expand_as(w), b))
    assert missing_teacher.value.reason == "teacher_mean_rank"


def test_native_frozen_rgb_and_synthetic_forecasts(tmp_path: Path) -> None:
    """Fit the final exposed features and forecast with each native model on CPU."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _native_probe(tmp_path)
    finally:
        torch.set_num_threads(previous)


@torch.no_grad()
def _native_probe(path: Path) -> None:
    g = 32
    cpu = torch.device("cpu")
    rng = torch.Generator().manual_seed(72)
    tensors = {
        "observations": torch.randn(g, 10, 12, 6, generator=rng),
        "actions": torch.randn(g, 10, 11, 24, generator=rng),
        "oracle_parameters": torch.zeros(g, 6),
        "episode_ids": torch.arange(g),
    }
    source = StateData("train", tensors, None, {})
    syn = run.build_control("full", 0, predictor_activation="gelu").eval()
    features = acquisition.train_contexts(syn, source, cpu, codes=False)
    data.prepare(
        path,
        data.DataConfig(
            train_groups=g, calibration_groups=4, validation_groups=4, episode_frames=16
        ),
    )
    frames = data.RenderedInputs(path, "train", cpu)
    rgb = build_model(0, phase="full", code_dim=4, context=10).eval()
    enc = cast(LaggedCodeEncoder, rgb.parameter_encoder)
    rows: list[Tensor] = []
    for batch in train.rendered_batches(frames, size=4):
        ob, ac = batch["observations"], batch["actions"]
        z = rgb.online_encoder.encode_sequence(ob[:, :, : rgb.context_len].flatten(0, 1))
        rows.append(
            enc.features(z, ac[:, :, : rgb.context_len - 1].flatten(0, 1)).reshape(
                *ob.shape[:2], -1
            )
        )
    for model, x, batch in (
        (syn, features, source.batch(torch.arange(2), cpu)),
        (rgb, torch.cat(rows), frames.batch(torch.arange(2))),
    ):
        layer = cast(
            nn.Linear, cast(nn.Sequential, model.parameter_encoder.get_submodule("head"))[-1]
        )
        fits, info = fit_readout(x, layer)
        before = cast(ModelOutput, model(batch["observations"], batch["actions"]))
        torch.testing.assert_close(
            before.parameters, x[:2].float() @ layer.weight.T + layer.bias, rtol=1e-4, atol=2e-6
        )
        assert info["design_rank"] >= layer.out_features
        assert info["spectrum_discrepancy"] < 1.0e-6
        layer.weight.copy_(fits["sibling"][0])
        layer.bias.copy_(fits["sibling"][1])
        after = cast(ModelOutput, model(batch["observations"], batch["actions"]))
        assert after.predictions.isfinite().all()
        assert not torch.equal(after.parameters, before.parameters)
        torch.testing.assert_close(after.online_states, before.online_states, rtol=0, atol=0)
