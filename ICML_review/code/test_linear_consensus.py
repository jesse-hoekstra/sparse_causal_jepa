"""Check the single-episode fit, chart preservation and native model interfaces."""

from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from diagnostics.masked_mlp_learned import acquisition, run
from diagnostics.shared.data import StateData
from diagnostics.vision_dimer import data
from diagnostics.vision_dimer.model import build_model
from scjepa.models.model import ModelOutput

from .linear_consensus import fit_readout, rgb_features, uncertainty


def test_independent_targets_and_unresolved_features() -> None:
    """Match an explicit fit and leave unseen feature directions unchanged."""
    g = torch.Generator().manual_seed(309)
    shared = torch.randn(64, 1, 2, generator=g, dtype=torch.double)
    noise = torch.randn(64, 10, 2, generator=g, dtype=torch.double)
    x = torch.cat((shared.expand_as(noise), noise, torch.ones_like(noise[:, :, :1])), dim=-1)
    layer = nn.Linear(5, 2).double()
    with torch.no_grad():
        layer.weight.copy_(torch.cat((torch.eye(2), torch.eye(2), torch.ones(2, 1)), dim=1))
    own = layer(x).detach()
    ref = torch.stack([own[:, list(set(range(10)) - {j})].mean(1) for j in range(10)], dim=1)
    fits, info = fit_readout(x, layer)
    w, b = fits["sibling"]
    trial = x @ w.T + b
    assert info["design_rank"] == 4
    torch.testing.assert_close(layer.weight, fits["own"][0], rtol=0, atol=0)
    torch.testing.assert_close(w[:, -1], layer.weight[:, -1], rtol=0, atol=1e-10)
    centered = x - x.mean((0, 1))
    normal = centered.flatten(0, 1).T @ (trial - ref).flatten(0, 1)
    torch.testing.assert_close(normal, torch.zeros_like(normal), rtol=0, atol=2e-9)
    spread = uncertainty(own, trial, torch.zeros(64), torch.zeros(64))
    assert spread["trial_rank"] == 2
    assert spread["trial_worst"] < spread["control_worst"] * 0.1
    assert spread["worst_change_95"][1] < 0
    with pytest.raises(ValueError, match="donor"):
        fit_readout(x[:, :1], layer)


def test_both_native_feature_readouts(tmp_path: Path) -> None:
    """Fit both native final layers and use them to forecast on CPU."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cpu = torch.device("cpu")
        tensors = {
            "observations": torch.randn(8, 10, 12, 6),
            "actions": torch.randn(8, 10, 11, 24),
            "oracle_parameters": torch.zeros(8, 6),
            "episode_ids": torch.arange(8),
        }
        ds = StateData("train", tensors, None, {})
        syn = run.build_control("full", 0, predictor_activation="gelu").eval()
        features = acquisition.train_contexts(syn, ds, cpu, codes=False)
        cohort = tmp_path / "frames"
        data.prepare(
            cohort,
            data.DataConfig(
                train_groups=8, calibration_groups=8, validation_groups=4, episode_frames=16
            ),
        )
        frames = data.RenderedInputs(cohort, "train", cpu)
        rgb = build_model(0, phase="full", code_dim=4, context=10).eval()
        image_features = rgb_features(rgb, frames)
        for model, x, batch in (
            (syn, features, ds.batch(torch.arange(2), cpu)),
            (rgb, image_features, frames.batch(torch.arange(2))),
        ):
            layer = cast(
                nn.Linear, cast(nn.Sequential, model.parameter_encoder.get_submodule("head"))[-1]
            )
            fits, _ = fit_readout(x, layer)
            with torch.no_grad():
                before = cast(ModelOutput, model(batch["observations"], batch["actions"]))
                torch.testing.assert_close(
                    x[:2].float() @ layer.weight.T + layer.bias,
                    before.parameters,
                    rtol=1e-4,
                    atol=1e-6,
                )
                layer.weight.copy_(fits["sibling"][0])
                layer.bias.copy_(fits["sibling"][1])
                after = cast(ModelOutput, model(batch["observations"], batch["actions"]))
            assert torch.isfinite(after.predictions).all()
            assert not torch.equal(before.parameters, after.parameters)
    finally:
        torch.set_num_threads(old)
