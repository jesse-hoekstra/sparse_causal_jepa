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


def test_long_rgb_episodes_score_native_first_window_after_readout(tmp_path: Path) -> None:
    """Corroborate both physical splits and the released checkpoint evaluator after each refit."""
    from diagnostics.vision_dimer import evaluate, train

    from .consensus_queries import sources
    from .linear_consensus import rgb_eval

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cpu = torch.device("cpu")
        cfg = data.DataConfig(
            train_groups=8,
            calibration_groups=12,
            validation_groups=8,
            test_groups=2,
            episode_frames=30,
        )
        directory = tmp_path / "frames"
        data.prepare(directory, cfg)
        inputs = {
            s: data.RenderedInputs(directory, s, cpu)
            for s in ("train", "calibration", "validation")
        }
        model = build_model(0, phase="full", context=cfg.context, code_dim=4).eval()
        layer = cast(
            nn.Linear, cast(nn.Sequential, model.parameter_encoder.get_submodule("head"))[-1]
        )
        fits, _ = fit_readout(rgb_features(model, inputs["train"]), layer)
        physical = {s: data.load_labels(directory, s) for s in ("calibration", "validation")}
        for labels in physical.values():
            assert labels["states"].shape[2] == cfg.episode_frames
        assert cfg.frames < cfg.episode_frames
        original_codes = None
        for mode in ("own", "sibling"):
            with torch.no_grad():
                weight, bias = fits[mode]
                layer.weight.copy_(weight)
                layer.bias.copy_(bias)
                metrics, samples = rgb_eval(model, inputs, data_dir=directory)
            expected_labels = {
                s: {"theta": z["theta"], "states": z["states"][:, :, : cfg.frames]}
                for s, z in physical.items()
            }
            for split in ("calibration", "validation"):
                assert (
                    samples[split]["states"].shape[:3] == expected_labels[split]["states"].shape[:3]
                )
                original = evaluate.evaluate(
                    model,
                    samples["calibration"],
                    expected_labels["calibration"],
                    samples[split],
                    expected_labels[split],
                )
                row = metrics[split]
                assert row["mse"] == original["prediction_mse"]
                assert row["nmse"] == original["nmse"]
                assert row["mcc"] == original["mcc"]
                assert row["r2"] == original["parameter_r2_min"]
                assert row["shd"] == sum(
                    original[k + "_shd"] for k in ("state", "action", "parameter")
                )
                assert row["spearman_assignment"] == original["spearman_assignment"]
            if mode == "own":
                original_codes = samples["validation"]["codes"]
            else:
                assert original_codes is not None
                assert not torch.equal(original_codes, samples["validation"]["codes"])
            source = tmp_path / f"{mode}.pt"
            torch.save(
                {
                    "meta": {
                        "seed": 0,
                        "context": cfg.context,
                        "code_dim": 4,
                        "shape": list(cfg.renderer.shape),
                        "stage": "full",
                    },
                    "config": {"stage": "full"},
                    "model": model.state_dict(),
                },
                source,
            )
            released = train.evaluate_checkpoint(
                source, directory, tmp_path / f"{mode}.json", device="cpu"
            )
            assert metrics["calibration"]["mse"] == released["calibration_prediction_mse"]
            assert metrics["validation"]["mse"] == released["prediction_mse"]
            assert metrics["validation"]["mcc"] == released["mcc"]
            assert metrics["validation"]["r2"] == released["parameter_r2_min"]
        queries = sources(
            "linear-consensus-vision-window", tmp_path, {"checkpoint_sha256": "digest"}
        )
        assert queries == {"sibling": ("full", tmp_path / "final.pt", "digest")}
    finally:
        torch.set_num_threads(old_threads)
