from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.coupled_duffing.evaluate import metrics
from diagnostics.coupled_duffing.experiment import (
    SCHEMA,
    StageConfig,
    calibrate_endpoint,
    load_data,
    train,
    verify_forecast_extension,
)
from diagnostics.coupled_duffing.model import build_model
from diagnostics.coupled_duffing.run import code_identity, forward_loss, objective_for
from diagnostics.coupled_duffing.visual_data import prepare as prepare_visual
from diagnostics.coupled_duffing_probe.data import ProbeDataConfig
from diagnostics.coupled_duffing_probe.data import prepare as prepare_physical
from diagnostics.shared.io import read_json, save_torch, sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import write_json
from scjepa.training.rng import preserve_rng


def test_long_rollout_keeps_prefix_predictions_and_attached_gradients() -> None:
    torch.set_num_threads(1)
    model = build_model(0, 6, code_dim=4, parameter_encoder="temporal-cnn").double().eval()
    inputs = {
        "observations": torch.randn(4, 3, 22, 1, 16, 32, dtype=torch.float64),
        "actions": torch.randn(4, 3, 21, 1, dtype=torch.float64),
    }
    objective = objective_for(
        "joint",
        0.4,
        torch.device("cpu"),
        state_regularization="projected-aligned",
        state_frames=8,
    ).double()
    captured: list[Tensor] = []

    def capture(_module: torch.nn.Module, _args: tuple[object, ...], code: Tensor) -> None:
        captured.append(code)

    handle = model.parameter_encoder.register_forward_hook(capture)
    with preserve_rng(torch.device("cpu")):
        loss, terms, output = forward_loss(model, inputs, objective)
    handle.remove()
    short = model(inputs["observations"][:, :, :8], inputs["actions"][:, :, :7])
    torch.testing.assert_close(output.predictions[:, :, :2], short.predictions, rtol=0, atol=0)
    torch.testing.assert_close(output.parameters, short.parameters, rtol=0, atol=0)
    assert len(output.dynamics) == 16
    assert not output.targets.requires_grad
    gradients = torch.autograd.grad(
        output.predictions[:, :, -1].square().mean(),
        (output.dynamics[0].prediction, captured[0]),
        retain_graph=True,
    )
    assert all(bool(g.abs().sum() > 0) and bool(torch.isfinite(g).all()) for g in gradients)
    ordinary = objective_for(
        "joint", 0.4, torch.device("cpu"), state_regularization="projected-aligned"
    ).double()
    ordinary.load_state_dict(objective.state_dict())
    with preserve_rng(torch.device("cpu")):
        _, expected = ordinary(output._replace(online_states=output.online_states[:, :, :8]))
    torch.testing.assert_close(
        terms["state_regularizer"], expected["state_regularizer"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        terms["prediction"], (output.predictions - output.targets).square().mean()
    )
    torch.testing.assert_close(loss, terms["prediction"] + 0.4 * terms["state_regularizer"])


def test_state_scoring_uses_common_frames_but_forecast_uses_all_targets() -> None:
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(11)
    labels = {
        "states": torch.randn(64, 2, 10, 2, generator=generator, dtype=torch.float64),
        "theta": torch.randn(64, 4, generator=generator, dtype=torch.float64),
    }
    samples = {
        "states": labels["states"].clone(),
        "codes": labels["theta"][:, None].expand(-1, 2, -1).clone(),
        "targets": labels["states"][:, :, 6:].clone(),
        "predictions": labels["states"][:, :, 6:].clone(),
    }
    original = metrics(samples, samples, labels, labels, 6, state_frames=8)
    changed = {k: v.clone() for k, v in samples.items()}
    changed["states"][:, :, 8:] *= -100
    changed["predictions"][:, :, -1] += 1
    scores = metrics(samples, changed, labels, labels, 6, state_frames=8)
    for key in original:
        if key.startswith(("state_", "coordinate_retention/")):
            assert scores[key] == original[key]
    assert scores["forecast/mse"] == 0.25
    assert scores["forecast/nmse_step_4"] > 0


def test_extended_adaptation_authenticates_prefix_and_resumes_long_sparse_training(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    physical, short, long = (tmp_path / name for name in ("physical", "short", "long"))
    prepare_physical(
        physical,
        ProbeDataConfig(
            train_groups=8,
            calibration_groups=8,
            validation_groups=8,
            views=2,
            context=6,
            forecast=8,
            forcing="driven",
        ),
    )
    prior = prepare_visual(short, physical)
    current = prepare_visual(long, physical, forecast=8)
    # Ordinary training and prefix verification must not open TRAIN labels.
    (short / "train-labels.pt").unlink()
    (long / "train-labels.pt").unlink()
    _, inputs = load_data(long, torch.device("cpu"), forecast_steps=8)
    receipt = verify_forecast_extension(short, prior["identity"], current, inputs)
    assert receipt["all_input_prefixes_exact"]
    inputs["train"]["observations"][0, 0, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="changed the train observations prefix"):
        verify_forecast_extension(short, prior["identity"], current, inputs)

    source = tmp_path / "projection"
    source.mkdir()
    model = build_model(0, 6, code_dim=4, parameter_encoder="temporal-cnn")
    objective = objective_for(
        "joint", 0.4, torch.device("cpu"), state_regularization="projected-aligned"
    )
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "phase": "projection",
        "seed": 0,
        "code_dim": 4,
        "parameter_encoder": "temporal-cnn",
        "state_regularization": "projected-aligned",
        "data_identity": prior["identity"],
        "code_identity": code_identity(),
    }
    record["identity"] = canonical_fingerprint(record)
    save_torch(
        source / "last.pt",
        {
            "identity": record["identity"],
            "model": model.state_dict(),
            "objective": objective.state_dict(),
        },
    )
    write_json(
        source / "completion.json",
        {
            **record,
            "hashes": {"last.pt": sha(source / "last.pt")},
        },
    )
    adaptation = StageConfig(
        str(long),
        str(tmp_path / "adaptation"),
        "adaptation",
        source_dir=str(source),
        source_data_dir=str(short),
        steps=2,
        batch_size=4,
        log_every=1,
        eval_every=1,
        device="cpu",
        wandb_mode="disabled",
        parameter_encoder="temporal-cnn",
        cache_device="cpu",
        state_regularization="projected-aligned",
        learning_rate=0.0001,
        synchronize_target=True,
        cache_frozen_states=True,
        forecast_steps=8,
        state_frames=8,
    )
    train(adaptation)
    finished = read_json(Path(adaptation.output_dir) / "completion.json")
    assert finished["source_forecast_extension"]["all_input_prefixes_exact"]
    assert finished["forecast_steps"] == 8
    calibration = tmp_path / "calibration.json"
    calibrate_endpoint(Path(adaptation.output_dir), calibration)
    full = replace(
        adaptation,
        phase="full",
        output_dir=str(tmp_path / "full"),
        source_dir=adaptation.output_dir,
        source_data_dir=None,
        synchronize_target=False,
        cache_frozen_states=False,
        gate_learning_rate=0.001,
        calibration_file=str(calibration),
    )
    train(full)
    path = Path(full.output_dir)
    before = cast(dict[str, Any], torch.load(path / "last.pt", weights_only=False))
    (path / "last.pt").write_bytes((path / "initial.pt").read_bytes())
    train(full, resume=True)
    after = cast(dict[str, Any], torch.load(path / "last.pt", weights_only=False))
    for category in ("model", "objective", "sampler"):
        torch.testing.assert_close(before[category], after[category], rtol=0, atol=0)
    torch.testing.assert_close(
        before["optimizer"]["state"], after["optimizer"]["state"], rtol=0, atol=0
    )
    assert before["optimizer"]["param_groups"] == after["optimizer"]["param_groups"]
    assert read_json(path / "completion.json")["state_frames"] == 8
    values = cast(dict[str, Tensor], torch.load(path / "validation.pt", weights_only=True))
    assert values["predictions"].shape[2] == 8
    assert values["states"].shape[2] == 14
    assert not list(tmp_path.rglob("test*.pt"))
