from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from diagnostics.coupled_duffing_probe.data import ProbeDataConfig, draw, prepare, window
from diagnostics.coupled_duffing_probe.run import ProbeConfig, build_model, forward_loss, train


def test_profiles_share_physics_and_windows_share_forecast_boundary() -> None:
    torch.set_num_threads(1)
    config = ProbeDataConfig(views=3, context=12, forecast=5)
    iid, theta = draw(8, 0, config)
    driven, other_theta = draw(8, 0, replace(config, forcing="driven"))
    torch.testing.assert_close(theta, other_theta, rtol=0, atol=0)
    torch.testing.assert_close(
        iid["observations"][:, :, :2], driven["observations"][:, :, :2], rtol=0, atol=0
    )
    assert bool((driven["actions"].abs() <= 1).all())
    assert not torch.equal(iid["observations"][:, :, 2:], driven["observations"][:, :, 2:])
    short, long = window(iid, 12, 5, 2), window(iid, 12, 12, 5)
    torch.testing.assert_close(short["observations"][:, :, 3:5], long["observations"][:, :, 10:12])
    torch.testing.assert_close(short["observations"][:, :, 5:], long["observations"][:, :, 12:14])
    torch.testing.assert_close(short["actions"][:, :, 3:], long["actions"][:, :, 10:13])
    _, validation_theta = draw(8, 2, config)
    assert not torch.equal(theta, validation_theta)
    with pytest.raises(ValueError, match="reserved test"):
        draw(8, 3, config)


def test_last_forecast_keeps_earlier_predictions_and_donor_codes_attached() -> None:
    torch.set_num_threads(1)
    inputs, theta = draw(4, 0, ProbeDataConfig(views=3, context=5, forecast=4))
    model = build_model(3, 5).eval()
    loss, output = forward_loss(model, inputs)
    assert torch.isfinite(loss)
    torch.testing.assert_close(output.online_states, inputs["observations"], rtol=0, atol=0)
    torch.testing.assert_close(output.targets, inputs["observations"][:, :, 5:], rtol=0, atol=0)
    assert len(output.dynamics) == 4
    first = output.dynamics[0].prediction
    first.retain_grad()
    final_loss = (output.predictions[:, :, -1] - output.targets[:, :, -1]).square().mean()
    final_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert first.grad is not None
    assert bool(first.grad.abs().sum() > 0)
    assert any(
        p.grad is not None and bool(p.grad.abs().sum() > 0)
        for p in model.parameter_encoder.parameters()
    )
    assert not list(model.online_encoder.parameters())
    with pytest.raises(ValueError, match="parameter labels"):
        forward_loss(model, {**inputs, "theta": theta})
    with pytest.raises(ValueError, match="exclude physical"):
        window({**inputs, "theta": theta}, 5, 5, 4)


def test_parameter_labels_are_unused_in_training_and_resume_is_exact(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    data = tmp_path / "data"
    prepare(
        data,
        ProbeDataConfig(
            train_groups=12,
            calibration_groups=8,
            validation_groups=8,
            views=3,
            context=6,
            forecast=4,
        ),
    )
    (data / "train-labels.pt").unlink()
    output = tmp_path / "run"
    config = ProbeConfig(
        str(data),
        str(output),
        context=5,
        horizon=4,
        steps=2,
        batch_size=4,
        eval_every=1,
        log_every=1,
        device="cpu",
        wandb_mode="disabled",
    )
    train(config)
    completed = cast(dict[str, Any], torch.load(output / "last.pt", weights_only=False))
    (output / "last.pt").write_bytes((output / "initial.pt").read_bytes())
    train(config, resume=True)
    resumed = cast(dict[str, Any], torch.load(output / "last.pt", weights_only=False))
    for key, value in completed["model"].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)
    assert completed["step"] == resumed["step"] == 2
    assert not list(data.glob("test*"))
