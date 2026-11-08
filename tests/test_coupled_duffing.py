from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from diagnostics.coupled_duffing.data import DataConfig, generate, prepare, render
from diagnostics.coupled_duffing.model import LaggedPredictor, build_model
from diagnostics.coupled_duffing.run import (
    TrainConfig,
    forward_loss,
    objective_for,
    phase_model,
    train,
)
from scjepa.models.model import ModelOutput


@pytest.fixture
def small_config() -> DataConfig:
    return DataConfig(
        train_groups=8, calibration_groups=8, validation_groups=8, views=2, context=6, forecast=2
    )


def test_discrete_equation_and_disjoint_systems(small_config: DataConfig) -> None:
    inputs, labels = generate(8, 0, small_config)
    _, other = generate(8, 1, small_config)
    assert set(inputs) == {"observations", "actions", "group_ids"}
    assert not set(int(value) for value in labels["group_ids"]) & set(
        int(value) for value in other["group_ids"]
    )
    states = labels["states"]
    previous, current, future = states[:, :, :-2], states[:, :, 1:-1], states[:, :, 2:]
    h = small_config.dt
    beta, damping = labels["theta"][:, None, None, 0::2], labels["theta"][:, None, None, 1::2]
    left = (
        (future - 2 * current + previous) / h**2
        + damping * (future - current) / h
        + small_config.linear_stiffness * current
        + beta * current.pow(3)
        + small_config.coupling * (current - current.flip(-1))
    )
    right = small_config.force_scale * inputs["actions"][:, :, 1:].double()
    torch.testing.assert_close(left, right.expand_as(left), atol=2e-7, rtol=2e-7)


def test_renderer_contains_both_positions_and_no_parameter_input(small_config: DataConfig) -> None:
    states = torch.tensor([[0.2, -0.4]], requires_grad=True)
    observation = render(states, small_config)
    assert observation.shape == (1, 1, 16, 32)
    weights = torch.arange(observation.numel()).reshape_as(observation)
    (observation * weights).sum().backward()  # pyright: ignore[reportUnknownMemberType]
    assert states.grad is not None
    assert bool((states.grad.abs() > 1e-3).all())
    changed = render(torch.tensor([[0.25, -0.4]]), small_config)
    assert not torch.equal(observation, changed)


def test_shared_coordinate_gate_closes_both_state_lags() -> None:
    model = LaggedPredictor(sparse=True).eval()
    assert model.state_logits is not None
    with torch.no_grad():
        model.state_logits[:, 0] = -3
    states = torch.randn(4, 2, 2, requires_grad=True)
    actions = torch.randn(4, 2, 1, requires_grad=True)
    code = torch.randn(4, 4, requires_grad=True)
    model(states, code, actions).prediction.sum().backward()  # pyright: ignore[reportUnknownMemberType]
    assert states.grad is not None
    assert actions.grad is not None
    assert code.grad is not None
    assert torch.count_nonzero(states.grad[:, :, 0]) == 0
    assert bool((actions.grad.abs().sum((0, 2)) > 0).all())
    assert torch.count_nonzero(code.grad) > 0
    assert model.parameter_logits is None
    assert bool(model.thresholded_graphs()["parameter"].all())


def test_labels_rejected_and_attached_multistep_gradients(small_config: DataConfig) -> None:
    inputs, labels = generate(4, 0, small_config)
    batch = {key: inputs[key] for key in ("observations", "actions")}
    model = build_model(0, small_config.context)
    objective = objective_for("joint", 0.4, torch.device("cpu"))
    with pytest.raises(ValueError, match="only observations and actions"):
        forward_loss(model, {**batch, "theta": labels["theta"]}, objective)
    model.predictor.requires_grad_(False)
    codes: list[Tensor] = []

    def retain_code(_module: nn.Module, _inputs: tuple[Any, ...], code: Tensor) -> None:
        code.retain_grad()
        codes.append(code)

    handle = model.parameter_encoder.register_forward_hook(retain_code)
    output = cast(ModelOutput, model(batch["observations"], batch["actions"]))
    handle.remove()
    output.dynamics[0].prediction.retain_grad()
    output.predictions[:, :, -1].square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
    assert codes[0].grad is not None
    assert float(codes[0].grad.abs().sum()) > 0
    assert output.dynamics[0].prediction.grad is not None
    assert float(output.dynamics[0].prediction.grad.abs().sum()) > 0
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in model.parameter_encoder.parameters()
    )
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_context_has_no_future_leakage(small_config: DataConfig) -> None:
    inputs, _ = generate(3, 0, small_config)
    model = build_model(3, small_config.context).eval()
    with torch.no_grad():
        original = cast(ModelOutput, model(inputs["observations"], inputs["actions"]))
        images = inputs["observations"].clone()
        actions = inputs["actions"].clone()
        images[:, :, small_config.context :] = torch.randn_like(
            images[:, :, small_config.context :]
        )
        actions[:, :, small_config.context - 1 :] *= -1
        changed = cast(ModelOutput, model(images, actions))
    torch.testing.assert_close(original.parameters, changed.parameters, rtol=0, atol=0)


def test_frozen_acquisition_and_sparse_initialization(small_config: DataConfig) -> None:
    model = build_model(1, small_config.context)
    old = {key: value.clone() for key, value in model.online_encoder.state_dict().items()}
    model = phase_model(model, "acquisition", 1, torch.device("cpu"))
    assert all(not p.requires_grad for p in model.online_encoder.parameters())
    for key, value in model.online_encoder.state_dict().items():
        torch.testing.assert_close(value, old[key], rtol=0, atol=0)
    predictor = {key: value.clone() for key, value in model.predictor.state_dict().items()}
    model = phase_model(model, "baseline", 1, torch.device("cpu"))
    assert all(p.requires_grad for p in model.online_encoder.parameters())
    assert isinstance(model.predictor, LaggedPredictor)
    assert model.predictor.parameter_logits is None
    for key, value in predictor.items():
        torch.testing.assert_close(model.predictor.state_dict()[key], value, rtol=0, atol=0)


def test_short_pipeline_and_exact_boundary_resume(tmp_path: Path, small_config: DataConfig) -> None:
    torch.set_num_threads(1)
    data, output = tmp_path / "data", tmp_path / "run"
    prepare(data, small_config)
    # Deleting train references proves that the complete ordinary runner never reads them.
    (data / "train-labels.pt").unlink()
    config = TrainConfig(
        str(data),
        str(output),
        joint_steps=2,
        acquisition_steps=2,
        baseline_steps=2,
        batch_size=4,
        eval_every=2,
        log_every=2,
        device="cpu",
        threads=1,
        wandb_mode="disabled",
    )
    train(config)
    complete = cast(dict[str, Any], torch.load(output / "last.pt", weights_only=False))
    assert complete["step"] == 6
    assert complete["phase"] == "baseline"
    assert complete["tau"] > 0
    assert not list(data.glob("test*"))
    (output / "last.pt").write_bytes((output / "joint.pt").read_bytes())
    train(config, resume=True)
    resumed = cast(dict[str, Any], torch.load(output / "last.pt", weights_only=False))
    for key, value in cast(dict[str, object], complete["model"]).items():
        if isinstance(value, Tensor):
            torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)
    assert resumed["objective"]["geco.num_updates"] == 2
