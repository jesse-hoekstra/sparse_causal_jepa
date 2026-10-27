from collections.abc import Generator
from typing import Any

import pytest
import torch
from torch import Tensor

from diagnostics.joint_acquisition.corrections import run_shared_corrections
from diagnostics.joint_acquisition.incentives import analyze_incentives
from scjepa.models.model import SCJEPA
from scjepa.training.factory import ModelConfig, build_model


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def source() -> tuple[SCJEPA, torch.optim.AdamW]:
    model = (
        build_model(
            ModelConfig(
                observation_dim=6,
                encoder_hidden_dims=(8,),
                state_dim=6,
                param_dim=6,
                action_dim=16,
                context_len=10,
                history_len=1,
                parameter_encoder_kind="transition",
                parameter_embed_dim=8,
                spartan_layers=1,
                spartan_embed_dim=4,
                spartan_mlp_hidden=8,
                spartan_mlp_layers=2,
                spartan_dense=True,
                autoregressive=True,
            )
        )
        .double()
        .eval()
    )
    model.online_encoder.requires_grad_(False)
    model.target_encoder.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def value(model: SCJEPA) -> Tensor:
    return (
        next(model.parameter_encoder.parameters()).sum() + next(model.predictor.parameters()).sum()
    )


def loss(model: SCJEPA) -> Tensor:
    return -value(model)


def objective(model: SCJEPA) -> dict[str, Tensor]:
    x = value(model)
    return {
        "prediction": -x,
        "sigreg": x,
        "consistency": x * 0,
        "logit": x * 0,
        "total": x * 0,
        "frozen_state": x.new_tensor(3.0),
    }


def assess(model: SCJEPA) -> dict[str, Any]:
    return {"response_loss": float(loss(model).detach())}


def test_incentives_expose_analytic_cancellation_and_saved_moment_opposition() -> None:
    model, optimizer = source()
    corrections = run_shared_corrections(model, optimizer, loss, assess, objective)
    report = analyze_incentives(model, objective, corrections, fit_loss=loss)
    row = report["cases"]["joint"]
    assert row["components"]["prediction"]["classification"] == "supportive"
    assert row["components"]["sigreg"]["classification"] == "opposing"
    assert row["components"]["total"]["classification"] == "undefined"
    assert row["components"]["frozen_state"]["directional_derivative"] == 0
    assert row["saved_optimizer_alignment"]["classification"] == "opposing"
    assert report["saved_optimizer_response"]["predicted_change"] > 0
    for component in row["components"].values():
        assert component["finite_steps"][-1]["linearization_valid"]


def test_saved_direction_components_have_finite_checks_without_mutating_moments() -> None:
    model, optimizer = source()
    corrections = run_shared_corrections(model, optimizer, loss, assess, objective)
    report = analyze_incentives(model, objective, corrections, fit_loss=loss)
    saved = report["saved_optimizer_components"]
    assert saved["prediction"]["directional_derivative"] > 0
    assert saved["sigreg"]["directional_derivative"] < 0
    assert saved["total"]["directional_derivative"] == 0
    assert all(row["finite_steps"][-1]["linearization_valid"] for row in saved.values())
