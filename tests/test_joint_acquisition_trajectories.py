import copy
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from diagnostics.joint_acquisition.trajectories import TrajectorySettings, run_trajectory
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


def model_fixture() -> SCJEPA:
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
    with torch.no_grad():
        next(model.parameter_encoder.parameters()).fill_(0)
    return model


def fit(model: SCJEPA) -> Tensor:
    value = next(model.parameter_encoder.parameters()).flatten()[0]
    return (value - 1).square()


def codes(model: SCJEPA) -> Tensor:
    return next(model.parameter_encoder.parameters()).flatten()[:6]


def assessment(model: SCJEPA) -> dict[str, Any]:
    value = float(codes(model)[0].detach())
    return {"delayed_benefit": max(0.0, value - 0.25)}


def test_trajectory_keeps_going_after_early_no_benefit_and_resumes_exactly(tmp_path: Path) -> None:
    model = model_fixture()
    before = copy.deepcopy(model.state_dict())
    random_state = torch.get_rng_state()
    settings = TrajectorySettings(updates=6, learning_rate=0.1, milestones=(0, 1, 3, 6))
    options: dict[str, Any] = {
        "case": "encoder",
        "identity": "fixture-groups-and-code",
        "local_checks": {"numerically_valid": True},
        "settings": settings,
    }
    uninterrupted = run_trajectory(
        model, fit, assessment, codes, checkpoint_path=tmp_path / "full.pt", **options
    )
    partial = run_trajectory(
        model,
        fit,
        assessment,
        codes,
        checkpoint_path=tmp_path / "resume.pt",
        stop_after=2,
        **options,
    )
    assert partial["status"] == "interrupted"
    resumed = run_trajectory(
        model,
        fit,
        assessment,
        codes,
        checkpoint_path=tmp_path / "resume.pt",
        resume=True,
        **options,
    )
    assert resumed == uninterrupted
    assert [row["step"] for row in resumed["records"]] == [0, 1, 3, 6]
    assert resumed["records"][1]["assessment"]["delayed_benefit"] == 0
    assert resumed["records"][-1]["assessment"]["delayed_benefit"] > 0
    assert resumed["records"][-1]["weight_displacement"]["predictor"] == 0
    assert (
        resumed["records"][-1]["weight_displacement"]["encoder"]
        > resumed["records"][1]["weight_displacement"]["encoder"]
    )
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    assert torch.equal(random_state, torch.get_rng_state())
    changed = dict(options, identity="other-groups")
    with pytest.raises(ValueError, match="identity"):
        run_trajectory(
            model,
            fit,
            assessment,
            codes,
            checkpoint_path=tmp_path / "resume.pt",
            resume=True,
            **changed,
        )


def test_trajectory_requires_valid_local_numerics_and_bounded_budget(tmp_path: Path) -> None:
    model = model_fixture()
    with pytest.raises(ValueError, match="numerical"):
        run_trajectory(
            model,
            fit,
            assessment,
            codes,
            case="joint",
            checkpoint_path=tmp_path / "x.pt",
            identity="x",
            local_checks={"numerically_valid": False},
        )
    with pytest.raises(ValueError, match="1000"):
        TrajectorySettings(updates=1001)


def test_resume_restores_fitting_random_streams_and_rejects_changed_source(tmp_path: Path) -> None:
    import random

    import numpy as np

    model = model_fixture()
    settings = TrajectorySettings(updates=4, learning_rate=0.03, milestones=(0, 1, 4))

    def stochastic_fit(candidate: SCJEPA) -> Tensor:
        offset = random.random() + float(np.random.uniform()) + torch.rand(())  # noqa: NPY002
        return (next(candidate.parameter_encoder.parameters()).flatten()[0] - offset).square()

    options: dict[str, Any] = {
        "case": "encoder",
        "identity": "rng-fixture",
        "local_checks": {"numerically_valid": True},
        "settings": settings,
    }
    full_path, resumed_path = tmp_path / "full.pt", tmp_path / "resumed.pt"
    full = run_trajectory(
        model, stochastic_fit, assessment, codes, checkpoint_path=full_path, **options
    )
    run_trajectory(
        model,
        stochastic_fit,
        assessment,
        codes,
        checkpoint_path=resumed_path,
        stop_after=2,
        **options,
    )
    resumed = run_trajectory(
        model,
        stochastic_fit,
        assessment,
        codes,
        checkpoint_path=resumed_path,
        resume=True,
        **options,
    )
    assert resumed == full
    completed = torch.load(full_path, weights_only=False)
    continued = torch.load(resumed_path, weights_only=False)
    torch.testing.assert_close(completed["model"], continued["model"], rtol=0, atol=0)
    torch.testing.assert_close(completed["optimizer"], continued["optimizer"], rtol=0, atol=0)
    with torch.no_grad():
        next(model.predictor.parameters()).add_(0.1)
    with pytest.raises(ValueError, match="identity"):
        run_trajectory(
            model,
            stochastic_fit,
            assessment,
            codes,
            checkpoint_path=resumed_path,
            resume=True,
            **options,
        )


def test_predictor_only_trajectory_preserves_codes_and_retention(tmp_path: Path) -> None:
    model = model_fixture()

    def predictor_loss(candidate: SCJEPA) -> Tensor:
        return (next(candidate.predictor.parameters()).flatten()[0] - 3).square()

    report = run_trajectory(
        model,
        predictor_loss,
        assessment,
        codes,
        case="predictor",
        checkpoint_path=tmp_path / "predictor.pt",
        identity="fixed-encoder",
        local_checks={"numerically_valid": True},
        settings=TrajectorySettings(updates=2, milestones=(0, 1, 2)),
    )
    for record in report["records"]:
        assert record["weight_displacement"]["encoder"] == 0
        assert record["code_displacement_rms"] == 0
        assert record["cumulative_code_path_rms"] == 0
