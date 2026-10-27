import copy
from collections.abc import Generator
from typing import Any

import pytest
import torch
from torch import Tensor

from diagnostics.joint_acquisition.corrections import CorrectionSettings, run_shared_corrections
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


def tiny_source() -> tuple[SCJEPA, torch.optim.AdamW]:
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
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.0)
    for p in parameters:
        p.grad = torch.ones_like(p)
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def coordinates(model: SCJEPA) -> tuple[Tensor, Tensor]:
    return next(model.parameter_encoder.parameters()).flatten()[0], next(
        model.predictor.parameters()
    ).flatten()[0]


def fit_loss(model: SCJEPA) -> Tensor:
    encoder, predictor = coordinates(model)
    return (encoder - 3).square() + (predictor - 3).square()


def components(model: SCJEPA) -> dict[str, Tensor]:
    loss = fit_loss(model)
    return {
        "prediction": loss,
        "sigreg": loss * 0,
        "consistency": loss * 0,
        "logit": loss * 0,
        "total": loss,
    }


def assess(model: SCJEPA) -> dict[str, Any]:
    encoder, predictor = coordinates(model)
    return {
        "encoder": float(encoder.detach()),
        "predictor": float(predictor.detach()),
        "heldout_loss": float(fit_loss(model).detach()),
    }


def test_shared_correction_uses_same_per_block_bounds_and_preserves_source() -> None:
    model, optimizer = tiny_source()
    before = copy.deepcopy(model.state_dict())
    moments = copy.deepcopy(optimizer.state_dict())
    random_state = torch.get_rng_state()
    result = run_shared_corrections(model, optimizer, fit_loss, assess, components)
    report = result.report
    assert report["numerically_valid"]
    assert (
        report["cases"]["encoder"]["finite_steps"][0]["assessment"]["predictor"]
        == assess(model)["predictor"]
    )
    assert (
        report["cases"]["predictor"]["finite_steps"][0]["assessment"]["encoder"]
        == assess(model)["encoder"]
    )
    for case, blocks in (
        ("encoder", ("encoder",)),
        ("predictor", ("predictor",)),
        ("joint", ("encoder", "predictor")),
    ):
        row = report["cases"][case]
        assert row["finite_steps"][0]["fit_loss"] < report["initial_fit_loss"]
        for block in blocks:
            assert row["movement"][block] == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
                report["reference_update_norms"][block]
            )
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state_dict(), moments, rtol=0, atol=0)
    assert torch.equal(random_state, torch.get_rng_state())


def test_shared_correction_rejects_missing_predictor_history_and_parameter_reordering() -> None:
    model, optimizer = tiny_source()
    for parameter in model.predictor.parameters():
        optimizer.state.pop(parameter, None)
    with pytest.raises(ValueError, match="history"):
        run_shared_corrections(model, optimizer, fit_loss, assess, components)
    model, optimizer = tiny_source()
    optimizer.param_groups[0]["params"].reverse()
    with pytest.raises(ValueError, match="order"):
        run_shared_corrections(model, optimizer, fit_loss, assess, components)


def test_settings_reject_validation_selected_or_nonshrinking_fractions() -> None:
    with pytest.raises(ValueError, match="fractions"):
        CorrectionSettings(fractions=(0.1, 1.0))


def test_zero_saved_update_is_undefined_and_missing_used_history_is_rejected() -> None:
    model, optimizer = tiny_source()
    optimizer.param_groups[0]["lr"] = 0.0
    result = run_shared_corrections(model, optimizer, fit_loss, assess, components)
    assert not result.report["numerically_valid"]
    assert all(not row["defined"] for row in result.report["cases"].values())
    model, optimizer = tiny_source()
    optimizer.state.pop(next(model.predictor.parameters()))
    with pytest.raises(ValueError, match="history"):
        run_shared_corrections(model, optimizer, fit_loss, assess, components)


def test_real_dgp_shared_corrections_keep_validation_and_source_isolated() -> None:
    from test_signal_path_geometry import tiny_data, tiny_model

    from diagnostics.joint_acquisition.baseline import (
        DiagnosticSelection,
        build_matched_batch,
        evaluate_responses,
        predict_matched,
        response_loss,
    )
    from diagnostics.joint_acquisition.incentives import (
        analyze_incentives,
        recorded_objective_components,
    )
    from scjepa.training.loop import ObjectiveConfig

    model = tiny_model()
    model.parameter_encoder.requires_grad_(True)
    model.predictor.requires_grad_(True)
    data, _ = tiny_data(model)
    data["validation"] = data.pop("test")
    source = copy.deepcopy(model.state_dict())
    cached = copy.deepcopy(data)
    selection = DiagnosticSelection((0, 1), (0, 1), (0,), 4, 5)
    fit = build_matched_batch(model, data, selection, "calibration").select_queries("fit")
    heldout = build_matched_batch(model, data, selection, "validation").select_queries("heldout")
    objective = ObjectiveConfig(
        state_regularizer="sigreg",
        parameter_regularizer="sigreg",
        state_regularizer_weight=0.4,
        parameter_regularizer_weight=0.4,
        parameter_consistency_weight=1,
        logit_weight=0.0001,
        regularizer_num_projections=8,
    )
    calibration = data["calibration"]
    batch = {
        "online": calibration["states_online"][:2],
        "target": calibration["states_target"][:2],
        "actions": calibration["actions"][:2],
    }

    def ordinary(candidate: SCJEPA) -> dict[str, Tensor]:
        return recorded_objective_components(candidate, batch, objective, seed=81)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0
    )
    # Give every potentially used parameter coherent history without changing the
    # frozen state maps used to produce the evaluation cache.
    ordinary(model)["total"].backward()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    model.load_state_dict(source)
    optimizer.zero_grad(set_to_none=True)

    def fitting(candidate: SCJEPA) -> Tensor:
        return response_loss(candidate, fit)

    def assessment(candidate: SCJEPA) -> dict[str, Any]:
        summary, _ = evaluate_responses(candidate, heldout, bootstrap_samples=16)
        with torch.no_grad():
            codes = predict_matched(candidate, heldout).codes
        summary["codes"] = codes.tolist()  # pyright: ignore[reportUnknownMemberType]
        return summary

    moments = copy.deepcopy(optimizer.state_dict())
    random_state = torch.get_rng_state()
    result = run_shared_corrections(model, optimizer, fitting, assessment, ordinary)
    assert (
        result.report["cases"]["predictor"]["finite_steps"][0]["assessment"]["codes"]
        == result.report["initial_assessment"]["codes"]
    )
    incentives = analyze_incentives(model, ordinary, result, fit_loss=fitting)
    assert len(incentives["raw_group_components"]["prediction"]) == 2
    for block in ("encoder", "predictor"):
        assert incentives["component_gradient_norms_by_block"][block]["frozen_state"] == 0
    assert torch.equal(random_state, torch.get_rng_state())
    torch.testing.assert_close(model.state_dict(), source, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state_dict(), moments, rtol=0, atol=0)
    torch.testing.assert_close(data, cached, rtol=0, atol=0)
