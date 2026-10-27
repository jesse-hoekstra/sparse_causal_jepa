import copy
from collections.abc import Generator

import pytest
import torch
from test_signal_path_geometry import tiny_data, tiny_model

from diagnostics.joint_acquisition.baseline import (
    DiagnosticSelection,
    build_matched_batch,
    predict_matched,
    response_report,
)


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_response_report_excludes_test_and_preserves_attached_two_step_chain() -> None:
    model = tiny_model()
    model.parameter_encoder.requires_grad_(True)
    data, _ = tiny_data(model)
    data["validation"] = data.pop("test")
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    selection = DiagnosticSelection((0, 1), (0, 1), (0,), 4, 5)
    batch = build_matched_batch(model, data, selection, "validation")
    assert batch.reconstruction_error < 1e-12
    assert torch.equal(batch.select_queries("fit").query_indices, torch.tensor([[1, 2, 3, 4]]))
    assert torch.equal(
        batch.select_queries("heldout").query_indices, torch.tensor([[5, 6, 7, 8, 9]])
    )
    output = predict_matched(model, batch)
    gradient = torch.autograd.grad(
        output.predictions[..., 1, :].square().sum(),
        tuple(model.parameter_encoder.parameters()),
        allow_unused=True,
    )
    assert any(value is not None and bool(value.abs().sum() > 0) for value in gradient)
    summary, raw = response_report(
        output.predictions.detach(),
        output.changed_predictions.detach(),
        batch,
        bootstrap_samples=32,
    )
    assert len(summary["per_parameter"]) == 6
    assert raw["response_error_energy"].shape == (2, 6)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(before, model.state_dict(), rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_zero_response_has_unit_error_and_zero_benefit() -> None:
    model = tiny_model()
    data, _ = tiny_data(model)
    data["validation"] = data.pop("test")
    selection = DiagnosticSelection((0, 1), (0, 1), (0,), 4, 5)
    batch = build_matched_batch(model, data, selection, "validation")
    summary, _ = response_report(
        batch.targets,
        batch.targets[:, :, None].expand_as(batch.changed_targets),
        batch,
        bootstrap_samples=32,
    )
    for row in summary["per_parameter"]:
        assert row["normalized_response_error"]["estimate"] == 1
        assert row["predictive_benefit"]["estimate"] == 0
        assert row["predictive_benefit"]["interval_95"] == [0, 0]


def test_invalid_reference_energy_is_undefined_and_changed_cache_is_rejected() -> None:
    from dataclasses import replace

    model = tiny_model()
    data, _ = tiny_data(model)
    data["validation"] = data.pop("test")
    selection = DiagnosticSelection((0, 1), (0, 1), (0,), 4, 5)
    batch = build_matched_batch(model, data, selection, "validation")
    zero_reference = replace(
        batch, changed_targets=batch.targets[:, :, None].expand_as(batch.changed_targets)
    )
    summary, _ = response_report(
        batch.targets, zero_reference.changed_targets, zero_reference, bootstrap_samples=16
    )
    assert all(
        row["normalized_response_error"]["estimate"] is None for row in summary["per_parameter"]
    )
    data["validation"]["states_online"][0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="Reconstructed"):
        build_matched_batch(model, data, selection, "validation")


def test_retention_uses_all_context_codes_and_never_reads_test_values() -> None:
    from diagnostics.joint_acquisition.baseline import validation_retention

    model = tiny_model()
    data, _ = tiny_data(model)
    data["validation"] = data.pop("test")
    selection = DiagnosticSelection((0, 1, 2), (0, 1, 2), (0,), 4, 5)
    report = validation_retention(model, data, selection)
    assert len(report["selected_validation_r2"]) == 6
    assert set(report["families"]) == {"linear", "cubic", "rbf_256_random_features"}
    assert "selected_test_r2" not in report
    assert report["contexts_per_group"] == 10


def test_compatible_identical_codes_give_zero_paired_difference() -> None:
    from diagnostics.joint_acquisition.baseline import compare_compatible_codes

    model = tiny_model()
    data, _ = tiny_data(model)
    data["validation"] = data.pop("test")
    selection = DiagnosticSelection((0, 1), (0, 1), (0,), 4, 5)
    summary, raw = compare_compatible_codes(
        model, copy.deepcopy(model), data, selection, bootstrap_samples=16
    )
    assert summary["paired_source_minus_current_ordinary_mse_by_horizon"] == [0, 0]
    assert torch.count_nonzero(raw["source_minus_current_predictive_benefit"]) == 0
    degraded = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in degraded.parameter_encoder.parameters():
            parameter.zero_()
    changed, _ = compare_compatible_codes(model, degraded, data, selection, bootstrap_samples=16)
    assert any(
        value != 0 for value in changed["paired_source_minus_current_ordinary_mse_by_horizon"]
    )
    with torch.no_grad():
        next(degraded.predictor.parameters()).add_(1)
    with pytest.raises(ValueError, match="identical frozen"):
        compare_compatible_codes(model, degraded, data, selection, bootstrap_samples=16)


def test_explicit_validation_geometry_checks_analytic_and_finite_responses() -> None:
    from diagnostics.joint_acquisition.baseline import numerical_checks

    model = tiny_model()
    data, _ = tiny_data(model)
    data["validation"] = data.pop("test")
    selection = DiagnosticSelection((0, 1), (0, 1), (0,), 4, 5)
    report, raw = numerical_checks(model, data, selection)
    assert report["passed"]
    assert report["geometry"]["split"] == "validation"
    assert raw["encoder_jacobian"].shape == (1, 1, 6, 6)
