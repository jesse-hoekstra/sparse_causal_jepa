import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from diagnostics.joint_acquisition.code_repair import fit_bounded_code


def mixed_predictor() -> Callable[[Tensor], Tensor]:
    mixing = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    return lambda code: (mixing @ code).reshape(1, 1, 2)


def test_full_code_repair_transfers_and_bounds_actual_displacement() -> None:
    predictor = mixed_predictor()
    code = torch.zeros(2, dtype=torch.float64)
    reference = torch.tensor([[[0.1, 0.0]]], dtype=torch.float64)
    result = fit_bounded_code(predictor, predictor, code, reference, reference, code_std=1.0)
    torch.testing.assert_close(result.delta, torch.tensor([0.05, 0.05], dtype=torch.float64))
    assert result.summary["heldout_relative_response_error"] < 1e-20
    assert result.summary["displacement_rms_in_calibration_std"] <= 0.1
    bounded = fit_bounded_code(predictor, predictor, code, reference * 10, reference, code_std=1.0)
    assert math.isclose(bounded.summary["displacement_rms_in_calibration_std"], 0.1)


def test_fit_success_does_not_hide_heldout_reversal_or_select_a_fraction() -> None:
    predictor = mixed_predictor()
    code = torch.zeros(2, dtype=torch.float64)
    reference = torch.tensor([[[0.1, 0.0]]], dtype=torch.float64)
    result = fit_bounded_code(
        predictor, lambda value: -predictor(value), code, reference, reference, code_std=1.0
    )
    assert result.summary["fit_relative_response_error"] < 1e-20
    assert math.isclose(result.summary["heldout_relative_response_error"], 4.0)
    torch.testing.assert_close(result.delta, torch.tensor([0.05, 0.05], dtype=torch.float64))


def test_degenerate_calibration_scale_is_explicitly_undefined() -> None:
    predictor = mixed_predictor()
    with pytest.raises(ValueError, match="degenerate calibration"):
        fit_bounded_code(
            predictor,
            predictor,
            torch.zeros(2),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2),
            code_std=0.0,
        )


def test_real_code_repair_preserves_sources_and_excludes_heldout_fitting_targets() -> None:
    import copy
    from dataclasses import replace

    from test_signal_path_geometry import tiny_data, tiny_model

    from diagnostics.joint_acquisition.baseline import DiagnosticSelection, build_matched_batch
    from diagnostics.joint_acquisition.code_repair import bounded_code_report

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = tiny_model()
        data, _ = tiny_data(model)
        data["validation"] = data.pop("test")
        selection = DiagnosticSelection((0, 1), (0, 1), (0,), 4, 5)
        batch = build_matched_batch(model, data, selection, "calibration")
        state = copy.deepcopy(model.state_dict())
        cache = copy.deepcopy(data)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        rng = torch.get_rng_state().clone()
        summary, raw = bounded_code_report(model, batch, code_std=1.0, bootstrap_samples=16)
        changed_targets = batch.changed_targets.clone()
        changed_targets[:, :, :, 4:] += 10
        changed = replace(batch, changed_targets=changed_targets)
        changed_summary, changed_raw = bounded_code_report(
            model, changed, code_std=1.0, bootstrap_samples=16
        )
        torch.testing.assert_close(
            raw["code_displacements"], changed_raw["code_displacements"], rtol=0, atol=0
        )
        assert summary["after"]["per_parameter"] != changed_summary["after"]["per_parameter"]
        assert all(
            row["displacement_rms_in_calibration_std"] <= 0.1 + 1e-14
            for row in summary["per_donor_fits"]
        )
        torch.testing.assert_close(state, model.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(cache, data, rtol=0, atol=0)
        assert torch.equal(rng, torch.get_rng_state())
        assert all(
            parameter.grad is not None and bool((parameter.grad == 1).all())
            for parameter in model.parameters()
        )
    finally:
        torch.set_num_threads(previous)
