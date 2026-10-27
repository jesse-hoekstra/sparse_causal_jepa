import copy
from collections.abc import Generator
from dataclasses import replace
from typing import Any

import pytest
import torch
from test_signal_path_geometry import tiny_data, tiny_model

from diagnostics.joint_acquisition.baseline import (
    DiagnosticSelection,
    MatchedBatch,
    build_matched_batch,
)
from diagnostics.joint_acquisition.corrections import CorrectionCase
from diagnostics.joint_acquisition.symmetric_audit import (
    load_repaired_model,
    paired_sign_report,
    verify_positive_reproduction,
)
from diagnostics.multiquery_control.data import module_fingerprint
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_signed_matched_batch_preserves_default_and_exact_negative_dynamics() -> None:
    model = tiny_model()
    data, observation_map = tiny_data(model)
    data["validation"] = data.pop("test")
    split = data["validation"]
    split["theta"][0, 0], split["theta"][1, 0] = 0.749, -0.749
    states = dgp.simulate(split["true_states"][:, :, 0], split["theta"][:, None], split["actions"])
    split["true_states"] = states
    with torch.no_grad():
        for key, encoder in (
            ("states_online", model.online_encoder),
            ("states_target", model.target_encoder),
        ):
            split[key] = encoder.encode_sequence(
                observation_map.forward(states).flatten(0, 1)
            ).reshape(3, 10, 12, 6)
    selection = DiagnosticSelection((0, 1), (0, 1), (0,), 4, 5)
    default = build_matched_batch(model, data, selection, "validation")
    positive = build_matched_batch(model, data, selection, "validation", perturbation_sign=1)
    negative = build_matched_batch(model, data, selection, "validation", perturbation_sign=-1)
    for name in ("targets", "changed_targets", "changed_donor_states", "finite_steps"):
        torch.testing.assert_close(getattr(default, name), getattr(positive, name), rtol=0, atol=0)
    torch.testing.assert_close(positive.finite_steps, -negative.finite_steps, rtol=0, atol=0)
    assert bool((negative.finite_steps < 0).all())
    assert bool((positive.finite_steps[:, 0] < 0.001).all())
    split = data["validation"]
    theta = split["theta"][:2]
    expected_steps = torch.minimum((0.75 - theta.abs()) / 2, torch.full_like(theta, 0.01))
    torch.testing.assert_close(negative.finite_steps, -expected_steps, rtol=0, atol=0)
    observation = VectorObservationMap(matrix=data["metadata"]["observation_matrix"], dimension=6)
    states = split["true_states"][:2, 1:10, 9]
    actions = split["true_actions"][:2, 1:10, 9:11]
    changed = theta.clone()
    changed[:, 0] += negative.finite_steps[:, 0]
    exact = dgp.simulate(states, changed[:, None], actions)[..., 1:, :]
    expected = model.target_encoder.encode_sequence(
        observation.forward(exact).flatten(0, 1)
    ).reshape(2, 1, 9, 2, 6)
    torch.testing.assert_close(negative.changed_targets[:, :, 0], expected, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match="exactly -1 or 1"):
        build_matched_batch(model, data, selection, "validation", perturbation_sign=0)  # pyright: ignore[reportArgumentType]


def test_paired_signs_cancel_linear_bias_without_hiding_negative_original_benefit() -> None:
    groups, donors, queries = 4, 1, 5
    target = torch.zeros(groups, donors, queries, 2, 6, dtype=torch.float64)
    positive_target = target[:, :, None].expand(-1, -1, 6, -1, -1, -1) + 0.01
    plus = MatchedBatch(
        torch.zeros(groups, donors, 10, 6),
        torch.zeros(groups, donors, 9, 16),
        torch.zeros(groups, donors, 6, 10, 6),
        torch.zeros(groups, donors, queries, 1, 6),
        torch.zeros(groups, donors, queries, 2, 16),
        target,
        positive_target,
        torch.full((groups, 6), 0.01),
        tuple(range(groups)),
        (0,),
        torch.arange(queries)[None],
        "validation",
        0,
        0.0,
    )
    minus = replace(plus, changed_targets=-positive_target, finite_steps=-plus.finite_steps)
    base = target + 1
    positive = base[:, :, None].expand_as(positive_target) + 0.01
    negative = base[:, :, None].expand_as(positive_target) - 0.01
    report = paired_sign_report(base, positive, negative, plus, minus, bootstrap_samples=32)
    original_report = report["reports"]["both_horizons"]["positive_original_criterion"]
    verify_positive_reproduction(original_report, original_report)
    with pytest.raises(ValueError, match="did not reproduce"):
        verify_positive_reproduction(
            original_report, {**original_report, "ordinary_mse_by_horizon": [100, 100]}
        )
    for horizon in report["reports"].values():
        original = horizon["positive_original_criterion"]["per_parameter"][0]
        paired = horizon["paired_sensitivity"]["per_parameter"][0]
        assert original["predictive_benefit"]["interval_95"][1] < 0
        assert abs(original["predictive_benefit"]["estimate"] + 0.0199) < 1e-12
        assert abs(paired["predictive_benefit"]["estimate"] - 0.0001) < 1e-12
        assert paired["normalized_response_error"]["estimate"] < 1e-20
        assert abs(paired["residual_cross_even"]["estimate"]) < 1e-14
        assert abs(paired["residual_cross_odd"]["estimate"] + 0.02) < 1e-12
    with pytest.raises(ValueError, match="exactly opposite"):
        paired_sign_report(
            base, positive, negative, plus, replace(minus, finite_steps=minus.finite_steps * 2)
        )


@pytest.mark.parametrize("case", ["encoder", "predictor"])
def test_repaired_endpoint_rejects_mislabeled_or_changed_inactive_block(
    case: CorrectionCase,
) -> None:
    primary = tiny_model()
    payload: dict[str, Any] = {
        "identity": "expected-trajectory",
        "status": "complete",
        "step": 1000,
        "protocol": {
            "case": case,
            "parent_identity": "parent",
            "source_model": {
                name: module_fingerprint(module)
                for name, module in (
                    ("online", primary.online_encoder),
                    ("target", primary.target_encoder),
                    ("encoder", primary.parameter_encoder),
                    ("predictor", primary.predictor),
                )
            },
        },
        "model": copy.deepcopy(primary.state_dict()),
    }
    record = {"identity": "expected-trajectory", "status": "complete", "updates": 1000}
    load_repaired_model(
        primary, payload, case=case, parent_identity="parent", trajectory_record=record
    )
    mislabeled = {**payload, "protocol": {**payload["protocol"], "case": "joint"}}
    with pytest.raises(ValueError, match="case or identity"):
        load_repaired_model(
            primary, mislabeled, case=case, parent_identity="parent", trajectory_record=record
        )
    with pytest.raises(ValueError, match="case or identity"):
        load_repaired_model(
            primary,
            {**payload, "identity": "different"},
            case=case,
            parent_identity="parent",
            trajectory_record=record,
        )
    changed = copy.deepcopy(primary)
    inactive = changed.predictor if case == "encoder" else changed.parameter_encoder
    with torch.no_grad():
        next(inactive.parameters()).add_(1)
    with pytest.raises(ValueError, match="inactive"):
        load_repaired_model(
            primary,
            {**payload, "model": changed.state_dict()},
            case=case,
            parent_identity="parent",
            trajectory_record=record,
        )
