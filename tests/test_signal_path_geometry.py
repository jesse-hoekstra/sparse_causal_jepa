import copy
from collections.abc import Generator
from typing import Any

import pytest
import torch
from torch import Tensor

from diagnostics.signal_path.geometry import (
    donor_code,
    exact_rollout,
    fit_code_directions,
    geometry_diagnostic,
    jacobian_columns,
    predict_rollout,
    response_metrics,
)
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
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


def tiny_model() -> SCJEPA:
    torch.default_generator.manual_seed(52)
    return (
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
        .requires_grad_(False)
    )


def tiny_data(model: SCJEPA) -> tuple[dict[str, Any], VectorObservationMap]:
    observation_map = VectorObservationMap(matrix=torch.eye(6, dtype=torch.float64), dimension=6)
    data: dict[str, Any] = {"metadata": {"observation_matrix": observation_map.matrix}}
    generator = torch.Generator().manual_seed(193)
    for name in ("calibration", "test"):
        theta = 0.5 * torch.rand(3, 6, generator=generator, dtype=torch.float64) - 0.25
        controls = torch.randn(3, 10, 11, 16, generator=generator, dtype=torch.float64)
        controls[..., 10:] = controls[..., 10:].tanh()
        actions = dgp.encode_controls(controls)
        initial = torch.randn(3, 10, 6, generator=generator, dtype=torch.float64)
        states = dgp.simulate(initial, theta[:, None], actions)
        with torch.no_grad():
            online = model.online_encoder.encode_sequence(
                observation_map.forward(states).flatten(0, 1)
            ).reshape(3, 10, 12, 6)
            target = model.target_encoder.encode_sequence(
                observation_map.forward(states).flatten(0, 1)
            ).reshape(3, 10, 12, 6)
        data[name] = {
            "theta": theta,
            "true_states": states,
            "true_actions": actions,
            "actions": actions.clone(),
            "states_online": online,
            "states_target": target,
        }
    return data, observation_map


def test_jacobian_columns_and_chain_are_exact_with_mixed_coordinates() -> None:
    torch.default_generator.manual_seed(20)
    first = torch.randn(6, 6, dtype=torch.float64)
    second = torch.randn(12, 6, dtype=torch.float64)
    point = torch.randn(6, dtype=torch.float64)

    def encoder(theta: Tensor) -> Tensor:
        return (first @ theta).sin()

    def predictor(code: Tensor) -> Tensor:
        return (second @ code).tanh().reshape(2, 6)

    a = jacobian_columns(encoder, point)
    p = jacobian_columns(predictor, encoder(point))
    direct = jacobian_columns(lambda value: predictor(encoder(value)), point)
    torch.testing.assert_close(a, torch.diag((first @ point).cos()) @ first)
    torch.testing.assert_close(p @ a, direct, atol=1e-12, rtol=1e-12)


def test_access_fit_uses_all_code_coordinates_and_rejects_inexpressible_factors() -> None:
    torch.default_generator.manual_seed(202)
    p = torch.randn(24, 6, dtype=torch.float64)
    v = torch.randn(6, 6, dtype=torch.float64)
    exact = p @ v
    fitted, _ = fit_code_directions(p, exact, 1e-10)
    torch.testing.assert_close(fitted, v, atol=1e-12, rtol=1e-12)
    change = torch.eye(6, dtype=torch.float64) + 0.1 * torch.randn(6, 6, dtype=torch.float64)
    changed_p = p @ change
    changed_v, _ = fit_code_directions(changed_p, exact, 1e-10)
    torch.testing.assert_close(changed_p @ changed_v, p @ fitted, atol=1e-12, rtol=1e-12)
    deficient = torch.zeros(12, 6, dtype=torch.float64)
    deficient[:2, :2] = torch.eye(2, dtype=torch.float64)
    target = torch.cat((torch.eye(6, dtype=torch.float64), torch.eye(6, dtype=torch.float64)))
    fitted, singular = fit_code_directions(deficient, target, 1e-6)
    assert int((singular > 1e-6).sum()) == 2
    metrics = response_metrics(deficient @ fitted, target)
    assert metrics["theta_3"]["response_energy_ratio"] == 0
    assert metrics["theta_3"]["relative_response_error"] == 1
    assert metrics["theta_3"]["response_cosine"] is None


def test_held_out_query_fit_detects_a_direction_that_only_fits_support_queries() -> None:
    p_fit = torch.eye(6, dtype=torch.float64)
    b_fit = torch.eye(6, dtype=torch.float64)
    direction, _ = fit_code_directions(p_fit, b_fit, 1e-6)
    p_test = -torch.eye(6, dtype=torch.float64)
    fit = response_metrics(p_fit @ direction, b_fit)
    held_out = response_metrics(p_test @ direction, b_fit)
    assert fit["theta_1"]["relative_response_error"] == 0
    assert held_out["theta_1"]["relative_response_error"] == 4
    assert held_out["theta_1"]["response_cosine"] == -1


def test_exact_rollout_broadcasts_theta_and_retains_both_step_derivatives() -> None:
    model = tiny_model()
    data, observation_map = tiny_data(model)
    split = data["test"]
    theta = split["theta"][0]
    initial = split["true_states"][0, :2, 9]
    actions = split["true_actions"][0, :2, 9:11]
    shared = exact_rollout(model, observation_map, initial, theta, actions)
    individual = exact_rollout(model, observation_map, initial, theta.expand(2, -1), actions)
    torch.testing.assert_close(shared, individual, rtol=0, atol=0)
    derivative = jacobian_columns(
        lambda value: exact_rollout(model, observation_map, initial, value, actions), theta
    )
    assert derivative.shape == (2, 2, 6, 6)
    step = 1e-5
    perturbation = torch.zeros_like(theta)
    perturbation[3] = step
    finite = (
        exact_rollout(model, observation_map, initial, theta + perturbation, actions)
        - exact_rollout(model, observation_map, initial, theta - perturbation, actions)
    ) / (2 * step)
    torch.testing.assert_close(derivative[..., 3], finite, atol=1e-7, rtol=1e-5)


def test_end_to_end_geometry_preserves_state_and_respects_query_partition() -> None:
    model = tiny_model()
    data, observation_map = tiny_data(model)
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state()
    summary, raw = geometry_diagnostic(
        model, data, groups=1, donor_contexts=2, fit_queries=1, test_queries=2
    )
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(before, model.state_dict(), atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert raw["encoder_jacobian"].shape == (1, 2, 6, 6)
    assert raw["predictor_jacobian"].shape == (1, 2, 3, 2, 6, 6)
    assert torch.equal(raw["query_indices"], torch.tensor([[[1, 2, 3], [2, 3, 4]]]))
    assert summary["checks"]["chain_rule_relative_error"] < 1e-8
    assert summary["checks"]["encoder_finite_difference_relative_error"] < 1e-5
    assert summary["checks"]["target_finite_difference_relative_error"] < 1e-5
    assert summary["checks"]["reconstructed_vs_cached_code_rms"] < 1e-12
    theta = data["test"]["theta"][0]
    manual = donor_code(
        model,
        observation_map,
        data["test"]["true_states"][0, 0, 0],
        theta,
        data["test"]["true_actions"][0, 0, :9],
    )
    torch.testing.assert_close(manual, raw["codes"][0, 0], rtol=0, atol=0)
    q = raw["codes"][0, 0]
    history = data["test"]["states_online"][0, 1:4, 9:10]
    actions = data["test"]["true_actions"][0, 1:4, 9:11]
    jacobian = jacobian_columns(
        lambda value: predict_rollout(model, history, value.expand(3, -1), actions), q
    )
    torch.testing.assert_close(jacobian, raw["predictor_jacobian"][0, 0], atol=1e-12, rtol=1e-12)


def test_invalid_query_partition_and_boundary_theta_fail_explicitly() -> None:
    model = tiny_model()
    data, _ = tiny_data(model)
    with pytest.raises(ValueError, match="nine non-self"):
        geometry_diagnostic(model, data, groups=1, donor_contexts=2, fit_queries=5, test_queries=5)
    data["test"]["theta"][0, 0] = dgp.PARAMETER_BOUND
    with pytest.raises(ValueError, match="strictly inside"):
        geometry_diagnostic(model, data, groups=1, donor_contexts=1, fit_queries=1, test_queries=1)
