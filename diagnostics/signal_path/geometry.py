"""Separate local context inference from predictive use in frozen checkpoints.

Every code direction remains a full six-vector. True parameter coordinates are
used only to generate diagnostic derivatives, never as learned-code coordinates.
"""

import math
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.multiquery_control.data import EvaluationData, TensorSplit
from diagnostics.multiquery_control.evaluation import encode_codes
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.models.model import SCJEPA


def predict_rollout(model: SCJEPA, history: Tensor, code: Tensor, actions: Tensor) -> Tensor:
    """Predict attached autoregressive steps from (N,1,6), (N,6), (N,H,16)."""
    predictions: list[Tensor] = []
    for step in range(actions.shape[1]):
        prediction = model.predictor(history, code, actions[:, step : step + 1]).prediction
        predictions.append(prediction)
        history = prediction[:, None]
    return torch.stack(predictions, dim=1)


def exact_rollout(
    model: SCJEPA,
    observation_map: VectorObservationMap,
    initial_states: Tensor,
    theta: Tensor,
    actions: Tensor,
) -> Tensor:
    """Encode exact future states, holding recipient initial states/actions fixed.

    Initial states have shape (N,6), actions (N,H,16), and theta is broadcastable
    to (N,6). Both the physical rollout and target encoder retain derivatives.
    """
    future = dgp.simulate(initial_states, theta, actions)[:, 1:]
    return model.target_encoder.encode_sequence(observation_map.forward(future))


def donor_code(
    model: SCJEPA,
    observation_map: VectorObservationMap,
    initial_state: Tensor,
    theta: Tensor,
    actions: Tensor,
) -> Tensor:
    """Infer one code through exact donor dynamics and the frozen online encoder."""
    states = dgp.simulate(initial_state, theta, actions)
    online = model.online_encoder.encode_sequence(observation_map.forward(states)[None])
    return cast(Tensor, model.parameter_encoder(online, actions[None]))[0]


def jacobian_columns(function: Callable[[Tensor], Tensor], point: Tensor) -> Tensor:
    """Return output axes followed by input-coordinate axes using exact JVPs."""
    if point.ndim != 1:
        raise ValueError("Jacobian point must be a vector")
    directions = torch.eye(len(point), dtype=point.dtype, device=point.device)
    columns: list[Tensor] = []
    with torch.enable_grad():
        for direction in directions:
            _, derivative = cast(
                tuple[Tensor, Tensor],
                torch.autograd.functional.jvp(  # pyright: ignore[reportUnknownMemberType]
                    function, point.detach(), direction, create_graph=False, strict=False
                ),
            )
            columns.append(derivative.detach())
    return torch.stack(columns, dim=-1)


def fit_code_directions(predictor: Tensor, target: Tensor, rcond: float) -> tuple[Tensor, Tensor]:
    """Fit six shared code directions from query/output-by-code Jacobian matrices.

    The truncated SVD is unregularized within retained directions. Validation on
    separate query episodes and a finite-displacement check are required: a good
    local fit alone does not establish usable conditioning or generalization.
    """
    if (
        predictor.ndim != 2
        or target.ndim != 2
        or predictor.shape[0] != target.shape[0]
        or not 0 < rcond < 1
    ):
        raise ValueError("Expected matching two-dimensional Jacobians and 0<rcond<1")
    u, singular, vh = cast(
        tuple[Tensor, Tensor, Tensor],
        torch.linalg.svd(predictor, full_matrices=False),  # pyright: ignore[reportUnknownMemberType]
    )  # pyright: ignore[reportUnknownMemberType]
    retained = singular > singular.max() * rcond
    inverse = torch.where(
        retained, singular.clamp_min(torch.finfo(singular.dtype).tiny).reciprocal(), 0
    )
    directions = (vh.T * inverse) @ u.T @ target
    return directions, singular


def response_metrics(response: Tensor, reference: Tensor) -> dict[str, Any]:
    """Aggregate response alignment by true factor; derivative baselines are zero."""
    if response.shape != reference.shape or response.shape[-1] != 6:
        raise ValueError("Response arrays must match and end in six true-factor columns")
    predicted = response.reshape(-1, 6)
    exact = reference.reshape(-1, 6)
    predicted_energy = predicted.square().sum(0)
    exact_energy = exact.square().sum(0)
    product = (predicted * exact).sum(0)
    difference = (predicted - exact).square().sum(0)
    rows: dict[str, Any] = {}
    for factor in range(6):
        ep, et = float(predicted_energy[factor]), float(exact_energy[factor])
        denominator = math.sqrt(ep * et)
        rows[f"theta_{factor + 1}"] = {
            "response_cosine": float(product[factor]) / denominator if denominator > 0 else None,
            "response_cosine_defined": denominator > 0,
            "response_energy_ratio": ep / et if et > 0 else None,
            "relative_response_error": float(difference[factor]) / et if et > 0 else None,
            "target_response_energy_mean": et / len(exact),
            "predicted_response_energy_mean": ep / len(exact),
        }
    return rows


def _vector(value: Tensor) -> list[float]:
    return cast(list[float], value.detach().cpu().tolist())  # pyright: ignore[reportUnknownMemberType]


def _relative_error(actual: Tensor, expected: Tensor) -> float:
    numerator = float((actual - expected).square().sum())
    denominator = float(expected.square().sum())
    return math.sqrt(numerator / denominator) if denominator > 0 else math.sqrt(numerator)


def _bounded_steps(theta: Tensor, displacement: float) -> Tensor:
    remaining = dgp.PARAMETER_BOUND - theta.abs()
    if not bool((remaining > 0).all()):
        raise ValueError("Differential diagnostic requires theta strictly inside its support")
    return remaining.mul(0.5).clamp(max=displacement)


def _finite_columns(function: Callable[[Tensor], Tensor], point: Tensor, steps: Tensor) -> Tensor:
    columns: list[Tensor] = []
    with torch.no_grad():
        for factor, step in enumerate(steps):
            displacement = torch.zeros_like(point)
            displacement[factor] = step
            columns.append(
                (function(point + displacement) - function(point - displacement)) / (2 * step)
            )
    return torch.stack(columns, -1)


def geometry_diagnostic(
    model: SCJEPA,
    data: EvaluationData,
    *,
    groups: int,
    donor_contexts: int,
    fit_queries: int,
    test_queries: int,
    rcond: float = 1e-6,
    displacement: float = 0.01,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Measure donor sensitivity, predictor transmission and held-out local access.

    Query selection is deterministic cyclic order excluding the donor. The first
    fit_queries fit a direction; the subsequent test_queries only evaluate it.
    Model parameters and their .grad buffers are never mutated. The diagnostic
    uses a double-precision evaluation copy and reconstructs smooth state maps.
    """
    parameter = next(model.parameters())
    if parameter.dtype != torch.float64 or model.training:
        raise ValueError("Use a float64 model in eval mode")
    if model.context_len != 10 or model.predictor.history_len != 1 or not model.autoregressive:
        raise ValueError("This diagnostic requires K10, L1 and autoregressive prediction")
    if min(groups, donor_contexts, fit_queries, test_queries) < 1:
        raise ValueError("Group, donor and query counts must be positive")
    if not 0 < rcond < 1 or not math.isfinite(displacement) or displacement <= 0:
        raise ValueError("Require 0<rcond<1 and finite positive displacement")
    split = cast(TensorSplit, data["test"])
    calibration = cast(TensorSplit, data["calibration"])
    metadata = cast(dict[str, Any], data["metadata"])
    views = split["true_states"].shape[1]
    if views != 10 or donor_contexts > views or fit_queries + test_queries > views - 1:
        raise ValueError("Use R10, at most ten donors and at most nine non-self query episodes")
    if groups > len(split["theta"]):
        raise ValueError("Requested groups exceed available test groups")
    observation_map = VectorObservationMap(
        matrix=cast(Tensor, metadata["observation_matrix"]), dimension=6
    )
    with torch.no_grad():
        calibration_codes = encode_codes(model, calibration, batch_size=128).double()
        code_std = float(calibration_codes.flatten(0, 1).var(0, correction=0).mean().sqrt())
    if not math.isfinite(code_std):
        raise FloatingPointError("Nonfinite calibration code scale")
    code_scale = code_std if code_std > 0 else 1.0
    store: dict[str, list[Tensor]] = {
        key: []
        for key in (
            "encoder_jacobian",
            "predictor_jacobian",
            "target_jacobian",
            "fitted_code_directions",
            "nonlinear_fitted_response",
            "finite_steps",
            "codes",
            "query_indices",
            "encoder_singular_values",
            "predictor_singular_values",
            "target_singular_values",
        )
    }
    checks: dict[str, float] = {}
    cached_code_error: list[Tensor] = []
    theta_values = split["theta"][:groups].to(parameter)
    states = split["true_states"][:groups].to(parameter)
    actions = split["true_actions"][:groups].to(parameter)
    for group in range(groups):
        theta = theta_values[group]
        steps = _bounded_steps(theta, displacement)
        for donor in range(donor_contexts):
            indices = [
                (donor + offset) % views for offset in range(1, fit_queries + test_queries + 1)
            ]
            initial = states[group, donor, 0]
            donor_actions = actions[group, donor, :9]
            query_states = states[group, indices, 9]
            query_actions = actions[group, indices, 9:11]
            with torch.no_grad():
                history = model.online_encoder.encode_sequence(
                    observation_map.forward(query_states[:, None])
                )

            def infer(
                value: Tensor,
                initial: Tensor = initial,
                donor_actions: Tensor = donor_actions,
            ) -> Tensor:
                return donor_code(model, observation_map, initial, value, donor_actions)

            with torch.no_grad():
                code = infer(theta)
                cached = cast(
                    Tensor,
                    model.parameter_encoder(
                        split["states_online"][group, donor, :10].to(parameter)[None],
                        split["actions"][group, donor, :9].to(parameter)[None],
                    ),
                )[0]
                cached_code_error.append((code - cached).square().mean().detach().cpu())

            def predict(
                value: Tensor,
                history: Tensor = history,
                queries: int = len(indices),
                query_actions: Tensor = query_actions,
            ) -> Tensor:
                return predict_rollout(model, history, value.expand(queries, -1), query_actions)

            def exact(
                value: Tensor,
                query_states: Tensor = query_states,
                query_actions: Tensor = query_actions,
            ) -> Tensor:
                return exact_rollout(model, observation_map, query_states, value, query_actions)

            a = jacobian_columns(infer, theta)
            p = jacobian_columns(predict, code)
            b = jacobian_columns(exact, theta)
            direction, singular = fit_code_directions(
                p[:fit_queries].reshape(-1, 6), b[:fit_queries].reshape(-1, 6), rcond
            )
            finite: list[Tensor] = []
            with torch.no_grad():
                for factor, step in enumerate(steps):
                    delta = step * direction[:, factor]
                    finite.append((predict(code + delta) - predict(code - delta)) / (2 * step))
            nonlinear = torch.stack(finite, -1)
            if group == 0 and donor == 0:

                def composed(value: Tensor) -> Tensor:
                    return predict(infer(value))

                direct = jacobian_columns(composed, theta)
                small_steps = _bounded_steps(theta, 1e-4)
                checks = {
                    "chain_rule_relative_error": _relative_error(p @ a, direct),
                    "encoder_finite_difference_relative_error": _relative_error(
                        _finite_columns(infer, theta, small_steps), a
                    ),
                    "composed_finite_difference_relative_error": _relative_error(
                        _finite_columns(composed, theta, small_steps), direct
                    ),
                    "target_finite_difference_relative_error": _relative_error(
                        _finite_columns(exact, theta, small_steps), b
                    ),
                }
            values = {
                "encoder_jacobian": a,
                "predictor_jacobian": p,
                "target_jacobian": b,
                "fitted_code_directions": direction,
                "nonlinear_fitted_response": nonlinear,
                "finite_steps": steps,
                "codes": code,
                "query_indices": torch.tensor(indices),
                "encoder_singular_values": cast(Tensor, torch.linalg.svdvals(a)),  # pyright: ignore[reportUnknownMemberType]
                "predictor_singular_values": singular,
                "target_singular_values": cast(
                    Tensor,
                    torch.linalg.svdvals(b[:fit_queries].reshape(-1, 6)),  # pyright: ignore[reportUnknownMemberType]
                ),  # pyright: ignore[reportUnknownMemberType]
            }
            for key, value in values.items():
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f"Nonfinite {key} for group {group}, donor {donor}")
                store[key].append(value.detach().cpu())
    raw = {
        key: torch.stack(values).reshape(groups, donor_contexts, *values[0].shape)
        for key, values in store.items()
    }
    raw["theta"] = theta_values.detach().cpu()
    raw["calibration_code_std"] = torch.tensor(code_std, dtype=torch.float64)
    a, p, b = (raw[key] for key in ("encoder_jacobian", "predictor_jacobian", "target_jacobian"))
    directions = raw["fitted_code_directions"]
    transmitted = torch.einsum("gdqhoi,gdit->gdqhot", p, a)
    fitted = torch.einsum("gdqhoi,gdit->gdqhot", p, directions)
    raw["transmitted_response"] = transmitted
    raw["fitted_linear_response"] = fitted
    sensitivity_energy = a.square().sum(-2)
    alignment_numerator = a.mean(1).square().sum(-2).sum(0)
    alignment_denominator = sensitivity_energy.mean(1).sum(0)
    alignment = [
        float(n / d) if donor_contexts > 1 and float(d) > 0 else None
        for n, d in zip(alignment_numerator, alignment_denominator, strict=True)
    ]
    corrected_alignment = [
        (donor_contexts * value - 1) / (donor_contexts - 1)
        if value is not None and donor_contexts > 1
        else None
        for value in alignment
    ]
    required_shift = raw["finite_steps"] * directions.square().mean(-2).sqrt() / code_scale
    raw["required_code_shift_rms"] = required_shift * code_scale
    if code_std > 0:
        raw["required_code_shift_in_calibration_std"] = required_shift
    summary: dict[str, Any] = {
        "groups": groups,
        "donor_contexts_per_group": donor_contexts,
        "fit_queries": fit_queries,
        "test_queries": test_queries,
        "rcond": rcond,
        "nominal_displacement": displacement,
        "calibration_code_std": code_std,
        "calibration_has_nonzero_variance": code_std > 0,
        "encoder_sensitivity_rms_per_theta_std_in_code_std": (
            _vector(
                sensitivity_energy.mean((0, 1)).sqrt()
                * (dgp.PARAMETER_BOUND / math.sqrt(3))
                / (math.sqrt(6) * code_scale)
            )
            if code_std > 0
            else None
        ),
        "across_context_derivative_alignment": alignment,
        "across_context_derivative_alignment_available": donor_contexts > 1,
        "across_context_independent_direction_alignment_baseline": 1 / donor_contexts,
        "across_context_cross_derivative_alignment_corrected": corrected_alignment,
        "required_code_shift_in_calibration_std_median": (
            _vector(required_shift.flatten(0, 1).median(0).values) if code_std > 0 else None
        ),
        "required_code_shift_in_calibration_std_maximum": (
            _vector(required_shift.flatten(0, 1).amax(0)) if code_std > 0 else None
        ),
        "encoder_singular_values_median": _vector(
            raw["encoder_singular_values"].flatten(0, 1).median(0).values
        ),
        "predictor_fit_singular_values_median": _vector(
            raw["predictor_singular_values"].flatten(0, 1).median(0).values
        ),
        "target_fit_singular_values_median": _vector(
            raw["target_singular_values"].flatten(0, 1).median(0).values
        ),
        "actual_transmission_test": response_metrics(
            transmitted[:, :, fit_queries:], b[:, :, fit_queries:]
        ),
        "best_local_access_fit": response_metrics(
            fitted[:, :, :fit_queries], b[:, :, :fit_queries]
        ),
        "best_local_access_test": response_metrics(
            fitted[:, :, fit_queries:], b[:, :, fit_queries:]
        ),
        "finite_fitted_access_test": response_metrics(
            raw["nonlinear_fitted_response"][:, :, fit_queries:], b[:, :, fit_queries:]
        ),
        "checks": {
            **checks,
            "reconstructed_vs_cached_code_rms": float(torch.stack(cached_code_error).mean().sqrt()),
        },
        "protocol": (
            "A=dq/dtheta through one regenerated K10 donor; P=d(two-step prediction)/dq; "
            "B=d(exact encoded two-step target)/dtheta with recipient state/actions fixed. "
            "Fit one full-code direction per true factor on other episodes; test separate "
            "episodes. No query target enters donor inference."
        ),
        "limits": [
            "Local conditional sensitivity is not global information recoverability or "
            "invariance across contexts.",
            "Large fitted code shifts or poor finite-displacement agreement undermine "
            "interpreting a tangent fit as usable predictor access. Fitted directions need "
            "not lie on the encoder image or in its empirical code distribution.",
            "Singular values, code-direction norms and derivative alignment depend on code "
            "parameterization; no code coordinate is assigned to a true factor.",
            "Response metrics aggregate correlated donors/queries within theta groups. "
            "No training, checkpoint selection, or causal claim about past optimization is made.",
        ],
    }
    return summary, raw
