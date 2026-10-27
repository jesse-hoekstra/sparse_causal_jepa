"""Direct supplied-theta forecasts and explicit validation/final-test counterfactuals."""

import math
from typing import Any, Literal, cast

import torch
from torch import Tensor

from diagnostics.joint_acquisition.baseline import MatchedBatch, response_report
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit
from diagnostics.signal_path.geometry import predict_rollout
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.models.model import SCJEPA

PARAMETER_STD = math.sqrt(0.1875)
Split = Literal["calibration", "validation", "test"]


def supplied_codes(theta: Tensor) -> Tensor:
    """Apply the fixed population standard deviation, without fitted normalization."""
    if theta.ndim != 2 or theta.shape[1] != 6 or not bool(torch.isfinite(theta).all()):
        raise ValueError("Exact diagnostic theta must be finite with shape Gx6")
    return theta / PARAMETER_STD


@torch.no_grad()
def rollout(model: SCJEPA, history: Tensor, codes: Tensor, actions: Tensor) -> Tensor:
    """Chunk inference while keeping the original two-step predictor semantics."""
    return torch.cat(
        [
            predict_rollout(
                model,
                history[start : start + 256],
                codes[start : start + 256],
                actions[start : start + 256],
            )
            for start in range(0, len(codes), 256)
        ]
    )


@torch.no_grad()
def forecast_report(model: SCJEPA, data: TensorSplit) -> dict[str, Any]:
    """Compare exact codes against zero codes on all explicitly supplied split groups."""
    reference = next(model.parameters())
    states, target, actions = (
        data[key].to(reference) for key in ("states_online", "states_target", "actions")
    )
    groups, views = states.shape[:2]
    codes = supplied_codes(data["theta"].to(reference))[:, None].expand(-1, views, -1).flatten(0, 1)
    history, query_actions = states[:, :, 9:10].flatten(0, 1), actions[:, :, 9:11].flatten(0, 1)
    truth = target[:, :, 10:12]
    variance = truth.permute(2, 0, 1, 3).flatten(1, 2).var(1, correction=0).mean(-1)
    result: dict[str, Any] = {
        "groups": groups,
        "target_variance_by_horizon": [float(x) for x in variance],
        "normalization": "Mean per-coordinate population variance across groups and views",
        "zero_code_scope": (
            "Input ablation of this same oracle predictor, not a fitted no-code baseline"
        ),
    }
    for name, values in (("exact_code", codes), ("zero_code", torch.zeros_like(codes))):
        predicted = rollout(model, history, values, query_actions).reshape(groups, views, 2, 6)
        error = (predicted - truth).square().mean((0, 1, 3))
        result[name] = {
            "mse_by_horizon": [float(x) for x in error],
            "normalized_mse_by_horizon": [float(x) for x in error / variance],
        }
    return result


@torch.no_grad()
def matched_batch(
    model: SCJEPA,
    data: EvaluationData,
    split: Split,
    indices: tuple[int, ...],
    *,
    final: bool = False,
) -> tuple[MatchedBatch, Tensor, Tensor]:
    """Build explicit split-specific exact futures; final-test access requires final=True."""
    if split not in ("calibration", "validation", "test") or (split == "test" and not final):
        raise ValueError("Test counterfactuals require the fixed final evaluation")
    if (
        model.training
        or model.context_len != 10
        or not model.autoregressive
        or model.predictor.history_len != 1
    ):
        raise ValueError("Require eval-mode original K10/L1 attached two-step model")
    source = cast(TensorSplit, data[split])
    if (
        not indices
        or min(indices) < 0
        or max(indices) >= len(source["theta"])
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("Invalid independent-group indices")
    reference = next(model.parameters())
    selected = {key: values[list(indices)].to(reference) for key, values in source.items()}
    theta, states, actions = (selected[key] for key in ("theta", "true_states", "true_actions"))
    groups = len(theta)
    remaining = 0.75 - theta.abs()
    if not bool((remaining > 0).all()):
        raise ValueError("Physical parameters must lie inside original support")
    steps = (remaining * 0.5).clamp(max=0.01)
    changed_theta = theta[:, None] + torch.diag_embed(steps)
    observation = VectorObservationMap(matrix=data["metadata"]["observation_matrix"], dimension=6)
    queries = torch.tensor([[5, 6, 7, 8, 9]])
    donor_actions = actions[:, :1, :9]
    initial = states[:, :1, 0]
    physical = dgp.simulate(initial, theta[:, None], donor_actions)
    donor = model.online_encoder.encode_sequence(
        observation.forward(physical).to(reference).flatten(0, 1)
    ).reshape(groups, 1, 10, 6)
    changed_physical = dgp.simulate(
        initial[:, :, None].expand(-1, -1, 6, -1),
        changed_theta[:, None],
        donor_actions[:, :, None].expand(-1, -1, 6, -1, -1),
    )
    changed_donor = model.online_encoder.encode_sequence(
        observation.forward(changed_physical).to(reference).flatten(0, 2)
    ).reshape(groups, 1, 6, 10, 6)
    query_states, query_actions = states[:, queries, 9], actions[:, queries, 9:11]
    history = model.online_encoder.encode_sequence(
        observation.forward(query_states).to(reference).reshape(-1, 1, 6)
    ).reshape(groups, 1, 5, 1, 6)
    futures = dgp.simulate(query_states, theta[:, None, None], query_actions)[..., 1:, :]
    targets = model.target_encoder.encode_sequence(
        observation.forward(futures).to(reference).flatten(0, 2)
    ).reshape(groups, 1, 5, 2, 6)
    changed_futures = dgp.simulate(
        query_states[:, :, None].expand(-1, -1, 6, -1, -1),
        changed_theta[:, None, :, None],
        query_actions[:, :, None].expand(-1, -1, 6, -1, -1, -1),
    )[..., 1:, :]
    changed_targets = model.target_encoder.encode_sequence(
        observation.forward(changed_futures).to(reference).flatten(0, 3)
    ).reshape(groups, 1, 6, 5, 2, 6)
    errors = [
        (physical - states[:, :1, :10]).abs().max(),
        (donor - selected["states_online"][:, :1, :10]).abs().max(),
        (history - selected["states_online"][:, queries, 9:10]).abs().max(),
        (targets - selected["states_target"][:, queries, 10:12]).abs().max(),
        (actions - selected["actions"]).abs().max(),
    ]
    error = max(float(value) for value in errors)
    if not math.isfinite(error) or error > 2e-5:
        raise ValueError("Counterfactual DGP/state maps do not reproduce cached split")
    batch = MatchedBatch(
        donor,
        donor_actions,
        changed_donor,
        history,
        query_actions,
        targets,
        changed_targets,
        steps,
        indices,
        (0,),
        queries,
        split,
        0,
        error,
    )
    return (
        batch,
        supplied_codes(theta),
        supplied_codes(changed_theta.flatten(0, 1)).reshape(groups, 6, 6),
    )


@torch.no_grad()
def assess(
    model: SCJEPA,
    data: EvaluationData,
    split: Split,
    *,
    groups: int,
    bootstrap_samples: int,
    seed: int,
    final: bool = False,
) -> dict[str, Any]:
    """Report oracle-code prediction and all-factor response without invoking E."""
    if split == "test" and not final:
        raise ValueError("Test reporting requires fixed final checkpoint")
    model.eval()
    source = cast(TensorSplit, data[split])
    ordinary = forecast_report(model, source)
    batch, codes, changed_codes = matched_batch(
        model, data, split, tuple(range(groups)), final=final
    )
    prediction = rollout(
        model,
        batch.history.flatten(0, 2),
        codes[:, None].expand(-1, 5, -1).flatten(0, 1),
        batch.query_actions.flatten(0, 2),
    ).reshape(groups, 1, 5, 2, 6)
    changed = rollout(
        model,
        batch.history[:, :, None].expand(-1, -1, 6, -1, -1, -1).flatten(0, 3),
        changed_codes[:, :, None].expand(-1, -1, 5, -1).flatten(0, 2),
        batch.query_actions[:, :, None].expand(-1, -1, 6, -1, -1, -1).flatten(0, 3),
    ).reshape(groups, 1, 6, 5, 2, 6)
    response, raw = response_report(
        prediction, changed, batch, bootstrap_samples=bootstrap_samples, seed=seed
    )
    response["interpretation"] = (
        "Exact-parameter diagnostic at fixed final test checkpoint; not label-free acquisition."
        if split == "test"
        else "Validation exact-parameter diagnostic; not label-free acquisition."
    )
    matched_variance = (
        batch.targets.permute(3, 0, 1, 2, 4).flatten(1, 3).var(1, correction=0).mean(-1)
    )
    response["ordinary_variance_normalized_mse_by_horizon"] = [
        float(error / variance)
        for error, variance in zip(
            response["ordinary_mse_by_horizon"], matched_variance, strict=True
        )
    ]
    response["target_variance_definition"] = (
        "Mean per-coordinate population variance across groups and queries"
    )
    for row in response["per_parameter"]:
        error, benefit = row["normalized_response_error"], row["predictive_benefit"]
        row["passes_predictive_use_bounds"] = bool(
            error["defined"]
            and benefit["defined"]
            and error["interval_95"][1] < 1
            and benefit["interval_95"][0] > 0
        )
    response["all_six_predictive_use_pass"] = all(
        row["passes_predictive_use_bounds"] for row in response["per_parameter"]
    )
    return {"split": split, "forecast": ordinary, "response": response, "raw": raw}
