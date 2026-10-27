"""Physical-coordinate counterfactuals and unchanged individual-context readouts."""

import copy
import math
from typing import Any, Literal, cast

import torch
from torch import Tensor

from diagnostics.joint_acquisition.baseline import (
    DiagnosticSelection,
    MatchedBatch,
    evaluate_responses,
    selected_split,
    validation_retention,
)
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit
from diagnostics.multiquery_control.evaluation import encode_codes, parameter_readouts
from diagnostics.physical_state_acquisition.data import PhysicalIdentityEncoder
from scjepa.data import simplified_gaussian as dgp
from scjepa.models.model import SCJEPA
from scjepa.training.rng import preserve_rng


def build_physical_matched_batch(
    model: SCJEPA,
    data: EvaluationData,
    selection: DiagnosticSelection,
    split: Literal["calibration", "validation", "test"],
    *,
    displacement: float = 0.01,
) -> MatchedBatch:
    """Regenerate raw physical contexts/futures; never apply the observation map.

    The code encoder still receives one context and only mixed observed actions.
    Test access is explicit, restricted by the caller to the fixed-budget endpoint.
    """
    if (
        model.training
        or model.context_len != 10
        or not model.autoregressive
        or model.predictor.history_len != 1
        or not isinstance(model.online_encoder, PhysicalIdentityEncoder)
        or not isinstance(model.target_encoder, PhysicalIdentityEncoder)
        or data["metadata"].get("state_coordinates") != "physical_identity"
    ):
        raise ValueError("Require the explicit eval-mode physical-coordinate K10/L1 model")
    if not math.isfinite(displacement) or displacement <= 0:
        raise ValueError("Physical perturbation must be finite and positive")
    calibration = selected_split(data, selection, "calibration")
    if split == "test":
        indices = selection.validation_indices
        source = cast(TensorSplit, data["test"])
        if not indices or min(indices) < 0 or max(indices) >= len(source["theta"]):
            raise ValueError("Test group indices exceed the independent test split")
        selected = {key: value[list(indices)] for key, value in source.items()}
    elif split in ("calibration", "validation"):
        indices = getattr(selection, split + "_indices")
        selected = selected_split(data, selection, split)
    else:
        raise ValueError("Unknown assessment split")
    if split != "calibration" and bool(
        (calibration["theta"][:, None] == selected["theta"][None]).all(-1).any()
    ):
        raise ValueError("Calibration and assessment parameter groups overlap")
    reference = next(model.parameters())
    theta, states, actions = (
        selected[key].to(reference) for key in ("theta", "true_states", "true_actions")
    )
    views = states.shape[1]
    donors = selection.donors
    if (
        views != 10
        or not donors
        or len(set(donors)) != len(donors)
        or min(donors) < 0
        or max(donors) >= views
        or min(selection.fit_queries, selection.heldout_queries) < 1
        or selection.fit_queries + selection.heldout_queries > views - 1
    ):
        raise ValueError("Require R10 donors and disjoint non-self recipient episodes")
    queries = torch.tensor(
        [
            [
                (donor + offset) % views
                for offset in range(1, selection.fit_queries + selection.heldout_queries + 1)
            ]
            for donor in donors
        ]
    )
    remaining = dgp.PARAMETER_BOUND - theta.abs()
    if not bool((remaining > 0).all()):
        raise ValueError("Parameters must be interior to their support")
    steps = remaining.mul(0.5).clamp(max=displacement)
    changed_theta = theta[:, None] + torch.diag_embed(steps)
    donor_actions = actions[:, donors, :9]
    initial = states[:, donors, 0]
    with torch.no_grad():
        donor_states = dgp.simulate(initial, theta[:, None], donor_actions)
        changed_donor = dgp.simulate(
            initial[:, :, None].expand(-1, -1, 6, -1),
            changed_theta[:, None],
            donor_actions[:, :, None].expand(-1, -1, 6, -1, -1),
        )
        query_states = states[:, queries, 9]
        query_actions = actions[:, queries, 9:11]
        targets = dgp.simulate(query_states, theta[:, None, None], query_actions)[..., 1:, :]
        changed_targets = dgp.simulate(
            query_states[:, :, None].expand(-1, -1, 6, -1, -1),
            changed_theta[:, None, :, None],
            query_actions[:, :, None].expand(-1, -1, 6, -1, -1, -1),
        )[..., 1:, :]
    errors = (
        (donor_states - states[:, donors, :10]).abs().max(),
        (donor_states - selected["states_online"][:, donors, :10].to(reference)).abs().max(),
        (query_states[:, :, :, None] - selected["states_online"][:, queries, 9:10].to(reference))
        .abs()
        .max(),
        (targets - selected["states_target"][:, queries, 10:12].to(reference)).abs().max(),
        (actions - selected["actions"].to(reference)).abs().max(),
    )
    maximum = max(float(value) for value in errors)
    if not math.isfinite(maximum) or maximum > 2e-5:
        raise ValueError(
            "Physical matched trajectories disagree with cached physical states/actions"
        )
    return MatchedBatch(
        donor_states,
        donor_actions,
        changed_donor,
        query_states[:, :, :, None],
        query_actions,
        targets,
        changed_targets,
        steps,
        indices,
        donors,
        queries,
        split,
        selection.fit_queries,
        maximum,
    )


def assess_model(
    model: SCJEPA,
    data: EvaluationData,
    selection: DiagnosticSelection,
    *,
    bootstrap_samples: int,
    seed: int,
    final: bool = False,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Keep validation diagnostic scores separate from the final independent test."""
    with preserve_rng(next(model.parameters()).device), torch.no_grad():
        candidate = copy.deepcopy(model).double().eval()
        split = "test" if final else "validation"
        matched = build_physical_matched_batch(candidate, data, selection, split).select_queries(
            "heldout"
        )
        response, raw = evaluate_responses(
            candidate, matched, bootstrap_samples=bootstrap_samples, seed=seed
        )
        if final:
            names = ("calibration", "validation", "test")
            for first, second in (("calibration", "test"), ("validation", "test")):
                if bool(
                    (data[first]["theta"][:, None] == data[second]["theta"][None]).all(-1).any()
                ):
                    raise ValueError("Final-test parameter groups overlap a probe-fitting split")
            retention = parameter_readouts(
                {name: encode_codes(candidate, data[name]) for name in names},
                {name: data[name] for name in names},
            )
            response["interpretation"] = (
                "Final fixed-budget independent-test response in privileged physical coordinates"
            )
        else:
            retention = validation_retention(candidate, data, selection)
            retention.pop("families")
        return {
            "response": response,
            "retention": retention,
            "physical_state_diagnostic": True,
        }, raw


def measurement_metrics(report: dict[str, Any], *, final: bool = False) -> dict[str, float]:
    """Expose every factor's recovery/use estimates and confidence bounds to W&B."""
    prefix = "test" if final else "validation"
    metrics: dict[str, float] = {
        f"{prefix}/response_group_count": float(len(report["response"]["group_indices"]))
    }
    for index, value in enumerate(
        report["retention"]["selected_test_r2" if final else "selected_validation_r2"], 1
    ):
        metrics[f"{prefix}/theta_{index}/retention_r2"] = float(value)
    for horizon, value in enumerate(report["response"]["ordinary_mse_by_horizon"], 1):
        metrics[f"{prefix}/ordinary_mse/horizon_{horizon}"] = float(value)
    for horizon, value in enumerate(
        report["response"]["ordinary_variance_normalized_mse_by_horizon"], 1
    ):
        if value is not None:
            metrics[f"{prefix}/variance_normalized_mse/horizon_{horizon}"] = float(value)
    for row in report["response"]["per_parameter"]:
        for name in ("normalized_response_error", "predictive_benefit"):
            entry = row[name]
            key = f"{prefix}/theta_{row['parameter']}/{name}"
            if entry["estimate"] is not None:
                metrics[key] = float(entry["estimate"])
            if entry["interval_95"] is not None:
                metrics[key + "/lower95"], metrics[key + "/upper95"] = entry["interval_95"]
    return metrics
