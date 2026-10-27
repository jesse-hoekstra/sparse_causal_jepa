"""Paired group-bootstrap MCC comparisons of final fixed-budget evaluations."""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from scjepa.eval.recovery import CoordinateMatch
from scjepa.experiments.storage import read_json


@dataclass(frozen=True)
class _Condition:
    directory: Path
    record: dict[str, object]
    provenance: dict[str, object]
    metrics: dict[str, float]
    groups: Tensor
    calibration_groups: Tensor
    states: Tensor
    online: Tensor
    match: CoordinateMatch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(directory: Path, fixed_budget: int) -> _Condition:
    evaluation = directory / "evaluation"
    record = read_json(directory / "run_config.json")
    provenance = read_json(evaluation / "evaluated_checkpoint.json")
    train = record.get("train")
    if (
        read_json(directory / "status.json").get("status") != "completed"
        or not isinstance(train, dict)
        or cast(dict[str, object], train).get("steps") != fixed_budget
        or provenance.get("evaluated_step") != fixed_budget
        or provenance.get("checkpoint_path") != "last.pt"
        or not isinstance(record.get("run_identity"), str)
        or not record["run_identity"]
        or provenance.get("run_identity") != record["run_identity"]
    ):
        raise ValueError("comparison requires verified final fixed-budget last.pt evaluations")
    samples_path = evaluation / "mcc_samples.pt"
    if provenance.get("checkpoint_sha256") != _sha256(directory / "last.pt") or provenance.get(
        "mcc_samples_sha256"
    ) != _sha256(samples_path):
        raise ValueError("checkpoint or MCC sample artifact identity changed")
    payload = cast(
        dict[str, object], torch.load(samples_path, map_location="cpu", weights_only=True)
    )
    if payload.get("format_version") != 1:
        raise ValueError("unsupported MCC sample artifact")
    tensors: dict[str, Tensor] = {}
    for key in ("group_ids", "calibration_group_ids", "states", "online", "state_match"):
        value = payload.get(key)
        if not isinstance(value, Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"missing or invalid MCC sample {key}")
        tensors[key] = value
    groups, calibration_groups = tensors["group_ids"], tensors["calibration_group_ids"]
    states, online, columns = tensors["states"], tensors["online"], tensors["state_match"]
    if (
        groups.ndim != 1
        or groups.dtype != torch.int64
        or groups.numel() < 2
        or len(set(int(value) for value in groups)) != groups.numel()
        or calibration_groups.ndim != 1
        or calibration_groups.dtype != torch.int64
        or calibration_groups.numel() < 2
        or len(set(int(value) for value in calibration_groups)) != calibration_groups.numel()
        or bool(torch.isin(groups, calibration_groups).any())
        or states.ndim != 4
        or states.shape != online.shape
        or states.shape[0] != groups.numel()
        or any(size == 0 for size in states.shape)
        or columns.ndim != 1
        or columns.dtype != torch.int64
        or columns.numel() != states.shape[-1]
    ):
        raise ValueError("invalid grouped MCC samples or calibration/test overlap")
    match = CoordinateMatch(tuple(int(value) for value in columns))
    values = read_json(evaluation / "metrics.json")
    if any(
        not isinstance(value, int | float) or not math.isfinite(value) for value in values.values()
    ):
        raise ValueError("nonfinite or nonscalar evaluation metrics")
    metrics = {key: float(value) for key, value in cast(dict[str, float], values).items()}
    return _Condition(
        directory, record, provenance, metrics, groups, calibration_groups, states, online, match
    )


def _scores(condition: _Condition, multiplicities: Tensor) -> Tensor:
    """Recompute pooled Pearson correlations from additive per-group moments."""
    truth = condition.states.double().flatten(1, 2)
    features = condition.online.double().flatten(1, 2)[:, :, list(condition.match.columns)]
    rows = truth.shape[1]
    count = multiplicities.sum(1, keepdim=True) * rows
    truth_sum = multiplicities @ truth.sum(1)
    feature_sum = multiplicities @ features.sum(1)
    truth_energy = (multiplicities @ truth.square().sum(1) - truth_sum.square() / count).clamp_min(
        0
    )
    feature_energy = (
        multiplicities @ features.square().sum(1) - feature_sum.square() / count
    ).clamp_min(0)
    covariance = multiplicities @ (truth * features).sum(1) - truth_sum * feature_sum / count
    correlation = covariance / (
        truth_energy.sqrt().clamp_min(1e-15) * feature_energy.sqrt().clamp_min(1e-15)
    )
    return correlation.abs().clamp(0, 1).mean(1)


def _condition_report(condition: _Condition, mcc: float) -> dict[str, object]:
    metrics = condition.metrics
    tau, base = metrics.get("geco/tau"), metrics.get("test/base_loss")
    feasible = None if tau is None or base is None else base <= tau
    if "test/feasible" in metrics and feasible != bool(metrics["test/feasible"]):
        raise ValueError("reported GECO feasibility disagrees with configured base-loss bound")
    return {
        "directory": str(condition.directory),
        "run_identity": condition.record["run_identity"],
        "checkpoint": condition.provenance,
        "state_mcc": mcc,
        "state_shd": metrics.get("graph/state_shd"),
        "parameter_routing_shd_descriptive": metrics.get("graph/parameter_shd"),
        "geco_feasible": feasible,
        "geco_tau": tau,
        "base_loss": base,
        "prediction_mse": metrics.get("test/prediction_mse"),
        "parameter_recovery_metrics": {
            key: value for key, value in metrics.items() if key.startswith("recovery/parameter")
        },
        "predictive_use_metrics": {
            key: value
            for key, value in metrics.items()
            if "shuffled_parameter" in key or key.startswith("predictive_use/")
        },
        "parameter_acceptance": (
            "not inferred from coordinate MCC or shuffled-code error; assess all-six "
            "whole-code retention and per-parameter response separately"
        ),
        "calibration_assignment": list(condition.match.columns),
        "assignment_sha256": hashlib.sha256(str(condition.match.columns).encode()).hexdigest(),
        "metrics": metrics,
    }


def paired_mcc_comparison(
    directories: dict[str, Path],
    *,
    fixed_budget: int,
    resamples: int = 2000,
    seed: int = 20261027,
) -> dict[str, object]:
    """Compare final dense/baseline/full MCC with frozen calibration assignments.

    Bootstrap samples draw the same independent parameter groups for every
    condition, retaining all their episodes and frames. No test or bootstrap
    matching is fitted. Intervals are conditional on the single training seed.
    Existing graph, constraint, prediction and parameter metrics remain separate.
    Historical test reuse is explicitly descriptive, never fresh confirmation.
    """
    if set(directories) != {"dense", "baseline", "full"}:
        raise ValueError("comparison requires exactly dense, baseline and full")
    if type(fixed_budget) is not int or fixed_budget < 1:
        raise ValueError("fixed_budget must be a positive integer")
    if type(resamples) is not int or resamples < 2:
        raise ValueError("resamples must be an integer at least two")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("bootstrap seed must be a nonnegative 63-bit integer")
    conditions = {
        name: _load(directories[name], fixed_budget) for name in ("dense", "baseline", "full")
    }
    reference = conditions["dense"]
    for condition in conditions.values():
        for key in ("dataset_identity", "experiment_identity", "seed"):
            expected = reference.record.get(key)
            if expected is None or condition.record.get(key) != expected:
                raise ValueError(f"unmatched {key} in paired MCC comparison")
        if (
            not torch.equal(condition.groups, reference.groups)
            or not torch.equal(condition.calibration_groups, reference.calibration_groups)
            or not torch.equal(condition.states, reference.states)
        ):
            raise ValueError("unmatched held-out groups, state targets or calibration groups")
    count = reference.groups.numel()
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(count, (resamples, count), generator=generator)
    multiplicities = torch.zeros(resamples + 1, count, dtype=torch.float64)
    multiplicities[0] = 1
    multiplicities[1:].scatter_add_(1, indices, torch.ones_like(indices, dtype=torch.float64))
    scores = {name: _scores(condition, multiplicities) for name, condition in conditions.items()}
    contrasts: dict[str, object] = {}
    for first, second in (("dense", "baseline"), ("baseline", "full")):
        difference = scores[second] - scores[first]
        interval = torch.quantile(difference[1:], torch.tensor([0.025, 0.975], dtype=torch.float64))
        lower, upper = float(interval[0]), float(interval[1])
        contrasts[f"{second}_minus_{first}"] = {
            "estimate": float(difference[0]),
            "interval_95": [lower, upper],
            "interval_above_zero": lower > 0,
            "interpretation": "positive effect"
            if lower > 0
            else "observed negative effect"
            if upper < 0
            else "inconclusive",
            "bootstrap_differences": [float(value) for value in difference[1:]],
            "available_mcc_headroom": 1 - float(scores[first][0]),
        }
    shds = [
        conditions[name].metrics.get("graph/state_shd") for name in ("dense", "baseline", "full")
    ]
    guardrail = (
        None
        if any(value is None for value in shds)
        else cast(float, shds[2]) <= cast(float, shds[1]) <= cast(float, shds[0])
    )
    return {
        "format_version": 1,
        "conditions": {
            name: _condition_report(condition, float(scores[name][0]))
            for name, condition in conditions.items()
        },
        "contrasts": contrasts,
        "state_shd_guardrail": guardrail,
        "fixed_budget": fixed_budget,
        "resamples": resamples,
        "resampling_seed": seed,
        "interval_method": (
            "paired percentile bootstrap, 95%; Pearson recomputed with frozen "
            "calibration assignment"
        ),
        "uncertainty_unit": "independent parameter group",
        "group_ids": [int(value) for value in reference.groups],
        "calibration_group_ids": [int(value) for value in reference.calibration_groups],
        "resampled_group_indices": cast(list[list[int]], indices.tolist()),  # pyright: ignore[reportUnknownMemberType]
        "training_seed": reference.record["seed"],
        "interpretation": (
            "held-out-group uncertainty conditional on training seed; does not establish "
            "robustness across training seeds; historical test reuse is descriptive, "
            "not fresh confirmation of an adaptively selected intervention"
        ),
        "scientific_acceptance": (
            "positive intervals are evidence of positive effects; desired gains are around "
            "0.05-0.10 where headroom permits; no 0.01 margin or near-complete cutoff is "
            "adopted; essentially complete recovery in both sparse conditions is a possible "
            "separate outcome, never inferred from similar imperfect scores or inconclusive "
            "contrasts"
        ),
    }
