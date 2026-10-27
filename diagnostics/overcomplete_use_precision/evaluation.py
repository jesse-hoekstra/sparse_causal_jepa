"""Aggregate once over all independent groups and verify the benefit identity from vectors."""

from collections.abc import Callable
from dataclasses import fields
from typing import Any

import torch
from torch import Tensor

from diagnostics.joint_acquisition.baseline import (
    MatchedBatch,
    _interval,  # pyright: ignore[reportPrivateUsage]
    response_report,
)
from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.oracle_code_control.evaluation import matched_batch
from diagnostics.overcomplete_acquisition.evaluation import predict_matched
from scjepa.models.model import SCJEPA


def direct_decomposition(
    base: Tensor, changed: Tensor, target: Tensor, changed_target: Tensor
) -> dict[str, Tensor]:
    """Compute B=F-2e·l directly, with F=2l·delta-l²; average within each group/factor."""
    base, changed, target, changed_target = (
        x.detach().double().cpu() for x in (base, changed, target, changed_target)
    )
    learned = changed - base[:, :, None]
    exact = changed_target - target[:, :, None]
    residual = base - target
    axes = (1, 3, 4, 5)
    finite_gain = (2 * learned * exact - learned.square()).mean(axes)
    cross = (2 * residual[:, :, None] * learned).mean(axes)
    benefit = (
        (base[:, :, None] - changed_target).square() - (changed - changed_target).square()
    ).mean(axes)
    torch.testing.assert_close(benefit, finite_gain - cross, atol=2e-15, rtol=1e-9)
    return {
        "finite_response_gain": finite_gain,
        "twice_residual_dot_learned_response": cross,
        "benefit_from_direct_vectors": finite_gain - cross,
        "benefit_direct_squared_error": benefit,
    }


def join_batches(parts: list[MatchedBatch]) -> MatchedBatch:
    """Concatenate only group axes, preserving donor/query/physical-factor identities."""
    if not parts:
        raise ValueError("Cannot join empty matched batches")
    first = parts[0]
    for part in parts[1:]:
        if (
            part.donors != first.donors
            or part.split != first.split
            or part.fit_queries != first.fit_queries
            or not torch.equal(part.query_indices, first.query_indices)
        ):
            raise ValueError("Matched geometry differs across chunks")
    values: dict[str, Any] = {}
    for field in fields(MatchedBatch):
        name = field.name
        if name in {"donors", "query_indices", "split", "fit_queries"}:
            values[name] = getattr(first, name)
        elif name == "group_indices":
            values[name] = tuple(index for part in parts for index in part.group_indices)
        elif name == "reconstruction_error":
            values[name] = max(part.reconstruction_error for part in parts)
        else:
            values[name] = torch.cat([getattr(part, name).cpu() for part in parts])
    if len(set(values["group_indices"])) != len(values["group_indices"]):
        raise ValueError("Repeated group in chunk aggregation")
    return MatchedBatch(**values)


@torch.no_grad()
def evaluate(
    model: SCJEPA,
    data: EvaluationData,
    *,
    chunk_size: int,
    bootstrap_samples: int,
    seed: int,
    progress: Callable[[int], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Tensor], dict[str, Tensor]]:
    """Use original positive-step assay and one bootstrap over the entire predeclared sample."""
    groups = len(data["validation"]["theta"])
    if chunk_size < 1 or model.training:
        raise ValueError("Require positive chunks and eval-mode model")
    batches: list[MatchedBatch] = []
    predictions: list[Tensor] = []
    changed: list[Tensor] = []
    for start in range(0, groups, chunk_size):
        indices = tuple(range(start, min(start + chunk_size, groups)))
        batch, _, _ = matched_batch(model, data, "validation", indices)
        result = predict_matched(model, batch)
        batches.append(batch.to(torch.zeros((), dtype=torch.float64)))
        predictions.append(result.predictions.cpu())
        changed.append(result.changed_predictions.cpu())
        if progress:
            progress(min(start + chunk_size, groups))
    batch = join_batches(batches)
    base, counterfactual = torch.cat(predictions), torch.cat(changed)
    report, raw = response_report(
        base, counterfactual, batch, bootstrap_samples=bootstrap_samples, seed=seed
    )
    decomposition = direct_decomposition(base, counterfactual, batch.targets, batch.changed_targets)
    torch.testing.assert_close(
        raw["predictive_benefit"],
        decomposition["benefit_from_direct_vectors"],
        atol=2e-15,
        rtol=1e-9,
    )
    raw.update(decomposition)
    report["direct_vector_identity_max_absolute_error"] = float(
        (raw["predictive_benefit"] - decomposition["benefit_from_direct_vectors"]).abs().max()
    )
    report["decomposition_means"] = {
        key: value.mean(0).numpy().tolist() for key, value in decomposition.items()
    }  # pyright: ignore[reportUnknownMemberType]
    generator = torch.Generator().manual_seed(seed)
    samples = torch.randint(groups, (bootstrap_samples, groups), generator=generator)
    report["decomposition_intervals"] = {
        key: [
            _interval(value[:, factor], torch.ones(groups, dtype=torch.float64), samples)
            for factor in range(6)
        ]
        for key, value in decomposition.items()
    }
    report["group_standard_deviation"] = {
        key: value.std(0, correction=1).numpy().tolist() for key, value in decomposition.items()
    }
    return (
        report,
        raw,
        {
            "factual_predictions": base,
            "counterfactual_predictions": counterfactual,
            "factual_targets": batch.targets,
            "counterfactual_targets": batch.changed_targets,
        },
    )
