"""Original six-code responses, aggregated once over every fixed validation group."""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from diagnostics.joint_acquisition.baseline import (
    MatchedBatch,
    _interval,  # pyright: ignore[reportPrivateUsage]
    predict_matched,
    response_report,
)
from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.oracle_code_control.evaluation import matched_batch
from diagnostics.overcomplete_use_precision.evaluation import direct_decomposition, join_batches
from scjepa.models.model import SCJEPA


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
    if model.parameter_encoder.param_dim != 6 or model.predictor.param_dim != 6:
        raise ValueError("Require six-code transfer/reference models")
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
