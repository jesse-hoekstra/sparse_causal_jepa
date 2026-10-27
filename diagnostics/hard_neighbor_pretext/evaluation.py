"""Original single-context retention plus independently formed held-out hard batches."""

import copy
from typing import Any

import torch

from diagnostics.group_contrastive_pretext.evaluation import assess_model as ordinary_assess
from diagnostics.group_contrastive_pretext.evaluation import measurement_metrics as ordinary_metrics
from diagnostics.group_contrastive_pretext.forward import code_statistics, contrastive_loss
from diagnostics.joint_acquisition.baseline import DiagnosticSelection, selected_split
from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.multiquery_control.evaluation import encode_codes
from scjepa.models.model import SCJEPA
from scjepa.training.rng import preserve_rng


def assess_model(
    model: SCJEPA,
    data: EvaluationData,
    selection: DiagnosticSelection,
    *,
    full: bool = False,
    final: bool = False,
) -> dict[str, Any]:
    """Hard-batch metrics use held-out means only to select groups, never infer codes."""
    report = ordinary_assess(model, data, selection, full=full, final=final)
    split = "test" if final else "validation"
    selected = data[split] if full or final else selected_split(data, selection, "validation")
    with preserve_rng(next(model.parameters()).device), torch.no_grad():
        candidate = copy.deepcopy(model).double().eval()
        codes = encode_codes(candidate, selected)
        means = codes.mean(1)
        batch_size = min(16, len(codes))
        rows: list[dict[str, float]] = []
        for anchor in range(len(codes)):
            distances = (means - means[anchor]).square().sum(-1)
            distances[anchor] = torch.inf
            neighbors = torch.argsort(distances, stable=True)[: batch_size - 1]
            indices = [anchor, *(int(value) for value in neighbors)]
            batch = codes[indices]
            rows.append(
                {
                    "contrastive": float(contrastive_loss(batch)),
                    **{key: float(value) for key, value in code_statistics(batch).items()},
                }
            )
        report["hard_pretext"] = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
        report["hard_candidate_groups"] = len(codes)
        report["hard_pretext_protocol"] = (
            "Every heldout group once as anchor plus its15 nearest distinct same-split groupmeans; "
            "current individual-context codes form the original symmetric loss. Groupmeans only "
            "select assessment batches. No TRAIN bank, other split or physical labels used. "
            "Use all available groups when verification has fewer than16. Candidate counts "
            "differ from TRAIN8192, so raw hard-loss gaps do not isolate memorization."
        )
    return report


def measurement_metrics(report: dict[str, Any], *, final: bool = False) -> dict[str, float]:
    """Keep ordinary metrics and separately name held-out hard-batch scores."""
    result = ordinary_metrics(report, final=final)
    prefix = "test" if final else "validation"
    result.update(
        {f"{prefix}/hard_{key}": float(value) for key, value in report["hard_pretext"].items()}
    )
    result[f"{prefix}/hard_candidate_groups"] = float(report["hard_candidate_groups"])
    return result
