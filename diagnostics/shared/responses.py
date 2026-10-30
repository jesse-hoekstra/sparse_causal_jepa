"""Matched physical-response tensors and group-bootstrap summaries."""

from dataclasses import dataclass, fields, replace
from typing import Any, Literal

import torch
from torch import Tensor

from scjepa.models.model import SCJEPA


@dataclass(frozen=True)
class MatchedBatch:
    """Matched tensors; G groups, D donors, F=6 factors, Q queries, H=2 horizons.

    donor_states/actions: G,D,K,S / G,D,K-1,A.
    changed_donor_states: G,D,F,K,S; actions remain fixed.
    history/actions: G,D,Q,1,S / G,D,Q,H,A.
    targets/changed_targets: G,D,Q,H,S / G,D,F,Q,H,S.
    finite_steps: G,F. query_indices: D,Q. Group IDs include split identity.
    """

    donor_states: Tensor
    donor_actions: Tensor
    changed_donor_states: Tensor
    history: Tensor
    query_actions: Tensor
    targets: Tensor
    changed_targets: Tensor
    finite_steps: Tensor
    group_indices: tuple[int, ...]
    donors: tuple[int, ...]
    query_indices: Tensor
    split: str
    fit_queries: int
    reconstruction_error: float

    def to(self, reference: Tensor) -> "MatchedBatch":
        """Copy floating tensors to a model's device and dtype without changing IDs."""
        return replace(
            self,
            **{
                name: getattr(self, name).to(reference)
                for name in (
                    "donor_states",
                    "donor_actions",
                    "changed_donor_states",
                    "history",
                    "query_actions",
                    "targets",
                    "changed_targets",
                    "finite_steps",
                )
            },
        )

    def select_queries(self, kind: Literal["fit", "heldout"]) -> "MatchedBatch":
        """Return disjoint support or assessment recipients, preserving donor context."""
        chosen = slice(0, self.fit_queries) if kind == "fit" else slice(self.fit_queries, None)
        if kind not in {"fit", "heldout"}:
            raise ValueError("Select fit or heldout queries")
        if not self.fit_queries or self.fit_queries >= self.history.shape[2]:
            raise ValueError("Query partition must contain both fit and heldout episodes")
        return replace(
            self,
            history=self.history[:, :, chosen],
            query_actions=self.query_actions[:, :, chosen],
            targets=self.targets[:, :, chosen],
            changed_targets=self.changed_targets[:, :, :, chosen],
            query_indices=self.query_indices[:, chosen],
            fit_queries=0,
        )


@dataclass(frozen=True)
class MatchedPredictions:
    """Attached predictions and individual context codes with the batch axes."""

    predictions: Tensor
    changed_predictions: Tensor
    codes: Tensor
    changed_codes: Tensor


def _interval(numerator: Tensor, denominator: Tensor, samples: Tensor) -> dict[str, Any]:
    valid = bool(torch.isfinite(numerator).all() and torch.isfinite(denominator).all())
    valid = valid and float(denominator.sum()) > 0
    if not valid:
        return {"estimate": None, "interval_95": None, "defined": False}
    totals = denominator[samples].sum(1)
    if not bool((totals > 0).all()):
        return {
            "estimate": float(numerator.sum() / denominator.sum()),
            "interval_95": None,
            "defined": False,
        }
    draws = numerator[samples].sum(1) / totals
    bounds = torch.quantile(draws, torch.tensor([0.025, 0.975], dtype=draws.dtype))
    return {
        "estimate": float(numerator.sum() / denominator.sum()),
        "interval_95": [float(value) for value in bounds],
        "defined": True,
    }


@torch.no_grad()
def response_report(
    predictions: Tensor,
    changed_predictions: Tensor,
    batch: MatchedBatch,
    *,
    bootstrap_samples: int = 1000,
    seed: int = 2026102701,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Report finite-change units with paired independent-group uncertainty."""
    if (
        predictions.shape != batch.targets.shape
        or changed_predictions.shape != batch.changed_targets.shape
    ):
        raise ValueError("Forecast tensors must match the explicit group/donor/factor/query axes")
    if bootstrap_samples < 2 or seed < 0:
        raise ValueError("Use at least two group-bootstrap samples and a nonnegative seed")
    base, changed, target, changed_target = (
        value.detach().double().cpu()
        for value in (predictions, changed_predictions, batch.targets, batch.changed_targets)
    )
    actual, exact = changed - base[:, :, None], changed_target - target[:, :, None]
    axes = (1, 3, 4, 5)
    error = (actual - exact).square().sum(axes)
    reference = exact.square().sum(axes)
    energy = actual.square().sum(axes)
    benefit = (
        (base[:, :, None] - changed_target).square() - (changed - changed_target).square()
    ).mean(axes)
    generator = torch.Generator().manual_seed(seed)
    samples = torch.randint(len(base), (bootstrap_samples, len(base)), generator=generator)
    rows: list[dict[str, Any]] = []
    for factor in range(6):
        rows.append(
            {
                "parameter": factor + 1,
                "normalized_response_error": _interval(
                    error[:, factor], reference[:, factor], samples
                ),
                "response_energy_ratio": _interval(
                    energy[:, factor], reference[:, factor], samples
                ),
                "predictive_benefit": _interval(
                    benefit[:, factor], torch.ones(len(base), dtype=torch.float64), samples
                ),
                "normalized_predictive_benefit": _interval(
                    benefit[:, factor] * actual[0, :, factor].numel(), reference[:, factor], samples
                ),
                "exact_response_energy": float(reference[:, factor].sum()),
                "learned_response_energy": float(energy[:, factor].sum()),
            }
        )
    ordinary = (base - target).square().mean((0, 1, 2, 4))
    variance = target.permute(3, 0, 1, 2, 4).flatten(1).var(1, correction=0)
    raw = {
        "response_error_energy": error,
        "exact_response_energy": reference,
        "learned_response_energy": energy,
        "predictive_benefit": benefit,
        "ordinary_mse_by_horizon": (base - target).square().mean((1, 2, 4)),
        "counterfactual_mse_by_horizon": (changed - changed_target).square().mean((1, 3, 5)),
        "finite_steps": batch.finite_steps.detach().cpu(),
        "group_indices": torch.tensor(batch.group_indices),
        "query_indices": batch.query_indices,
    }
    summary: dict[str, Any] = {
        "split": batch.split,
        "group_indices": list(batch.group_indices),
        "donors": list(batch.donors),
        "query_indices": [[int(value) for value in row] for row in batch.query_indices],
        "ordinary_mse_by_horizon": [float(value) for value in ordinary],
        "ordinary_variance_normalized_mse_by_horizon": [
            float(e / v) if float(v) > 0 else None for e, v in zip(ordinary, variance, strict=True)
        ],
        "per_parameter": rows,
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "unit": "independent parameter group",
            "confidence": 0.95,
        },
        "normalization": (
            "Ratio of aggregate squared finite response error to aggregate exact finite "
            "response energy; zero response scores one."
        ),
        "reconstruction_max_absolute_error": batch.reconstruction_error,
        "interpretation": (
            "Validation diagnostic conditional on this checkpoint/training seed; not final test "
            "evidence or proof of global incapacity."
        ),
    }
    return summary, raw


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


def predict_rollout(model: SCJEPA, history: Tensor, code: Tensor, actions: Tensor) -> Tensor:
    """Predict attached autoregressive steps from (N,1,6), (N,6), (N,H,16)."""
    predictions: list[Tensor] = []
    for step in range(actions.shape[1]):
        prediction = model.predictor(history, code, actions[:, step : step + 1]).prediction
        predictions.append(prediction)
        history = prediction[:, None]
    return torch.stack(predictions, dim=1)
