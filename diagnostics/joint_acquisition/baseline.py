"""Explicit held-out groups and matched physical responses for joint diagnostics."""

from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import torch
from torch import Tensor

from diagnostics.multiquery_control.data import EvaluationData, TensorSplit
from scjepa.models.model import SCJEPA


@dataclass(frozen=True)
class DiagnosticSelection:
    """Fixed calibration/validation group indices and disjoint recipient episodes."""

    calibration_indices: tuple[int, ...] = tuple(range(64))
    validation_indices: tuple[int, ...] = tuple(range(64))
    donors: tuple[int, ...] = (0,)
    fit_queries: int = 4
    heldout_queries: int = 5


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


def selected_split(
    data: EvaluationData,
    selection: DiagnosticSelection,
    split: Literal["calibration", "validation"],
) -> TensorSplit:
    """Select explicit diagnostic groups without accessing final-test values."""
    if split not in {"calibration", "validation"}:
        raise ValueError("Diagnostic fitting/assessment excludes final test")
    indices = getattr(selection, f"{split}_indices")
    source = cast(TensorSplit, data[split])
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError("Group selections must be nonempty, distinct nonnegative indices")
    if max(indices) >= len(source["theta"]):
        raise ValueError("Selected groups exceed the requested split")
    return {name: values[list(indices)] for name, values in source.items()}


def build_matched_batch(
    model: SCJEPA,
    data: EvaluationData,
    selection: DiagnosticSelection,
    split: Literal["calibration", "validation"],
    *,
    displacement: float = 0.01,
) -> MatchedBatch:
    """Regenerate donor contexts and exact futures with fixed recipient inputs."""
    import math

    from scjepa.data import simplified_gaussian as dgp
    from scjepa.data.controlled_vector import VectorObservationMap

    if model.training or model.context_len != 10 or not model.autoregressive:
        raise ValueError("Use an eval-mode K10 autoregressive model")
    if model.predictor.history_len != 1 or not math.isfinite(displacement) or displacement <= 0:
        raise ValueError("Require L1 and a positive finite physical perturbation")
    calibration = selected_split(data, selection, "calibration")
    validation = selected_split(data, selection, "validation")
    if bool((calibration["theta"][:, None] == validation["theta"][None]).all(-1).any()):
        raise ValueError("Calibration and validation must have independent parameter groups")
    selected = calibration if split == "calibration" else validation
    if split not in {"calibration", "validation"}:
        raise ValueError("Diagnostic fitting/assessment excludes final test")
    reference = next(model.parameters())
    theta, states, actions = (
        selected[key].to(reference) for key in ("theta", "true_states", "true_actions")
    )
    groups, views = states.shape[:2]
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
        raise ValueError("Use distinct R10 donors and disjoint non-self fit/heldout queries")
    queries = torch.tensor(
        [
            [
                (donor + offset) % views
                for offset in range(1, selection.fit_queries + selection.heldout_queries + 1)
            ]
            for donor in donors
        ],
        device=reference.device,
    )
    remaining = dgp.PARAMETER_BOUND - theta.abs()
    if not bool((remaining > 0).all()):
        raise ValueError("Parameters must be strictly inside their support")
    steps = remaining.mul(0.5).clamp(max=displacement)
    changed_theta = theta[:, None, :] + torch.diag_embed(steps)
    observation = VectorObservationMap(matrix=data["metadata"]["observation_matrix"], dimension=6)
    donor_actions = actions[:, donors, :9]
    initial = states[:, donors, 0]
    with torch.no_grad():
        regenerated = dgp.simulate(initial, theta[:, None], donor_actions)
        donor_states = model.online_encoder.encode_sequence(
            observation.forward(regenerated).flatten(0, 1)
        ).reshape(groups, len(donors), 10, 6)
        physical_changed = dgp.simulate(
            initial[:, :, None].expand(-1, -1, 6, -1),
            changed_theta[:, None],
            donor_actions[:, :, None].expand(-1, -1, 6, -1, -1),
        )
        changed_donor_states = model.online_encoder.encode_sequence(
            observation.forward(physical_changed).flatten(0, 2)
        ).reshape(groups, len(donors), 6, 10, 6)
        query_states = states[:, queries, 9]
        query_actions = actions[:, queries, 9:11]
        history = model.online_encoder.encode_sequence(
            observation.forward(query_states).reshape(-1, 1, 6)
        ).reshape(groups, len(donors), queries.shape[1], 1, 6)
        exact_states = dgp.simulate(query_states, theta[:, None, None], query_actions)[..., 1:, :]
        targets = model.target_encoder.encode_sequence(
            observation.forward(exact_states).flatten(0, 2)
        ).reshape(groups, len(donors), queries.shape[1], 2, 6)
        changed_states = dgp.simulate(
            query_states[:, :, None].expand(-1, -1, 6, -1, -1),
            changed_theta[:, None, :, None],
            query_actions[:, :, None].expand(-1, -1, 6, -1, -1, -1),
        )[..., 1:, :]
        changed_targets = model.target_encoder.encode_sequence(
            observation.forward(changed_states).flatten(0, 3)
        ).reshape(groups, len(donors), 6, queries.shape[1], 2, 6)
    errors = [
        (regenerated - states[:, donors, :10]).abs().max(),
        (donor_states - selected["states_online"][:, donors, :10].to(reference)).abs().max(),
        (history - selected["states_online"][:, queries, 9:10].to(reference)).abs().max(),
        (targets - selected["states_target"][:, queries, 10:12].to(reference)).abs().max(),
        (actions - selected["actions"].to(reference)).abs().max(),
    ]
    reconstruction_error = max(float(error) for error in errors)
    if not math.isfinite(reconstruction_error) or reconstruction_error > 2e-5:
        raise ValueError(
            f"Reconstructed DGP/state maps differ from cached inputs: {reconstruction_error}"
        )
    return MatchedBatch(
        donor_states,
        donor_actions,
        changed_donor_states,
        history,
        query_actions,
        targets,
        changed_targets,
        steps,
        getattr(selection, f"{split}_indices"),
        donors,
        queries.detach().cpu(),
        split,
        selection.fit_queries,
        reconstruction_error,
    )


def predict_matched(model: SCJEPA, batch: MatchedBatch) -> MatchedPredictions:
    """Infer full context codes and retain both attached autoregressive steps."""
    from diagnostics.signal_path.geometry import predict_rollout

    batch = batch.to(next(model.parameters()))
    groups, donors, queries = batch.history.shape[:3]
    codes = cast(
        Tensor,
        model.parameter_encoder(
            batch.donor_states.flatten(0, 1), batch.donor_actions.flatten(0, 1)
        ),
    ).reshape(groups, donors, 6)
    changed_codes = cast(
        Tensor,
        model.parameter_encoder(
            batch.changed_donor_states.flatten(0, 2),
            batch.donor_actions[:, :, None].expand(-1, -1, 6, -1, -1).flatten(0, 2),
        ),
    ).reshape(groups, donors, 6, 6)
    predictions = predict_rollout(
        model,
        batch.history.flatten(0, 2),
        codes[:, :, None].expand(-1, -1, queries, -1).flatten(0, 2),
        batch.query_actions.flatten(0, 2),
    ).reshape(groups, donors, queries, 2, 6)
    changed_predictions = predict_rollout(
        model,
        batch.history[:, :, None].expand(-1, -1, 6, -1, -1, -1).flatten(0, 3),
        changed_codes[:, :, :, None].expand(-1, -1, -1, queries, -1).flatten(0, 3),
        batch.query_actions[:, :, None].expand(-1, -1, 6, -1, -1, -1).flatten(0, 3),
    ).reshape(groups, donors, 6, queries, 2, 6)
    return MatchedPredictions(predictions, changed_predictions, codes, changed_codes)


def response_loss(model: SCJEPA, batch: MatchedBatch) -> Tensor:
    """Diagnostic supervision: mean factor-wise finite-response normalized error."""
    output = predict_matched(model, batch)
    expected = (batch.changed_targets - batch.targets[:, :, None]).to(output.predictions)
    actual = output.changed_predictions - output.predictions[:, :, None]
    energy = expected.square().sum((0, 1, 3, 4, 5))
    if not bool(torch.isfinite(energy).all()) or not bool((energy > 0).all()):
        raise ValueError("Undefined diagnostic response normalization")
    return ((actual - expected).square().sum((0, 1, 3, 4, 5)) / energy).mean()


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


@torch.no_grad()
def evaluate_responses(
    model: SCJEPA, batch: MatchedBatch, *, bootstrap_samples: int = 1000, seed: int = 2026102701
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Evaluate all factors with unchanged recipient inputs and matched donor contexts."""
    output = predict_matched(model, batch)
    return response_report(
        output.predictions,
        output.changed_predictions,
        batch,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


@torch.no_grad()
def validation_retention(
    model: SCJEPA, data: EvaluationData, selection: DiagnosticSelection
) -> dict[str, Any]:
    """Reuse whole-code probe families while reporting diagnostic validation only.

    The historical evaluator requires three split slots. Its scoring slot is
    deliberately supplied the same selected validation data, never final-test
    data. All outputs are relabelled, and their selection optimism is explicit.
    """
    from diagnostics.multiquery_control.evaluation import encode_codes, parameter_readouts

    calibration = selected_split(data, selection, "calibration")
    validation = selected_split(data, selection, "validation")
    cal_codes = encode_codes(model, calibration)
    val_codes = encode_codes(model, validation)
    proxy: EvaluationData = {
        "calibration": calibration,
        "validation": validation,
        "test": validation,
    }
    result = parameter_readouts(
        {"calibration": cal_codes, "validation": val_codes, "test": val_codes}, proxy
    )
    families: dict[str, Any] = {}
    for name, family in result["families"].items():
        families[name] = {
            "validation_r2": family["validation_r2"],
            "selected_hyperparameters_per_parameter": family[
                "selected_hyperparameters_per_parameter"
            ],
            "validation_group_mse_by_parameter": family["test_group_mse_by_parameter"],
        }
    return {
        "families": families,
        "selected_methods_per_parameter": result["selected_methods_per_parameter"],
        "selected_validation_r2": result["selected_test_r2"],
        "validation_group_mse_by_parameter": result["selected_test_group_mse_by_parameter"],
        "calibration_code_covariance_eigenvalues": result[
            "calibration_code_covariance_eigenvalues"
        ],
        "calibration_code_std": float(
            cal_codes.double().flatten(0, 1).var(0, correction=0).mean().sqrt()
        ),
        "contexts_per_group": int(cal_codes.shape[1]),
        "protocol": (
            "Fit full individual-context code probes on selected calibration groups; select and "
            "score on selected validation groups. No context averaging. Selection-optimistic "
            "diagnostic scores, not independent final test evidence."
        ),
    }


@torch.no_grad()
def compare_compatible_codes(
    source_model: SCJEPA,
    current_model: SCJEPA,
    data: EvaluationData,
    selection: DiagnosticSelection,
    *,
    displacement: float = 0.01,
    bootstrap_samples: int = 1000,
    seed: int = 2026102701,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Compare actual encoders with one identical predictor and identical recipient queries."""
    from diagnostics.frozen_predictor_control.source import frozen_module_hashes

    if frozen_module_hashes(source_model) != frozen_module_hashes(current_model):
        raise ValueError("Compatible code substitution requires identical frozen modules")
    if source_model.get_extra_state() != current_model.get_extra_state():
        raise ValueError("Compatible code substitution requires identical prediction settings")
    batch = build_matched_batch(
        source_model, data, selection, "validation", displacement=displacement
    ).select_queries("heldout")
    source, source_raw = evaluate_responses(
        source_model, batch, bootstrap_samples=bootstrap_samples, seed=seed
    )
    current, current_raw = evaluate_responses(
        current_model, batch, bootstrap_samples=bootstrap_samples, seed=seed
    )
    raw = {f"source/{key}": value for key, value in source_raw.items()}
    raw.update({f"current/{key}": value for key, value in current_raw.items()})
    difference = source_raw["predictive_benefit"] - current_raw["predictive_benefit"]
    raw["source_minus_current_predictive_benefit"] = difference
    raw["source_minus_current_ordinary_mse_by_horizon"] = (
        source_raw["ordinary_mse_by_horizon"] - current_raw["ordinary_mse_by_horizon"]
    )
    generator = torch.Generator().manual_seed(seed)
    samples = torch.randint(
        len(difference), (bootstrap_samples, len(difference)), generator=generator
    )
    paired: list[dict[str, Any]] = []
    for factor in range(6):
        paired.append(
            {
                "parameter": factor + 1,
                "source_minus_current_predictive_benefit": _interval(
                    difference[:, factor], torch.ones(len(difference), dtype=torch.float64), samples
                ),
                "source_minus_current_normalized_response_error": _interval(
                    source_raw["response_error_energy"][:, factor]
                    - current_raw["response_error_energy"][:, factor],
                    source_raw["exact_response_energy"][:, factor],
                    samples,
                ),
            }
        )
    return {
        "source": source,
        "current": current,
        "paired_source_minus_current_ordinary_mse_by_horizon": [
            float(value) for value in raw["source_minus_current_ordinary_mse_by_horizon"].mean(0)
        ],
        "paired_per_parameter": paired,
        "interpretation": (
            "The source encoder is an imperfect comparator, not an exact oracle. Better "
            "source-code performance indicates a code-dependent limitation without uniquely "
            "blaming the encoder. Poor performance in both keeps predictor adequacy open. These "
            "codes must not be substituted into the unrelated primary joint predictor."
        ),
    }, raw


def numerical_checks(
    model: SCJEPA, data: EvaluationData, selection: DiagnosticSelection
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Validate chain-rule and finite-response calculations on one disposable copy."""
    import copy
    import math

    from diagnostics.signal_path.geometry import geometry_diagnostic

    summary, raw = geometry_diagnostic(
        copy.deepcopy(model).double().eval().requires_grad_(False),
        data,
        groups=1,
        donor_contexts=1,
        fit_queries=selection.fit_queries,
        test_queries=selection.heldout_queries,
        split_name="validation",
        group_indices=selection.validation_indices[:1],
        calibration_indices=selection.calibration_indices,
    )
    tolerances = {
        "chain_rule_relative_error": 1e-6,
        "encoder_finite_difference_relative_error": 1e-3,
        "composed_finite_difference_relative_error": 1e-3,
        "target_finite_difference_relative_error": 1e-3,
        "reconstructed_vs_cached_code_rms": 2e-5,
    }
    passed = all(
        math.isfinite(summary["checks"][name]) and summary["checks"][name] <= tolerance
        for name, tolerance in tolerances.items()
    )
    return {
        "passed": passed,
        "tolerances": tolerances,
        "geometry": summary,
        "scope": (
            "First selected validation group, donor episode 0; numerical validation only. "
            "Thresholds fixed before inspecting scientific validation outcomes."
        ),
    }, raw
