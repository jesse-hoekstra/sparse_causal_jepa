"""Calibration-frozen MCC and graph SHD for registered vector DGP experiments.

MCC uses absolute Pearson correlation and a Hungarian assignment, following
Lachapelle et al.'s mechanism-sparsity implementation. We fit that assignment on
calibration episodes and score independent test episodes. Graph SHD counts false
positives plus false negatives after applying those same assignments. Local
SPARTAN graphs describe residual-inclusive value-routing support, conditional on
learned codes or explicitly provided oracle parameters, rather than an identified
causal graph.
"""

import json
import math
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.simplified_gaussian import PARAMETER_BOUND
from scjepa.data.types import DGPConfig, GraphMasks
from scjepa.eval.recovery import (
    CoordinateMatch,
    fit_coordinate_match,
    mean_correlation_coefficient,
)
from scjepa.eval.state_probes import fit_state_probe, state_probe_metrics
from scjepa.models.model import SCJEPA, ModelOutput, cross_episode_parameters
from scjepa.training.loop import ObjectiveConfig, TrainingObjective
from scjepa.training.rng import preserve_rng


@dataclass(frozen=True)
class EvaluationConfig:
    """Held-out independent-group counts and predeclared routing aggregation.

    Each grouped item/batch unit contains R independent trajectories sharing one
    parameter. Every recipient uses one other context's code, with equal weight.
    """

    calibration_episodes: int = 512
    validation_episodes: int = 256
    test_episodes: int = 512
    batch_size: int = 64
    seed: int = 42
    graph_support_threshold: float = 0.0

    def __post_init__(self) -> None:
        """Require defined correlation/SIGReg statistics and a finite threshold."""
        for name in ("calibration_episodes", "validation_episodes", "test_episodes", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 2:
                raise ValueError(f"{name} must be an integer at least two")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("evaluation seed must be a nonnegative 63-bit integer")
        if (
            not math.isfinite(self.graph_support_threshold)
            or not 0 <= self.graph_support_threshold < 1
        ):
            raise ValueError("graph_support_threshold must lie in [0, 1)")


@dataclass(frozen=True)
class _EncodedSplit:
    config: DGPConfig
    reference_graphs: GraphMasks
    ids: Tensor
    states: Tensor
    parameters: Tensor
    online: Tensor
    target: Tensor
    target_states: Tensor
    codes: Tensor
    graph_frequencies: dict[str, Tensor]
    prediction_mse: float
    base_loss: float
    sparsity: float
    shuffled_prediction_mse: float | None = None


@contextmanager
def _evaluation_context(model: SCJEPA, seed: int = 42) -> Generator[None]:
    """Isolate random projections and restore module modes even after an error."""
    modes = [(module, module.training) for module in model.modules()]
    device = next(model.parameters()).device
    try:
        with preserve_rng(device):
            torch.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            model.eval()
            yield
    finally:
        for module, training in modes:
            module.training = training


def _check_model(model: SCJEPA, data: DGPConfig) -> None:
    if data.family not in {"controlled_vector", "independent_gaussian", "simplified_gaussian"}:
        raise ValueError(
            "this experiment evaluates controlled_vector, independent_gaussian "
            "or simplified_gaussian data"
        )
    specification = build_dgp_dataset(data).specification
    dimensions = tuple(specification[name] for name in ("state_dim", "param_dim", "action_dim"))
    predictor = model.predictor
    if (
        model.online_encoder.observation_ndim != 1
        or (predictor.state_dim, predictor.param_dim, predictor.action_dim) != dimensions
        or predictor.history_len != 1
        or model.context_len < 2
    ):
        label = "/".join(str(value) for value in dimensions)
        raise ValueError(f"vector experiment requires dimensions {label}, L=1, K>=2")
    if data.trajectory_length <= model.context_len:
        raise ValueError("trajectory_length must include a target after the context")
    minimum_context = cast(int, specification["minimum_context_len"])
    if model.context_len < minimum_context:
        raise ValueError(f"{data.family} evaluation requires K>={minimum_context}")
    if data.family == "simplified_gaussian" and data.episodes_per_parameter < 2:
        raise ValueError("simplified_gaussian evaluation requires at least two episodes per theta")


def _batch_sizes(count: int, batch_size: int) -> list[int]:
    sizes = [min(batch_size, count - begin) for begin in range(0, count, batch_size)]
    if len(sizes) > 1 and sizes[-1] == 1:
        tail = sizes.pop()
        sizes[-1] += tail
    return sizes


@torch.no_grad()
def _shuffled_code_prediction_mse(model: SCJEPA, result: ModelOutput, actions: Tensor) -> float:
    """Derange parameter groups and recompute the complete autoregressive predictor.

    A cyclic group shift ensures every donor has a different theta; the view
    shift retains the normal single-context donor convention. In oracle mode,
    every view already contains the group's provided raw theta, so this shifts
    true parameters between groups. Gates and attention are recomputed from the
    modified inputs at every prediction step.
    """
    if result.parameters.ndim != 3 or result.parameters.shape[0] < 2 or not model.autoregressive:
        raise ValueError("code shuffling requires at least two parameter groups and autoregression")
    start = result.prediction_start
    first = start - model.predictor.history_len + 1
    codes = cross_episode_parameters(result.parameters.roll(1, dims=0)).flatten(0, 1)
    history = result.online_states[:, :, first : start + 1].flatten(0, 1)
    predictions = model.rollout(
        history,
        codes,
        result.targets.shape[-2],
        actions[:, :, first:].flatten(0, 1),
    )
    return float((predictions - result.targets.flatten(0, 1)).square().mean())


@torch.no_grad()
def _collect(
    model: SCJEPA,
    base: DGPConfig,
    matrix: Tensor,
    config: EvaluationConfig,
    split: str,
    count: int,
    objective: ObjectiveConfig | None = None,
    *,
    isolate_gate_rng: bool = False,
    shuffle_parameters: bool = False,
) -> _EncodedSplit:
    data = replace(base, split=split, num_episodes=count)
    process = build_dgp_dataset(data, observation_matrix=matrix).process
    reference = next(model.online_encoder.parameters())
    criterion = TrainingObjective(objective or ObjectiveConfig()).to(reference)
    buckets: dict[str, list[Tensor]] = {
        key: []
        for key in ("ids", "states", "parameters", "online", "target", "target_states", "codes")
    }
    graph_counts: dict[str, Tensor] = {}
    transition_count = 0
    prediction_sum = base_sum = sparsity_sum = shuffled_sum = 0.0
    begin = 0
    for batch_index, size in enumerate(_batch_sizes(count, config.batch_size)):
        episodes = [process.generate(index) for index in range(begin, begin + size)]
        begin += size
        observations = torch.stack([item.observations for item in episodes]).to(reference)
        actions = torch.stack([item.actions for item in episodes]).to(reference)
        parameters = torch.stack([item.ground_truth["parameters"] for item in episodes])
        forward_arguments = (
            {"oracle_parameters": parameters.to(reference)}
            if model.parameter_source == "oracle"
            else {}
        )
        if isolate_gate_rng:
            # Gate draws must not change the SIGReg projection sequence below.
            with preserve_rng(reference.device):
                gate_seed = (config.seed + 1_000_003 + batch_index) % (2**63)
                torch.default_generator.manual_seed(gate_seed)
                if reference.device.type == "cuda":
                    with torch.cuda.device(reference.device):
                        torch.cuda.manual_seed(gate_seed)
                result = model(observations, actions, **forward_arguments)
        else:
            result = model(observations, actions, **forward_arguments)
        _, terms = criterion.forward(result)
        prediction_sum += float(terms["prediction"]) * size
        base_sum += float(terms["base"]) * size
        sparsity_sum += float(terms["sparsity"]) * size
        if shuffle_parameters:
            shuffled_sum += _shuffled_code_prediction_mse(model, result, actions) * size
        ids = torch.tensor([item.episode_id for item in episodes], dtype=torch.int64)
        states = torch.stack([item.ground_truth["states"] for item in episodes])
        online, target, codes = (
            result.online_states.cpu(),
            result.targets.cpu(),
            result.parameters.cpu(),
        )
        if data.episodes_per_parameter >= 2:
            # Recovery scores each context code against its shared theta. Only
            # diagnostics flatten views; the objective above keeps groups independent.
            ids = ids.repeat_interleave(data.episodes_per_parameter)
            states, online, target, codes = (
                value.flatten(0, 1) for value in (states, online, target, codes)
            )
            parameters = parameters.repeat_interleave(data.episodes_per_parameter, dim=0)
        buckets["ids"].append(ids)
        buckets["states"].append(states)
        buckets["parameters"].append(parameters)
        buckets["online"].append(online)
        buckets["target"].append(target)
        buckets["target_states"].append(states[:, result.prediction_start + 1 :])
        buckets["codes"].append(codes)
        for dynamics in result.dynamics:
            for name in ("state", "action", "parameter"):
                paths = cast(Tensor, getattr(dynamics.graphs, name))
                counts = (paths > 0).double().sum(0).cpu()
                graph_counts[name] = graph_counts.get(name, torch.zeros_like(counts)) + counts
            transition_count += size * data.episodes_per_parameter
    values = {key: torch.cat(items) for key, items in buckets.items()}
    if any(not bool(torch.isfinite(value).all()) for value in values.values()):
        raise FloatingPointError("nonfinite evaluation encodings")
    return _EncodedSplit(
        data,
        reference_graphs=process.graphs(),
        **values,
        graph_frequencies={key: value / transition_count for key, value in graph_counts.items()},
        prediction_mse=prediction_sum / count,
        base_loss=base_sum / count,
        sparsity=sparsity_sum / count,
        shuffled_prediction_mse=shuffled_sum / count if shuffle_parameters else None,
    )


def _loss_metrics(split: _EncodedSplit, prefix: str) -> dict[str, float]:
    metrics = {
        f"{prefix}/prediction_mse": split.prediction_mse,
        f"{prefix}/base_loss": split.base_loss,
        f"{prefix}/sparsity": split.sparsity,
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("nonfinite evaluation loss")
    return metrics


def validate_model(
    model: SCJEPA,
    base_data: DGPConfig,
    observation_matrix: Tensor,
    config: EvaluationConfig,
    *,
    objective: ObjectiveConfig | None = None,
) -> dict[str, float]:
    """Score validation loss and calibration-aligned MCC/SHD without test data.

    Base loss includes MSE, weighted state/parameter SIGReg, parameter consistency
    and logit regularization, excluding structural sparsity. Projections and evaluation
    batching are fixed by ``config`` and do not advance the training RNG.
    Label-based recovery diagnostics do not enter the loss or checkpoint selector.
    """
    _check_model(model, base_data)
    with _evaluation_context(model, config.seed):
        calibration = _collect(
            model,
            base_data,
            observation_matrix,
            config,
            "evaluation_calibration",
            config.calibration_episodes,
        )
        with _evaluation_context(model, config.seed):
            split = _collect(
                model,
                base_data,
                observation_matrix,
                config,
                "validation",
                config.validation_episodes,
                objective,
            )
        diagnostics, _, _ = _recovery_metrics(calibration, split, config.graph_support_threshold)
    # Keep direction/horizon means separate; use all validation groups, not batch variances.
    targets = split.target.double().reshape(
        -1, split.config.episodes_per_parameter, *split.target.shape[1:]
    )
    target_variance = float(targets.var(dim=0, correction=0).mean())
    return {
        **_loss_metrics(split, "validation"),
        "validation/target_variance": target_variance,
        "validation/normalized_prediction_mse": split.prediction_mse / max(target_variance, 1e-12),
        **{f"validation/{name}": value for name, value in diagnostics.items()},
    }


def calibrate_tau(
    model: SCJEPA,
    base_data: DGPConfig,
    observation_matrix: Tensor,
    config: EvaluationConfig,
    *,
    objective: ObjectiveConfig | None = None,
    training_gates: bool = False,
) -> dict[str, float]:
    """Measure a selected dense model on the independent calibration split.

    No evaluation labels or coordinate matching enter this loss, except for the
    raw theta input explicitly provided by the oracle-parameter control. The
    caller persists the measured base loss as the shared GECO threshold for both
    sparse conditions.
    ``training_gates`` samples predictor gates for a startup loss comparison while
    keeping encoders in evaluation mode and retaining the same SIGReg projections.
    """
    _check_model(model, base_data)
    with _evaluation_context(model, config.seed):
        if training_gates:
            model.predictor.train()
        split = _collect(
            model,
            base_data,
            observation_matrix,
            config,
            "evaluation_calibration",
            config.calibration_episodes,
            objective,
            isolate_gate_rng=training_gates,
        )
    return _loss_metrics(split, "calibration")


def _aligned_support(
    frequencies: dict[str, Tensor],
    state_match: CoordinateMatch,
    parameter_match: CoordinateMatch,
    threshold: float,
    *,
    target_match: CoordinateMatch | None = None,
) -> dict[str, Tensor]:
    """Align EMA output rows separately from online-state and parameter inputs."""
    state_columns, parameter_columns = list(state_match.columns), list(parameter_match.columns)
    output_rows = list((target_match or state_match).columns)
    return {
        "state": (frequencies["state"][output_rows][:, :, state_columns] > threshold),
        "action": frequencies["action"][output_rows] > threshold,
        "parameter": frequencies["parameter"][output_rows][:, parameter_columns] > threshold,
    }


def _gaussianized_parameters(parameters: Tensor) -> Tensor:
    """Map the known uniform reference distribution to coordinatewise N(0,1)."""
    probability = (parameters.double() / PARAMETER_BOUND + 1) / 2
    epsilon = torch.finfo(torch.float64).eps
    return math.sqrt(2) * torch.erfinv(2 * probability.clamp(epsilon, 1 - epsilon) - 1)


def _group_linear_r2(
    calibration_features: Tensor,
    calibration_truth: Tensor,
    scored_features: Tensor,
    scored_truth: Tensor,
    groups: tuple[tuple[int, ...], ...],
    match: CoordinateMatch,
) -> float:
    """Average held-out linear R² across calibration-assigned coordinate groups."""
    scores: list[float] = []
    for group in groups:
        columns = [match.columns[index] for index in group]
        truth_columns = list(group)
        probe = fit_state_probe(
            calibration_features[:, columns], calibration_truth[:, truth_columns], ridge=1e-3
        )
        scores.append(
            state_probe_metrics(
                probe.predict(scored_features[:, columns]), scored_truth[:, truth_columns]
            )["r2"]
        )
    return sum(scores) / len(scores)


def _paired_code_variance(codes: Tensor, episodes_per_parameter: int = 2) -> dict[str, float]:
    """Score individual context codes, using every available view per test theta."""
    if (
        codes.ndim != 2
        or episodes_per_parameter < 2
        or codes.shape[0] % episodes_per_parameter
        or codes.shape[0] < 2 * episodes_per_parameter
    ):
        raise ValueError(
            "code variance requires complete groups with at least two views and groups"
        )
    paired = codes.double().reshape(-1, episodes_per_parameter, codes.shape[-1])
    # Retain the exact legacy two-view calculation; for R>2 this is the same
    # unbiased within-group variance, not the variance of an averaged code.
    within = (
        float((paired[:, 0] - paired[:, 1]).square().mean() / 2)
        if episodes_per_parameter == 2
        else float(paired.var(dim=1, correction=1).mean())
    )
    # Estimate marginal variance across independent groups separately for each view.
    total = float(paired.var(dim=0, correction=1).mean())
    return {
        "parameter_within_theta_variance": within,
        "parameter_total_variance": total,
        "parameter_within_theta_variance_ratio": within / max(total, 1e-12),
    }


def _simplified_recovery_metrics(
    calibration: _EncodedSplit,
    scored: _EncodedSplit,
    matches: dict[str, CoordinateMatch],
) -> dict[str, float]:
    """Add inexpensive group and context diagnostics for the approved new family."""
    width = scored.states.shape[-1] // 2
    rows = (tuple(range(width)), tuple(range(width, 2 * width)))
    pairs = tuple((index, index + width) for index in range(width))
    gaussian_calibration = _gaussianized_parameters(calibration.parameters)
    gaussian_scored = _gaussianized_parameters(scored.parameters)
    matches["parameter_gaussianized"] = fit_coordinate_match(
        calibration.codes, gaussian_calibration, method="pearson"
    )
    return {
        "state_row_block_linear_r2": _group_linear_r2(
            calibration.online.flatten(0, 1),
            calibration.states.flatten(0, 1),
            scored.online.flatten(0, 1),
            scored.states.flatten(0, 1),
            rows,
            matches["state"],
        ),
        "parameter_column_pair_linear_r2": _group_linear_r2(
            calibration.codes,
            calibration.parameters,
            scored.codes,
            scored.parameters,
            pairs,
            matches["parameter"],
        ),
        "parameter_gaussianized_mcc": mean_correlation_coefficient(
            matches["parameter_gaussianized"], scored.codes, gaussian_scored
        ),
        **_paired_code_variance(scored.codes, scored.config.episodes_per_parameter),
    }


def _recovery_metrics(
    calibration: _EncodedSplit, scored: _EncodedSplit, threshold: float
) -> tuple[dict[str, float], dict[str, CoordinateMatch], dict[str, Tensor]]:
    """Fit only calibration assignments and score a separate validation/test split."""
    matches = {
        "state": fit_coordinate_match(
            calibration.online.flatten(0, 1), calibration.states.flatten(0, 1), method="pearson"
        ),
        "parameter": fit_coordinate_match(
            calibration.codes, calibration.parameters, method="pearson"
        ),
        "target": fit_coordinate_match(
            calibration.target.flatten(0, 1),
            calibration.target_states.flatten(0, 1),
            method="pearson",
        ),
    }
    metrics = {
        "state_mcc": mean_correlation_coefficient(
            matches["state"], scored.online.flatten(0, 1), scored.states.flatten(0, 1)
        ),
        "parameter_mcc": mean_correlation_coefficient(
            matches["parameter"], scored.codes, scored.parameters
        ),
    }
    support = _aligned_support(
        scored.graph_frequencies,
        matches["state"],
        matches["parameter"],
        threshold,
        target_match=matches["target"],
    )
    truth = scored.reference_graphs
    for name, predicted in support.items():
        target = cast(Tensor, getattr(truth, name))
        metrics[f"{name}_shd"] = float((predicted != target).sum())
        if scored.config.family == "simplified_gaussian":
            metrics[f"{name}_false_positives"] = float((predicted & ~target).sum())
            metrics[f"{name}_false_negatives"] = float((~predicted & target).sum())
    if scored.config.family == "simplified_gaussian":
        metrics.update(_simplified_recovery_metrics(calibration, scored, matches))
    return metrics, matches, support


def evaluate_model(
    model: SCJEPA,
    base_data: DGPConfig,
    observation_matrix: Tensor,
    config: EvaluationConfig,
    artifact_dir: Path,
    *,
    objective: ObjectiveConfig | None = None,
) -> dict[str, float]:
    """Report held-out MCC and permutation-aligned residual-inclusive routing SHD.

    The default global support is the union of deterministic local routing paths
    over every test transition. For a positive predeclared threshold an edge must
    occur in a strictly greater fraction of transitions. The threshold is never
    fitted to graph labels. SHD counts false positives plus false negatives;
    poor MCC or nonlinear coordinate mixing limits its causal interpretation.
    """
    _check_model(model, base_data)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with _evaluation_context(model, config.seed):
        calibration = _collect(
            model,
            base_data,
            observation_matrix,
            config,
            "evaluation_calibration",
            config.calibration_episodes,
            objective,
        )
        # The test loss uses the same fixed projection stream regardless of
        # how many calibration batches were needed to fit its permutations.
        with _evaluation_context(model, config.seed):
            test = _collect(
                model,
                base_data,
                observation_matrix,
                config,
                "test",
                config.test_episodes,
                objective,
                shuffle_parameters=base_data.family == "simplified_gaussian",
            )
        diagnostics, matches, support = _recovery_metrics(
            calibration, test, config.graph_support_threshold
        )
        metrics = _loss_metrics(test, "test")
        for name, value in diagnostics.items():
            prefix = (
                "graph"
                if name.endswith(("_shd", "_false_positives", "_false_negatives"))
                else "recovery"
            )
            metrics[f"{prefix}/{name}"] = value
        if test.shuffled_prediction_mse is not None:
            metrics["test/shuffled_parameter_prediction_mse"] = test.shuffled_prediction_mse
            metrics["test/shuffled_parameter_prediction_mse_delta"] = (
                test.shuffled_prediction_mse - test.prediction_mse
            )
        truth = test.reference_graphs
        views = base_data.episodes_per_parameter
        torch.save(
            {
                "format_version": 1,
                "group_ids": test.ids.reshape(-1, views)[:, 0],
                "calibration_group_ids": calibration.ids.reshape(-1, views)[:, 0],
                "states": test.states.reshape(-1, views, *test.states.shape[1:]),
                "online": test.online.reshape(-1, views, *test.online.shape[1:]),
                "state_match": torch.tensor(matches["state"].columns),
            },
            artifact_dir / "mcc_samples.pt",
        )
        torch.save(
            {
                "matrix": observation_matrix.detach().cpu(),
                "calibration_episode_ids": calibration.ids,
                "test_episode_ids": test.ids,
                **{f"{name}_match": torch.tensor(match.columns) for name, match in matches.items()},
                "routing_frequencies": test.graph_frequencies,
                "aligned_support": support,
                "reference_graphs": {name: getattr(truth, name) for name in support},
            },
            artifact_dir / "evaluation_references.pt",
        )
        metadata = {
            "format_version": 2,
            "evaluation_config": asdict(config),
            "calibration_data": asdict(calibration.config),
            "test_data": asdict(test.config),
            "observation_matrix_sha256": VectorObservationMap(
                matrix=observation_matrix, dimension=observation_matrix.shape[0]
            ).fingerprint,
            "mcc": "mean absolute Pearson correlation with calibration-only Hungarian assignment",
            "parameter_source": model.parameter_source,
            "parameter_recovery_interpretation": (
                "provided raw-theta reference scores, not learned parameter recovery; "
                "raw MCC and column-pair linear R2 should be approximately one, and "
                "within-theta variance zero; Gaussianized-reference MCC need not equal one"
                if model.parameter_source == "oracle"
                else "recovery of learned individual context-derived parameter codes"
            ),
            "matching_reference": "https://github.com/slachapelle/disentanglement_via_mechanism_sparsity/blob/7fd50ded090a51a7527520f07d933450aa5b872a/metrics.py",
            "state_rows": (
                "all frames from every view; parameters have one row per context code "
                "with shared truth repeated; episode IDs identify independent parameter groups"
                if base_data.episodes_per_parameter >= 2
                else "all frames; parameters have one row per distinct episode"
            ),
            "prediction_context": (
                "paired episodes: opposite episode's code in both directions"
                if base_data.episodes_per_parameter == 2
                else (
                    "grouped episodes: cyclic view shift by one; one donor context code "
                    "per recipient, without averaging codes"
                    if base_data.episodes_per_parameter > 2
                    else "same-episode context"
                )
            ),
            "prediction_mode": "autoregressive" if model.autoregressive else "teacher_forced",
            "shd": "false positives plus false negatives; no reversal discount",
            "graph_aggregation": (
                "frequency > predeclared threshold over all test transitions; zero means union"
            ),
            "graph_semantics": (
                "residual-inclusive local one-step value-routing support at each predictor "
                "call, without composing paths across rollout steps; conditional on inferred "
                "parameters; not a causal guarantee"
            ),
            "graph_alignment": (
                "EMA target assignment on output rows; online state assignment on state "
                "columns; parameter assignment on parameter columns; observed actions retain "
                "their ordering; oldest-first lags; all assignments fitted on calibration only"
            ),
            "interpretation": (
                "MCC is a linear coordinate-recovery measure; poor MCC or nonlinear mixing "
                "prevents causal interpretation of routing SHD"
            ),
        }
        if model.parameter_source == "oracle":
            metadata["prediction_context"] = (
                "provided raw ground-truth theta per independent parameter group; shared "
                "unchanged across its views and prediction steps; parameter encoder bypassed"
            )
            metadata["graph_semantics"] = (
                "residual-inclusive local one-step value-routing support at each predictor "
                "call, without composing paths across rollout steps; conditional on provided "
                "raw ground-truth parameters; not a causal guarantee"
            )
        if base_data.family == "simplified_gaussian":
            metadata["additional_diagnostics"] = {
                "group_recovery": (
                    "linear ridge R2, ridge=0.001; two state row blocks and parameter column "
                    "pairs; feature subsets selected by calibration-only coordinate matches; "
                    "fit only on calibration, score held-out groups; nonlinear within-group "
                    "changes and failed coordinate matching can lower these scores"
                ),
                "parameter_gaussianized_mcc": (
                    "known Uniform(-0.75,0.75) CDF followed by the standard Gaussian inverse "
                    "CDF; independent calibration-only assignment; raw MCC retained"
                ),
                "parameter_context_variance": (
                    "all independently initialized test views per shared theta; unbiased "
                    "within-group sample variance of individual codes (R=2: squared difference "
                    "/ 2), normalized by marginal variance across groups "
                    "averaged over views; denominator floored at 1e-12; inspect total "
                    "variance because constant codes also have zero within-theta variance"
                ),
                "parameter_shuffle": (
                    "cyclic one-group shift within each test batch, never the same theta; "
                    + (
                        "provided raw theta shifted between groups and repeated across views; "
                        if model.parameter_source == "oracle"
                        else "single-context cyclic-view donor retained; "
                    )
                    + "autoregressive prediction and input-dependent "
                    "gates recomputed; report MSE and delta from unchanged-code test MSE"
                ),
            }
        (artifact_dir / "protocol.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
        (artifact_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    return metrics
