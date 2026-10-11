"""Held-out evaluation for the controlled-vector comparison.

All fitting and matching use evaluation-calibration episodes. Physical predictions
are decoded by a frozen nonlinear probe fitted to the EMA target coordinates.
This is a measured approximation to an inverse, not an exact inverse encoder.
No labels, fitted probes, or same-parameter pairings enter model training.
"""

import json
import math
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from scjepa.data.controlled_vector import ParameterGraph, VectorObservationMap, simulate
from scjepa.data.processes import ControlledVectorDGP
from scjepa.data.random import episode_id, generator
from scjepa.data.types import DGPConfig, ScientificEpisode
from scjepa.eval.recovery import (
    FrozenProbe,
    ProbeConfig,
    fit_coordinate_match,
    fit_group_probes,
    fit_nonlinear_probe,
    probe_metrics,
    score_coordinate_match,
)
from scjepa.models.model import SCJEPA
from scjepa.training.rng import preserve_rng


@dataclass(frozen=True)
class EvaluationConfig:
    """Explicit evaluation sizes and probe choices, independent of training data."""

    calibration_episodes: int = 512
    validation_episodes: int = 256
    test_episodes: int = 512
    paired_episodes: int = 128
    batch_size: int = 64
    rollout_steps: int = 8
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    validation_probe_steps: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        """Reject undefined evaluation statistics and invalid stream addresses."""
        for name in ("calibration_episodes", "validation_episodes", "test_episodes"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be at least two")
        if not 2 <= self.paired_episodes <= self.test_episodes:
            raise ValueError("paired_episodes must lie between two and test_episodes")
        if min(self.batch_size, self.rollout_steps, self.validation_probe_steps) < 1:
            raise ValueError(
                "batch_size, rollout_steps, and validation_probe_steps must be positive"
            )
        if not 0 <= self.seed < 2**63:
            raise ValueError("evaluation seed must be a nonnegative 63-bit integer")


@dataclass(frozen=True)
class _EncodedSplit:
    config: DGPConfig
    ids: Tensor
    states: Tensor
    parameters: Tensor
    actions: Tensor
    online: Tensor
    target: Tensor
    codes: Tensor
    predictions: Tensor
    start: int
    graphs: dict[str, Tensor]


@contextmanager
def _evaluation_context(model: SCJEPA) -> Generator[None]:
    """Preserve all module modes and host/device random streams, including errors."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        with preserve_rng(next(model.parameters()).device):
            model.eval()
            yield
    finally:
        for module, training in modes:
            module.training = training


def _check_model(model: SCJEPA, data: DGPConfig, config: EvaluationConfig) -> None:
    if data.family != "controlled_vector":
        raise ValueError("this experiment evaluates only controlled_vector data")
    predictor = model.predictor
    if (
        model.online_encoder.observation_ndim != 1
        or predictor.state_dim != 10
        or predictor.param_dim != 10
        or predictor.action_dim != 10
        or predictor.history_len != 1
        or model.context_len < 2
    ):
        raise ValueError(
            "vector experiment requires vector observations, dimensions 10/10/10, L=1, K>=2"
        )
    if data.trajectory_length < model.context_len + config.rollout_steps:
        raise ValueError("trajectory_length must cover the context and requested rollout_steps")


def _split_config(base: DGPConfig, split: str, count: int) -> DGPConfig:
    return replace(base, split=split, num_episodes=count)


@torch.no_grad()
def _encode_episodes(
    model: SCJEPA,
    data: DGPConfig,
    episodes: Iterator[ScientificEpisode],
    batch_size: int,
) -> _EncodedSplit:
    reference = next(model.online_encoder.parameters())
    buckets: dict[str, list[Tensor]] = {
        key: []
        for key in (
            "ids",
            "states",
            "parameters",
            "actions",
            "online",
            "target",
            "codes",
            "predictions",
        )
    }
    graph_blocks: dict[str, Tensor] = {}
    start = model.context_len - 1
    batch: list[ScientificEpisode] = []

    def consume() -> None:
        observations = torch.stack([item.observations for item in batch]).to(reference)
        actions = torch.stack([item.actions for item in batch]).to(reference)
        result = model(observations, actions)
        buckets["ids"].append(torch.tensor([item.episode_id for item in batch], dtype=torch.int64))
        buckets["states"].append(torch.stack([item.ground_truth["states"] for item in batch]))
        buckets["parameters"].append(
            torch.stack([item.ground_truth["parameters"] for item in batch])
        )
        buckets["actions"].append(actions.cpu())
        buckets["online"].append(result.online_states.cpu())
        buckets["target"].append(model.target_encoder.encode_sequence(observations).cpu())
        buckets["codes"].append(result.parameters.cpu())
        buckets["predictions"].append(result.predictions.cpu())
        if not graph_blocks:
            graph = result.dynamics[0].graphs
            graph_blocks.update(
                {
                    "state": graph.state[0].cpu(),
                    "action": graph.action[0].cpu(),
                    "parameter": graph.parameter[0].cpu(),
                }
            )

    for episode in episodes:
        batch.append(episode)
        if len(batch) == batch_size:
            consume()
            batch.clear()
    if batch:
        consume()
    values = {key: torch.cat(items) for key, items in buckets.items()}
    for name, value in values.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"nonfinite evaluation values in {name}")
    return _EncodedSplit(data, **values, start=start, graphs=graph_blocks)


def _collect(
    model: SCJEPA, base: DGPConfig, matrix: Tensor, config: EvaluationConfig, split: str, count: int
) -> _EncodedSplit:
    data = _split_config(base, split, count)
    process = ControlledVectorDGP(data)
    process.restore_observation_matrix(matrix)
    return _encode_episodes(
        model, data, (process.generate(index) for index in range(count)), config.batch_size
    )


def _diagnostics(values: Tensor, prefix: str) -> dict[str, float]:
    rows = values.detach().cpu().double().reshape(-1, values.shape[-1])
    centered = rows - rows.mean(0)
    variance = centered.square().mean(0)
    eigenvalues = cast(
        Tensor,
        torch.linalg.eigvalsh(centered.T @ centered / rows.shape[0]),  # pyright: ignore[reportUnknownMemberType]
    ).clamp(min=0)
    total = eigenvalues.sum()
    probabilities = eigenvalues / total.clamp(min=1e-30)
    entropy = -(probabilities * probabilities.clamp(min=1e-30).log()).sum()
    return {
        f"{prefix}/std_mean": float(variance.sqrt().mean()),
        f"{prefix}/std_min": float(variance.sqrt().min()),
        f"{prefix}/effective_rank": float(entropy.exp()) if float(total) > 0 else 0.0,
        f"{prefix}/collapsed_coordinates": float((variance <= 1e-12).sum()),
    }


def _physical_metrics(
    predicted: Tensor, truth: Tensor, variance: Tensor, prefix: str
) -> dict[str, float]:
    if predicted.shape != truth.shape:
        raise ValueError("physical predictions and truth must have identical shapes")
    squared = (predicted.double().cpu() - truth.double().cpu()).square()
    return {
        f"{prefix}_mse": float(squared.mean()),
        f"{prefix}_nrmse": float((squared / variance).mean().sqrt()),
    }


def _calibration_variance(calibration: _EncodedSplit) -> Tensor:
    variance = calibration.states.flatten(0, 1).var(0, unbiased=False)
    if (variance <= 0).any() or not torch.isfinite(variance).all():
        raise ValueError("calibration physical-coordinate variances must be positive and finite")
    return variance


def _check_metrics(metrics: dict[str, float]) -> dict[str, float]:
    if any(not math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("evaluation produced a nonfinite metric")
    return metrics


def _probe_config(
    config: EvaluationConfig, offset: int = 0, *, validation: bool = False
) -> ProbeConfig:
    return replace(
        config.probe,
        steps=config.validation_probe_steps if validation else config.probe.steps,
        seed=(config.seed + config.probe.seed + offset) % (2**63),
    )


def _target_probe(calibration: _EncodedSplit, config: ProbeConfig) -> FrozenProbe:
    return fit_nonlinear_probe(
        calibration.target.flatten(0, 1),
        calibration.states.flatten(0, 1),
        config=config,
        episode_ids=calibration.ids[:, None].expand(calibration.target.shape[:2]).flatten(),
    )


def _physical_evaluation(
    split: _EncodedSplit, probe: FrozenProbe, variance: Tensor, prefix: str
) -> dict[str, float]:
    truth = split.states[:, split.start + 1 :]
    metrics = _physical_metrics(
        probe.predict(split.predictions), truth, variance, f"{prefix}/physical_prediction"
    )
    metrics.update(
        _physical_metrics(
            probe.predict(split.target[:, split.start + 1 :]),
            truth,
            variance,
            f"{prefix}/encoded_target_decoding",
        )
    )
    metrics[f"{prefix}/latent_prediction_mse"] = float(
        (split.predictions - split.target[:, split.start + 1 :]).square().mean()
    )
    metrics[f"{prefix}/online_target_coordinate_mse"] = float(
        (split.online - split.target).square().mean()
    )
    metrics.update(_diagnostics(split.online, f"{prefix}/state"))
    metrics.update(_diagnostics(split.codes, f"{prefix}/parameter"))
    return metrics


def validate_model(
    model: SCJEPA,
    base_data: DGPConfig,
    observation_matrix: Tensor,
    config: EvaluationConfig,
) -> dict[str, float]:
    """Fit a calibration-only target probe and score validation; never read test data.

    The physical NRMSE selector includes probe approximation error. Its encoded-
    target decoding baseline and collapse diagnostics must be reported alongside it.
    Probe settings are fixed in advance; validation is not used to fit the probe.
    """
    _check_model(model, base_data, config)
    with _evaluation_context(model):
        calibration = _collect(
            model,
            base_data,
            observation_matrix,
            config,
            "evaluation_calibration",
            config.calibration_episodes,
        )
        validation = _collect(
            model, base_data, observation_matrix, config, "validation", config.validation_episodes
        )
        probe = _target_probe(calibration, _probe_config(config, validation=True))
        metrics = _physical_evaluation(
            validation, probe, _calibration_variance(calibration), "validation"
        )
    return _check_metrics(metrics)


@torch.no_grad()
def _rollout(model: SCJEPA, split: _EncodedSplit, config: EvaluationConfig) -> Tensor:
    reference = next(model.online_encoder.parameters())
    predictions: list[Tensor] = []
    for begin in range(0, len(split.ids), config.batch_size):
        rows = slice(begin, begin + config.batch_size)
        start = split.start
        history = split.online[rows, start : start + 1].to(reference)
        codes = split.codes[rows].to(reference)
        actions = split.actions[rows, start : start + config.rollout_steps].to(reference)
        predictions.append(model.rollout(history, codes, config.rollout_steps, actions).cpu())
    return torch.cat(predictions)


def _paired_episodes(
    base: DGPConfig, matrix: Tensor, original: _EncodedSplit, count: int
) -> Iterator[ScientificEpisode]:
    """Keep each test theta, independently redraw its initial state and every action."""
    observation = VectorObservationMap(matrix=matrix)
    dtype = torch.float32 if base.dtype == "float32" else torch.float64
    variant = cast(ParameterGraph, base.parameter_graph)
    for index in range(count):
        initial = torch.randn(
            10,
            dtype=torch.float64,
            generator=generator(
                base.root_seed, "test", index, "parameter_consistency/initial_position"
            ),
        )
        actions = (
            2
            * torch.rand(
                base.trajectory_length - 1,
                10,
                dtype=torch.float64,
                generator=generator(base.root_seed, "test", index, "parameter_consistency/actions"),
            )
            - 1
        )
        parameters = original.parameters[index]
        states = simulate(initial, parameters, actions, variant)
        # A paired trajectory is tagged by its original theta-group ID. It is an
        # evaluation-only replicate, never a new independent parameter sample.
        yield ScientificEpisode(
            observations=observation.forward(states).to(dtype),
            actions=actions.to(dtype),
            episode_id=episode_id(base.root_seed, "test", index),
            ground_truth={"states": states, "parameters": parameters.clone()},
        )


@torch.no_grad()
def _swapped_predictions(
    model: SCJEPA, original: _EncodedSplit, codes: Tensor, batch_size: int
) -> Tensor:
    reference = next(model.online_encoder.parameters())
    predictions: list[Tensor] = []
    for begin in range(0, len(codes), batch_size):
        rows = slice(begin, min(begin + batch_size, len(codes)))
        parameters = codes[rows].to(reference)
        steps: list[Tensor] = []
        for source in range(original.start, original.online.shape[1] - 1):
            states = original.online[rows, source : source + 1].to(reference)
            actions = original.actions[rows, source : source + 1].to(reference)
            steps.append(model.predictor(states, parameters, actions).prediction.cpu())
        predictions.append(torch.stack(steps, dim=1))
    return torch.cat(predictions)


def _paired_metrics(
    model: SCJEPA,
    base: DGPConfig,
    matrix: Tensor,
    test: _EncodedSplit,
    config: EvaluationConfig,
    probe: FrozenProbe,
    variance: Tensor,
) -> tuple[dict[str, float], dict[str, Tensor]]:
    count = config.paired_episodes
    paired_config = _split_config(base, "test", count)
    paired = _encode_episodes(
        model, paired_config, _paired_episodes(base, matrix, test, count), config.batch_size
    )
    first, second = test.codes[:count].double(), paired.codes.double()
    within = (first - second).square().mean(0) / 2
    between = ((first + second) / 2).var(0, unbiased=False)
    metrics = {
        "consistency/within_parameter_variance": float(within.mean()),
        "consistency/between_pair_mean_variance": float(between.mean()),
        "consistency/within_over_between_variance": float(
            within.mean() / between.mean().clamp(min=1e-12)
        ),
        "consistency/collapsed_code": float(
            (torch.cat((first, second)).var(0, unbiased=False) <= 1e-12).all()
        ),
    }
    truth = test.states[:count, test.start + 1 :]
    own = test.predictions[:count]
    same = _swapped_predictions(model, test, paired.codes, config.batch_size)
    different = _swapped_predictions(model, test, test.codes[:count].roll(1, 0), config.batch_size)
    for name, prediction in (
        ("own", own),
        ("same_parameter_swap", same),
        ("different_parameter_swap", different),
    ):
        metrics.update(
            _physical_metrics(probe.predict(prediction), truth, variance, f"consistency/{name}")
        )
    metrics["consistency/same_parameter_swap_nrmse_change"] = (
        metrics["consistency/same_parameter_swap_nrmse"] - metrics["consistency/own_nrmse"]
    )
    metrics["consistency/different_parameter_swap_nrmse_change"] = (
        metrics["consistency/different_parameter_swap_nrmse"] - metrics["consistency/own_nrmse"]
    )
    metrics["consistency/same_parameter_latent_prediction_change_mse"] = float(
        (same - own).square().mean()
    )
    metrics["consistency/different_parameter_latent_prediction_change_mse"] = float(
        (different - own).square().mean()
    )
    return metrics, {
        "group_episode_ids": test.ids[:count],
        "true_parameters": test.parameters[:count],
        "original_codes": first,
        "paired_codes": second,
        "paired_true_parameters": paired.parameters,
    }


def _recovery_metrics(
    calibration: _EncodedSplit,
    test: _EncodedSplit,
    config: EvaluationConfig,
    artifact_dir: Path,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    state_groups = (tuple(range(5)), tuple(range(5, 10)))
    variant = calibration.config.parameter_graph
    parameter_groups = (
        (tuple(range(10)),)
        if variant == "dense"
        else tuple((index, index + 5) for index in range(5))
        if variant == "complementary"
        else tuple((index,) for index in range(10))
    )
    for offset, (
        name,
        fit_features,
        fit_truth,
        test_features,
        test_truth,
        groups,
        ids,
    ) in enumerate(
        (
            (
                "state",
                calibration.online.flatten(0, 1),
                calibration.states.flatten(0, 1),
                test.online.flatten(0, 1),
                test.states.flatten(0, 1),
                state_groups,
                calibration.ids[:, None].expand(calibration.online.shape[:2]).flatten(),
            ),
            (
                "parameter",
                calibration.codes,
                calibration.parameters,
                test.codes,
                test.parameters,
                parameter_groups,
                calibration.ids,
            ),
        )
    ):
        matching = fit_coordinate_match(fit_features, fit_truth)
        scores = score_coordinate_match(matching, test_features, test_truth)
        metrics.update(
            {f"recovery/{name}/coordinate/{key}": value for key, value in scores.items()}
        )
        probe_config = _probe_config(config, 100 + offset)
        full_probe = fit_nonlinear_probe(
            fit_features, fit_truth, config=probe_config, episode_ids=ids
        )
        metrics.update(
            {
                f"recovery/{name}/full/{key}": value
                for key, value in probe_metrics(
                    full_probe.predict(test_features), test_truth
                ).items()
            }
        )
        group_probes = fit_group_probes(
            fit_features, fit_truth, groups, match=matching, config=probe_config, episode_ids=ids
        )
        group_scores: list[float] = []
        group_payloads: list[dict[str, object]] = []
        for index, group in enumerate(group_probes):
            scores = probe_metrics(
                group.predict(test_features), test_truth[:, list(group.truth_columns)]
            )
            metrics.update(
                {f"recovery/{name}/group_{index}/{key}": value for key, value in scores.items()}
            )
            group_scores.append(scores["r2"])
            group_payloads.append(
                {
                    "truth_columns": list(group.truth_columns),
                    "feature_columns": list(group.feature_columns),
                    "probe": group.probe.to_payload(),
                }
            )
        metrics[f"recovery/{name}/group_mean_r2"] = sum(group_scores) / len(group_scores)
        torch.save(
            {
                "coordinate_match": torch.tensor(matching.columns),
                "full_probe": full_probe.to_payload(),
                "group_probes": group_payloads,
            },
            artifact_dir / f"{name}_recovery.pt",
        )
    return metrics


def evaluate_model(
    model: SCJEPA,
    base_data: DGPConfig,
    observation_matrix: Tensor,
    config: EvaluationConfig,
    artifact_dir: Path,
) -> dict[str, float]:
    """Fit frozen calibration maps and report final held-out test measurements.

    Saves weights-only-loadable probe/matching payloads, actual observation map,
    paired-code audit tensors, and JSON results/provenance. Group feature selection
    uses calibration coordinate matching; it is not invariant to within-group
    rotations. Full-vector probes measure recoverable information separately.
    """
    _check_model(model, base_data, config)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with _evaluation_context(model):
        calibration = _collect(
            model,
            base_data,
            observation_matrix,
            config,
            "evaluation_calibration",
            config.calibration_episodes,
        )
        test = _collect(model, base_data, observation_matrix, config, "test", config.test_episodes)
        probe = _target_probe(calibration, _probe_config(config))
        variance = _calibration_variance(calibration)
        metrics = _physical_evaluation(test, probe, variance, "test")
        metrics.update(_recovery_metrics(calibration, test, config, artifact_dir))
        rollout = _rollout(model, test, config)
        truth = test.states[:, test.start + 1 : test.start + 1 + config.rollout_steps]
        decoded = probe.predict(rollout)
        metrics.update(_physical_metrics(decoded, truth, variance, "test/rollout"))
        for step in range(config.rollout_steps):
            metrics.update(
                _physical_metrics(
                    decoded[:, step], truth[:, step], variance, f"test/rollout_step_{step + 1}"
                )
            )
        paired_metrics, paired_payload = _paired_metrics(
            model, base_data, observation_matrix, test, config, probe, variance
        )
        metrics.update(paired_metrics)
        for name, paths in test.graphs.items():
            support = paths > 0
            aggregate = support.any(1) if name in {"state", "action"} else support
            metrics[f"graph/{name}/lag_resolved_edges"] = float(support.sum())
            metrics[f"graph/{name}/edges"] = float(aggregate.sum())
            metrics[f"graph/{name}/density"] = float(aggregate.double().mean())
        _check_metrics(metrics)
        torch.save(probe.to_payload(), artifact_dir / "target_state_probe.pt")
        torch.save(paired_payload, artifact_dir / "parameter_consistency.pt")
        torch.save(
            {
                "matrix": observation_matrix.detach().cpu(),
                "calibration_episode_ids": calibration.ids,
                "test_episode_ids": test.ids,
                "physical_normalization_variance": variance,
                **test.graphs,
            },
            artifact_dir / "evaluation_references.pt",
        )
        metadata = {
            "format_version": 1,
            "evaluation_config": asdict(config),
            "calibration_data": asdict(calibration.config),
            "test_data": asdict(test.config),
            "observation_matrix_sha256": VectorObservationMap(
                matrix=observation_matrix
            ).fingerprint,
            "physical_decoder": "frozen nonlinear calibration probe on complete EMA target encoder",
            "normalization": "per-coordinate population variance of all calibration true states",
            "state_rows": (
                "all frames, equal episode length; parameter rows one per distinct episode"
            ),
            "group_selection": (
                "calibration Spearman coordinate assignment, then corresponding learned subsets"
            ),
            "rollout_coordinates": (
                "online encoded initial history; appended predictions are EMA target coordinates"
            ),
            "consistency": (
                "two independent initial-state/action trajectories per test theta; "
                "cyclic different-theta swaps"
            ),
            "graph_semantics": (
                "residual-inclusive structural reachability conditional on supplied parameter code"
            ),
        }
        (artifact_dir / "protocol.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
        (artifact_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    return metrics
