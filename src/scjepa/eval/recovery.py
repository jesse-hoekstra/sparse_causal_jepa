"""Calibration-fitted, frozen representation-recovery diagnostics.

Coordinate assignment maximizes absolute Spearman correlation on calibration
rows only. Nonlinear probes are evaluation-only CPU regressors: fixed training
steps and calibration-only standardization avoid selecting anything on test
labels. These finite-sample diagnostics do not establish identifiability.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TypedDict, cast

import torch
from torch import Tensor, nn


def _matrix(values: Tensor, name: str) -> Tensor:
    """Validate and detach a finite nonempty sample-by-coordinate matrix."""
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError(f"{name} must have shape (at least two rows, nonempty coordinates)")
    if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must contain finite floating-point values")
    return values.detach().cpu().double()


def _paired(features: Tensor, truth: Tensor) -> tuple[Tensor, Tensor]:
    """Validate two reference-aligned matrices without retaining gradients."""
    features, truth = _matrix(features, "features"), _matrix(truth, "truth")
    if features.shape[0] != truth.shape[0]:
        raise ValueError("features and truth must contain the same number of rows")
    return features, truth


def _ranks(values: Tensor) -> Tensor:
    """Compute column-wise average ranks, including exact ties."""
    result = torch.empty_like(values)
    for column in range(values.shape[1]):
        ordered, indices = values[:, column].sort(stable=True)
        _, counts = cast(
            tuple[Tensor, Tensor],
            torch.unique_consecutive(ordered, return_counts=True),  # pyright: ignore[reportUnknownMemberType]
        )
        ends = counts.cumsum(0).double()
        averages = ends - (counts.double() - 1.0) / 2.0
        result[indices, column] = averages.repeat_interleave(counts)
    return result


def spearman_matrix(features: Tensor, truth: Tensor) -> Tensor:
    """Return absolute rank correlation, with true rows and learned columns.

    Equal values receive average ranks. Constant coordinates have correlation
    zero with every coordinate, including other constants, instead of NaN.
    """
    features, truth = _paired(features, truth)
    feature_ranks, truth_ranks = _ranks(features), _ranks(truth)
    feature_ranks -= feature_ranks.mean(dim=0)
    truth_ranks -= truth_ranks.mean(dim=0)
    feature_ranks /= feature_ranks.square().sum(dim=0).sqrt().clamp_min(1e-15)
    truth_ranks /= truth_ranks.square().sum(dim=0).sqrt().clamp_min(1e-15)
    return (truth_ranks.T @ feature_ranks).abs().clamp(0.0, 1.0)


def _maximum_assignment(scores: Tensor) -> tuple[int, ...]:
    """Solve square assignment exactly with the O(d³) Hungarian algorithm."""
    # The shortest augmenting path formulation maintains dual row/column
    # potentials. Ascending iteration makes equal-score assignments repeatable.
    size = scores.shape[0]
    costs = cast(list[list[float]], (-scores).tolist())  # pyright: ignore[reportUnknownMemberType]
    row_potential, column_potential = [0.0] * (size + 1), [0.0] * (size + 1)
    column_row, predecessor = [0] * (size + 1), [0] * (size + 1)
    for row in range(1, size + 1):
        column_row[0] = row
        column = 0
        minimum, used = [float("inf")] * (size + 1), [False] * (size + 1)
        while True:
            used[column] = True
            current_row = column_row[column]
            delta, next_column = float("inf"), 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[current_row - 1][candidate - 1]
                    - row_potential[current_row]
                    - column_potential[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < delta:
                    delta, next_column = minimum[candidate], candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    row_potential[column_row[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if column_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            column_row[column] = column_row[previous]
            column = previous
            if column == 0:
                break
    assignment = [0] * size
    for column in range(1, size + 1):
        assignment[column_row[column] - 1] = column - 1
    return tuple(assignment)


@dataclass(frozen=True)
class CoordinateMatch:
    """Frozen calibration assignment, mapping true index to learned index."""

    columns: tuple[int, ...]

    def __post_init__(self) -> None:
        """Require an exact permutation of equally many learned coordinates."""
        if not self.columns or sorted(self.columns) != list(range(len(self.columns))):
            raise ValueError("coordinate assignment must be a nonempty permutation")


def fit_coordinate_match(features: Tensor, truth: Tensor) -> CoordinateMatch:
    """Fit a one-to-one assignment using only caller-supplied calibration rows."""
    scores = spearman_matrix(features, truth)
    if scores.shape[0] != scores.shape[1]:
        raise ValueError("one-to-one recovery requires equal learned and true dimensions")
    return CoordinateMatch(_maximum_assignment(scores))


def score_coordinate_match(
    match: CoordinateMatch, features: Tensor, truth: Tensor
) -> dict[str, float]:
    """Score a frozen assignment without rematching on the scored rows."""
    features, truth = _paired(features, truth)
    size = len(match.columns)
    if features.shape[1] != size or truth.shape[1] != size:
        raise ValueError("scored dimensions must agree with the calibration assignment")
    scores = spearman_matrix(features, truth)[torch.arange(size), list(match.columns)]
    result = {
        "spearman_mean": float(scores.mean()),
        "spearman_min": float(scores.min()),
        "constant_truth_coordinates": float((truth.amax(0) == truth.amin(0)).sum()),
        "constant_learned_coordinates": float((features.amax(0) == features.amin(0)).sum()),
    }
    result.update({f"spearman_{index}": float(score) for index, score in enumerate(scores)})
    return result


@dataclass(frozen=True)
class ProbeConfig:
    """Fixed calibration-only probe training choices, without test selection.

    A ridge-initialized linear path and a two-layer tanh residual accommodate
    both linear and nonlinear mixing. ``steps`` counts Adam minibatch updates;
    ``ridge`` is used only for the initial linear least-squares fit.
    """

    hidden_dim: int = 64
    steps: int = 300
    batch_size: int = 256
    lr: float = 1e-3
    ridge: float = 1e-3
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject invalid or nonfinite calibration settings before fitting."""
        for name in ("hidden_dim", "steps", "batch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"probe {name} must be a positive integer")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("probe lr must be positive and finite")
        if not math.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("probe ridge must be positive and finite")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("probe seed must be an integer in [0, 2**63)")


class _ProbeNetwork(nn.Module):
    """Small residual nonlinear regressor whose weights are never trained online."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, features: Tensor) -> Tensor:
        """Add a tanh residual to the ridge-initialized linear prediction."""
        return self.linear(features) + self.output(torch.tanh(self.hidden(features)))


class ProbePayload(TypedDict):
    """Tensor-only weights and primitive metadata accepted by safe torch loading."""

    version: int
    config: dict[str, int | float]
    feature_mean: Tensor
    feature_std: Tensor
    target_mean: Tensor
    target_std: Tensor
    state_dict: dict[str, Tensor]
    calibration_episode_ids: Tensor


class FrozenProbe:
    """A frozen CPU regressor with calibration statistics and portable weights."""

    def __init__(
        self,
        network: nn.Module,
        feature_mean: Tensor,
        feature_std: Tensor,
        target_mean: Tensor,
        target_std: Tensor,
        config: ProbeConfig,
        calibration_episode_ids: Tensor,
    ) -> None:
        """Freeze a calibrated module; use ``fit_nonlinear_probe`` to construct it."""
        self.network = network.cpu().float().eval().requires_grad_(False)
        self.network.zero_grad(set_to_none=True)
        self.feature_mean = feature_mean.detach().cpu().clone()
        self.feature_std = feature_std.detach().cpu().clone()
        self.target_mean = target_mean.detach().cpu().clone()
        self.target_std = target_std.detach().cpu().clone()
        self.config = config
        self.calibration_episode_ids = calibration_episode_ids.detach().cpu().clone()

    @torch.no_grad()
    def predict(self, features: Tensor, *, batch_size: int = 4096) -> Tensor:
        """Decode arbitrary leading dimensions; return detached CPU float64 values."""
        if batch_size < 1 or features.ndim < 1 or features.shape[-1] != self.feature_mean.numel():
            raise ValueError("probe features must end in the fitted feature dimension")
        if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
            raise ValueError("probe features must be finite floating-point values")
        original_shape = features.shape[:-1]
        values = features.detach().cpu().double().reshape(-1, features.shape[-1])
        results: list[Tensor] = []
        for start in range(0, values.shape[0], batch_size):
            chunk = values[start : start + batch_size]
            normalized = ((chunk - self.feature_mean) / self.feature_std).float()
            decoded: Tensor = self.network(normalized)
            results.append(decoded.double() * self.target_std + self.target_mean)
        output_dim = self.target_mean.numel()
        if not results:
            return torch.empty((*original_shape, output_dim), dtype=torch.float64)
        return torch.cat(results).reshape(*original_shape, output_dim)

    def to_payload(self) -> ProbePayload:
        """Copy all calibrated state for reproducible, weights-only serialization."""
        return {
            "version": 1,
            "config": asdict(self.config),
            "feature_mean": self.feature_mean.clone(),
            "feature_std": self.feature_std.clone(),
            "target_mean": self.target_mean.clone(),
            "target_std": self.target_std.clone(),
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in self.network.state_dict().items()
            },
            "calibration_episode_ids": self.calibration_episode_ids.clone(),
        }

    @classmethod
    def from_payload(cls, payload: ProbePayload) -> "FrozenProbe":
        """Restore frozen weights without consuming the caller's Torch RNG."""
        if payload["version"] != 1:
            raise ValueError("unsupported nonlinear probe payload version")
        config_values = payload["config"]
        config = ProbeConfig(
            hidden_dim=int(config_values["hidden_dim"]),
            steps=int(config_values["steps"]),
            batch_size=int(config_values["batch_size"]),
            lr=float(config_values["lr"]),
            ridge=float(config_values["ridge"]),
            seed=int(config_values["seed"]),
        )
        feature_mean, target_mean = payload["feature_mean"], payload["target_mean"]
        for name in ("feature_mean", "feature_std", "target_mean", "target_std"):
            values = cast(Tensor, cast(Mapping[str, object], payload)[name])
            if values.ndim != 1 or values.numel() < 1 or not bool(torch.isfinite(values).all()):
                raise ValueError("probe payload statistics must be finite nonempty vectors")
        if (
            payload["feature_std"].shape != feature_mean.shape
            or payload["target_std"].shape != target_mean.shape
            or bool((payload["feature_std"] <= 0).any())
            or bool((payload["target_std"] <= 0).any())
        ):
            raise ValueError(
                "probe payload scales must be positive and match the fitted dimensions"
            )
        with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
            network = _ProbeNetwork(
                feature_mean.numel(), target_mean.numel(), config.hidden_dim
            ).float()
        network.load_state_dict(payload["state_dict"], strict=True)
        if any(not bool(torch.isfinite(value).all()) for value in network.state_dict().values()):
            raise ValueError("probe payload weights must be finite")
        return cls(
            network,
            feature_mean,
            payload["feature_std"],
            target_mean,
            payload["target_std"],
            config,
            payload["calibration_episode_ids"],
        )


def fit_nonlinear_probe(
    features: Tensor,
    truth: Tensor,
    *,
    config: ProbeConfig | None = None,
    episode_ids: Tensor | None = None,
) -> FrozenProbe:
    """Fit on calibration rows only with detached labels and isolated randomness.

    There is no internal train/test split or early stopping: every fit choice
    is fixed by ``config``. Repeated frame rows may share an episode ID, which
    is recorded once for provenance. Callers must keep calibration episodes
    disjoint from test episodes. No row-level split is performed.
    """
    features, truth = _paired(features, truth)
    config = config or ProbeConfig()
    if episode_ids is None:
        unique_ids = torch.empty(0, dtype=torch.int64)
    else:
        if episode_ids.shape != (features.shape[0],) or episode_ids.dtype != torch.int64:
            raise ValueError("probe episode_ids must be one int64 identifier per calibration row")
        unique_ids = cast(Tensor, episode_ids.detach().cpu().unique(sorted=True))  # pyright: ignore[reportUnknownMemberType]
    feature_mean, target_mean = features.mean(0), truth.mean(0)
    feature_std = features.std(0, unbiased=False).clamp_min(1e-8)
    target_std = truth.std(0, unbiased=False).clamp_min(1e-8)
    normalized_features = (features - feature_mean) / feature_std
    normalized_truth = (truth - target_mean) / target_std
    count = features.shape[0]
    gram = normalized_features.T @ normalized_features / count
    gram += config.ridge * torch.eye(features.shape[1], dtype=torch.float64)
    linear_weight = cast(
        Tensor,
        torch.linalg.solve(gram, normalized_features.T @ normalized_truth / count),  # pyright: ignore[reportUnknownMemberType]
    )
    x, y = normalized_features.float(), normalized_truth.float()
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        torch.default_generator.manual_seed(config.seed)
        generator = torch.Generator().manual_seed(config.seed)
        network = _ProbeNetwork(features.shape[1], truth.shape[1], config.hidden_dim).float()
        with torch.no_grad():
            network.linear.weight.copy_(linear_weight.T.float())
            network.linear.bias.zero_()
            network.output.weight.zero_()
            network.output.bias.zero_()
        optimizer = torch.optim.Adam(network.parameters(), lr=config.lr)
        with torch.enable_grad():
            for _ in range(config.steps):
                indices = torch.randint(
                    count, (min(count, config.batch_size),), generator=generator
                )
                optimizer.zero_grad(set_to_none=True)
                loss = (network(x[indices]) - y[indices]).square().mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        "nonlinear recovery probe fit produced a nonfinite loss"
                    )
                loss.backward()  # pyright: ignore[reportUnknownMemberType]
                optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    return FrozenProbe(
        network,
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        config,
        unique_ids,
    )


def probe_metrics(prediction: Tensor, truth: Tensor) -> dict[str, float]:
    """Report unclipped held-out R² and MSE, flagging constant true coordinates.

    Constant targets receive R² zero rather than an undefined score. Their
    prediction errors still contribute to MSE. Per-coordinate R² is retained.
    """
    prediction, truth = _paired(prediction, truth)
    if prediction.shape != truth.shape:
        raise ValueError("probe predictions and truth must have identical shapes")
    errors = (prediction - truth).square().mean(0)
    variances = (truth - truth.mean(0)).square().mean(0)
    constants = variances == 0
    scores = torch.where(constants, 0.0, 1.0 - errors / variances.clamp_min(1e-30))
    metrics = {
        "r2": float(scores.mean()),
        "mse": float(errors.mean()),
        "constant_truth_coordinates": float(constants.sum()),
    }
    metrics.update({f"r2_{index}": float(score) for index, score in enumerate(scores)})
    return metrics


@dataclass(frozen=True)
class GroupProbe:
    """A frozen diagnostic for explicitly chosen true and learned subsets."""

    truth_columns: tuple[int, ...]
    feature_columns: tuple[int, ...]
    probe: FrozenProbe

    def predict(self, features: Tensor) -> Tensor:
        """Decode only the group's fixed calibration-selected feature subset."""
        return self.probe.predict(features[..., list(self.feature_columns)])


def fit_group_probes(
    features: Tensor,
    truth: Tensor,
    groups: Sequence[Sequence[int]],
    *,
    match: CoordinateMatch,
    config: ProbeConfig | None = None,
    episode_ids: Tensor | None = None,
) -> tuple[GroupProbe, ...]:
    """Fit group probes using coordinate-assigned calibration feature subsets.

    This is an explicit diagnostic convention, not an invariant optimizer over
    all possible learned groups. Severe within-group nonlinear mixing can
    defeat coordinate matching; full-vector probes are needed alongside it.
    """
    features, truth = _paired(features, truth)
    if features.shape[1] != len(match.columns) or truth.shape[1] != len(match.columns):
        raise ValueError("group probes require dimensions matching the coordinate assignment")
    all_columns = [index for group in groups for index in group]
    if sorted(all_columns) != list(range(truth.shape[1])) or any(not group for group in groups):
        raise ValueError("groups must be a nonempty disjoint partition of true coordinates")
    result: list[GroupProbe] = []
    for group in groups:
        true_columns = tuple(group)
        feature_columns = tuple(match.columns[index] for index in group)
        probe = fit_nonlinear_probe(
            features[:, list(feature_columns)],
            truth[:, list(true_columns)],
            config=config,
            episode_ids=episode_ids,
        )
        result.append(GroupProbe(true_columns, feature_columns, probe))
    return tuple(result)
