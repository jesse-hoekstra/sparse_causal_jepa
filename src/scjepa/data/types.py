"""Small shared contracts for scientific generators and training-safe episode adapters."""

import math
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol

import torch
from torch import Tensor


@dataclass(frozen=True)
class RendererConfig:
    """Fixed Gaussian appearance; these defaults are explicit implementation choices."""

    sigma: float = 3.0
    amplitude: float = 1.0
    track_y: tuple[float, float] = (20.0, 44.0)

    def __post_init__(self) -> None:
        """Reject degenerate appearances; retain floating-point positions and intensity."""
        object.__setattr__(self, "track_y", tuple(self.track_y))
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("renderer sigma must be finite and positive")
        if not math.isfinite(self.amplitude) or self.amplitude <= 0:
            raise ValueError("renderer amplitude must be finite and positive")
        if len(self.track_y) != 2 or any(
            not math.isfinite(y) or not 0 <= y < 64 for y in self.track_y
        ):
            raise ValueError("track_y must contain two finite positions in [0,64)")
        if self.track_y[0] == self.track_y[1]:
            raise ValueError("the two horizontal tracks must differ")


@dataclass(frozen=True)
class DGPConfig:
    """Resolved data choices, separate from architecture, losses, and scientific protocols.

    ``dtype`` is the stored/training dtype; simulation and reference ground truth
    use float64. Graph variants never enter sampling seeds. Context length belongs
    to the model configuration; the spring preset supplies eight frames there.
    ``num_episodes`` counts independent parameter draws; with
    ``episodes_per_parameter=2``, each dataset item contains two trajectories.
    """

    family: str = "controlled_vector"
    parameter_graph: str | None = None
    root_seed: int = 0
    split: str = "train"
    num_episodes: int = 8
    episodes_per_parameter: int = 1
    trajectory_length: int = 16
    dtype: str = "float32"
    observation_seed: int = 1729
    position_mean: float = 0.0
    position_std: float = 1.0
    velocity_mean: float = 0.0
    velocity_std: float = 1.0
    renderer: RendererConfig = field(default_factory=RendererConfig)

    def __post_init__(self) -> None:
        """Validate reproducible stream addresses and configuration-level dimensions."""
        # New registered families validate their own scientific fields.
        if not self.family:
            raise ValueError("family must be nonempty")
        if self.parameter_graph is None:
            object.__setattr__(
                self, "parameter_graph", "diagonal" if self.family == "coupled_springs" else "dense"
            )
        if self.parameter_graph not in {"dense", "complementary", "diagonal"}:
            raise ValueError("unknown parameter_graph")
        if self.family == "coupled_springs" and self.parameter_graph != "diagonal":
            raise ValueError("coupled_springs has the paper's diagonal parameter graph")
        if self.family == "controlled_vector" and (
            self.position_mean,
            self.position_std,
            self.velocity_mean,
            self.velocity_std,
        ) != (0.0, 1.0, 0.0, 1.0):
            raise ValueError("Gaussian overrides apply only to coupled_springs; vector uses N(0,I)")
        if self.split not in {"train", "validation", "evaluation_calibration", "test"}:
            raise ValueError("split must be train, validation, evaluation_calibration, or test")
        if not 0 <= self.root_seed < 2**31 or not 0 <= self.observation_seed < 2**63:
            raise ValueError(
                "root_seed must be 31-bit nonnegative; observation_seed 63-bit nonnegative"
            )
        if not 1 <= self.num_episodes <= 2**30:
            raise ValueError("num_episodes must be in [1,2**30]")
        if self.episodes_per_parameter not in (1, 2):
            raise ValueError("episodes_per_parameter must be 1 or 2")
        if self.episodes_per_parameter == 2 and self.family != "controlled_vector":
            raise ValueError("paired episodes are supported only for controlled_vector")
        minimum = 3 if self.family == "coupled_springs" else 2
        if self.trajectory_length < minimum:
            raise ValueError(f"trajectory_length must be at least {minimum}")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        for value in (self.position_mean, self.velocity_mean, self.position_std, self.velocity_std):
            if not math.isfinite(value):
                raise ValueError("initial Gaussian settings must be finite")
        if min(self.position_std, self.velocity_std) <= 0:
            raise ValueError("initial Gaussian scales must be positive")


class AggregatedGraphs(NamedTuple):
    """Logical union over input lags, with rows indexing output coordinates."""

    state: Tensor
    action: Tensor
    parameter: Tensor


@dataclass(frozen=True)
class GraphMasks:
    """Boolean structural graphs: state/action ``(output, lag, input)``."""

    state: Tensor
    action: Tensor
    parameter: Tensor

    def __post_init__(self) -> None:
        """Validate orientations and common output and history axes."""
        if any(value.dtype != torch.bool for value in (self.state, self.action, self.parameter)):
            raise ValueError("structural graph masks must be boolean")
        if self.state.ndim != 3 or self.action.ndim != 3 or self.parameter.ndim != 2:
            raise ValueError("graphs require (output,lag,input) and (output,parameter)")
        if (
            self.state.shape[:2] != self.action.shape[:2]
            or self.state.shape[0] != self.parameter.shape[0]
        ):
            raise ValueError("graph output/history axes must agree")

    def aggregate_lags(self) -> AggregatedGraphs:
        """Union dependencies, counting an input coordinate once across all lags."""
        return AggregatedGraphs(self.state.any(dim=1), self.action.any(dim=1), self.parameter)


@dataclass(frozen=True)
class TransitionJacobians:
    """Numerical derivative blocks with the same output/lag/input graph orientation."""

    state: Tensor
    action: Tensor
    parameter: Tensor


@dataclass(frozen=True)
class ScientificEpisode:
    """One episode or shared-parameter pair with a separate evaluation payload."""

    observations: Tensor
    actions: Tensor
    episode_id: int
    ground_truth: dict[str, Tensor]


class DGP(Protocol):
    """A registered scientific process; no model or trainer dependencies."""

    config: DGPConfig
    observation_matrix: Tensor | None

    def restore_observation_matrix(self, matrix: Tensor) -> None:
        """Use a persisted realized map; non-matrix observers reject this operation."""
        ...

    def specification(self) -> dict[str, object]:
        """Describe dimensions, observation map, dynamics, and indexing conventions."""
        ...

    def graphs(self) -> GraphMasks:
        """Return structural dependency masks independently of a sampled episode."""
        ...

    def generate(self, episode_index: int) -> ScientificEpisode:
        """Deterministically generate one episode by stable integer address."""
        ...
