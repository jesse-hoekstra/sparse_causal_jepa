"""Typed, reproducible settings for the controlled-vector comparison."""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, ListConfig, OmegaConf

from scjepa.data.types import DGPConfig
from scjepa.experiments.evaluation import EvaluationConfig
from scjepa.training import ObjectiveConfig, TrainConfig
from scjepa.training.factory import ModelConfig
from scjepa.training.logging import WandbConfig


def _model() -> ModelConfig:
    return ModelConfig(
        observation_dim=10,
        state_dim=10,
        param_dim=10,
        action_dim=10,
        context_len=2,
        history_len=1,
    )


def _training() -> TrainConfig:
    return TrainConfig(
        steps=10_000,
        batch_size=64,
        checkpoint_every=250,
        log_every=25,
        objective=ObjectiveConfig(
            state_sparsity_weight=0.001,
            action_sparsity_weight=0.001,
            parameter_sparsity_weight=0.001,
            state_regularizer="sigreg",
            state_regularizer_weight=0.1,
            parameter_regularizer="sigreg",
            parameter_regularizer_weight=0.1,
        ),
    )


@dataclass
class ExperimentConfig:
    """An explicit comparison protocol; numeric defaults are untuned choices.

    ``train.objective.parameter_sparsity_weight`` is the full-model weight.
    The baseline changes only that weight to zero. Model seeds do not change
    the scientific data seed, observation matrix, or calibration/test splits.
    """

    name: str = "controlled-vector"
    output_dir: str = "outputs/controlled_vector"
    graphs: tuple[str, ...] = ("dense", "complementary", "diagonal")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    train_episodes: int = 8192
    trajectory_length: int = 16
    data_seed: int = 0
    observation_seed: int = 1729
    validation_every: int = 1000
    upload_evaluation_artifacts: bool = True
    model: ModelConfig = field(default_factory=_model)
    train: TrainConfig = field(default_factory=_training)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def validate(self) -> None:
        """Reject ambiguous comparisons before creating outputs or tracking runs."""
        self.train.__post_init__()
        self.train.objective.__post_init__()
        self.wandb.__post_init__()
        self.evaluation.__post_init__()
        if self.train.seed != 0:
            raise ValueError("set experiment seeds rather than the derived train.seed")
        if not self.name or not self.output_dir:
            raise ValueError("experiment name and output_dir must be nonempty")
        if (
            not self.graphs
            or len(set(self.graphs)) != len(self.graphs)
            or any(graph not in {"dense", "complementary", "diagonal"} for graph in self.graphs)
        ):
            raise ValueError("graphs must be distinct named controlled-vector variants")
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(not 0 <= seed < 2**31 for seed in self.seeds)
        ):
            raise ValueError("seeds must be distinct nonnegative 31-bit integers")
        if self.train_episodes < self.train.batch_size or self.validation_every < 1:
            raise ValueError("need at least a full training batch and positive validation_every")
        model = self.model
        if (
            model.observation_kind != "vector"
            or (model.observation_dim, model.state_dim, model.param_dim, model.action_dim)
            != (10, 10, 10, 10)
            or model.history_len != 1
            or not 2 <= model.context_len < self.trajectory_length
        ):
            raise ValueError("controlled comparison requires vector 10/10/10/10, L=1 and K>=2")
        if model.spartan_dense or model.spartan_identity:
            raise ValueError("the sparse baseline/full comparison requires learned gates")
        objective = self.train.objective
        if (
            objective.state_regularizer != "sigreg"
            or objective.parameter_regularizer != "sigreg"
            or min(objective.state_regularizer_weight, objective.parameter_regularizer_weight) <= 0
        ):
            raise ValueError("both representations require explicitly weighted SIGReg")
        if (
            min(
                objective.state_sparsity_weight,
                objective.action_sparsity_weight,
                objective.parameter_sparsity_weight,
            )
            <= 0
        ):
            raise ValueError("specify positive state/action and full-model parameter penalties")
        if self.train.steps % self.train.accumulation_steps:
            raise ValueError("experiment steps must end at a complete accumulation boundary")
        if self.evaluation.rollout_steps > self.trajectory_length - model.context_len:
            raise ValueError("evaluation rollout does not fit after the parameter context")
        self.data_config(self.graphs[0])

    def data_config(self, graph: str) -> DGPConfig:
        """Resolve the training data definition without using model random seeds."""
        return DGPConfig(
            family="controlled_vector",
            parameter_graph=graph,
            root_seed=self.data_seed,
            observation_seed=self.observation_seed,
            num_episodes=self.train_episodes,
            trajectory_length=self.trajectory_length,
        )

    def training_config(self, condition: str, seed: int, out_dir: Path) -> TrainConfig:
        """Make paired conditions differ only in parameter-graph sparsity."""
        if condition not in {"baseline", "full"}:
            raise ValueError("condition must be baseline or full")
        objective = replace(
            self.train.objective,
            parameter_sparsity_weight=self.train.objective.parameter_sparsity_weight
            if condition == "full"
            else 0.0,
        )
        return replace(self.train, seed=seed, out_dir=str(out_dir), objective=objective)

    def definition(self) -> dict[str, object]:
        """Return scientific identity, excluding execution/logging destinations."""
        result = asdict(self)
        result.pop("output_dir")
        result.pop("wandb")
        result.pop("upload_evaluation_artifacts")
        training = cast(dict[str, object], result["train"])
        for field_name in ("out_dir", "device", "num_workers", "prefetch_factor", "log_every"):
            training.pop(field_name)
        return result


def _unlock(config: DictConfig | ListConfig) -> None:
    OmegaConf.set_readonly(config, False)
    for child in config.values() if isinstance(config, DictConfig) else config:
        if isinstance(child, DictConfig | ListConfig):
            _unlock(child)


def load_experiment_config(path: Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Resolve YAML and dotlist overrides against a strict nested schema."""
    schema = OmegaConf.structured(ExperimentConfig)
    _unlock(schema)
    merged = OmegaConf.merge(schema, OmegaConf.load(path), OmegaConf.from_dotlist(overrides or []))
    result = cast(ExperimentConfig, OmegaConf.to_object(merged))
    result.validate()
    return result
