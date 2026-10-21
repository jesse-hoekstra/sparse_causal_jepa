"""Typed, reproducible settings for registered vector DGP comparisons."""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, ListConfig, OmegaConf

from scjepa.data.types import DGPConfig
from scjepa.experiments.evaluation import EvaluationConfig
from scjepa.training import ObjectiveConfig, TrainConfig
from scjepa.training.factory import ModelConfig
from scjepa.training.lagrangian import GECOConfig
from scjepa.training.logging import WandbConfig


def _model() -> ModelConfig:
    return ModelConfig(
        observation_dim=10,
        state_dim=10,
        param_dim=10,
        action_dim=10,
        context_len=2,
        history_len=1,
        autoregressive=True,
    )


def _training() -> TrainConfig:
    return TrainConfig(
        steps=10_000,
        batch_size=64,
        checkpoint_every=250,
        log_every=25,
        objective=ObjectiveConfig(
            state_sparsity_weight=1.0,
            action_sparsity_weight=1.0,
            parameter_sparsity_weight=1.0,
            logit_weight=1e-4,
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
    The baseline sets that weight to zero and forces direct parameter-to-output
    gates open. Model seeds do not change the scientific data seed, observation
    matrix, or calibration/test splits. With paired episodes, dataset counts and
    batch sizes count independent parameter groups, each containing two trajectories.
    """

    name: str = "complementary-geco"
    output_dir: str = "outputs/complementary_geco"
    dense_reference_dir: str | None = None
    data_family: str = "controlled_vector"
    graphs: tuple[str, ...] = ("complementary",)
    seeds: tuple[int, ...] = (0,)
    train_episodes: int = 8192
    trajectory_length: int = 4
    episodes_per_parameter: int = 2
    data_seed: int = 0
    observation_seed: int = 1729
    validation_every: int = 1000
    upload_evaluation_artifacts: bool = False
    tau_multiplier: float = 1.0
    model: ModelConfig = field(default_factory=_model)
    train: TrainConfig = field(default_factory=_training)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def validate(self) -> None:
        """Reject ambiguous comparisons before creating outputs or tracking runs."""
        self.train.__post_init__()
        self.train.objective.__post_init__()
        self.train.geco.__post_init__()
        self.wandb.__post_init__()
        self.evaluation.__post_init__()
        if self.train.seed != 0:
            raise ValueError("set experiment seeds rather than the derived train.seed")
        if self.train.geco.enabled or self.train.geco.tau is not None:
            raise ValueError("GECO is enabled per sparse condition; tau comes from the dense run")
        if not 1.0 <= self.tau_multiplier < float("inf"):
            raise ValueError("tau_multiplier must be finite and at least one")
        if not self.name or not self.output_dir:
            raise ValueError("experiment name and output_dir must be nonempty")
        if self.data_family not in {
            "controlled_vector",
            "independent_gaussian",
            "simplified_gaussian",
        }:
            raise ValueError(
                "data_family must be controlled_vector, independent_gaussian or simplified_gaussian"
            )
        if self.dense_reference_dir is not None:
            if not self.dense_reference_dir.strip():
                raise ValueError("dense_reference_dir must be nonempty when supplied")
            if Path(self.dense_reference_dir).resolve() == Path(self.output_dir).resolve():
                raise ValueError("dense_reference_dir requires a different output_dir")
        if (
            not self.graphs
            or len(set(self.graphs)) != len(self.graphs)
            or any(graph not in {"dense", "complementary", "diagonal"} for graph in self.graphs)
        ):
            raise ValueError("graphs must be distinct named vector variants")
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(not 0 <= seed < 2**31 for seed in self.seeds)
        ):
            raise ValueError("seeds must be distinct nonnegative 31-bit integers")
        if self.train_episodes < self.train.batch_size or self.validation_every < 1:
            raise ValueError("need at least a full training batch and positive validation_every")
        model = self.model
        dimensions = (
            (6, 6, 6, 16) if self.data_family == "simplified_gaussian" else (10, 10, 10, 10)
        )
        if (
            model.observation_kind != "vector"
            or (model.observation_dim, model.state_dim, model.param_dim, model.action_dim)
            != dimensions
            or model.history_len != 1
            or not 2 <= model.context_len < self.trajectory_length
        ):
            raise ValueError(
                f"{self.data_family} controlled comparison requires vector "
                f"{'/'.join(str(value) for value in dimensions)}, L=1 and K>=2"
            )
        if (
            model.spartan_dense
            or model.spartan_identity
            or model.spartan_parameter_gates_open
            or model.spartan_action_gates_open
        ):
            raise ValueError("comparison sets gate modes per condition")
        if self.data_family in {"independent_gaussian", "simplified_gaussian"}:
            valid_context = (
                model.context_len == 3
                if self.data_family == "independent_gaussian"
                else model.context_len >= 3
            )
            if (
                tuple(self.graphs) != ("complementary",)
                or not valid_context
                or self.trajectory_length != model.context_len + 2
                or self.episodes_per_parameter != 2
                or not model.autoregressive
            ):
                context_rule = "K=3" if self.data_family == "independent_gaussian" else "K>=3"
                raise ValueError(
                    f"{self.data_family} comparison requires complementary graphs, "
                    f"{context_rule}, T=K+2, paired episodes and two-step autoregressive prediction"
                )
        objective = self.train.objective
        if objective.state_regularizer != "sigreg" or objective.parameter_regularizer != "sigreg":
            raise ValueError("use SIGReg for both representations; zero weights disable it")
        if (
            min(
                objective.state_sparsity_weight,
                objective.action_sparsity_weight,
                objective.parameter_sparsity_weight,
            )
            <= 0
        ):
            raise ValueError(
                f"{self.data_family}: specify positive state/action "
                "and full-model parameter penalties"
            )
        if objective.logit_weight <= 0:
            raise ValueError("SPARTAN comparison requires positive attention-logit regularization")
        if self.train.steps % self.train.accumulation_steps:
            raise ValueError("experiment steps must end at a complete accumulation boundary")
        self.data_config(self.graphs[0])

    @property
    def conditions(self) -> tuple[str, ...]:
        """Train only sparse conditions when reusing an existing dense reference."""
        return ("baseline", "full") if self.dense_reference_dir else ("dense", "baseline", "full")

    def data_config(self, graph: str) -> DGPConfig:
        """Resolve the training data definition without using model random seeds."""
        return DGPConfig(
            family=self.data_family,
            parameter_graph=graph,
            root_seed=self.data_seed,
            observation_seed=self.observation_seed,
            num_episodes=self.train_episodes,
            trajectory_length=self.trajectory_length,
            episodes_per_parameter=self.episodes_per_parameter,
        )

    def condition_model(self, condition: str) -> ModelConfig:
        """Derive dense reference, open parameter baseline, or fully learned gates."""
        if condition not in {"dense", "baseline", "full"}:
            raise ValueError("condition must be dense, baseline or full")
        return replace(
            self.model,
            spartan_dense=condition == "dense",
            spartan_parameter_gates_open=condition == "baseline",
            spartan_action_gates_open=False,
        )

    def training_config(
        self, condition: str, seed: int, out_dir: Path, *, tau: float | None = None
    ) -> TrainConfig:
        """Use one calibrated bound and add the parameter path penalty only for full."""
        self.condition_model(condition)
        objective = replace(
            self.train.objective,
            state_sparsity_weight=self.train.objective.state_sparsity_weight
            if condition != "dense"
            else 0.0,
            action_sparsity_weight=self.train.objective.action_sparsity_weight
            if condition != "dense"
            else 0.0,
            parameter_sparsity_weight=self.train.objective.parameter_sparsity_weight
            if condition == "full"
            else 0.0,
        )
        geco: GECOConfig = replace(self.train.geco, enabled=condition != "dense", tau=tau)
        return replace(
            self.train,
            seed=seed,
            out_dir=str(out_dir),
            objective=objective,
            geco=geco,
            lr_decay_steps=(
                self.train.lr_decay_steps or self.train.steps // self.train.accumulation_steps
            )
            if condition == "dense"
            else None,
        )

    def definition(self) -> dict[str, object]:
        """Return scientific identity, excluding execution/logging destinations."""
        result = asdict(self)
        result.pop("output_dir")
        result.pop("wandb")
        result.pop("upload_evaluation_artifacts")
        # Preserve definitions written before explicit family selection existed.
        # New families remain explicit and cannot share a legacy experiment identity.
        if self.data_family == "controlled_vector":
            result.pop("data_family")
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
