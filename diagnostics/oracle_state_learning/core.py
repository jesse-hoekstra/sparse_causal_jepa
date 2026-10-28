"""Raw-observation oracle controls; physical states remain evaluation-only.

All conditions share the native encoder initialization and attached two-step
SCJEPA forward. MLP has a prediction-only objective, never fabricated attention
graphs. Known raw parameters are supplied inputs, not training targets.
"""

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import torch
from torch import Tensor

from diagnostics.mlp_oracle_code_control.model import MLPPredictor, initialize_mlp
from scjepa.data.datasets import canonical_fingerprint
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.identity import training_tensors_fingerprint
from scjepa.data.types import DGPConfig, RendererConfig
from scjepa.experiments.oracle import OracleParameterEpisodes
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.loop import ObjectiveConfig, TrainingObjective
from scjepa.training.rng import preserve_rng

type Condition = Literal["dense", "baseline", "full", "mlp"]

TRAINING_KEYS = frozenset({"observations", "actions", "oracle_parameters", "episode_ids"})


@dataclass(frozen=True)
class LiveStateData:
    """Addressed raw data with a strict training interface and separate truth."""

    config: DGPConfig
    tensors: dict[str, Tensor]
    states: Tensor | None
    fingerprint: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Reject mixed training/evaluation payloads and malformed cached shapes."""
        if frozenset(self.tensors) != TRAINING_KEYS:
            raise ValueError("Training tensors must contain observations/actions/oracle/IDs only")
        groups, views = self.config.num_episodes, self.config.episodes_per_parameter
        expected = {
            "observations": (groups, views, 12, 6),
            "actions": (groups, views, 11, 16),
            "oracle_parameters": (groups, 6),
            "episode_ids": (groups,),
        }
        for name, shape in expected.items():
            value = self.tensors[name]
            if value.shape != shape or value.requires_grad or value.device.type != "cpu":
                raise ValueError(f"Invalid frozen CPU cache tensor {name}: expected {shape}")
            if value.dtype != (torch.int64 if name == "episode_ids" else torch.float32):
                raise ValueError(f"Invalid cache dtype for {name}")
        if self.config.split == "train" and self.states is not None:
            raise ValueError("Training cache must not store physical states")
        if self.states is not None and self.states.shape != expected["observations"]:
            raise ValueError("Evaluation physical states must align with observations")

    @property
    def views(self) -> int:
        """Return the number of trajectories sharing each supplied parameter vector."""
        return self.config.episodes_per_parameter

    def __len__(self) -> int:
        """Return the number of independent parameter groups."""
        return self.config.num_episodes

    def batch(self, indices: Tensor, device: torch.device | str) -> dict[str, Tensor]:
        """Transfer only explicitly authorized inputs; never expose physical states."""
        indices = indices.detach().cpu()
        return {key: value[indices].to(device) for key, value in self.tensors.items()}

    def payload(self) -> dict[str, Any]:
        """Serialize plain metadata and tensors, compatible with weights-only loading."""
        return {
            "config": asdict(self.config),
            "tensors": self.tensors,
            "states": self.states,
            "fingerprint": self.fingerprint,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LiveStateData":
        """Restore the same strict interface from a trusted diagnostic cache."""
        config = dict(payload["config"])
        config["renderer"] = RendererConfig(**config["renderer"])
        return cls(
            DGPConfig(**config),
            payload["tensors"],
            payload["states"],
            str(payload["fingerprint"]),
            dict(payload["metadata"]),
        )


def build_data(split: str, groups: int, seed: int = 0, views: int = 10) -> LiveStateData:
    """Cache unchanged registered R10/K10/T12 raw observations and supplied theta.

    ``calibration`` is the named convenience alias for the registered addressed
    ``evaluation_calibration`` split. States are omitted entirely on train.
    """
    if views != 10:
        raise ValueError("The fixed scientific protocol requires ten views per group")
    config = DGPConfig(
        family="simplified_gaussian",
        root_seed=seed,
        split="evaluation_calibration" if split == "calibration" else split,
        num_episodes=groups,
        episodes_per_parameter=views,
        trajectory_length=12,
        observation_seed=1729,
    )
    source = build_dgp_dataset(config)
    tensors: dict[str, list[Tensor]] = {key: [] for key in TRAINING_KEYS}
    states: list[Tensor] = []
    for index in range(groups):
        episode = source.process.generate(index)
        tensors["observations"].append(episode.observations)
        tensors["actions"].append(episode.actions)
        tensors["oracle_parameters"].append(
            episode.ground_truth["parameters"].detach().to(episode.observations)
        )
        tensors["episode_ids"].append(torch.tensor(episode.episode_id, dtype=torch.int64))
        if config.split != "train":
            states.append(episode.ground_truth["states"].detach().cpu())
    stacked = {key: torch.stack(parts) for key, parts in tensors.items()}
    metadata: dict[str, Any] = {
        "schema": "oracle-state-learning-raw-data-v1",
        "dataset_fingerprint": OracleParameterEpisodes(source).fingerprint,
        "training_tensor_fingerprint": training_tensors_fingerprint(stacked),
        "parameter_source": "supplied_raw_ground_truth",
        "physical_states_training": False,
        "views_per_group": views,
        "context_length": 10,
        "trajectory_length": 12,
    }
    return LiveStateData(
        config,
        stacked,
        torch.stack(states) if states else None,
        canonical_fingerprint(metadata),
        metadata,
    )


def build_control(condition: Condition, seed: int) -> SCJEPA:
    """Initialize native oracle state learning with identical encoder/SPARTAN tensors."""
    if condition not in {"dense", "baseline", "full", "mlp"}:
        raise ValueError("Unknown oracle state-learning condition")
    config = ModelConfig(
        observation_dim=6,
        state_dim=6,
        param_dim=6,
        action_dim=16,
        context_len=10,
        history_len=1,
        encoder_hidden_dims=(64, 64),
        parameter_encoder_kind="transition",
        parameter_source="oracle",
        spartan_layers=2,
        spartan_embed_dim=64,
        spartan_mlp_hidden=64,
        spartan_mlp_layers=3,
        spartan_dense=condition == "dense",
        spartan_parameter_gates_open=condition == "baseline",
        ema_decay=0.996,
        autoregressive=True,
    )
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(seed)
        model = build_model(config)
        if condition == "mlp":
            initialize_mlp(model, seed)
    return model


def objective_config(condition: Condition, logit_weight: float = 1e-4) -> ObjectiveConfig:
    """Keep state SIGReg common; sparsify only the declared SPARTAN graph blocks."""
    if condition not in {"dense", "baseline", "full", "mlp"}:
        raise ValueError("Unknown oracle state-learning condition")
    return ObjectiveConfig(
        state_sparsity_weight=float(condition in {"baseline", "full"}),
        action_sparsity_weight=float(condition in {"baseline", "full"}),
        parameter_sparsity_weight=float(condition == "full"),
        logit_weight=0.0 if condition == "mlp" else logit_weight,
        state_regularizer="sigreg",
        state_regularizer_weight=0.4,
        regularizer_num_projections=256,
    )


def forward_loss(
    model: SCJEPA,
    batch: Mapping[str, Tensor],
    objective: TrainingObjective,
    *,
    forward_seed: int | None = None,
) -> tuple[Tensor, dict[str, Tensor], ModelOutput]:
    """Run exactly two attached predictions, excluding all unauthorized labels."""
    if frozenset(batch) != TRAINING_KEYS:
        raise ValueError("Training batch accepts observations/actions/oracle_parameters/IDs only")
    if (
        objective.config.parameter_regularizer_weight
        or objective.config.parameter_consistency_weight
    ):
        raise ValueError("Supplied oracle parameters must have no parameter-learning penalties")
    observations = batch["observations"]
    if observations.ndim != 4 or observations.shape[1:] != (10, 12, 6):
        raise ValueError("This control requires grouped R10/T12 six-coordinate observations")
    context = preserve_rng(observations.device) if forward_seed is not None else nullcontext()
    with context:
        if forward_seed is not None:
            torch.default_generator.manual_seed(forward_seed)
            if observations.device.type == "cuda":
                with torch.cuda.device(observations.device):
                    torch.cuda.manual_seed(forward_seed)
        output = cast(
            ModelOutput,
            model(observations, batch["actions"], oracle_parameters=batch["oracle_parameters"]),
        )
    if output.prediction_start != 9 or len(output.dynamics) != 2:
        raise RuntimeError("The oracle comparison requires exactly two attached forecast steps")
    if not isinstance(model.predictor, MLPPredictor):
        loss, terms = objective.forward(output)
        return loss, terms, output
    config = objective.config
    if any(
        (
            config.state_sparsity_weight,
            config.action_sparsity_weight,
            config.parameter_sparsity_weight,
            config.logit_weight,
        )
    ):
        raise ValueError("The global MLP has no attention logits or graph penalties")
    prediction = (output.predictions - output.targets).square().mean()
    terms = {"prediction": prediction}
    base = prediction
    if config.state_regularizer_weight:
        states = output.online_states.flatten(1, 2).transpose(0, 1)
        terms["state_regularizer"] = objective.state_regularizer(states)
        base = base + config.state_regularizer_weight * terms["state_regularizer"]
    terms.update(base=base, sparsity=base.new_zeros(()))
    return base, terms, output


@dataclass(frozen=True)
class EvaluationResult:
    """Held-out metrics and group-preserving samples for calibration-fitted MCC."""

    metrics: dict[str, float]
    samples: dict[str, Tensor]


def _state_health(states: Tensor, prefix: str) -> dict[str, float]:
    rows = states.flatten(0, -2).double()
    centered = rows - rows.mean(0)
    covariance = centered.T @ centered / len(rows)
    eigenvalues = cast(Tensor, torch.linalg.eigvalsh(covariance)).clamp_min(0)  # pyright: ignore[reportUnknownMemberType]
    variances = covariance.diagonal()
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-15)
    effective_rank = (-probabilities * probabilities.clamp_min(1e-15).log()).sum().exp()
    return {
        f"{prefix}/variance_mean": float(variances.mean()),
        f"{prefix}/variance_min": float(variances.min()),
        f"{prefix}/covariance_eigenvalue_min": float(eigenvalues.min()),
        f"{prefix}/covariance_eigenvalue_max": float(eigenvalues.max()),
        f"{prefix}/effective_rank": float(effective_rank) if bool(eigenvalues.sum() > 0) else 0.0,
        f"{prefix}/mean_abs": float(rows.mean(0).abs().mean()),
    }


@torch.no_grad()
def evaluate(
    model: SCJEPA,
    data: LiveStateData,
    batch_size: int,
    device: torch.device | str,
    objective: TrainingObjective,
) -> EvaluationResult:
    """Collect deterministic held-out encodings without fitting or selecting on them.

    Online/target MCC assignments must be fitted on calibration samples by the
    caller and frozen when scoring validation/test. All sample tensors retain
    their independent group axis. Routing is descriptive and SPARTAN-only.
    """
    if data.states is None or batch_size < 2:
        raise ValueError("Evaluation requires held-out physical states and batch_size >= 2")
    device = torch.device(device)
    modes = [(module, module.training) for module in (*model.modules(), *objective.modules())]
    parts: dict[str, list[Tensor]] = {
        key: [] for key in ("online_states", "target_states", "predictions", "targets")
    }
    totals: dict[str, float] = {}
    try:
        with preserve_rng(device):
            torch.default_generator.manual_seed(42)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(42)
            model.eval()
            objective.eval()
            for start in range(0, len(data), batch_size):
                indices = torch.arange(start, min(start + batch_size, len(data)))
                batch = data.batch(indices, device)
                loss, terms, output = forward_loss(model, batch, objective, forward_seed=42 + start)
                for name, value in dict(terms, optimized_total=loss).items():
                    totals[name] = totals.get(name, 0.0) + float(value) * len(indices)
                parts["online_states"].append(output.online_states.detach().cpu())
                encoded = model.target_encoder.encode_sequence(batch["observations"].flatten(0, 1))
                parts["target_states"].append(encoded.reshape(len(indices), 10, 12, 6).cpu())
                parts["predictions"].append(output.predictions.cpu())
                parts["targets"].append(output.targets.cpu())
                if not isinstance(model.predictor, MLPPredictor):
                    for name in ("state", "action", "parameter"):
                        routed = torch.stack(
                            [
                                getattr(item.graphs.aggregate_lags(), name) > 0
                                for item in output.dynamics
                            ],
                            dim=1,
                        ).float()
                        grouped = routed.reshape(len(indices), 10, 2, *routed.shape[-2:])
                        parts.setdefault(f"graph_{name}", []).append(grouped.mean((1, 2)).cpu())
    finally:
        for module, training in modes:
            module.training = training
    samples = {key: torch.cat(values) for key, values in parts.items()}
    samples["states"] = data.states
    samples["group_ids"] = data.tensors["episode_ids"]
    metrics = {name: value / len(data) for name, value in totals.items()}
    for name in ("online_states", "target_states"):
        metrics.update(_state_health(samples[name], name.removesuffix("_states")))
    targets = samples["targets"].flatten(0, -2).double()
    variance = targets.var(0, correction=0).mean()
    metrics["forecast/target_variance"] = float(variance)
    metrics["forecast/normalized_mse"] = metrics["prediction"] / max(float(variance), 1e-15)
    coordinate_errors = (
        (samples["predictions"] - samples["targets"]).flatten(0, -2).double().square().mean(0)
    )
    normalized = coordinate_errors / targets.var(0, correction=0).clamp_min(1e-15)
    for coordinate in range(6):
        metrics[f"forecast/coordinate_{coordinate + 1}_normalized_mse"] = float(
            normalized[coordinate]
        )
    for horizon in range(2):
        error = samples["predictions"][:, :, horizon] - samples["targets"][:, :, horizon]
        metrics[f"forecast/horizon_{horizon + 1}_mse"] = float(error.square().mean())
    for name in ("state", "action", "parameter"):
        if f"graph_{name}" in samples:
            metrics[f"routing/{name}_support_mean"] = float(samples[f"graph_{name}"].mean())
    return EvaluationResult(metrics, samples)
