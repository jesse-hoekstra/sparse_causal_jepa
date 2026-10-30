"""Oracle forecast objective and group-preserving held-out encodings."""

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import torch
from torch import Tensor

from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.loop import ObjectiveConfig, TrainingObjective
from scjepa.training.rng import preserve_rng

type Condition = Literal["dense", "baseline", "full"]
TRAINING_KEYS = frozenset({"observations", "actions", "oracle_parameters", "episode_ids"})


class LiveStateData(Protocol):
    """Read-only interface shared by the experiment's authenticated cache views."""

    states: Tensor | None
    tensors: dict[str, Tensor]
    metadata: dict[str, Any]

    def __len__(self) -> int:
        """Count independent parameter groups."""
        ...

    def batch(self, indices: Tensor, device: torch.device | str) -> dict[str, Tensor]:
        """Transfer only the declared input tensors."""
        ...


def objective_config(condition: Condition, logit_weight: float = 1e-4) -> ObjectiveConfig:
    """Keep state SIGReg common; sparsify only the declared predictor graph blocks."""
    if condition not in {"dense", "baseline", "full"}:
        raise ValueError("Unknown oracle state-learning condition")
    return ObjectiveConfig(
        state_sparsity_weight=float(condition in {"baseline", "full"}),
        action_sparsity_weight=float(condition in {"baseline", "full"}),
        parameter_sparsity_weight=float(condition == "full"),
        logit_weight=logit_weight,
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
    loss, terms = objective.forward(output)
    return loss, terms, output


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
    their independent group axis. Routing describes sampled or thresholded predictor gates.
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
