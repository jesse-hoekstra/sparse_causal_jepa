"""Masked MLP state learning with a learned parameter encoder (full Experiment 1 setting).

Adapted from the masked-MLP oracle runners (``diagnostics/masked_mlp_oracle``). The
only scientific change is the parameter source: a transition-set parameter encoder
infers a code from each episode's first ten online state encodings and nine
actions, and that attached code predicts another episode of the same parameter
group (cyclic donor/recipient routing). True theta never enters training; it is
used only to evaluate the code. There is no parameter regularizer and no
consistency term. Everything else is unchanged: the authenticated Paley24 cache,
the chart-equivariant masked MLP, the 6-64-64-6 state encoder with Gaussian head,
EMA .996, state SIGReg .4, AdamW .001, clip 1, B16/R10/K10/T12/L1 and the attached
two-step forecast from online state 9.

Protocol: fresh dense arms (all masks open) for 100k updates, by default with the
audited acquisition recipe (twelve-dimensional code, encoder gains, shared-feature
output rows; see ``acquisition``); ``project`` maps each twelve-code dense endpoint
to a six-code model through TRAIN shared modes; ``calibrate`` fixes ``tau`` as 1.1
times the seed-zero six-code dense calibration prediction MSE; baseline and full
arms start from their seed's six-code dense endpoint (encoders, parameter encoder,
trunk, readouts; fresh mask logits, AdamW, GECO, sampler) and run 100k or 200k
sparse updates with GECO on the normalized prediction loss.
"""

import argparse
import json
import math
import random
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment  # pyright: ignore[reportUnknownVariableType]
from torch import Tensor, nn

from diagnostics.balanced_action_oracle.data import StateData, load_data
from diagnostics.isotropic_oracle.dgp import graphs as true_graphs
from diagnostics.masked_mlp_learned.acquisition import (
    apply_encoder_gains,
    initialize_shared_output,
    project_code,
)
from diagnostics.masked_mlp_oracle import run as oracle
from diagnostics.masked_mlp_oracle.model import (
    MaskedTrunkPredictor,
    _gaussian_,  # pyright: ignore[reportPrivateUsage]
    trunk_width,
)
from diagnostics.oracle_state_learning.core import EvaluationResult, objective_config
from diagnostics.oracle_state_learning.data import read_json, save_torch, sha
from diagnostics.oracle_state_learning.run import recovery_metrics
from diagnostics.oracle_state_predictor.run import GroupSampler, resolve_device
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.experiments.storage import file_lock, write_json
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.models.spartan import Spartan
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.lagrangian import GECOConfig
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import TrainingObjective, seed_everything
from scjepa.training.rng import preserve_rng

SCHEMA = "masked-mlp-learned-state-learning-v2"
NORMALIZED_MSE_CAP = 0.15
SEEDS = (0, 1, 2, 3, 4)
SPARSE_CODE_DIM = 6
CODE_DIMS = (6, 12)
PAIRS = ((0, 3), (1, 4), (2, 5))
INPUT_KEYS = frozenset({"observations", "actions"})
MASK_LOGITS = frozenset(f"predictor.{name}_logits" for name in ("state", "action", "parameter"))
type Condition = Literal["dense", "baseline", "full"]
type Profile = Literal["exploratory", "smoke"]
type Initializer = Literal["native", "gain", "shared_feature"]
INITIALIZERS: tuple[Initializer, ...] = ("native", "gain", "shared_feature")


def code_identity() -> str:
    """Bind the masked-MLP oracle identity (cache, model, helpers) plus this runner."""
    files = {
        name: sha(Path(__file__).with_name(name))
        for name in ("__init__.py", "acquisition.py", "run.py")
    }
    return canonical_fingerprint({"oracle_code_identity": oracle.code_identity(), "files": files})


def build_control(
    condition: Condition,
    seed: int,
    code_dim: int = SPARSE_CODE_DIM,
    initializer: Initializer = "native",
) -> SCJEPA:
    """Pair encoder, parameter encoder, trunk and readouts across conditions for one seed.

    ``gain`` and ``shared_feature`` scale the parameter encoder; the data-dependent
    shared-feature output rows are fitted afterwards by ``initialize_shared_output``.
    """
    if code_dim not in CODE_DIMS or initializer not in INITIALIZERS:
        raise ValueError("unknown code dimension or initializer")
    config = ModelConfig(
        observation_dim=6,
        state_dim=6,
        param_dim=code_dim,
        action_dim=24,
        context_len=10,
        history_len=1,
        encoder_hidden_dims=(64, 64),
        parameter_encoder_kind="transition",
        parameter_source="learned",
        spartan_layers=2,
        spartan_embed_dim=64,
        spartan_mlp_hidden=64,
        spartan_mlp_layers=3,
        spartan_dense=True,
        ema_decay=0.996,
        autoregressive=True,
    )
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(seed)
        model = build_model(config)
        torch.default_generator.manual_seed(seed)
        head = model.online_encoder.head
        encoder = model.parameter_encoder
        if not isinstance(head, nn.Linear) or not isinstance(encoder, TransitionParameterEncoder):
            raise TypeError("expected a linear state head and a transition parameter encoder")
        first = encoder.transition[0]
        if not isinstance(first, nn.Linear):
            raise TypeError("expected a linear first transition layer")
        # Gaussian laws keep the initialization invariant to re-charting of states.
        for layer in (head, first):
            _gaussian_(layer.weight, layer.in_features)
            _gaussian_(layer.bias, layer.in_features)
        model.target_encoder.load_state_dict(model.online_encoder.state_dict())
        if initializer != "native":
            apply_encoder_gains(model)
        if condition not in {"dense", "baseline", "full"}:
            raise ValueError("Unknown masked MLP condition")
        sparse = condition != "dense"
        predictor = MaskedTrunkPredictor(
            param_dim=code_dim,
            learn_state=sparse,
            learn_action=sparse,
            learn_parameter=condition == "full",
        )
    model.predictor = cast(Spartan, predictor)
    return model


def _predictor(model: SCJEPA) -> MaskedTrunkPredictor:
    predictor = cast(nn.Module, model.predictor)
    if not isinstance(predictor, MaskedTrunkPredictor):
        raise TypeError("this runner trains only the masked MLP predictor")
    return predictor


def forward_loss(
    model: SCJEPA,
    batch: Mapping[str, Tensor],
    objective: TrainingObjective,
    *,
    forward_seed: int | None = None,
) -> tuple[Tensor, dict[str, Tensor], ModelOutput]:
    """Run two attached predictions from observations and actions only; never true theta."""
    observations = batch["observations"]
    if observations.ndim != 4 or observations.shape[1:] != (10, 12, 6):
        raise ValueError("This control requires grouped R10/T12 six-coordinate observations")
    if (
        objective.config.parameter_regularizer_weight
        or objective.config.parameter_consistency_weight
    ):
        raise ValueError("The declared objective has no parameter regularizer or consistency term")
    context = preserve_rng(observations.device) if forward_seed is not None else nullcontext()
    with context:
        if forward_seed is not None:
            torch.default_generator.manual_seed(forward_seed)
            if observations.device.type == "cuda":
                with torch.cuda.device(observations.device):
                    torch.cuda.manual_seed(forward_seed)
        output = cast(ModelOutput, model(observations, batch["actions"]))
    if output.prediction_start != 9 or len(output.dynamics) != 2:
        raise RuntimeError("The comparison requires exactly two attached forecast steps")
    loss, terms = objective.forward(output)
    return loss, terms, output


class _ShuffledCodes(nn.Module):
    """Wrap the parameter encoder so every group receives another group's codes."""

    def __init__(self, inner: nn.Module, views: int) -> None:
        super().__init__()
        self.inner = inner
        self.views = views

    def forward(self, states: Tensor, actions: Tensor | None = None) -> Tensor:
        codes = cast(Tensor, self.inner(states, actions))
        grouped = codes.reshape(-1, self.views, codes.shape[-1])
        return grouped.roll(1, dims=0).flatten(0, 1)


@contextmanager
def shuffled_codes(model: SCJEPA, views: int) -> Generator[None]:
    """Temporarily route codes across parameter groups (predictive-use diagnostic)."""
    original = model.parameter_encoder
    model.parameter_encoder = cast(Any, _ShuffledCodes(original, views))
    try:
        yield
    finally:
        model.parameter_encoder = original


def _state_health(states: Tensor, prefix: str) -> dict[str, float]:
    rows = states.flatten(0, -2).double()
    centered = rows - rows.mean(0)
    covariance = centered.T @ centered / len(rows)
    eigenvalues = cast(Tensor, torch.linalg.eigvalsh(covariance)).clamp_min(0)  # pyright: ignore[reportUnknownMemberType]
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-15)
    effective_rank = (-probabilities * probabilities.clamp_min(1e-15).log()).sum().exp()
    return {
        f"{prefix}/variance_mean": float(covariance.diagonal().mean()),
        f"{prefix}/variance_min": float(covariance.diagonal().min()),
        f"{prefix}/effective_rank": float(effective_rank) if bool(eigenvalues.sum() > 0) else 0.0,
    }


@torch.no_grad()
def evaluate(
    model: SCJEPA,
    data: StateData,
    batch_size: int,
    device: torch.device,
    objective: TrainingObjective,
) -> EvaluationResult:
    """Held-out encodings, codes and forecasts; factual and group-shuffled codes."""
    if data.states is None or batch_size < 2:
        raise ValueError("Evaluation requires held-out physical states and batch_size >= 2")
    modes = [(module, module.training) for module in (*model.modules(), *objective.modules())]
    parts: dict[str, list[Tensor]] = {
        key: []
        for key in ("online_states", "target_states", "predictions", "targets", "codes", "theta")
    }
    totals: dict[str, float] = {}
    shuffled_error = 0.0
    try:
        with preserve_rng(device):
            torch.default_generator.manual_seed(42)
            model.eval()
            objective.eval()
            for start in range(0, len(data), batch_size):
                indices = torch.arange(start, min(start + batch_size, len(data)))
                if len(indices) < 2:
                    break
                batch = data.batch(indices, device)
                loss, terms, output = forward_loss(model, batch, objective, forward_seed=42 + start)
                for name, value in dict(terms, optimized_total=loss).items():
                    totals[name] = totals.get(name, 0.0) + float(value) * len(indices)
                with shuffled_codes(model, batch["observations"].shape[1]):
                    _, shuffled_terms, _ = forward_loss(
                        model, batch, objective, forward_seed=42 + start
                    )
                shuffled_error += float(shuffled_terms["prediction"]) * len(indices)
                parts["online_states"].append(output.online_states.cpu())
                encoded = model.target_encoder.encode_sequence(batch["observations"].flatten(0, 1))
                parts["target_states"].append(encoded.reshape(len(indices), 10, 12, 6).cpu())
                parts["predictions"].append(output.predictions.cpu())
                parts["targets"].append(output.targets.cpu())
                parts["codes"].append(output.parameters.cpu())
                parts["theta"].append(batch["oracle_parameters"].cpu())
    finally:
        for module, training in modes:
            module.training = training
    samples = {key: torch.cat(values) for key, values in parts.items()}
    groups = len(samples["codes"])
    samples["states"] = data.states[:groups]
    metrics = {name: value / groups for name, value in totals.items()}
    for name in ("online_states", "target_states"):
        metrics.update(_state_health(samples[name], name.removesuffix("_states")))
    targets = samples["targets"].flatten(0, -2).double()
    variance = float(targets.var(0, correction=0).mean())
    metrics["forecast/target_variance"] = variance
    metrics["forecast/normalized_mse"] = metrics["prediction"] / max(variance, 1e-15)
    metrics["forecast/shuffled_code_normalized_mse"] = (
        shuffled_error / groups / max(variance, 1e-15)
    )
    metrics["forecast/code_use_ratio"] = metrics["forecast/shuffled_code_normalized_mse"] / max(
        metrics["forecast/normalized_mse"], 1e-15
    )
    return EvaluationResult(metrics, samples)


def _design(values: Tensor, family: str) -> Tensor:
    columns = [values]
    if family == "cubic":
        width = values.shape[1]
        columns += [
            (values[:, i] * values[:, j])[:, None] for i in range(width) for j in range(i, width)
        ]
        columns += [
            (values[:, i] * values[:, j] * values[:, k])[:, None]
            for i in range(width)
            for j in range(i, width)
            for k in range(j, width)
        ]
    return torch.cat([*columns, values.new_ones(len(values), 1)], 1)


def _ridge(design: Tensor, target: Tensor, strength: float = 1e-3) -> Tensor:
    gram = design.T @ design / len(design)
    identity = torch.eye(gram.shape[0], dtype=gram.dtype)
    identity[-1, -1] = 0
    return cast(
        Tensor,
        torch.linalg.solve(gram + strength * identity, design.T @ target / len(design)),  # pyright: ignore[reportUnknownMemberType]
    )


def _r2(prediction: Tensor, target: Tensor) -> Tensor:
    residual = (target - prediction).square().sum(0)
    total = (target - target.mean(0)).square().sum(0).clamp_min(1e-15)
    return 1 - residual / total


def code_metrics(calibration: EvaluationResult, result: EvaluationResult) -> dict[str, float]:
    """Whole-code theta retention, code MCC and parameter-pair purity.

    Probes fit on the first half of the calibration groups; the linear or cubic
    family is selected per parameter on the second half; scores use the held-out
    split. Codes are whitened with calibration statistics. Complementary graphs
    identify parameters only up to mixing within the pairs (1,4), (2,5), (3,6).
    """

    def rows(sample: EvaluationResult) -> tuple[Tensor, Tensor]:
        codes = sample.samples["codes"].double()
        theta = sample.samples["theta"].double()[:, None].expand(-1, codes.shape[1], -1)
        return codes.flatten(0, 1), theta.flatten(0, 1)

    codes, theta = rows(calibration)
    scored_codes, scored_theta = rows(result)
    mean, std = codes.mean(0), codes.std(0).clamp_min(1e-10)
    codes, scored_codes = (codes - mean) / std, (scored_codes - mean) / std
    half = len(codes) // 2
    metrics: dict[str, float] = {}
    best: list[float] = []
    selections: dict[str, tuple[Tensor, Tensor]] = {}
    for family in ("linear", "cubic"):
        weights = _ridge(_design(codes[:half], family), theta[:half])
        selection = _r2(_design(codes[half:], family) @ weights, theta[half:])
        scored = _r2(_design(scored_codes, family) @ weights, scored_theta)
        selections[family] = (selection, scored)
    for k in range(6):
        family = max(selections, key=lambda name: float(selections[name][0][k]))
        score = float(selections[family][1][k])
        metrics[f"code_r2/theta_{k + 1}"] = score
        best.append(score)
    metrics["code_r2_min"] = min(best)
    metrics["code_r2_mean"] = sum(best) / len(best)
    correlation = _abs_correlation(scored_codes, scored_theta)
    rows_, cols = cast(tuple[Any, Any], linear_sum_assignment(-correlation.numpy()))
    metrics["code_mcc"] = float(correlation[rows_, cols].mean())
    purities: list[float] = []
    for j in range(codes.shape[1]):
        scores: list[float] = []
        for pair in PAIRS:
            weights = _ridge(_design(theta[:, list(pair)], "linear"), codes[:, j : j + 1])
            prediction = _design(scored_theta[:, list(pair)], "linear") @ weights
            scores.append(float(_r2(prediction, scored_codes[:, j : j + 1])[0]))
        purities.append(max(scores))
    metrics["code_pair_purity"] = sum(purities) / len(purities)
    return metrics


def _abs_correlation(left: Tensor, right: Tensor) -> Tensor:
    left = left - left.mean(0)
    right = right - right.mean(0)
    left = left / left.square().sum(0).sqrt().clamp_min(1e-15)
    right = right / right.square().sum(0).sqrt().clamp_min(1e-15)
    return (left.T @ right).abs()


@dataclass(frozen=True)
class TrainConfig:
    """One fixed-budget arm. Sparse arms start from a dense endpoint and read a bound."""

    output_dir: str
    data_dir: str
    condition: Condition
    profile: Profile = "exploratory"
    seed: int = 0
    steps: int = 100_000
    batch_size: int = 16
    lr: float = 0.001
    grad_clip: float = 1.0
    calibration_dir: str | None = None
    init_dir: str | None = None
    geco_step_size: float = 0.1
    code_dim: int | None = None
    initializer: Initializer | None = None
    device: str = "cpu"
    threads: int = 1
    log_every: int = 100
    evaluate_every: int = 5000
    checkpoint_every: int = 5000
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None
    wandb_group: str = "masked-mlp-learned"

    def __post_init__(self) -> None:
        """Freeze scientific settings and the dense/sparse protocol."""
        if self.profile not in {"exploratory", "smoke"}:
            raise ValueError("profile must be exploratory or smoke")
        if self.condition not in {"dense", "baseline", "full"}:
            raise ValueError("unknown masked MLP condition")
        if self.seed not in SEEDS:
            raise ValueError(f"training seed must be one of {SEEDS}")
        sparse = self.condition != "dense"
        if sparse != (self.calibration_dir is not None) or sparse != (self.init_dir is not None):
            raise ValueError("exactly the sparse arms need a calibration and a dense endpoint")
        if sparse and (self.code_dim is not None or self.initializer is not None):
            raise ValueError("sparse arms inherit a six-code model from their dense endpoint")
        if self.code_dim is not None and self.code_dim not in CODE_DIMS:
            raise ValueError(f"code dimension must be one of {CODE_DIMS}")
        if self.initializer is not None and self.initializer not in INITIALIZERS:
            raise ValueError(f"initializer must be one of {INITIALIZERS}")
        if self.profile == "exploratory":
            if (self.batch_size, self.lr, self.grad_clip) != (16, 0.001, 1.0):
                raise ValueError("exploratory optimizer settings are fixed")
            allowed = (100_000, 200_000) if sparse else (100_000,)
            if self.steps not in allowed:
                raise ValueError(f"exploratory budget must be one of {allowed}")
        for name in (
            "steps",
            "batch_size",
            "threads",
            "log_every",
            "evaluate_every",
            "checkpoint_every",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size < 2:
            raise ValueError("SIGReg requires at least two independent groups")
        values = (self.lr, self.grad_clip, self.geco_step_size)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("learning rate, clipping and GECO step must be finite and positive")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)

    @property
    def model_code_dim(self) -> int:
        """Dense arms default to the audited twelve-code interface; sparse arms use six."""
        if self.condition != "dense":
            return SPARSE_CODE_DIM
        return 12 if self.code_dim is None else self.code_dim

    @property
    def model_initializer(self) -> Initializer:
        """Dense arms default to the audited recipe; sparse arms load every weight."""
        if self.condition != "dense":
            return "native"
        return "shared_feature" if self.initializer is None else self.initializer


def verify_completion(
    directory: Path, *, kind: str | None = None, require_current_source: bool = False
) -> dict[str, Any]:
    """Authenticate all declared output files without requiring the original host."""
    result = read_json(directory / "completion.json")
    if result.get("schema") != SCHEMA:
        raise ValueError("completion schema differs")
    if kind is not None and result.get("kind") != kind:
        raise ValueError("completion stage differs")
    if require_current_source and result.get("code_identity") != code_identity():
        raise ValueError("completion numerical source differs")
    hashes = result.get("hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("completion lacks artifact hashes")
    for name, expected in cast(dict[str, str], hashes).items():
        if Path(name).name != name or sha(directory / name) != expected:
            raise ValueError(f"artifact hash differs: {name}")
    return result


def _fresh_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def _state_identity(module: nn.Module, *, exclude_logits: bool = False) -> str:
    state = cast(dict[str, object], module.state_dict())
    return canonical_fingerprint(
        {
            key: tensor_fingerprint(value) if isinstance(value, Tensor) else value
            for key, value in state.items()
            if not (exclude_logits and key.endswith("_logits"))
        }
    )


def _initial_identities(model: SCJEPA) -> dict[str, str]:
    return {
        "initial_encoder_identity": _state_identity(model.online_encoder),
        "initial_parameter_encoder_identity": _state_identity(model.parameter_encoder),
        "initial_trunk_identity": _state_identity(model.predictor, exclude_logits=True),
        "initial_model_identity": _state_identity(model),
    }


def _six_code_dense(directory: Path, profile: str) -> dict[str, Any]:
    """Authenticate a six-code dense endpoint: a six-code fit or a projected twelve-code fit."""
    completion = verify_completion(directory, require_current_source=True)
    if completion["kind"] not in {"train", "projection"}:
        raise ValueError("dense endpoint must be a training or projection completion")
    if completion["condition"] != "dense" or completion["profile"] != profile:
        raise ValueError("reference must be a matching-profile dense endpoint")
    if completion["code_dim"] != SPARSE_CODE_DIM:
        raise ValueError("project the twelve-code dense endpoint to six codes first")
    return completion


@dataclass(frozen=True)
class CalibrateConfig:
    """Fix the prediction bound from the one seed-zero dense endpoint."""

    output_dir: str
    dense_dir: str
    tau_factor: float = 1.1
    profile: Profile = "exploratory"

    def __post_init__(self) -> None:
        """Require a positive prespecified bound multiplier."""
        if self.profile not in {"exploratory", "smoke"}:
            raise ValueError("profile must be exploratory or smoke")
        if not math.isfinite(self.tau_factor) or self.tau_factor <= 0:
            raise ValueError("tau_factor must be finite and positive")
        if self.profile == "exploratory" and self.tau_factor != 1.1:
            raise ValueError("exploratory tau_factor is prespecified as 1.1")


def calibrate(config: CalibrateConfig) -> dict[str, Any]:
    """Create an immutable prediction bound from calibration loss, never test or MCC."""
    output = Path(config.output_dir).resolve()
    with file_lock(output.parent / f".{output.name}.lock"):
        _fresh_output(output)
        directory = Path(config.dense_dir).resolve()
        completion = _six_code_dense(directory, config.profile)
        if completion["seed"] != 0:
            raise ValueError("calibration requires the seed-zero dense reference")
        metrics = read_json(directory / "calibration.json")["metrics"]
        prediction = float(metrics["prediction"])
        if not math.isfinite(prediction) or prediction <= 0:
            raise ValueError("dense calibration prediction loss must be finite and positive")
        record: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": "calibration",
            "profile": config.profile,
            "code_identity": code_identity(),
            "data_identity": completion["data_identity"],
            "tau_prediction": config.tau_factor * prediction,
            "tau_factor": config.tau_factor,
            "reference": {
                "directory": str(directory),
                "kind": completion["kind"],
                "identity": completion["identity"],
                "completion_sha256": sha(directory / "completion.json"),
                "prediction_loss": prediction,
                "steps": completion["steps"],
            },
            "policy": (
                "tau_factor times six-code dense calibration prediction MSE; no weights transferred"
            ),
        }
        record["identity"] = canonical_fingerprint(record)
        write_json(output / "calibration.json", record)
        completion = dict(record, hashes={"calibration.json": sha(output / "calibration.json")})
        write_json(output / "completion.json", completion)
        return completion


def _calibration(config: TrainConfig, data_identity: str) -> dict[str, Any] | None:
    if config.calibration_dir is None:
        return None
    directory = Path(config.calibration_dir)
    completion = verify_completion(directory, kind="calibration", require_current_source=True)
    record = read_json(directory / "calibration.json")
    if record["identity"] != completion["identity"] or record["profile"] != config.profile:
        raise ValueError("calibration identity or profile differs")
    if record["data_identity"] != data_identity:
        raise ValueError("calibration data differs")
    tau = float(record["tau_prediction"])
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("calibrated bound must be finite and positive")
    return record


def _dense_endpoint(config: TrainConfig, data_identity: str) -> dict[str, Any] | None:
    if config.init_dir is None:
        return None
    directory = Path(config.init_dir).resolve()
    completion = _six_code_dense(directory, config.profile)
    if completion["seed"] != config.seed:
        raise ValueError("initialization must be the seed-matched dense endpoint")
    if completion["data_identity"] != data_identity:
        raise ValueError("dense endpoint data identity differs")
    checkpoint = cast(
        dict[str, Any], torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    )
    if checkpoint.get("identity") != completion["identity"]:
        raise ValueError("dense checkpoint does not belong to its completion record")
    if int(checkpoint["step"]) != int(completion["steps"]):
        raise ValueError("dense checkpoint is not the final endpoint")
    return {
        "record": {
            "directory": str(directory),
            "kind": completion["kind"],
            "identity": completion["identity"],
            "completion_sha256": sha(directory / "completion.json"),
            "checkpoint_sha256": sha(directory / "last.pt"),
            "seed": completion["seed"],
            "steps": completion["steps"],
        },
        "state": checkpoint["model"],
    }


def _load_dense(model: SCJEPA, state: dict[str, Tensor]) -> None:
    result = model.load_state_dict(state, strict=False)
    unexpected = cast(list[str], result.unexpected_keys)  # pyright: ignore[reportUnknownMemberType]
    missing = cast(list[str], result.missing_keys)  # pyright: ignore[reportUnknownMemberType]
    if unexpected or not set(missing) <= MASK_LOGITS:
        raise ValueError("dense endpoint and sparse model differ beyond mask logits")


def _graph_metrics(predictor: MaskedTrunkPredictor, columns: list[int]) -> dict[str, float]:
    """Aligned graphs; an open twelve-code predictor uses a padded true parameter graph.

    The six extra code columns of the padded graph have no true edges.
    """
    if predictor.param_dim == SPARSE_CODE_DIM:
        return oracle.graph_metrics(predictor, columns)
    if predictor.learned_logits():
        raise ValueError("only open dense predictors use a twelve-dimensional code")
    truth = true_graphs()
    metrics: dict[str, float] = {}
    for name, expected in (("state", truth.state[:, 0]), ("action", truth.action[:, 0])):
        metrics[f"masks/{name}_edges"] = float(expected.numel())
        metrics[f"masks/{name}_true_edges"] = float(expected.sum())
        metrics[f"masks/{name}_aligned_shd"] = float((~expected).sum())
        metrics[f"masks/{name}_missing_true_edges"] = 0.0
    edges = predictor.state_dim * predictor.param_dim
    true_edges = float(truth.parameter.sum())
    metrics["masks/parameter_edges"] = float(edges)
    metrics["masks/parameter_true_edges"] = true_edges
    metrics["masks/parameter_aligned_shd"] = edges - true_edges
    metrics["masks/parameter_missing_true_edges"] = 0.0
    return metrics


def _collect(
    model: SCJEPA,
    data: dict[str, StateData],
    batch_size: int,
    device: torch.device,
    objective: TrainingObjective,
    tau_prediction: float | None,
) -> tuple[dict[str, EvaluationResult], dict[str, list[int]]]:
    calibration = evaluate(model, data["calibration"], batch_size, device, objective)
    results = {"calibration": calibration}
    matching: dict[str, list[int]] = {}
    for split in ("calibration", "validation"):
        result = (
            calibration
            if split == "calibration"
            else evaluate(model, data[split], batch_size, device, objective)
        )
        metrics, matching = recovery_metrics(calibration, result)
        metrics["forecast_quality_pass"] = float(
            metrics["normalized_prediction_mse"] <= NORMALIZED_MSE_CAP
        )
        metrics |= oracle.block_purity(calibration, result)
        metrics |= _graph_metrics(_predictor(model), matching["online_columns"])
        metrics |= code_metrics(calibration, result)
        if objective.geco is not None and tau_prediction is not None:
            relative = result.metrics["prediction"] / tau_prediction
            metrics.update(
                {
                    "geco/tau_prediction": tau_prediction,
                    "geco/relative_prediction": relative,
                    "geco/feasible": float(relative <= 1),
                    "geco/penalty_weight": float(objective.geco.penalty_weight),
                }
            )
        results[split] = EvaluationResult(result.metrics | metrics, result.samples)
    return results, matching


def _scientific_config(config: TrainConfig) -> dict[str, Any]:
    values = {
        key: value
        for key, value in asdict(config).items()
        if key
        not in {
            "output_dir",
            "data_dir",
            "calibration_dir",
            "init_dir",
            "device",
            "threads",
            "log_every",
            "evaluate_every",
            "checkpoint_every",
            "wandb_mode",
            "wandb_project",
            "wandb_entity",
            "wandb_group",
        }
    }
    code_dim = config.model_code_dim
    return values | {
        "code_dim": code_dim,
        "initializer": config.model_initializer,
        "parameter_source": "learned transition-set encoder, cross-episode donor codes",
        "parameter_training": "forecast only; no parameter regularizer or consistency term",
        "objective": asdict(objective_config(config.condition, 0.0)),
        "constraint": "sparse arms: GECO on prediction MSE / tau_prediction <= 1",
        "initialization": (
            "fresh paired Gaussian encoder head, parameter-encoder input layer, trunk, "
            f"readouts; parameter-encoder policy {config.model_initializer}"
            if config.condition == "dense"
            else "seed-matched six-code dense endpoint; fresh mask logits, AdamW, GECO, sampler"
        ),
        "ema_decay": 0.996,
        "weight_decay": 0.0,
        "predictor": (
            f"shared {30 + code_dim}-{trunk_width()}-{trunk_width()} ReLU trunk, "
            "per-output readouts"
        ),
        "forecast": "attached two-step prediction from online state 9; EMA targets 10/11",
        "test_policy": "validation-only screen; no test command",
    }


def _log(logger: WandbLogger, output: Path, step: int, metrics: dict[str, float]) -> None:
    if any(not math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("nonfinite reported metric")
    logger.log(step, metrics)
    with (output / "history.jsonl").open("a") as stream:
        stream.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")


def train(config: TrainConfig, *, resume: bool = False) -> dict[str, Any]:
    """Hold an exclusive output lock while training or resuming exactly one arm."""
    output = Path(config.output_dir).resolve()
    with file_lock(output.parent / f".{output.name}.lock"):
        return _train(config, output, resume=resume)


def _train(config: TrainConfig, output: Path, *, resume: bool) -> dict[str, Any]:
    torch.set_num_threads(config.threads)
    device = resolve_device(config.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not resume:
        _fresh_output(output)
    elif not (output / "last.pt").is_file():
        raise FileNotFoundError("resume requires last.pt")
    data, cache = load_data(Path(config.data_dir))
    if cache["settings"]["profile"] != config.profile:
        raise ValueError("training and cache profiles differ")
    if config.batch_size > len(data["train"]):
        raise ValueError("training data must contain a full group minibatch")
    data_identity = str(cache["identity"])
    calibration = _calibration(config, data_identity)
    endpoint = _dense_endpoint(config, data_identity)
    tau_prediction = float(calibration["tau_prediction"]) if calibration else None
    seed_everything(config.seed)
    model = build_control(
        config.condition, config.seed, config.model_code_dim, config.model_initializer
    ).to(device)
    initialization: dict[str, Any] | None = None
    if resume:
        # The data-dependent initializer is restored, never refitted, on resume.
        saved = cast(
            dict[str, Any],
            torch.load(output / "initial.pt", map_location="cpu", weights_only=False),
        )
        model.load_state_dict(saved["model"], strict=True)
        initialization = saved.get("initialization")
    elif endpoint is not None:
        _load_dense(model, endpoint["state"])
    elif config.model_initializer == "shared_feature":
        initialization = initialize_shared_output(model, data["train"], device)
    initials = _initial_identities(model)
    controller = (
        GECOConfig(enabled=True, tau=1.0, step_size=config.geco_step_size)
        if calibration
        else GECOConfig()
    )
    objective = TrainingObjective(objective_config(config.condition, 0.0), controller).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if any(parameter.requires_grad for parameter in model.target_encoder.parameters()):
        raise RuntimeError("target encoder must remain outside gradient optimization")
    if not any(parameter.requires_grad for parameter in model.parameter_encoder.parameters()):
        raise RuntimeError("the parameter encoder must be trained")
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=0)
    sampler = GroupSampler(len(data["train"]), config.batch_size, config.seed)
    source_identity = code_identity()
    specification: dict[str, Any] = {
        "scientific_config": _scientific_config(config),
        "candidate_settings": cache["settings"],
        "data_identity": data_identity,
        "cache_identity": canonical_fingerprint(cache),
        "code_identity": source_identity,
        "calibration_identity": calibration["identity"] if calibration else None,
        "tau_prediction": tau_prediction,
        "geco": asdict(controller),
        "dense_endpoint": endpoint["record"] if endpoint else None,
        "initialization": initialization,
        **initials,
    }
    identity = canonical_fingerprint(specification)
    record = {
        "schema": SCHEMA,
        "identity": identity,
        **specification,
        "config": asdict(config),
        "physical_state_training": False,
        "true_parameter_training": False,
        "target_policy": "EMA .996, no optimizer gradients",
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    restored: dict[str, Any] = {}
    start = 0
    if resume:
        restored = cast(
            dict[str, Any], torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        )
        if restored.get("schema") != SCHEMA or restored.get("identity") != identity:
            raise ValueError("checkpoint configuration, source, initialization or data differs")
        if restored.get("initial_checkpoint_sha256") != sha(output / "initial.pt"):
            raise ValueError("saved initial checkpoint changed")
        model.load_state_dict(restored["model"], strict=True)
        objective.load_state_dict(restored["objective"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        sampler.load_state_dict(restored["sampler"])
        start = int(restored["step"])
        if not 0 <= start <= config.steps:
            raise ValueError("checkpoint step exceeds its fixed budget")
        for state in optimizer.state.values():
            if int(cast(Tensor, state["step"])) != start:
                raise ValueError("Adam state and update count disagree")
        if objective.geco is not None:
            objective.geco.validate_state()
            if int(objective.geco.num_updates) != start:
                raise ValueError("GECO and optimizer update counts disagree")
        torch.set_rng_state(restored["torch_rng"])
        random.setstate(restored["python_rng"])
        np.random.set_state(restored["numpy_rng"])  # noqa: NPY002
        if device.type == "cuda":
            torch.cuda.set_rng_state(restored["cuda_rng"], device)
        if (output / "completion.json").exists():
            completed = verify_completion(output, kind="train", require_current_source=True)
            if completed["identity"] != identity or completed["steps"] != config.steps:
                raise ValueError("completed run differs from requested continuation")
            return completed
    else:
        save_torch(
            output / "initial.pt",
            {"model": model.state_dict(), "initialization": initialization, **initials},
        )
    initial_checkpoint_sha256 = sha(output / "initial.pt")
    record["initial_checkpoint_sha256"] = initial_checkpoint_sha256
    write_json(output / "config.json", record)
    logger = WandbLogger(
        WandbConfig(
            mode=config.wandb_mode,
            project=config.wandb_project,
            entity=config.wandb_entity,
            group=config.wandb_group,
            name=f"masked-mlp-learned-{config.condition}-s{config.seed}",
            job_type="masked-mlp-learned",
            tags=["diagnostic", "learned-parameters", "masked-mlp", config.condition],
        ),
        resolved_config=cast(dict[str, object], record),
        dataset_identity=data_identity,
        out_dir=output,
        resume_state=restored.get("wandb"),
    )
    write_json(output / "wandb.json", logger.metadata)

    def save(step: int) -> None:
        save_torch(
            output / "last.pt",
            {
                "schema": SCHEMA,
                "identity": identity,
                "initial_checkpoint_sha256": initial_checkpoint_sha256,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "objective": objective.state_dict(),
                "sampler": sampler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),  # noqa: NPY002
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "wandb": logger.metadata,
            },
        )

    def collect() -> tuple[dict[str, EvaluationResult], dict[str, list[int]]]:
        return _collect(model, data, config.batch_size, device, objective, tau_prediction)

    def flat(results: dict[str, EvaluationResult]) -> dict[str, float]:
        return {
            f"{split}/{key}": value
            for split, result in results.items()
            for key, value in result.metrics.items()
        }

    started = time.monotonic()
    try:
        if not resume:
            initial, _ = collect()
            write_json(
                output / "startup.json",
                {**initials, "metrics": {split: r.metrics for split, r in initial.items()}},
            )
            _log(logger, output, 0, flat(initial))
            save(0)
        model.train()
        objective.train()
        for step in range(start + 1, config.steps + 1):
            batch = data["train"].batch(sampler.next(), device)
            inputs = {key: value for key, value in batch.items() if key in INPUT_KEYS}
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed((1 << 40) + config.seed + step)  # pyright: ignore[reportUnknownMemberType]
            loss, terms, _ = forward_loss(model, inputs, objective, forward_seed=config.seed + step)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss at update {step}")
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            gradient = torch.nn.utils.clip_grad_norm_(
                parameters, config.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            model.update_target()
            if objective.geco is not None and tau_prediction is not None:
                objective.geco.update(terms["prediction"].detach() / tau_prediction)
            if step % config.log_every == 0 or step == config.steps:
                metrics = {f"loss/{key}": float(value.detach()) for key, value in terms.items()}
                metrics.update(
                    {
                        "loss/total": float(loss.detach()),
                        "train/gradient_norm": float(gradient),
                        "train/successful_updates": float(step),
                        "train/elapsed_seconds": time.monotonic() - started,
                    }
                )
                for name, logits in _predictor(model).learned_logits().items():
                    metrics[f"masks/{name}_open_probability"] = float(
                        torch.sigmoid(logits.detach()).mean()
                    )
                    metrics[f"masks/{name}_edges"] = float((logits.detach() > 0).sum())
                if objective.geco is not None:
                    metrics.update(
                        {
                            "geco/penalty_weight": float(objective.geco.penalty_weight),
                            "geco/log10_penalty_weight": -float(objective.geco.log_lambda)
                            / math.log(10),
                            "geco/moving_violation": float(objective.geco.ma_error),
                        }
                    )
                _log(logger, output, step, metrics)
            if step % config.evaluate_every == 0 and step != config.steps:
                interim, _ = collect()
                _log(logger, output, step, flat(interim))
            if step % config.checkpoint_every == 0 or step == config.steps:
                save(step)
        names = [
            "initial.pt",
            "config.json",
            "startup.json",
            "history.jsonl",
            "last.pt",
            "wandb.json",
        ]
        final, matching = collect()
        write_json(output / "matching.json", matching)
        names.append("matching.json")
        for split, result in final.items():
            write_json(output / f"{split}.json", {"metrics": result.metrics})
            save_torch(
                output / f"{split}.pt", {"metrics": result.metrics, "samples": result.samples}
            )
            names += [f"{split}.json", f"{split}.pt"]
        _log(logger, output, config.steps, flat(final))
        _log(logger, output, config.steps, {"diagnostic/complete": 1.0})
        logger.finish()
        if code_identity() != source_identity:
            raise RuntimeError("numerical source changed during training; completion refused")
        completion: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": "train",
            "identity": identity,
            "code_identity": source_identity,
            "profile": config.profile,
            "condition": config.condition,
            "seed": config.seed,
            "code_dim": config.model_code_dim,
            "steps": config.steps,
            "successful_updates": config.steps,
            "skipped_updates": 0,
            "data_identity": data_identity,
            "calibration_identity": calibration["identity"] if calibration else None,
            "tau_prediction": tau_prediction,
            "dense_endpoint": endpoint["record"] if endpoint else None,
            **initials,
            "wandb": logger.metadata,
            "hashes": {name: sha(output / name) for name in names},
        }
        write_json(output / "completion.json", completion)
        return completion
    except BaseException:
        logger.finish(exit_code=1)
        raise


@dataclass(frozen=True)
class ProjectConfig:
    """Map one twelve-code dense endpoint to six codes through TRAIN shared modes."""

    output_dir: str
    dense_dir: str
    data_dir: str
    profile: Profile = "exploratory"
    batch_size: int = 16
    device: str = "cpu"
    threads: int = 1

    def __post_init__(self) -> None:
        """Keep the evaluation batch of the training runs."""
        if self.profile not in {"exploratory", "smoke"}:
            raise ValueError("profile must be exploratory or smoke")
        if self.profile == "exploratory" and self.batch_size != 16:
            raise ValueError("exploratory evaluation batch is fixed")
        if type(self.batch_size) is not int or self.batch_size < 2:
            raise ValueError("batch_size must be an integer of at least two")


def project(config: ProjectConfig) -> dict[str, Any]:
    """Write a six-code dense endpoint; only TRAIN observations, actions and groups enter."""
    output = Path(config.output_dir).resolve()
    with file_lock(output.parent / f".{output.name}.lock"):
        _fresh_output(output)
        torch.set_num_threads(config.threads)
        device = resolve_device(config.device)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        directory = Path(config.dense_dir).resolve()
        completion = verify_completion(directory, kind="train", require_current_source=True)
        if completion["condition"] != "dense" or completion["profile"] != config.profile:
            raise ValueError("projection source must be a matching-profile dense fit")
        if completion["code_dim"] <= SPARSE_CODE_DIM:
            raise ValueError("projection source must have more than six code coordinates")
        data, cache = load_data(Path(config.data_dir))
        if str(cache["identity"]) != completion["data_identity"]:
            raise ValueError("projection data differs from the dense fit")
        checkpoint = cast(
            dict[str, Any],
            torch.load(directory / "last.pt", map_location="cpu", weights_only=False),
        )
        if checkpoint.get("identity") != completion["identity"]:
            raise ValueError("dense checkpoint does not belong to its completion record")
        if int(checkpoint["step"]) != int(completion["steps"]):
            raise ValueError("dense checkpoint is not the final endpoint")
        seed = int(completion["seed"])
        source = build_control("dense", seed, int(completion["code_dim"])).to(device)
        source.load_state_dict(checkpoint["model"], strict=True)
        model = build_control("dense", seed, SPARSE_CODE_DIM).to(device)
        receipt = project_code(source, model, data["train"], device)
        objective = TrainingObjective(objective_config("dense", 0.0), GECOConfig()).to(device)
        results, matching = _collect(model, data, config.batch_size, device, objective, None)
        source_metrics = read_json(directory / "validation.json")["metrics"]
        specification: dict[str, Any] = {
            "code_identity": code_identity(),
            "data_identity": completion["data_identity"],
            "source": {
                "directory": str(directory),
                "identity": completion["identity"],
                "completion_sha256": sha(directory / "completion.json"),
                "checkpoint_sha256": sha(directory / "last.pt"),
                "code_dim": completion["code_dim"],
            },
            "projection": receipt,
            "policy": (
                "top six TRAIN shared modes of the twelve-dimensional code; encoder output "
                "rows composed with the projection, predictor code columns with its "
                "least-squares reconstruction; no theta, held-out data or optimization"
            ),
        }
        identity = canonical_fingerprint(specification)
        save_torch(
            output / "last.pt",
            {
                "schema": SCHEMA,
                "identity": identity,
                "step": completion["steps"],
                "model": model.state_dict(),
            },
        )
        write_json(
            output / "projection.json",
            {
                **specification,
                "identity": identity,
                "source_validation": {
                    key: source_metrics.get(key)
                    for key in ("normalized_prediction_mse", "code_r2_min", "code_r2_mean")
                },
            },
        )
        write_json(output / "matching.json", matching)
        names = ["last.pt", "projection.json", "matching.json"]
        for split, result in results.items():
            write_json(output / f"{split}.json", {"metrics": result.metrics})
            save_torch(
                output / f"{split}.pt", {"metrics": result.metrics, "samples": result.samples}
            )
            names += [f"{split}.json", f"{split}.pt"]
        record: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": "projection",
            "identity": identity,
            "code_identity": specification["code_identity"],
            "profile": config.profile,
            "condition": "dense",
            "seed": seed,
            "code_dim": SPARSE_CODE_DIM,
            "steps": completion["steps"],
            "data_identity": completion["data_identity"],
            "source": specification["source"],
            "hashes": {name: sha(output / name) for name in names},
        }
        write_json(output / "completion.json", record)
        return record


SUMMARY_KEYS = (
    "state_mcc",
    "block_purity",
    "normalized_prediction_mse",
    "online_state_probe/r2",
    "code_r2_min",
    "code_r2_mean",
    "code_mcc",
    "code_pair_purity",
    "forecast/code_use_ratio",
    "masks/state_aligned_shd",
    "masks/parameter_edges",
    "masks/parameter_aligned_shd",
    "geco/relative_prediction",
)


def summarize(directories: list[str]) -> list[dict[str, Any]]:
    """Read authenticated start and final validation metrics; no model or data is loaded."""
    rows: list[dict[str, Any]] = []
    for value in directories:
        directory = Path(value).resolve()
        completion = verify_completion(directory)
        if completion["kind"] not in {"train", "projection"}:
            raise ValueError(f"{directory} is not a training or projection endpoint")
        final = read_json(directory / "validation.json")["metrics"]
        start: dict[str, Any] = (
            read_json(directory / "startup.json")["metrics"]["validation"]
            if completion["kind"] == "train"
            else {}
        )
        rows.append(
            {
                "condition": completion["condition"],
                "kind": completion["kind"],
                "seed": completion["seed"],
                "code_dim": completion["code_dim"],
                "steps": completion["steps"],
                "initial_state_mcc": start.get("state_mcc"),
                **{key: final.get(key) for key in SUMMARY_KEYS},
            }
        )
    return sorted(rows, key=lambda row: (row["condition"], row["kind"], row["seed"]))


def main() -> None:
    """Train one arm, project a dense code, fix the dense bound, or summarize endpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--data-dir", required=True)
    train_parser.add_argument("--condition", choices=("dense", "baseline", "full"), required=True)
    train_parser.add_argument("--profile", choices=("exploratory", "smoke"), default="exploratory")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--steps", type=int, default=100_000)
    train_parser.add_argument("--calibration-dir")
    train_parser.add_argument("--init-dir")
    train_parser.add_argument("--geco-step-size", type=float, default=0.1)
    train_parser.add_argument("--code-dim", type=int, choices=CODE_DIMS)
    train_parser.add_argument("--initializer", choices=INITIALIZERS)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--threads", type=int, default=1)
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--wandb-mode", default="disabled")
    train_parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    train_parser.add_argument("--wandb-entity")
    train_parser.add_argument("--wandb-group", default="masked-mlp-learned")
    project_parser = commands.add_parser("project")
    project_parser.add_argument("--output-dir", required=True)
    project_parser.add_argument("--dense-dir", required=True)
    project_parser.add_argument("--data-dir", required=True)
    project_parser.add_argument(
        "--profile", choices=("exploratory", "smoke"), default="exploratory"
    )
    project_parser.add_argument("--device", default="cpu")
    project_parser.add_argument("--threads", type=int, default=1)
    calibrate_parser = commands.add_parser("calibrate")
    calibrate_parser.add_argument("--output-dir", required=True)
    calibrate_parser.add_argument("--dense-dir", required=True)
    calibrate_parser.add_argument(
        "--profile", choices=("exploratory", "smoke"), default="exploratory"
    )
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("--runs", nargs="+", required=True)
    arguments = parser.parse_args()
    if arguments.command == "train":
        result: Any = train(
            TrainConfig(
                output_dir=arguments.output_dir,
                data_dir=arguments.data_dir,
                condition=arguments.condition,
                profile=arguments.profile,
                seed=arguments.seed,
                steps=arguments.steps,
                calibration_dir=arguments.calibration_dir,
                init_dir=arguments.init_dir,
                geco_step_size=arguments.geco_step_size,
                code_dim=arguments.code_dim,
                initializer=arguments.initializer,
                device=arguments.device,
                threads=arguments.threads,
                wandb_mode=arguments.wandb_mode,
                wandb_project=arguments.wandb_project,
                wandb_entity=arguments.wandb_entity,
                wandb_group=arguments.wandb_group,
            ),
            resume=arguments.resume,
        )
    elif arguments.command == "project":
        result = project(
            ProjectConfig(
                arguments.output_dir,
                arguments.dense_dir,
                arguments.data_dir,
                profile=arguments.profile,
                device=arguments.device,
                threads=arguments.threads,
            )
        )
    elif arguments.command == "calibrate":
        result = calibrate(
            CalibrateConfig(arguments.output_dir, arguments.dense_dir, profile=arguments.profile)
        )
    else:
        result = summarize(arguments.runs)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
