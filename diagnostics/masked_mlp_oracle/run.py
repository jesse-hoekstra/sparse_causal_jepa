"""Oracle-theta state learning with chart-equivariant masked MLP predictors.

Adapted from the frozen balanced-action runner at commit `b4d2808`. The Paley24
cache, attached two-step forecast, state encoder, EMA, state SIGReg, optimizer
and evaluation are reused unchanged. Only the predictor and the GECO constraint
differ: sparse arms bound the normalized prediction MSE, ``E[pred / tau] <= 1``,
with ``tau`` fixed from the seed-zero dense masked MLP's final calibration loss.
SIGReg stays in the optimized loss but outside that constraint.
"""

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn

from diagnostics.balanced_action_oracle.data import StateData, load_data
from diagnostics.balanced_action_oracle.data import code_identity as cache_code_identity
from diagnostics.isotropic_oracle.dgp import graphs as true_graphs
from diagnostics.masked_mlp_oracle.model import (
    INITIAL_LOGIT,
    Condition,
    MaskedTrunkPredictor,
    build_control,
    trunk_width,
)
from diagnostics.oracle_state_learning.core import (
    EvaluationResult,
    LiveStateData,
    evaluate,
    forward_loss,
    objective_config,
)
from diagnostics.oracle_state_learning.data import read_json, save_torch, sha
from diagnostics.oracle_state_learning.run import recovery_metrics
from diagnostics.oracle_state_predictor.run import GroupSampler, resolve_device
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.experiments.storage import file_lock, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.lagrangian import GECOConfig
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import TrainingObjective, seed_everything

SCHEMA = "masked-mlp-oracle-state-learning-v1"
NORMALIZED_MSE_CAP = 0.15
SEEDS = (0, 1, 2)
BLOCKS = ((0, 1, 2), (3, 4, 5))
type Profile = Literal["exploratory", "smoke"]


def code_identity() -> str:
    """Bind the authenticated Paley cache source plus this diagnostic's numerical files."""
    files = {
        name: sha(Path(__file__).with_name(name)) for name in ("__init__.py", "model.py", "run.py")
    }
    return canonical_fingerprint({"cache_code_identity": cache_code_identity(), "files": files})


@dataclass(frozen=True)
class TrainConfig:
    """One fresh fixed-budget fit; sparse arms read a bound but no parent weights."""

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
    device: str = "cpu"
    threads: int = 1
    log_every: int = 100
    evaluate_every: int = 5000
    checkpoint_every: int = 5000
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None
    wandb_group: str = "masked-mlp-oracle"

    def __post_init__(self) -> None:
        """Freeze scientific settings and identify reduced fixtures explicitly."""
        if self.profile not in {"exploratory", "smoke"}:
            raise ValueError("profile must be exploratory or smoke")
        if self.condition not in {"dense", "baseline", "full"}:
            raise ValueError("unknown masked MLP condition")
        if self.seed not in SEEDS:
            raise ValueError(f"training seed must be one of {SEEDS}")
        if self.profile == "exploratory" and (self.batch_size, self.lr, self.grad_clip) != (
            16,
            0.001,
            1.0,
        ):
            raise ValueError("exploratory optimizer settings are fixed")
        if self.profile == "exploratory" and self.steps != 100_000:
            raise ValueError("exploratory budget is fixed at 100000 updates")
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
        if any(not math.isfinite(value) or value <= 0 for value in (self.lr, self.grad_clip)):
            raise ValueError("learning rate and clipping must be finite and positive")
        if (self.condition != "dense") != (self.calibration_dir is not None):
            raise ValueError("exactly the sparse arms require the calibration directory")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


@dataclass(frozen=True)
class CalibrateConfig:
    """Fix the prediction bound from the one seed-zero dense masked MLP."""

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
        "initial_trunk_identity": _state_identity(model.predictor, exclude_logits=True),
        "initial_predictor_identity": _state_identity(model.predictor),
        "initial_model_identity": _state_identity(model),
    }


def _predictor(model: SCJEPA) -> MaskedTrunkPredictor:
    predictor = cast(nn.Module, model.predictor)
    if not isinstance(predictor, MaskedTrunkPredictor):
        raise TypeError("this runner trains only the masked MLP predictor")
    return predictor


def _scientific_config(config: TrainConfig) -> dict[str, Any]:
    values = asdict(config)
    for name in (
        "output_dir",
        "data_dir",
        "calibration_dir",
        "device",
        "threads",
        "log_every",
        "evaluate_every",
        "checkpoint_every",
        "wandb_mode",
        "wandb_project",
        "wandb_entity",
        "wandb_group",
    ):
        values.pop(name)
    return values | {
        "initialization": "fresh paired Gaussian encoder head, trunk and readouts",
        "parameter_source": "supplied raw ground-truth theta",
        "objective": asdict(objective_config(config.condition, 0.0)),
        "constraint": "sparse arms: GECO on prediction MSE / tau_prediction <= 1",
        "ema_decay": 0.996,
        "weight_decay": 0.0,
        "scheduler": "constant",
        "state_encoder": "6-64-64-6 native VectorEncoder, Gaussian head",
        "predictor": f"shared 36-{trunk_width()}-{trunk_width()} ReLU trunk, per-output readouts",
        "masks": {
            "dense": "all open",
            "baseline": "learned state/action, parameter fixed open",
            "full": "learned state/action/parameter",
        }[config.condition],
        "initial_logit": INITIAL_LOGIT,
        "forecast": "attached two-step prediction from online state9; EMA targets10/11",
        "test_policy": "validation-only screen; no test command in this diagnostic",
        "normalized_mse_interpretation_cap": NORMALIZED_MSE_CAP,
    }


def _finite_nonconstant(result: dict[str, Any]) -> bool:
    metrics = cast(dict[str, float], result["metrics"])
    return all(math.isfinite(value) for value in metrics.values()) and all(
        metrics[f"{prefix}/variance_min"] > 0 for prefix in ("online", "target")
    )


def calibrate(config: CalibrateConfig) -> dict[str, Any]:
    """Create an immutable prediction bound from calibration loss, never test or MCC."""
    output = Path(config.output_dir).resolve()
    with file_lock(output.parent / f".{output.name}.lock"):
        _fresh_output(output)
        directory = Path(config.dense_dir).resolve()
        completion = verify_completion(directory, kind="train", require_current_source=True)
        if completion["condition"] != "dense" or completion["profile"] != config.profile:
            raise ValueError("calibration reference must be a matching-profile dense fit")
        if completion["seed"] != 0:
            raise ValueError("calibration requires the seed-zero dense reference")
        calibration = read_json(directory / "calibration.json")
        if not _finite_nonconstant(calibration):
            raise ValueError("dense calibration has nonfinite or constant state encodings")
        prediction = float(calibration["metrics"]["prediction"])
        if prediction <= 0:
            raise ValueError("dense calibration prediction loss must be positive")
        reference = {
            "directory": str(directory),
            "identity": completion["identity"],
            "completion_sha256": sha(directory / "completion.json"),
            "checkpoint_sha256": sha(directory / "last.pt"),
            "seed": completion["seed"],
            "profile": completion["profile"],
            "data_identity": completion["data_identity"],
            "initial_encoder_identity": completion["initial_encoder_identity"],
            "initial_trunk_identity": completion["initial_trunk_identity"],
            "prediction_loss": prediction,
            "base_loss": float(calibration["metrics"]["base"]),
            "scientific_config": read_json(directory / "config.json")["scientific_config"],
            "calibration_normalized_prediction_mse": calibration["metrics"][
                "normalized_prediction_mse"
            ],
        }
        record: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": "calibration",
            "profile": config.profile,
            "code_identity": code_identity(),
            "data_identity": reference["data_identity"],
            "tau_prediction": config.tau_factor * prediction,
            "tau_factor": config.tau_factor,
            "references": [reference],
            "policy": "tau_factor times dense calibration prediction MSE; no weights transferred",
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
    if record["identity"] != completion["identity"]:
        raise ValueError("calibration identity differs")
    if record["data_identity"] != data_identity or record["profile"] != config.profile:
        raise ValueError("calibration data or profile differs")
    tau = float(record["tau_prediction"])
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("calibrated bound must be finite and positive")
    reference = record["references"][0]["scientific_config"]
    for name in ("steps", "batch_size", "lr", "grad_clip"):
        if reference[name] != getattr(config, name):
            raise ValueError(f"Sparse setting {name} differs from its dense reference")
    return record


def block_purity(calibration: EvaluationResult, result: EvaluationResult) -> dict[str, float]:
    """Best single-block linear R² per learned coordinate, fitted on calibration rows only."""

    def rows(sample: EvaluationResult, key: str) -> Tensor:
        return sample.samples[key].reshape(-1, 6).double()

    learned, truth = rows(calibration, "online_states"), rows(calibration, "states")
    scored, scored_truth = rows(result, "online_states"), rows(result, "states")
    metrics: dict[str, float] = {}
    best: list[float] = []
    for coordinate in range(6):
        scores: list[float] = []
        for block in BLOCKS:
            design = torch.cat((truth[:, block], truth.new_ones(len(truth), 1)), 1)
            fit = torch.linalg.lstsq(design, learned[:, coordinate : coordinate + 1])  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            weights = cast(Tensor, fit.solution)  # pyright: ignore[reportUnknownMemberType]
            held = torch.cat(
                (scored_truth[:, block], scored_truth.new_ones(len(scored_truth), 1)), 1
            )
            target = scored[:, coordinate]
            residual = target - (held @ weights)[:, 0]
            total = (target - target.mean()).square().sum().clamp_min(1e-15)
            scores.append(float(1 - residual.square().sum() / total))
        best.append(max(scores))
        metrics[f"block_purity/coordinate_{coordinate + 1}"] = best[-1]
    metrics["block_purity"] = sum(best) / len(best)
    return metrics


def graph_metrics(predictor: MaskedTrunkPredictor, columns: list[int]) -> dict[str, float]:
    """Thresholded global masks, aligned to true coordinates by the calibration matching."""
    learned = predictor.thresholded_graphs()
    truth = true_graphs()
    expected = {
        "state": truth.state[:, 0],
        "action": truth.action[:, 0],
        "parameter": truth.parameter,
    }
    order = torch.tensor(columns)
    aligned = {
        "state": learned["state"][order][:, order],
        "action": learned["action"][order],
        "parameter": learned["parameter"][order],
    }
    metrics: dict[str, float] = {}
    for name, graph in aligned.items():
        metrics[f"masks/{name}_edges"] = float(graph.sum())
        metrics[f"masks/{name}_true_edges"] = float(expected[name].sum())
        metrics[f"masks/{name}_aligned_shd"] = float((graph != expected[name]).sum())
        metrics[f"masks/{name}_missing_true_edges"] = float((~graph & expected[name]).sum())
    for name, logits in predictor.learned_logits().items():
        metrics[f"masks/{name}_open_probability"] = float(torch.sigmoid(logits.detach()).mean())
    return metrics


def _collect(
    model: SCJEPA,
    data: dict[str, StateData],
    config: TrainConfig,
    device: torch.device,
    objective: TrainingObjective,
    tau_prediction: float | None,
) -> tuple[dict[str, EvaluationResult], dict[str, list[int]]]:
    # The frozen evaluator uses only states, batch(), and len(); action dimension
    # is read from the predictor. This explicit adapter leaves its old protocol intact.
    calibration = evaluate(
        model, cast(LiveStateData, data["calibration"]), config.batch_size, device, objective
    )
    results = {"calibration": calibration}
    matching: dict[str, list[int]] = {}
    for split in ("calibration", "validation"):
        result = (
            calibration
            if split == "calibration"
            else evaluate(
                model, cast(LiveStateData, data[split]), config.batch_size, device, objective
            )
        )
        metrics, matching = recovery_metrics(calibration, result)
        metrics["forecast_quality_pass"] = float(
            metrics["normalized_prediction_mse"] <= NORMALIZED_MSE_CAP
        )
        metrics |= block_purity(calibration, result)
        metrics |= graph_metrics(_predictor(model), matching["online_columns"])
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
    calibration = _calibration(config, str(cache["identity"]))
    tau_prediction = float(calibration["tau_prediction"]) if calibration else None
    seed_everything(config.seed)
    model = build_control(config.condition, config.seed).to(device)
    initials = _initial_identities(model)
    if calibration is not None:
        reference = calibration["references"][0]
        if config.seed == reference["seed"]:
            for key in ("initial_encoder_identity", "initial_trunk_identity"):
                if initials[key] != reference[key]:
                    raise ValueError("fresh sparse initialization differs from its dense reference")
    # The normalized constraint keeps GECO's default step size on a unit scale.
    controller = GECOConfig(enabled=True, tau=1.0) if calibration else GECOConfig()
    objective = TrainingObjective(objective_config(config.condition, 0.0), controller).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if any(parameter.requires_grad for parameter in model.target_encoder.parameters()):
        raise RuntimeError("target encoder must remain outside gradient optimization")
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=0)
    sampler = GroupSampler(len(data["train"]), config.batch_size, config.seed)
    source_identity = code_identity()
    specification: dict[str, Any] = {
        "scientific_config": _scientific_config(config),
        "candidate_settings": cache["settings"],
        "data_identity": cache["identity"],
        "cache_identity": canonical_fingerprint(cache),
        "code_identity": source_identity,
        "calibration_identity": calibration["identity"] if calibration else None,
        "tau_prediction": tau_prediction,
        "geco": asdict(controller),
        **initials,
    }
    identity = canonical_fingerprint(specification)
    record = {
        "schema": SCHEMA,
        "identity": identity,
        **specification,
        "config": asdict(config),
        "privileged_control": True,
        "physical_state_training": False,
        "parameter_encoder_training": False,
        "target_policy": "EMA .996, no optimizer gradients",
        "normalized_mse_cap": NORMALIZED_MSE_CAP,
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
        if read_json(output / "config.json")["identity"] != identity:
            raise ValueError("saved run configuration differs")
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
        save_torch(output / "initial.pt", {"model": model.state_dict(), **initials})
    initial_checkpoint_sha256 = sha(output / "initial.pt")
    record["initial_checkpoint_sha256"] = initial_checkpoint_sha256
    write_json(output / "config.json", record)
    logger = WandbLogger(
        WandbConfig(
            mode=config.wandb_mode,
            project=config.wandb_project,
            entity=config.wandb_entity,
            group=config.wandb_group,
            name=f"masked-mlp-paley-{config.condition}-s{config.seed}",
            job_type="masked-mlp-oracle",
            tags=["diagnostic", "oracle", "masked-mlp", "balanced-action-dgp", config.condition],
        ),
        resolved_config=cast(dict[str, object], record),
        dataset_identity=str(cache["identity"]),
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
        return _collect(model, data, config, device, objective, tau_prediction)

    started = time.monotonic()
    try:
        if not resume:
            initial, _ = collect()
            write_json(
                output / "startup.json",
                {
                    **initials,
                    "metrics": {split: result.metrics for split, result in initial.items()},
                },
            )
            _log(
                logger,
                output,
                0,
                {
                    f"{split}/{key}": value
                    for split, result in initial.items()
                    for key, value in result.metrics.items()
                },
            )
            save(0)
        model.train()
        objective.train()
        for step in range(start + 1, config.steps + 1):
            batch = data["train"].batch(sampler.next(), device)
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed((1 << 40) + config.seed + step)  # pyright: ignore[reportUnknownMemberType]
            loss, terms, _ = forward_loss(model, batch, objective, forward_seed=config.seed + step)
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
                    probability = torch.sigmoid(logits.detach())
                    metrics[f"masks/{name}_open_probability"] = float(probability.mean())
                    metrics[f"masks/{name}_edges"] = float((logits.detach() > 0).sum())
                if objective.geco is not None:
                    metrics.update(
                        {
                            "geco/penalty_weight": float(objective.geco.penalty_weight),
                            "geco/moving_violation": float(objective.geco.ma_error),
                        }
                    )
                _log(logger, output, step, metrics)
            if step % config.evaluate_every == 0 and step != config.steps:
                interim, _ = collect()
                _log(
                    logger,
                    output,
                    step,
                    {
                        f"{split}/{key}": value
                        for split, result in interim.items()
                        for key, value in result.metrics.items()
                    },
                )
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
            _log(
                logger,
                output,
                config.steps,
                {f"{split}/{key}": value for key, value in result.metrics.items()},
            )
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
            "steps": config.steps,
            "successful_updates": config.steps,
            "skipped_updates": 0,
            "data_identity": cache["identity"],
            "calibration_identity": calibration["identity"] if calibration else None,
            "tau_prediction": tau_prediction,
            **initials,
            "wandb": logger.metadata,
            "hashes": {name: sha(output / name) for name in names},
        }
        write_json(output / "completion.json", completion)
        return completion
    except BaseException:
        logger.finish(exit_code=1)
        raise


SUMMARY_KEYS = (
    "state_mcc",
    "block_purity",
    "normalized_prediction_mse",
    "online_state_probe/r2",
    "masks/state_edges",
    "masks/state_aligned_shd",
    "masks/action_edges",
    "masks/parameter_edges",
    "masks/parameter_aligned_shd",
    "geco/relative_prediction",
)


def summarize(directories: list[str]) -> list[dict[str, Any]]:
    """Read authenticated final validation metrics; no model or data is loaded."""
    rows: list[dict[str, Any]] = []
    for value in directories:
        directory = Path(value).resolve()
        completion = verify_completion(directory, kind="train")
        metrics = read_json(directory / "validation.json")["metrics"]
        rows.append(
            {
                "condition": completion["condition"],
                "seed": completion["seed"],
                "steps": completion["steps"],
                **{key: metrics.get(key) for key in SUMMARY_KEYS},
            }
        )
    return sorted(rows, key=lambda row: (row["condition"], row["seed"]))


def main() -> None:
    """Train one arm, fix the dense bound, or summarize completed endpoints."""
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
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--threads", type=int, default=1)
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--wandb-mode", default="disabled")
    train_parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    train_parser.add_argument("--wandb-entity")
    train_parser.add_argument("--wandb-group", default="masked-mlp-oracle")
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
        result = train(
            TrainConfig(
                output_dir=arguments.output_dir,
                data_dir=arguments.data_dir,
                condition=arguments.condition,
                profile=arguments.profile,
                seed=arguments.seed,
                steps=arguments.steps,
                calibration_dir=arguments.calibration_dir,
                device=arguments.device,
                threads=arguments.threads,
                wandb_mode=arguments.wandb_mode,
                wandb_project=arguments.wandb_project,
                wandb_entity=arguments.wandb_entity,
                wandb_group=arguments.wandb_group,
            ),
            resume=arguments.resume,
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
