"""Train fresh oracle-parameter controls with learned online and EMA state encoders."""

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn

from diagnostics.oracle_state_learning.core import (
    Condition,
    EvaluationResult,
    LiveStateData,
    build_control,
    evaluate,
    forward_loss,
    objective_config,
)
from diagnostics.oracle_state_learning.data import (
    PrepareConfig,
    code_identity,
    load_data,
    prepare,
    read_json,
    save_torch,
    sha,
)
from diagnostics.oracle_state_predictor.run import GroupSampler, resolve_device
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.eval.recovery import (
    CoordinateMatch,
    fit_coordinate_match,
    mean_correlation_coefficient,
    pearson_matrix,
)
from scjepa.eval.state_probes import fit_state_probe, state_probe_metrics
from scjepa.experiments.storage import file_lock, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.lagrangian import GECOConfig
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import TrainingObjective, seed_everything

SCHEMA = "oracle-state-learning-v1"
NORMALIZED_MSE_CAP = 0.05
type Profile = Literal["scientific", "smoke"]


@dataclass(frozen=True)
class TrainConfig:
    """One fresh fixed-budget fit; sparse arms read a bound but no parent weights."""

    output_dir: str
    data_dir: str
    condition: Condition
    profile: Profile = "scientific"
    seed: int = 0
    steps: int = 100_000
    batch_size: int = 16
    lr: float = 0.001
    grad_clip: float = 1.0
    logit_weight: float = 0.0001
    calibration_dir: str | None = None
    device: str = "cpu"
    threads: int = 1
    log_every: int = 100
    evaluate_every: int = 1000
    checkpoint_every: int = 1000
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None
    wandb_group: str = "diagnostic-oracle-state-learning"

    def __post_init__(self) -> None:
        """Freeze scientific settings and identify reduced fixtures explicitly."""
        if self.profile not in {"scientific", "smoke"}:
            raise ValueError("profile must be scientific or smoke")
        if self.condition not in {"dense", "baseline", "full", "mlp"}:
            raise ValueError("unknown oracle state-learning condition")
        if self.seed not in (0, 1):
            raise ValueError("paired training seeds are fixed to zero and one")
        if self.profile == "scientific" and (
            self.steps,
            self.batch_size,
            self.lr,
            self.grad_clip,
            self.logit_weight,
        ) != (100_000, 16, 0.001, 1.0, 0.0001):
            raise ValueError("scientific budget, optimizer and logit weight are fixed")
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
        if not math.isfinite(self.logit_weight) or self.logit_weight < 0:
            raise ValueError("logit weight must be finite and nonnegative")
        if (self.condition in {"baseline", "full"}) != (self.calibration_dir is not None):
            raise ValueError("exactly the sparse arms require the shared calibration directory")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


@dataclass(frozen=True)
class CalibrateConfig:
    """Freeze the maximum final calibration base loss across both dense seeds."""

    output_dir: str
    dense_dirs: tuple[str, ...]
    profile: Profile = "scientific"

    def __post_init__(self) -> None:
        """Require the prospectively paired references, including in software fixtures."""
        if self.profile not in {"scientific", "smoke"} or len(self.dense_dirs) != 2:
            raise ValueError("calibration requires exactly two paired dense endpoints")


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


def _state_identity(module: nn.Module) -> str:
    state = cast(dict[str, object], module.state_dict())
    return canonical_fingerprint(
        {
            key: tensor_fingerprint(value) if isinstance(value, Tensor) else value
            for key, value in state.items()
        }
    )


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
        "initialization": "fresh paired native tensors; no pretrained weights",
        "parameter_source": "supplied raw ground-truth theta",
        "objective": asdict(objective_config(config.condition, config.logit_weight)),
        "ema_decay": 0.996,
        "weight_decay": 0.0,
        "scheduler": "constant",
        "state_encoder": "6-64-64-6 native VectorEncoder",
        "predictor": "28-211-211-6 ReLU MLP" if config.condition == "mlp" else "native SPARTAN",
        "forecast": "attached two-step prediction from online state9; EMA targets10/11",
        "test_policy": "no test inference"
        if config.condition == "dense"
        else "fixed endpoint only",
    }


def _finite_nonconstant(result: dict[str, Any]) -> bool:
    metrics = cast(dict[str, float], result["metrics"])
    return all(math.isfinite(value) for value in metrics.values()) and all(
        metrics[f"{prefix}/variance_min"] > 0 for prefix in ("online", "target")
    )


def calibrate(config: CalibrateConfig) -> dict[str, Any]:
    """Create an immutable common bound from calibration losses, never test or MCC."""
    output = Path(config.output_dir).resolve()
    with file_lock(output.parent / f".{output.name}.lock"):
        _fresh_output(output)
        references: list[dict[str, Any]] = []
        for value in config.dense_dirs:
            directory = Path(value).resolve()
            completion = verify_completion(directory, kind="train", require_current_source=True)
            if completion["condition"] != "dense" or completion["profile"] != config.profile:
                raise ValueError("calibration reference must be a matching-profile dense fit")
            record = read_json(directory / "config.json")
            calibration = read_json(directory / "calibration.json")
            if not _finite_nonconstant(calibration):
                raise ValueError("dense calibration has nonfinite or constant state encodings")
            loss = float(calibration["metrics"]["base"])
            if loss <= 0:
                raise ValueError("dense calibration base loss must be positive")
            references.append(
                {
                    "directory": str(directory),
                    "identity": completion["identity"],
                    "completion_sha256": sha(directory / "completion.json"),
                    "checkpoint_sha256": sha(directory / "last.pt"),
                    "seed": completion["seed"],
                    "profile": completion["profile"],
                    "data_identity": completion["data_identity"],
                    "initial_encoder_identity": completion["initial_encoder_identity"],
                    "initial_predictor_identity": completion["initial_predictor_identity"],
                    "base_loss": loss,
                    "scientific_config": record["scientific_config"],
                    "calibration_normalized_prediction_mse": calibration["metrics"][
                        "normalized_prediction_mse"
                    ],
                }
            )
        references.sort(key=lambda item: int(item["seed"]))
        if [item["seed"] for item in references] != [0, 1]:
            raise ValueError("calibration requires exactly dense seeds zero and one")
        if len({item["data_identity"] for item in references}) != 1:
            raise ValueError("dense references have different data identities")
        comparable = [dict(item["scientific_config"]) for item in references]
        for item in comparable:
            item.pop("seed")
        if comparable[0] != comparable[1]:
            raise ValueError("dense reference scientific settings differ beyond seed")
        record: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": "calibration",
            "profile": config.profile,
            "code_identity": code_identity(),
            "data_identity": references[0]["data_identity"],
            "tau": max(float(item["base_loss"]) for item in references),
            "references": references,
            "policy": "maximum final dense calibration base loss; no weights transferred",
            "quality_policy": (
                "normalized forecast MSE <= .05 informs interpretation, not admission"
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
    if record["identity"] != completion["identity"]:
        raise ValueError("calibration identity differs")
    if record["data_identity"] != data_identity or record["profile"] != config.profile:
        raise ValueError("calibration data or profile differs")
    if not math.isfinite(float(record["tau"])) or float(record["tau"]) <= 0:
        raise ValueError("calibrated bound must be finite and positive")
    return record


def recovery_metrics(
    calibration: EvaluationResult, result: EvaluationResult
) -> tuple[dict[str, float], dict[str, list[int]]]:
    """Fit assignment and a fixed ridge readout on calibration rows alone."""
    matches: dict[str, CoordinateMatch] = {}
    metrics: dict[str, float] = {}
    truth = calibration.samples["states"].reshape(-1, 6)
    scored_truth = result.samples["states"].reshape(-1, 6)
    for prefix in ("online", "target"):
        features = calibration.samples[f"{prefix}_states"].reshape(-1, 6)
        scored = result.samples[f"{prefix}_states"].reshape(-1, 6)
        match = fit_coordinate_match(features, truth, method="pearson")
        matches[prefix] = match
        key = "state_mcc" if prefix == "online" else "target_state_mcc"
        metrics[key] = mean_correlation_coefficient(match, scored, scored_truth)
        correlations = pearson_matrix(scored, scored_truth)
        metrics.update(
            {
                f"{prefix}_state_correlation/coordinate_{index + 1}": float(
                    correlations[index, column]
                )
                for index, column in enumerate(match.columns)
            }
        )
        probe = fit_state_probe(features, truth, ridge=1e-3)
        scores = state_probe_metrics(probe.predict(scored), scored_truth)
        metrics.update({f"{prefix}_state_probe/{name}": value for name, value in scores.items()})
        if prefix == "target":
            predicted = result.samples["predictions"].reshape(-1, 6)
            future_truth = result.samples["states"][:, :, 10:12].reshape(-1, 6)
            scores = state_probe_metrics(probe.predict(predicted), future_truth)
            metrics.update(
                {f"forecast_state_probe/{name}": value for name, value in scores.items()}
            )
    metrics["normalized_prediction_mse"] = result.metrics["forecast/normalized_mse"]
    metrics["forecast_quality_pass"] = float(
        metrics["normalized_prediction_mse"] <= NORMALIZED_MSE_CAP
    )
    return metrics, {f"{prefix}_columns": list(match.columns) for prefix, match in matches.items()}


def _collect(
    model: SCJEPA,
    data: dict[str, LiveStateData],
    config: TrainConfig,
    device: torch.device,
    objective: TrainingObjective,
    *,
    include_test: bool,
) -> tuple[dict[str, EvaluationResult], dict[str, list[int]]]:
    calibration = evaluate(model, data["calibration"], config.batch_size, device, objective)
    results = {"calibration": calibration}
    matching: dict[str, list[int]] = {}
    for split in (
        ("calibration", "validation", "test") if include_test else ("calibration", "validation")
    ):
        result = (
            calibration
            if split == "calibration"
            else evaluate(model, data[split], config.batch_size, device, objective)
        )
        metrics, matching = recovery_metrics(calibration, result)
        if objective.geco is not None:
            residual = result.metrics["base"] - objective.geco.tau
            metrics.update(
                {
                    "geco/tau": objective.geco.tau,
                    "geco/residual": residual,
                    "geco/feasible": float(residual <= 0),
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
    if config.profile == "scientific" and (
        tuple(len(data[split]) for split in ("train", "calibration", "validation", "test"))
        != (8192, 512, 256, 512)
        or any(item.config.root_seed != 0 or item.views != 10 for item in data.values())
    ):
        raise ValueError("scientific cache dimensions, seed or views differ")
    calibration = _calibration(config, str(cache["identity"]))
    seed_everything(config.seed)
    model = build_control(config.condition, config.seed).to(device)
    initials = {
        "initial_encoder_identity": _state_identity(model.online_encoder),
        "initial_predictor_identity": _state_identity(model.predictor),
        "initial_model_identity": _state_identity(model),
    }
    if calibration is not None:
        reference = next(item for item in calibration["references"] if item["seed"] == config.seed)
        for key in ("initial_encoder_identity", "initial_predictor_identity"):
            if initials[key] != reference[key]:
                raise ValueError(
                    "fresh sparse initialization differs from its paired dense reference"
                )
    controller = (
        GECOConfig(enabled=True, tau=float(calibration["tau"])) if calibration else GECOConfig()
    )
    objective = TrainingObjective(
        objective_config(config.condition, config.logit_weight), controller
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if any(parameter.requires_grad for parameter in model.target_encoder.parameters()):
        raise RuntimeError("target encoder must remain outside gradient optimization")
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=0)
    sampler = GroupSampler(len(data["train"]), config.batch_size, config.seed)
    source_identity = code_identity()
    specification: dict[str, Any] = {
        "scientific_config": _scientific_config(config),
        "data_identity": cache["identity"],
        "cache_identity": canonical_fingerprint(cache),
        "code_identity": source_identity,
        "calibration_identity": calibration["identity"] if calibration else None,
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
            name=f"oracle-state-{config.condition}-s{config.seed}",
            job_type="oracle-state-learning",
            tags=["diagnostic", "oracle", config.condition],
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

    started = time.monotonic()
    try:
        if not resume:
            initial, _ = _collect(model, data, config, device, objective, include_test=False)
            write_json(
                output / "startup.json",
                {
                    **initials,
                    "encoder_identity": initials["initial_encoder_identity"],
                    "predictor_identity": initials["initial_predictor_identity"],
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
            if objective.geco is not None:
                objective.geco.update(terms["base"])
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
                if objective.geco is not None:
                    metrics.update(
                        {
                            "geco/tau": objective.geco.tau,
                            "geco/penalty_weight": float(objective.geco.penalty_weight),
                            "geco/moving_violation": float(objective.geco.ma_error),
                        }
                    )
                _log(logger, output, step, metrics)
            if step % config.evaluate_every == 0 and step != config.steps:
                interim, _ = _collect(model, data, config, device, objective, include_test=False)
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
        final, matching = _collect(
            model, data, config, device, objective, include_test=config.condition != "dense"
        )
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
            **initials,
            "wandb": logger.metadata,
            "hashes": {name: sha(output / name) for name in names},
        }
        write_json(output / "completion.json", completion)
        return completion
    except BaseException:
        logger.finish(exit_code=1)
        raise


def main() -> None:
    """Run one isolated stage; orchestration owns detached execution and GPU allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage, cls, required in (
        ("prepare", PrepareConfig, {"output_dir"}),
        ("train", TrainConfig, {"output_dir", "data_dir", "condition"}),
        ("calibrate", CalibrateConfig, {"output_dir", "dense_dirs"}),
    ):
        command = stages.add_parser(stage)
        for field in fields(cls):
            name = "--" + field.name.replace("_", "-")
            if field.name == "dense_dirs":
                command.add_argument(
                    "--dense-dir", dest="dense_dirs", action="append", required=True
                )
            elif field.name in required:
                command.add_argument(name, required=True)
            elif isinstance(field.default, bool):
                command.add_argument(name, action="store_true", default=field.default)
            else:
                command.add_argument(
                    name,
                    default=field.default,
                    type=type(field.default) if isinstance(field.default, int | float) else str,
                )
        if stage in {"prepare", "train"}:
            command.add_argument("--resume", action="store_true")
    arguments = vars(parser.parse_args())
    stage = arguments.pop("stage")
    resume = bool(arguments.pop("resume", False))
    if stage == "prepare":
        result = prepare(PrepareConfig(**arguments), resume=resume)
    elif stage == "train":
        result = train(TrainConfig(**arguments), resume=resume)
    else:
        arguments["dense_dirs"] = tuple(arguments["dense_dirs"])
        result = calibrate(CalibrateConfig(**arguments))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
