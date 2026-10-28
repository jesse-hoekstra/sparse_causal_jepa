"""Fit the fixed-chart MLP comparator and authenticate its untrained conjugate."""

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor

from diagnostics.chart_sparsity.core import Chart, chart_metadata, transform_data
from diagnostics.chart_sparsity.run import (
    code_identity as chart_code_identity,
)
from diagnostics.chart_sparsity.run import load_data, read_json, save_torch, sha
from diagnostics.mlp_chart_control.core import (
    chart_transport,
    closure_audit,
    conjugate,
    evaluate,
    rollout,
)
from diagnostics.mlp_oracle_code_control.model import MLPPredictor
from diagnostics.oracle_state_predictor.run import GroupSampler, resolve_device, write_json
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.training import seed_everything
from scjepa.training.logging import WandbConfig, WandbLogger

SCHEMA = "fixed-chart-mlp-v1"
PARAMETER_COUNT = 52_123
QUALITY_CAP = 0.02
Profile = Literal["scientific", "smoke"]


@dataclass(frozen=True)
class TrainConfig:
    """One fresh MLP fit with a fixed endpoint and supplied physical parameters."""

    output_dir: str
    data_dir: str
    chart: Chart
    profile: Profile = "scientific"
    seed: int = 0
    steps: int = 100_000
    batch_size: int = 16
    lr: float = 0.001
    grad_clip: float = 1.0
    device: str = "cpu"
    threads: int = 1
    log_every: int = 100
    evaluate_every: int = 1000
    checkpoint_every: int = 1000
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None
    wandb_group: str = "diagnostic-fixed-chart-mlp"

    def __post_init__(self) -> None:
        """Keep scientific settings fixed while permitting explicit tiny CPU fixtures."""
        if self.profile not in ("scientific", "smoke"):
            raise ValueError("profile must be scientific or smoke")
        if self.chart not in ("physical", "mixed") or self.seed not in (0, 1):
            raise ValueError("require physical/mixed chart and paired training seed 0 or 1")
        if self.profile == "scientific" and (
            self.steps,
            self.batch_size,
            self.lr,
            self.grad_clip,
        ) != (100_000, 16, 0.001, 1.0):
            raise ValueError("scientific budget and optimizer are fixed by protocol")
        for name in (
            "steps",
            "batch_size",
            "threads",
            "log_every",
            "evaluate_every",
            "checkpoint_every",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if any(not math.isfinite(value) or value <= 0 for value in (self.lr, self.grad_clip)):
            raise ValueError("learning rate and gradient clip must be finite and positive")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def code_identity() -> str:
    """Bind the reused numerical implementation and this control's training sources."""
    digest = hashlib.sha256(chart_code_identity().encode())
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "mlp_chart_control/core.py",
        "mlp_chart_control/run.py",
        "mlp_oracle_code_control/model.py",
    ):
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def verify_completion(directory: Path, *, require_current_source: bool = False) -> dict[str, Any]:
    """Authenticate a finished MLP run without requiring its original machine."""
    result = read_json(directory / "completion.json")
    if result.get("schema") != SCHEMA or result.get("kind") != "train":
        raise ValueError("completion schema or stage differs")
    if require_current_source and result.get("code_identity") != code_identity():
        raise ValueError("completion numerical source identity differs")
    hashes = result.get("hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("completion lacks artifact hashes")
    for name, expected in cast(dict[str, str], hashes).items():
        if Path(name).name != name or sha(directory / name) != expected:
            raise ValueError(f"artifact hash differs: {name}")
    return result


def _scientific_config(config: TrainConfig) -> dict[str, Any]:
    result = asdict(config)
    for name in (
        "output_dir",
        "data_dir",
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
        result.pop(name)
    return result | {
        "architecture": "28-211-211-6 ReLU; biased linear layers; no normalization",
        "parameter_count": PARAMETER_COUNT,
        "objective": "attached two-step raw forecast MSE",
        "weight_decay": 0.0,
        "scheduler": "constant",
    }


def _state_identity(model: MLPPredictor) -> str:
    state = cast(dict[str, Tensor], model.state_dict())
    return canonical_fingerprint({name: tensor_fingerprint(value) for name, value in state.items()})


def _checkpoint(path: Path) -> dict[str, Any]:
    result = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))
    if result.get("schema") != SCHEMA:
        raise ValueError("checkpoint schema differs")
    return result


def _log(logger: WandbLogger, output: Path, step: int, metrics: dict[str, float]) -> None:
    logger.log(step, metrics)
    with (output / "history.jsonl").open("a") as stream:
        stream.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")


def train(config: TrainConfig, *, resume: bool = False) -> dict[str, Any]:
    """Train only the native-chart predictor; conjugated endpoint copies never learn."""
    torch.set_num_threads(config.threads)
    device = resolve_device(config.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(config.output_dir).resolve()
    if not resume:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"output is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
    elif not (output / "last.pt").is_file():
        raise FileNotFoundError("resume requires a last.pt checkpoint")

    physical, cache = load_data(Path(config.data_dir))
    if cache.get("profile") != config.profile:
        raise ValueError("training and cache profiles differ")
    if config.profile == "scientific" and (
        tuple(len(physical[split]) for split in ("train", "calibration", "validation", "test"))
        != (8192, 512, 256, 512)
        or any(item.views != 10 or item.config.root_seed != 0 for item in physical.values())
    ):
        raise ValueError("scientific cache dimensions, views or data seed differ")
    if config.batch_size > len(physical["train"]):
        raise ValueError("training must contain a full parameter-group minibatch")
    data = {
        split: transform_data(physical[split], config.chart)
        for split in ("train", "calibration", "validation")
    }
    seed_everything(config.seed)
    model = MLPPredictor().to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise ValueError("declared MLP architecture changed")
    initial_identity = _state_identity(model)
    source_identity = code_identity()
    specification = {
        "scientific_config": _scientific_config(config),
        "data_identity": cache["identity"],
        "data_sha256": cache["hashes"]["data.pt"],
        "cache_code_identity": cache["code_identity"],
        "code_identity": source_identity,
        "initial_state_identity": initial_identity,
    }
    identity = canonical_fingerprint(specification)
    record = {
        "schema": SCHEMA,
        "identity": identity,
        **specification,
        "config": asdict(config),
        "chart_definition": chart_metadata(config.chart),
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "privileged_control": True,
        "test_policy": "no test inference; fixed final calibration and validation only",
        "attention_penalty": "not applicable: this MLP has no attention logits or gates",
        "quality_cap": QUALITY_CAP,
        "transport_policy": (
            "After fixed endpoint, analytically conjugate a copy's first/last linear layers; "
            "no optimizer, extra training, source mutation or forward-time coordinate wrapper"
        ),
    }
    if resume and (output / "completion.json").is_file():
        completed = verify_completion(output, require_current_source=True)
        if completed["identity"] != identity:
            raise ValueError("completed run provenance differs")
        return completed

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0)
    sampler = GroupSampler(len(data["train"]), config.batch_size, config.seed)
    restored: dict[str, Any] = {}
    start = 0
    if resume:
        restored = _checkpoint(output / "last.pt")
        if restored.get("identity") != identity:
            raise ValueError("checkpoint configuration, source or data differs")
        model.load_state_dict(restored["model"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        sampler.load_state_dict(restored["sampler"])
        start = int(restored["step"])
        if not 0 <= start <= config.steps:
            raise ValueError("checkpoint step exceeds its fixed budget")
        for state in optimizer.state.values():
            if int(cast(Tensor, state["step"])) != start:
                raise ValueError("Adam state and successful-update count disagree")
        torch.set_rng_state(restored["torch_rng"])
        random.setstate(restored["python_rng"])
        np.random.set_state(restored["numpy_rng"])  # noqa: NPY002 - restore seeded global RNG
        if device.type == "cuda":
            torch.cuda.set_rng_state(restored["cuda_rng"], device)
    write_json(output / "config.json", record)
    logger = WandbLogger(
        WandbConfig(
            mode=config.wandb_mode,
            project=config.wandb_project,
            entity=config.wandb_entity,
            group=config.wandb_group,
            name=f"mlp-chart-{config.chart}-s{config.seed}",
            job_type="fixed-chart-mlp",
            tags=["diagnostic", "oracle", "mlp", config.chart],
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
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sampler": sampler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),  # noqa: NPY002 - preserve seeded global RNG
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "wandb": logger.metadata,
            },
        )

    started = time.monotonic()
    try:
        if not resume:
            initial = evaluate(model, data["validation"], config.batch_size, device)
            write_json(output / "startup.json", initial.metrics)
            _log(logger, output, 0, {f"validation/{k}": v for k, v in initial.metrics.items()})
            save(0)
        model.train()
        for step in range(start + 1, config.steps + 1):
            batch = data["train"].batch(sampler.next(), device)
            optimizer.zero_grad(set_to_none=True)
            predicted = rollout(model, batch.initial_states, batch.parameters, batch.actions)
            loss = (predicted - batch.targets).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss at successful update {step}")
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            if step % config.log_every == 0 or step == config.steps:
                _log(
                    logger,
                    output,
                    step,
                    {
                        "loss/prediction": float(loss.detach()),
                        "loss/base": float(loss.detach()),
                        "loss/total": float(loss.detach()),
                        "train/successful_updates": float(step),
                        "train/elapsed_seconds": time.monotonic() - started,
                    },
                )
            if step % config.evaluate_every == 0 and step != config.steps:
                result = evaluate(model, data["validation"], config.batch_size, device)
                _log(
                    logger, output, step, {f"validation/{k}": v for k, v in result.metrics.items()}
                )
            if step % config.checkpoint_every == 0 or step == config.steps:
                save(step)

        names = ["config.json", "wandb.json", "startup.json", "history.jsonl", "last.pt"]
        native_identity = _state_identity(model)
        opposite: Chart = "mixed" if config.chart == "physical" else "physical"
        matrix = chart_transport(config.chart, opposite)
        transported = conjugate(model, matrix)
        save_torch(
            output / "transported.pt",
            {
                "schema": SCHEMA,
                "kind": "analytic_conjugate",
                "source_identity": identity,
                "source_checkpoint_sha256": sha(output / "last.pt"),
                "source_chart": config.chart,
                "target_chart": opposite,
                "matrix": matrix,
                "model": transported.state_dict(),
                "optimization_steps": 0,
            },
        )
        names.append("transported.pt")
        checks: dict[str, dict[str, float]] = {}
        quality: dict[str, bool] = {}
        for split in ("calibration", "validation"):
            native = evaluate(model, data[split], config.batch_size, device)
            other = evaluate(
                transported, transform_data(physical[split], opposite), config.batch_size, device
            )
            for prefix, result in ((split, native), (f"transported_{split}", other)):
                write_json(output / f"{prefix}.json", {"metrics": result.metrics})
                save_torch(output / f"{prefix}.pt", {"samples": result.samples})
                names += [f"{prefix}.json", f"{prefix}.pt"]
                _log(
                    logger,
                    output,
                    config.steps,
                    {f"{prefix}/{k}": v for k, v in result.metrics.items()},
                )
            checks[split] = closure_audit(native, other, matrix)
            quality[split] = native.metrics["prediction_mse"] <= QUALITY_CAP
            _log(
                logger,
                output,
                config.steps,
                {f"transport/{split}/{k}": v for k, v in checks[split].items()},
            )
        if _state_identity(model) != native_identity:
            raise RuntimeError("conjugation or evaluation mutated the trained predictor")
        write_json(
            output / "transport.json",
            {
                "source_chart": config.chart,
                "target_chart": opposite,
                "matrix": [[float(value) for value in row] for row in matrix],
                "source_model_identity": native_identity,
                "transported_model_identity": _state_identity(transported),
                "source_checkpoint_sha256": sha(output / "last.pt"),
                "optimization_steps": 0,
                "checks": checks,
                "interpretation": (
                    "Analytic same-architecture representational witness, not training"
                ),
            },
        )
        names.append("transport.json")
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
            "chart": config.chart,
            "seed": config.seed,
            "steps": config.steps,
            "successful_updates": config.steps,
            "skipped_updates": 0,
            "data_identity": cache["identity"],
            "initial_state_identity": initial_identity,
            "native_quality": quality,
            "closure_passed": True,
            "wandb": logger.metadata,
            "hashes": {name: sha(output / name) for name in names},
        }
        write_json(output / "completion.json", completion)
        return completion
    except BaseException:
        logger.finish(exit_code=1)
        raise


def main() -> None:
    """Run one fixed-budget arm; the caller owns detached execution and GPU allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    training = commands.add_parser("train")
    required = ("output_dir", "data_dir", "chart")
    for name in required:
        training.add_argument("--" + name.replace("_", "-"), required=True)
    for field in fields(TrainConfig):
        if field.name in required:
            continue
        default = field.default
        training.add_argument(
            "--" + field.name.replace("_", "-"),
            default=default,
            type=type(default) if isinstance(default, int | float) else str,
        )
    training.add_argument("--resume", action="store_true")
    arguments = vars(parser.parse_args())
    arguments.pop("stage")
    resume = bool(arguments.pop("resume"))
    result = train(TrainConfig(**arguments), resume=resume)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
