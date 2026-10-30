"""Sparse masked-MLP arms initialized from each seed's completed dense endpoint.

In the fresh-start comparison the prediction bound was reached only after about
60-70k updates, so sparsification ran for a small part of the budget. This
adapter loads the seed-matched dense endpoint's online encoder, EMA target, trunk
and readouts, adds fresh mask logits, and then spends the whole fixed budget on
the sparse objective with a fresh AdamW state, GECO controller and group sampler.
Every arm therefore starts from the same mixed (O(6)-chance) chart as its dense
reference. ``run.py`` and ``model.py`` are imported unchanged, so the dense
endpoints and their calibration stay authenticated under their own identity.
"""

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from diagnostics.masked_mlp_oracle import run
from diagnostics.masked_mlp_oracle.model import build_control
from diagnostics.shared.data import load_data
from diagnostics.shared.io import read_json, save_torch, sha
from diagnostics.shared.oracle import EvaluationResult, forward_loss, objective_config
from diagnostics.shared.sampling import GroupSampler, resolve_device
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.lagrangian import GECOConfig
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import TrainingObjective, seed_everything

SCHEMA = "masked-mlp-oracle-from-dense-v1"
MASK_LOGITS = frozenset(f"predictor.{name}_logits" for name in ("state", "action", "parameter"))


def code_identity() -> str:
    """Bind the unchanged fresh-start identity plus this adapter."""
    return canonical_fingerprint(
        {"parent_code_identity": run.code_identity(), "from_dense.py": sha(Path(__file__))}
    )


@dataclass(frozen=True)
class FromDenseConfig:
    """One sparse arm; ``train`` carries every unchanged fresh-start setting."""

    train: run.TrainConfig
    init_dir: str
    geco_step_size: float = 0.1
    geco_lambda_init: float = 1000.0
    sparse_steps: int | None = None

    @property
    def budget(self) -> int:
        """Sparse updates; ``train.steps`` stays the dense reference budget."""
        return self.train.steps if self.sparse_steps is None else self.sparse_steps

    def __post_init__(self) -> None:
        """Only the sparse arms start from a dense endpoint."""
        if self.train.condition == "dense":
            raise ValueError("a dense endpoint initializes only baseline or full arms")
        if type(self.budget) is not int or self.budget < 1:
            raise ValueError("sparse_steps must be a positive integer")
        if self.train.profile == "exploratory" and self.budget not in (100_000, 200_000):
            raise ValueError("exploratory sparse budget is 100000 or 200000 updates")
        if not math.isfinite(self.geco_step_size) or self.geco_step_size <= 0:
            raise ValueError("geco_step_size must be finite and positive")
        # GECOConfig validates the initial multiplier against its log bounds.
        GECOConfig(
            enabled=True,
            tau=1.0,
            step_size=self.geco_step_size,
            lambda_init=self.geco_lambda_init,
        )


def verify_completion(
    directory: Path, *, kind: str | None = None, require_current_source: bool = False
) -> dict[str, Any]:
    """Authenticate all declared output files of a dense-initialized arm."""
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


def _dense_endpoint(config: FromDenseConfig, data_identity: str) -> dict[str, Any]:
    """Authenticate the seed-matched dense endpoint and return its record and weights."""
    directory = Path(config.init_dir).resolve()
    completion = run.verify_completion(directory, kind="train", require_current_source=True)
    if completion["condition"] != "dense":
        raise ValueError("initialization must be a dense endpoint")
    if completion["seed"] != config.train.seed:
        raise ValueError("dense endpoint seed differs from the sparse arm seed")
    if completion["profile"] != config.train.profile:
        raise ValueError("dense endpoint profile differs")
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
            "identity": completion["identity"],
            "completion_sha256": sha(directory / "completion.json"),
            "checkpoint_sha256": sha(directory / "last.pt"),
            "seed": completion["seed"],
            "steps": completion["steps"],
        },
        "state": checkpoint["model"],
    }


def _load_dense(model: SCJEPA, state: dict[str, Tensor]) -> None:
    """Copy every dense tensor; only the sparse arm's new mask logits may be absent."""
    result = model.load_state_dict(state, strict=False)
    unexpected = cast(list[str], result.unexpected_keys)  # pyright: ignore[reportUnknownMemberType]
    missing = cast(list[str], result.missing_keys)  # pyright: ignore[reportUnknownMemberType]
    if unexpected or not set(missing) <= MASK_LOGITS:
        raise ValueError("dense endpoint and sparse model differ beyond mask logits")


def _log(logger: WandbLogger, output: Path, step: int, metrics: dict[str, float]) -> None:
    if any(not math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("nonfinite reported metric")
    logger.log(step, metrics)
    with (output / "history.jsonl").open("a") as stream:
        stream.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")


def train(config: FromDenseConfig, *, resume: bool = False) -> dict[str, Any]:
    """Hold an exclusive output lock while training or resuming exactly one arm."""
    output = Path(config.train.output_dir).resolve()
    with file_lock(output.parent / f".{output.name}.lock"):
        return _train(config, output, resume=resume)


def _train(config: FromDenseConfig, output: Path, *, resume: bool) -> dict[str, Any]:
    settings = config.train
    budget = config.budget
    torch.set_num_threads(settings.threads)
    device = resolve_device(settings.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not resume:
        run._fresh_output(output)  # pyright: ignore[reportPrivateUsage]
    elif not (output / "last.pt").is_file():
        raise FileNotFoundError("resume requires last.pt")
    data, cache = load_data(Path(settings.data_dir))
    if cache["settings"]["profile"] != settings.profile:
        raise ValueError("training and cache profiles differ")
    if settings.batch_size > len(data["train"]):
        raise ValueError("training data must contain a full group minibatch")
    calibration = run._calibration(settings, str(cache["identity"]))  # pyright: ignore[reportPrivateUsage]
    if calibration is None:
        raise ValueError("sparse arms require the dense calibration")
    tau_prediction = float(calibration["tau_prediction"])
    endpoint = _dense_endpoint(config, str(cache["identity"]))
    seed_everything(settings.seed)
    model = build_control(settings.condition, settings.seed)
    _load_dense(model, endpoint["state"])
    model = model.to(device)
    initials = run._initial_identities(model)  # pyright: ignore[reportPrivateUsage]
    controller = GECOConfig(
        enabled=True,
        tau=1.0,
        step_size=config.geco_step_size,
        lambda_init=config.geco_lambda_init,
    )
    objective = TrainingObjective(objective_config(settings.condition, 0.0), controller).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if any(parameter.requires_grad for parameter in model.target_encoder.parameters()):
        raise RuntimeError("target encoder must remain outside gradient optimization")
    optimizer = torch.optim.AdamW(parameters, lr=settings.lr, weight_decay=0)
    sampler = GroupSampler(len(data["train"]), settings.batch_size, settings.seed)
    source_identity = code_identity()
    scientific = run._scientific_config(settings) | {  # pyright: ignore[reportPrivateUsage]
        "initialization": (
            "seed-matched final dense endpoint (online/EMA encoder, trunk, readouts); "
            "fresh mask logits, AdamW, GECO and sampler"
        ),
        "sparse_steps": budget,
    }
    specification: dict[str, Any] = {
        "scientific_config": scientific,
        "candidate_settings": cache["settings"],
        "data_identity": cache["identity"],
        "cache_identity": canonical_fingerprint(cache),
        "code_identity": source_identity,
        "calibration_identity": calibration["identity"],
        "tau_prediction": tau_prediction,
        "geco": asdict(controller),
        "dense_endpoint": endpoint["record"],
        **initials,
    }
    identity = canonical_fingerprint(specification)
    record = {
        "schema": SCHEMA,
        "identity": identity,
        **specification,
        "config": {"train": asdict(settings), "init_dir": config.init_dir},
        "privileged_control": True,
        "physical_state_training": False,
        "parameter_encoder_training": False,
        "target_policy": "EMA .996 continued from the dense endpoint, no optimizer gradients",
        "normalized_mse_cap": run.NORMALIZED_MSE_CAP,
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
        if not 0 <= start <= budget:
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
            if completed["identity"] != identity or completed["steps"] != budget:
                raise ValueError("completed run differs from requested continuation")
            return completed
    else:
        save_torch(output / "initial.pt", {"model": model.state_dict(), **initials})
    initial_checkpoint_sha256 = sha(output / "initial.pt")
    record["initial_checkpoint_sha256"] = initial_checkpoint_sha256
    write_json(output / "config.json", record)
    logger = WandbLogger(
        WandbConfig(
            mode=settings.wandb_mode,
            project=settings.wandb_project,
            entity=settings.wandb_entity,
            group=settings.wandb_group,
            name=f"masked-mlp-paley-fromdense-{settings.condition}-s{settings.seed}",
            job_type="masked-mlp-oracle-from-dense",
            tags=["diagnostic", "oracle", "masked-mlp", "from-dense", settings.condition],
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
        return run._collect(model, data, settings, device, objective, tau_prediction)  # pyright: ignore[reportPrivateUsage]

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
        for step in range(start + 1, budget + 1):
            batch = data["train"].batch(sampler.next(), device)
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed((1 << 40) + settings.seed + step)  # pyright: ignore[reportUnknownMemberType]
            loss, terms, _ = forward_loss(
                model, batch, objective, forward_seed=settings.seed + step
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss at update {step}")
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            gradient = torch.nn.utils.clip_grad_norm_(
                parameters, settings.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            model.update_target()
            assert objective.geco is not None
            objective.geco.update(terms["prediction"].detach() / tau_prediction)
            if step % settings.log_every == 0 or step == budget:
                metrics = {f"loss/{key}": float(value.detach()) for key, value in terms.items()}
                metrics.update(
                    {
                        "loss/total": float(loss.detach()),
                        "train/gradient_norm": float(gradient),
                        "train/successful_updates": float(step),
                        "train/elapsed_seconds": time.monotonic() - started,
                        "geco/penalty_weight": float(objective.geco.penalty_weight),
                        "geco/log10_penalty_weight": -float(objective.geco.log_lambda)
                        / math.log(10),
                        "geco/moving_violation": float(objective.geco.ma_error),
                    }
                )
                for name, logits in run._predictor(model).learned_logits().items():  # pyright: ignore[reportPrivateUsage]
                    metrics[f"masks/{name}_open_probability"] = float(
                        torch.sigmoid(logits.detach()).mean()
                    )
                    metrics[f"masks/{name}_edges"] = float((logits.detach() > 0).sum())
                _log(logger, output, step, metrics)
            if step % settings.evaluate_every == 0 and step != budget:
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
            if step % settings.checkpoint_every == 0 or step == budget:
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
                budget,
                {f"{split}/{key}": value for key, value in result.metrics.items()},
            )
        _log(logger, output, budget, {"diagnostic/complete": 1.0})
        logger.finish()
        if code_identity() != source_identity:
            raise RuntimeError("numerical source changed during training; completion refused")
        completion: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": "train",
            "identity": identity,
            "code_identity": source_identity,
            "profile": settings.profile,
            "condition": settings.condition,
            "seed": settings.seed,
            "steps": budget,
            "successful_updates": budget,
            "skipped_updates": 0,
            "data_identity": cache["identity"],
            "calibration_identity": calibration["identity"],
            "tau_prediction": tau_prediction,
            "dense_endpoint": endpoint["record"],
            **initials,
            "wandb": logger.metadata,
            "hashes": {name: sha(output / name) for name in names},
        }
        write_json(output / "completion.json", completion)
        return completion
    except BaseException:
        logger.finish(exit_code=1)
        raise


def summarize(directories: list[str]) -> list[dict[str, Any]]:
    """Read authenticated start and final validation metrics; no model or data is loaded."""
    rows: list[dict[str, Any]] = []
    for value in directories:
        directory = Path(value).resolve()
        completion = verify_completion(directory, kind="train")
        final = read_json(directory / "validation.json")["metrics"]
        start = read_json(directory / "startup.json")["metrics"]["validation"]
        rows.append(
            {
                "condition": completion["condition"],
                "seed": completion["seed"],
                "steps": completion["steps"],
                "initial_state_mcc": start.get("state_mcc"),
                "initial_block_purity": start.get("block_purity"),
                **{key: final.get(key) for key in run.SUMMARY_KEYS},
            }
        )
    return sorted(rows, key=lambda row: (row["condition"], row["seed"]))


def main() -> None:
    """Train one dense-initialized sparse arm or summarize completed arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--data-dir", required=True)
    train_parser.add_argument("--init-dir", required=True)
    train_parser.add_argument("--calibration-dir", required=True)
    train_parser.add_argument("--condition", choices=("baseline", "full"), required=True)
    train_parser.add_argument("--profile", choices=("exploratory", "smoke"), default="exploratory")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--steps", type=int, default=100_000, help="dense reference budget")
    train_parser.add_argument("--sparse-steps", type=int, help="sparse budget (default: --steps)")
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--threads", type=int, default=1)
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--wandb-mode", default="disabled")
    train_parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    train_parser.add_argument("--wandb-entity")
    train_parser.add_argument("--wandb-group", default="masked-mlp-oracle-from-dense")
    train_parser.add_argument("--geco-step-size", type=float, default=0.1)
    train_parser.add_argument("--geco-lambda-init", type=float, default=1000.0)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("--runs", nargs="+", required=True)
    arguments = parser.parse_args()
    if arguments.command == "train":
        result = train(
            FromDenseConfig(
                run.TrainConfig(
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
                arguments.init_dir,
                geco_step_size=arguments.geco_step_size,
                geco_lambda_init=arguments.geco_lambda_init,
                sparse_steps=arguments.sparse_steps,
            ),
            resume=arguments.resume,
        )
    else:
        result = summarize(arguments.runs)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
