"""Compare parameter SIGReg weights with fixed informative state encoders.

Run from the repository root with python -m diagnostics.parameter_sigreg_control.run.
This diagnostic trains fresh original MLP parameter encoders and dense predictors;
it never updates either state encoder, applies EMA, or pools inference contexts.
"""

import argparse
import copy
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from diagnostics.evaluate_context_variance import load_frozen_model
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, build_cache
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.multiquery_control.objective import FrozenStateFastObjective
from diagnostics.multiquery_control.run import (
    atomic_save,
    fresh_model,
    frozen_hashes,
    resolve_device,
    scalar_metrics,
    sha,
)
from diagnostics.parameter_sigreg_control.evaluation import code_scale_diagnostics, evaluate
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, source_fingerprint, write_json
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.factory import build_model
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng

PARAMETER_SIGREG_WEIGHTS = (0.4, 0.1, 0.0)


def condition_name(weight: float) -> str:
    """Name the three authorized parameter-SIGReg settings without float ambiguity."""
    names = {0.4: "sigreg_0p4", 0.1: "sigreg_0p1", 0.0: "sigreg_0"}
    if weight not in names:
        raise ValueError("parameter SIGReg weight must be 0.4, 0.1 or 0")
    return names[weight]


@dataclass(frozen=True)
class Config:
    """Fixed-budget paired comparison; counts denote independent theta groups."""

    checkpoint: str
    output_dir: str
    device: str = "cpu"
    seeds: tuple[int, ...] = (202610261301,)
    parameter_sigreg_weights: tuple[float, ...] = PARAMETER_SIGREG_WEIGHTS
    steps: int = 30000
    train_groups: int = 8192
    calibration_groups: int = 512
    validation_groups: int = 256
    test_groups: int = 512
    evaluation_seed: int = 202610261601
    batch_size: int = 16
    lr: float = 0.001
    threads: int = 1
    log_every: int = 100
    evaluate_every: int = 1000
    checkpoint_every: int = 1000
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None
    wandb_group: str = "frozen-states-parameter-sigreg-control"

    def __post_init__(self) -> None:
        """Reject ineffective batch sizes and invalid experiment configurations."""
        for name in ("steps", "threads", "log_every", "evaluate_every", "checkpoint_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.batch_size < 2:
            raise ValueError("batch_size must be >=2")
        for name in ("train_groups", "calibration_groups", "validation_groups", "test_groups"):
            if getattr(self, name) < max(4, self.batch_size):
                raise ValueError(f"{name} must be at least max(4, batch_size)")
            if getattr(self, name) % self.batch_size:
                raise ValueError(f"{name} must be divisible by batch_size for matched SIGReg")
        if not 0 < self.lr < float("inf"):
            raise ValueError("lr must be finite and positive")
        if not self.seeds or min(self.seeds) < 0 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct nonnegative integers")
        weights = self.parameter_sigreg_weights
        if not weights or len(set(weights)) != len(weights):
            raise ValueError("parameter SIGReg weights must be nonempty and distinct")
        for weight in weights:
            condition_name(weight)
        if self.evaluation_seed < 0 or torch.device(self.device).type not in {"cpu", "cuda"}:
            raise ValueError("need a nonnegative evaluation seed and CPU or CUDA device")


def code_identity() -> str:
    """Include disposable code as well as production code in resume compatibility."""
    digest = hashlib.sha256(source_fingerprint().encode())
    digest.update(Path(__file__).parents[1].joinpath("evaluate_context_variance.py").read_bytes())
    for directory in (Path(__file__).parent, Path(__file__).parents[1] / "multiquery_control"):
        for path in sorted(directory.glob("*.py")):
            digest.update(str(path.relative_to(Path(__file__).parents[1])).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare(config: Config, device: torch.device, *, resume: bool) -> dict[str, Any]:
    """Validate the source and lock paired inputs, initializations, and protocol before training."""
    out = Path(config.output_dir)
    source, observation_map, provenance = load_frozen_model(Path(config.checkpoint), device)
    record = cast(dict[str, Any], read_json(Path(config.checkpoint).parent / "run_config.json"))
    if (
        not isinstance(source.parameter_encoder, TransitionParameterEncoder)
        or source.parameter_source != "learned"
        or source.context_len != 10
        or source.predictor.history_len != 1
        or source.predictor.state_dim != 6
        or source.predictor.param_dim != 6
        or source.predictor.action_dim != 16
        or not source.autoregressive
        or not all(layer.dense for layer in source.predictor.layers)
    ):
        raise ValueError("Use the original learned-parameter K10 MLP dense checkpoint")
    objective = ObjectiveConfig(**record["train"]["objective"])
    if (
        objective.parameter_regularizer != "sigreg"
        or objective.state_regularizer != "sigreg"
        or objective.parameter_regularizer_weight != 0.4
        or objective.state_regularizer_weight != 0.4
        or any(
            (
                objective.state_sparsity_weight,
                objective.action_sparsity_weight,
                objective.parameter_sparsity_weight,
            )
        )
    ):
        raise ValueError("Use the dense SIGReg0.4 source with zero sparsity weights")
    # Consistency is held fixed across every arm, regardless of the source record.
    objective.parameter_consistency_weight = 1.0
    immutable = asdict(config)
    # Logging choices may change during recovery; scientific settings and device may not.
    for key in ("wandb_mode", "wandb_project", "wandb_entity", "output_dir", "checkpoint"):
        immutable.pop(key)
    identity = canonical_fingerprint(
        {
            "settings": immutable,
            "source": provenance["sha256"],
            "code": code_identity(),
            "torch": str(torch.__version__),
        }
    )
    protocol_path = out / "protocol.json"
    if resume:
        protocol = cast(dict[str, Any], read_json(protocol_path))
        if protocol["identity"] != identity:
            raise ValueError(
                "Resume settings, source, software, or code differ; use a fresh output directory"
            )
        for name, expected in protocol["artifacts"].items():
            if sha(out / name) != expected:
                raise ValueError(f"Input artifact changed: {name}")
        return protocol
    if protocol_path.exists() or any(p.name != ".lock" for p in out.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {out}; use --resume or a new directory"
        )
    print(
        json.dumps({"preparing_frozen_cache": True, "train_groups": config.train_groups}),
        flush=True,
    )
    train, evaluation_data = build_cache(
        source,
        observation_map,
        record["data"],
        train_groups=config.train_groups,
        calibration_groups=config.calibration_groups,
        validation_groups=config.validation_groups,
        test_groups=config.test_groups,
        eval_seed=config.evaluation_seed,
        batch_size=32,
    )
    if set(train) != {"online", "target", "actions"}:
        raise ValueError("Training cache must contain only frozen encodings and observed actions")
    atomic_save(out / "train_inputs.pt", train)
    atomic_save(out / "evaluation_data.pt", evaluation_data)
    source.cpu()
    initializations: dict[str, Any] = {}
    for seed in config.seeds:
        model = fresh_model(source, record["model"], seed)
        path = out / f"initial_seed_{seed}.pt"
        atomic_save(
            path, {"model": model.state_dict(), "seed": seed, "source_sha256": provenance["sha256"]}
        )
        initializations[str(seed)] = {"path": path.name, "sha256": sha(path)}
    protocol = {
        "identity": identity,
        "config": asdict(config),
        "source": provenance,
        "model_config": record["model"],
        "objective_config": asdict(objective),
        "initializations": initializations,
        "frozen_hashes": frozen_hashes(source),
        "code_identity": code_identity(),
        "torch": str(torch.__version__),
        "artifacts": {
            name: sha(out / name)
            for name in [
                "train_inputs.pt",
                "evaluation_data.pt",
                *[item["path"] for item in initializations.values()],
            ]
        },
        "conditions": {
            condition_name(weight): {
                "queries": 1,
                "consistency_weight": 1.0,
                "parameter_regularizer_weight": weight,
            }
            for weight in config.parameter_sigreg_weights
        },
        "pairing": (
            "Same serialized fresh initialization, cached training groups, batch order, "
            "per-update RNG seed, AdamW and update budget within each seed; "
            "all arms run on same device. Each step reseeds before forward; disabling SIGReg "
            "does not shift any subsequent update's donor or projection draws."
        ),
        "scope": (
            "One context per inferred code. Frozen state encoders, no EMA, labels excluded "
            "from training. Dense prediction only; no state-MCC or sparsity comparison."
        ),
        "interpretation": (
            "Only parameter SIGReg weight changes. At zero, consistency can decrease through "
            "code shrinkage and inverse predictor rescaling; use normalized variance, joint "
            "parameter readouts and intervention/shuffle tests rather than raw consistency alone."
        ),
        "selection": (
            "Final fixed-budget checkpoint, no label-driven checkpoint selection or early stopping."
        ),
    }
    write_json(protocol_path, protocol)
    return protocol


def arm_objective(protocol: dict[str, Any], weight: float) -> ObjectiveConfig:
    """Change only parameter SIGReg weight while retaining a fixed consistency signal."""
    condition_name(weight)
    config = ObjectiveConfig(**protocol["objective_config"])
    config.parameter_regularizer_weight = weight
    config.parameter_consistency_weight = 1.0
    return config


def train_condition(
    config: Config,
    protocol: dict[str, Any],
    train: TensorSplit,
    evaluation_data: EvaluationData,
    device: torch.device,
    parameter_sigreg_weight: float,
    seed: int,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Train one paired arm, resuming its exact next batch and optimizer state if requested."""
    condition = condition_name(parameter_sigreg_weight)
    if parameter_sigreg_weight not in config.parameter_sigreg_weights or seed not in config.seeds:
        raise ValueError("Arm is not part of the locked configuration")
    out = Path(config.output_dir)
    dest = out / f"seed-{seed}" / condition
    dest.mkdir(parents=True, exist_ok=True)
    result_path, last = dest / "results.json", dest / "last.pt"
    if (dest / "completed.json").exists():
        completed = read_json(dest / "completed.json")
        if (
            completed["identity"] != protocol["identity"]
            or completed["checkpoint_sha256"] != sha(last)
            or completed["results_sha256"] != sha(result_path)
        ):
            raise ValueError("Completed arm artifacts changed")
        print(json.dumps({"skipping_completed": condition, "seed": seed}), flush=True)
        return cast(dict[str, Any], read_json(result_path))
    initial = out / protocol["initializations"][str(seed)]["path"]
    if sha(initial) != protocol["initializations"][str(seed)]["sha256"]:
        raise ValueError("Paired initialization artifact changed")
    payload = cast(
        dict[str, Any],
        torch.load(
            last if resume and last.exists() else initial, map_location="cpu", weights_only=False
        ),
    )
    model = build_model(cast(DictConfig, OmegaConf.create(protocol["model_config"])))
    model.load_state_dict(payload["model"])
    model.online_encoder.requires_grad_(False).eval()
    model.target_encoder.requires_grad_(False).eval()
    model.to(device)
    if frozen_hashes(model) != protocol["frozen_hashes"]:
        raise ValueError("State encoder tensors differ from the selected source")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=0.0)
    start = 0
    if "step" in payload:
        if (
            payload["identity"] != protocol["identity"]
            or payload["condition"] != condition
            or payload["seed"] != seed
            or payload["initial_sha256"] != sha(initial)
            or payload["parameter_regularizer_weight"] != parameter_sigreg_weight
            or payload["consistency_weight"] != 1.0
            or payload["queries"] != 1
        ):
            raise ValueError("Checkpoint is not from this paired arm")
        optimizer.load_state_dict(payload["optimizer"])
        start = int(payload["step"])
    queries, consistency = 1, 1.0
    objective_config = arm_objective(protocol, parameter_sigreg_weight)
    objective = FrozenStateFastObjective(objective_config).to(device)
    logger = WandbLogger(
        WandbConfig(
            mode=config.wandb_mode,
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"param-{condition.replace('_', '-')}-seed-{seed}",
            group=config.wandb_group,
            tags=["diagnostic", "frozen-states", "single-context", condition],
            job_type="parameter-sigreg-control",
        ),
        resolved_config={
            **asdict(config),
            "condition": condition,
            "seed": seed,
            "queries": queries,
            "consistency_weight": consistency,
            "parameter_regularizer_weight": parameter_sigreg_weight,
            "state_regularizer_weight": objective_config.state_regularizer_weight,
            "source_sha256": protocol["source"]["sha256"],
        },
        dataset_identity=protocol["identity"],
        out_dir=dest,
        resume_state=payload.get("wandb"),
    )
    started = time.perf_counter()
    elapsed = float(payload.get("elapsed_seconds", 0.0))

    def save(step: int, path: Path) -> None:
        if frozen_hashes(model) != protocol["frozen_hashes"]:
            raise RuntimeError("Frozen state encoder changed during training")
        atomic_save(
            path,
            {
                "identity": protocol["identity"],
                "condition": condition,
                "seed": seed,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "source_sha256": protocol["source"]["sha256"],
                "initial_sha256": sha(initial),
                "queries": queries,
                "consistency_weight": consistency,
                "parameter_regularizer_weight": parameter_sigreg_weight,
                "frozen_hashes": protocol["frozen_hashes"],
                "wandb": logger.metadata,
                "elapsed_seconds": elapsed + time.perf_counter() - started,
            },
        )

    try:
        if start == 0:
            save(0, last)
        batches_per_epoch = config.train_groups // config.batch_size
        current_epoch, batches = -1, []
        for step in range(start + 1, config.steps + 1):
            epoch, position = divmod(step - 1, batches_per_epoch)
            if epoch != current_epoch:
                batches = list(
                    GlobalBatchSampler(config.train_groups, config.batch_size, seed, epoch)
                )
                current_epoch = epoch
            indices = batches[position]
            batch = {name: values[indices].to(device) for name, values in train.items()}
            torch.manual_seed(seed * 100000 + step)  # pyright: ignore[reportUnknownMemberType]
            model.train()
            model.online_encoder.eval()
            model.target_encoder.eval()
            optimizer.zero_grad(set_to_none=True)
            output = cached_multiquery_forward(model, batch, queries=queries)
            loss, terms = objective(output)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss in {condition} at step {step}")
            should_log = step % config.log_every == 0 or step == config.steps
            # Both code and projection scale refer to this forward's weights.
            scales = code_scale_diagnostics(model, output.parameters) if should_log else {}
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError(f"Nonfinite gradient in {condition} at step {step}")
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            if should_log:
                metrics = {f"train/{name}": float(value.detach()) for name, value in terms.items()}
                metrics.update({f"train/{name}": value for name, value in scales.items()})
                metrics.update(
                    {
                        "train/queries": float(queries),
                        "train/consistency_weight": consistency,
                        "train/parameter_regularizer_weight": parameter_sigreg_weight,
                        "train/gradient_norm": float(norm),
                        "train/elapsed_seconds": elapsed + time.perf_counter() - started,
                    }
                )
                logger.log(step, metrics)
            if step % config.evaluate_every == 0 or step == config.steps:
                with preserve_rng(device):
                    monitored = evaluate(
                        copy.deepcopy(model).cpu().eval(),
                        evaluation_data,
                        objective_config,
                        batch_size=config.batch_size,
                        final=False,
                    )
                write_json(dest / "validation.json", {"step": step, "metrics": monitored})
                logger.log(step, scalar_metrics(monitored, "validation"))
            if step % config.checkpoint_every == 0 or step == config.steps:
                save(step, last)
        with preserve_rng(device):
            metrics = evaluate(
                copy.deepcopy(model).cpu().eval(),
                evaluation_data,
                objective_config,
                batch_size=config.batch_size,
                final=True,
            )
        result = {
            "condition": condition,
            "seed": seed,
            "steps": config.steps,
            "queries": queries,
            "consistency_weight": consistency,
            "parameter_regularizer_weight": parameter_sigreg_weight,
            "metrics": metrics,
            "source_sha256": protocol["source"]["sha256"],
            "initial_sha256": sha(initial),
            "checkpoint_sha256": sha(last),
            "identity": protocol["identity"],
            "elapsed_seconds": elapsed + time.perf_counter() - started,
            "frozen_hashes": frozen_hashes(model),
        }
        write_json(result_path, result)
        logger.log(config.steps, scalar_metrics(metrics, "test"))
        write_json(
            dest / "completed.json",
            {
                "identity": protocol["identity"],
                "checkpoint_sha256": sha(last),
                "results_sha256": sha(result_path),
            },
        )
        logger.finish()
        return result
    except BaseException:
        logger.finish(exit_code=1)
        raise


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Run the selected parameter-SIGReg arms sequentially and persist all results."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use ordinary python on one GPU; this diagnostic does not use torchrun")
    torch.set_num_threads(config.threads)
    device = resolve_device(config.device)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with file_lock(out / ".lock"):
        protocol = prepare(config, device, resume=resume)
        train = cast(
            TensorSplit, torch.load(out / "train_inputs.pt", map_location="cpu", weights_only=True)
        )
        evaluation_data = cast(
            EvaluationData,
            torch.load(out / "evaluation_data.pt", map_location="cpu", weights_only=True),
        )
        if set(train) != {"online", "target", "actions"}:
            raise ValueError("Unexpected fields in label-free training cache")
        results: list[dict[str, Any]] = []
        for seed in config.seeds:
            for weight in config.parameter_sigreg_weights:
                condition = condition_name(weight)
                print(
                    json.dumps({"starting": condition, "seed": seed, "device": str(device)}),
                    flush=True,
                )
                results.append(
                    train_condition(
                        config,
                        protocol,
                        train,
                        evaluation_data,
                        device,
                        weight,
                        seed,
                        resume=resume,
                    )
                )
                write_json(
                    out / "comparison.json", {"identity": protocol["identity"], "results": results}
                )
    return {"identity": protocol["identity"], "results": results}


def main() -> None:
    """Parse the portable server command; a new output directory is required unless resuming."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[202610261301])
    parser.add_argument(
        "--parameter-sigreg-weights",
        nargs="+",
        type=float,
        choices=list(PARAMETER_SIGREG_WEIGHTS),
        default=list(PARAMETER_SIGREG_WEIGHTS),
    )
    for name, default in (
        ("steps", 30000),
        ("train-groups", 8192),
        ("calibration-groups", 512),
        ("validation-groups", 256),
        ("test-groups", 512),
        ("evaluation-seed", 202610261601),
        ("batch-size", 16),
        ("threads", 1),
        ("log-every", 100),
        ("evaluate-every", 1000),
        ("checkpoint-every", 1000),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="disabled"
    )
    parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="frozen-states-parameter-sigreg-control")
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    resume = bool(args.pop("resume"))
    args["checkpoint"] = str(Path(args["checkpoint"]).resolve())
    args["output_dir"] = str(Path(args["output_dir"]).resolve())
    args["seeds"] = tuple(args["seeds"])
    args["parameter_sigreg_weights"] = tuple(args["parameter_sigreg_weights"])
    run(Config(**args), resume=resume)


if __name__ == "__main__":
    main()
