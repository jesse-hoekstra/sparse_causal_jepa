"""Compare query count and explicit consistency with frozen informative states.

Run from the repository root with python -m diagnostics.multiquery_control.run.
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
from torch import nn

from diagnostics.evaluate_context_variance import load_frozen_model
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, build_cache
from diagnostics.multiquery_control.evaluation import evaluate
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.multiquery_control.objective import FrozenStateFastObjective
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, source_fingerprint, write_json
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.factory import build_model
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng

CONDITIONS: dict[str, tuple[int, float]] = {
    "q3_c0": (3, 0.0),
    "q1_c0": (1, 0.0),
    "q3_c1": (3, 1.0),
    "q1_c1": (1, 1.0),
}


@dataclass(frozen=True)
class Config:
    """Fixed-budget paired comparison; counts denote independent theta groups."""

    checkpoint: str
    output_dir: str
    device: str = "cpu"
    seeds: tuple[int, ...] = (202610261301, 202610261302)
    conditions: tuple[str, ...] = tuple(CONDITIONS)
    steps: int = 30000
    train_groups: int = 8192
    calibration_groups: int = 512
    validation_groups: int = 256
    test_groups: int = 512
    evaluation_seed: int = 202610261401
    batch_size: int = 16
    lr: float = 0.001
    threads: int = 1
    log_every: int = 100
    evaluate_every: int = 1000
    checkpoint_every: int = 1000
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None
    wandb_group: str = "single-context-multiquery-consistency-control"

    def __post_init__(self) -> None:
        """Reject ineffective batch sizes and invalid experiment configurations."""
        for name in ("steps", "threads", "log_every", "evaluate_every", "checkpoint_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("train_groups", "calibration_groups", "validation_groups", "test_groups"):
            if getattr(self, name) < max(4, self.batch_size):
                raise ValueError(f"{name} must be at least max(4, batch_size)")
            if getattr(self, name) % self.batch_size:
                raise ValueError(f"{name} must be divisible by batch_size for matched SIGReg")
        if self.batch_size < 2 or not 0 < self.lr < float("inf"):
            raise ValueError("batch_size must be >=2 and lr finite and positive")
        if not self.seeds or min(self.seeds) < 0 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct nonnegative integers")
        if not self.conditions or len(set(self.conditions)) != len(self.conditions):
            raise ValueError("conditions must be nonempty and distinct")
        if any(name not in CONDITIONS for name in self.conditions):
            raise ValueError("unknown query/consistency condition")
        if self.evaluation_seed < 0 or torch.device(self.device).type not in {"cpu", "cuda"}:
            raise ValueError("need a nonnegative evaluation seed and CPU or CUDA device")


def sha(path: Path) -> str:
    """Fingerprint a trusted local artifact without loading its contents as code."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_hash(module: nn.Module) -> str:
    """Fingerprint frozen encoder tensor names, shapes, and values on either device."""
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frozen_hashes(model: SCJEPA) -> dict[str, str]:
    """Bind both online and target state encoders to their source tensors."""
    return {
        name: module_hash(getattr(model, name)) for name in ("online_encoder", "target_encoder")
    }


def code_identity() -> str:
    """Include disposable code as well as production code in resume compatibility."""
    digest = hashlib.sha256(source_fingerprint().encode())
    digest.update(Path(__file__).parents[1].joinpath("evaluate_context_variance.py").read_bytes())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    """Accept bare CUDA while ensuring set_device receives an indexed device."""
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device.index is None:  # pyright: ignore[reportUnnecessaryComparison]
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    return device


def atomic_save(path: Path, payload: dict[str, Any]) -> None:
    """Replace one complete checkpoint atomically, preserving its previous version on error."""
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def scalar_metrics(value: dict[str, Any], prefix: str) -> dict[str, float]:
    """Flatten metric dictionaries and short coordinate lists for W&B and JSON logging."""
    result: dict[str, float] = {}
    for key, item in value.items():
        name = f"{prefix}/{key}"
        if isinstance(item, dict):
            result.update(scalar_metrics(cast(dict[str, Any], item), name))
        elif isinstance(item, float | int):
            result[name] = float(item)
        elif isinstance(item, list) and len(cast(list[Any], item)) <= 6:
            for i, element in enumerate(cast(list[Any], item), 1):
                if isinstance(element, float | int):
                    if key.endswith("_ci95") and len(cast(list[Any], item)) == 2:
                        coordinate = "lower" if i == 1 else "upper"
                    elif key in {"linear_r2", "selected_test_r2", "test_r2", "validation_r2"}:
                        coordinate = f"theta_{i}"
                    else:
                        coordinate = f"coordinate_{i}"
                    result[f"{name}/{coordinate}"] = float(element)
    return result


def fresh_model(source: SCJEPA, model_config: dict[str, Any], seed: int) -> SCJEPA:
    """Initialize only q and the predictor afresh; retain both informative state encoders."""
    with preserve_rng(torch.device("cpu")):
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
        model = build_model(OmegaConf.create(model_config))
    model.online_encoder.load_state_dict(source.online_encoder.state_dict())
    model.target_encoder.load_state_dict(source.target_encoder.state_dict())
    model.online_encoder.requires_grad_(False).eval()
    model.target_encoder.requires_grad_(False).eval()
    assert frozen_hashes(model) == frozen_hashes(source)
    return model


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
            name: {"queries": q, "consistency_weight": c} for name, (q, c) in CONDITIONS.items()
        },
        "pairing": (
            "Same serialized fresh initialization, cached training groups, batch order, "
            "per-update RNG seed, AdamW and update budget within each seed; "
            "all arms run on same device."
        ),
        "scope": (
            "One context per inferred code. Frozen state encoders, no EMA, labels excluded "
            "from training. Dense prediction only; no state-MCC or sparsity comparison."
        ),
        "interpretation": (
            "Three queries reduce pairing noise but preserve expected prediction objective. "
            "Equal updates give three times as many prediction pairs. Prediction alone "
            "does not certify code invariance or injectivity."
        ),
        "selection": (
            "Final fixed-budget checkpoint, no label-driven checkpoint selection. Q3 saves "
            "an extra steps//3 checkpoint for secondary equal prediction-pair-count comparison."
        ),
    }
    write_json(protocol_path, protocol)
    return protocol


def train_condition(
    config: Config,
    protocol: dict[str, Any],
    train: TensorSplit,
    evaluation_data: EvaluationData,
    device: torch.device,
    condition: str,
    seed: int,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Train one paired arm, resuming its exact next batch and optimizer state if requested."""
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
        ):
            raise ValueError("Checkpoint is not from this paired arm")
        optimizer.load_state_dict(payload["optimizer"])
        start = int(payload["step"])
    queries, consistency = CONDITIONS[condition]
    objective_config = ObjectiveConfig(**protocol["objective_config"])
    objective_config.parameter_consistency_weight = consistency
    objective = FrozenStateFastObjective(objective_config).to(device)
    logger = WandbLogger(
        WandbConfig(
            mode=config.wandb_mode,
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"multiquery-{condition}-seed-{seed}",
            group=config.wandb_group,
            tags=["diagnostic", "frozen-states", "single-context", condition],
            job_type="multiquery-control",
        ),
        resolved_config={
            **asdict(config),
            "condition": condition,
            "queries": queries,
            "consistency_weight": consistency,
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
            loss, terms = objective(cached_multiquery_forward(model, batch, queries=queries))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss in {condition} at step {step}")
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError(f"Nonfinite gradient in {condition} at step {step}")
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            if step % config.log_every == 0 or step == config.steps:
                metrics = {f"train/{name}": float(value.detach()) for name, value in terms.items()}
                metrics.update(
                    {
                        "train/queries": float(queries),
                        "train/consistency_weight": consistency,
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
            if queries == 3 and step == config.steps // 3:
                save(step, dest / f"step_{step}.pt")
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
    """Run the selected factorial arms sequentially on one device and persist all results."""
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
            for condition in config.conditions:
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
                        condition,
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[202610261301, 202610261302])
    parser.add_argument(
        "--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS)
    )
    for name, default in (
        ("steps", 30000),
        ("train-groups", 8192),
        ("calibration-groups", 512),
        ("validation-groups", 256),
        ("test-groups", 512),
        ("evaluation-seed", 202610261401),
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
    parser.add_argument("--wandb-group", default="single-context-multiquery-consistency-control")
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    resume = bool(args.pop("resume"))
    args["checkpoint"] = str(Path(args["checkpoint"]).resolve())
    args["output_dir"] = str(Path(args["output_dir"]).resolve())
    args["seeds"], args["conditions"] = tuple(args["seeds"]), tuple(args["conditions"])
    run(Config(**args), resume=resume)


if __name__ == "__main__":
    main()
