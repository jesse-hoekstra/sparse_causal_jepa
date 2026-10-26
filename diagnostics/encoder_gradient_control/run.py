"""Continue prediction weight one with three matched encoder-gradient controls."""

import argparse
import copy
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.encoder_gradient_control.source import (
    SourceBundle,
    build_source_model,
    load_source,
)
from diagnostics.encoder_gradient_control.updates import backward_and_clip
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.multiquery_control.run import (
    atomic_save,
    frozen_hashes,
    resolve_device,
    scalar_metrics,
    sha,
)
from diagnostics.parameter_sigreg_control.evaluation import code_scale_diagnostics
from diagnostics.predictive_supervision_control.evaluation import evaluate
from diagnostics.predictive_supervision_control.objective import (
    PredictiveSupervisionObjective,
    gradient_diagnostics,
)
from diagnostics.predictive_supervision_control.run import code_identity as source_code_identity
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng

CONDITIONS = ("joint_clip", "separate_clip", "separate_clip_encoder4")


@dataclass(frozen=True)
class Config:
    """Additional update budget; source checkpoint supplies batch size and Adam settings."""

    source_dir: str
    output_dir: str
    seed: int = 202610261301
    device: str = "cpu"
    steps: int = 30000
    conditions: tuple[str, ...] = CONDITIONS
    encoder_prediction_scale: float = 4.0
    threads: int = 1
    log_every: int = 100
    evaluate_every: int = 1000
    checkpoint_every: int = 1000
    gradient_every: int = 1000
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None
    wandb_group: str = "prediction1-encoder-gradient-continuation"

    def __post_init__(self) -> None:
        """Require the matched three-arm protocol and an independent output directory."""
        for name in (
            "steps",
            "threads",
            "log_every",
            "evaluate_every",
            "checkpoint_every",
            "gradient_every",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.seed < 0 or torch.device(self.device).type not in {"cpu", "cuda"}:
            raise ValueError("Use a nonnegative seed and CPU or CUDA device")
        if (
            not self.conditions
            or len(set(self.conditions)) != len(self.conditions)
            or any(c not in CONDITIONS for c in self.conditions)
        ):
            raise ValueError("conditions must be distinct supported arms")
        if not math.isfinite(self.encoder_prediction_scale) or self.encoder_prediction_scale != 4:
            raise ValueError("This comparison fixes encoder_prediction_scale at 4")
        source, output = Path(self.source_dir).resolve(), Path(self.output_dir).resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Source and output directories must be separate, not nested")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def code_identity() -> str:
    """Bind continuation semantics and all reused training/evaluation helpers."""
    digest = hashlib.sha256(source_code_identity().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare(config: Config, bundle: SourceBundle, *, resume: bool) -> dict[str, Any]:
    """Lock source artifacts and continuation settings without copying or modifying inputs."""
    settings = asdict(config)
    for key in ("source_dir", "output_dir", "wandb_mode", "wandb_project", "wandb_entity"):
        settings.pop(key)
    evidence = {
        "settings": settings,
        "source_identity": bundle.protocol["identity"],
        "source_hashes": bundle.hashes,
        "code_identity": code_identity(),
        "torch": str(torch.__version__),
    }
    identity = canonical_fingerprint(evidence)
    out = Path(config.output_dir)
    if resume:
        protocol = cast(dict[str, Any], read_json(out / "protocol.json"))
        if protocol["identity"] != identity:
            raise ValueError("Resume source, settings or code changed; use a new output directory")
        return protocol
    if any(path.name != ".lock" for path in out.iterdir()):
        raise FileExistsError("Output directory is not empty; use --resume or a new directory")
    protocol = {
        "identity": identity,
        **evidence,
        "config": asdict(config),
        "source_step": bundle.source_step,
        "additional_steps": config.steps,
        "final_absolute_step": bundle.source_step + config.steps,
        "source_checkpoint": str(
            bundle.source_dir / f"seed-{config.seed}" / "prediction_1" / "last.pt"
        ),
        "batch_size": bundle.batch_size,
        "train_groups": bundle.train_groups,
        "model_config": bundle.protocol["model_config"],
        "objective_config": bundle.protocol["objective_config"],
        "frozen_hashes": bundle.protocol["frozen_hashes"],
        "conditions": {
            condition: {
                "prediction_weight": 1.0,
                "encoder_prediction_scale": config.encoder_prediction_scale
                if condition == "separate_clip_encoder4"
                else 1.0,
                "clipping": "global_norm_1" if condition == "joint_clip" else "per_module_norm_1",
                "parameter_sigreg_weight": 0.4,
                "consistency_weight": 1.0,
            }
            for condition in config.conditions
        },
        "pairing": (
            "Same final prediction_1 weights, Adam moments/hyperparameters, label-free training "
            "cache, future batch sequence and per-absolute-step RNG seed. "
            "Each arm continues the source step counter."
        ),
        "scope": (
            "Original transition MLP and dense predictor; both informative state encoders "
            "frozen; no EMA or parameter labels in training."
        ),
        "interpretation": (
            "All evaluation uses the common prediction-weight-1 objective. Encoder4 changes "
            "the encoder backward gradient only; its update rule is not the gradient of one "
            "shared scalar objective. Separate clipping also changes block step scales and "
            "can strengthen auxiliary gradients; it is not an isolated change to prediction "
            "weighting."
        ),
        "selection": (
            "Final additional-update budget; no early stopping or label-driven "
            "checkpoint selection."
        ),
    }
    write_json(out / "protocol.json", protocol)
    return protocol


def _completed(dest: Path, identity: str) -> dict[str, Any]:
    completed = read_json(dest / "completed.json")
    if completed["identity"] != identity:
        raise ValueError("Completed continuation belongs to another protocol")
    for filename, key in (
        ("last.pt", "checkpoint_sha256"),
        ("results.json", "results_sha256"),
        ("gradient_history.json", "gradient_history_sha256"),
    ):
        if sha(dest / filename) != completed[key]:
            raise ValueError(f"Completed continuation artifact changed: {filename}")
    return cast(dict[str, Any], read_json(dest / "results.json"))


def train_condition(
    config: Config,
    protocol: dict[str, Any],
    bundle: SourceBundle,
    device: torch.device,
    condition: str,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Train one continuation, restoring only its own checkpoint and W&B identity on resume."""
    if condition not in config.conditions:
        raise ValueError("Condition is absent from the locked protocol")
    dest = Path(config.output_dir) / f"seed-{config.seed}" / condition
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "completed.json").exists():
        if not resume:
            raise FileExistsError("Completed arm exists; use --resume")
        result = _completed(dest, protocol["identity"])
        print(json.dumps({"skipping_completed": condition}), flush=True)
        return result
    model, optimizer = build_source_model(bundle, device)
    source_checkpoint = f"seed-{config.seed}/prediction_1/last.pt"
    source_hash = bundle.hashes[source_checkpoint]
    if sha(bundle.source_dir / source_checkpoint) != source_hash:
        raise ValueError("Source checkpoint changed after loading the comparison")
    start = 0
    history: list[dict[str, Any]] = []
    clipped_counts = {"encoder": 0, "predictor": 0}
    elapsed = 0.0
    resume_wandb: dict[str, object] | None = None
    last = dest / "last.pt"
    if resume and last.exists():
        payload = cast(dict[str, Any], torch.load(last, map_location="cpu", weights_only=False))
        start = int(payload["continuation_step"])
        if (
            payload["identity"] != protocol["identity"]
            or payload["condition"] != condition
            or payload["source_sha256"] != source_hash
            or payload["seed"] != config.seed
            or payload["source_step"] != bundle.source_step
            or payload["step"] != bundle.source_step + start
            or payload["prediction_weight"] != 1.0
            or payload["condition_config"] != protocol["conditions"][condition]
            or not 0 <= start <= config.steps
        ):
            raise ValueError("Checkpoint is not from this continuation arm")
        expected_extra_state = model.get_extra_state()
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        if (
            frozen_hashes(model) != protocol["frozen_hashes"]
            or model.get_extra_state() != expected_extra_state
        ):
            raise ValueError(
                "Frozen encoders or prediction mode changed in continuation checkpoint"
            )
        history = cast(list[dict[str, Any]], payload["gradient_history"])
        clipped_counts = cast(dict[str, int], payload["clipped_updates"])
        if set(clipped_counts) != {"encoder", "predictor"} or any(
            not 0 <= value <= start for value in clipped_counts.values()
        ):
            raise ValueError("Invalid clipping counters")
        history_steps = [row["continuation_step"] for row in history]
        if history_steps != sorted(set(history_steps)) or any(
            not 1 <= step <= start for step in history_steps
        ):
            raise ValueError("Invalid gradient history")
        elapsed = float(payload["elapsed_seconds"])
        resume_wandb = payload.get("wandb")
    write_json(
        dest / "gradient_history.json", {"identity": protocol["identity"], "history": history}
    )
    objective_config = ObjectiveConfig(**protocol["objective_config"])
    objective = PredictiveSupervisionObjective(objective_config, 1.0).to(device)
    logger = WandbLogger(
        WandbConfig(
            mode=config.wandb_mode,
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"prediction1-{condition}-seed-{config.seed}",
            group=config.wandb_group,
            tags=["diagnostic", "frozen-states", "prediction-1", "continuation", condition],
            job_type="encoder-gradient-control",
        ),
        resolved_config={
            **asdict(config),
            "condition": condition,
            **protocol["conditions"][condition],
            "source_step": bundle.source_step,
            "final_absolute_step": bundle.source_step + config.steps,
            "batch_size": bundle.batch_size,
            "train_groups": bundle.train_groups,
            "source_sha256": source_hash,
            "source_optimizer_preserved": True,
        },
        dataset_identity=protocol["identity"],
        out_dir=dest,
        resume_state=resume_wandb,
    )
    started = time.perf_counter()

    def save(step: int) -> None:
        if frozen_hashes(model) != protocol["frozen_hashes"]:
            raise RuntimeError("Frozen state encoders changed during continuation")
        atomic_save(
            last,
            {
                "identity": protocol["identity"],
                "condition": condition,
                "seed": config.seed,
                "source_step": bundle.source_step,
                "continuation_step": step,
                "step": bundle.source_step + step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "prediction_weight": 1.0,
                "condition_config": protocol["conditions"][condition],
                "clipped_updates": clipped_counts,
                "gradient_history": history,
                "source_sha256": source_hash,
                "frozen_hashes": frozen_hashes(model),
                "queries": 1,
                "consistency_weight": 1.0,
                "parameter_regularizer_weight": 0.4,
                "wandb": logger.metadata,
                "elapsed_seconds": elapsed + time.perf_counter() - started,
            },
        )

    def validation(step: int) -> None:
        with preserve_rng(device):
            monitored = evaluate(
                copy.deepcopy(model).cpu().eval(),
                bundle.evaluation,
                objective_config,
                batch_size=bundle.batch_size,
                final=False,
                prediction_weight=1.0,
                prediction_weights=(1.0,),
            )
        write_json(
            dest / "validation.json",
            {"continuation_step": step, "step": bundle.source_step + step, "metrics": monitored},
        )
        logger.log(
            step,
            {
                **scalar_metrics(monitored, "validation"),
                "train/absolute_step": float(bundle.source_step + step),
            },
        )

    try:
        if start == 0:
            save(0)
            validation(0)
        batches_per_epoch = bundle.train_groups // bundle.batch_size
        current_epoch, batches = -1, []
        for local_step in range(start + 1, config.steps + 1):
            step = bundle.source_step + local_step
            epoch, position = divmod(step - 1, batches_per_epoch)
            if epoch != current_epoch:
                batches = list(
                    GlobalBatchSampler(bundle.train_groups, bundle.batch_size, config.seed, epoch)
                )
                current_epoch = epoch
            batch = {
                key: value[batches[position]].to(device) for key, value in bundle.train.items()
            }
            torch.manual_seed(config.seed * 100000 + step)  # pyright: ignore[reportUnknownMemberType]
            model.train()
            model.online_encoder.eval()
            model.target_encoder.eval()
            optimizer.zero_grad(set_to_none=True)
            output = cached_multiquery_forward(model, batch, queries=1)
            loss, terms = objective(output)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Nonfinite loss in {condition} at continuation step {local_step}"
                )
            diagnose = (
                local_step == 1
                or local_step % config.gradient_every == 0
                or local_step == config.steps
            )
            log = diagnose or local_step % config.log_every == 0
            reference_gradients = (
                gradient_diagnostics(model, output, terms, objective_config, 1.0)
                if diagnose
                else {}
            )
            scales = code_scale_diagnostics(model, output.parameters) if log else {}
            updates = backward_and_clip(
                model,
                loss,
                terms["prediction"],
                condition=condition,
                encoder_prediction_scale=config.encoder_prediction_scale,
                measure_prediction=log,
            )
            for block in clipped_counts:
                clipped_counts[block] += int(updates[f"{block}/clip_coefficient"] < 1.0)
            update_blocks = {
                "encoder": tuple(
                    p for p in model.parameter_encoder.parameters() if p.requires_grad
                ),
                "predictor": tuple(p for p in model.predictor.parameters() if p.requires_grad),
            }
            before_update = (
                {
                    block: tuple(p.detach().clone() for p in parameters)
                    for block, parameters in update_blocks.items()
                }
                if log
                else {}
            )
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            if log:
                for block, parameters in update_blocks.items():
                    changes = [
                        (p.detach() - old).double().square().sum()
                        for p, old in zip(parameters, before_update[block], strict=True)
                    ]
                    updates[f"{block}/adam_parameter_update_norm"] = float(
                        torch.stack(changes).sum().sqrt()
                    )
                metrics = {f"train/{key}": float(value.detach()) for key, value in terms.items()}
                metrics.update({f"train/{key}": value for key, value in scales.items()})
                metrics.update(
                    {
                        f"reference_gradients/{key}": value
                        for key, value in reference_gradients.items()
                    }
                )
                metrics.update({f"updates/{key}": value for key, value in updates.items()})
                metrics.update(
                    {
                        f"updates/{block}/cumulative_clipped_fraction": count / local_step
                        for block, count in clipped_counts.items()
                    }
                )
                metrics.update(
                    {
                        "train/absolute_step": float(step),
                        "train/continuation_step": float(local_step),
                        "train/prediction_weight": 1.0,
                        "train/encoder_prediction_scale": float(
                            protocol["conditions"][condition]["encoder_prediction_scale"]
                        ),
                        "train/elapsed_seconds": elapsed + time.perf_counter() - started,
                    }
                )
                logger.log(local_step, metrics)
            if diagnose:
                history.append(
                    {
                        "continuation_step": local_step,
                        "step": step,
                        "reference_gradients": reference_gradients,
                        "applied_updates": updates,
                    }
                )
                write_json(
                    dest / "gradient_history.json",
                    {"identity": protocol["identity"], "history": history},
                )
            if local_step % config.checkpoint_every == 0 or local_step == config.steps:
                save(local_step)
            if local_step % config.evaluate_every == 0 or local_step == config.steps:
                validation(local_step)
        with preserve_rng(device):
            metrics = evaluate(
                copy.deepcopy(model).cpu().eval(),
                bundle.evaluation,
                objective_config,
                batch_size=bundle.batch_size,
                final=True,
                prediction_weight=1.0,
                prediction_weights=(1.0,),
            )
        if (
            sha(bundle.source_dir / f"seed-{config.seed}" / "prediction_1" / "last.pt")
            != source_hash
        ):
            raise RuntimeError("Source checkpoint changed during continuation")
        result = {
            "identity": protocol["identity"],
            "condition": condition,
            "seed": config.seed,
            "source_step": bundle.source_step,
            "additional_steps": config.steps,
            "steps": bundle.source_step + config.steps,
            "prediction_weight": 1.0,
            "condition_config": protocol["conditions"][condition],
            "metrics": metrics,
            "source_sha256": source_hash,
            "checkpoint_sha256": sha(last),
            "gradient_history_sha256": sha(dest / "gradient_history.json"),
            "clipped_updates": clipped_counts,
            "frozen_hashes": frozen_hashes(model),
            "wandb": logger.metadata,
            "elapsed_seconds": elapsed + time.perf_counter() - started,
        }
        write_json(dest / "results.json", result)
        logger.log(config.steps, scalar_metrics(metrics, "test"))
        for path, artifact_type in (
            (dest / "results.json", "evaluation"),
            (dest / "gradient_history.json", "gradient-diagnostics"),
            (Path(config.output_dir) / "protocol.json", "protocol"),
        ):
            logger.log_artifact(
                path,
                name=f"encoder-gradient-{condition}-{artifact_type}-{protocol['identity'][:10]}",
                artifact_type=artifact_type,
            )
        logger.finish()
        write_json(
            dest / "completed.json",
            {
                "identity": protocol["identity"],
                "checkpoint_sha256": sha(last),
                "results_sha256": sha(dest / "results.json"),
                "gradient_history_sha256": sha(dest / "gradient_history.json"),
            },
        )
        print(
            json.dumps(
                {
                    "completed": condition,
                    "additional_steps": config.steps,
                    "absolute_step": bundle.source_step + config.steps,
                }
            ),
            flush=True,
        )
        return result
    except BaseException:
        logger.finish(exit_code=1)
        raise


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Run matched arms sequentially on one device, leaving the source experiment read-only."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use ordinary Python on one device, not torchrun")
    torch.set_num_threads(config.threads)
    device = resolve_device(config.device)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with file_lock(out / ".lock"), preserve_rng(device):
        bundle = load_source(Path(config.source_dir), config.seed)
        protocol = prepare(config, bundle, resume=resume)
        results: list[dict[str, Any]] = []
        for condition in config.conditions:
            print(
                json.dumps(
                    {
                        "starting": condition,
                        "source_step": bundle.source_step,
                        "additional_steps": config.steps,
                        "device": str(device),
                    }
                ),
                flush=True,
            )
            results.append(
                train_condition(config, protocol, bundle, device, condition, resume=resume)
            )
            write_json(
                out / "comparison.json", {"identity": protocol["identity"], "results": results}
            )
    return {"identity": protocol["identity"], "results": results}


def main() -> None:
    """Parse an additional-step continuation command with three fresh W&B runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    for name, default in (
        ("seed", 202610261301),
        ("steps", 30000),
        ("threads", 1),
        ("log-every", 100),
        ("evaluate-every", 1000),
        ("checkpoint-every", 1000),
        ("gradient-every", 1000),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--encoder-prediction-scale", type=float, default=4.0)
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="disabled"
    )
    parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="prediction1-encoder-gradient-continuation")
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    resume = bool(args.pop("resume"))
    args["conditions"] = tuple(args["conditions"])
    for name in ("source_dir", "output_dir"):
        args[name] = str(Path(args[name]).resolve())
    run(Config(**args), resume=resume)


if __name__ == "__main__":
    main()
