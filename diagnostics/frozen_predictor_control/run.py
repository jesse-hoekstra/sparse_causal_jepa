"""Train a fresh transition parameter encoder against a frozen all-six predictor."""

import argparse
import copy
import hashlib
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.frozen_predictor_control.diagnostics import (
    encoder_gradient_stats,
    predictor_code_sensitivity,
)
from diagnostics.frozen_predictor_control.source import (
    SourceBundle,
    build_fresh_model,
    frozen_module_hashes,
    load_source_bundle,
)
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.multiquery_control.run import atomic_save, resolve_device, scalar_metrics, sha
from diagnostics.parameter_sigreg_control.evaluation import code_scale_diagnostics
from diagnostics.predictive_supervision_control.evaluation import evaluate
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective
from diagnostics.predictive_supervision_control.run import code_identity as source_code_identity
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng


@dataclass(frozen=True)
class Config:
    """One fresh encoder, one fixed predictor, and a fixed update budget."""

    source_dir: str
    output_dir: str
    seed: int = 202610261301
    device: str = "cpu"
    steps: int = 100000
    batch_size: int = 16
    lr: float = 0.001
    threads: int = 1
    log_every: int = 100
    evaluate_every: int = 1000
    checkpoint_every: int = 1000
    gradient_every: int = 1000
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None
    wandb_group: str = "fresh-encoder-frozen-all6-predictor"

    def __post_init__(self) -> None:
        """Reject ambiguous budgets, unsupported devices, and source overwrites."""
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
        if self.batch_size < 2 or not math.isfinite(self.lr) or self.lr != 0.001:
            raise ValueError("Use batch_size>=2 and the original Adam lr=0.001")
        source, output = Path(self.source_dir).resolve(), Path(self.output_dir).resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Source and output directories must be separate, not nested")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def code_identity() -> str:
    """Bind the diagnostic and reused training/evaluation helper implementations."""
    digest = hashlib.sha256(source_code_identity().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare(config: Config, bundle: SourceBundle, *, resume: bool) -> dict[str, Any]:
    """Record immutable experiment settings and the verified portable source identity."""
    if config.batch_size != bundle.batch_size:
        raise ValueError("Batch size differs from the validated source protocol")
    settings = asdict(config)
    for key in ("source_dir", "output_dir", "wandb_mode", "wandb_project", "wandb_entity"):
        settings.pop(key)
    evidence = {
        "settings": settings,
        "source_identity": bundle.manifest["identity"],
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
        "model_config": bundle.model_config,
        "objective_config": bundle.objective_config,
        "frozen_hashes": bundle.manifest["frozen_hashes"],
        "source_manifest": bundle.manifest,
        "training": {
            "prediction_weight": 1.0,
            "parameter_sigreg_weight": 0.4,
            "consistency_weight": 1.0,
            "logit_weight": bundle.objective_config["logit_weight"],
            "optimizer": "fresh Adam, encoder parameters only, default betas/eps, no weight decay",
            "gradient_clip_norm": 1.0,
            "initial_step": 0,
            "train_groups": bundle.train_groups,
            "training_labels_absent": True,
        },
        "scope": (
            "Fresh original transition MLP with ordinary initialization; online state encoder, "
            "target state encoder and validated all-six predictor are fixed. Gradients pass "
            "through the predictor and both autoregressive forecast steps to the encoder. "
            "No EMA, parameter supervision, informative-code targets or code rescaling."
        ),
        "interpretation": (
            "Conditional acquisition diagnostic. The fixed predictor came from label-assisted "
            "preparation followed by original-objective washout. Success here is not an "
            "end-to-end unsupervised result or proof of robustness across initializations."
        ),
        "selection": "Final fixed-budget checkpoint; labels used only in held-out evaluation.",
    }
    write_json(out / "protocol.json", protocol)
    return protocol


def _completed(dest: Path, identity: str) -> dict[str, Any]:
    completed = cast(dict[str, Any], read_json(dest / "completed.json"))
    if completed["identity"] != identity:
        raise ValueError("Completed run belongs to another protocol")
    for filename in ("last.pt", "results.json", "gradient_history.json", "sensitivity.json"):
        if sha(dest / filename) != completed["hashes"][filename]:
            raise ValueError(f"Completed artifact changed: {filename}")
    return cast(dict[str, Any], read_json(dest / "results.json"))


def train(
    config: Config,
    protocol: dict[str, Any],
    bundle: SourceBundle,
    device: torch.device,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Train only a fresh encoder; resume can restore only this experiment's own state."""
    dest = Path(config.output_dir) / f"seed-{config.seed}"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "completed.json").exists():
        return _completed(dest, protocol["identity"])
    model = build_fresh_model(bundle, config.seed, device)
    initial_encoder_sha256 = module_fingerprint(model.parameter_encoder)
    expected_extra_state = model.get_extra_state()
    parameters = tuple(p for p in model.parameter_encoder.parameters() if p.requires_grad)
    if {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in parameters}:
        raise ValueError("Only the parameter encoder may be trainable")
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    start, clipped_updates, elapsed = 0, 0, 0.0
    history: list[dict[str, Any]] = []
    initial_sensitivity: dict[str, Any] | None = None
    resume_wandb: dict[str, object] | None = None
    last = dest / "last.pt"
    if resume and last.exists():
        payload = cast(dict[str, Any], torch.load(last, map_location="cpu", weights_only=False))
        start = int(payload["step"])
        if (
            payload["identity"] != protocol["identity"]
            or payload["seed"] != config.seed
            or payload["initial_encoder_sha256"] != initial_encoder_sha256
            or payload["frozen_hashes"] != protocol["frozen_hashes"]
            or not 0 <= start <= config.steps
        ):
            raise ValueError("Checkpoint is not from this fresh-encoder experiment")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        if (
            frozen_module_hashes(model) != protocol["frozen_hashes"]
            or model.get_extra_state() != expected_extra_state
        ):
            raise ValueError("Frozen modules or model prediction mode changed in checkpoint")
        history = cast(list[dict[str, Any]], payload["gradient_history"])
        history_steps = [row["step"] for row in history]
        if history_steps != sorted(set(history_steps)) or any(
            not 1 <= step <= start for step in history_steps
        ):
            raise ValueError("Invalid checkpoint gradient history")
        clipped_updates = int(payload["clipped_updates"])
        if not 0 <= clipped_updates <= start:
            raise ValueError("Invalid clipping counter")
        initial_sensitivity = payload["initial_sensitivity"]
        elapsed = float(payload["elapsed_seconds"])
        resume_wandb = payload.get("wandb")
    write_json(
        dest / "gradient_history.json", {"identity": protocol["identity"], "history": history}
    )
    objective_config = ObjectiveConfig(**bundle.objective_config)
    objective = PredictiveSupervisionObjective(objective_config, 1.0).to(device)
    logger = WandbLogger(
        WandbConfig(
            mode=config.wandb_mode,
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"fresh-encoder-frozen-all6-seed-{config.seed}",
            group=config.wandb_group,
            tags=["diagnostic", "fresh-encoder", "frozen-all6-predictor", "prediction-1"],
            job_type="frozen-predictor-control",
        ),
        resolved_config={
            **asdict(config),
            **protocol["training"],
            "source_identity": bundle.manifest["identity"],
            "initial_encoder_sha256": initial_encoder_sha256,
            "source_optimizer_preserved": False,
            "predictor_frozen": True,
        },
        dataset_identity=protocol["identity"],
        out_dir=dest,
        resume_state=resume_wandb,
    )
    started = time.perf_counter()

    def save(step: int) -> None:
        if (
            frozen_module_hashes(model) != protocol["frozen_hashes"]
            or model.get_extra_state() != expected_extra_state
        ):
            raise RuntimeError("Frozen modules or prediction mode changed during training")
        atomic_save(
            last,
            {
                "identity": protocol["identity"],
                "seed": config.seed,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "initial_encoder_sha256": initial_encoder_sha256,
                "initial_sensitivity": initial_sensitivity,
                "frozen_hashes": protocol["frozen_hashes"],
                "gradient_history": history,
                "clipped_updates": clipped_updates,
                "wandb": logger.metadata,
                "elapsed_seconds": elapsed + time.perf_counter() - started,
            },
        )

    def sensitivity() -> dict[str, Any]:
        calibration = bundle.evaluation["calibration"]
        inputs = {
            "online": calibration["states_online"],
            "target": calibration["states_target"],
            "actions": calibration["actions"],
        }
        return predictor_code_sensitivity(
            model, inputs, reference_stats=bundle.manifest["reference_code_stats"], max_groups=16
        )

    def validation(step: int) -> None:
        with preserve_rng(device):
            monitored = evaluate(
                copy.deepcopy(model).cpu().eval(),
                bundle.evaluation,
                objective_config,
                batch_size=config.batch_size,
                final=False,
                prediction_weight=1.0,
                prediction_weights=(1.0,),
            )
        write_json(dest / "validation.json", {"step": step, "metrics": monitored})
        logger.log(step, scalar_metrics(monitored, "validation"))

    try:
        if initial_sensitivity is None:
            initial_sensitivity = sensitivity()
            logger.log(0, scalar_metrics(initial_sensitivity, "initial_sensitivity"))
        if start == 0:
            save(0)
            validation(0)
        batches_per_epoch = bundle.train_groups // config.batch_size
        current_epoch, batches = -1, []
        for step in range(start + 1, config.steps + 1):
            epoch, position = divmod(step - 1, batches_per_epoch)
            if epoch != current_epoch:
                batches = list(
                    GlobalBatchSampler(bundle.train_groups, config.batch_size, config.seed, epoch)
                )
                current_epoch = epoch
            batch = {
                key: value[batches[position]].to(device) for key, value in bundle.train.items()
            }
            torch.manual_seed(config.seed * 100000 + step)  # pyright: ignore[reportUnknownMemberType]
            model.train()
            model.online_encoder.eval()
            model.target_encoder.eval()
            model.predictor.eval()
            optimizer.zero_grad(set_to_none=True)
            output = cached_multiquery_forward(model, batch, queries=1)
            loss, terms = objective(output)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite training loss at step {step}")
            diagnose = step == 1 or step % config.gradient_every == 0 or step == config.steps
            log = diagnose or step % config.log_every == 0
            gradients = (
                encoder_gradient_stats(model, output, terms, objective_config) if diagnose else {}
            )
            scales = code_scale_diagnostics(model, output.parameters) if log else {}
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
            coefficient = min(1.0, 1.0 / (norm + 1e-6))
            clipped_updates += int(coefficient < 1.0)
            before = tuple(p.detach().clone() for p in parameters) if log else ()
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            updates: dict[str, float] = {}
            if log:
                change = (
                    torch.stack(
                        [
                            (p.detach() - old).double().square().sum()
                            for p, old in zip(parameters, before, strict=True)
                        ]
                    )
                    .sum()
                    .sqrt()
                )
                updates = {
                    "encoder/gradient_norm_before_clip": norm,
                    "encoder/gradient_norm_after_clip": norm * coefficient,
                    "encoder/clip_coefficient": coefficient,
                    "encoder/adam_parameter_update_norm": float(change),
                    "encoder/cumulative_clipped_fraction": clipped_updates / step,
                }
                if diagnose:
                    updates["encoder/prediction_gradient_norm_after_clip"] = (
                        gradients["encoder/prediction_norm"] * coefficient
                    )
                metrics = {f"train/{key}": float(value.detach()) for key, value in terms.items()}
                metrics.update({f"train/{key}": value for key, value in scales.items()})
                metrics.update({f"gradients/{key}": value for key, value in gradients.items()})
                metrics.update({f"updates/{key}": value for key, value in updates.items()})
                metrics["train/prediction_weight"] = 1.0
                metrics["train/elapsed_seconds"] = elapsed + time.perf_counter() - started
                logger.log(step, metrics)
            if diagnose:
                history.append({"step": step, "gradients": gradients, "updates": updates})
                write_json(
                    dest / "gradient_history.json",
                    {"identity": protocol["identity"], "history": history},
                )
            if step % config.checkpoint_every == 0 or step == config.steps:
                save(step)
            if step % config.evaluate_every == 0 or step == config.steps:
                validation(step)
        with preserve_rng(device):
            metrics = evaluate(
                copy.deepcopy(model).cpu().eval(),
                bundle.evaluation,
                objective_config,
                batch_size=config.batch_size,
                final=True,
                prediction_weight=1.0,
                prediction_weights=(1.0,),
            )
        final_sensitivity = sensitivity()
        write_json(
            dest / "sensitivity.json",
            {
                "identity": protocol["identity"],
                "initial": initial_sensitivity,
                "final": final_sensitivity,
            },
        )
        if frozen_module_hashes(model) != protocol["frozen_hashes"]:
            raise RuntimeError("Frozen modules changed during final evaluation")
        result = {
            "identity": protocol["identity"],
            "seed": config.seed,
            "steps": config.steps,
            "prediction_weight": 1.0,
            "metrics": metrics,
            "source_identity": bundle.manifest["identity"],
            "frozen_hashes": protocol["frozen_hashes"],
            "initial_encoder_sha256": initial_encoder_sha256,
            "final_encoder_sha256": module_fingerprint(model.parameter_encoder),
            "checkpoint_sha256": sha(last),
            "gradient_history_sha256": sha(dest / "gradient_history.json"),
            "sensitivity_sha256": sha(dest / "sensitivity.json"),
            "clipped_updates": clipped_updates,
            "wandb": logger.metadata,
            "elapsed_seconds": elapsed + time.perf_counter() - started,
        }
        write_json(dest / "results.json", result)
        logger.log(config.steps, scalar_metrics(metrics, "test"))
        logger.log(config.steps, scalar_metrics(final_sensitivity, "final_sensitivity"))
        for path, artifact_type in (
            (dest / "results.json", "evaluation"),
            (dest / "gradient_history.json", "gradient-diagnostics"),
            (dest / "sensitivity.json", "sensitivity-diagnostics"),
            (Path(config.output_dir) / "protocol.json", "protocol"),
        ):
            logger.log_artifact(
                path,
                name=f"frozen-predictor-{artifact_type}-{protocol['identity'][:10]}",
                artifact_type=artifact_type,
            )
        logger.finish()
        write_json(
            dest / "completed.json",
            {
                "identity": protocol["identity"],
                "hashes": {
                    filename: sha(dest / filename)
                    for filename in (
                        "last.pt",
                        "results.json",
                        "gradient_history.json",
                        "sensitivity.json",
                    )
                },
            },
        )
        return result
    except BaseException:
        logger.finish(exit_code=1)
        raise


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Run one fresh-encoder control on one device; source artifacts remain read-only."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use ordinary Python on one device, not torchrun")
    torch.set_num_threads(config.threads)
    device = resolve_device(config.device)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with file_lock(out / ".lock"), preserve_rng(device):
        bundle = load_source_bundle(Path(config.source_dir))
        protocol = prepare(config, bundle, resume=resume)
        return train(config, protocol, bundle, device, resume=resume)


def main() -> None:
    """Parse the single 100k-step encoder experiment with optional W&B logging."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    for name, default in (
        ("seed", 202610261301),
        ("steps", 100000),
        ("batch-size", 16),
        ("threads", 1),
        ("log-every", 100),
        ("evaluate-every", 1000),
        ("checkpoint-every", 1000),
        ("gradient-every", 1000),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="disabled"
    )
    parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="fresh-encoder-frozen-all6-predictor")
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    resume = bool(args.pop("resume"))
    for name in ("source_dir", "output_dir"):
        args[name] = str(Path(args[name]).resolve())
    run(Config(**args), resume=resume)


if __name__ == "__main__":
    main()
