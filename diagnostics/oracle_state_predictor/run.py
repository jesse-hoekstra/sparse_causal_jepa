"""Fit the existing dense SPARTAN to fixed true states with known parameters.

Run from the repository root using python -m diagnostics.oracle_state_predictor.run.
This standalone diagnostic never constructs SCJEPA or changes its training interface.
"""

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.oracle_state_predictor.data import OracleStateData, build_data
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import source_fingerprint
from scjepa.models.spartan import Spartan
from scjepa.training import seed_everything
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.rng import preserve_rng


@dataclass(frozen=True)
class Config:
    """One predictor-only experiment; counts and batch size denote parameter groups."""

    output_dir: str = "outputs/diagnostics/oracle_state_predictor/seed-0"
    steps: int = 100_000
    seed: int = 0
    data_seed: int = 0
    batch_size: int = 16
    train_groups: int = 8192
    validation_groups: int = 256
    test_groups: int = 512
    views: int = 10
    lr: float = 0.001
    logit_weight: float = 0.0001
    grad_clip: float = 1.0
    embed_dim: int = 64
    mlp_hidden: int = 64
    device: str = "cpu"
    threads: int = 4
    log_every: int = 100
    evaluate_every: int = 1000
    checkpoint_every: int = 1000
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None
    wandb_group: str = "diagnostic-oracle-state-parameter-predictor"

    def __post_init__(self) -> None:
        """Reject invalid runs before creating output files or contacting W&B."""
        for name in (
            "steps",
            "batch_size",
            "train_groups",
            "validation_groups",
            "test_groups",
            "views",
            "embed_dim",
            "mlp_hidden",
            "threads",
            "log_every",
            "evaluate_every",
            "checkpoint_every",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.views < 2 or self.batch_size > self.train_groups:
            raise ValueError("need at least two views and batch_size <= train_groups")
        if self.seed < 0 or self.data_seed < 0:
            raise ValueError("seeds must be nonnegative")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(self.grad_clip) or self.grad_clip <= 0:
            raise ValueError("grad_clip must be finite and positive")
        if not math.isfinite(self.logit_weight) or self.logit_weight < 0:
            raise ValueError("logit_weight must be finite and nonnegative")
        if torch.device(self.device).type not in {"cpu", "cuda"}:
            raise ValueError("this diagnostic supports CPU or CUDA")


def build_predictor(config: Config) -> Spartan:
    """Use the current predictor architecture with every routing gate open."""
    return Spartan(
        state_dim=6,
        param_dim=6,
        action_dim=16,
        history_len=1,
        num_layers=2,
        embed_dim=config.embed_dim,
        mlp_hidden_size=config.mlp_hidden,
        mlp_num_layers=3,
        temperature=1.0,
        dense=True,
    )


def rollout(
    model: Spartan, initial_states: Tensor, parameters: Tensor, actions: Tensor
) -> tuple[Tensor, Tensor]:
    """Forecast two steps with fixed raw theta and an attached first prediction."""
    if actions.ndim != 3 or actions.shape[1:] != (2, 16):
        raise ValueError("actions must have shape (batch * views, 2, 16)")
    state = initial_states
    predictions: list[Tensor] = []
    penalties: list[Tensor] = []
    for horizon in range(2):
        result = model(state, parameters, actions[:, horizon : horizon + 1])
        predictions.append(result.prediction)
        penalties.append(result.logit_penalty)
        state = result.prediction.unsqueeze(1)
    return torch.stack(predictions, dim=1), torch.stack(penalties).mean()


def forecast_metrics(predictions: Tensor, targets: Tensor) -> dict[str, float]:
    """Pool errors globally; normalize by mean per-coordinate population variance.

    Combined variance pools samples AND the two horizons before centering each
    coordinate. Per-horizon variance pools all group/view samples at that horizon.
    """
    if predictions.shape != targets.shape or targets.ndim != 3 or targets.shape[1:] != (2, 6):
        raise ValueError("predictions and targets must both have shape (N, 2, 6)")
    predictions, targets = predictions.double(), targets.double()
    if not bool(torch.isfinite(predictions).all() & torch.isfinite(targets).all()):
        raise FloatingPointError("nonfinite forecasts or targets")
    errors = (predictions - targets).square()
    variance = float(targets.flatten(0, 1).var(dim=0, correction=0).mean())
    if variance <= 0:
        raise ValueError("target variance must be positive")
    metrics = {"prediction_mse": float(errors.mean()), "target_variance": variance}
    metrics["normalized_prediction_mse"] = metrics["prediction_mse"] / variance
    for horizon in range(2):
        mse = float(errors[:, horizon].mean())
        variance = float(targets[:, horizon].var(dim=0, correction=0).mean())
        if variance <= 0:
            raise ValueError("per-horizon target variance must be positive")
        metrics[f"horizon_{horizon + 1}_mse"] = mse
        metrics[f"horizon_{horizon + 1}_target_variance"] = variance
        metrics[f"horizon_{horizon + 1}_normalized_mse"] = mse / variance
    return metrics


@torch.no_grad()
def evaluate(
    model: Spartan,
    data: OracleStateData,
    batch_size: int,
    device: torch.device,
    logit_weight: float,
) -> dict[str, float]:
    """Evaluate every held-out group without changing training flags or RNG streams."""
    flags = [(module, module.training) for module in model.modules()]
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    logit_sum = 0.0
    try:
        with preserve_rng(device):
            model.eval()
            for start in range(0, len(data), batch_size):
                indices = torch.arange(start, min(start + batch_size, len(data)))
                batch = data.batch(indices, device)
                prediction, penalty = rollout(
                    model, batch.initial_states, batch.parameters, batch.actions
                )
                predictions.append(prediction.cpu())
                targets.append(batch.targets.cpu())
                logit_sum += float(penalty) * len(indices)
    finally:
        for module, training in flags:
            module.training = training
    metrics = forecast_metrics(torch.cat(predictions), torch.cat(targets))
    metrics["logit"] = logit_sum / len(data)
    metrics["base_loss"] = metrics["prediction_mse"] + logit_weight * metrics["logit"]
    return metrics


@torch.no_grad()
def reference_metrics(data: OracleStateData) -> dict[str, float]:
    """Check the analytic forecast and a persistence baseline on fixed supplied inputs."""
    theta = data.parameters[:, None, :].expand(-1, data.targets.shape[1], -1).double()
    state = data.initial_states[:, :, 0].double()
    predictions: list[Tensor] = []
    for horizon in range(2):
        state = dgp.transition(state, theta, data.actions[:, :, horizon].double())
        predictions.append(state)
    targets = data.targets.flatten(0, 1)
    exact = forecast_metrics(torch.stack(predictions, dim=2).flatten(0, 1), targets)
    persistence = forecast_metrics(
        data.initial_states.expand_as(data.targets).flatten(0, 1), targets
    )
    return {f"analytic/{key}": value for key, value in exact.items()} | {
        f"persistence/{key}": value for key, value in persistence.items()
    }


class GroupSampler:
    """Shuffle independent parameter groups without replacement; drop epoch remainders."""

    def __init__(self, groups: int, batch_size: int, seed: int) -> None:
        """Own the complete sampling stream so checkpoints resume its exact position."""
        self.groups, self.batch_size = groups, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.empty(0, dtype=torch.long)
        self.offset = 0

    def next(self) -> Tensor:
        """Return a full batch of distinct groups."""
        if self.offset + self.batch_size > len(self.order):
            self.order = torch.randperm(self.groups, generator=self.generator)
            self.offset = 0
        result = self.order[self.offset : self.offset + self.batch_size]
        self.offset += self.batch_size
        return result

    def state_dict(self) -> dict[str, object]:
        """Serialize the sampler without consuming its random stream."""
        return {"rng": self.generator.get_state(), "order": self.order, "offset": self.offset}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the next group batch from a trusted local checkpoint."""
        self.generator.set_state(state["rng"])
        self.order, self.offset = state["order"], state["offset"]


def write_json(path: Path, value: object) -> None:
    """Atomically replace a local JSON result."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def code_identity() -> str:
    """Bind checkpoints to production code and this disposable diagnostic's code."""
    digest = hashlib.sha256(source_fingerprint().encode())
    for name in ("data.py", "run.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def run(config: Config, *, resume: bool = False) -> dict[str, float]:
    """Train only the predictor, preserving checkpoints, histories and final test results."""
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    out = Path(config.output_dir)
    checkpoint_path = out / "last.pt"
    if resume and not checkpoint_path.is_file():
        raise FileNotFoundError(f"no diagnostic checkpoint to resume: {checkpoint_path}")
    if not resume and out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {out}; choose a new path or --resume"
        )
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    datasets = {
        split: build_data(split, groups, seed=config.data_seed, views=config.views, progress=True)
        for split, groups in (
            ("train", config.train_groups),
            ("validation", config.validation_groups),
            ("test", config.test_groups),
        )
    }
    immutable = asdict(config)
    for name in (
        "output_dir",
        "steps",
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
        immutable.pop(name)
    identities = {split: data.fingerprint for split, data in datasets.items()}
    identity = canonical_fingerprint(
        {"config": immutable, "data": identities, "code": code_identity()}
    )
    model = build_predictor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)
    sampler = GroupSampler(config.train_groups, config.batch_size, config.seed)
    start_step, best = 0, math.inf
    restored: dict[str, Any] = {}
    if resume:
        restored = cast(
            dict[str, Any], torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        )
        if (
            restored.get("schema") != "oracle-state-predictor-v1"
            or restored.get("identity") != identity
        ):
            raise ValueError("checkpoint is not this diagnostic or its code/data/config changed")
        model.load_state_dict(restored["model"])
        optimizer.load_state_dict(restored["optimizer"])
        sampler.load_state_dict(restored["sampler"])
        start_step, best = restored["step"], restored["best_validation_mse"]
        if config.steps < start_step:
            raise ValueError("--steps cannot be smaller than the checkpoint's step")
        torch.set_rng_state(restored["torch_rng"])
        if device.type == "cuda" and restored["cuda_rng"] is not None:
            torch.cuda.set_rng_state(restored["cuda_rng"], device)
    record: dict[str, object] = {
        "schema": "oracle-state-predictor-v1",
        "config": asdict(config),
        "identity": identity,
        "dataset_identities": identities,
        "source_fingerprint": code_identity(),
        "protocol": (
            "true z9 and raw theta; observed actions9,10; attached two-step forecast; "
            "fixed true z10,z11 targets"
        ),
        "data": {split: data.metadata for split, data in datasets.items()},
        "objective": "mean two-step MSE + logit_weight * mean SPARTAN logit penalty",
        "architecture": {
            "predictor": "Spartan",
            "dense": True,
            "layers": 2,
            "history_len": 1,
            "embed_dim": config.embed_dim,
            "mlp_hidden": config.mlp_hidden,
            "mlp_layers": 3,
        },
    }
    write_json(out / "config.json", record)
    logger = WandbLogger(
        WandbConfig(
            mode=config.wandb_mode,
            project=config.wandb_project,
            entity=config.wandb_entity,
            group=config.wandb_group,
            name=f"oracle-state-theta-dense-seed{config.seed}-logit{config.logit_weight:g}",
            job_type="oracle-state-predictor-diagnostic",
            tags=["diagnostic", "oracle-states", "oracle-parameters", "dense"],
        ),
        resolved_config=record,
        dataset_identity=identities["train"],
        out_dir=out,
        resume_state=restored.get("wandb"),
    )

    def log(step: int, metrics: dict[str, float]) -> None:
        logger.log(step, metrics)
        with (out / "history.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")

    def save(step: int, name: str) -> None:
        payload = {
            "schema": "oracle-state-predictor-v1",
            "identity": identity,
            "config": asdict(config),
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sampler": sampler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "best_validation_mse": best,
            "wandb": logger.metadata,
        }
        temporary = out / f"{name}.tmp"
        torch.save(payload, temporary)
        temporary.replace(out / name)

    try:
        if not resume:
            references = reference_metrics(datasets["validation"])
            write_json(out / "reference.json", references)
            log(0, {f"reference/{key}": value for key, value in references.items()})
            initial = evaluate(
                model, datasets["validation"], config.batch_size, device, config.logit_weight
            )
            log(0, {f"validation/{key}": value for key, value in initial.items()})
            save(0, "last.pt")
        model.train()
        interval_start, interval_steps = time.perf_counter(), 0
        for step in range(start_step + 1, config.steps + 1):
            batch = datasets["train"].batch(sampler.next(), device)
            optimizer.zero_grad(set_to_none=True)
            predictions, logit = rollout(
                model, batch.initial_states, batch.parameters, batch.actions
            )
            mse = (predictions - batch.targets).square().mean()
            loss = mse + config.logit_weight * logit
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss at update {step}")
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            interval_steps += 1
            if step % config.log_every == 0 or step == config.steps:
                log(
                    step,
                    {
                        "loss/prediction": float(mse.detach()),
                        "loss/logit": float(logit.detach()),
                        "loss/total": float(loss.detach()),
                        "train/lr": config.lr,
                        "train/seconds_per_update": (time.perf_counter() - interval_start)
                        / interval_steps,
                    },
                )
                interval_start, interval_steps = time.perf_counter(), 0
            if step % config.evaluate_every == 0 or step == config.steps:
                metrics = evaluate(
                    model, datasets["validation"], config.batch_size, device, config.logit_weight
                )
                log(step, {f"validation/{key}": value for key, value in metrics.items()})
                if metrics["prediction_mse"] < best:
                    best = metrics["prediction_mse"]
                    save(step, "best.pt")
            if step % config.checkpoint_every == 0 or step == config.steps:
                save(step, "last.pt")
        # Test the final model, never select a checkpoint using the held-out test.
        metrics = evaluate(model, datasets["test"], config.batch_size, device, config.logit_weight)
        write_json(out / "test_metrics.json", {"evaluated_step": config.steps, **metrics})
        log(config.steps, {f"test/{key}": value for key, value in metrics.items()})
    except BaseException:
        logger.finish(exit_code=1)
        raise
    logger.finish()
    return metrics


def main() -> None:
    """Parse standalone command-line settings; production experiment configs are untouched."""
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = cast(dict[str, str | int | float | None], asdict(Config()))
    for name, value in defaults.items():
        kind = type(value) if value is not None else str
        parser.add_argument("--" + name.replace("_", "-"), default=value, type=kind)
    parser.add_argument("--resume", action="store_true", help="resume this diagnostic's last.pt")
    arguments = vars(parser.parse_args())
    resume = cast(bool, arguments.pop("resume"))
    run(Config(**arguments), resume=resume)


if __name__ == "__main__":
    main()
