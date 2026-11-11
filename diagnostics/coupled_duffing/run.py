"""Fixed-budget, label-free baseline screen with staged parameter acquisition."""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from diagnostics.coupled_duffing.data import DataConfig, prepare
from diagnostics.coupled_duffing.evaluate import collect, metrics, prediction_mse
from diagnostics.coupled_duffing.model import LaggedPredictor, build_model
from diagnostics.coupled_duffing.regularization import (
    STATE_REGULARIZATIONS,
    StateRegularizationObjective,
)
from diagnostics.masked_mlp_learned.run import code_identity as parent_identity
from diagnostics.shared.io import read_json, save_torch, sha
from diagnostics.shared.sampling import GroupSampler, resolve_device
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import write_json
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.models.spartan import Spartan
from scjepa.training.lagrangian import GECOConfig
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig, TrainingObjective, seed_everything
from scjepa.training.rng import preserve_rng


@dataclass(frozen=True)
class TrainConfig:
    """One prespecified seed; all phases share one W&B run and one final endpoint."""

    data_dir: str
    output_dir: str
    seed: int = 0
    joint_steps: int = 50000
    acquisition_steps: int = 50000
    baseline_steps: int = 150000
    batch_size: int = 16
    learning_rate: float = 0.001
    sigreg_weight: float = 0.4
    eval_every: int = 5000
    log_every: int = 250
    device: str = "cuda:0"
    threads: int = 2
    wandb_mode: str = "online"
    wandb_group: str = "coupled-duffing-baseline-20261108"


def code_identity() -> str:
    """Bind all executed numerical sources without changing any historical identity."""
    return canonical_fingerprint(
        {
            "parent": parent_identity(),
            "files": {path.name: sha(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
        }
    )


def forward_loss(
    model: SCJEPA, inputs: dict[str, Tensor], objective: TrainingObjective
) -> tuple[Tensor, dict[str, Tensor], ModelOutput]:
    """Reject physical labels and keep every autoregressive donor-code path attached."""
    if set(inputs) != {"observations", "actions"}:
        raise ValueError("ordinary training accepts only observations and actions")
    output = cast(ModelOutput, model(inputs["observations"], inputs["actions"]))
    loss, terms = objective.forward(output)
    return loss, terms, output


def phase_model(model: SCJEPA, phase: str, seed: int, device: torch.device) -> SCJEPA:
    """Start fresh frozen-state acquisition, or open joint baseline optimization."""
    if phase == "acquisition":
        fresh = build_model(seed + 100000, model.context_len).to(device)
        fresh.online_encoder.load_state_dict(model.online_encoder.state_dict())
        fresh.target_encoder.load_state_dict(model.target_encoder.state_dict())
        fresh.online_encoder.requires_grad_(False)
        return fresh
    if phase == "baseline":
        fresh_predictor = LaggedPredictor(sparse=True).to(device)
        missing, unexpected = cast(
            tuple[list[str], list[str]],
            fresh_predictor.load_state_dict(model.predictor.state_dict(), strict=False),
        )
        if set(missing) != {"state_logits", "action_logits"} or unexpected:
            raise ValueError("dense-to-baseline inheritance changed")
        model.predictor = cast(Spartan, fresh_predictor)
        model.online_encoder.requires_grad_(True)
    return model


def objective_for(
    phase: str,
    weight: float,
    device: torch.device,
    *,
    state_regularization: str = "aligned",
    seed: int = 0,
    state_frames: int | None = None,
) -> TrainingObjective:
    """Use the existing SIGReg and prediction-calibrated GECO convention."""
    if state_regularization not in STATE_REGULARIZATIONS:
        raise ValueError("unknown state regularization")
    sparse = phase in {"baseline", "full"}
    config = ObjectiveConfig(
        state_sparsity_weight=float(sparse),
        action_sparsity_weight=float(sparse),
        parameter_sparsity_weight=float(phase == "full"),
        state_regularizer="sigreg",
        state_regularizer_weight=weight,
        regularizer_num_projections=256,
    )
    geco = GECOConfig(enabled=True, tau=1.0) if sparse else None
    if state_regularization == "aligned":
        if state_frames is not None:
            raise ValueError("a fixed state-frame prefix requires an auxiliary objective")
        return TrainingObjective(config, geco).to(device)
    return StateRegularizationObjective(
        config, geco, mode=state_regularization, seed=seed, state_frames=state_frames
    ).to(device)


def _load_inputs(directory: Path, split: str, device: torch.device) -> dict[str, Tensor]:
    payload = cast(
        dict[str, Tensor], torch.load(directory / f"{split}-inputs.pt", weights_only=True)
    )
    return {name: payload[name].to(device) for name in ("observations", "actions")}


def train(config: TrainConfig, *, resume: bool = False) -> None:
    """Train one baseline seed and authenticate all fixed endpoints and source artifacts."""
    torch.set_num_threads(config.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(config.seed)
    device = resolve_device(config.device)
    directory, output_dir = Path(config.data_dir), Path(config.output_dir)
    data_record = read_json(directory / "manifest.json")
    expected_data_identity = canonical_fingerprint(
        {key: value for key, value in data_record.items() if key != "identity"}
    )
    if expected_data_identity != data_record["identity"]:
        raise ValueError("data manifest changed")
    for name in (
        "train-inputs.pt",
        "calibration-inputs.pt",
        "validation-inputs.pt",
        "calibration-labels.pt",
        "validation-labels.pt",
    ):
        if sha(directory / name) != data_record["hashes"][name]:
            raise ValueError(f"data artifact changed: {name}")
    data_config = DataConfig(**data_record["config"])
    total = config.joint_steps + config.acquisition_steps + config.baseline_steps
    if (
        min(config.joint_steps, config.acquisition_steps, config.baseline_steps, config.eval_every)
        < 1
    ):
        raise ValueError("all phase budgets and evaluation intervals must be positive")
    if output_dir.exists() and not resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=resume)
    identity = code_identity()
    specification: dict[str, Any] = {
        "config": asdict(config),
        "data_identity": data_record["identity"],
        "code_identity": identity,
        "parameter_gates": "fixed_open",
        "true_state_training": False,
        "true_parameter_training": False,
        "test_evaluation": False,
    }
    run_identity = canonical_fingerprint(specification)
    inputs = {
        name: _load_inputs(directory, name, device)
        for name in ("train", "calibration", "validation")
    }
    # The train label file is deliberately never opened by this runner.
    labels = {
        name: cast(
            dict[str, Tensor], torch.load(directory / f"{name}-labels.pt", weights_only=True)
        )
        for name in ("calibration", "validation")
    }
    sampler = GroupSampler(len(inputs["train"]["observations"]), config.batch_size, config.seed)
    saved: dict[str, Any] = {}
    step, phase, tau = 0, "joint", None
    model = build_model(config.seed, data_config.context).to(device)
    if resume:
        saved = cast(
            dict[str, Any],
            torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False),
        )
        if saved["identity"] != run_identity:
            raise ValueError("checkpoint source, data or fixed configuration changed")
        phase, step, tau = saved["phase"], int(saved["step"]), saved["tau"]
        model = phase_model(model, phase, config.seed, device)
        model.load_state_dict(saved["model"], strict=True)
    write_json(output_dir / "config.json", {**specification, "identity": run_identity})
    objective = objective_for(phase, config.sigreg_weight, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0)
    if resume:
        optimizer.load_state_dict(saved["optimizer"])
        objective.load_state_dict(saved["objective"])
        sampler.load_state_dict(saved["sampler"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
    logger = WandbLogger(
        WandbConfig(
            mode=config.wandb_mode,
            project="sparse_causal_jepa",
            entity="jesse-hoekstra-university-of-oxford",
            group=config.wandb_group,
            name=f"duffing-baseline-s{config.seed}",
            job_type="baseline-screen",
            tags=["duffing", "baseline", "learned-parameters", "validation-only"],
        ),
        resolved_config=cast(dict[str, object], specification),
        dataset_identity=data_record["identity"],
        out_dir=output_dir,
        resume_state=saved.get("wandb"),
    )
    write_json(output_dir / "wandb.json", logger.metadata)

    def save(name: str = "last.pt") -> None:
        save_torch(
            output_dir / name,
            {
                "identity": run_identity,
                "step": step,
                "phase": phase,
                "tau": tau,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "objective": objective.state_dict(),
                "sampler": sampler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "wandb": logger.metadata,
            },
        )

    def evaluate(*, final: bool = False) -> dict[str, float]:
        with preserve_rng(device):
            cal = collect(
                model, inputs["calibration"], config.batch_size, limit=None if final else 128
            )
            val = collect(
                model, inputs["validation"], config.batch_size, limit=None if final else 128
            )
            result = metrics(
                cal, val, labels["calibration"], labels["validation"], data_config.context
            )
            shuffled = collect(
                model,
                inputs["validation"],
                config.batch_size,
                limit=None if final else 128,
                shuffle=True,
            )
            result["code_shuffle/mse_ratio"] = prediction_mse(shuffled) / max(
                prediction_mse(val), 1e-12
            )
            if tau is not None:
                result["geco/relative_prediction"] = prediction_mse(val) / tau
            predictor = cast(LaggedPredictor, model.predictor)
            for name, graph in predictor.thresholded_graphs().items():
                result[f"gates/{name}_open"] = float(graph.sum())
            if final:
                save_torch(output_dir / "calibration.pt", cal)
                save_torch(output_dir / "validation.pt", val)
            with (output_dir / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, "phase": phase, **result}) + "\n")
            logger.log(step, {f"validation/{key}": value for key, value in result.items()})
        return result

    started = time.monotonic()
    start = step
    exit_code = 1
    try:
        if not resume:
            save("initial.pt")
            evaluate()
        for step in range(start + 1, total + 1):
            next_phase = (
                "joint"
                if step <= config.joint_steps
                else "acquisition"
                if step <= config.joint_steps + config.acquisition_steps
                else "baseline"
            )
            if next_phase != phase:
                if next_phase == "baseline":
                    with preserve_rng(device):
                        calibration = collect(model, inputs["calibration"], config.batch_size)
                    tau = prediction_mse(calibration)
                    if not math.isfinite(tau) or tau <= 0:
                        raise ValueError("invalid calibration prediction MSE")
                    write_json(
                        output_dir / "calibration.json",
                        {
                            "tau": tau,
                            "factor": 1.0,
                            "source_checkpoint_sha256": sha(output_dir / "acquisition.pt"),
                        },
                    )
                model = phase_model(model, next_phase, config.seed, device)
                phase = next_phase
                objective = objective_for(phase, config.sigreg_weight, device)
                parameters = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0)
            model.train()
            if phase == "acquisition":
                model.online_encoder.eval()
            indices = sampler.next().to(device)
            batch = {name: value[indices] for name, value in inputs["train"].items()}
            optimizer.zero_grad(set_to_none=True)
            loss, terms, _ = forward_loss(model, batch, objective)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            gradient = nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            if phase != "acquisition":
                model.update_target()
            if objective.geco is not None:
                assert tau is not None
                objective.geco.update(terms["prediction"].detach() / tau)
            if step % config.log_every == 0:
                values = {f"train/{key}": float(value.detach()) for key, value in terms.items()}
                values.update(
                    {
                        "train/loss": float(loss.detach()),
                        "train/gradient_norm": float(gradient),
                        "train/phase": float(("joint", "acquisition", "baseline").index(phase)),
                        "runtime/seconds_per_step": (time.monotonic() - started) / (step - start),
                    }
                )
                if device.type == "cuda":
                    values["runtime/gpu_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1e9
                if objective.geco is not None:
                    values["train/sparsity_weight"] = float(objective.geco.penalty_weight)
                logger.log(step, values)
            boundary = step in {config.joint_steps, config.joint_steps + config.acquisition_steps}
            if step % config.eval_every == 0 or boundary or step == total:
                evaluate(final=step == total)
                save()
            if boundary:
                save(f"{phase}.pt")
        if code_identity() != identity:
            raise RuntimeError("source files changed during training")
        final_metrics = read_json_lines(output_dir / "validation.jsonl")[-1]
        write_json(
            output_dir / "completion.json",
            {
                "identity": run_identity,
                "steps": step,
                "phase": phase,
                "metrics": final_metrics,
                "elapsed_seconds": time.monotonic() - started,
                "hashes": {
                    name: sha(output_dir / name)
                    for name in (
                        "last.pt",
                        "initial.pt",
                        "joint.pt",
                        "acquisition.pt",
                        "config.json",
                        "calibration.json",
                        "calibration.pt",
                        "validation.pt",
                        "validation.jsonl",
                    )
                },
            },
        )
        exit_code = 0
    finally:
        logger.finish(exit_code)


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    """Read the complete fixed-checkpoint evaluation record."""
    return [
        cast(dict[str, Any], json.loads(line)) for line in path.read_text().splitlines() if line
    ]


def main() -> None:
    """Prepare the fixed cache, train a seed, or summarize completed baseline seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("prepare")
    data.add_argument("--output-dir", required=True)
    run = sub.add_parser("train")
    run.add_argument("--data-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--wandb-group", default="coupled-duffing-baseline-20261108")
    run.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    run.add_argument("--resume", action="store_true")
    for name, default in (
        ("joint-steps", 50000),
        ("acquisition-steps", 50000),
        ("baseline-steps", 150000),
        ("eval-every", 5000),
        ("log-every", 250),
    ):
        run.add_argument(f"--{name}", type=int, default=default)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        torch.set_num_threads(2)
        prepare(Path(args["output_dir"]), DataConfig())
    else:
        resume = args.pop("resume")
        train(TrainConfig(**args), resume=resume)


if __name__ == "__main__":
    main()
