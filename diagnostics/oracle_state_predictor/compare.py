"""Compare dense, baseline and full routing with supplied true states and parameters.

This disposable experiment imports existing SPARTAN/GECO components without
changing the ordinary dataset, learned-representation pipeline or point-1 runner.
"""

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.oracle_state_predictor.data import OracleStateData, build_data
from diagnostics.oracle_state_predictor.graphs import (
    Condition,
    build_predictor,
    evaluate_graphs,
    rollout_graphs,
)
from diagnostics.oracle_state_predictor.reference import build_calibration, load_dense_reference
from diagnostics.oracle_state_predictor.run import (
    Config,
    GroupSampler,
    code_identity,
    resolve_device,
    write_json,
)
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.models.spartan import GraphPenalties, Spartan
from scjepa.training import seed_everything
from scjepa.training.lagrangian import SparsityLagrangian
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.rng import preserve_rng

SCHEMA = "oracle-state-graph-comparison-v1"


@dataclass(frozen=True)
class ComparisonConfig(Config):
    """Shared settings; steps means updates per condition, not total updates."""

    output_dir: str = "outputs/diagnostics/oracle_state_graphs/seed-0"
    wandb_group: str = "diagnostic-oracle-state-parameter-graphs"
    dense_reference_dir: str | None = None
    calibration_groups: int = 512
    tau_multiplier: float = 1.0

    def __post_init__(self) -> None:
        """Validate prediction settings and an explicit positive calibration multiplier."""
        super().__post_init__()
        if self.calibration_groups < 1:
            raise ValueError("calibration_groups must be positive")
        if not math.isfinite(self.tau_multiplier) or self.tau_multiplier <= 0:
            raise ValueError("tau_multiplier must be finite and positive")


def predictor_config(config: ComparisonConfig, output: Path) -> Config:
    """Strip comparison-only settings for the unchanged point-1 components."""
    values = {field.name: getattr(config, field.name) for field in fields(Config)}
    return replace(Config(**values), output_dir=str(output))


def comparison_source() -> str:
    """Fingerprint this comparison separately from the unchanged point-1 runner."""
    digest = hashlib.sha256(code_identity().encode())
    for name in ("compare.py", "graphs.py", "reference.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def state_identity(state: dict[str, Tensor]) -> str:
    """Record paired tensor initialization independently of module gate overrides."""
    return canonical_fingerprint({key: tensor_fingerprint(value) for key, value in state.items()})


def read_checkpoint(path: Path) -> dict[str, Any]:
    """Read only a caller-selected trusted local checkpoint."""
    return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))


def read_json(path: Path) -> dict[str, Any]:
    """Read a local diagnostic manifest or result."""
    return cast(dict[str, Any], json.loads(path.read_text()))


def sha256(path: Path) -> str:
    """Identify exact checkpoint bytes before any warm start."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sparsity_loss(penalties: GraphPenalties, condition: Condition) -> Tensor:
    """Match production block weights: none, state/action, or all three means."""
    if condition == "dense":
        return penalties.state.new_zeros(())
    return (
        penalties.state
        + penalties.action
        + (penalties.parameter if condition == "full" else penalties.parameter.new_zeros(()))
    )


def make_controller(tau: float, device: torch.device) -> SparsityLagrangian:
    """Reuse the production GECO settings and its inverse-multiplier convention."""
    return SparsityLagrangian(
        tau=tau,
        step_size=0.1,
        lambda_init=1000.0,
        momentum=0.99,
        log_lambda_min=-12.0,
        log_lambda_max=12.0,
    ).to(device)


def _logger(
    config: ComparisonConfig,
    condition: Condition,
    output: Path,
    record: dict[str, object],
    dataset_identity: str,
    resume_state: dict[str, object] | None,
) -> WandbLogger:
    return WandbLogger(
        WandbConfig(
            mode=config.wandb_mode,
            project=config.wandb_project,
            entity=config.wandb_entity,
            group=config.wandb_group,
            name=f"oracle-state-theta-{condition}-seed{config.seed}",
            job_type="oracle-state-graph-comparison",
            tags=["diagnostic", "oracle-states", "oracle-parameters", condition],
        ),
        resolved_config=record,
        dataset_identity=dataset_identity,
        out_dir=output,
        resume_state=resume_state,
    )


def _log(logger: WandbLogger, output: Path, step: int, metrics: dict[str, float]) -> None:
    logger.log(step, metrics)
    with (output / "history.jsonl").open("a") as stream:
        stream.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")


def _evaluate(
    model: Spartan,
    data: OracleStateData,
    config: ComparisonConfig,
    device: torch.device,
    tau: float | None,
) -> tuple[dict[str, float], dict[str, object]]:
    metrics, graphs = evaluate_graphs(model, data, config.batch_size, device, config.logit_weight)
    if tau is not None:
        metrics["constraint"] = metrics["base_loss"] - tau
        metrics["feasible"] = float(metrics["base_loss"] <= tau)
    return metrics, graphs


def train_condition(
    config: ComparisonConfig,
    condition: Condition,
    datasets: dict[str, OracleStateData],
    comparison_identity: str,
    *,
    initial: dict[str, Tensor] | None = None,
    calibration: dict[str, object] | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Train one condition, restoring its own optimizer/controller on continuation."""
    if (condition == "dense" and (initial is not None or calibration is not None)) or (
        condition != "dense" and (initial is None or calibration is None)
    ):
        raise ValueError("sparse conditions require a shared dense state and calibration")
    device = resolve_device(config.device)
    seed_everything(config.seed)
    output = Path(config.output_dir) / condition
    model = build_predictor(predictor_config(config, output), condition).to(device)
    if initial is not None:
        model.load_state_dict(initial)
    initial_fingerprint = state_identity(model.state_dict())
    tau = float(cast(float, calibration["tau"])) if calibration is not None else None
    identity = canonical_fingerprint(
        {
            "comparison": comparison_identity,
            "condition": condition,
            "initial_state_fingerprint": initial_fingerprint,
            "calibration": calibration,
        }
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)
    sampler = GroupSampler(config.train_groups, config.batch_size, config.seed)
    controller = make_controller(tau, device) if tau is not None else None
    checkpoint_path = output / "last.pt"
    restored: dict[str, Any] = {}
    start, best = 0, math.inf
    if output.exists() and any(output.iterdir()):
        if not resume or not checkpoint_path.is_file():
            raise FileExistsError(
                f"existing {condition} output requires its checkpoint and --resume"
            )
        restored = read_checkpoint(checkpoint_path)
        if restored.get("schema") != SCHEMA or restored.get("identity") != identity:
            raise ValueError(f"{condition} checkpoint provenance differs")
        model.load_state_dict(restored["model"])
        optimizer.load_state_dict(restored["optimizer"])
        sampler.load_state_dict(restored["sampler"])
        if controller is not None:
            controller.load_state_dict(restored["controller"])
            controller.validate_state()
        start, best = restored["step"], restored["best_validation_mse"]
        if start > config.steps:
            raise ValueError("checkpoint is beyond requested steps")
        torch.set_rng_state(restored["torch_rng"])
        if device.type == "cuda" and restored["cuda_rng"] is not None:
            torch.cuda.set_rng_state(restored["cuda_rng"], device)
        result_path = output / "result.json"
        if start == config.steps and result_path.is_file():
            result = read_json(result_path)
            if result.get("identity") != identity or result.get("last_checkpoint_sha256") != sha256(
                checkpoint_path
            ):
                raise ValueError(f"{condition} completion record differs from checkpoint")
            return result
    output.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "schema": SCHEMA,
        "identity": identity,
        "comparison_identity": comparison_identity,
        "condition": condition,
        "config": asdict(config),
        "source_fingerprint": comparison_source(),
        "initial_state_fingerprint": initial_fingerprint,
        "tau_calibration": calibration,
        "parameter_source": "raw_true_parameters",
        "state_source": "raw_true_states",
        "parameter_gates_open": condition == "baseline",
        "dense": condition == "dense",
        "objective": (
            "two-step prediction MSE + logit_weight * logit + GECO weight * block path means"
        ),
        "sparsity_weights": {
            "state": int(condition != "dense"),
            "action": int(condition != "dense"),
            "parameter": int(condition == "full"),
        },
        "data": {split: data.metadata for split, data in datasets.items()},
        "geco": None
        if controller is None
        else {
            "tau": tau,
            "step_size": 0.1,
            "momentum": 0.99,
            "lambda_init": 1000.0,
            "log_lambda_min": -12.0,
            "log_lambda_max": 12.0,
        },
    }
    write_json(output / "config.json", record)
    logger = _logger(
        config, condition, output, record, datasets["train"].fingerprint, restored.get("wandb")
    )

    def save(step: int, name: str) -> None:
        payload = {
            "schema": SCHEMA,
            "identity": identity,
            "comparison_identity": comparison_identity,
            "condition": condition,
            "config": asdict(config),
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sampler": sampler.state_dict(),
            "controller": None if controller is None else controller.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "best_validation_mse": best,
            "wandb": logger.metadata,
            "initial_state_fingerprint": initial_fingerprint,
            "tau_calibration": calibration,
        }
        temporary = output / f"{name}.tmp"
        torch.save(payload, temporary)
        temporary.replace(output / name)

    try:
        if not restored:
            metrics, _ = _evaluate(model, datasets["validation"], config, device, tau)
            _log(logger, output, 0, {f"validation/{key}": value for key, value in metrics.items()})
            if controller is not None:
                # Expose stochastic warm-start feasibility without consuming training draws.
                with preserve_rng(device), torch.no_grad():
                    model.train()
                    batch = datasets["validation"].batch(
                        torch.arange(min(config.batch_size, len(datasets["validation"]))), device
                    )
                    predicted, logit, _, _ = rollout_graphs(
                        model, batch.initial_states, batch.parameters, batch.actions
                    )
                    sampled_base = (
                        predicted - batch.targets
                    ).square().mean() + config.logit_weight * logit
                _log(
                    logger,
                    output,
                    0,
                    {
                        "warm_start/sampled_base_loss": float(sampled_base),
                        "geco/tau": cast(float, tau),
                    },
                )
            save(0, "last.pt")
        model.train()
        for step in range(start + 1, config.steps + 1):
            batch = datasets["train"].batch(sampler.next(), device)
            optimizer.zero_grad(set_to_none=True)
            predicted, logit, penalties, _ = rollout_graphs(
                model, batch.initial_states, batch.parameters, batch.actions
            )
            mse = (predicted - batch.targets).square().mean()
            base = mse + config.logit_weight * logit
            sparse = sparsity_loss(penalties, condition)
            weight = (
                base.new_zeros(())
                if controller is None
                else controller.penalty_weight.detach().clone()
            )
            loss = base + weight * sparse
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite {condition} loss at {step}")
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            if controller is not None:
                controller.update(base.detach())
            if step % config.log_every == 0 or step == config.steps:
                values = {
                    "loss/prediction": float(mse.detach()),
                    "loss/base": float(base.detach()),
                    "loss/logit": float(logit.detach()),
                    "loss/sparsity": float(sparse.detach()),
                    "loss/state_sparsity": float(penalties.state.detach()),
                    "loss/action_sparsity": float(penalties.action.detach()),
                    "loss/parameter_sparsity": float(penalties.parameter.detach()),
                    "loss/total": float(loss.detach()),
                    "train/lr": config.lr,
                }
                if controller is not None:
                    values |= {
                        "geco/tau": cast(float, tau),
                        "geco/constraint": float(base.detach()) - cast(float, tau),
                        "geco/applied_sparsity_weight": float(weight),
                        "geco/sparsity_weight": float(controller.penalty_weight),
                        "geco/moving_average_constraint": float(controller.ma_error),
                    }
                _log(logger, output, step, values)
            if step % config.evaluate_every == 0 or step == config.steps:
                metrics, _ = _evaluate(model, datasets["validation"], config, device, tau)
                _log(
                    logger,
                    output,
                    step,
                    {f"validation/{key}": value for key, value in metrics.items()},
                )
                if metrics["prediction_mse"] < best:
                    best = metrics["prediction_mse"]
                    save(step, "best.pt")
            if step % config.checkpoint_every == 0 or step == config.steps:
                save(step, "last.pt")
        metrics, graphs = _evaluate(model, datasets["test"], config, device, tau)
        _log(logger, output, config.steps, {f"test/{key}": value for key, value in metrics.items()})
        write_json(output / "test_graphs.json", graphs)
        result: dict[str, object] = {
            "schema": SCHEMA,
            "identity": identity,
            "condition": condition,
            "evaluated_step": config.steps,
            "metrics": metrics,
            "last_checkpoint_sha256": sha256(checkpoint_path),
            "wandb": logger.metadata,
        }
        write_json(output / "result.json", result)
    except BaseException:
        logger.finish(exit_code=1)
        raise
    logger.finish()
    return result


def evaluate_dense_reference(
    config: ComparisonConfig,
    datasets: dict[str, OracleStateData],
    comparison_identity: str,
    reference: dict[str, object],
    *,
    resume: bool,
) -> dict[str, object]:
    """Log an evaluation-only dense condition, leaving its original run untouched."""
    output = Path(config.output_dir) / "dense"
    result_path = output / "result.json"
    identity = canonical_fingerprint({"comparison": comparison_identity, "reference": reference})
    if resume and result_path.is_file():
        result = read_json(result_path)
        if result.get("identity") != identity:
            raise ValueError("dense reference result identity differs")
        return result
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    checkpoint = read_checkpoint(Path(cast(str, reference["last_checkpoint"])))
    model = build_predictor(predictor_config(config, output), "dense").to(device)
    model.load_state_dict(checkpoint["model"])
    record: dict[str, object] = {
        "schema": SCHEMA,
        "identity": identity,
        "condition": "dense",
        "config": asdict(config),
        "evaluation_only": True,
        "reference": reference,
        "comparison_identity": comparison_identity,
        "state_source": "raw_true_states",
        "parameter_source": "raw_true_parameters",
    }
    write_json(output / "config.json", record)
    logger = _logger(config, "dense", output, record, datasets["train"].fingerprint, None)
    result: dict[str, object] = {}
    try:
        for split in ("validation", "test"):
            metrics, graphs = _evaluate(model, datasets[split], config, device, None)
            _log(
                logger,
                output,
                checkpoint["step"],
                {f"{split}/{key}": value for key, value in metrics.items()},
            )
            if split == "test":
                write_json(output / "test_graphs.json", graphs)
                result = {
                    "schema": SCHEMA,
                    "identity": identity,
                    "condition": "dense",
                    "evaluated_step": checkpoint["step"],
                    "metrics": metrics,
                    "evaluation_only": True,
                    "reference": reference,
                    "wandb": logger.metadata,
                }
                write_json(result_path, result)
    except BaseException:
        logger.finish(exit_code=1)
        raise
    logger.finish()
    return result


def run_comparison(config: ComparisonConfig, *, resume: bool = False) -> dict[str, object]:
    """Run the three conditions with a shared selected dense warm start and tau."""
    torch.set_num_threads(config.threads)
    device = resolve_device(config.device)
    output = Path(config.output_dir)
    manifest_path = output / "comparison.json"
    if resume and not manifest_path.is_file():
        raise FileNotFoundError("--resume requires this comparison's comparison.json")
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError("comparison output is not empty; choose a new directory or --resume")
    dense_reference: dict[str, object] | None = None
    selected: dict[str, Any] = {}
    if config.dense_reference_dir is not None:
        source = Path(config.dense_reference_dir).resolve()
        destination = output.resolve()
        if source == destination or source in destination.parents or destination in source.parents:
            raise ValueError("dense reference and comparison output must be separate directories")
        selected, dense_reference = load_dense_reference(source, predictor_config(config, output))
    definition = asdict(config)
    for name in (
        "output_dir",
        "device",
        "threads",
        "log_every",
        "checkpoint_every",
        "wandb_mode",
        "wandb_project",
        "wandb_entity",
        "wandb_group",
    ):
        definition.pop(name)
    identity = canonical_fingerprint(
        {
            "schema": SCHEMA,
            "definition": definition,
            "source": comparison_source(),
            "dense_reference": dense_reference,
        }
    )
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "identity": identity,
        "definition": definition,
        "source_fingerprint": comparison_source(),
        "dense_reference": dense_reference,
    }
    if resume and read_json(manifest_path) != manifest:
        raise ValueError("comparison code, settings or dense reference changed; start a new output")
    output.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    datasets = {
        split: build_data(split, groups, seed=config.data_seed, views=config.views, progress=True)
        for split, groups in (
            ("train", config.train_groups),
            ("validation", config.validation_groups),
            ("test", config.test_groups),
        )
    }
    if dense_reference is None:
        dense_result = train_condition(config, "dense", datasets, identity, resume=resume)
        best_path, last_path = output / "dense" / "best.pt", output / "dense" / "last.pt"
        selected = read_checkpoint(best_path)
        dense_reference = {
            "best_checkpoint": str(best_path.resolve()),
            "last_checkpoint": str(last_path.resolve()),
            "best_checkpoint_sha256": sha256(best_path),
            "last_checkpoint_sha256": sha256(last_path),
            "selected_step": selected["step"],
            "completed_step": config.steps,
            "identity": selected["identity"],
        }
    else:
        dense_result = evaluate_dense_reference(
            config, datasets, identity, dense_reference, resume=resume
        )
    calibration_data = build_calibration(
        config.calibration_groups, seed=config.data_seed, views=config.views, progress=True
    )
    with preserve_rng(device):
        dense = build_predictor(predictor_config(config, output), "dense").to(device)
        dense.load_state_dict(selected["model"])
        calibrated, _ = _evaluate(dense, calibration_data, config, device, None)
    calibration: dict[str, object] = {
        "tau": config.tau_multiplier * calibrated["base_loss"],
        "tau_multiplier": config.tau_multiplier,
        "calibration_base_loss": calibrated["base_loss"],
        "calibration_prediction_mse": calibrated["prediction_mse"],
        "calibration_dataset_identity": calibration_data.fingerprint,
        "selected_step": selected["step"],
        "checkpoint_sha256": dense_reference["best_checkpoint_sha256"],
        "dense_reference": dense_reference,
    }
    calibration_path = output / "calibration.json"
    if resume and calibration_path.is_file():
        saved_calibration = read_json(calibration_path)
        # Preserve the original scalar exactly on resume, including across GPU hardware.
        for key in (
            "calibration_dataset_identity",
            "selected_step",
            "checkpoint_sha256",
            "tau_multiplier",
        ):
            if saved_calibration.get(key) != calibration[key]:
                raise ValueError("saved calibration provenance differs")
        calibration = saved_calibration
    else:
        write_json(calibration_path, calibration)
    print(json.dumps({"calibration": calibration}), flush=True)
    results = [dense_result]
    for condition in ("baseline", "full"):
        results.append(
            train_condition(
                config,
                cast(Condition, condition),
                datasets,
                identity,
                initial=selected["model"],
                calibration=calibration,
                resume=resume,
            )
        )
    summary: dict[str, object] = {
        "schema": SCHEMA,
        "identity": identity,
        "calibration": calibration,
        "conditions": results,
        "expected_edges": {"dense": [36, 96, 36], "baseline": [18, 96, 36], "full": [18, 96, 12]},
        "edge_order": ["state", "action", "parameter"],
    }
    write_json(output / "comparison_summary.json", summary)
    return summary


def main() -> None:
    """Launch all conditions, optionally reusing a completed point-1 dense run."""
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = cast(dict[str, str | int | float | None], asdict(ComparisonConfig()))
    for name, value in defaults.items():
        kind = type(value) if value is not None else str
        parser.add_argument("--" + name.replace("_", "-"), default=value, type=kind)
    parser.add_argument(
        "--resume", action="store_true", help="resume unfinished stages; skip completed stages"
    )
    arguments = vars(parser.parse_args())
    resume = cast(bool, arguments.pop("resume"))
    run_comparison(ComparisonConfig(**arguments), resume=resume)


if __name__ == "__main__":
    main()
