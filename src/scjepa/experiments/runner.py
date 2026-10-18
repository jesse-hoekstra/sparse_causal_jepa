"""Resumable paired controlled-vector experiments with held-out evaluation."""

import hashlib
import math
import signal
import time
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from types import FrameType
from typing import cast

import torch
from omegaconf import OmegaConf
from torch import Tensor

from scjepa.data.datasets import canonical_fingerprint
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.identity import training_tensors_fingerprint
from scjepa.experiments.config import ExperimentConfig
from scjepa.experiments.evaluation import calibrate_tau, evaluate_model, validate_model
from scjepa.experiments.reference import dense_directory, external_dense_manifest
from scjepa.experiments.reporting import aggregate_results
from scjepa.experiments.storage import (
    claim_execution,
    file_lock,
    json_safe,
    prepare_experiment,
    read_json,
    write_json,
)
from scjepa.models.model import SCJEPA
from scjepa.training import Trainer, seed_everything
from scjepa.training.distributed import (
    any_rank,
    destroy_distributed,
    gather_rank_objects,
    initialize_distributed,
    is_main_process,
    world_size,
)
from scjepa.training.factory import build_model
from scjepa.training.logging import WandbLogger


class ExperimentLogger:
    """Keep the experiment dashboard focused on its objective and recovery scores."""

    def __init__(self, logger: WandbLogger) -> None:
        """Wrap the existing W&B/terminal sink without changing its identity."""
        self.logger = logger

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Forward only interpretable losses, GECO controls and scientific scores."""
        selected = {
            key: value
            for key, value in metrics.items()
            if key
            in {
                "loss/prediction",
                "loss/base",
                "loss/sparsity",
                "geco/tau",
                "geco/constraint",
                "geco/sparsity_weight",
                "health/skipped_steps",
            }
            or (
                key.startswith(("validation/", "test/", "recovery/", "graph/"))
                and key != "validation/selection_score"
            )
        }
        if selected:
            self.logger.log(step, selected)


class StopController:
    """Signal handler that requests a checkpoint at the next optimizer boundary."""

    def __init__(self) -> None:
        """Start with no pending termination request."""
        self.requested = False

    def __call__(self) -> bool:
        """Report a signal request without interrupting a checkpoint write."""
        return self.requested

    def handle(self, _signum: int, _frame: FrameType | None) -> None:
        """Defer SIGINT/SIGTERM to the synchronized training loop."""
        self.requested = True


@contextmanager
def _signals() -> Generator[StopController, None, None]:
    controller = StopController()
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        for number in previous:
            signal.signal(number, controller.handle)
        yield controller
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _rank_zero[T](operation: Callable[[], T], device: torch.device) -> T | None:
    result: T | None = None
    failure: Exception | None = None
    if is_main_process():
        try:
            result = operation()
        except Exception as error:
            failure = error
    if any_rank(failure is not None, device):
        if failure is not None:
            raise failure
        raise RuntimeError("experiment operation failed on rank zero; see its traceback")
    return result


def experiment_plan(
    config: ExperimentConfig,
    *,
    graph: str | None = None,
    seed: int | None = None,
    condition: str | None = None,
) -> list[dict[str, str | int]]:
    """Expand configured pairs; optional selectors never alter experiment identity."""
    config.validate()
    if graph is not None and graph not in config.graphs:
        raise ValueError("selected graph is not in the experiment definition")
    if seed is not None and seed not in config.seeds:
        raise ValueError("selected seed is not in the experiment definition")
    if condition is not None and condition not in config.conditions:
        raise ValueError(f"condition must be one of {config.conditions} for this configuration")
    return [
        {
            "graph": g,
            "seed": s,
            "condition": c,
            "directory": str(Path(config.output_dir) / g / f"seed-{s}" / c),
        }
        for g in config.graphs
        if graph is None or graph == g
        for s in config.seeds
        if seed is None or seed == s
        for c in config.conditions
        if condition is None or condition == c
    ]


def _status(directory: Path, status: str, identity: str, **details: object) -> None:
    write_json(
        directory / "status.json",
        {
            "status": status,
            "run_identity": identity,
            "updated_at_unix": time.time(),
            **details,
        },
    )


def _selected_checkpoint(path: Path, trainer: Trainer, identity: str) -> dict[str, object]:
    """Verify selected weights against the current run and its selection record."""
    payload = cast(dict[str, object], torch.load(path, map_location="cpu", weights_only=False))
    if (
        payload.get("format_version") != 1
        or payload.get("dataset_identity") != trainer.dataset_identity
        or payload.get("model_config") != trainer.model_config
        or payload.get("world_size") != world_size()
    ):
        raise ValueError("best checkpoint provenance differs from this experiment run")
    metadata = payload.get("run_metadata")
    if (
        not isinstance(metadata, dict)
        or cast(dict[str, object], metadata).get("experiment_run_identity") != identity
    ):
        raise ValueError("best checkpoint belongs to a different experiment run")
    saved_config = payload.get("train_config")
    if not isinstance(saved_config, dict):
        raise ValueError("best checkpoint lacks its training configuration")
    saved_config = cast(dict[str, object], saved_config)
    expected = asdict(trainer.config)
    runtime_keys = {"out_dir", "device", "num_workers", "prefetch_factor", "log_every"}
    if any(
        saved_config.get(key) != value for key, value in expected.items() if key not in runtime_keys
    ):
        raise ValueError("best checkpoint training configuration differs")
    selection = payload.get("validation_state")
    current = trainer.validation_state
    if not isinstance(selection, dict) or any(
        cast(dict[str, object], selection).get(key) != current.get(key)
        for key in ("selection_metric", "best_score", "best_step")
    ):
        raise ValueError("best checkpoint does not match the saved validation selection")
    score = current.get("best_score")
    if (
        payload.get("step") != current.get("best_step")
        or not isinstance(payload.get("step"), int)
        or cast(int, payload["step"]) < 0
        or cast(dict[str, object], selection).get("last_step") != payload["step"]
        or not isinstance(score, int | float)
        or not math.isfinite(score)
        or current.get("selection_metric")
        != ("validation/selection_score" if trainer.config.geco.enabled else "validation/base_loss")
    ):
        raise ValueError("best checkpoint has an invalid selection step or score")
    return payload


def _checkpoint_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_tau(
    config: ExperimentConfig, manifest: dict[str, object], graph: str, seed: int
) -> dict[str, object]:
    """Require the completed dense reference and verify the bound's provenance."""
    directory = dense_directory(config, graph, seed)
    if not (directory / "tau.json").is_file():
        raise ValueError(f"dense tau calibration is missing at {directory}")
    reference_manifest = (
        external_dense_manifest(config, manifest, graph, seed)
        if config.dense_reference_dir
        else manifest
    )
    record = read_json(directory / "tau.json")
    identity = canonical_fingerprint([reference_manifest["identity"], graph, seed, "dense"])
    status = read_json(directory / "status.json")
    if (
        record.get("dense_run_identity") != identity
        or record.get("experiment_identity") != reference_manifest["identity"]
        or status.get("status") != "completed"
        or status.get("run_identity") != identity
        or record.get("checkpoint_sha256") != _checkpoint_hash(directory / "best.pt")
        or record.get("tau_multiplier") != config.tau_multiplier
    ):
        raise ValueError("dense tau calibration provenance differs from this experiment")
    tau, base = record.get("tau"), record.get("calibration/base_loss")
    if (
        not isinstance(tau, int | float)
        or not isinstance(base, int | float)
        or not math.isfinite(tau)
        or tau <= 0
        or tau != base * config.tau_multiplier
    ):
        raise ValueError("dense tau calibration is invalid")
    return record


def _sparse_selection_score(base: float, sparsity: float, tau: float) -> float:
    """Rank feasible checkpoints by sparsity, then infeasible ones by base loss."""
    if base <= tau:
        return sparsity / (1.0 + sparsity)
    return 1.0 + (base - tau) / (base + tau)


def _run_one(
    config: ExperimentConfig,
    manifest: dict[str, object],
    matrix: Tensor,
    task: dict[str, str | int],
    device: torch.device,
    stop: StopController,
    *,
    resume: bool,
    evaluate_only: bool,
) -> bool:
    graph, seed, condition = str(task["graph"]), int(task["seed"]), str(task["condition"])
    directory = Path(str(task["directory"]))
    identity = canonical_fingerprint([manifest["identity"], graph, seed, condition])
    logger: WandbLogger | None = None
    exit_code = 1
    with ExitStack() as ownership:
        _rank_zero(lambda: ownership.enter_context(file_lock(directory / ".run.lock")), device)

        def check_existing() -> bool:
            status_path = directory / "status.json"
            if status_path.exists():
                status = read_json(status_path)
                if status.get("run_identity") != identity:
                    raise ValueError("run identity differs from existing output")
                if not resume and not evaluate_only:
                    raise ValueError("run output exists; use --resume or choose a new output_dir")
                if status.get("status") == "completed" and not evaluate_only:
                    if not (directory / "evaluation" / "metrics.json").exists():
                        raise ValueError("completed run is missing evaluation; use --evaluate-only")
                    if condition == "dense":
                        _load_tau(config, manifest, graph, seed)
                    else:
                        current_calibration = _load_tau(config, manifest, graph, seed)
                        if (
                            read_json(directory / "run_config.json").get("tau_calibration")
                            != current_calibration
                        ):
                            raise ValueError(
                                "completed run tau calibration differs from dense reference"
                            )
                    return True
            elif any(
                (directory / name).exists() for name in ("last.pt", "best.pt", "run_config.json")
            ):
                raise ValueError("run artifacts exist without a status record")
            if evaluate_only and not (directory / "best.pt").exists():
                raise ValueError("--evaluate-only requires a selected best.pt checkpoint")
            return False

        skip = _rank_zero(check_existing, device)
        if any_rank(bool(skip), device):
            return False
        try:
            _rank_zero(lambda: _status(directory, "preparing", identity), device)
            seed_everything(seed)
            data_config = config.data_config(graph)
            data = build_dgp_dataset(data_config, observation_matrix=matrix)
            model_config = config.condition_model(condition)
            model = build_model(model_config)
            calibration: dict[str, object] | None = None
            if condition != "dense":
                result = _rank_zero(lambda: _load_tau(config, manifest, graph, seed), device)
                calibration = cast(dict[str, object], gather_rank_objects(result)[0])
                # _load_tau verifies this selected checkpoint's hash. Copy the
                # complete model, including EMA, but leave the new Trainer's
                # optimizer, GECO controller, RNG and step counters fresh.
                dense_checkpoint = torch.load(
                    dense_directory(config, graph, seed) / "best.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                model.load_state_dict(dense_checkpoint["model"])
            initial_identity = training_tensors_fingerprint(
                {
                    name: value
                    for name, value in model.state_dict().items()
                    if isinstance(value, Tensor)
                }
            )
            training = replace(
                config.training_config(
                    condition,
                    seed,
                    directory,
                    tau=float(cast(float, calibration["tau"])) if calibration else None,
                ),
                device=str(device),
            )
            trainer = Trainer(model, training, data, model_config=asdict(model_config))
            checkpoint = directory / ("last.pt" if (directory / "last.pt").exists() else "best.pt")
            restoring = (resume or evaluate_only) and checkpoint.exists()
            if restoring:
                trainer.load_checkpoint(checkpoint)
                if trainer.run_metadata.get("experiment_run_identity") != identity:
                    raise ValueError("checkpoint belongs to a different experiment run")
                expected_training, restored_training = asdict(training), asdict(trainer.config)
                if restored_training != expected_training:
                    raise ValueError("checkpoint training configuration differs from experiment")
            trainer.run_metadata["experiment_run_identity"] = identity
            if calibration is not None:
                if trainer.run_metadata.get("tau_calibration", calibration) != calibration:
                    raise ValueError("checkpoint tau calibration differs from the dense reference")
                trainer.run_metadata["tau_calibration"] = calibration
            record: dict[str, object] = {
                "experiment_identity": manifest["identity"],
                "run_identity": identity,
                "graph": graph,
                "seed": seed,
                "condition": condition,
                "initial_state_fingerprint": initial_identity,
                "dataset_identity": trainer.dataset_identity,
                "model": asdict(model_config),
                "train": asdict(trainer.config),
                "data": asdict(data_config),
                "evaluation": asdict(config.evaluation),
                "world_size": world_size(),
                "source_fingerprint": manifest["source_fingerprint"],
                "tau_calibration": calibration,
            }

            def start_logging() -> None:
                nonlocal logger
                saved_path = directory / "run_config.json"
                if saved_path.exists():
                    saved = read_json(saved_path)
                    for key in (
                        "run_identity",
                        "dataset_identity",
                        "initial_state_fingerprint",
                        "world_size",
                    ):
                        if saved.get(key) != record[key]:
                            raise ValueError(f"saved run {key} differs")
                write_json(saved_path, record)
                OmegaConf.save(OmegaConf.structured(config), directory / "resolved_experiment.yaml")
                saved_wandb = trainer.run_metadata.get("wandb")
                metadata_path = directory / "run_metadata.json"
                if saved_wandb is None and resume and metadata_path.exists():
                    saved_wandb = read_json(metadata_path).get("wandb")
                if saved_wandb is not None and not isinstance(saved_wandb, dict):
                    raise ValueError("invalid saved W&B identity")
                label = f"{graph}-{condition}-seed{seed}"
                logging_config = replace(
                    config.wandb,
                    name=f"{config.wandb.name}-{label}" if config.wandb.name else label,
                    group=config.wandb.group or config.name,
                    tags=[*config.wandb.tags, graph, condition, "sigreg"],
                )
                logger = WandbLogger(
                    logging_config,
                    resolved_config=cast(dict[str, object], json_safe(record)),
                    dataset_identity=trainer.dataset_identity,
                    out_dir=directory,
                    resume_state=cast(dict[str, object] | None, saved_wandb),
                )
                trainer.logger = ExperimentLogger(logger)
                if logger.metadata:
                    trainer.run_metadata["wandb"] = logger.metadata
                write_json(metadata_path, trainer.run_metadata)
                _status(
                    directory,
                    "evaluating" if evaluate_only else "running",
                    identity,
                    step=trainer.step,
                )

            _rank_zero(start_logging, device)
            if not evaluate_only:
                if calibration is not None and not restoring:

                    def check_initial_loss() -> None:
                        assert calibration is not None
                        dense_base = float(cast(float, calibration["calibration/base_loss"]))
                        losses = [
                            calibrate_tau(
                                trainer.model,
                                data_config,
                                matrix,
                                config.evaluation,
                                objective=training.objective,
                                training_gates=sampled,
                            )["calibration/base_loss"]
                            for sampled in (False, True)
                        ]
                        print(
                            f"{condition} warm-start base loss: dense={dense_base:.6g}, "
                            f"eval gates={losses[0]:.6g} (change={losses[0] - dense_base:+.6g}), "
                            f"sampled gates={losses[1]:.6g} "
                            f"(change={losses[1] - dense_base:+.6g}), "
                            f"tau={float(cast(float, calibration['tau'])):.6g}",
                            flush=True,
                        )

                    _rank_zero(check_initial_loss, device)

                def validation(current: SCJEPA) -> dict[str, float]:
                    scores = validate_model(
                        current,
                        data_config,
                        matrix,
                        config.evaluation,
                        objective=training.objective,
                    )
                    if calibration is not None:
                        bound = float(cast(float, calibration["tau"]))
                        scores["validation/selection_score"] = _sparse_selection_score(
                            scores["validation/base_loss"], scores["validation/sparsity"], bound
                        )
                        scores["validation/feasible"] = float(
                            scores["validation/base_loss"] <= bound
                        )
                    return scores

                trainer.train(
                    validation_callback=validation,
                    validation_every=config.validation_every,
                    selection_metric=(
                        "validation/selection_score"
                        if calibration is not None
                        else "validation/base_loss"
                    ),
                    stop_requested=stop,
                )
            if trainer.interrupted or any_rank(stop(), device):
                _rank_zero(
                    lambda: _status(directory, "interrupted", identity, step=trainer.step), device
                )
                exit_code = 130
                return True

            def final_evaluation() -> None:
                best_path = directory / "best.pt"
                if not best_path.exists():
                    raise ValueError("training finished without a finite selected checkpoint")
                payload = _selected_checkpoint(best_path, trainer, identity)
                selected = build_model(model_config).to(device)
                selected.load_state_dict(cast(dict[str, object], payload["model"]))
                metrics = evaluate_model(
                    selected,
                    data_config,
                    matrix,
                    config.evaluation,
                    directory / "evaluation",
                    objective=training.objective,
                )
                metrics["evaluation/selected_step"] = float(cast(int, payload["step"]))
                metrics["evaluation/selection_score"] = float(
                    cast(float, trainer.validation_state["best_score"])
                )
                checkpoint_fingerprint = _checkpoint_hash(best_path)
                if condition == "dense":
                    bound = calibrate_tau(
                        selected,
                        data_config,
                        matrix,
                        config.evaluation,
                        objective=training.objective,
                    )
                    tau = bound["calibration/base_loss"] * config.tau_multiplier
                    if not math.isfinite(tau) or tau <= 0:
                        raise ValueError("dense calibration must produce a finite positive tau")
                    write_json(
                        directory / "tau.json",
                        {
                            "experiment_identity": manifest["identity"],
                            "dense_run_identity": identity,
                            "checkpoint_sha256": checkpoint_fingerprint,
                            "selected_step": payload["step"],
                            "tau_multiplier": config.tau_multiplier,
                            "tau": tau,
                            **bound,
                        },
                    )
                    metrics["geco/tau"] = tau
                elif calibration is not None:
                    metrics["geco/tau"] = float(cast(float, calibration["tau"]))
                    metrics["test/feasible"] = float(
                        metrics["test/base_loss"] <= metrics["geco/tau"]
                    )
                write_json(
                    directory / "evaluation" / "selected_checkpoint.json",
                    {
                        "run_identity": identity,
                        "checkpoint_sha256": checkpoint_fingerprint,
                        "selected_step": payload["step"],
                        "selection_metric": trainer.validation_state["selection_metric"],
                        "selection_score": trainer.validation_state["best_score"],
                    },
                )
                write_json(directory / "evaluation" / "metrics.json", metrics)
                assert logger is not None
                logger.log(
                    trainer.step,
                    {
                        key: value
                        for key, value in metrics.items()
                        if not key.startswith("evaluation/")
                    },
                )
                if config.upload_evaluation_artifacts:
                    logger.log_artifact(
                        directory / "evaluation" / "metrics.json",
                        name=f"evaluation-{identity[:16]}",
                    )
                _status(
                    directory,
                    "completed",
                    identity,
                    step=trainer.step,
                    selected_step=payload["step"],
                )

            _rank_zero(final_evaluation, device)
            exit_code = 0
            return False
        except BaseException as error:
            if is_main_process():
                _status(directory, "failed", identity, error=f"{type(error).__name__}: {error}")
            raise
        finally:

            def finish_logging() -> None:
                if logger is not None:
                    try:
                        logger.finish(exit_code=exit_code)
                    except Exception as error:
                        _status(directory, "failed", identity, error=f"W&B finalization: {error}")
                        raise

            _rank_zero(finish_logging, device)


def run_experiment(
    config: ExperimentConfig,
    *,
    resume: bool = False,
    evaluate_only: bool = False,
    graph: str | None = None,
    seed: int | None = None,
    condition: str | None = None,
) -> dict[str, object]:
    """Run selected paired tasks, safely resuming and aggregating completed results."""
    plan = experiment_plan(config, graph=graph, seed=seed, condition=condition)
    device = initialize_distributed(config.train.device)
    try:
        if config.train.batch_size % world_size() or config.train.batch_size // world_size() < 2:
            raise ValueError(
                "SIGReg needs at least two episodes per rank and equal global sharding"
            )
        if config.evaluation.batch_size != config.train.batch_size // world_size():
            raise ValueError(
                "evaluation.batch_size must equal train.batch_size / world_size so tau "
                "and training use the same SIGReg statistic"
            )
        if any(
            count % config.evaluation.batch_size
            for count in (
                config.evaluation.calibration_episodes,
                config.evaluation.validation_episodes,
                config.evaluation.test_episodes,
            )
        ):
            raise ValueError("held-out episode counts must be divisible by evaluation.batch_size")
        _rank_zero(lambda: prepare_experiment(config), device)
        manifest, matrix = prepare_experiment(config)
        _rank_zero(lambda: claim_execution(config, str(manifest["identity"]), world_size()), device)
        with _signals() as stop:
            for task in plan:
                interrupted = _run_one(
                    config,
                    manifest,
                    matrix,
                    task,
                    device,
                    stop,
                    resume=resume,
                    evaluate_only=evaluate_only,
                )
                if interrupted:
                    break
        result = _rank_zero(lambda: aggregate_results(config), device)
        if result is not None and stop():
            result["interrupted"] = True
        return result or {}
    finally:
        destroy_distributed()
