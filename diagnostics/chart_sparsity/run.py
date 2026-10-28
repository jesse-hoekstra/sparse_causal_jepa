"""Run the isolated fixed-chart sparsity diagnostic with authenticated artifacts."""

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

from diagnostics.chart_sparsity.core import Chart, chart_metadata, evaluate, transform_data
from diagnostics.oracle_state_predictor.compare import make_controller, sparsity_loss
from diagnostics.oracle_state_predictor.data import OracleStateData, build_data
from diagnostics.oracle_state_predictor.graphs import Condition, build_predictor, rollout_graphs
from diagnostics.oracle_state_predictor.reference import build_calibration
from diagnostics.oracle_state_predictor.run import Config, GroupSampler, resolve_device, write_json
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.experiments.storage import source_fingerprint
from scjepa.training import seed_everything
from scjepa.training.logging import WandbConfig, WandbLogger

SCHEMA = "fixed-chart-sparsity-v1"
SPLITS = ("train", "calibration", "validation", "test")
Profile = Literal["scientific", "smoke"]


@dataclass(frozen=True)
class PrepareConfig:
    """Build one physical cache shared by every chart, policy and training seed."""

    output_dir: str
    profile: Profile = "scientific"
    data_seed: int = 0
    train_groups: int = 8192
    calibration_groups: int = 512
    validation_groups: int = 256
    test_groups: int = 512
    views: int = 10
    threads: int = 1
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None
    wandb_group: str = "diagnostic-fixed-chart-sparsity"

    def __post_init__(self) -> None:
        """Reject invalid cache sizes before generating any trajectories."""
        if self.profile not in ("scientific", "smoke"):
            raise ValueError("profile must be scientific or smoke")
        if self.profile == "scientific" and (
            self.data_seed,
            self.train_groups,
            self.calibration_groups,
            self.validation_groups,
            self.test_groups,
            self.views,
        ) != (0, 8192, 512, 256, 512, 10):
            raise ValueError("scientific cache sizes, seed and views are fixed by protocol")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)
        if self.data_seed < 0 or self.views < 2 or self.threads < 1:
            raise ValueError("require nonnegative data seed, at least two views and one thread")
        if any(getattr(self, f"{split}_groups") < 2 for split in SPLITS):
            raise ValueError("each split must have at least two independent groups")


@dataclass(frozen=True)
class TrainConfig:
    """One fixed endpoint; the coordinate chart and physical parameter inputs never learn."""

    output_dir: str
    data_dir: str
    chart: Chart
    condition: Condition
    profile: Profile = "scientific"
    seed: int = 0
    steps: int = 100_000
    batch_size: int = 16
    lr: float = 0.001
    logit_weight: float = 0.0001
    grad_clip: float = 1.0
    embed_dim: int = 64
    mlp_hidden: int = 64
    device: str = "cpu"
    threads: int = 1
    log_every: int = 100
    evaluate_every: int = 1000
    checkpoint_every: int = 1000
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None
    wandb_group: str = "diagnostic-fixed-chart-sparsity"
    dense_dir: str | None = None
    calibration_dir: str | None = None

    def __post_init__(self) -> None:
        """Validate the scientific arm and positive fixed budget."""
        if self.profile not in ("scientific", "smoke"):
            raise ValueError("profile must be scientific or smoke")
        if self.profile == "scientific" and (
            self.steps,
            self.batch_size,
            self.embed_dim,
            self.mlp_hidden,
            self.lr,
            self.logit_weight,
            self.grad_clip,
        ) != (100_000, 16, 64, 64, 0.001, 0.0001, 1.0):
            raise ValueError("scientific budget, architecture and optimizer are fixed by protocol")
        if self.chart not in ("physical", "mixed") or self.condition not in (
            "dense",
            "baseline",
            "full",
        ):
            raise ValueError("invalid chart or condition")
        if self.seed not in (0, 1):
            raise ValueError("the paired diagnostic uses only predeclared training seeds 0 and 1")
        for name in (
            "steps",
            "batch_size",
            "embed_dim",
            "mlp_hidden",
            "threads",
            "log_every",
            "evaluate_every",
            "checkpoint_every",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if any(not math.isfinite(v) or v <= 0 for v in (self.lr, self.grad_clip)):
            raise ValueError("learning rate and gradient clip must be finite and positive")
        if not math.isfinite(self.logit_weight) or self.logit_weight < 0:
            raise ValueError("logit weight must be finite and nonnegative")
        if self.condition == "dense":
            if self.dense_dir is not None or self.calibration_dir is not None:
                raise ValueError("a dense fit must start fresh, without a sparse-stage reference")
        elif self.dense_dir is None or self.calibration_dir is None:
            raise ValueError("sparse arms require their dense endpoint and shared calibration")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


@dataclass(frozen=True)
class CalibrateConfig:
    """Authenticate exactly four dense endpoints before fixing one shared constraint."""

    output_dir: str
    dense_dirs: tuple[str, ...]
    profile: Profile = "scientific"
    quality_cap: float = 0.02

    def __post_init__(self) -> None:
        """Keep the diagnostic threshold fixed outside explicitly marked CPU smoke tests."""
        if self.profile not in ("scientific", "smoke"):
            raise ValueError("profile must be scientific or smoke")
        if not math.isfinite(self.quality_cap) or self.quality_cap <= 0:
            raise ValueError("quality cap must be finite and positive")
        if self.profile == "scientific" and self.quality_cap != 0.02:
            raise ValueError("scientific forecast-MSE cap is fixed at .02")


def sha(path: Path) -> str:
    """Hash exact bytes without serializing or changing the artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_identity() -> str:
    """Bind numerical production code and every imported diagnostic implementation."""
    digest = hashlib.sha256(source_fingerprint().encode())
    root = Path(__file__).resolve().parents[1]
    paths = [root / "chart_sparsity" / name for name in ("core.py", "run.py")]
    paths += [
        root / "oracle_state_predictor" / name
        for name in ("data.py", "run.py", "graphs.py", "compare.py", "reference.py")
    ]
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from a trusted local artifact directory."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def save_torch(path: Path, value: object) -> None:
    """Replace a tensor artifact only after its complete serialization."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _fresh_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _finish_manifest(output: Path, record: dict[str, object], names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "code_identity": code_identity(),
        **record,
        "hashes": {name: sha(output / name) for name in names},
    }
    write_json(output / "completion.json", result)
    return result


def verify_completion(
    directory: Path, *, kind: str | None = None, require_current_source: bool = False
) -> dict[str, Any]:
    """Verify every declared artifact and reject stale code or mismatched stage types."""
    result = read_json(directory / "completion.json")
    if result.get("schema") != SCHEMA:
        raise ValueError("completion schema differs")
    if require_current_source and result.get("code_identity") != code_identity():
        raise ValueError("completion numerical source identity differs")
    if kind is not None and result.get("kind") != kind:
        raise ValueError("completion stage differs")
    hashes = result.get("hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("completion lacks artifact hashes")
    for name, expected in cast(dict[str, str], hashes).items():
        if Path(name).name != name or sha(directory / name) != expected:
            raise ValueError(f"artifact hash differs: {name}")
    return result


def prepare(config: PrepareConfig) -> dict[str, Any]:
    """Generate disjoint physical groups once, with online progress when configured."""
    torch.set_num_threads(config.threads)
    output = Path(config.output_dir).resolve()
    _fresh_output(output)
    logger = WandbLogger(
        WandbConfig(
            mode=config.wandb_mode,
            project=config.wandb_project,
            entity=config.wandb_entity,
            group=config.wandb_group,
            name="chart-data-prepare",
            job_type="fixed-chart-data",
        ),
        resolved_config=cast(dict[str, object], asdict(config)),
        dataset_identity=None,
        out_dir=output,
    )
    write_json(output / "wandb.json", logger.metadata)
    datasets: dict[str, OracleStateData] = {}
    try:
        for step, split in enumerate(SPLITS, 1):
            groups = getattr(config, f"{split}_groups")
            datasets[split] = (
                build_calibration(groups, seed=config.data_seed, views=config.views, progress=True)
                if split == "calibration"
                else build_data(
                    split, groups, seed=config.data_seed, views=config.views, progress=True
                )
            )
            logger.log(step, {f"prepared/{split}_groups": float(groups)})
        config_values = asdict(config)
        for name in (
            "output_dir",
            "threads",
            "wandb_mode",
            "wandb_project",
            "wandb_entity",
            "wandb_group",
        ):
            config_values.pop(name)
        metadata = {
            "schema": SCHEMA,
            "kind": "data",
            "config": config_values,
            "code_identity": code_identity(),
            "datasets": {split: data.metadata for split, data in datasets.items()},
            "charts": {chart: chart_metadata(chart) for chart in ("physical", "mixed")},
            "privileged_control": True,
            "protocol": (
                "true s9 and theta; observed a9/a10; true s10/s11; attached two-step rollout"
            ),
        }
        metadata["identity"] = canonical_fingerprint(metadata)
        save_torch(output / "data.pt", datasets)
        write_json(output / "metadata.json", metadata)
        logger.log(len(SPLITS) + 1, {"diagnostic/complete": 1.0})
        logger.finish()
        return _finish_manifest(
            output,
            {
                "kind": "data",
                "identity": metadata["identity"],
                "profile": config.profile,
                "wandb": logger.metadata,
            },
            ["data.pt", "metadata.json", "wandb.json"],
        )
    except BaseException:
        logger.finish(exit_code=1)
        raise


def load_data(directory: Path) -> tuple[dict[str, OracleStateData], dict[str, Any]]:
    """Authenticate a shared cache before loading its trusted tensor payload."""
    completion = verify_completion(directory, kind="data", require_current_source=True)
    metadata = read_json(directory / "metadata.json")
    if metadata.get("identity") != completion.get("identity"):
        raise ValueError("cache metadata identity differs")
    payload = cast(
        dict[str, OracleStateData],
        torch.load(directory / "data.pt", map_location="cpu", weights_only=False),
    )
    if set(payload) != set(SPLITS):
        raise ValueError("cache must contain exactly train/calibration/validation/test")
    for split, data in payload.items():
        if canonical_fingerprint(data.metadata) != canonical_fingerprint(
            metadata["datasets"][split]
        ):
            raise ValueError(f"cache descriptor differs for {split}")
    return payload, completion


def _model_config(config: TrainConfig, data: dict[str, OracleStateData]) -> Config:
    return Config(
        output_dir=config.output_dir,
        steps=config.steps,
        seed=config.seed,
        data_seed=data["train"].config.root_seed,
        batch_size=config.batch_size,
        train_groups=len(data["train"]),
        validation_groups=len(data["validation"]),
        test_groups=len(data["test"]),
        views=data["train"].views,
        lr=config.lr,
        logit_weight=config.logit_weight,
        grad_clip=config.grad_clip,
        embed_dim=config.embed_dim,
        mlp_hidden=config.mlp_hidden,
        device=config.device,
        threads=config.threads,
        log_every=config.log_every,
        evaluate_every=config.evaluate_every,
        checkpoint_every=config.checkpoint_every,
        wandb_mode=config.wandb_mode,
        wandb_project=config.wandb_project,
        wandb_entity=config.wandb_entity,
        wandb_group=config.wandb_group,
    )


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
        "dense_dir",
        "calibration_dir",
    ):
        result.pop(name)
    return result


def _state_identity(state: dict[str, Tensor]) -> str:
    return canonical_fingerprint({name: tensor_fingerprint(value) for name, value in state.items()})


def _checkpoint(path: Path) -> dict[str, Any]:
    result = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))
    if result.get("schema") != SCHEMA:
        raise ValueError("checkpoint schema differs")
    return result


def _dense_reference(
    directory: Path, config: TrainConfig, cache: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    completed = verify_completion(directory, kind="train", require_current_source=True)
    record = read_json(directory / "config.json")
    expected = _scientific_config(config)
    source = dict(record["scientific_config"])
    for name in ("steps", "condition"):
        expected.pop(name)
        source.pop(name)
    if completed.get("condition") != "dense" or source != expected:
        raise ValueError("dense reference chart, seed or scientific configuration differs")
    if record.get("data_identity") != cache["identity"]:
        raise ValueError("dense reference used a different physical cache")
    checkpoint = _checkpoint(directory / "last.pt")
    if checkpoint["identity"] != completed["identity"] or checkpoint["step"] != completed["steps"]:
        raise ValueError("dense reference is not its fixed final checkpoint")
    return checkpoint, {
        "directory": str(directory.resolve()),
        "identity": completed["identity"],
        "completion_sha256": sha(directory / "completion.json"),
        "checkpoint_sha256": sha(directory / "last.pt"),
        "chart": config.chart,
        "seed": config.seed,
        "steps": checkpoint["step"],
    }


def _calibration_reference(
    directory: Path, dense: dict[str, Any], profile: Profile
) -> dict[str, Any]:
    completed = verify_completion(directory, kind="calibration", require_current_source=True)
    result = read_json(directory / "calibration.json")
    if result.get("identity") != completed["identity"] or result.get("profile") != profile:
        raise ValueError("shared calibration identity or profile differs")
    if profile == "scientific" and result.get("prediction_mse_bound") != 0.02:
        raise ValueError("scientific prediction-quality bound differs")
    if not math.isfinite(float(result["tau"])) or float(result["tau"]) <= 0:
        raise ValueError("shared GECO bound must be finite and positive")
    if not result.get("admitted"):
        raise ValueError("dense feasibility admission failed; sparse training is prohibited")
    sources = cast(list[dict[str, Any]], result["dense_sources"])
    if not any(
        row["identity"] == dense["identity"]
        and row["checkpoint_sha256"] == dense["checkpoint_sha256"]
        and row["completion_sha256"] == dense["completion_sha256"]
        for row in sources
    ):
        raise ValueError("dense source is not pinned by the shared calibration")
    return result | {"artifact_sha256": sha(directory / "calibration.json")}


def _log(logger: WandbLogger, output: Path, step: int, metrics: dict[str, float]) -> None:
    logger.log(step, metrics)
    with (output / "history.jsonl").open("a") as stream:
        stream.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")


def _add_bounds(metrics: dict[str, float], calibration: dict[str, Any] | None) -> None:
    if calibration is not None:
        metrics["base_minus_tau"] = metrics["base_loss"] - float(calibration["tau"])
        metrics["feasible"] = float(metrics["base_minus_tau"] <= 0)
        bound = float(calibration["prediction_mse_bound"])
        metrics["prediction_mse_minus_bound"] = metrics["prediction_mse"] - bound
        metrics["prediction_quality_pass"] = float(metrics["prediction_mse"] <= bound)


def train(config: TrainConfig, *, resume: bool = False) -> dict[str, Any]:
    """Fit one predictor to a fixed chart, preserving exact continuation provenance."""
    torch.set_num_threads(config.threads)
    device = resolve_device(config.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(config.output_dir).resolve()
    if not resume:
        _fresh_output(output)
    elif not (output / "last.pt").is_file():
        raise FileNotFoundError("resume requires a last.pt checkpoint")
    physical, cache = load_data(Path(config.data_dir))
    if cache.get("profile") != config.profile:
        raise ValueError("training and cache profiles differ")
    data = {split: transform_data(item, config.chart) for split, item in physical.items()}
    seed_everything(config.seed)
    model = build_predictor(_model_config(config, data), config.condition).to(device)
    dense: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    if config.condition != "dense":
        checkpoint, dense = _dense_reference(Path(cast(str, config.dense_dir)), config, cache)
        model.load_state_dict(checkpoint["model"], strict=True)
        calibration = _calibration_reference(
            Path(cast(str, config.calibration_dir)), dense, config.profile
        )
    initial_identity = _state_identity(cast(dict[str, Tensor], model.state_dict()))
    specification = {
        "scientific_config": _scientific_config(config),
        "data_identity": cache["identity"],
        "data_sha256": cache["hashes"]["data.pt"],
        "code_identity": code_identity(),
        "dense_source": dense,
        "calibration": calibration,
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
        "objective": (
            f"two-step MSE + {config.logit_weight} logit; GECO-weighted declared block path means"
        ),
        "test_policy": "dense: no test inference; sparse: final fixed-budget endpoint only",
    }
    if resume and (output / "completion.json").is_file():
        completed = verify_completion(output, kind="train", require_current_source=True)
        if completed["identity"] != identity:
            raise ValueError("completed run provenance differs")
        return completed
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0)
    sampler = GroupSampler(len(data["train"]), config.batch_size, config.seed)
    controller = None if calibration is None else make_controller(float(calibration["tau"]), device)
    restored: dict[str, Any] = {}
    start = 0
    if resume:
        restored = _checkpoint(output / "last.pt")
        if restored.get("identity") != identity:
            raise ValueError("checkpoint configuration, source, data or ancestry differs")
        model.load_state_dict(restored["model"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        sampler.load_state_dict(restored["sampler"])
        start = int(restored["step"])
        if not 0 <= start <= config.steps:
            raise ValueError("checkpoint step exceeds its fixed budget")
        if controller is not None:
            controller.load_state_dict(restored["controller"])
            controller.validate_state()
            if int(controller.num_updates) != start:
                raise ValueError("controller and optimizer update counts disagree")
        for state in optimizer.state.values():
            if int(cast(Tensor, state["step"])) != start:
                raise ValueError("Adam state and successful-update count disagree")
        torch.set_rng_state(restored["torch_rng"])
        random.setstate(restored["python_rng"])
        np.random.set_state(restored["numpy_rng"])  # noqa: NPY002 - restore seed_everything global RNG
        if device.type == "cuda":
            torch.cuda.set_rng_state(restored["cuda_rng"], device)
    write_json(output / "config.json", record)
    logger = WandbLogger(
        WandbConfig(
            mode=config.wandb_mode,
            project=config.wandb_project,
            entity=config.wandb_entity,
            group=config.wandb_group,
            name=f"chart-{config.chart}-{config.condition}-s{config.seed}",
            job_type="fixed-chart-sparsity",
            tags=["diagnostic", "oracle", config.chart, config.condition],
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
                "controller": None if controller is None else controller.state_dict(),
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
            initial = evaluate(
                model,
                data["validation"],
                config.batch_size,
                device,
                config.logit_weight,
                config.chart,
            )
            _add_bounds(initial.metrics, calibration)
            write_json(output / "startup.json", initial.metrics)
            _log(logger, output, 0, {f"validation/{k}": v for k, v in initial.metrics.items()})
            save(0)
        model.train()
        for step in range(start + 1, config.steps + 1):
            batch = data["train"].batch(sampler.next(), device)
            optimizer.zero_grad(set_to_none=True)
            prediction, logit, penalties, _ = rollout_graphs(
                model, batch.initial_states, batch.parameters, batch.actions
            )
            mse = (prediction - batch.targets).square().mean()
            base = mse + config.logit_weight * logit
            sparse = sparsity_loss(penalties, config.condition)
            weight = (
                base.new_zeros(())
                if controller is None
                else controller.penalty_weight.detach().clone()
            )
            loss = base + weight * sparse
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss at successful update {step}")
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
                    "loss/total": float(loss.detach()),
                    "train/successful_updates": float(step),
                    "train/elapsed_seconds": time.monotonic() - started,
                    "loss/state_sparsity": float(penalties.state.detach()),
                    "loss/action_sparsity": float(penalties.action.detach()),
                    "loss/parameter_sparsity": float(penalties.parameter.detach()),
                }
                if controller is not None:
                    values |= {
                        "geco/tau": controller.tau,
                        "geco/constraint": float(base.detach()) - controller.tau,
                        "geco/applied_sparsity_weight": float(weight),
                        "geco/sparsity_weight": float(controller.penalty_weight),
                        "geco/moving_average_constraint": float(controller.ma_error),
                    }
                _log(logger, output, step, values)
            if step % config.evaluate_every == 0 and step != config.steps:
                evaluated = evaluate(
                    model,
                    data["validation"],
                    config.batch_size,
                    device,
                    config.logit_weight,
                    config.chart,
                )
                _add_bounds(evaluated.metrics, calibration)
                _log(
                    logger,
                    output,
                    step,
                    {f"validation/{k}": v for k, v in evaluated.metrics.items()},
                )
            if step % config.checkpoint_every == 0 or step == config.steps:
                save(step)
        names = ["config.json", "wandb.json", "startup.json", "history.jsonl", "last.pt"]
        for split in (
            ("calibration", "validation") if config.condition == "dense" else ("validation", "test")
        ):
            evaluated = evaluate(
                model, data[split], config.batch_size, device, config.logit_weight, config.chart
            )
            _add_bounds(evaluated.metrics, calibration)
            write_json(
                output / f"{split}.json", {"metrics": evaluated.metrics, "graphs": evaluated.graphs}
            )
            save_torch(output / f"{split}.pt", {"samples": evaluated.samples})
            names += [f"{split}.json", f"{split}.pt"]
            _log(
                logger,
                output,
                config.steps,
                {f"{split}/{k}": v for k, v in evaluated.metrics.items()},
            )
        _log(logger, output, config.steps, {"diagnostic/complete": 1.0})
        logger.finish()
        return _finish_manifest(
            output,
            {
                "kind": "train",
                "identity": identity,
                "condition": config.condition,
                "profile": config.profile,
                "chart": config.chart,
                "seed": config.seed,
                "steps": config.steps,
                "successful_updates": config.steps,
                "skipped_updates": 0,
                "data_identity": cache["identity"],
                "initial_state_identity": initial_identity,
                "wandb": logger.metadata,
            },
            names,
        )
    except BaseException:
        logger.finish(exit_code=1)
        raise


def calibration_rule(rows: list[dict[str, Any]], quality_cap: float = 0.02) -> dict[str, Any]:
    """Apply one prospective feasibility cap and max-calibration-base GECO rule."""
    if not rows or not math.isfinite(quality_cap) or quality_cap <= 0:
        raise ValueError("calibration requires finite positive cap and endpoint rows")
    failures: list[dict[str, object]] = []
    for row in rows:
        for split in ("calibration", "validation"):
            mse = float(row[f"{split}_metrics"]["prediction_mse"])
            base = float(row[f"{split}_metrics"]["base_loss"])
            if not all(math.isfinite(v) and v >= 0 for v in (mse, base)):
                raise ValueError("nonfinite or negative dense endpoint loss")
            if mse > quality_cap:
                failures.append(
                    {
                        "chart": row["chart"],
                        "seed": row["seed"],
                        "split": split,
                        "prediction_mse": mse,
                    }
                )
    tau = max(float(row["calibration_metrics"]["base_loss"]) for row in rows)
    if tau <= 0:
        raise ValueError("calibrated GECO bound must be strictly positive")
    return {
        "admitted": not failures,
        "failures": failures,
        "tau": tau,
        "prediction_mse_bound": quality_cap,
        "rule": "tau=max four dense calibration base losses; every calibration/validation MSE<=cap",
        "interpretation": (
            "forecast feasibility diagnostic; not theorem certification or Experiment1 success"
        ),
    }


def calibrate(config: CalibrateConfig) -> dict[str, Any]:
    """Pin four final dense fits and derive the prospectively declared common bounds."""
    output = Path(config.output_dir).resolve()
    _fresh_output(output)
    if len(config.dense_dirs) != 4:
        raise ValueError("calibration requires physical/mixed x seeds0/1, exactly four dense fits")
    rows: list[dict[str, Any]] = []
    signatures: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for item in config.dense_dirs:
        directory = Path(item).resolve()
        completed = verify_completion(directory, kind="train", require_current_source=True)
        record = read_json(directory / "config.json")
        scientific = dict(record["scientific_config"])
        if scientific["condition"] != "dense" or scientific["profile"] != config.profile:
            raise ValueError("shared calibration requires matching-profile dense endpoints")
        pair = (str(scientific.pop("chart")), int(scientific.pop("seed")))
        if pair in seen:
            raise ValueError("duplicate chart/seed in dense calibration")
        seen.add(pair)
        checkpoint = _checkpoint(directory / "last.pt")
        if (
            checkpoint["identity"] != completed["identity"]
            or checkpoint["step"] != scientific["steps"]
        ):
            raise ValueError("dense endpoint is not the declared final checkpoint")
        signatures.append(
            {
                "scientific": scientific,
                "data_identity": record["data_identity"],
                "data_sha256": record["data_sha256"],
            }
        )
        rows.append(
            {
                "directory": str(directory),
                "identity": completed["identity"],
                "completion_sha256": sha(directory / "completion.json"),
                "checkpoint_sha256": sha(directory / "last.pt"),
                "chart": pair[0],
                "seed": pair[1],
                "steps": checkpoint["step"],
                "calibration_metrics": read_json(directory / "calibration.json")["metrics"],
                "validation_metrics": read_json(directory / "validation.json")["metrics"],
            }
        )
    if seen != {(chart, seed) for chart in ("physical", "mixed") for seed in (0, 1)}:
        raise ValueError("dense chart/seed coverage differs from the fixed protocol")
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("dense fits do not share scientific settings, budget and physical cache")
    result = {
        "schema": SCHEMA,
        "profile": config.profile,
        "dense_sources": rows,
        "shared_configuration": signatures[0],
        **calibration_rule(rows, config.quality_cap),
    }
    result["identity"] = canonical_fingerprint(result)
    write_json(output / "calibration.json", result)
    return _finish_manifest(
        output,
        {
            "kind": "calibration",
            "profile": config.profile,
            "identity": result["identity"],
            "admitted": result["admitted"],
        },
        ["calibration.json"],
    )


def main() -> None:
    """Expose independent preparation, training and calibration stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--output-dir", required=True)
    for field in fields(PrepareConfig):
        if field.name == "output_dir":
            continue
        default = field.default
        preparation.add_argument(
            "--" + field.name.replace("_", "-"),
            default=default,
            type=type(default) if isinstance(default, int | float) else str,
        )
    training = commands.add_parser("train")
    for name in ("output_dir", "data_dir", "chart", "condition"):
        training.add_argument("--" + name.replace("_", "-"), required=True)
    for field in fields(TrainConfig):
        if field.name in ("output_dir", "data_dir", "chart", "condition"):
            continue
        default = field.default
        training.add_argument(
            "--" + field.name.replace("_", "-"),
            default=default,
            type=type(default) if isinstance(default, int | float) else str,
        )
    training.add_argument("--resume", action="store_true")
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--output-dir", required=True)
    calibration.add_argument("--dense-dir", action="append", required=True)
    calibration.add_argument("--profile", default="scientific")
    calibration.add_argument("--quality-cap", type=float, default=0.02)
    arguments = vars(parser.parse_args())
    stage = arguments.pop("stage")
    if stage == "prepare":
        result = prepare(PrepareConfig(**arguments))
    elif stage == "train":
        resume = bool(arguments.pop("resume"))
        result = train(TrainConfig(**arguments), resume=resume)
    else:
        result = calibrate(
            CalibrateConfig(
                arguments["output_dir"],
                tuple(arguments["dense_dir"]),
                arguments["profile"],
                arguments["quality_cap"],
            )
        )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
