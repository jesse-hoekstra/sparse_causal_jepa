"""Experiment 1's complete acquisition protocol, adapted only to visual two-lag mechanics."""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from diagnostics.coupled_duffing.acquisition import initialize_shared_output, project
from diagnostics.coupled_duffing.data import DataConfig, prepare
from diagnostics.coupled_duffing.evaluate import collect, graph_metrics, metrics, prediction_mse
from diagnostics.coupled_duffing.model import LaggedPredictor, ParameterArchitecture, build_model
from diagnostics.coupled_duffing.regularization import (
    STATE_REGULARIZATIONS,
    StateRegularization,
    StateRegularizationObjective,
)
from diagnostics.coupled_duffing.run import code_identity, forward_loss, objective_for
from diagnostics.masked_mlp_learned.selection import training_score
from diagnostics.shared.io import read_json, save_torch, sha
from diagnostics.shared.sampling import GroupSampler, resolve_device
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import write_json
from scjepa.models.model import SCJEPA
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import seed_everything
from scjepa.training.rng import preserve_rng

SCHEMA = "duffing-experiment1-protocol-v1"
BUDGETS = {
    "joint": 100000,
    "acquisition": 100000,
    "adaptation": 50000,
    "baseline": 350000,
    "full": 350000,
}


def experiment_data() -> DataConfig:
    """Match Experiment 1's system/episode counts, context and attached forecast length."""
    return DataConfig(train_groups=8192, validation_groups=256, views=10, context=10, forecast=2)


@dataclass(frozen=True)
class StageConfig:
    """One fixed-budget phase; acquisition attempts differ only in initialization seed."""

    data_dir: str
    output_dir: str
    phase: str
    source_dir: str | None = None
    seed: int = 0
    acquisition_seed: int | None = None
    steps: int | None = None
    batch_size: int = 16
    log_every: int = 100
    eval_every: int = 5000
    device: str = "cuda:0"
    threads: int = 1
    wandb_mode: str = "online"
    wandb_group: str = "coupled-duffing-experiment1-20261108"
    parameter_encoder: ParameterArchitecture = "mlp"
    cache_device: Literal["model", "cpu"] = "model"
    state_regularization: StateRegularization = "aligned"
    source_code_identity: str | None = None
    calibration_file: str | None = None
    learning_rate: float = 0.001

    def __post_init__(self) -> None:
        """Reject ambiguous stages or invalid fixed budgets before opening an output."""
        if self.phase not in BUDGETS:
            raise ValueError("unknown training phase")
        if self.parameter_encoder not in {"mlp", "temporal-cnn"}:
            raise ValueError("unknown parameter encoder architecture")
        if self.cache_device not in {"model", "cpu"}:
            raise ValueError("cache_device must be model or cpu")
        if self.state_regularization not in STATE_REGULARIZATIONS:
            raise ValueError("unknown state regularization")
        if (self.phase == "joint") != (self.source_dir is None):
            raise ValueError("only joint training starts without a source endpoint")
        if (self.phase == "acquisition") != (self.acquisition_seed is not None):
            raise ValueError("only acquisition requires a distinct initialization seed")
        if self.source_code_identity is not None and self.source_dir is None:
            raise ValueError("an expected historical source requires a source endpoint")
        if self.calibration_file is not None and self.phase not in {"baseline", "full"}:
            raise ValueError("shared calibration applies only to sparse training")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if min(self.updates, self.batch_size, self.log_every, self.eval_every, self.threads) < 1:
            raise ValueError("budgets, intervals, batch size and threads must be positive")

    @property
    def updates(self) -> int:
        """Return the successful Experiment 1 budget, unless explicitly overridden for a test."""
        return BUDGETS[self.phase] if self.steps is None else self.steps

    @property
    def code_dim(self) -> int:
        """Use an overcomplete eight-code warm-up, then four projected coordinates."""
        return 8 if self.phase in {"joint", "acquisition"} else 4


def load_data(
    directory: Path, device: torch.device
) -> tuple[dict[str, Any], dict[str, dict[str, Tensor]]]:
    """Authenticate input caches only; never read the training reference labels."""
    record = read_json(directory / "manifest.json")
    if (
        canonical_fingerprint({k: v for k, v in record.items() if k != "identity"})
        != record["identity"]
    ):
        raise ValueError("changed data manifest")
    if record["config"]["forecast"] != 2:
        raise ValueError("the matched protocol requires exactly two attached forecast steps")
    inputs: dict[str, dict[str, Tensor]] = {}
    for split in ("train", "calibration", "validation"):
        name = f"{split}-inputs.pt"
        if sha(directory / name) != record["hashes"][name]:
            raise ValueError(f"changed input cache: {name}")
        payload = cast(dict[str, Tensor], torch.load(directory / name, weights_only=True))
        inputs[split] = {key: payload[key].to(device) for key in ("observations", "actions")}
    return record, inputs


def verify(directory: Path, *, expected_code_identity: str | None = None) -> dict[str, Any]:
    """Authenticate a finished stage without opening metrics, tensors or physical labels."""
    record = read_json(directory / "completion.json")
    expected = code_identity() if expected_code_identity is None else expected_code_identity
    if record["schema"] != SCHEMA or record["code_identity"] != expected:
        raise ValueError("source endpoint belongs to a different numerical implementation")
    for name, digest in record["hashes"].items():
        if sha(directory / name) != digest:
            raise ValueError(f"changed endpoint artifact: {directory / name}")
    return record


def load_model(
    directory: Path,
    context: int,
    device: torch.device,
    *,
    expected_code_identity: str | None = None,
) -> tuple[SCJEPA, dict[str, Any]]:
    """Load an authenticated model for a named next stage without using evaluation labels."""
    record = verify(directory, expected_code_identity=expected_code_identity)
    model = build_model(
        record["seed"],
        context,
        code_dim=record["code_dim"],
        sparse=record["phase"] in {"baseline", "full"},
        parameter_encoder=record.get("parameter_encoder", "mlp"),
        parameter_sparsity=record["phase"] == "full",
    ).to(device)
    state = cast(
        dict[str, Any], torch.load(directory / "last.pt", map_location=device, weights_only=False)
    )
    if state["identity"] != record["identity"]:
        raise ValueError("checkpoint identity differs from completion")
    model.load_state_dict(state["model"], strict=True)
    return model, record


def _parent(record: dict[str, Any], directory: str) -> dict[str, Any]:
    return {
        "directory": str(Path(directory).resolve()),
        "identity": record["identity"],
        "checkpoint_sha256": record["hashes"]["last.pt"],
        "phase": record["phase"],
        "code_identity": record["code_identity"],
    }


def calibrate_endpoint(
    source_dir: Path, output: Path, source_code_identity: str | None = None
) -> dict[str, Any]:
    """Set one tolerance from the authenticated adapted seed-zero calibration forecasts."""
    if output.exists():
        raise FileExistsError(output)
    source = verify(source_dir, expected_code_identity=source_code_identity)
    if source["phase"] != "adaptation" or source["seed"] != 0 or source["code_dim"] != 4:
        raise ValueError("shared calibration requires the adapted four-code seed-zero endpoint")
    samples = cast(dict[str, Tensor], torch.load(source_dir / "calibration.pt", weights_only=True))
    tau = prediction_mse(samples)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("invalid calibration prediction MSE")
    result = {
        "schema": "duffing-shared-calibration-v1",
        "code_identity": code_identity(),
        "source": _parent(source, str(source_dir)),
        "data_identity": source["data_identity"],
        "parameter_encoder": source["parameter_encoder"],
        "state_regularization": source["state_regularization"],
        "tau": tau,
        "factor": 1.0,
        "groups": len(samples["states"]),
        "weight_updates": 0,
        "test_evaluation": False,
    }
    result["identity"] = canonical_fingerprint(result)
    write_json(output, result)
    return result


def _shared_calibration(config: StageConfig, data_identity: str) -> dict[str, Any] | None:
    if config.calibration_file is None:
        return None
    record = read_json(Path(config.calibration_file))
    if (
        record["schema"] != "duffing-shared-calibration-v1"
        or record["identity"]
        != canonical_fingerprint({key: value for key, value in record.items() if key != "identity"})
        or record["code_identity"] != code_identity()
        or record["data_identity"] != data_identity
        or record["parameter_encoder"] != config.parameter_encoder
        or record["state_regularization"] != config.state_regularization
        or record["factor"] != 1.0
        or not math.isfinite(record["tau"])
        or record["tau"] <= 0
    ):
        raise ValueError("shared calibration identity or scientific settings differ")
    source = verify(
        Path(record["source"]["directory"]),
        expected_code_identity=record["source"]["code_identity"],
    )
    if source["identity"] != record["source"]["identity"]:
        raise ValueError("shared calibration source differs")
    return record


def _complete(
    directory: Path, specification: dict[str, Any], names: list[str], **extra: object
) -> None:
    if code_identity() != specification["code_identity"]:
        raise RuntimeError("numerical source changed during this stage")
    write_json(
        directory / "completion.json",
        {
            **specification,
            **extra,
            "hashes": {name: sha(directory / name) for name in names},
        },
    )


def train(config: StageConfig, *, resume: bool = False) -> None:
    """Train one phase with the original optimizer, donor RNG, objective and EMA conventions."""
    torch.set_num_threads(config.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = resolve_device(config.device)
    seed_everything(config.seed)
    data_dir, output = Path(config.data_dir), Path(config.output_dir)
    data_record, inputs = load_data(
        data_dir, torch.device("cpu") if config.cache_device == "cpu" else device
    )
    context = int(data_record["config"]["context"])
    frozen = config.phase in {"acquisition", "adaptation"}
    sparse = config.phase in {"baseline", "full"}
    calibration = _shared_calibration(config, data_record["identity"])
    model = build_model(
        config.acquisition_seed if config.acquisition_seed is not None else config.seed,
        context,
        code_dim=config.code_dim,
        sparse=sparse,
        parameter_encoder=config.parameter_encoder,
        parameter_sparsity=config.phase == "full",
    ).to(device)
    source: dict[str, Any] | None = None
    if config.source_dir is not None:
        inherited, record = load_model(
            Path(config.source_dir),
            context,
            device,
            expected_code_identity=config.source_code_identity,
        )
        expected = {
            "acquisition": "joint",
            "adaptation": "projection",
            "baseline": "adaptation",
            "full": "adaptation",
        }[config.phase]
        if (
            record["phase"] != expected
            or record["seed"] != config.seed
            or record["data_identity"] != data_record["identity"]
            or record.get("parameter_encoder", "mlp") != config.parameter_encoder
            or record.get("state_regularization", "aligned") != config.state_regularization
        ):
            raise ValueError("source phase, training seed, architecture or data differs")
        source = _parent(record, config.source_dir)
        if config.phase == "acquisition":
            model.online_encoder.load_state_dict(inherited.online_encoder.state_dict())
            model.target_encoder.load_state_dict(inherited.target_encoder.state_dict())
        elif config.phase == "adaptation":
            model.load_state_dict(inherited.state_dict(), strict=True)
        else:
            missing, unexpected = cast(
                tuple[list[str], list[str]],
                model.load_state_dict(inherited.state_dict(), strict=False),
            )
            expected_missing = {"predictor.state_logits", "predictor.action_logits"}
            if config.phase == "full":
                expected_missing.add("predictor.parameter_logits")
            if set(missing) != expected_missing or unexpected:
                raise ValueError("sparse initialization changed inherited tensors")
        del inherited
    model.online_encoder.requires_grad_(not frozen)
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=resume)
    specification: dict[str, Any] = {
        "schema": SCHEMA,
        "config": asdict(config),
        "phase": config.phase,
        "seed": config.seed,
        "steps": config.updates,
        "code_dim": config.code_dim,
        "parameter_encoder": config.parameter_encoder,
        "state_regularization": config.state_regularization,
        "state_projector": [2, 128, 128, 2]
        if config.state_regularization.startswith("projected-")
        else None,
        "state_variance_floor_weight": 0.4
        if config.state_regularization.endswith("-variance")
        else 0.0,
        "state_variance_floor_std": 1.0,
        "state_variance_floor_epsilon": 1e-4,
        "sigreg_sampling": "absolute_time"
        if "mixture" not in config.state_regularization
        else "within_system_time_view_mixture",
        "code_identity": code_identity(),
        "data_identity": data_record["identity"],
        "source": source,
        "true_parameter_training": False,
        "true_state_training": False,
        "parameter_gates": "learned" if config.phase == "full" else "fixed_open",
        "shared_calibration": calibration,
        "test_evaluation": False,
        "learning_rate": config.learning_rate,
        "weight_decay": 0.0,
        "gradient_clip": 1.0,
        "ema_decay": 0.996,
        "sigreg_weight": 0.4,
        "sigreg_directions": 256,
        "parameter_auxiliary_loss": False,
        "initialization": "shared_feature"
        if config.phase in {"joint", "acquisition"}
        else "inherited",
        "history_len": 2,
        "forecast_steps": 2,
        "initial_gate_logit": 3.0,
    }
    specification["identity"] = canonical_fingerprint(specification)
    saved: dict[str, Any] = {}
    start = 0
    if resume:
        saved = cast(
            dict[str, Any], torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        )
        if saved["identity"] != specification["identity"]:
            raise ValueError("resume configuration, source or data changed")
        model.load_state_dict(saved["model"], strict=True)
        start = int(saved["step"])
    elif config.phase in {"joint", "acquisition"}:
        write_json(output / "initialization.json", initialize_shared_output(model, inputs["train"]))
    write_json(output / "config.json", specification)
    tau: float | None = None
    if sparse:
        if resume:
            tau = float(saved["tau"])
        elif calibration is not None:
            tau = float(calibration["tau"])
            write_json(output / "calibration.json", calibration)
        else:
            with preserve_rng(device):
                # Calibrate the dense initializer, before adding stochastic gates.
                dense, _ = load_model(
                    Path(cast(str, config.source_dir)),
                    context,
                    device,
                    expected_code_identity=config.source_code_identity,
                )
                tau = prediction_mse(collect(dense, inputs["calibration"], config.batch_size))
                del dense
            if not math.isfinite(tau) or tau <= 0:
                raise ValueError("invalid calibration prediction MSE")
            write_json(
                output / "calibration.json",
                {
                    "tau": tau,
                    "factor": 1.0,
                    "source": source,
                    "groups": len(inputs["calibration"]["observations"]),
                    "weight_updates": 0,
                },
            )
    objective = objective_for(
        config.phase if sparse else "joint",
        0.4,
        device,
        state_regularization=config.state_regularization,
        seed=config.seed,
    )
    if isinstance(objective, StateRegularizationObjective) and config.source_dir is not None:
        parent_checkpoint = cast(
            dict[str, Any],
            torch.load(
                Path(config.source_dir) / "last.pt", map_location=device, weights_only=False
            ),
        )
        objective.projector.load_state_dict(
            {
                key.removeprefix("projector."): value
                for key, value in parent_checkpoint["objective"].items()
                if key.startswith("projector.")
            },
            strict=True,
        )
        del parent_checkpoint
    objective.requires_grad_(not frozen)
    parameters = [
        p for module in (model, objective) for p in module.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0.0)
    sampler = GroupSampler(len(inputs["train"]["observations"]), config.batch_size, config.seed)
    if resume:
        optimizer.load_state_dict(saved["optimizer"])
        objective.load_state_dict(saved["objective"])
        sampler.load_state_dict(saved["sampler"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        # Drop events newer than the resumed checkpoint; preserve the original full log.
        for name in ("history.jsonl", "validation.jsonl"):
            path = output / name
            backup = output / f"{name}.before-resume-{time.time_ns()}"
            backup.write_bytes(path.read_bytes())
            path.write_text(
                "".join(
                    line + "\n"
                    for line in path.read_text().splitlines()
                    if json.loads(line)["step"] <= start
                )
            )
    labels: dict[str, dict[str, Tensor]] = {}
    for split in ("calibration", "validation"):
        name = f"{split}-labels.pt"
        if sha(data_dir / name) != data_record["hashes"][name]:
            raise ValueError("changed evaluation reference labels")
        labels[split] = cast(dict[str, Tensor], torch.load(data_dir / name, weights_only=True))
    logger = WandbLogger(
        WandbConfig(
            mode=config.wandb_mode,
            entity="jesse-hoekstra-university-of-oxford",
            project="sparse_causal_jepa",
            group=config.wandb_group,
            name=f"duffing-{config.phase}-s{config.seed}"
            + (f"-a{config.acquisition_seed}" if config.acquisition_seed is not None else ""),
            job_type=config.phase,
            tags=[
                "duffing",
                "experiment1-protocol",
                "validation-only",
                config.parameter_encoder,
                config.state_regularization,
            ],
        ),
        resolved_config=cast(dict[str, object], specification),
        dataset_identity=data_record["identity"],
        out_dir=output,
        resume_state=saved.get("wandb"),
    )
    write_json(output / "wandb.json", logger.metadata)

    def save(step: int, name: str = "last.pt") -> None:
        save_torch(
            output / name,
            {
                "identity": specification["identity"],
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "objective": objective.state_dict(),
                "sampler": sampler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "wandb": logger.metadata,
                "tau": tau,
            },
        )

    def evaluate(step: int, *, final: bool = False) -> dict[str, float]:
        with preserve_rng(device):
            cal = collect(model, inputs["calibration"], config.batch_size)
            val = collect(model, inputs["validation"], config.batch_size)
            scores = metrics(cal, val, labels["calibration"], labels["validation"], context)
            shuffled = collect(model, inputs["validation"], config.batch_size, shuffle=True)
            scores["code_shuffle/mse_ratio"] = prediction_mse(shuffled) / max(
                prediction_mse(val), 1e-12
            )
            if tau is not None:
                scores["geco/relative_prediction"] = prediction_mse(val) / tau
            graphs = cast(LaggedPredictor, model.predictor).thresholded_graphs()
            for name, graph in graphs.items():
                scores[f"gates/{name}_open"] = float(graph.sum())
            if config.code_dim == 4:
                scores.update(graph_metrics(graphs, cal, labels["calibration"]))
            with (output / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, **scores}) + "\n")
            logger.log(step, {f"validation/{key}": value for key, value in scores.items()})
            if final:
                save_torch(output / "calibration.pt", cal)
                save_torch(output / "validation.pt", val)
        return scores

    started, status = time.monotonic(), 1
    try:
        if not resume:
            save(0, "initial.pt")
            save(0)
            evaluate(0)
        final_metrics: dict[str, float] = {}
        for step in range(start + 1, config.updates + 1):
            model.train()
            if frozen:
                model.online_encoder.eval()
            indices = sampler.next().to(inputs["train"]["observations"].device)
            batch = {name: values[indices].to(device) for name, values in inputs["train"].items()}
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed((1 << 40) + config.seed + step)  # pyright: ignore[reportUnknownMemberType]
            with preserve_rng(device):
                torch.default_generator.manual_seed(config.seed + step)
                if device.type == "cuda":
                    torch.cuda.manual_seed(config.seed + step)
                loss, terms, result = forward_loss(model, batch, objective)
            if len(result.dynamics) != 2 or not bool(torch.isfinite(loss)):
                raise FloatingPointError("expected two finite attached forecast steps")
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            if not frozen:
                model.update_target()
            if objective.geco is not None:
                assert tau is not None
                objective.geco.update(terms["prediction"].detach() / tau)
            if step % config.log_every == 0 or step == config.updates:
                values = {f"loss/{key}": float(value.detach()) for key, value in terms.items()}
                values.update(
                    {
                        "loss/total": float(loss.detach()),
                        "train/gradient_norm": float(gradient),
                        "runtime/seconds_per_step": (time.monotonic() - started) / (step - start),
                    }
                )
                if objective.geco is not None:
                    values["geco/penalty_weight"] = float(objective.geco.penalty_weight)
                if device.type == "cuda":
                    values["runtime/gpu_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1e9
                with (output / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": step, **values}) + "\n")
                logger.log(step, values)
            if step % config.eval_every == 0 or step == config.updates:
                final_metrics = evaluate(step, final=step == config.updates)
                save(step)
        if start == config.updates:
            final_metrics = evaluate(start, final=True)
        names = [
            "config.json",
            "initial.pt",
            "last.pt",
            "history.jsonl",
            "validation.jsonl",
            "calibration.pt",
            "validation.pt",
        ]
        if sparse:
            names.append("calibration.json")
        if config.phase in {"joint", "acquisition"}:
            names.append("initialization.json")
        _complete(
            output,
            specification,
            names,
            metrics=final_metrics,
            elapsed_seconds=time.monotonic() - started,
        )
        status = 0
    finally:
        logger.finish(status)


def select(
    directories: list[Path],
    *,
    window: int = 5000,
    maximum_loss: float = 0.025,
    expected_candidates: int = 16,
) -> dict[str, Any]:
    """Use exactly the final-window TRAIN-loss rule; physical labels never select a restart."""
    if (
        len(directories) != expected_candidates
        or not math.isfinite(maximum_loss)
        or maximum_loss <= 0
    ):
        raise ValueError("provide the complete prespecified pool and a finite positive threshold")
    common: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    seeds: set[int] = set()
    for path in directories:
        record = verify(path)
        if record["phase"] != "acquisition":
            raise ValueError("selection accepts frozen-state acquisitions only")
        specification = {
            key: record[key]
            for key in ("source", "seed", "steps", "code_dim", "data_identity", "code_identity")
        }
        specification["parameter_encoder"] = record.get("parameter_encoder", "mlp")
        specification["state_regularization"] = record.get("state_regularization", "aligned")
        if common is not None and specification != common:
            raise ValueError("candidate budgets, source or data differ")
        common = specification
        initialization = int(record["config"]["acquisition_seed"])
        if initialization in seeds:
            raise ValueError("acquisition seeds must differ")
        seeds.add(initialization)
        score = training_score(
            path, steps=record["steps"], window=window, log_every=record["config"]["log_every"]
        )
        candidates.append(
            {
                "directory": str(path.resolve()),
                "identity": record["identity"],
                "completion_sha256": sha(path / "completion.json"),
                "acquisition_seed": initialization,
                "train_prediction_mean": score,
            }
        )
    selected = min(
        candidates, key=lambda row: (row["train_prediction_mean"], row["acquisition_seed"])
    )
    result = {
        "schema": SCHEMA,
        "common": common,
        "candidates": candidates,
        "selected": selected,
        "window": window,
        "maximum_loss": maximum_loss,
        "admitted": selected["train_prediction_mean"] <= maximum_loss,
        "selection_used_parameter_labels": False,
    }
    return {**result, "identity": canonical_fingerprint(result)}


def project_endpoint(
    data_dir: Path,
    source_dir: Path,
    output_dir: Path,
    device_name: str = "cuda:0",
    cache_device: str = "model",
) -> None:
    """Project only the selected acquisition, preserving its source and an exact transfer record."""
    device = resolve_device(device_name)
    if cache_device not in {"model", "cpu"}:
        raise ValueError("cache_device must be model or cpu")
    record, inputs = load_data(data_dir, torch.device("cpu") if cache_device == "cpu" else device)
    context = int(record["config"]["context"])
    source, inherited = load_model(source_dir, context, device)
    if (
        inherited["phase"] != "acquisition"
        or inherited["code_dim"] != 8
        or inherited["data_identity"] != record["identity"]
    ):
        raise ValueError("projection requires a matched eight-code acquisition endpoint")
    architecture = inherited.get("parameter_encoder", "mlp")
    target = build_model(inherited["seed"], context, code_dim=4, parameter_encoder=architecture).to(
        device
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    receipt, matrices = project(source, target, inputs["train"])
    specification: dict[str, Any] = {
        "schema": SCHEMA,
        "phase": "projection",
        "seed": inherited["seed"],
        "code_dim": 4,
        "parameter_encoder": architecture,
        "state_regularization": inherited.get("state_regularization", "aligned"),
        "code_identity": code_identity(),
        "data_identity": record["identity"],
        "source": _parent(inherited, str(source_dir)),
        "projection": receipt,
    }
    specification["identity"] = canonical_fingerprint(specification)
    write_json(output_dir / "config.json", specification)
    save_torch(output_dir / "matrices.pt", matrices)
    source_checkpoint = cast(
        dict[str, Any],
        torch.load(source_dir / "last.pt", map_location="cpu", weights_only=False),
    )
    save_torch(
        output_dir / "last.pt",
        {
            "identity": specification["identity"],
            "model": target.state_dict(),
            "objective": source_checkpoint["objective"],
        },
    )
    _complete(output_dir, specification, ["config.json", "matrices.pt", "last.pt"])


def main() -> None:
    """Prepare matched data, train individual stages, select restarts or project the winner."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("prepare")
    data.add_argument("--output-dir", type=Path, required=True)
    run = sub.add_parser("train")
    for name in ("data-dir", "output-dir", "phase"):
        run.add_argument(f"--{name}", required=True)
    run.add_argument("--source-dir")
    run.add_argument("--source-code-identity")
    run.add_argument("--calibration-file")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--acquisition-seed", type=int)
    run.add_argument("--steps", type=int)
    run.add_argument("--learning-rate", type=float, default=0.001)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    run.add_argument("--wandb-group", default="coupled-duffing-experiment1-20261108")
    run.add_argument("--parameter-encoder", choices=("mlp", "temporal-cnn"), default="mlp")
    run.add_argument("--cache-device", choices=("model", "cpu"), default="model")
    run.add_argument("--state-regularization", choices=STATE_REGULARIZATIONS, default="aligned")
    run.add_argument("--eval-every", type=int, default=5000)
    run.add_argument("--log-every", type=int, default=100)
    run.add_argument("--resume", action="store_true")
    selection = sub.add_parser("select")
    selection.add_argument("--runs", nargs="+", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)
    projection = sub.add_parser("project")
    for name in ("data-dir", "source-dir", "output-dir"):
        projection.add_argument(f"--{name}", type=Path, required=True)
    projection.add_argument("--device-name", default="cuda:0")
    projection.add_argument("--cache-device", choices=("model", "cpu"), default="model")
    calibration = sub.add_parser("calibrate")
    calibration.add_argument("--source-dir", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--source-code-identity")
    args = vars(parser.parse_args())
    command = args.pop("command")
    torch.set_num_threads(1)
    if command == "prepare":
        prepare(args["output_dir"], experiment_data())
    elif command == "train":
        resume = args.pop("resume")
        train(StageConfig(**args), resume=resume)
    elif command == "project":
        project_endpoint(**args)
    elif command == "calibrate":
        print(json.dumps(calibrate_endpoint(**args), indent=2))
    else:
        if args["output"].exists():
            raise FileExistsError(args["output"])
        result = select(args["runs"])
        write_json(args["output"], result)
        print(json.dumps(result, indent=2))
        if not result["admitted"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
