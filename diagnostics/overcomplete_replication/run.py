"""Paired second-seed gained six/twelve-code fresh acquisition controls."""

import argparse
import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.group_latent_control.run import (
    _restore_rng,  # pyright: ignore[reportPrivateUsage]
    _rng,  # pyright: ignore[reportPrivateUsage]
)
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.mlp_acquisition_control.evaluation import assess_model as assess_six
from diagnostics.mlp_acquisition_control.evaluation import measurement_metrics
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.mlp_forecast_only_acquisition.run import training_loss as loss_six
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.overcomplete_acquisition.evaluation import assess_model as assess_twelve
from diagnostics.overcomplete_acquisition.forward import training_loss as loss_twelve
from diagnostics.overcomplete_replication.source import SOURCE_SEED, TRAINING_SEED, Arm, load_inputs
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig


def training_loss(
    model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
) -> tuple[Tensor, dict[str, Tensor]]:
    """Use unchanged forecast-only adapters; all twelve or six code coordinates stay attached."""
    return (loss_twelve if model.predictor.param_dim == 12 else loss_six)(model, batch, objective)


def assess_model(
    model: SCJEPA,
    data: EvaluationData,
    selection: DiagnosticSelection,
    *,
    bootstrap_samples: int,
    seed: int,
    final: bool = False,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Use each frozen whole-code evaluator with a separate, fixed evaluation seed."""
    evaluator = assess_twelve if model.predictor.param_dim == 12 else assess_six
    return evaluator(
        model, data, selection, bootstrap_samples=bootstrap_samples, seed=seed, final=final
    )


@dataclass(frozen=True)
class Config:
    """Two declared second-seed reference arms; no dimension or seed sweep."""

    source_dir: str
    output_dir: str
    training_source_root: str | None = None
    arm: Arm = "gain12"
    source_seed: int = SOURCE_SEED
    seed: int = TRAINING_SEED
    evaluation_seed: int = SOURCE_SEED
    device: str = "cpu"
    steps: int = 100000
    groups: int = 64
    bootstrap_samples: int = 1000
    threads: int = 1
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Protect immutable sources and fix the scientific training/evaluation budget."""
        if self.verification:
            if not 1 <= self.steps <= 3 or not 2 <= self.groups <= 4:
                raise ValueError("Verification permits at most three updates and 2..4 groups")
        elif (
            self.steps,
            self.groups,
            self.bootstrap_samples,
            self.seed,
            self.source_seed,
            self.evaluation_seed,
        ) != (
            100000,
            64,
            1000,
            TRAINING_SEED,
            SOURCE_SEED,
            SOURCE_SEED,
        ):
            raise ValueError("Replication fixes100k updates,64groups and all three declared seeds")
        if self.arm not in ("gain6", "gain12") or self.seed == self.source_seed:
            raise ValueError("Choose a declared arm and distinct training/source seeds")
        if (
            self.threads < 1
            or self.bootstrap_samples < 2
            or min(self.seed, self.source_seed, self.evaluation_seed) < 0
        ):
            raise ValueError("Invalid threads, bootstrap count or seed")
        source, output = Path(self.source_dir).resolve(), Path(self.output_dir).resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Output must be separate from immutable source")
        if self.training_source_root:
            archive = Path(self.training_source_root).resolve()
            if output == archive or output in archive.parents or archive in output.parents:
                raise ValueError("Output must be separate from the archived training source")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def fixed_hashes(model: SCJEPA) -> dict[str, str]:
    """Only the state maps remain fixed; E and P both train."""
    return {
        name: module_fingerprint(getattr(model, name))
        for name in ("online_encoder", "target_encoder")
    }


def code_identity() -> str:
    """Bind this independent runner and every unchanged acquisition dependency."""
    from diagnostics.overcomplete_acquisition.run import code_identity as mlp_code_identity

    digest = hashlib.sha256(mlp_code_identity().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Train E/P without physical labels and evaluate only the fixed final test."""
    source = Path(config.source_dir)
    inputs = load_inputs(
        source,
        config.source_seed,
        config.seed,
        arm=config.arm,
        training_source_root=Path(config.training_source_root)
        if config.training_source_root
        else None,
        verification=config.verification,
    )
    bundle = inputs.bundle
    if not config.verification and (bundle.train_groups, bundle.batch_size) != (8192, 16):
        raise ValueError("Scientific acquisition requires original8192 groups and B16")
    device, out = torch.device(config.device), Path(config.output_dir)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        with file_lock(out / ".lock"):
            model = inputs.model.requires_grad_(False)
            source_hashes = inputs.source_hashes
            initialization = inputs.source_initialization
            replication_initialization = inputs.replication_initialization
            model = model.to(device)
            model.predictor.requires_grad_(True)
            model.parameter_encoder.requires_grad_(True)
            frozen, flags = fixed_hashes(model), model.get_extra_state()
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(parameters, lr=0.001)
            objective = AcquisitionObjective(
                ObjectiveConfig(**bundle.protocol["objective_config"])
            ).to(device)
            selection = DiagnosticSelection(
                tuple(range(config.groups)), tuple(range(config.groups))
            )
            initial_encoder_hash = module_fingerprint(model.parameter_encoder)
            initial_predictor_hash = module_fingerprint(model.predictor)
            settings = asdict(config)
            for name in (
                "source_dir",
                "output_dir",
                "training_source_root",
                "wandb_mode",
                "wandb_entity",
            ):
                settings.pop(name)
            evidence: dict[str, Any] = {
                "settings": settings,
                "source_hashes": source_hashes,
                "source_initialization": initialization,
                "initial_encoder_sha256": initial_encoder_hash,
                "replication_initialization": replication_initialization,
                "initial_predictor_sha256": initial_predictor_hash,
                "frozen_hashes": frozen,
                "code_identity": code_identity(),
                "torch": str(torch.__version__),
            }
            identity = canonical_fingerprint(evidence)
            protocol: dict[str, Any] = {
                "identity": identity,
                **evidence,
                "config": asdict(config),
                "hypothesis": (
                    "A genuinely fresh independent optimization seed tests whether the gained "
                    "twelve-code advantage persists relative to its paired gained six-code arm. "
                    "This is conditional replication on fixed data/state maps, not a new DGP."
                ),
                "controlled_change": (
                    "Authenticate originalsource_seed and reproduce its archived full-config "
                    "fresh E/P initializer. Construct full original model at newtraining_seed, "
                    "copy fresh E only, initialize fresh MLP P at same newseed, apply fixed gains. "
                    "gain12 adds unchangedseed+1/+2 expansion; gain6 skips it. No learned E/P "
                    "or Adam source history inherited. Fixed100k updates with common newseed "
                    "batches/update RNG, originalevalbootstrap seed and original fixed caches."
                ),
                "code_dimension": 12 if config.arm == "gain12" else 6,
                "physical_parameter_dimension": 6,
                "training_parameter_labels_absent": True,
                "production_training_modified": False,
                "optimized_loss_arm": "forecast_only",
                "objective": (
                    "Unchanged attached two-step forecast MSE; original auxiliary diagnostics "
                    "computed with zero optimized weights. Parameter SIGReg draw width follows "
                    "code dimension; update reseeding preserves shared donor RNG schedule."
                ),
                "evaluation": (
                    "Frozen original six-code or declared twelve-code evaluator by arm. "
                    "Same family grids, calibration fit, validation selection and fixed-final "
                    "512/256/512retention plus64test responses. Every1000monitor64cal/val. "
                    "Bootstrap seed remains original; training seed changes initialization, "
                    "batch order and per-update RNG only. Twelve-code cubic probe454vs83features."
                ),
                "decision": (
                    "Twelve-only all-six supports an interface advantage across two optimization "
                    "seeds; both succeeding shows six-code success depends on optimization seed. "
                    "Neither supports robustness, while six-only opposes a general12D advantage. "
                    "Retention and use remain separate; no test selection or graph claim."
                ),
            }
            if resume:
                if read_json(out / "protocol.json")["identity"] != identity:
                    raise ValueError("Resume settings, code or source changed")
            elif any(path.name != ".lock" for path in out.iterdir()):
                raise FileExistsError("Use a fresh output directory or explicit resume")
            else:
                write_json(out / "protocol.json", protocol)
                atomic_save(
                    out / "initial_modules.pt",
                    {
                        "identity": identity,
                        "replication_initialization": replication_initialization,
                        "predictor": model.predictor.state_dict(),
                        "parameter_encoder": model.parameter_encoder.state_dict(),
                    },
                )
            initial_payload = cast(
                dict[str, Any],
                torch.load(out / "initial_modules.pt", map_location="cpu", weights_only=False),
            )
            if (
                initial_payload["identity"] != identity
                or initial_payload["replication_initialization"] != replication_initialization
            ):
                raise ValueError("Persisted MLP initialization identity changed")
            initial_copy = copy.deepcopy(model.predictor).cpu()
            initial_copy.load_state_dict(initial_payload["predictor"], strict=True)
            if module_fingerprint(initial_copy) != initial_predictor_hash:
                raise ValueError("Persisted MLP initialization weights changed")
            encoder_copy = copy.deepcopy(model.parameter_encoder).cpu()
            encoder_copy.load_state_dict(initial_payload["parameter_encoder"], strict=True)
            if module_fingerprint(encoder_copy) != initial_encoder_hash:
                raise ValueError("Persisted encoder initialization weights changed")
            initial_sha256 = sha(out / "initial_modules.pt")
            if (out / "completed.json").exists():
                completion = cast(dict[str, Any], read_json(out / "completed.json"))
                if completion["identity"] != identity or any(
                    sha(out / name) != value for name, value in completion["hashes"].items()
                ):
                    raise ValueError("Completed artifacts changed")
                return cast(dict[str, Any], read_json(out / "results.json"))
            start = 0
            records: list[dict[str, Any]] = []
            raw: dict[int, Any] = {}
            tracking = (
                read_json(out / "tracking.json") if (out / "tracking.json").exists() else None
            )
            if resume and (out / "last.pt").exists():
                checkpoint = cast(
                    dict[str, Any],
                    torch.load(out / "last.pt", map_location="cpu", weights_only=False),
                )
                if checkpoint["identity"] != identity:
                    raise ValueError("Checkpoint identity differs")
                model.load_state_dict(checkpoint["model"], strict=True)
                optimizer.load_state_dict(checkpoint["optimizer"])
                start, records, raw = checkpoint["step"], checkpoint["records"], checkpoint["raw"]
                tracking = checkpoint["tracking"]
                _restore_rng(checkpoint["rng"], device)
            if not 0 <= start <= config.steps:
                raise ValueError("Invalid checkpoint update")

            def check_fixed() -> None:
                if sha(out / "initial_modules.pt") != initial_sha256:
                    raise RuntimeError("Persisted MLP initialization artifact changed")
                if fixed_hashes(model) != frozen or model.get_extra_state() != flags:
                    raise RuntimeError("Frozen encoders or rollout settings changed")

            check_fixed()
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=f"overcomplete-replication-{out.parent.name}",
                    job_type="diagnostic",
                ),
                resolved_config=protocol,
                dataset_identity=bundle.protocol["identity"],
                out_dir=out,
                resume_state=tracking,
            )
            write_json(out / "tracking.json", logger.metadata)

            def save(step: int) -> None:
                check_fixed()
                atomic_save(
                    out / "last.pt",
                    {
                        "identity": identity,
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng": _rng(device),
                        "records": records,
                        "raw": raw,
                        "tracking": logger.metadata,
                    },
                )

            def validate(step: int) -> None:
                check_fixed()
                report, measurements = assess_model(
                    model,
                    bundle.evaluation,
                    selection,
                    bootstrap_samples=config.bootstrap_samples,
                    seed=config.evaluation_seed,
                )
                raw[step] = measurements
                records.append({"step": step, **report})
                logger.log(step, measurement_metrics(report))
                write_json(out / "validation.json", {"identity": identity, "records": records})

            step = start
            try:
                if not records:
                    validate(0)
                    save(0)
                loaded_epoch, batches = -1, []
                for step in range(start + 1, config.steps + 1):
                    epoch, position = divmod(step - 1, bundle.train_groups // bundle.batch_size)
                    if epoch != loaded_epoch:
                        batches = list(
                            GlobalBatchSampler(
                                bundle.train_groups,
                                bundle.batch_size,
                                config.seed,
                                epoch,
                            )
                        )
                        loaded_epoch = epoch
                    selected = batches[position]
                    batch = {
                        name: value[selected].to(device) for name, value in bundle.train.items()
                    }
                    torch.manual_seed(config.seed * 100000 + step)  # pyright: ignore[reportUnknownMemberType]
                    model.train()
                    model.online_encoder.eval()
                    model.target_encoder.eval()
                    optimizer.zero_grad(set_to_none=True)
                    loss, terms = training_loss(model, batch, objective)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"Nonfinite diagnostic loss at update{step}")
                    loss.backward()  # pyright: ignore[reportUnknownMemberType]
                    norm = float(
                        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    )
                    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
                    if step == 1 or step % 100 == 0 or step == config.steps:
                        logger.log(
                            step,
                            {
                                **{
                                    f"train/{name}": float(value.detach())
                                    for name, value in terms.items()
                                },
                                "train/gradient_norm": norm,
                            },
                        )
                    if config.verification or step % 1000 == 0 or step == config.steps:
                        validate(step)
                        save(step)
                if (out / "final_test.json").exists():
                    final_record = cast(dict[str, Any], read_json(out / "final_test.json"))
                    if (
                        final_record["identity"] != identity
                        or final_record["checkpoint_sha256"] != sha(out / "last.pt")
                        or final_record["raw_sha256"] != sha(out / "final_test_measurements.pt")
                    ):
                        raise ValueError("Final test differs from fixed final checkpoint")
                    final = final_record["metrics"]
                    final_raw = torch.load(
                        out / "final_test_measurements.pt", map_location="cpu", weights_only=False
                    )
                else:
                    final, final_raw = assess_model(
                        model,
                        bundle.evaluation,
                        selection,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.evaluation_seed,
                        final=True,
                    )
                    atomic_save(out / "final_test_measurements.pt", final_raw)
                    write_json(
                        out / "final_test.json",
                        {
                            "identity": identity,
                            "checkpoint_sha256": sha(out / "last.pt"),
                            "raw_sha256": sha(out / "final_test_measurements.pt"),
                            "metrics": final,
                        },
                    )
                check_fixed()
                for name, expected in source_hashes.items():
                    if sha(source / name) != expected:
                        raise RuntimeError("Immutable source changed")
                result = {
                    "identity": identity,
                    "step": step,
                    "records": records,
                    "final_test": final,
                    "tracking": logger.metadata,
                    "scope": (
                        "Paired second-seed gained six/twelve-code optimization diagnostic, "
                        "not original SPARTAN experiment success"
                    ),
                }
                write_json(out / "results.json", result)
                atomic_save(
                    out / "measurements.pt",
                    {"identity": identity, "validation": raw, "test": final_raw},
                )
                logger.log(step, measurement_metrics(final, final=True))
                logger.log(step, {"diagnostic/complete": 1.0})
                for name in ("results.json", "protocol.json"):
                    logger.log_artifact(
                        out / name,
                        name=f"overcomplete-replication-{identity[:12]}-{Path(name).stem}",
                    )
                logger.finish()
                write_json(
                    out / "completed.json",
                    {
                        "identity": identity,
                        "hashes": {
                            name: sha(out / name)
                            for name in (
                                "protocol.json",
                                "initial_modules.pt",
                                "results.json",
                                "last.pt",
                                "validation.json",
                                "measurements.pt",
                                "final_test.json",
                                "final_test_measurements.pt",
                            )
                        },
                    },
                )
                return cast(dict[str, Any], read_json(out / "results.json"))
            except BaseException as error:
                write_json(
                    out / "failure.json", {"identity": identity, "step": step, "error": str(error)}
                )
                logger.log(step, {"diagnostic/failed": 1.0})
                logger.log_artifact(
                    out / "failure.json",
                    name=f"overcomplete-replication-{identity[:12]}-failure",
                )
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Launch one declared arm of the paired second-seed replication."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--training-source-root")
    parser.add_argument("--arm", choices=("gain6", "gain12"), default="gain12")
    for name, default in (
        ("seed", TRAINING_SEED),
        ("source-seed", SOURCE_SEED),
        ("evaluation-seed", SOURCE_SEED),
        ("steps", 100000),
        ("groups", 64),
        ("bootstrap-samples", 1000),
        ("threads", 1),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--verification", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    resume = bool(args.pop("resume"))
    run(Config(**args), resume=resume)


if __name__ == "__main__":
    main()
