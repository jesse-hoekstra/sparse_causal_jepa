"""Test one TRAIN-shared-feature initialization policy at two declared seeds."""

import argparse
import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.group_latent_control.run import (
    _restore_rng,  # pyright: ignore[reportPrivateUsage]
    _rng,  # pyright: ignore[reportPrivateUsage]
)
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.overcomplete_acquisition.evaluation import assess_model, measurement_metrics
from diagnostics.overcomplete_acquisition.forward import training_loss
from diagnostics.overcomplete_replication.source import SOURCE_SEED, TRAINING_SEED
from diagnostics.shared_feature_acquisition.source import load_inputs, verify_sources
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig


@dataclass(frozen=True)
class Config:
    """One fixed TRAIN-only projection policy at the two declared training seeds."""

    source_dir: str
    initial_source_dir: str
    output_dir: str
    training_source_root: str | None = None
    audit_projection: str | None = None
    source_seed: int = SOURCE_SEED
    seed: int = SOURCE_SEED
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
            (
                self.steps,
                self.groups,
                self.bootstrap_samples,
                self.source_seed,
                self.evaluation_seed,
            )
            != (100000, 64, 1000, SOURCE_SEED, SOURCE_SEED)
            or self.seed not in (SOURCE_SEED, TRAINING_SEED)
            or self.audit_projection is None
        ):
            raise ValueError(
                "Shared-feature control fixes100k,64groups,two seeds and audit reference"
            )
        if (
            self.threads < 1
            or self.bootstrap_samples < 2
            or min(self.seed, self.source_seed, self.evaluation_seed) < 0
        ):
            raise ValueError("Invalid threads, bootstrap count or seed")
        source, output = Path(self.source_dir).resolve(), Path(self.output_dir).resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Output must be separate from immutable source")
        initial = Path(self.initial_source_dir).resolve()
        if output == initial or output in initial.parents or initial in output.parents:
            raise ValueError("Output must be separate from immutable initial source")
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
    from diagnostics.overcomplete_replication.run import code_identity as mlp_code_identity

    digest = hashlib.sha256(mlp_code_identity().encode())
    digest.update((Path(__file__).parents[1] / "table_settling_control/source.py").read_bytes())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Train E/P without physical labels and evaluate only the fixed final test."""
    source = Path(config.source_dir)
    inputs = load_inputs(
        source,
        Path(config.initial_source_dir),
        config.seed,
        source_seed=config.source_seed,
        audit_projection=Path(config.audit_projection) if config.audit_projection else None,
        training_source_root=Path(config.training_source_root)
        if config.training_source_root
        else None,
        verification=config.verification,
    )
    bundle = inputs.bundle
    if not config.verification and (bundle.train_groups, bundle.batch_size) != (8192, 16):
        raise ValueError("Scientific acquisition requires original8192 groups and B16")
    output = Path(config.output_dir).resolve()
    roots = {Path(name).parent for name in inputs.hashes}
    if any(output == root or output in root.parents or root in output.parents for root in roots):
        raise ValueError("Output overlaps an immutable ancestor source directory")
    device, out = torch.device(config.device), Path(config.output_dir)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        with file_lock(out / ".lock"):
            model = inputs.model.requires_grad_(False)
            source_hashes = inputs.hashes
            initialization = inputs.provenance
            projection = inputs.projection
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
                "initial_source_dir",
                "audit_projection",
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
                "projection": projection.receipt,
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
                    "Random initial output projection attenuates shared system information. "
                    "A TRAIN-only shared-covariance projection into twelve coordinates improves "
                    "fresh all-factor acquisition across both declared seeds, without new labels."
                ),
                "controlled_change": (
                    "Restore actual authenticated saved INITIAL E/P from each completed q12 "
                    "parent, retain hidden/P/state tensors, replace only E output weight/bias. "
                    "Fit one fixed covariance recipe on all8192 TRAIN groups/R10; do not use "
                    "calibration/validation/test or theta in fitting. Fresh Adam.001 clip1, "
                    "same seed-specific global batches/update RNG and forecast-only100k."
                ),
                "code_dimension": 12,
                "physical_parameter_dimension": 6,
                "training_parameter_labels_absent": True,
                "production_training_modified": False,
                "optimized_loss_arm": "forecast_only",
                "projection_policy": projection.receipt["formula"],
                "numerical_audit": inputs.provenance["numerical_audit"],
                "numerical_audit_limits": (
                    "Canonical-sign float32weight/bias relative Frobeniusdifference<=1e-4; "
                    "first1024 TRAINcontext code RMSdifference<=1e-4*originalcodeRMS+1e-7. "
                    "Numerical identity only, not physical recovery/heldout model selection. "
                    "Persist actual server tensor hashes; CPUbackend roundoff may differ."
                ),
                "objective": (
                    "Unchanged attached two-step forecast MSE; original auxiliary losses "
                    "computed/logged with zero optimized weights. Original per-context forward; "
                    "group means are used only once to fit initialization, never at inference."
                ),
                "evaluation": (
                    "Frozen q12 evaluator, original evaluation seed, same probe grids; "
                    "calibration fit/validation selection/fixed100k test. Every1000monitor64; "
                    "final512/256/512retention plus64test responses. No teacher or new data."
                ),
                "decision": (
                    "Both seeds acquiring allsix supports this initializer for replication; "
                    "one seed only leaves robustness unresolved, neither rejects this fixed "
                    "projection remedy. Shared rank/SNR alone is not recovery; retention and "
                    "use remain distinct. No covariance/feature/dimension hyperparameter sweep."
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
                    out / "projection.pt",
                    {
                        "identity": identity,
                        "weight": projection.weight,
                        "bias": projection.bias,
                        "receipt": projection.receipt,
                    },
                )
                atomic_save(
                    out / "initial_modules.pt",
                    {
                        "identity": identity,
                        "projection": projection.receipt,
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
                or initial_payload["projection"] != projection.receipt
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
            persisted = cast(
                dict[str, Any],
                torch.load(out / "projection.pt", map_location="cpu", weights_only=False),
            )
            if (
                persisted["identity"] != identity
                or persisted["receipt"] != projection.receipt
                or not torch.equal(persisted["weight"], projection.weight)
                or not torch.equal(persisted["bias"], projection.bias)
            ):
                raise ValueError("Persisted TRAIN projection changed")
            projection_sha256 = sha(out / "projection.pt")
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
                if (
                    sha(out / "initial_modules.pt") != initial_sha256
                    or sha(out / "projection.pt") != projection_sha256
                ):
                    raise RuntimeError("Persisted MLP initialization artifact changed")
                if fixed_hashes(model) != frozen or model.get_extra_state() != flags:
                    raise RuntimeError("Frozen encoders or rollout settings changed")

            check_fixed()
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=f"shared-feature-acquisition-{out.parent.name}",
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
                verify_sources(inputs)
                result = {
                    "identity": identity,
                    "step": step,
                    "records": records,
                    "final_test": final,
                    "tracking": logger.metadata,
                    "scope": (
                        "TRAIN-shared-feature initialization diagnostic at two declared seeds, "
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
                        name=f"shared-feature-acquisition-{identity[:12]}-{Path(name).stem}",
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
                                "projection.pt",
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
                    name=f"shared-feature-acquisition-{identity[:12]}-failure",
                )
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Launch one declared seed of the TRAIN-shared-feature initialization control."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "initial-source-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--training-source-root")
    parser.add_argument("--audit-projection")
    for name, default in (
        ("seed", SOURCE_SEED),
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
