"""Distill fixed compressed six-vector targets, fit P, then wash out auxiliary targets."""

import argparse
import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from diagnostics.compressed_teacher_curriculum.evaluation import assess_model, measurement_metrics
from diagnostics.compressed_teacher_curriculum.source import (
    load_inputs,
    prepare_targets,
    verify_sources,
)
from diagnostics.group_latent_control.run import (
    _restore_rng,  # pyright: ignore[reportPrivateUsage]
    _rng,  # pyright: ignore[reportPrivateUsage]
)
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective, cached_forward
from diagnostics.mlp_acquisition_control.run import fixed_hashes
from diagnostics.mlp_oracle_code_control.model import initialize_mlp
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.multiteacher_curriculum.forward import infer_codes
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.models.model import SCJEPA, cross_episode_parameters
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig

Stage = Literal["distill", "predictor", "washout"]


def configure_phase(stage: Stage, model: SCJEPA) -> list[nn.Parameter]:
    """Train E only, P only, then E/P; no auxiliary heads are trainable."""
    if stage not in ("distill", "predictor", "washout"):
        raise ValueError("Unknown compressed-teacher curriculum phase")
    model.requires_grad_(False)
    model.parameter_encoder.requires_grad_(stage != "predictor")
    model.predictor.requires_grad_(stage != "distill")
    model.zero_grad(set_to_none=True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def phase_training_loss(
    stage: Stage,
    model: SCJEPA,
    batch: dict[str, Tensor],
    targets: Tensor | None,
    objective: AcquisitionObjective,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Use fixed six-vector targets directly, without any learned decoder gradient path."""
    if stage == "distill":
        codes = infer_codes(model, batch)
        if targets is None or targets.shape != codes.shape or targets.requires_grad:
            raise ValueError("Distillation requires detached BxRx6 compressed targets")
        if (
            targets.dtype != codes.dtype
            or targets.device != codes.device
            or not bool(torch.isfinite(targets).all())
        ):
            raise ValueError("Compressed targets must be finite and match student codes")
        cross_episode_parameters(codes, training=model.training)
        online = batch["online"]
        torch.randn(
            online.shape[-1], objective.projections, device=online.device, dtype=online.dtype
        )
        regularizer = objective.regularizer(codes.transpose(0, 1))
        consistency = 2 * codes.var(dim=1, correction=1).mean()
        distillation = (codes - targets).square().mean()
        total = distillation + 0.4 * regularizer + consistency
        return total, {
            "distillation": distillation,
            "parameter_regularizer": regularizer,
            "parameter_consistency": consistency,
            "optimized_total": total,
        }
    if stage not in ("predictor", "washout") or targets is not None:
        raise ValueError("Predictor fitting and washout accept no compressed targets")
    output = cached_forward(model, batch)
    full, terms = objective(output)
    terms["ordinary_total_without_constant_state_sigreg"] = full
    if stage == "predictor":
        if output.parameters.requires_grad:
            raise ValueError("Predictor fitting requires a frozen E")
        terms["optimized_total"] = terms["prediction"]
        return terms["prediction"], terms
    return full, terms


@dataclass(frozen=True)
class Config:
    """Fixed 30k/30k/10k exploratory curriculum with authenticated fixed teachers."""

    compression_source_dir: str
    output_dir: str
    seed: int = 202610261301
    device: str = "cpu"
    distill_steps: int = 30000
    predictor_steps: int = 30000
    washout_steps: int = 10000
    groups: int = 64
    bootstrap_samples: int = 1000
    threads: int = 1
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Keep teacher/output paths disjoint and reject undeclared scientific budgets."""
        budgets = (self.distill_steps, self.predictor_steps, self.washout_steps)
        if self.verification:
            if not all(1 <= value <= 3 for value in budgets) or not 2 <= self.groups <= 4:
                raise ValueError("Verification permits one to three updates per phase,2..4 groups")
        elif (budgets, self.seed, self.groups, self.bootstrap_samples) != (
            (30000, 30000, 10000),
            202610261301,
            64,
            1000,
        ):
            raise ValueError("Scientific curriculum fixes 30k/30k/10k, original seed and64groups")
        if self.threads < 1 or self.bootstrap_samples < 2 or self.seed < 0:
            raise ValueError("Invalid CPU threads, bootstrap count or seed")
        output = Path(self.output_dir).resolve()
        source = Path(self.compression_source_dir).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Output must be separate from immutable compression source")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)

    @property
    def steps(self) -> int:
        """Total fixed budget; every phase must run to its own endpoint."""
        return self.distill_steps + self.predictor_steps + self.washout_steps


def stage_at(step: int, config: Config) -> tuple[Stage, int]:
    """Map an upcoming global update to phase and zero-based phase offset."""
    if not 1 <= step <= config.steps:
        raise ValueError("Global update lies outside the fixed curriculum")
    if step <= config.distill_steps:
        return "distill", 0
    if step <= config.distill_steps + config.predictor_steps:
        return "predictor", config.distill_steps
    return "washout", config.distill_steps + config.predictor_steps


def code_identity() -> str:
    """Bind this curriculum and unchanged evaluator/MLP/acquisition dependencies."""
    from diagnostics.teacher_code_compression.run import code_identity as compression_code_identity

    digest = hashlib.sha256(compression_code_identity().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Train on teacher-derived targets only; physical labels remain evaluation-only."""
    inputs = load_inputs(Path(config.compression_source_dir), verification=config.verification)
    if inputs.bundle.seed != config.seed:
        raise ValueError("Student seed differs from authenticated parent source")
    output = Path(config.output_dir).resolve()
    source_roots = {Path(name).resolve().parent for name in inputs.hashes}
    source_roots.add(inputs.bundle.source_dir.resolve())
    if any(
        output == root or output in root.parents or root in output.parents for root in source_roots
    ):
        raise ValueError("Output overlaps an immutable source directory")
    bundle, out, device = inputs.bundle, Path(config.output_dir), torch.device(config.device)
    if not config.verification and (bundle.train_groups, bundle.batch_size) != (8192, 16):
        raise ValueError("Require original8192 TRAIN groups and B16")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        with file_lock(out / ".lock"):
            model = inputs.model.requires_grad_(False)
            mlp_initialization = initialize_mlp(model, config.seed)
            model = model.to(device)
            state_hashes, flags = fixed_hashes(model), model.get_extra_state()
            initial_encoder_hash = module_fingerprint(model.parameter_encoder)
            objective = AcquisitionObjective(
                ObjectiveConfig(**bundle.protocol["objective_config"])
            ).to(device)
            settings = asdict(config)
            for key in ("compression_source_dir", "output_dir", "wandb_mode", "wandb_entity"):
                settings.pop(key)
            evidence: dict[str, Any] = {
                "settings": settings,
                "source_hashes": inputs.hashes,
                "compression_provenance": inputs.provenance,
                "source_initialization": inputs.initialization,
                "initial_encoder_sha256": initial_encoder_hash,
                "mlp_initialization": mlp_initialization,
                "state_hashes": state_hashes,
                "code_identity": code_identity(),
                "torch": str(torch.__version__),
            }
            identity = canonical_fingerprint(evidence)
            protocol: dict[str, Any] = {
                "identity": identity,
                **evidence,
                "config": asdict(config),
                "hypothesis": (
                    "Direct fixed six-vector targets bypass learned-head/encoder coordination "
                    "and permit original E to acquire the compressor's all-six information."
                ),
                "selection_scope": (
                    "Exploratory artifact consolidation. Teachers selected retrospectively using "
                    "physical-parameter results including previously seen tests. Compressor passed "
                    "a secondary adaptive validation assay using fixed decoder features; its "
                    "original raw-code probe criterion did not pass. Not blinded discovery."
                ),
                "training_parameter_labels_absent": True,
                "algorithm": {
                    "targets": (
                        "Fixed C applied separately to each TRAIN context's normalized18D teacher "
                        "union, then average OTHER nine compressed6vectors. No new normalization."
                    ),
                    "distill": (
                        "30000 Eonly updates: direct6vector MSE + SIGReg.4 + consistency1. "
                        "P fixed; no learned heads."
                    ),
                    "predictor": (
                        "30000 Ponly forecastMSE updates, E fixed; constant auxiliary RNG retained."
                    ),
                    "washout": (
                        "10000 E/P forecastMSE+SIGReg.4+consistency1; "
                        "no targets or auxiliary heads."
                    ),
                    "optimizer": (
                        "Fresh Adam .001 each phase; globalnorm1; B16/R10/K10/T12. No MLP logits."
                    ),
                    "sampling": (
                        "Continuous original batch schedule; per-update seed*100000+global_step."
                    ),
                },
                "evaluation": (
                    "Original directq probes AND separately expanded family from fixed Cheads(q), "
                    "no extra context/input. Calibration fits and validation selects all families. "
                    "Final70000 only: full512/256/512 retention and64 testgroup responses. "
                    "Responses use actual studentq/P, never decoder-feature interventions."
                ),
                "decision": (
                    "Failure on fixed usable targets points to E amortization "
                    "or objective pressure. "
                    "Expanded-only recovery remains distinct from original probe criterion. "
                    "Success is conditional consolidation, not original SPARTAN graph recovery "
                    "or a blind discovery algorithm."
                ),
            }
            if resume:
                if read_json(out / "protocol.json")["identity"] != identity:
                    raise ValueError("Resume settings, source or code changed")
            elif any(path.name != ".lock" for path in out.iterdir()):
                raise FileExistsError("Use a fresh output directory or explicit resume")
            else:
                write_json(out / "protocol.json", protocol)
                atomic_save(
                    out / "initial_modules.pt",
                    {
                        "identity": identity,
                        "encoder": model.parameter_encoder.state_dict(),
                        "predictor": model.predictor.state_dict(),
                    },
                )
            initial = cast(
                dict[str, Any],
                torch.load(out / "initial_modules.pt", map_location="cpu", weights_only=False),
            )
            if initial["identity"] != identity:
                raise ValueError("Initial snapshot identity changed")
            for key, module, expected in (
                ("encoder", model.parameter_encoder, initial_encoder_hash),
                ("predictor", model.predictor, mlp_initialization["module_sha256"]),
            ):
                candidate = copy.deepcopy(module).cpu()
                candidate.load_state_dict(initial[key], strict=True)
                if module_fingerprint(candidate) != expected:
                    raise ValueError("Initial snapshot module changed")
            initial_sha256 = sha(out / "initial_modules.pt")
            if (out / "completed.json").exists():
                completed = cast(dict[str, Any], read_json(out / "completed.json"))
                if completed["identity"] != identity or any(
                    sha(out / name) != value for name, value in completed["hashes"].items()
                ):
                    raise ValueError("Completed curriculum artifacts changed")
                return cast(dict[str, Any], read_json(out / "results.json"))
            tracking = (
                read_json(out / "tracking.json") if (out / "tracking.json").exists() else None
            )
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=out.parent.name,
                    job_type="diagnostic",
                ),
                resolved_config=protocol,
                dataset_identity=bundle.protocol["identity"],
                out_dir=out,
                resume_state=tracking,
            )
            write_json(out / "tracking.json", logger.metadata)
            step = 0
            try:
                if (out / "target_manifest.json").exists():
                    manifest = read_json(out / "target_manifest.json")
                    if manifest["identity"] != identity or manifest["sha256"] != sha(
                        out / "teacher_targets.pt"
                    ):
                        raise ValueError("Teacher target cache changed")
                    payload = cast(
                        dict[str, Any],
                        torch.load(
                            out / "teacher_targets.pt", map_location="cpu", weights_only=False
                        ),
                    )
                    if payload["identity"] != identity:
                        raise ValueError("Teacher target identity changed")
                    targets = cast(Tensor, payload["targets"])
                else:
                    logger.log(0, {"preparation/targets_ready": 0.0})

                    def progress(done: int, total: int) -> None:
                        logger.log(
                            0,
                            {
                                "preparation/groups_encoded": float(done),
                                "preparation/fraction": done / total,
                            },
                        )

                    prepared = prepare_targets(inputs, device=device, progress=progress)
                    targets = prepared.targets
                    atomic_save(
                        out / "teacher_targets.pt",
                        {
                            "identity": identity,
                            "raw": prepared.raw,
                            "targets": targets,
                            "receipt": prepared.receipt,
                        },
                    )
                    manifest = {
                        "identity": identity,
                        "sha256": sha(out / "teacher_targets.pt"),
                        "receipt": prepared.receipt,
                    }
                    write_json(out / "target_manifest.json", manifest)
                if (
                    targets.shape != (bundle.train_groups, 10, 6)
                    or targets.requires_grad
                    or not bool(torch.isfinite(targets).all())
                ):
                    raise ValueError("Invalid detached TRAIN teacher targets")
                target_manifest_hash = sha(out / "target_manifest.json")
                logger.log(0, {"preparation/targets_ready": 1.0})
                stage: Stage = "distill"
                params = configure_phase(stage, model)
                optimizer = torch.optim.Adam(params, lr=0.001)
                frozen_phase_hash = module_fingerprint(model.predictor)
                records: list[dict[str, Any]] = []
                raw: dict[int, Any] = {}
                if resume and (out / "last.pt").exists():
                    checkpoint = cast(
                        dict[str, Any],
                        torch.load(out / "last.pt", map_location="cpu", weights_only=False),
                    )
                    if (
                        checkpoint["identity"] != identity
                        or checkpoint["target_manifest_sha256"] != target_manifest_hash
                    ):
                        raise ValueError("Checkpoint source or target cache changed")
                    model.load_state_dict(checkpoint["model"], strict=True)
                    stage = checkpoint["stage"]
                    if checkpoint["heads"] is not None:
                        raise ValueError("This curriculum must never contain learned heads")
                    params = configure_phase(stage, model)
                    optimizer = torch.optim.Adam(params, lr=0.001)
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    step, records, raw = (
                        checkpoint["step"],
                        checkpoint["records"],
                        checkpoint["raw"],
                    )
                    frozen_phase_hash = checkpoint["frozen_phase_hash"]
                    _restore_rng(checkpoint["rng"], device)
                if not 0 <= step <= config.steps or (step and stage != stage_at(step, config)[0]):
                    raise ValueError("Invalid checkpoint phase/update")
                selection = DiagnosticSelection(
                    tuple(range(config.groups)), tuple(range(config.groups))
                )

                def check_fixed() -> None:
                    if fixed_hashes(model) != state_hashes or model.get_extra_state() != flags:
                        raise RuntimeError("Frozen state maps or rollout flags changed")
                    if sha(out / "initial_modules.pt") != initial_sha256:
                        raise RuntimeError("Initial snapshot changed")
                    if (
                        stage == "distill"
                        and module_fingerprint(model.predictor) != frozen_phase_hash
                    ):
                        raise RuntimeError("Predictor changed during distillation")
                    if (
                        stage == "predictor"
                        and module_fingerprint(model.parameter_encoder) != frozen_phase_hash
                    ):
                        raise RuntimeError("Encoder changed during predictor fitting")

                def save() -> None:
                    check_fixed()
                    payload = {
                        "identity": identity,
                        "step": step,
                        "stage": stage,
                        "model": model.state_dict(),
                        "heads": None,
                        "optimizer": optimizer.state_dict(),
                        "rng": _rng(device),
                        "frozen_phase_hash": frozen_phase_hash,
                        "records": records,
                        "raw": raw,
                        "target_manifest_sha256": target_manifest_hash,
                    }
                    atomic_save(out / "last.pt", payload)
                    boundary = {
                        "distill": config.distill_steps,
                        "predictor": config.distill_steps + config.predictor_steps,
                        "washout": config.steps,
                    }[stage]
                    if step == boundary:
                        atomic_save(out / f"{stage}_final.pt", payload)

                def assess() -> None:
                    check_fixed()
                    report, measurements = assess_model(
                        model,
                        bundle.evaluation,
                        selection,
                        heads=inputs.heads,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                    )
                    records.append({"step": step, "stage": stage, **report})
                    raw[step] = measurements
                    logger.log(step, measurement_metrics(report))
                    write_json(out / "validation.json", {"identity": identity, "records": records})

                if not records:
                    assess()
                    save()
                loaded_epoch, batches = -1, []
                start = step
                for step in range(start + 1, config.steps + 1):
                    next_stage, _ = stage_at(step, config)
                    if next_stage != stage:
                        check_fixed()
                        stage = next_stage
                        params = configure_phase(stage, model)
                        optimizer = torch.optim.Adam(params, lr=0.001)
                        frozen_phase_hash = (
                            module_fingerprint(model.parameter_encoder)
                            if stage == "predictor"
                            else ""
                        )
                    epoch, position = divmod(step - 1, bundle.train_groups // bundle.batch_size)
                    if epoch != loaded_epoch:
                        batches = list(
                            GlobalBatchSampler(
                                bundle.train_groups, bundle.batch_size, config.seed, epoch
                            )
                        )
                        loaded_epoch = epoch
                    selected = batches[position]
                    batch = {
                        name: value[selected].to(device) for name, value in bundle.train.items()
                    }
                    teacher_targets = (
                        targets[selected].to(next(model.parameters()))
                        if stage == "distill"
                        else None
                    )
                    torch.manual_seed(config.seed * 100000 + step)  # pyright: ignore[reportUnknownMemberType]
                    model.train()
                    model.online_encoder.eval()
                    model.target_encoder.eval()
                    if stage == "predictor":
                        model.parameter_encoder.eval()
                    optimizer.zero_grad(set_to_none=True)
                    loss, terms = phase_training_loss(
                        stage, model, batch, teacher_targets, objective
                    )
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite curriculum loss")
                    loss.backward()  # pyright: ignore[reportUnknownMemberType]
                    norm = float(
                        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                    )
                    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
                    if step == 1 or step % 100 == 0 or config.verification:
                        logger.log(
                            step,
                            {
                                **{
                                    f"train/{key}": float(value.detach())
                                    for key, value in terms.items()
                                },
                                "train/gradient_norm": norm,
                                "train/phase": float(
                                    ("distill", "predictor", "washout").index(stage)
                                ),
                            },
                        )
                    if (
                        config.verification
                        or step % 1000 == 0
                        or step
                        in (
                            config.distill_steps,
                            config.distill_steps + config.predictor_steps,
                            config.steps,
                        )
                    ):
                        assess()
                        save()
                if (out / "final_test.json").exists():
                    final_record = cast(dict[str, Any], read_json(out / "final_test.json"))
                    if (
                        final_record["identity"] != identity
                        or final_record["checkpoint_sha256"] != sha(out / "last.pt")
                        or final_record["raw_sha256"] != sha(out / "final_test_measurements.pt")
                    ):
                        raise ValueError("Final test differs from fixed-budget endpoint")
                    final = final_record["metrics"]
                else:
                    final, final_raw = assess_model(
                        model,
                        bundle.evaluation,
                        selection,
                        heads=inputs.heads,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
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
                if (
                    sha(out / "target_manifest.json") != target_manifest_hash
                    or sha(out / "teacher_targets.pt") != manifest["sha256"]
                ):
                    raise RuntimeError("Teacher targets changed during curriculum")
                result = {
                    "identity": identity,
                    "step": step,
                    "records": records,
                    "final_test": final,
                    "tracking": logger.metadata,
                    "scope": protocol["selection_scope"],
                }
                write_json(out / "results.json", result)
                logger.log(step, measurement_metrics(final, final=True))
                logger.log(step, {"diagnostic/complete": 1.0})
                logger.log_artifact(
                    out / "results.json", name=f"compressed-teacher-{identity[:12]}-results"
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
                                "target_manifest.json",
                                "teacher_targets.pt",
                                "last.pt",
                                "distill_final.pt",
                                "predictor_final.pt",
                                "washout_final.pt",
                                "validation.json",
                                "final_test.json",
                                "final_test_measurements.pt",
                                "results.json",
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
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    """Launch only the exploratory fixed compressed-target curriculum."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compression-source-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cpu")
    for name, value in (
        ("seed", 202610261301),
        ("distill-steps", 30000),
        ("predictor-steps", 30000),
        ("washout-steps", 10000),
        ("groups", 64),
        ("bootstrap-samples", 1000),
        ("threads", 1),
    ):
        parser.add_argument("--" + name, type=int, default=value)
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
