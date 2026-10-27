"""Compare full and forecast-only bounded continuations from the same fixed70k student."""

import argparse
import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from diagnostics.compressed_teacher_continuation.source import load_inputs, verify_sources
from diagnostics.compressed_teacher_curriculum.evaluation import assess_model, measurement_metrics
from diagnostics.group_latent_control.run import (
    _restore_rng,  # pyright: ignore[reportPrivateUsage]
    _rng,  # pyright: ignore[reportPrivateUsage]
)
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.mlp_acquisition_control.forward import training_loss as ordinary_training_loss
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.run import atomic_save, sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig

ObjectiveArm = Literal["original", "forecast_only"]


def training_loss(
    model: SCJEPA,
    batch: dict[str, Tensor],
    objective: AcquisitionObjective,
    objective_arm: ObjectiveArm,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Preserve original loss arithmetic/RNG; select only the optimized tensor."""
    if objective_arm not in ("original", "forecast_only"):
        raise ValueError("Unknown continuation objective arm")
    ordinary_total, terms = ordinary_training_loss(model, batch, objective)
    loss = ordinary_total if objective_arm == "original" else terms["prediction"]
    terms["ordinary_total_without_constant_state_sigreg"] = ordinary_total
    terms["optimized_total"] = loss
    terms["effective_parameter_regularizer_weight"] = loss.new_tensor(
        0.4 if objective_arm == "original" else 0.0
    )
    terms["effective_parameter_consistency_weight"] = loss.new_tensor(
        1.0 if objective_arm == "original" else 0.0
    )
    return loss, terms


@dataclass(frozen=True)
class Config:
    """A fixed paired continuation arm with immutable endpoint ancestry."""

    source_dir: str
    output_dir: str
    objective_arm: ObjectiveArm = "original"
    seed: int = 202610261301
    device: str = "cpu"
    steps: int = 30000
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
        elif (self.steps, self.groups, self.bootstrap_samples, self.seed) != (
            30000,
            64,
            1000,
            202610261301,
        ):
            raise ValueError(
                "Continuation fixes 30k additional updates, 64 groups and original seed"
            )
        if self.objective_arm not in ("original", "forecast_only"):
            raise ValueError("Choose original or forecast_only")
        if self.threads < 1 or self.bootstrap_samples < 2 or self.seed < 0:
            raise ValueError("Invalid threads, bootstrap count or seed")
        source, output = Path(self.source_dir).resolve(), Path(self.output_dir).resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Output must be separate from immutable source")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def fixed_hashes(model: SCJEPA) -> dict[str, str]:
    """Only the state maps remain fixed; E and P both train."""
    return {
        name: module_fingerprint(getattr(model, name))
        for name in ("online_encoder", "target_encoder")
    }


def code_identity() -> str:
    """Bind this independent runner and every unchanged acquisition dependency."""
    from diagnostics.compressed_teacher_curriculum.run import code_identity as parent_code_identity

    digest = hashlib.sha256(parent_code_identity().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Train E/P without physical labels and evaluate only the fixed final test."""
    inputs = load_inputs(Path(config.source_dir), verification=config.verification)
    bundle = inputs.bundle
    source_step = int(inputs.provenance["source_step"])
    if bundle.seed != config.seed:
        raise ValueError("Continuation seed differs from source")
    if not config.verification and (source_step, bundle.train_groups, bundle.batch_size) != (
        70000,
        8192,
        16,
    ):
        raise ValueError("Require fixed70k source and original8192 groups/B16")
    output = Path(config.output_dir).resolve()
    source_roots = {Path(name).resolve().parent for name in inputs.hashes}
    source_roots.add(bundle.source_dir.resolve())
    if any(
        output == root or output in root.parents or root in output.parents for root in source_roots
    ):
        raise ValueError("Output overlaps an immutable ancestor source directory")
    device, out = torch.device(config.device), Path(config.output_dir)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        with file_lock(out / ".lock"):
            model = inputs.model.to(device).requires_grad_(False)
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
            fixed_heads_hash = module_fingerprint(inputs.heads)
            settings = asdict(config)
            for name in (
                "source_dir",
                "output_dir",
                "wandb_mode",
                "wandb_entity",
            ):
                settings.pop(name)
            evidence: dict[str, Any] = {
                "settings": settings,
                "source_hashes": inputs.hashes,
                "source_provenance": inputs.provenance,
                "initial_encoder_sha256": initial_encoder_hash,
                "initial_predictor_sha256": initial_predictor_hash,
                "fixed_heads_sha256": fixed_heads_hash,
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
                    "From the same near-all-six70k endpoint, removing combined auxiliary-loss "
                    "pressure improves retained information and predictive use beyond equal "
                    "additional optimization with the ordinary objective."
                ),
                "controlled_change": (
                    "Paired original MSE+.4SIGReg+consistency1 versus forecast-only MSE. "
                    "Both restore identical completed70k E/P, reset Adam .001 clip1, keep states "
                    "fixed, and continue the original global batch/RNG schedule70001..100000. "
                    "No teacher targets or trainable heads. Auxiliary values/draws retained."
                ),
                "training_parameter_labels_absent": True,
                "objective_arm": config.objective_arm,
                "optimized_loss_arm": config.objective_arm,
                "production_training_modified": False,
                "optimizer_history": "Fresh Adam for BOTH arms; do not restore source moments",
                "source_step": source_step,
                "final_global_step": source_step + config.steps,
                "objective": (
                    "original: mean two-step MSE+.4parameterSIGReg+consistency1; "
                    "forecast_only: return original prediction MSE tensor directly. "
                    "Frozen-state SIGReg constant omitted with projection draw preserved; "
                    "MLP has no attention logits in either arm."
                ),
                "evaluation": (
                    "Every1000additional updates: primary64cal/val direct and fixed-head-expanded "
                    "retention plus matched responses. Only fixed100k endpoint: "
                    "full512/256/512cal/val/test retention and64test responses. "
                    "Fixed heads use only studentq; response uses actualq/P."
                ),
                "selection_scope": (
                    "Adaptive repair of known failed70k endpoint, selected after parameter/test "
                    "results. Original70k failure remains unchanged; not independent replication. "
                    "Upstream teachers and compressor were retrospectively selected."
                ),
                "decision": (
                    "Forecast-only improvement beyond ordinary continuation supports combined "
                    "auxiliary maintenance pressure. Comparable improvement supports extra "
                    "optimization time; neither improving leaves this bounded repair unresolved. "
                    "Retention and predictive use remain distinct. No individual auxiliary-term "
                    "attribution, early stopping, test selection or original SPARTAN success claim."
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
                        "source_step": source_step,
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
                or initial_payload["source_step"] != source_step
            ):
                raise ValueError("Persisted continuation initialization identity changed")
            initial_copy = copy.deepcopy(model.predictor).cpu()
            initial_copy.load_state_dict(initial_payload["predictor"], strict=True)
            if module_fingerprint(initial_copy) != initial_predictor_hash:
                raise ValueError("Persisted continuation initialization weights changed")
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
                start, records, raw = (
                    checkpoint["updates"],
                    checkpoint["records"],
                    checkpoint["raw"],
                )
                if checkpoint["step"] != source_step + start:
                    raise ValueError("Checkpoint global step differs from source plus updates")
                tracking = checkpoint["tracking"]
                _restore_rng(checkpoint["rng"], device)
            if not 0 <= start <= config.steps:
                raise ValueError("Invalid checkpoint update")

            def check_fixed() -> None:
                if module_fingerprint(inputs.heads) != fixed_heads_hash:
                    raise RuntimeError("Fixed evaluation heads changed")
                if sha(out / "initial_modules.pt") != initial_sha256:
                    raise RuntimeError("Persisted continuation initialization artifact changed")
                if fixed_hashes(model) != frozen or model.get_extra_state() != flags:
                    raise RuntimeError("Frozen encoders or rollout settings changed")

            check_fixed()
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

            def save(step: int) -> None:
                check_fixed()
                atomic_save(
                    out / "last.pt",
                    {
                        "identity": identity,
                        "step": source_step + step,
                        "updates": step,
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
                    seed=config.seed,
                    heads=inputs.heads,
                )
                raw[source_step + step] = measurements
                records.append({"step": source_step + step, "updates": step, **report})
                logger.log(source_step + step, measurement_metrics(report))
                write_json(out / "validation.json", {"identity": identity, "records": records})

            step = start
            try:
                if not records:
                    validate(0)
                    save(0)
                loaded_epoch, batches = -1, []
                for step in range(start + 1, config.steps + 1):
                    global_step = source_step + step
                    epoch, position = divmod(
                        global_step - 1, bundle.train_groups // bundle.batch_size
                    )
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
                    torch.manual_seed(config.seed * 100000 + global_step)  # pyright: ignore[reportUnknownMemberType]
                    model.train()
                    model.online_encoder.eval()
                    model.target_encoder.eval()
                    optimizer.zero_grad(set_to_none=True)
                    loss, terms = training_loss(model, batch, objective, config.objective_arm)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"Nonfinite diagnostic loss at update{step}")
                    loss.backward()  # pyright: ignore[reportUnknownMemberType]
                    norm = float(
                        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    )
                    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
                    if step == 1 or step % 100 == 0 or step == config.steps:
                        logger.log(
                            global_step,
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
                        seed=config.seed,
                        final=True,
                        heads=inputs.heads,
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
                    "step": source_step + step,
                    "updates": step,
                    "records": records,
                    "final_test": final,
                    "tracking": logger.metadata,
                    "scope": (
                        "Adaptive paired compressed-student continuation, "
                        "not original SPARTAN experiment success"
                    ),
                }
                write_json(out / "results.json", result)
                atomic_save(
                    out / "measurements.pt",
                    {"identity": identity, "validation": raw, "test": final_raw},
                )
                logger.log(source_step + step, measurement_metrics(final, final=True))
                logger.log(source_step + step, {"diagnostic/complete": 1.0})
                for name in ("results.json", "protocol.json"):
                    logger.log_artifact(
                        out / name,
                        name=f"compressed-teacher-continuation-{identity[:12]}-{Path(name).stem}",
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
                    out / "failure.json",
                    {
                        "identity": identity,
                        "step": source_step + step,
                        "updates": step,
                        "error": str(error),
                    },
                )
                logger.log(source_step + step, {"diagnostic/failed": 1.0})
                logger.log_artifact(
                    out / "failure.json",
                    name=f"compressed-teacher-continuation-{identity[:12]}-failure",
                )
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Launch one matched bounded continuation arm from the fixed70k source."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--objective-arm", choices=("original", "forecast_only"), default="original"
    )
    for name, default in (
        ("seed", 202610261301),
        ("steps", 30000),
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
