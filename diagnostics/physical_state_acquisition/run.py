"""Fresh label-free E/P acquisition with explicitly privileged physical state coordinates."""

import argparse
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.group_latent_control.run import (
    _restore_rng,  # pyright: ignore[reportPrivateUsage]
    _rng,  # pyright: ignore[reportPrivateUsage]
    initialize_control,
)
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.multiquery_control.data import TensorSplit, module_fingerprint
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.physical_state_acquisition.data import (
    PhysicalIdentityEncoder,
    build_physical_cache,
    source_data_config,
)
from diagnostics.physical_state_acquisition.evaluation import assess_model, measurement_metrics
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective
from diagnostics.predictive_supervision_control.run import code_identity as shared_code_identity
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng


@dataclass(frozen=True)
class Config:
    """One predeclared physical-coordinate control, with isolated source/output paths."""

    source_dir: str
    output_dir: str
    training_source_root: str | None = None
    source_run_config: str | None = None
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
            raise ValueError("Physical diagnostic fixes 30k updates, 64 groups and original seed")
        if self.threads < 1 or self.bootstrap_samples < 2 or self.seed < 0:
            raise ValueError("Invalid threads, bootstrap count or seed")
        source, output = Path(self.source_dir).resolve(), Path(self.output_dir).resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Output must be separate from immutable source")
        if self.training_source_root:
            archive = Path(self.training_source_root).resolve()
            if output == archive or output in archive.parents or archive in output.parents:
                raise ValueError("Output must be separate from the archived training source")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def training_loss(
    model: SCJEPA, batch: TensorSplit, objective: PredictiveSupervisionObjective
) -> tuple[Tensor, dict[str, Tensor]]:
    """Use the original attached cross-context objective with no true-parameter inputs."""
    if set(batch) != {"online", "target", "actions"}:
        raise ValueError("Physical acquisition training receives no parameter labels")
    if not isinstance(model.online_encoder, PhysicalIdentityEncoder) or not isinstance(
        model.target_encoder, PhysicalIdentityEncoder
    ):
        raise ValueError("Training requires explicit physical identity state encoders")
    return objective(cached_multiquery_forward(model, batch, queries=1))


def code_identity() -> str:
    """Bind current production code and every reused diagnostic implementation."""
    digest = hashlib.sha256(shared_code_identity().encode())
    root = Path(__file__).parents[1]
    for folder in (
        Path(__file__).parent,
        root / "group_latent_control",
        root / "joint_acquisition",
        root / "encoder_gradient_control",
    ):
        for path in sorted(folder.glob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Run one fixed-budget diagnostic; assess test only at the final checkpoint."""
    source, out = Path(config.source_dir), Path(config.output_dir)
    device = torch.device(config.device)
    bundle = load_primary_source(
        source,
        config.seed,
        expected_step=2 if config.verification else 30000,
        **({"expected_checkpoint_sha256": None} if config.verification else {}),
        training_source_root=Path(config.training_source_root)
        if config.training_source_root
        else None,
    )
    if not config.verification and (bundle.train_groups, bundle.batch_size) != (8192, 16):
        raise ValueError("Scientific physical control requires 8192 TRAIN groups and B16")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        with file_lock(out / ".lock"), preserve_rng(device):
            model, source_optimizer = build_source_model(bundle, device)
            del source_optimizer
            discarded_table, initialization, source_hashes = initialize_control(
                model, bundle, "fresh_zero"
            )
            del discarded_table
            data_config, data_record = source_data_config(
                bundle, Path(config.source_run_config) if config.source_run_config else None
            )
            settings = asdict(config)
            for name in ("output_dir", "wandb_mode", "wandb_entity"):
                settings.pop(name)
            evidence: dict[str, Any] = {
                "settings": settings,
                "source_hashes": source_hashes,
                "source_data_record": data_record,
                "initialization": initialization,
                "code_identity": code_identity(),
                "torch": str(torch.__version__),
                "state_coordinates": "physical_identity",
            }
            identity = canonical_fingerprint(evidence)
            protocol: dict[str, Any] = {
                "identity": identity,
                **evidence,
                "config": asdict(config),
                "hypothesis": (
                    "The learned-state coordinate pipeline obstructs parameter acquisition."
                ),
                "controlled_change": (
                    "Replace only frozen online/target state coordinates by raw physical states; "
                    "same authenticated fresh parameter encoder/predictor initialization, "
                    "mixed observed actions, DGP groups, original objective and 30k joint updates."
                ),
                "decision": (
                    "All-six recovery/use supports a learned-state pipeline limitation. Failure "
                    "despite a successful true-parameter predictor control implicates inference "
                    "or coupled learning even with privileged state coordinates."
                ),
                "scope": (
                    "Privileged diagnostic only, not production acquisition or state recovery. "
                    "State geometry, information and online/target chart mismatch change together."
                ),
                "training_parameter_labels_absent": True,
                "action_unmixing_in_model": False,
                "objective": bundle.protocol["objective_config"],
                "optimizer": {
                    "kind": "Adam",
                    "lr": 0.001,
                    "weight_decay": 0.0,
                    "clip_norm": 1.0,
                    "fresh": True,
                },
                "evaluation": (
                    "Validation every 1000; final fixed-budget test once; individual-context "
                    "whole-code probes on all 512/256/512 groups at final evaluation; "
                    "matched physical .01 two-step responses on first 64 assessment groups"
                ),
            }
            if resume:
                if read_json(out / "protocol.json")["identity"] != identity:
                    raise ValueError("Resume source, settings or code changed")
            elif any(path.name != ".lock" for path in out.iterdir()):
                raise FileExistsError("Use a fresh output directory or explicit resume")
            else:
                write_json(out / "protocol.json", protocol)
            if (out / "completed.json").exists():
                complete = cast(dict[str, Any], read_json(out / "completed.json"))
                if complete["identity"] != identity or any(
                    sha(out / name) != value for name, value in complete["hashes"].items()
                ):
                    raise ValueError("Completed physical diagnostic artifacts changed")
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
                manifest_path = out / "cache_manifest.json"
                manifest: dict[str, Any]
                if manifest_path.exists():
                    manifest = cast(dict[str, Any], read_json(manifest_path))
                    if manifest["identity"] != identity or any(
                        sha(out / name) != value for name, value in manifest["hashes"].items()
                    ):
                        raise ValueError("Physical diagnostic cache changed")
                    train = cast(
                        TensorSplit,
                        torch.load(
                            out / "physical_train_inputs.pt", map_location="cpu", weights_only=False
                        ),
                    )
                    evaluation = torch.load(
                        out / "physical_evaluation_data.pt", map_location="cpu", weights_only=False
                    )
                else:
                    logger.log(
                        0, {"preparation/cache_ready": 0.0, "preparation/groups_verified": 0.0}
                    )

                    def progress(done: int, total: int) -> None:
                        if done % 256 == 0 or done == total:
                            logger.log(
                                0,
                                {
                                    "preparation/groups_verified": float(done),
                                    "preparation/fraction": done / total,
                                },
                            )

                    train, evaluation, receipt = build_physical_cache(
                        model, bundle.train, bundle.evaluation, data_config, progress=progress
                    )
                    atomic_save(out / "physical_train_inputs.pt", train)
                    atomic_save(out / "physical_evaluation_data.pt", evaluation)
                    write_json(out / "cache_receipt.json", {"identity": identity, **receipt})
                    manifest = {
                        "identity": identity,
                        "hashes": {
                            name: sha(out / name)
                            for name in (
                                "physical_train_inputs.pt",
                                "physical_evaluation_data.pt",
                                "cache_receipt.json",
                            )
                        },
                    }
                    write_json(manifest_path, manifest)
                if set(train) != {"online", "target", "actions"}:
                    raise ValueError("Physical training cache includes forbidden fields")
                logger.log(0, {"preparation/cache_ready": 1.0})
                reference = next(model.parameters())
                model.online_encoder = PhysicalIdentityEncoder().to(reference).requires_grad_(False)
                model.target_encoder = PhysicalIdentityEncoder().to(reference).requires_grad_(False)
                model.parameter_encoder.requires_grad_(True)
                model.predictor.requires_grad_(True)
                state_hashes = {
                    name: module_fingerprint(getattr(model, name + "_encoder"))
                    for name in ("online", "target")
                }
                for name in state_hashes:
                    if state_hashes[name] != evaluation["metadata"][f"state_{name}_sha256"]:
                        raise ValueError("Physical identity encoder differs from cache metadata")
                parameters = [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ]
                optimizer = torch.optim.Adam(parameters, lr=0.001)
                objective = PredictiveSupervisionObjective(
                    ObjectiveConfig(**bundle.protocol["objective_config"]), 1.0
                ).to(device)
                selection = DiagnosticSelection(
                    tuple(range(config.groups)), tuple(range(config.groups))
                )
                records: list[dict[str, Any]] = []
                raw: dict[int, dict[str, Tensor]] = {}
                if resume and (out / "last.pt").exists():
                    checkpoint = cast(
                        dict[str, Any],
                        torch.load(out / "last.pt", map_location="cpu", weights_only=False),
                    )
                    if checkpoint["identity"] != identity or checkpoint[
                        "cache_manifest_sha256"
                    ] != sha(manifest_path):
                        raise ValueError("Physical checkpoint source/cache mismatch")
                    model.load_state_dict(checkpoint["model"], strict=True)
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    step, records, raw = (
                        checkpoint["step"],
                        checkpoint["records"],
                        checkpoint["raw"],
                    )
                    _restore_rng(checkpoint["rng"], device)
                if not 0 <= step <= config.steps:
                    raise ValueError("Invalid physical acquisition checkpoint step")

                def save() -> None:
                    if any(
                        module_fingerprint(getattr(model, name + "_encoder")) != value
                        for name, value in state_hashes.items()
                    ):
                        raise RuntimeError("Physical identity encoders changed")
                    atomic_save(
                        out / "last.pt",
                        {
                            "identity": identity,
                            "step": step,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "records": records,
                            "raw": raw,
                            "rng": _rng(device),
                            "cache_manifest_sha256": sha(manifest_path),
                        },
                    )

                def assess() -> None:
                    report, measurements = assess_model(
                        model,
                        evaluation,
                        selection,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                    )
                    records.append({"step": step, **report})
                    raw[step] = measurements
                    logger.log(step, measurement_metrics(report))
                    write_json(
                        out / "measurements.json", {"identity": identity, "records": records}
                    )

                if not records:
                    assess()
                    save()
                batches_per_epoch = bundle.train_groups // bundle.batch_size
                epoch_cached, batches = -1, []
                start = step
                for step in range(start + 1, config.steps + 1):
                    epoch, position = divmod(step - 1, batches_per_epoch)
                    if epoch != epoch_cached:
                        batches = list(
                            GlobalBatchSampler(
                                bundle.train_groups, bundle.batch_size, config.seed, epoch
                            )
                        )
                        epoch_cached = epoch
                    batch = {
                        key: values[batches[position]].to(device) for key, values in train.items()
                    }
                    torch.manual_seed(config.seed * 100000 + step)  # pyright: ignore[reportUnknownMemberType]
                    model.train()
                    model.online_encoder.eval()
                    model.target_encoder.eval()
                    optimizer.zero_grad(set_to_none=True)
                    loss, terms = training_loss(model, batch, objective)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite physical-state acquisition loss")
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
                                    f"train/{key}": float(value.detach())
                                    for key, value in terms.items()
                                },
                                "train/gradient_norm": norm,
                            },
                        )
                    if config.verification or step % 1000 == 0 or step == config.steps:
                        assess()
                        save()
                final: dict[str, Any]
                if (out / "final_test.json").exists():
                    final = cast(dict[str, Any], read_json(out / "final_test.json"))
                    if (
                        final["identity"] != identity
                        or final["checkpoint_sha256"] != sha(out / "last.pt")
                        or final["raw_sha256"] != sha(out / "final_test_measurements.pt")
                    ):
                        raise ValueError(
                            "Final-test record differs from final fixed-budget checkpoint"
                        )
                else:
                    report, final_raw = assess_model(
                        model,
                        evaluation,
                        selection,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                        final=True,
                    )
                    atomic_save(out / "final_test_measurements.pt", final_raw)
                    final = {
                        "identity": identity,
                        "checkpoint_sha256": sha(out / "last.pt"),
                        "raw_sha256": sha(out / "final_test_measurements.pt"),
                        "metrics": report,
                    }
                    write_json(out / "final_test.json", final)
                logger.log(step, measurement_metrics(final["metrics"], final=True))
                for name, expected in source_hashes.items():
                    if sha(source / name) != expected:
                        raise RuntimeError("Immutable source changed during physical diagnostic")
                if sha(Path(data_record["path"])) != data_record["sha256"]:
                    raise RuntimeError("Source run configuration changed")
                if any(sha(out / name) != value for name, value in manifest["hashes"].items()):
                    raise RuntimeError("Physical caches changed during training")
                result = {
                    "identity": identity,
                    "step": step,
                    "records": records,
                    "final_test": final["metrics"],
                    "tracking": logger.metadata,
                    "scope": protocol["scope"],
                }
                write_json(out / "results.json", result)
                atomic_save(out / "measurements.pt", {"identity": identity, "raw": raw})
                logger.log_artifact(
                    out / "results.json", name=f"physical-acquisition-{identity[:12]}-results"
                )
                logger.log(step, {"diagnostic/complete": 1.0})
                logger.finish()
                write_json(
                    out / "completed.json",
                    {
                        "identity": identity,
                        "hashes": {
                            name: sha(out / name)
                            for name in (
                                "protocol.json",
                                "cache_manifest.json",
                                "cache_receipt.json",
                                "physical_train_inputs.pt",
                                "physical_evaluation_data.pt",
                                "results.json",
                                "last.pt",
                                "final_test.json",
                                "final_test_measurements.pt",
                                "measurements.pt",
                            )
                        },
                    },
                )
                return result
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
    """Launch the isolated physical-state diagnostic with explicit source paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-source-root")
    parser.add_argument("--source-run-config")
    parser.add_argument("--device", default="cpu")
    for name, value in (
        ("seed", 202610261301),
        ("steps", 30000),
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
