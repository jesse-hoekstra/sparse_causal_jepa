"""Fit a fixed six-dimensional teacher-code bottleneck; validation only, no forecasting."""

import argparse
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.group_latent_control.run import (
    _restore_rng,  # pyright: ignore[reportPrivateUsage]
    _rng,  # pyright: ignore[reportPrivateUsage]
)
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.multiteacher_curriculum.run import code_identity as curriculum_code_identity
from diagnostics.teacher_code_compression.data import load_reference, verify_reference
from diagnostics.teacher_code_compression.evaluation import assess, metrics, prepare_evaluation
from diagnostics.teacher_code_compression.model import initialize_modules, training_loss
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig


@dataclass(frozen=True)
class Config:
    """Fixed-budget exploratory compression with the selected immutable curriculum input."""

    curriculum_dir: str
    output_dir: str
    seed: int = 202610261301
    device: str = "cpu"
    steps: int = 30000
    groups: int = 64
    threads: int = 1
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Reject budget changes and writes into the immutable parent curriculum."""
        if self.verification:
            if not 1 <= self.steps <= 3 or not 2 <= self.groups <= 4:
                raise ValueError("Verification permits one to three updates and2..4 groups")
        elif (self.seed, self.steps, self.groups) != (202610261301, 30000, 64):
            raise ValueError("Scientific compression fixes original seed,30k updates,64groups")
        if self.seed < 0 or self.threads < 1:
            raise ValueError("Invalid seed or CPU thread count")
        _separate(Path(self.output_dir), Path(self.curriculum_dir))
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def _separate(output: Path, source: Path) -> None:
    output, source = output.resolve(), source.resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("Output must be separate from every immutable source")


def code_identity() -> str:
    """Bind this independent implementation and every unchanged curriculum dependency."""
    digest = hashlib.sha256(curriculum_code_identity().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Train only compression encoder/heads; retain the final validation-only endpoint."""
    inputs = load_reference(Path(config.curriculum_dir), verification=config.verification)
    bundle, out, device = inputs.parent.bundle, Path(config.output_dir), torch.device(config.device)
    for source in [
        bundle.source_dir,
        *[
            Path(record["checkpoint"]).parent
            for record in inputs.parent.teacher_provenance["teachers"]
        ],
    ]:
        _separate(out, source)
    if any(
        inputs.reference[key] != getattr(config, key)
        for key in ("seed", "device", "threads", "groups")
    ):
        raise ValueError("Compression seed/device/threads/groups must match parent distillation")
    if not config.verification and (bundle.train_groups, bundle.batch_size) != (8192, 16):
        raise ValueError("Require original8192 TRAIN groups and B16")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        with file_lock(out / ".lock"):
            encoder, heads, initialization = initialize_modules(
                config.seed, dtype=inputs.train.dtype
            )
            heads.load_state_dict(inputs.heads_state, strict=True)
            if module_fingerprint(heads) != inputs.reference["exact_initial_heads_sha256"]:
                raise ValueError("Initial heads differ from the matched curriculum")
            encoder, heads = encoder.to(device), heads.to(device)
            optimizer = torch.optim.Adam([*encoder.parameters(), *heads.parameters()], lr=0.001)
            objective = AcquisitionObjective(
                ObjectiveConfig(**bundle.protocol["objective_config"])
            ).to(device)
            settings = asdict(config)
            for name in ("curriculum_dir", "output_dir", "wandb_mode", "wandb_entity"):
                settings.pop(name)
            evidence = {
                "settings": settings,
                "reference": inputs.reference,
                "source_hashes": inputs.parent.hashes,
                "reference_hashes": inputs.hashes,
                "initialization": initialization,
                "code_identity": code_identity(),
                "torch": str(torch.__version__),
            }
            identity = canonical_fingerprint(evidence)
            protocol: dict[str, Any] = {
                "identity": identity,
                **evidence,
                "config": asdict(config),
                "hypothesis": (
                    "The same regularized teacher reconstruction task can retain all six when "
                    "temporal inference is bypassed with the same context's teacher-code union."
                ),
                "controlled_change": (
                    "Replace the temporal student encoder input/network by a fresh18-128-128-6 "
                    "GELU network over that individual context's TRAIN-normalized teacher codes. "
                    "Exact initial heads, fixed targets, losses, sampler, seed, optimizer and "
                    "budget match distillation. This is an encoder-input/architecture intervention."
                ),
                "objective": "Mean teacher MSE + .4 parameter SIGReg + consistency1; no predictor",
                "optimizer": "Fresh Adam .001, clip global encoder+heads gradient norm1",
                "rng": "Original donor-pair and state-projection draws; seed*100000+update",
                "training_parameter_labels_absent": True,
                "evaluation": (
                    "Every1000: first64 calibration/validation groups, ten separate contexts. "
                    "Fixed30000 additionally full512 calibration/256 validation. Existing "
                    "linear/cubic/RFF probes fit calibration and select/score validation. "
                    "No test data evaluated, no prediction/use criterion claimed."
                ),
                "decision": (
                    "All-six compression supports a temporal amortization/optimization barrier "
                    "in the original student contrast. Failure leaves this bottleneck, heads, "
                    "target geometry and objective optimization unresolved; it does not rule "
                    "out global nonlinear6D compression or original encoder capacity."
                ),
                "selection_scope": inputs.parent.teacher_provenance["selection_scope"],
            }
            if resume:
                if read_json(out / "protocol.json")["identity"] != identity:
                    raise ValueError("Resume code, settings or immutable source changed")
            elif any(path.name != ".lock" for path in out.iterdir()):
                raise FileExistsError("Use a fresh output directory or explicit resume")
            else:
                write_json(out / "protocol.json", protocol)
                atomic_save(
                    out / "initial_modules.pt",
                    {
                        "identity": identity,
                        "encoder": encoder.state_dict(),
                        "heads": heads.state_dict(),
                    },
                )
            initial = cast(
                dict[str, Any],
                torch.load(out / "initial_modules.pt", map_location="cpu", weights_only=False),
            )
            check_encoder, check_heads, _ = initialize_modules(
                config.seed, dtype=inputs.train.dtype
            )
            check_encoder.load_state_dict(initial["encoder"], strict=True)
            check_heads.load_state_dict(initial["heads"], strict=True)
            if (
                initial["identity"] != identity
                or module_fingerprint(check_encoder) != initialization["encoder_sha256"]
                or module_fingerprint(check_heads) != initialization["heads_sha256"]
            ):
                raise ValueError("Persisted initial modules changed")
            initial_hash = sha(out / "initial_modules.pt")
            if (out / "completed.json").exists():
                completion = cast(dict[str, Any], read_json(out / "completed.json"))
                if completion["identity"] != identity or any(
                    sha(out / name) != value for name, value in completion["hashes"].items()
                ):
                    raise ValueError("Completed compression artifacts changed")
                return cast(dict[str, Any], read_json(out / "results.json"))
            tracking = (
                read_json(out / "tracking.json") if (out / "tracking.json").exists() else None
            )
            start = 0
            records: list[dict[str, Any]] = []
            if resume and (out / "last.pt").exists():
                checkpoint = cast(
                    dict[str, Any],
                    torch.load(out / "last.pt", map_location="cpu", weights_only=False),
                )
                if (
                    checkpoint["identity"] != identity
                    or checkpoint["initial_sha256"] != initial_hash
                ):
                    raise ValueError("Checkpoint source or initialization identity differs")
                encoder.load_state_dict(checkpoint["encoder"], strict=True)
                heads.load_state_dict(checkpoint["heads"], strict=True)
                optimizer.load_state_dict(checkpoint["optimizer"])
                start, records = checkpoint["step"], checkpoint["records"]
                tracking = checkpoint["tracking"]
                _restore_rng(checkpoint["rng"], device)
            if not 0 <= start <= config.steps:
                raise ValueError("Invalid saved compression update")
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
            step = start
            try:
                logger.log(step, {"preparation/evaluation_ready": 0.0})
                evaluation = prepare_evaluation(inputs, device)
                logger.log(step, {"preparation/evaluation_ready": 1.0})

                def save() -> None:
                    if sha(out / "initial_modules.pt") != initial_hash:
                        raise ValueError("Initial modules changed during compression")
                    atomic_save(
                        out / "last.pt",
                        {
                            "identity": identity,
                            "step": step,
                            "encoder": encoder.state_dict(),
                            "heads": heads.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "rng": _rng(device),
                            "records": records,
                            "tracking": logger.metadata,
                            "initial_sha256": initial_hash,
                        },
                    )

                def validate() -> None:
                    report = assess(encoder, heads, evaluation, groups=config.groups)
                    records.append({"step": step, **report})
                    logger.log(step, metrics(report))
                    write_json(out / "validation.json", {"identity": identity, "records": records})

                if not records:
                    validate()
                    save()
                loaded_epoch, batches = -1, []
                for step in range(start + 1, config.steps + 1):
                    epoch, position = divmod(step - 1, bundle.train_groups // bundle.batch_size)
                    if epoch != loaded_epoch:
                        batches = list(
                            GlobalBatchSampler(
                                bundle.train_groups, bundle.batch_size, config.seed, epoch
                            )
                        )
                        loaded_epoch = epoch
                    selected = batches[position]
                    features, targets = (
                        inputs.train[selected].to(device),
                        inputs.targets[selected].to(device),
                    )
                    torch.manual_seed(config.seed * 100000 + step)  # pyright: ignore[reportUnknownMemberType]
                    encoder.train()
                    heads.train()
                    optimizer.zero_grad(set_to_none=True)
                    loss, terms = training_loss(encoder, heads, features, targets, objective)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"Nonfinite compression loss at update{step}")
                    loss.backward()  # pyright: ignore[reportUnknownMemberType]
                    norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            [*encoder.parameters(), *heads.parameters()],
                            1.0,
                            error_if_nonfinite=True,
                        )
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
                        validate()
                        save()
                final = assess(encoder, heads, evaluation, groups=None)
                verify_reference(inputs)
                if sha(out / "initial_modules.pt") != initial_hash:
                    raise ValueError("Initial modules changed")
                result = {
                    "identity": identity,
                    "step": step,
                    "records": records,
                    "final_full_validation": final,
                    "tracking": logger.metadata,
                    "scope": (
                        "Exploratory teacher-code compression; validation only, "
                        "no test or predictive-use claim"
                    ),
                }
                write_json(out / "results.json", result)
                logger.log(step, metrics(final, "full_validation"))
                logger.log(step, {"diagnostic/complete": 1.0})
                for name in ("protocol.json", "results.json"):
                    logger.log_artifact(
                        out / name, name=f"teacher-compression-{identity[:12]}-{Path(name).stem}"
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
                                "results.json",
                                "last.pt",
                                "initial_modules.pt",
                                "validation.json",
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
                    out / "failure.json", name=f"teacher-compression-{identity[:12]}-failure"
                )
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    """Run only after the original fixed distillation boundary exists and is authenticated."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    for name, value in (("seed", 202610261301), ("steps", 30000), ("groups", 64), ("threads", 1)):
        parser.add_argument("--" + name, type=int, default=value)
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
