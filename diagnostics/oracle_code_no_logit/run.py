"""Match the completed SPARTAN oracle control while optimizing only forecast MSE."""

import argparse
import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.group_latent_control.forward import group_teacher_forward
from diagnostics.group_latent_control.run import (
    _restore_rng,  # pyright: ignore[reportPrivateUsage]
    _rng,  # pyright: ignore[reportPrivateUsage]
)
from diagnostics.multiquery_control.data import TensorSplit, module_fingerprint
from diagnostics.multiquery_control.run import atomic_save, scalar_metrics, sha
from diagnostics.oracle_code_control.evaluation import PARAMETER_STD, assess, supplied_codes
from diagnostics.oracle_code_control.source import load_inputs
from diagnostics.oracle_code_no_logit.reference import verify_reference
from diagnostics.predictive_supervision_control.run import code_identity as shared_code_identity
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.rng import preserve_rng


@dataclass(frozen=True)
class Config:
    """Fixed-budget diagnostic with explicit physical labels, never ordinary training."""

    source_dir: str
    audit_dir: str
    output_dir: str
    training_source_root: str | None = None
    reference_run_dir: str | None = None
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
        """Protect immutable inputs and prevent silent budget changes."""
        if self.verification:
            if not 1 <= self.steps <= 3 or not 2 <= self.groups <= 4:
                raise ValueError(
                    "Verification permits at most three updates and two to four groups"
                )
        elif (self.steps, self.groups, self.bootstrap_samples) != (30000, 64, 1000):
            raise ValueError("Scientific diagnostic requires 30000 updates,64groups,1000 resamples")
        if not self.verification and (self.reference_run_dir is None or self.seed != 202610261301):
            raise ValueError(
                "Scientific no-logit control requires the original seed and completed reference"
            )
        if self.seed < 0 or self.threads < 1 or self.bootstrap_samples < 2:
            raise ValueError("Invalid seed, thread count or bootstrap budget")
        output = Path(self.output_dir).resolve()
        inputs = (
            self.source_dir,
            self.audit_dir,
            self.reference_run_dir,
            self.training_source_root,
        )
        for name in inputs:
            if name is None:
                continue
            path = Path(name).resolve()
            if output == path or output in path.parents or path in output.parents:
                raise ValueError("Output must be separate from immutable inputs")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def fixed_hashes(model: SCJEPA) -> dict[str, str]:
    """Bind every encoder; only predictor weights may change in this diagnostic."""
    return {
        name: module_fingerprint(getattr(model, name))
        for name in ("online_encoder", "target_encoder", "parameter_encoder")
    }


def training_loss(
    model: SCJEPA, batch: TensorSplit, theta: Tensor
) -> tuple[Tensor, dict[str, Tensor]]:
    """Explicit diagnostic labels supply the code; ordinary cache remains label-free."""
    if set(batch) != {"online", "target", "actions"}:
        raise ValueError("Keep physical labels separate from original cached state/action inputs")
    output = group_teacher_forward(model, batch, supplied_codes(theta))
    prediction = (output.predictions - output.targets).square().mean()
    logit = torch.stack([result.logit_penalty for result in output.dynamics]).mean()
    original = prediction + 0.0001 * logit
    return prediction, {
        "prediction": prediction,
        "logit": logit,
        "original_optimized_total": original,
        "original_weighted_logit_report_only": 0.0001 * logit,
        "effective_logit_weight": prediction.new_zeros(()),
        "optimized_total": prediction,
    }


def code_identity() -> str:
    """Include all active source, forward and evaluation dependencies in resume identity."""
    digest = hashlib.sha256(shared_code_identity().encode())
    for directory in (
        Path(__file__).parent,
        Path(__file__).parents[1] / "oracle_code_control",
        Path(__file__).parents[1] / "group_latent_control",
        Path(__file__).parents[1] / "encoder_gradient_control",
        Path(__file__).parents[1] / "joint_acquisition",
        Path(__file__).parents[1] / "signal_path",
        Path(__file__).parents[1] / "table_settling_control",
    ):
        for path in sorted(directory.glob("*.py")):
            digest.update(str(path.relative_to(Path(__file__).parents[1])).encode())
            digest.update(path.read_bytes())
    digest.update(Path(__file__).parents[1].joinpath("group_latent_table_audit.py").read_bytes())
    return digest.hexdigest()


def metrics(report: dict[str, Any], prefix: str) -> dict[str, float]:
    """Log forecast and six physical-response summaries without group-index metadata."""
    values = scalar_metrics(report["forecast"], prefix + "/forecast")
    for row in report["response"]["per_parameter"]:
        values.update(scalar_metrics(row, f"{prefix}/response/theta_{row['parameter']}"))
    values[prefix + "/all_six_predictive_use_pass"] = float(
        report["response"]["all_six_predictive_use_pass"]
    )
    return values


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Train only P with authenticated oracle codes; evaluate final test exactly at budget."""
    inputs = load_inputs(
        Path(config.source_dir),
        Path(config.audit_dir),
        config.seed,
        training_source_root=Path(config.training_source_root)
        if config.training_source_root
        else None,
        verification=config.verification,
    )
    if not config.verification and (inputs.bundle.train_groups, inputs.bundle.batch_size) != (
        8192,
        16,
    ):
        raise ValueError("Scientific diagnostic requires original8192training groups andB16")
    device, out = torch.device(config.device), Path(config.output_dir)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        with file_lock(out / ".lock"):
            model = inputs.model.to(device)
            model.predictor.requires_grad_(True)
            frozen, flags = fixed_hashes(model), model.get_extra_state()
            reference = verify_reference(
                Path(config.reference_run_dir) if config.reference_run_dir else None,
                inputs,
                seed=config.seed,
                steps=config.steps,
                groups=config.groups,
                bootstrap_samples=config.bootstrap_samples,
                device=config.device,
                threads=config.threads,
                verification=config.verification,
            )
            optimizer = torch.optim.Adam(model.predictor.parameters(), lr=0.001)
            settings = asdict(config)
            for name in (
                "source_dir",
                "audit_dir",
                "output_dir",
                "training_source_root",
                "reference_run_dir",
                "wandb_mode",
                "wandb_entity",
            ):
                settings.pop(name)
            evidence: dict[str, Any] = {
                "settings": settings,
                "source_hashes": inputs.hashes,
                "initialization": inputs.initialization,
                "matched_reference": reference,
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
                    "Attention-logit pressure limits the original SPARTAN predictor with "
                    "exact physical codes in the current learned state coordinates."
                ),
                "controlled_change": (
                    "Remove only the optimized 1e-4 logit term from the matched oracle control. "
                    "Fresh archived P receives theta/sqrt(.1875), not inferred E codes. "
                    "P-only fresh Adam .001 clip1, original B16/R10/K10/T12 two-step cached "
                    "forecasts; all encoders fixed."
                ),
                "diagnostic_physical_labels_in_training": True,
                "ordinary_training_modified": False,
                "parameter_population_std": PARAMETER_STD,
                "objective": (
                    "Mean two-step forecast MSE only. The same forward still computes "
                    "the original logit penalty and original weighted total for reporting. "
                    "No SIGReg/consistency gradients or optimizer history."
                ),
                "rng": (
                    "Original donor-pair RNG draw retained by repeated-code forward; no unused "
                    "SIGReg projection draws; seed per update follows original 30k control."
                ),
                "evaluation": (
                    "All validation groups for ordinary exact/zero-code forecasts; first 64 "
                    "validation groups for primary one-sided .01 bounded physical responses, "
                    "donor0 heldout queries5..9; full test and all test-group responses only "
                    "at final 30000. Zero codes ablate the same trained oracle predictor."
                ),
                "decision": (
                    "Improvement versus the matched original SPARTAN control implicates logit "
                    "pressure. Residual differences versus the MLP comparator keep the "
                    "predictor architecture/optimization pipeline open. Unchanged performance "
                    "does not prove global incapacity. No parameter acquisition claim follows "
                    "because exact physical codes are supplied."
                ),
            }
            if resume:
                if read_json(out / "protocol.json")["identity"] != identity:
                    raise ValueError("Resume settings, code or source changed")
            elif any(path.name != ".lock" for path in out.iterdir()):
                raise FileExistsError("Use a fresh output directory or explicit resume")
            else:
                write_json(out / "protocol.json", protocol)
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
                if fixed_hashes(model) != frozen or model.get_extra_state() != flags:
                    raise RuntimeError("Frozen encoders or rollout settings changed")

            check_fixed()
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=f"oracle-code-no-logit-{out.parent.name}",
                    job_type="diagnostic",
                ),
                resolved_config=protocol,
                dataset_identity=inputs.bundle.protocol["identity"],
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
                with preserve_rng(device):
                    report = assess(
                        copy.deepcopy(model).double().eval(),
                        inputs.bundle.evaluation,
                        "validation",
                        groups=config.groups,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                    )
                raw[step] = report.pop("raw")
                row = {"step": step, **report}
                records.append(row)
                logger.log(step, metrics(report, "validation"))
                write_json(out / "validation.json", {"identity": identity, "records": records})

            step = start
            try:
                if not records:
                    validate(0)
                    save(0)
                loaded_epoch, batches = -1, []
                for step in range(start + 1, config.steps + 1):
                    epoch, position = divmod(
                        step - 1, inputs.bundle.train_groups // inputs.bundle.batch_size
                    )
                    if epoch != loaded_epoch:
                        batches = list(
                            GlobalBatchSampler(
                                inputs.bundle.train_groups,
                                inputs.bundle.batch_size,
                                config.seed,
                                epoch,
                            )
                        )
                        loaded_epoch = epoch
                    selected = batches[position]
                    batch = {
                        name: value[selected].to(device)
                        for name, value in inputs.bundle.train.items()
                    }
                    theta = inputs.theta[selected].to(next(model.parameters()))
                    torch.manual_seed(config.seed * 100000 + step)  # pyright: ignore[reportUnknownMemberType]
                    model.train()
                    model.online_encoder.eval()
                    model.target_encoder.eval()
                    model.parameter_encoder.eval()
                    optimizer.zero_grad(set_to_none=True)
                    loss, terms = training_loss(model, batch, theta)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"Nonfinite diagnostic loss at update{step}")
                    loss.backward()  # pyright: ignore[reportUnknownMemberType]
                    norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.predictor.parameters(), 1.0, error_if_nonfinite=True
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
                        validate(step)
                        save(step)
                with preserve_rng(device):
                    test_groups = len(cast(TensorSplit, inputs.bundle.evaluation["test"])["theta"])
                    final = assess(
                        copy.deepcopy(model).double().eval(),
                        inputs.bundle.evaluation,
                        "test",
                        groups=test_groups,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                        final=True,
                    )
                final_raw = final.pop("raw")
                check_fixed()
                for name, expected in inputs.hashes.items():
                    if sha(Path(name)) != expected:
                        raise RuntimeError("Immutable source or audited labels changed")
                if config.reference_run_dir:
                    reference_root = Path(config.reference_run_dir)
                    if sha(reference_root / "completed.json") != reference[
                        "completed_sha256"
                    ] or any(
                        sha(reference_root / name) != value
                        for name, value in reference["artifact_hashes"].items()
                    ):
                        raise RuntimeError("Immutable matched reference changed")
                result = {
                    "identity": identity,
                    "step": step,
                    "records": records,
                    "final_test": final,
                    "tracking": logger.metadata,
                    "scope": (
                        "Matched no-logit exact-parameter-code diagnostic, "
                        "not label-free acquisition"
                    ),
                    "matched_reference": reference,
                }
                write_json(out / "results.json", result)
                atomic_save(
                    out / "measurements.pt",
                    {"identity": identity, "validation": raw, "test": final_raw},
                )
                logger.log(step, metrics(final, "test"))
                logger.log(step, {"diagnostic/complete": 1.0})
                for name in ("results.json", "protocol.json"):
                    logger.log_artifact(
                        out / name, name=f"oracle-code-no-logit-{identity[:12]}-{Path(name).stem}"
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
                                "validation.json",
                                "measurements.pt",
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
                    out / "failure.json", name=f"oracle-code-no-logit-{identity[:12]}-failure"
                )
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Launch only the explicit supervised diagnostic in its own output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "audit-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--training-source-root")
    parser.add_argument("--reference-run-dir")
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
