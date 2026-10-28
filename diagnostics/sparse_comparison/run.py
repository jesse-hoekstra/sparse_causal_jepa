"""Run one matched fixed30k dense/baseline/full arm through the unchanged native Trainer."""

import argparse
import math
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.global_code_calibration.source import verify_files
from diagnostics.mlp_acquisition_control.evaluation import measurement_metrics
from diagnostics.multiquery_control.run import sha
from diagnostics.overcomplete_use_precision.run import verify_replay
from diagnostics.sparse_comparison.artifacts import (
    code_identity,
    disjoint,
    save_checkpoint,
    verify_checkpoint,
)
from diagnostics.sparse_comparison.evaluation import final, monitor
from diagnostics.sparse_comparison.source import STREAM_SEED, load_inputs
from diagnostics.sparse_comparison.training import (
    CONDITIONS,
    Condition,
    assert_trainer,
    make_trainer,
    step,
)
from diagnostics.table_settling_control.source import authenticated_directory, model_fingerprint
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.rng import preserve_rng


@dataclass(frozen=True)
class Config:
    """One predeclared arm; the same fixed additional budget applies to all three."""

    shared_dir: str
    output_dir: str
    condition: Condition
    device: str = "cpu"
    threads: int = 1
    steps: int = 30000
    groups: int = 64
    bootstrap_samples: int = 1000
    chunk_size: int = 64
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Forbid adaptive budgets and overlap with the shared preparation."""
        if self.condition not in CONDITIONS:
            raise ValueError("Require dense, baseline or full")
        if self.verification:
            if not 1 <= self.steps <= 3 or not 2 <= self.groups <= 4:
                raise ValueError("Verification allows1..3 updates and2..4 monitor groups")
        elif (self.steps, self.groups, self.bootstrap_samples) != (30000, 64, 1000):
            raise ValueError("Scientific comparison fixes30000updates/64monitor/1000bootstrap")
        if min(self.threads, self.chunk_size) < 1 or self.bootstrap_samples < 2:
            raise ValueError("Require positive execution settings")
        disjoint(Path(self.output_dir), [Path(self.shared_dir)])
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Train one fixed arm, preserving native optimizer/GECO/RNG state on exact resume."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    out, shared, device = (
        Path(config.output_dir),
        Path(config.shared_dir),
        torch.device(config.device),
    )
    try:
        complete = authenticated_directory(
            shared,
            {
                "protocol.json",
                "results.json",
                "tau.json",
                "initial_model.pt",
                "startup.json",
                "preflight.json",
            },
        )
        prepared = cast(dict[str, Any], read_json(shared / "protocol.json"))
        if (
            prepared["code_identity"] != code_identity()
            or prepared["verification"] != config.verification
        ):
            raise ValueError("Shared preparation code or verification role changed")
        endpoint = Path(prepared["config"]["endpoint_dir"])
        inputs = load_inputs(endpoint, device=device, verification=config.verification)
        if (
            prepared["source_hashes"] != inputs.hashes
            or prepared["source_provenance"] != inputs.provenance
            or (not config.verification and inputs.seed != STREAM_SEED)
        ):
            raise ValueError("Prepared raw source or stream changed")
        tau_record = cast(dict[str, Any], read_json(shared / "tau.json"))
        tau = float(tau_record["tau"])
        if (
            tau_record["identity"] != complete["identity"]
            or tau_record["source_checkpoint_sha256"] != inputs.provenance["checkpoint_sha256"]
            or tau_record["source_model_sha256"] != model_fingerprint(inputs.model)
            or tau_record["tau_multiplier"] != 1.0
            or tau != tau_record["calibration/base_loss"]
            or not math.isfinite(tau)
            or tau <= 0
        ):
            raise ValueError("Shared full-base calibration identity or bound differs")
        if not config.verification and not inputs.admission["passed"]:
            raise ValueError("Raw-source amended admission no longer passes")
        shared_hashes = {
            str((shared / name).resolve()): value for name, value in complete["hashes"].items()
        }
        shared_hashes[str((shared / "completed.json").resolve())] = sha(shared / "completed.json")
        disjoint(out, [Path(name).parent for name in {**inputs.hashes, **shared_hashes}])
        with file_lock(out / ".lock"):
            trainer = make_trainer(inputs, config.condition, out, tau=tau, steps=config.steps)
            expected_training = asdict(trainer.config)
            settings = asdict(config)
            for key in ("shared_dir", "output_dir", "wandb_mode", "wandb_entity"):
                settings.pop(key)
            evidence = {
                "experiment_identity": complete["identity"],
                "source_hashes": inputs.hashes,
                "shared_hashes": shared_hashes,
                "settings": settings,
                "code_identity": code_identity(),
                "source_step": inputs.source_step,
                "stream_seed": inputs.seed,
                "raw_initializer_sha256": inputs.provenance["source_model_sha256"],
            }
            identity = canonical_fingerprint(evidence)
            protocol: dict[str, Any] = {
                "identity": identity,
                **evidence,
                "config": asdict(config),
                "command": shlex.join([sys.executable, *sys.argv]),
                "admission": inputs.admission,
                "tau": tau_record,
                "scope": (
                    "Matched fixed30000 additional updates from raw200k, fresh identical "
                    "AdamW, separate SIGReg stream, native gate policies and fresh sparse "
                    "GECO. All three final checkpoints reported regardless of scientific "
                    "success. No checkpoint selection, automatic extension, source "
                    "calibration composition or new post-fit correction;100k longer "
                    "comparison remains a separate question."
                ),
                "evaluation": (
                    "Two matched historically reused cohorts: native512cal/256val/512test "
                    "for full-base/MCC/SHD; diagnostic512cal/256val/512test for complete-code "
                    "retention and all512 test physical responses. Validation monitors every5000; "
                    "fixedfinal test only; historical positive-benefit criterion remains reported "
                    "separately from amended retention/fidelity admission."
                ),
            }
            if (out / "protocol.json").exists():
                if not resume or read_json(out / "protocol.json")["identity"] != identity:
                    raise ValueError(
                        "Matching explicit resume required; comparison identity changed"
                    )
            elif any(path.name != ".lock" for path in out.iterdir()):
                raise FileExistsError("Use a fresh output or matching explicit resume")
            else:
                write_json(out / "protocol.json", protocol)
                write_json(
                    out / "run_config.json",
                    {
                        "run_identity": identity,
                        "experiment_identity": complete["identity"],
                        "condition": config.condition,
                        "seed": inputs.seed,
                        "dataset_identity": inputs.provenance["dataset_identity"],
                        "model": trainer.model_config,
                        "data": asdict(inputs.data_config),
                        "train": expected_training,
                        "evaluation": asdict(inputs.native_evaluation),
                        "tau_calibration": tau_record,
                        "source_global_step": inputs.source_step,
                    },
                )
            if (out / "completed.json").exists():
                result_complete = authenticated_directory(
                    out,
                    {
                        "protocol.json",
                        "results.json",
                        "run_config.json",
                        "status.json",
                        "last.pt",
                        "last.receipt.json",
                        "final.receipt.json",
                        "evaluation/mcc_samples.pt",
                        "evaluation/diagnostic_final.json",
                    },
                )
                if result_complete["identity"] != identity:
                    raise ValueError("Completed comparison identity differs")
                return cast(dict[str, Any], read_json(out / "results.json"))
            trainer.run_metadata.update(
                {
                    "experiment_run_identity": identity,
                    "comparison_identity": complete["identity"],
                    "records": [],
                    "raw": {},
                }
            )
            if resume and (out / "last.pt").exists():
                receipt = verify_checkpoint(out, identity)
                trainer.load_checkpoint(out / "last.pt")
                if (
                    trainer.step != receipt["step"]
                    or trainer.run_metadata.get("experiment_run_identity") != identity
                    or trainer.run_metadata.get("comparison_identity") != complete["identity"]
                    or asdict(trainer.config) != expected_training
                ):
                    raise ValueError("Native resumed checkpoint settings or identity differ")
            assert_trainer(trainer, inputs, config.condition, tau=tau)
            if not 0 <= trainer.step <= config.steps:
                raise ValueError("Resumed local update count outside fixed budget")
            records = cast(list[dict[str, Any]], trainer.run_metadata["records"])
            raw = cast(dict[int, Any], trainer.run_metadata["raw"])
            if records and records[-1]["step"] > trainer.step:
                raise ValueError("Saved validation record is later than checkpoint")
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
                dataset_identity=inputs.provenance["dataset_identity"],
                out_dir=out,
                resume_state=tracking if tracking and tracking.get("id") else None,
            )
            write_json(out / "tracking.json", logger.metadata)
            write_json(
                out / "status.json",
                {"status": "running", "run_identity": identity, "step": trainer.step},
            )
            try:

                def validate() -> None:
                    with preserve_rng(device):
                        report, measurements = monitor(
                            trainer.model,
                            inputs,
                            groups=config.groups,
                            bootstrap_samples=config.bootstrap_samples,
                            seed=inputs.seed,
                            objective=trainer.config.objective,
                            tau=tau if config.condition != "dense" else None,
                        )
                    if trainer.step == 0 and config.condition == "dense":
                        verify_replay(
                            report["diagnostic"]["response"], inputs.archived_validation["response"]
                        )
                    records.append(
                        {
                            "step": trainer.step,
                            "global_step": inputs.source_step + trainer.step,
                            **report,
                        }
                    )
                    raw[trainer.step] = measurements
                    write_json(out / "validation.json", {"identity": identity, "records": records})
                    logger.log(
                        inputs.source_step + trainer.step,
                        {
                            **report["native"],
                            **measurement_metrics(report["diagnostic"]),
                            "validation/revised_admission_pass": float(
                                report["admission"]["revised_admission_pass"]
                            ),
                            "validation/original_retention_and_use_gate_pass": float(
                                report["admission"]["original_retention_and_use_gate_pass"]
                            ),
                        },
                    )
                    print(
                        {
                            "phase": "validation",
                            "step": trainer.step,
                            "condition": config.condition,
                            "native": report["native"],
                            "admission": report["admission"],
                        },
                        flush=True,
                    )

                if not records:
                    validate()
                    save_checkpoint(trainer, out, identity)
                for _ in range(trainer.step, config.steps):
                    started = time.perf_counter()
                    metrics = step(trainer, inputs)
                    metrics["train/seconds_per_microbatch"] = time.perf_counter() - started
                    if trainer.step == 1 or trainer.step % 100 == 0 or trainer.step == config.steps:
                        logger.log(inputs.source_step + trainer.step, metrics)
                        print(
                            {"step": trainer.step, "condition": config.condition, **metrics},
                            flush=True,
                        )
                    if (
                        config.verification
                        or trainer.step % 5000 == 0
                        or trainer.step == config.steps
                    ):
                        validate()
                    if (
                        config.verification
                        or trainer.step % 250 == 0
                        or trainer.step == config.steps
                    ):
                        assert_trainer(trainer, inputs, config.condition, tau=tau)
                        save_checkpoint(trainer, out, identity)
                # Every scientific arm reaches the same fixed endpoint, irrespective of metrics.
                checkpoint_sha = sha(out / "last.pt")
                if (out / "final.receipt.json").exists():
                    receipt = cast(dict[str, Any], read_json(out / "final.receipt.json"))
                    if (
                        receipt["identity"] != identity
                        or receipt["checkpoint_sha256"] != checkpoint_sha
                    ):
                        raise ValueError("Final assessment source changed")
                    names = list(cast(dict[str, str], receipt["hashes"]))
                    for name, expected in receipt["hashes"].items():
                        if sha(out / name) != expected:
                            raise ValueError("Completed final evaluation artifact changed")
                    final_report = cast(
                        dict[str, Any], read_json(out / "evaluation/diagnostic_final.json")
                    )
                else:
                    print(
                        {
                            "phase": "fixed_final_evaluation",
                            "step": trainer.step,
                            "condition": config.condition,
                        },
                        flush=True,
                    )
                    with preserve_rng(device):
                        final_report, names = final(
                            trainer.model,
                            inputs,
                            out,
                            objective=trainer.config.objective,
                            tau=tau if config.condition != "dense" else None,
                            identity=identity,
                            condition=config.condition,
                            step=trainer.step,
                            fixed_budget=config.steps,
                            verification=config.verification,
                            chunk_size=config.chunk_size,
                            bootstrap_samples=config.bootstrap_samples,
                            seed=inputs.seed,
                        )
                    write_json(
                        out / "final.receipt.json",
                        {
                            "identity": identity,
                            "checkpoint_sha256": checkpoint_sha,
                            "hashes": {name: sha(out / name) for name in names},
                        },
                    )
                assert_trainer(trainer, inputs, config.condition, tau=tau)
                verify_files({**inputs.hashes, **shared_hashes})
                if sha(out / "last.pt") != checkpoint_sha:
                    raise RuntimeError("Final evaluation altered native training checkpoint")
                result = {
                    "identity": identity,
                    "condition": config.condition,
                    "steps": trainer.step,
                    "global_step": inputs.source_step + trainer.step,
                    "source_admission": inputs.admission,
                    "records": records,
                    "final": final_report,
                    "tracking": logger.metadata,
                    "scope": protocol["scope"],
                }
                write_json(out / "results.json", result)
                write_json(
                    out / "status.json",
                    {"status": "completed", "run_identity": identity, "step": trainer.step},
                )
                logger.log(
                    inputs.source_step + trainer.step,
                    {
                        **final_report["native"],
                        **measurement_metrics(final_report["diagnostic"], final=True),
                        "diagnostic/complete": 1.0,
                    },
                )
                for name in ("protocol.json", "results.json", "evaluation/metrics.json"):
                    logger.log_artifact(
                        out / name,
                        name=f"sparse-{config.condition}-{identity[:12]}-{Path(name).stem}",
                    )
                names = [
                    "protocol.json",
                    "run_config.json",
                    "last.pt",
                    "last.receipt.json",
                    "validation.json",
                    "final.receipt.json",
                    "results.json",
                    "status.json",
                    "tracking.json",
                    *names,
                ]
                write_json(
                    out / "completed.json",
                    {"identity": identity, "hashes": {name: sha(out / name) for name in names}},
                )
                logger.finish()
                return result
            except BaseException:
                write_json(
                    out / "status.json",
                    {
                        "status": "interrupted_or_failed",
                        "run_identity": identity,
                        "step": trainer.step,
                    },
                )
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Launch one fixed30k arm from the prepared raw source and common GECO bound."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("shared-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--resume", action="store_true")
    values = vars(parser.parse_args())
    resume = bool(values.pop("resume"))
    run(Config(**values), resume=resume)


if __name__ == "__main__":
    main()
