"""Evaluate each predeclared fixed predictor arm on the same existing precision validation."""

import argparse
import hashlib
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.global_code_calibration.source import verify_files
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.overcomplete_use_precision.run import code_identity as precision_code_identity
from diagnostics.overcomplete_use_precision.run import verify_replay
from diagnostics.predictor_transfer_control.model import audit_model_fingerprint
from diagnostics.predictor_transfer_control.run import code_identity as transfer_code_identity
from diagnostics.predictor_transfer_precision.evaluation import evaluate
from diagnostics.predictor_transfer_precision.source import Arm, load_inputs
from diagnostics.table_settling_control.source import authenticated_directory
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.training.logging import WandbConfig, WandbLogger


@dataclass(frozen=True)
class Config:
    """One of three predeclared models, with a fixed final-checkpoint SHA supplied before launch."""

    endpoint_dir: str
    source_dir: str
    precision_dir: str
    output_dir: str
    arm: Arm
    checkpoint_sha256: str
    device: str = "cpu"
    chunk_size: int = 64
    bootstrap_samples: int = 1000
    evaluation_seed: int = 202610261301
    threads: int = 1
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Reject adaptive scientific seeds/bootstraps and protect completed source directories."""
        if self.arm not in ("spartan", "mlp", "reference"):
            raise ValueError("Require one of all three predeclared arms")
        if len(self.checkpoint_sha256) != 64 or any(
            x not in "0123456789abcdef" for x in self.checkpoint_sha256
        ):
            raise ValueError("Explicit full SHA256 checkpoint pin required")
        if (
            min(self.chunk_size, self.threads) < 1
            or self.bootstrap_samples < 2
            or self.evaluation_seed < 0
        ):
            raise ValueError("Require valid evaluation sizes/seed")
        if not self.verification and (self.bootstrap_samples, self.evaluation_seed) != (
            1000,
            202610261301,
        ):
            raise ValueError("Original scientific bootstrap and evaluation seed are fixed")
        output = Path(self.output_dir).resolve()
        for name in (self.endpoint_dir, self.source_dir, self.precision_dir):
            source = Path(name).resolve()
            if output == source or output in source.parents or source in output.parents:
                raise ValueError("Output must be disjoint from immutable source directories")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def code_identity() -> str:
    """Bind the original learned endpoints, unchanged physical evaluator and this adapter."""
    digest = hashlib.sha256((transfer_code_identity() + precision_code_identity()).encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Replay original validation, then evaluate all8192 existing groups without any fitting."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    out = Path(config.output_dir)
    try:
        with file_lock(out / ".lock"):
            inputs = load_inputs(
                Path(config.endpoint_dir),
                Path(config.source_dir),
                Path(config.precision_dir),
                arm=config.arm,
                checkpoint_sha256=config.checkpoint_sha256,
                verification=config.verification,
            )
            if config.evaluation_seed != inputs.evaluation_seed:
                raise ValueError("Preserve the original transfer evaluation seed")
            model = inputs.model.to(config.device)
            frozen = audit_model_fingerprint(model)
            protocol: dict[str, Any] = {
                "config": asdict(config),
                "source_hashes": inputs.hashes,
                "source_identity": inputs.source_protocol["identity"],
                "source_model_sha256": frozen,
                "chart_provenance": inputs.chart_provenance,
                "code_identity": code_identity(),
                "command": shlex.join([sys.executable, *sys.argv]),
                "predeclared_arms": ["spartan", "mlp", "reference"],
                "hypothesis": (
                    "With the learned six-code encoder fixed, this precision audit separates "
                    "persistent predictor response/bias failures from noise in the "
                    "original64-group use assay."
                ),
                "controlled_change": (
                    "Only read-only evaluation sample size: use the existing8192-group precision "
                    "VALIDATIONcache. Both fixed100k trained arms and the frozen composedMLP6 "
                    "reference are all predeclared and must all be reported; separate jobs allow "
                    "controls to run while SPARTAN finishes."
                ),
                "evaluation": (
                    "Original donor0/query5..9, positive min(.01,.5*(.75-abs(theta))), attached "
                    "two steps and one1000-draw bootstrap over all groups using the original "
                    "evaluation seed. Replay original64validation first; save direct vectors "
                    "verifying B=F-2e·l."
                ),
                "decision": (
                    "Use all3 reports together: successfulSPARTAN supportsgraph-bearingtransfer; "
                    "MLP-onlysuccess localizes testedSPARTANtraining; "
                    "bothfreshfail/composedreferenceworks leavesfreshpredictorlearning limiting. "
                    "No checkpoint, seed, model or subset selection."
                ),
                "scope": (
                    "Adaptive reused-validation diagnostic; no TRAIN/PCA/probe fitting, no new "
                    "data or test predictions, no replacement of original64test. Separate fresh12D "
                    "confirmation and full state-MCC goals remain distinct."
                ),
            }
            protocol["identity"] = canonical_fingerprint(
                {k: v for k, v in protocol.items() if k != "command"}
            )
            identity = protocol["identity"]
            if (out / "completed.json").exists():
                completed = authenticated_directory(
                    out, {"protocol.json", "results.json", "measurements.pt", "replay.json"}
                )
                if completed["identity"] != identity:
                    raise ValueError("Completed precision identity changed")
                return cast(dict[str, Any], read_json(out / "results.json"))
            if (out / "protocol.json").exists():
                if not resume or read_json(out / "protocol.json")["identity"] != identity:
                    raise ValueError("Matching explicit resume required")
            else:
                write_json(out / "protocol.json", protocol)
            tracking = (
                cast(dict[str, Any], read_json(out / "tracking.json"))
                if (out / "tracking.json").exists()
                else None
            )
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=f"predictor-transfer-precision-{config.arm}-{out.parent.name}",
                    job_type="diagnostic",
                ),
                resolved_config=protocol,
                dataset_identity=inputs.source_protocol["identity"],
                out_dir=out,
                resume_state=tracking if tracking and tracking.get("id") else None,
            )
            step = int(tracking.get("last_event_step", 0)) if tracking else 0
            write_json(out / "tracking.json", {**logger.metadata, "last_event_step": step})

            def log(metrics: dict[str, float]) -> None:
                nonlocal step
                step += 1
                write_json(out / "tracking.json", {**logger.metadata, "last_event_step": step})
                logger.log(step, metrics)
                print({"step": step, **metrics}, flush=True)

            try:
                replay, _, _ = evaluate(
                    model,
                    inputs.original_validation,
                    chunk_size=config.chunk_size,
                    bootstrap_samples=inputs.original_bootstrap_samples,
                    seed=inputs.evaluation_seed,
                )
                verify_replay(replay, inputs.archived_report)
                write_json(
                    out / "replay.json", {"identity": identity, "report": replay, "matched": True}
                )
                log({"progress/original_validation_replay_complete": 1.0})
                report, raw, vectors = evaluate(
                    model,
                    inputs.precision_data,
                    chunk_size=config.chunk_size,
                    bootstrap_samples=config.bootstrap_samples,
                    seed=inputs.evaluation_seed,
                    progress=lambda n: log({"progress/precision_validation_groups": float(n)}),
                )
                passed = [
                    row["parameter"]
                    for row in report["per_parameter"]
                    if row["predictive_benefit"]["interval_95"][0] > 0
                    and row["normalized_response_error"]["interval_95"][1] < 1
                ]
                result = {
                    "identity": identity,
                    "arm": config.arm,
                    "report": report,
                    "use_pass_parameters": passed,
                    "all_six_use_pass": len(passed) == 6,
                    "tracking": logger.metadata,
                    "scope": protocol["scope"],
                    "chart_provenance": inputs.chart_provenance,
                }
                metrics = {"validation/all_six_use_pass": float(len(passed) == 6)}
                for row in report["per_parameter"]:
                    for name in ("predictive_benefit", "normalized_response_error"):
                        key = f"validation/theta_{row['parameter']}/{name}"
                        metrics[key] = float(row[name]["estimate"])
                        metrics[key + "/lower95"], metrics[key + "/upper95"] = row[name][
                            "interval_95"
                        ]
                log(metrics)
                if audit_model_fingerprint(model) != frozen or any(
                    p.grad is not None or p.requires_grad for p in model.parameters()
                ):
                    raise RuntimeError("Frozen predictor-transfer model changed")
                verify_files(inputs.hashes)
                write_json(out / "results.json", result)
                atomic_save(
                    out / "measurements.pt", {"identity": identity, "raw": raw, "vectors": vectors}
                )
                for name in ("protocol.json", "results.json", "replay.json"):
                    logger.log_artifact(
                        out / name, name=f"transfer-precision-{config.arm}-{Path(name).stem}"
                    )
                log({"diagnostic/complete": 1.0})
                write_json(
                    out / "completed.json",
                    {
                        "identity": identity,
                        "hashes": {
                            name: sha(out / name)
                            for name in (
                                "protocol.json",
                                "results.json",
                                "replay.json",
                                "measurements.pt",
                                "tracking.json",
                            )
                        },
                    },
                )
                logger.finish()
                return result
            except BaseException:
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Expose the fixed validation assay; root supplies the completed checkpoint's exact SHA."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("endpoint-dir", "source-dir", "precision-dir", "output-dir", "checkpoint-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--arm", choices=("spartan", "mlp", "reference"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--resume", action="store_true")
    values = vars(parser.parse_args())
    resume = bool(values.pop("resume"))
    run(Config(**values), resume=resume)


if __name__ == "__main__":
    main()
