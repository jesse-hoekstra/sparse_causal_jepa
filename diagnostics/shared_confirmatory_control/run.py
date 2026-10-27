"""One predeclared fresh confirmation of both source/calibrated acquisition pairs."""

import argparse
import hashlib
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.global_code_calibration.source import verify_files
from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.multiquery_control.evaluation import encode_codes
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.overcomplete_use_precision.evaluation import evaluate
from diagnostics.shared_code_calibration.run import code_identity as parent_code_identity
from diagnostics.shared_confirmatory_control.data import build_fresh, independence
from diagnostics.shared_confirmatory_control.probes import (
    fit_probes,
    predict,
    score,
    verify_archived,
)
from diagnostics.shared_confirmatory_control.source import load_inputs
from diagnostics.table_settling_control.source import authenticated_directory, model_fingerprint
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.training.logging import WandbConfig, WandbLogger

ARMS = ("seed1/source", "seed1/calibrated", "seed2/source", "seed2/calibrated")


@dataclass(frozen=True)
class Config:
    """One fixed confirmatory sample; verification uses small genuine fixtures only."""

    calibration_seed1_dir: str
    calibration_seed2_dir: str
    source_dir: str
    precision_dir: str
    output_dir: str
    groups: int = 32768
    seed: int = 202610272901
    bootstrap_samples: int = 1000
    evaluation_seed: int = 202610261301
    chunk_size: int = 128
    threads: int = 1
    device: str = "cpu"
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Reject undeclared sample sizes/seeds and output paths overlapping immutable sources."""
        if self.verification:
            if not 2 <= self.groups <= 8:
                raise ValueError("Verification requires2..8 fresh groups")
        elif (self.groups, self.seed, self.bootstrap_samples, self.evaluation_seed) != (
            32768,
            202610272901,
            1000,
            202610261301,
        ):
            raise ValueError(
                "Confirmatory count, draw seed and original bootstrap protocol are fixed"
            )
        if (
            min(self.chunk_size, self.threads) < 1
            or self.bootstrap_samples < 2
            or min(self.seed, self.evaluation_seed) < 0
        ):
            raise ValueError("Require positive sizes and valid seeds")
        output = Path(self.output_dir).resolve()
        for name in (
            self.calibration_seed1_dir,
            self.calibration_seed2_dir,
            self.source_dir,
            self.precision_dir,
        ):
            source = Path(name).resolve()
            if output == source or source in output.parents or output in source.parents:
                raise ValueError("Output must be disjoint from immutable sources")
        if Path(self.calibration_seed1_dir).resolve() == Path(self.calibration_seed2_dir).resolve():
            raise ValueError("Both distinct training-seed sources are required")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def code_identity() -> str:
    """Bind this isolated evaluator and every unchanged acquisition/calibration dependency."""
    digest = hashlib.sha256(parent_code_identity().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def checked_payload(out: Path, name: str, identity: str) -> dict[str, Any]:
    """Load a durable stage only when both protocol identity and exact file hash agree."""
    receipt = read_json(out / f"{name}.receipt.json")
    if receipt["identity"] != identity or receipt["sha256"] != sha(out / f"{name}.pt"):
        raise ValueError(f"Saved {name} stage changed")
    return cast(
        dict[str, Any], torch.load(out / f"{name}.pt", map_location="cpu", weights_only=False)
    )


def save_payload(out: Path, name: str, payload: dict[str, Any], identity: str) -> None:
    """Make the persisted stage and its identity/hash receipt durable before advancing."""
    atomic_save(out / f"{name}.pt", payload)
    write_json(
        out / f"{name}.receipt.json", {"identity": identity, "sha256": sha(out / f"{name}.pt")}
    )


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Freeze old-data probes, generate one new test sample, then report all four fixed arms."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    out = Path(config.output_dir)
    try:
        with file_lock(out / ".lock"):
            inputs = load_inputs(
                (Path(config.calibration_seed1_dir), Path(config.calibration_seed2_dir)),
                Path(config.source_dir),
                Path(config.precision_dir),
                verification=config.verification,
            )
            if tuple(inputs.models) != ARMS:
                raise ValueError("Require all four predeclared arms in fixed order")
            frozen = {key: model_fingerprint(value) for key, value in inputs.models.items()}
            protocol: dict[str, Any] = {
                "config": asdict(config),
                "source_hashes": inputs.hashes,
                "source_model_sha256": frozen,
                "code_identity": code_identity(),
                "command": shlex.join([sys.executable, *sys.argv]),
                "hypothesis": (
                    "TRAIN-only calibration and shared-feature acquisition retain all six factors "
                    "and improve all six physical responses on independent groups in BOTH declared "
                    "seeds."
                ),
                "controlled_change": (
                    "One fixed fresh32768-group test sample after models and "
                    "original-calibration/validation-selected probes are frozen. No training, "
                    "source selection, retries, subsets or sample-size adaptation."
                ),
                "evaluation": (
                    "All four arms, same groups; original donor0/query5..9, positive "
                    "min(.01,.5*(.75-abs(theta))) interventions, attached two horizons, "
                    "original1000 group-bootstrap draws. Every use factor needs benefit lower95>0 "
                    "and normalized-response-error upper95<1. Every retention factor needs "
                    "single-context full-code R2>=.80."
                ),
                "retention": (
                    "Before generating fresh groups, freeze the exact old512cal/256val "
                    "linear/cubic/RFF256 family/ridge/width selection and coefficient tensors for "
                    "each source. Undo each known calibrated code translation and reuse its source "
                    "probe. No fresh labels enter fitting/selection."
                ),
                "decision": (
                    "Both calibrated seeds pass all factors: confirm this diagnostic recipe "
                    "conditional on two seeds/frozen charts. Either fails: report the failure "
                    "without retry or extra sample. Preserve uncalibrated references and "
                    "original64 tests. Production transfer/state-MCC remain separate."
                ),
                "limits": (
                    "Adaptively developed recipe, then fresh fixed confirmation; two training "
                    "seeds share one training dataset/state chart. Individual per-factor95% "
                    "intervals are the original criterion, not simultaneous coverage or broad seed "
                    "generalization."
                ),
                "train_identity_only": inputs.train_config,
            }
            protocol["identity"] = canonical_fingerprint(
                {k: v for k, v in protocol.items() if k != "command"}
            )
            identity = protocol["identity"]
            if (out / "completed.json").exists():
                complete = authenticated_directory(
                    out, {"protocol.json", "results.json", "probes.pt", "fresh_test.pt"}
                )
                if complete["identity"] != identity:
                    raise ValueError("Completed confirmatory identity changed")
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
                    name=f"shared-confirmatory-{out.parent.name}",
                    job_type="diagnostic",
                ),
                resolved_config=protocol,
                dataset_identity=identity,
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
                if (out / "probes.pt").exists():
                    probes = checked_payload(out, "probes", identity)
                else:
                    if (out / "fresh_test.pt").exists():
                        raise ValueError("Fresh data cannot exist before immutable probe freezing")
                    probes: dict[str, Any] = {}
                    for prefix in ("seed1", "seed2"):
                        model = inputs.models[prefix + "/source"].to(config.device)
                        cal, val = inputs.old_data["calibration"], inputs.old_data["validation"]
                        probes[prefix] = fit_probes(
                            encode_codes(model, cal),
                            encode_codes(model, val),
                            cal["theta"],
                            val["theta"],
                        )
                        verify_archived(probes[prefix], inputs.endpoint_results[prefix])
                        log({f"progress/{prefix}_probe_frozen": 1.0})
                    save_payload(out, "probes", probes, identity)
                if set(probes) != {"seed1", "seed2"} or not all(
                    p["selection_frozen_before_fresh_generation"] for p in probes.values()
                ):
                    raise ValueError("Both old-data probes must be frozen before fresh generation")
                # This call is deliberately below the immutable probe receipt above.
                if (out / "fresh_test.pt").exists():
                    fresh = checked_payload(out, "fresh_test", identity)
                else:
                    fresh = build_fresh(
                        inputs.models["seed1/source"].to(config.device),
                        inputs.old_data["metadata"],
                        groups=config.groups,
                        seed=config.seed,
                        batch_size=config.chunk_size,
                        progress=lambda n: log({"progress/fresh_test_cache_groups": float(n)}),
                    )
                    save_payload(out, "fresh_test", fresh, identity)
                if (
                    len(fresh["test"]["theta"]) != config.groups
                    or fresh["metadata"]["seed"] != config.seed
                ):
                    raise ValueError("Fresh sample size or seed differs")
                independent = independence(
                    fresh, inputs.old_data, inputs.precision_data, inputs.train_config
                )
                write_json(out / "independence.json", {"identity": identity, **independent})
                # The unchanged response helper names its only slot 'validation'; role remains test.
                adapted: EvaluationData = {
                    "metadata": fresh["metadata"],
                    "validation": fresh["test"],
                }
                results: dict[str, Any] = {}
                for arm in ARMS:
                    stage = "arm_" + arm.replace("/", "_")
                    if (out / f"{stage}.pt").exists():
                        payload = checked_payload(out, stage, identity)
                        results[arm] = payload["report"]
                        continue
                    model = inputs.models[arm].to(config.device)
                    response, raw, vectors = evaluate(
                        model,
                        adapted,
                        chunk_size=config.chunk_size,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.evaluation_seed,
                        progress=lambda n, stage=stage: log({f"progress/{stage}_groups": float(n)}),
                    )
                    codes = encode_codes(model, fresh["test"], batch_size=config.chunk_size)
                    prefix, condition = arm.split("/")
                    delta = inputs.translations[prefix] if condition == "calibrated" else None
                    predictions = predict(probes[prefix], codes, delta)
                    retention, errors = score(predictions, fresh["test"]["theta"])
                    use_pass = [
                        row["parameter"]
                        for row in response["per_parameter"]
                        if row["predictive_benefit"]["interval_95"][0] > 0
                        and row["normalized_response_error"]["interval_95"][1] < 1
                    ]
                    report: dict[str, Any] = {
                        "role": "confirmatory_test",
                        "response": response,
                        "retention": retention,
                        "use_pass_parameters": use_pass,
                        "all_six_use_pass": len(use_pass) == 6,
                        "all_six_retention_and_use_pass": len(use_pass) == 6
                        and retention["all_six_above_0_80"],
                    }
                    save_payload(
                        out,
                        stage,
                        {
                            "report": report,
                            "raw": raw,
                            "vectors": vectors,
                            "retention_predictions": predictions,
                            "retention_group_mse": errors,
                        },
                        identity,
                    )
                    results[arm] = report
                    metrics = {
                        f"{stage}/retention/minimum_r2": retention["minimum_r2"],
                        f"{stage}/all_six_retention_and_use_pass": float(
                            report["all_six_retention_and_use_pass"]
                        ),
                    }
                    for factor, row in enumerate(response["per_parameter"]):
                        metrics[f"{stage}/theta_{factor + 1}/retention_r2"] = retention["r2"][
                            factor
                        ]
                        for name in ("predictive_benefit", "normalized_response_error"):
                            key = f"{stage}/theta_{factor + 1}/{name}"
                            metrics[key] = row[name]["estimate"]
                            metrics[key + "/lower95"], metrics[key + "/upper95"] = row[name][
                                "interval_95"
                            ]
                    log(metrics)
                for prefix in ("seed1", "seed2"):
                    source = checked_payload(out, "arm_" + prefix + "_source", identity)
                    calibrated = checked_payload(out, "arm_" + prefix + "_calibrated", identity)
                    torch.testing.assert_close(
                        source["retention_predictions"],
                        calibrated["retention_predictions"],
                        atol=1e-9,
                        rtol=1e-7,
                    )
                for arm, model in inputs.models.items():
                    if model_fingerprint(model) != frozen[arm] or any(
                        p.requires_grad or p.grad is not None for p in model.parameters()
                    ):
                        raise RuntimeError("Frozen confirmatory model changed")
                verify_files(inputs.hashes)
                result = {
                    "identity": identity,
                    "reports": results,
                    "both_calibrated_seeds_pass": all(
                        results[p + "/calibrated"]["all_six_retention_and_use_pass"]
                        for p in ("seed1", "seed2")
                    ),
                    "independence": independent,
                    "tracking": logger.metadata,
                    "scope": protocol["limits"],
                }
                write_json(out / "results.json", result)
                for name in ("protocol.json", "results.json", "independence.json"):
                    logger.log_artifact(out / name, name=f"shared-confirmatory-{Path(name).stem}")
                log({"diagnostic/complete": 1.0})
                names = [
                    "protocol.json",
                    "results.json",
                    "independence.json",
                    "tracking.json",
                    "probes.pt",
                    "probes.receipt.json",
                    "fresh_test.pt",
                    "fresh_test.receipt.json",
                ]
                names += [
                    "arm_" + arm.replace("/", "_") + suffix
                    for arm in ARMS
                    for suffix in (".pt", ".receipt.json")
                ]
                write_json(
                    out / "completed.json",
                    {"identity": identity, "hashes": {name: sha(out / name) for name in names}},
                )
                logger.finish()
                return result
            except BaseException:
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Expose fixed confirmation with explicit, identity-preserving interruption recovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "calibration-seed1-dir",
        "calibration-seed2-dir",
        "source-dir",
        "precision-dir",
        "output-dir",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=128)
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
