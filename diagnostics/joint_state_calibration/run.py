"""Calibrate six constant code offsets on TRAIN, then assess both fixed validation samples."""

import argparse
import hashlib
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.global_code_calibration.run import code_identity as optimizer_code_identity
from diagnostics.global_code_calibration.run import compact_fit
from diagnostics.global_code_calibration.source import verify_files
from diagnostics.joint_state_calibration.source import (
    PINNED,
    Arm,
    encode_observations,
    load_evaluation,
    load_training,
)
from diagnostics.joint_state_survival.run import code_identity as survival_code_identity
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.overcomplete_use_precision.run import verify_replay
from diagnostics.predictor_transfer_calibration.objective import (
    TrainObjective,
    compose_translation,
    encode_train,
)
from diagnostics.predictor_transfer_calibration.optimization import (
    EmbeddedObjective,
    active_state,
    fit,
    verify_embedding,
)
from diagnostics.predictor_transfer_calibration.run import code_identity as evaluation_code_identity
from diagnostics.predictor_transfer_precision.evaluation import evaluate
from diagnostics.table_settling_control.source import authenticated_directory, model_fingerprint
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.training.logging import WandbConfig, WandbLogger


@dataclass(frozen=True)
class Config:
    """Both trained arms are predeclared, each with one bounded TRAIN-only calibration."""

    endpoint_dir: str
    source_dir: str
    precision_dir: str
    output_dir: str
    arm: Arm
    checkpoint_sha256: str
    device: str = "cpu"
    chunk_size: int = 16
    threads: int = 1
    max_iterations: int = 25
    max_evaluations: int = 50
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Protect fixed budgets, source checkpoints and the scientific two-arm scope."""
        if self.arm not in ("frozen_state", "live_state"):
            raise ValueError("Require one of both declared trained predictor arms")
        if len(self.checkpoint_sha256) != 64 or any(
            x not in "0123456789abcdef" for x in self.checkpoint_sha256
        ):
            raise ValueError("Full fixed-final checkpoint SHA required")
        if min(self.chunk_size, self.threads, self.max_iterations, self.max_evaluations) < 1:
            raise ValueError("Positive sizes and fixed budgets required")
        if self.verification:
            if self.max_iterations > 3 or self.max_evaluations > 8:
                raise ValueError("Verification permits at most3accepted/8full evaluations")
        elif (self.max_iterations, self.max_evaluations) != (25, 50):
            raise ValueError("Scientific calibration fixes25accepted/50unique full evaluations")
        if not self.verification and self.checkpoint_sha256 != PINNED[self.arm]:
            raise ValueError("Require the explicitly pinned fixed110k endpoint")
        output = Path(self.output_dir).resolve()
        for name in (self.endpoint_dir, self.source_dir, self.precision_dir):
            source = Path(name).resolve()
            if output == source or output in source.parents or source in output.parents:
                raise ValueError("Output must be disjoint from immutable source directories")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def code_identity() -> str:
    """Bind the unchanged optimizer, transfer/evaluation chain and six-code adapter."""
    digest = hashlib.sha256(
        (optimizer_code_identity() + evaluation_code_identity() + survival_code_identity()).encode()
    )
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Fit only six offsets; deserialize held-out input caches only after the fixed fit ends."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    out = Path(config.output_dir)
    try:
        with file_lock(out / ".lock"):
            inputs = load_training(
                Path(config.endpoint_dir),
                Path(config.source_dir),
                Path(config.precision_dir),
                arm=config.arm,
                checkpoint_sha256=config.checkpoint_sha256,
                verification=config.verification,
            )
            if any(
                out.resolve() == Path(name).resolve().parent
                or out.resolve() in Path(name).resolve().parents
                or Path(name).resolve().parent in out.resolve().parents
                for name in inputs.hashes
            ):
                raise ValueError("Output overlaps an immutable ancestor source")
            protocol: dict[str, Any] = {
                "config": asdict(config),
                "source_hashes": inputs.hashes,
                "source_identity": inputs.protocol["identity"],
                "source_model_sha256": inputs.original_model_sha256,
                "source_float32_model_sha256": inputs.float32_model_sha256,
                "code_identity": code_identity(),
                "command": shlex.join([sys.executable, *sys.argv]),
                "predeclared_arms": ["frozen_state", "live_state"],
                "hypothesis": (
                    "A factual TRAIN-only offset may repair benefit lost during "
                    "ordinary joint-state continuation with retention/fidelity preserved; "
                    "persistent response "
                    "errors cannot generally be repaired by a constant offset."
                ),
                "fit": (
                    "Six new zero-initialized offsets relative to each fixed110k endpoint. "
                    "Frozen E/P/state weights; "
                    "all8192 TRAIN groups x10 contexts x9 nonself donors x2 attached horizons x6 "
                    "states, exact factual forecast MSE, float64. No physical labels, validation, "
                    "counterfactual or distillation losses. Explicitly exclude SPARTAN logit "
                    "penalty: this is a post-fit operating-point diagnostic. Endpoint state charts "
                    "re-encode observed TRAIN in float32; E/P and fitting use float64. "
                    "No EMA; earlier source offsets are not reapplied."
                ),
                "optimizer": (
                    "Unchanged bounded LBFGS lr1/history10/strongWolfe,25accepted/50unique "
                    "full-evaluation caps, exact trial cache and last-accepted rollback, "
                    "gradienttol1e-9/changetol1e-12. Its12-slot implementation embeds six active "
                    "offsets plus six fixed zeros; inactive points/gradients/directions/curvature "
                    "histories must remain exactly zero. This has the same six-offset objective "
                    "and mathematical search geometry, without claiming bitwise equivalence to an "
                    "independently implemented6D reduction."
                ),
                "evaluation": (
                    "Only after fitting and composition are durable, replay original64 AND8192 "
                    "validation for each endpoint source and assess "
                    "source/calibrated models on original64 plus existing8192 validation. "
                    "Original1000 group-bootstrap seed and positive.01 interventions unchanged. No "
                    "test/probe fitting or reuse of the32768 confirmatory test."
                ),
                "decision": (
                    "Report both fixed endpoint calibrations regardless of either outcome. "
                    "All-six use supports a bias repair conditional on predictor accuracy; "
                    "persistent fidelity failures or nonpositive benefit leave that arm "
                    "unresolved. No model/seed/endpoint selection or extra budget."
                ),
                "checkpoint_loading_scope": (
                    "last.pt contains monitoring-derived records/raw; the loader discards "
                    "those tensors. No held-out INPUT CACHE is opened before fitting; the fitter "
                    "receives only fixed weights and observed TRAIN-derived encodings."
                ),
                "production_training_modified": False,
            }
            protocol["identity"] = canonical_fingerprint(
                {k: v for k, v in protocol.items() if k != "command"}
            )
            identity = protocol["identity"]
            if (out / "completed.json").exists():
                complete = authenticated_directory(
                    out,
                    {
                        "protocol.json",
                        "results.json",
                        "fit_state.pt",
                        "translation.pt",
                        "composed_model.pt",
                    },
                )
                if complete["identity"] != identity:
                    raise ValueError("Completed calibration identity changed")
                return cast(dict[str, Any], read_json(out / "results.json"))
            if (out / "protocol.json").exists():
                if not resume or read_json(out / "protocol.json")["identity"] != identity:
                    raise ValueError("Matching explicit resume required")
            elif any(path.name != ".lock" for path in out.iterdir()):
                raise FileExistsError("Use a fresh output directory or matching explicit resume")
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
                    name=f"joint-state-calibration-{config.arm}-{out.parent.name}",
                    job_type="diagnostic",
                ),
                resolved_config=protocol,
                dataset_identity=inputs.protocol["identity"],
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
                model = inputs.model.to(config.device)
                fixed = model_fingerprint(model)
                if (out / "train_encodings.pt").exists():
                    encoded_receipt = read_json(out / "encoding_receipt.json")
                    if (
                        encoded_receipt["identity"] != identity
                        or encoded_receipt["source_model_sha256"] != fixed
                        or encoded_receipt["sha256"] != sha(out / "train_encodings.pt")
                    ):
                        raise ValueError("Endpoint TRAIN encodings changed")
                    train = cast(
                        dict[str, Tensor],
                        torch.load(
                            out / "train_encodings.pt", map_location="cpu", weights_only=True
                        ),
                    )
                else:
                    train = encode_observations(
                        model,
                        inputs.observations,
                        chunk_size=config.chunk_size,
                        progress=lambda n: log({"progress/train_state_groups": float(n)}),
                    )
                    atomic_save(out / "train_encodings.pt", train)
                    write_json(
                        out / "encoding_receipt.json",
                        {
                            "identity": identity,
                            "sha256": sha(out / "train_encodings.pt"),
                            "source_model_sha256": fixed,
                            "factual_state_dtype": "float32",
                            "all_train_groups_reencoded": True,
                        },
                    )
                if (out / "train_codes.pt").exists():
                    receipt = read_json(out / "cache_receipt.json")
                    if receipt["identity"] != identity or receipt["sha256"] != sha(
                        out / "train_codes.pt"
                    ):
                        raise ValueError("TRAIN code cache changed")
                    payload = cast(
                        dict[str, Any],
                        torch.load(out / "train_codes.pt", map_location="cpu", weights_only=False),
                    )
                    if payload["identity"] != identity:
                        raise ValueError("TRAIN code payload identity changed")
                    codes = cast(Tensor, payload["codes"])
                else:
                    codes = encode_train(
                        model,
                        train,
                        config.chunk_size,
                        lambda n: log({"progress/train_groups": float(n)}),
                    )
                    atomic_save(out / "train_codes.pt", {"identity": identity, "codes": codes})
                    write_json(
                        out / "cache_receipt.json",
                        {"identity": identity, "sha256": sha(out / "train_codes.pt")},
                    )
                state = None
                if (out / "fit_state.pt").exists():
                    payload = cast(
                        dict[str, Any],
                        torch.load(out / "fit_state.pt", map_location="cpu", weights_only=False),
                    )
                    if payload["identity"] != identity:
                        raise ValueError("Accepted optimizer checkpoint identity changed")
                    state = payload["state"]
                    verify_embedding(state)

                def save(state: dict[str, Any]) -> None:
                    verify_embedding(state)
                    if model_fingerprint(model) != fixed:
                        raise RuntimeError("Calibration changed frozen source weights")
                    atomic_save(out / "fit_state.pt", {"identity": identity, "state": state})
                    metrics = {
                        "fit/full_evaluations": float(state["full_evaluations"]),
                        "fit/accepted_iterations": float(state["accepted_iterations"]),
                    }
                    if state["accepted_loss"] is not None:
                        metrics["fit/accepted_loss"] = float(state["accepted_loss"])
                        metrics["fit/gradient_infinity_norm"] = float(
                            state["accepted_gradient"].abs().max()
                        )
                    log(metrics)

                terminal_reasons = {
                    "gradient_tolerance",
                    "evaluation_budget_rollback",
                    "no_accepted_displacement",
                    "accepted_iteration_budget",
                }
                if state is not None and (
                    (state["max_iterations"], state["max_evaluations"])
                    != (config.max_iterations, config.max_evaluations)
                    or not 0 <= state["accepted_iterations"] <= config.max_iterations
                    or not 0 <= state["full_evaluations"] <= config.max_evaluations
                    or state["stop_reason"] not in terminal_reasons | {"running"}
                ):
                    raise ValueError("Saved calibration budget or phase differs")
                if state is not None and state["stop_reason"] in terminal_reasons:
                    # A completed fixed fit is immutable even if later evaluation was interrupted.
                    fitted = state
                    log({"fit/reused_terminal_state": 1.0})
                else:
                    fitted = fit(
                        EmbeddedObjective(TrainObjective(model, train, codes, config.chunk_size)),
                        device=torch.device(config.device),
                        max_iterations=config.max_iterations,
                        max_evaluations=config.max_evaluations,
                        state=state,
                        on_state=save,
                    )
                active = active_state(fitted)
                summary = compact_fit(active, codes)
                summary["active_parameters"] = 6
                summary["inactive_optimizer_slots_exactly_zero"] = True
                write_json(out / "fit.json", {"identity": identity, **summary})
                delta = cast(Tensor, active["translation"]).to(next(model.parameters()))
                candidate = compose_translation(model, delta)
                atomic_save(
                    out / "translation.pt",
                    {
                        "identity": identity,
                        "translation": delta.detach().cpu(),
                        "source_model_sha256": fixed,
                    },
                )
                atomic_save(
                    out / "composed_model.pt",
                    {
                        "identity": identity,
                        "model": candidate.state_dict(),
                        "source_model_sha256": fixed,
                        "composed_model_sha256": model_fingerprint(candidate),
                    },
                )
                # First access to any held-out input cache is below the completed fixed fit.
                evaluation = load_evaluation(
                    inputs,
                    Path(config.source_dir),
                    Path(config.precision_dir),
                    device=torch.device(config.device),
                    verification=config.verification,
                )
                reports: dict[str, Any] = {}
                measurements: dict[str, Any] = {}
                for name, data, bootstrap in (
                    (
                        "original64",
                        evaluation.original_validation,
                        evaluation.original_bootstrap_samples,
                    ),
                    (
                        "precision8192",
                        evaluation.precision_data,
                        evaluation.original_bootstrap_samples,
                    ),
                ):
                    for arm, evaluated in (("source", model), ("calibrated", candidate)):
                        report, raw, vectors = evaluate(
                            evaluated,
                            data,
                            chunk_size=config.chunk_size,
                            bootstrap_samples=bootstrap,
                            seed=evaluation.evaluation_seed,
                            progress=lambda n: log({"progress/evaluation_groups": float(n)}),
                        )
                        if arm == "source":
                            verify_replay(report, evaluation.archived_reports[name])
                        key = name + "/" + arm
                        reports[key], measurements[key] = report, {"raw": raw, "vectors": vectors}
                        metrics: dict[str, float] = {}
                        for row in report["per_parameter"]:
                            for metric in ("predictive_benefit", "normalized_response_error"):
                                prefix = f"{key}/theta_{row['parameter']}/{metric}"
                                metrics[prefix] = float(row[metric]["estimate"])
                                metrics[prefix + "/lower95"], metrics[prefix + "/upper95"] = row[
                                    metric
                                ]["interval_95"]
                        log(metrics)
                if model_fingerprint(model) != fixed or any(
                    p.requires_grad or p.grad is not None for p in model.parameters()
                ):
                    raise RuntimeError("Source model changed or accumulated parameter gradients")
                verify_files(inputs.hashes)
                result = {
                    "identity": identity,
                    "arm": config.arm,
                    "fit": summary,
                    "reports": reports,
                    "tracking": logger.metadata,
                    "scope": protocol["evaluation"],
                    "retention": (
                        "Invertible constant translation preserves code information; "
                        "no probes fitted or selected."
                    ),
                }
                write_json(out / "results.json", result)
                atomic_save(out / "measurements.pt", {"identity": identity, **measurements})
                for name in ("protocol.json", "results.json", "fit.json", "translation.pt"):
                    logger.log_artifact(
                        out / name, name=f"joint-state-calibration-{config.arm}-{Path(name).stem}"
                    )
                log({"diagnostic/complete": 1.0})
                names = (
                    "protocol.json",
                    "results.json",
                    "fit.json",
                    "fit_state.pt",
                    "train_codes.pt",
                    "cache_receipt.json",
                    "train_encodings.pt",
                    "encoding_receipt.json",
                    "translation.pt",
                    "composed_model.pt",
                    "measurements.pt",
                    "tracking.json",
                )
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
    """Expose the fixed six-offset diagnostic with explicit interruption recovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("endpoint-dir", "source-dir", "precision-dir", "output-dir", "checkpoint-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--arm", choices=("frozen_state", "live_state"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=16)
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
