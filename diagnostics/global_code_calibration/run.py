"""Fit only a global code translation, then assess fixed validation sets without selection."""

import argparse
import hashlib
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.global_code_calibration.objective import (
    TrainObjective,
    compose_translation,
    encode_train,
)
from diagnostics.global_code_calibration.optimization import fit
from diagnostics.global_code_calibration.source import load_inputs, verify_files
from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.overcomplete_use_precision.evaluation import evaluate
from diagnostics.overcomplete_use_precision.run import code_identity as precision_code_identity
from diagnostics.overcomplete_use_precision.run import verify_replay
from diagnostics.table_settling_control.source import authenticated_directory, model_fingerprint
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.training.logging import WandbConfig, WandbLogger


@dataclass(frozen=True)
class Config:
    """One predeclared calibration budget with no held-out endpoint selection."""

    endpoint_dir: str
    source_dir: str
    precision_dir: str
    output_dir: str
    device: str = "cpu"
    chunk_size: int = 64
    threads: int = 1
    max_iterations: int = 25
    max_evaluations: int = 50
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Keep production budgets fixed and all completed sources immutable."""
        if min(self.chunk_size, self.threads, self.max_iterations, self.max_evaluations) < 1:
            raise ValueError("Positive sizes and budgets required")
        if self.verification:
            if self.max_iterations > 3 or self.max_evaluations > 8:
                raise ValueError("Verification permits at most3iterations/8evaluations")
        elif (self.max_iterations, self.max_evaluations) != (25, 50):
            raise ValueError("Scientific calibration fixes25acceptediterations/50full evaluations")
        output = Path(self.output_dir).resolve()
        for name in (self.endpoint_dir, self.source_dir, self.precision_dir):
            source = Path(name).resolve()
            if output == source or output in source.parents or source in output.parents:
                raise ValueError("Output must be disjoint from immutable source directories")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def code_identity() -> str:
    """Bind this isolated calibration and every unchanged prediction/evaluation dependency."""
    digest = hashlib.sha256(precision_code_identity().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compact_fit(state: dict[str, Any], codes: Tensor) -> dict[str, Any]:
    """Separate accepted updates, unique full evaluations, stationarity and displacement."""
    delta = cast(Tensor, state["translation"])
    gradient = cast(Tensor, state["accepted_gradient"])
    scale = codes.flatten(0, 1).std(0, correction=0).clamp_min(1e-12)
    return {
        "accepted_iterations": state["accepted_iterations"],
        "full_evaluations": state["full_evaluations"],
        "closure_calls_including_cache_hits": state["closure_calls"],
        "stop_reason": state["stop_reason"],
        "converged_gradient": state["converged_gradient"],
        "initial_loss": state["initial_loss"],
        "accepted_loss": state["accepted_loss"],
        "initial_gradient": state["initial_gradient"].numpy().tolist(),
        "accepted_gradient": gradient.numpy().tolist(),
        "gradient_infinity_norm": float(gradient.abs().max()),
        "translation": delta.numpy().tolist(),
        "translation_l2": float(delta.square().sum().sqrt()),
        "translation_in_train_coordinate_std": (delta / scale).numpy().tolist(),
    }


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Calibrate on TRAIN only; deserialize held-out tensors only after fitting terminates."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    out = Path(config.output_dir)
    try:
        with file_lock(out / ".lock"):
            inputs = load_inputs(
                Path(config.endpoint_dir),
                Path(config.source_dir),
                Path(config.precision_dir),
                verification=config.verification,
            )
            protocol: dict[str, Any] = {
                "config": asdict(config),
                "source_hashes": inputs.hashes,
                "source_identity": inputs.protocol["identity"],
                "source_model_sha256": inputs.original_model_sha256,
                "code_identity": code_identity(),
                "command": shlex.join([sys.executable, *sys.argv]),
                "hypothesis": (
                    "A global code operating-point correction can reduce factual "
                    "residual bias; varying physical code directions may leave "
                    "cross-term bias despite mean translation-gradient "
                    "stationarity."
                ),
                "fit": (
                    "Only12sharedtranslationparameters, zero initialization, "
                    "float64model/input arithmetic, exactmean of all8192TRAINgrou "
                    "ps*10contexts*9nonselfdonors*2horizons*6states. "
                    "AllsourceE/P/stateweights frozen; cachedsingle-contextcodes; "
                    "both autoregressive steps attached. No theta, validation or "
                    "counterfactual losses."
                ),
                "optimizer": (
                    "DeterministicLBFGS lr1/history10/strongWolfe, "
                    "max_iter1peracceptedstep, gradienttol1e-9/changetol1e-12. "
                    "Maximum25acceptedupdates/50unique fullobjectiveevaluations; "
                    "exactpointcache. Budget interruption rolls back to "
                    "lastacceptedtranslation/optimizerstate."
                ),
                "evaluation": (
                    "Afterfixedfitonly: replayoriginal64validation "
                    "andpredeclared8192precisionvalidation, "
                    "thenevaluatecomposedbias onboth "
                    "withsameoriginalresponse/bootstrap. No "
                    "testinference/probes/endpointselection; "
                    "originalcompletedtestcriterionunchanged."
                ),
                "production_training_modified": False,
            }
            protocol["identity"] = canonical_fingerprint(
                {key: value for key, value in protocol.items() if key != "command"}
            )
            if (out / "completed.json").exists():
                complete = authenticated_directory(
                    out,
                    {
                        "protocol.json",
                        "results.json",
                        "fit_state.pt",
                        "train_codes.pt",
                        "cache_receipt.json",
                        "translation.pt",
                        "composed_model.pt",
                        "measurements.pt",
                    },
                )
                if complete["identity"] != protocol["identity"]:
                    raise ValueError("Completed calibration identity changed")
                return cast(dict[str, Any], read_json(out / "results.json"))
            if (out / "protocol.json").exists():
                if (
                    not resume
                    or read_json(out / "protocol.json")["identity"] != protocol["identity"]
                ):
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
                    name=f"global-code-calibration-{out.parent.name}",
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
                if (out / "train_codes.pt").exists():
                    receipt = read_json(out / "cache_receipt.json")
                    if receipt["identity"] != protocol["identity"] or receipt["sha256"] != sha(
                        out / "train_codes.pt"
                    ):
                        raise ValueError("TRAIN code cache changed")
                    cached_codes = cast(
                        dict[str, Any],
                        torch.load(out / "train_codes.pt", map_location="cpu", weights_only=False),
                    )
                    if cached_codes["identity"] != protocol["identity"]:
                        raise ValueError("TRAIN code payload identity changed")
                    codes = cast(Tensor, cached_codes["codes"])
                else:
                    codes = encode_train(
                        model,
                        inputs.train,
                        config.chunk_size,
                        lambda n: log({"progress/train_context_groups": float(n)}),
                    )
                    atomic_save(
                        out / "train_codes.pt", {"identity": protocol["identity"], "codes": codes}
                    )
                    write_json(
                        out / "cache_receipt.json",
                        {"identity": protocol["identity"], "sha256": sha(out / "train_codes.pt")},
                    )
                state = None
                if (out / "fit_state.pt").exists():
                    payload = cast(
                        dict[str, Any],
                        torch.load(out / "fit_state.pt", map_location="cpu", weights_only=False),
                    )
                    if payload["identity"] != protocol["identity"]:
                        raise ValueError("Accepted calibration checkpoint identity changed")
                    state = payload["state"]

                def save(state: dict[str, Any]) -> None:
                    if model_fingerprint(model) != fixed:
                        raise RuntimeError("Calibration modified frozen source model")
                    atomic_save(
                        out / "fit_state.pt", {"identity": protocol["identity"], "state": state}
                    )
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

                fitted = fit(
                    TrainObjective(model, inputs.train, codes, config.chunk_size),
                    device=torch.device(config.device),
                    max_iterations=config.max_iterations,
                    max_evaluations=config.max_evaluations,
                    state=state,
                    on_state=save,
                )
                summary = compact_fit(fitted, codes)
                write_json(out / "fit.json", {"identity": protocol["identity"], **summary})
                translation = cast(Tensor, fitted["translation"]).to(next(model.parameters()))
                candidate = compose_translation(model, translation)
                atomic_save(
                    out / "translation.pt",
                    {
                        "identity": protocol["identity"],
                        "translation": translation.detach().cpu(),
                        "source_model_sha256": fixed,
                    },
                )
                atomic_save(
                    out / "composed_model.pt",
                    {
                        "identity": protocol["identity"],
                        "model": candidate.state_dict(),
                        "source_model_sha256": fixed,
                        "composed_model_sha256": model_fingerprint(candidate),
                    },
                )
                # This is the first deserialization of any held-out arrays.
                old = cast(
                    EvaluationData,
                    torch.load(
                        Path(config.source_dir) / "evaluation_data.pt",
                        map_location="cpu",
                        weights_only=False,
                    ),
                )
                del old["test"]
                count = inputs.protocol["config"]["groups"]
                old64: EvaluationData = {
                    "metadata": old["metadata"],
                    "validation": {key: value[:count] for key, value in old["validation"].items()},
                }
                precision = cast(
                    EvaluationData,
                    torch.load(
                        Path(config.precision_dir) / "validation_data.pt",
                        map_location="cpu",
                        weights_only=False,
                    ),
                )
                precision_result = cast(
                    dict[str, Any], read_json(Path(config.precision_dir) / "results.json")
                )
                reports: dict[str, Any] = {}
                measurements: dict[str, Any] = {}
                for name, data, archived, bootstrap in (
                    (
                        "original64",
                        old64,
                        inputs.archived_validation,
                        inputs.protocol["config"]["bootstrap_samples"],
                    ),
                    (
                        "precision8192",
                        precision,
                        precision_result["report"],
                        inputs.precision_protocol["config"]["bootstrap_samples"],
                    ),
                ):
                    for arm, evaluated_model in (("source", model), ("calibrated", candidate)):
                        report, raw, vectors = evaluate(
                            evaluated_model,
                            data,
                            chunk_size=config.chunk_size,
                            bootstrap_samples=bootstrap,
                            seed=inputs.protocol["config"]["seed"],
                            progress=lambda n: log({"progress/evaluation_groups": float(n)}),
                        )
                        if arm == "source":
                            verify_replay(report, archived)
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
                    p.grad is not None or p.requires_grad for p in model.parameters()
                ):
                    raise RuntimeError("Source model changed or accumulated parameter gradients")
                verify_files(inputs.hashes)
                result = {
                    "identity": protocol["identity"],
                    "fit": summary,
                    "reports": reports,
                    "tracking": logger.metadata,
                    "scope": protocol["evaluation"],
                    "retention": (
                        "An invertible constant translation preserves information "
                        "and centered probe features; no probe fitted or "
                        "selected."
                    ),
                }
                write_json(out / "results.json", result)
                atomic_save(
                    out / "measurements.pt", {"identity": protocol["identity"], **measurements}
                )
                for name in ("protocol.json", "results.json", "fit.json", "translation.pt"):
                    logger.log_artifact(
                        out / name, name=f"global-code-calibration-{Path(name).stem}"
                    )
                log({"diagnostic/complete": 1.0})
                write_json(
                    out / "completed.json",
                    {
                        "identity": protocol["identity"],
                        "hashes": {
                            name: sha(out / name)
                            for name in (
                                "protocol.json",
                                "results.json",
                                "fit.json",
                                "fit_state.pt",
                                "train_codes.pt",
                                "cache_receipt.json",
                                "translation.pt",
                                "composed_model.pt",
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
    """Expose the fixed scientific calibration, with authenticated interruption recovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("endpoint-dir", "source-dir", "precision-dir", "output-dir"):
        parser.add_argument(f"--{name}", required=True)
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
