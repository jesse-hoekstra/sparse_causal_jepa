"""Continue the fixed live dense model and AdamW, then calibrate its fixed final endpoint."""

import argparse
import hashlib
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.global_code_calibration.source import verify_files
from diagnostics.group_latent_control.run import (
    _restore_rng,  # pyright: ignore[reportPrivateUsage]
    _rng,  # pyright: ignore[reportPrivateUsage]
)
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.joint_dense_continuation.calibration import fit_final
from diagnostics.joint_dense_continuation.source import CHECKPOINT_SHA256, load_inputs
from diagnostics.joint_state_calibration.run import code_identity as parent_code_identity
from diagnostics.joint_state_survival.data import refresh_evaluation
from diagnostics.joint_state_survival.evaluation import (
    monitor,
    paired_changes,
    precision_assessment,
)
from diagnostics.joint_state_survival.forward import after_update, training_loss
from diagnostics.mlp_acquisition_control.evaluation import measurement_metrics
from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.overcomplete_use_precision.run import verify_replay
from diagnostics.predictor_transfer_precision.evaluation import evaluate
from diagnostics.shared_confirmatory_control.run import code_identity as stage_code_identity
from diagnostics.table_settling_control.source import authenticated_directory, model_fingerprint
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.rng import preserve_rng


@dataclass(frozen=True)
class Config:
    """One fixed90k extension of the original live model, followed by one bounded offset fit."""

    endpoint_dir: str
    source_dir: str
    precision_dir: str
    output_dir: str
    checkpoint_sha256: str
    device: str = "cpu"
    steps: int = 90000
    seed: int = 202610261301
    groups: int = 64
    bootstrap_samples: int = 1000
    chunk_size: int = 16
    threads: int = 1
    max_iterations: int = 25
    max_evaluations: int = 50
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Fix the scientific source/budgets and keep every ancestor artifact immutable."""
        if len(self.checkpoint_sha256) != 64 or any(
            x not in "0123456789abcdef" for x in self.checkpoint_sha256
        ):
            raise ValueError("Require the full explicit source checkpoint SHA256 pin")
        if self.verification:
            if (
                not 1 <= self.steps <= 3
                or not 2 <= self.groups <= 4
                or not 1 <= self.max_iterations <= 3
                or not 1 <= self.max_evaluations <= 8
            ):
                raise ValueError(
                    "Verification permits1..3updates/2..4groups/at most3accepted8full fits"
                )
        else:
            if self.checkpoint_sha256 != CHECKPOINT_SHA256:
                raise ValueError("Require the pinned uncalibrated live110k source checkpoint")
            if (
                self.steps,
                self.seed,
                self.groups,
                self.bootstrap_samples,
                self.max_iterations,
                self.max_evaluations,
            ) != (90000, 202610261301, 64, 1000, 25, 50):
                raise ValueError("Scientific continuation fixes90k/64/1000 and25accepted50full fit")
        if min(self.chunk_size, self.threads) < 1 or self.bootstrap_samples < 2 or self.seed < 0:
            raise ValueError("Require positive execution settings")
        output = Path(self.output_dir).resolve()
        for name in (self.endpoint_dir, self.source_dir, self.precision_dir):
            source = Path(name).resolve()
            if output == source or output in source.parents or source in output.parents:
                raise ValueError("Output must be disjoint from immutable source directories")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def code_identity() -> str:
    """Bind the new adapter and the unchanged training/calibration/evaluation dependencies."""
    digest = hashlib.sha256((parent_code_identity() + stage_code_identity()).encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Preserve the original dense trajectory; fit only TRAIN before fixed final inference."""
    device, out = torch.device(config.device), Path(config.output_dir)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        inputs = load_inputs(
            Path(config.endpoint_dir),
            Path(config.source_dir),
            Path(config.precision_dir),
            checkpoint_sha256=config.checkpoint_sha256,
            device=device,
            verification=config.verification,
        )
        train = inputs.observations
        train_groups = len(train["observations"])
        if inputs.seed != config.seed or (
            not config.verification
            and (inputs.source_step, inputs.optimizer_step, train_groups, inputs.batch_size)
            != (110000, 10000, 8192, 16)
        ):
            raise ValueError("Require original seed/fixed110k/10000AdamW updates/8192TRAIN/B16")
        if any(
            out.resolve() == Path(name).resolve().parent
            or out.resolve() in Path(name).resolve().parents
            or Path(name).resolve().parent in out.resolve().parents
            for name in inputs.hashes
        ):
            raise ValueError("Output overlaps an immutable ancestor source")
        with file_lock(out / ".lock"):
            model, optimizer = inputs.model, inputs.optimizer
            parameters = [p for p in model.parameters() if p.requires_grad]
            initial_hash = model_fingerprint(model)
            source_step = inputs.source_step
            final_step = source_step + config.steps
            selection = DiagnosticSelection(
                tuple(range(config.groups)), tuple(range(config.groups))
            )
            settings = asdict(config)
            for key in (
                "endpoint_dir",
                "source_dir",
                "precision_dir",
                "output_dir",
                "wandb_mode",
                "wandb_entity",
            ):
                settings.pop(key)
            evidence = {
                "settings": settings,
                "source_hashes": inputs.hashes,
                "source_provenance": inputs.provenance,
                "initial_model_sha256": initial_hash,
                "code_identity": code_identity(),
                "torch": str(torch.__version__),
            }
            identity = canonical_fingerprint(evidence)
            protocol: dict[str, Any] = {
                "identity": identity,
                **evidence,
                "config": asdict(config),
                "command": shlex.join([sys.executable, *sys.argv]),
                "hypothesis": (
                    "Ordinary live dense training may preserve acquired "
                    "information/fidelity and improve forecast stability over100k total "
                    "joint updates, followed by the same bounded TRAIN-only offset "
                    "procedure."
                ),
                "controlled_change": (
                    "Only training duration: original UNCALIBRATED live110k endpoint "
                    "to200k,90000additional updates. Preserve model, target chart, exact "
                    "AdamW moments/counters, parameter order and global batch/donor "
                    "stream. No source calibration offset, reset, architecture/loss "
                    "change or sweep."
                ),
                "training": (
                    "Original full ordinary forecast "
                    "+.4stateSIGReg+.4parameterSIGReg+consistency1+.0001logit/256projections; "
                    "all-open dense/noGECO. Train online/E/P, frozen target with EMA.996 "
                    "once per update; AdamW.001/weight_decay0/clip1. Observations/actions "
                    "only, original8192/B16/R10/K10/T12, globalsteps110001..200000; "
                    "attached donors and two autoregressive steps."
                ),
                "calibration": (
                    "After durable uncalibrated final last.pt: freeze final E/P/state, "
                    "EMAoff, new zero6offset; float32 current-state TRAIN encodings then "
                    "double E/P fitting. Unchanged factual all8192x10x9nonselfx2 "
                    "objective and bounded25accepted/50full evaluations. Reuse exact "
                    "cached/rollback optimizer with six dormant zero slots. No labels or "
                    "held-out inputs in fitter; prior64validation monitoring exists and "
                    "is never used by fitting or stopping. Terminal fit must be durable "
                    "before final test/precision inference."
                ),
                "evaluation": (
                    "Replay source0 original64validation. Monitor every5000updates; "
                    "fixedfinal raw originalfull512/256/512retention "
                    "and64test/currentchartMCC only after fitting. Raw and calibrated "
                    "original64validation plus existing8192validation responses. "
                    "Original512cal/256val probe selection only; no calibrated probe "
                    "refit or test inference. No32768reuse/newdata. Current state "
                    "encoders re-encode every physical trajectory; no stale state cache."
                ),
                "interpretation": (
                    "Earlier source all-six use remains unmet. Retention/fidelity "
                    "survival tests dense stability; fixed final calibration all-six "
                    "benefit/fidelity passes would support this staged procedure "
                    "conditionally. Negative benefit rejects a factor's use; an interval "
                    "crossing zero remains unresolved. All-six success still requires "
                    "R2>=.8, benefit lower95>0 and response-error upper95<1. StateMCC "
                    "improvement alone is neither sparsity evidence nor full Experiment1 "
                    "success."
                ),
                "source_step": source_step,
                "final_global_step": final_step,
                "source_optimizer_step": inputs.optimizer_step,
                "production_training_modified": False,
                "training_parameter_labels_absent": True,
            }
            if (out / "protocol.json").exists():
                if not resume or read_json(out / "protocol.json")["identity"] != identity:
                    raise ValueError(
                        "Matching explicit resume required; settings/source/code identity changed"
                    )
            elif any(path.name != ".lock" for path in out.iterdir()):
                raise FileExistsError("Use a fresh output or matching explicit resume")
            else:
                write_json(out / "protocol.json", protocol)
                atomic_save(
                    out / "initial_model.pt",
                    {
                        "identity": identity,
                        "model_sha256": initial_hash,
                        "model": model.state_dict(),
                    },
                )
            if (out / "completed.json").exists():
                complete = authenticated_directory(
                    out,
                    {
                        "protocol.json",
                        "results.json",
                        "initial_model.pt",
                        "last.pt",
                        "endpoint_assessment.json",
                        "endpoint_assessment.pt",
                        "calibration/completed.json",
                        "calibration/fit_state.pt",
                        "calibration/composed_model.pt",
                    },
                )
                if complete["identity"] != identity:
                    raise ValueError("Completed continuation identity changed")
                return cast(dict[str, Any], read_json(out / "results.json"))
            initial = cast(
                dict[str, Any],
                torch.load(out / "initial_model.pt", map_location="cpu", weights_only=False),
            )
            if initial["identity"] != identity or initial["model_sha256"] != initial_hash:
                raise ValueError("Persisted initial model identity differs")
            model.load_state_dict(initial["model"], strict=True)
            if model_fingerprint(model) != initial_hash:
                raise ValueError("Persisted initial model weights differ")
            initial_sha = sha(out / "initial_model.pt")
            records: list[dict[str, Any]] = []
            raw: dict[int, Any] = {}
            start = 0
            tracking = (
                read_json(out / "tracking.json") if (out / "tracking.json").exists() else None
            )
            _restore_rng(inputs.rng, device)
            if resume and (out / "last.pt").exists():
                receipt = read_json(out / "last.receipt.json")
                if receipt["identity"] != identity or receipt["sha256"] != sha(out / "last.pt"):
                    raise ValueError("Saved continuation checkpoint hash or identity changed")
                checkpoint = cast(
                    dict[str, Any],
                    torch.load(out / "last.pt", map_location="cpu", weights_only=False),
                )
                if checkpoint["identity"] != identity:
                    raise ValueError("Checkpoint identity differs")
                start = int(checkpoint["updates"])
                if (
                    checkpoint["step"] != source_step + start
                    or checkpoint["optimizer_steps"] != inputs.optimizer_step + start
                ):
                    raise ValueError("Checkpoint global/AdamW counters differ")
                model.load_state_dict(checkpoint["model"], strict=True)
                optimizer.load_state_dict(checkpoint["optimizer"])
                records, raw = checkpoint["records"], checkpoint["raw"]
                tracking = checkpoint["tracking"]
                _restore_rng(checkpoint["rng"], device)
            if not 0 <= start <= config.steps:
                raise ValueError("Invalid completed continuation count")
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=out.parent.name,
                    job_type="diagnostic",
                ),
                resolved_config=protocol,
                dataset_identity=inputs.provenance["source_identity"],
                out_dir=out,
                resume_state=tracking if tracking and tracking.get("id") else None,
            )
            write_json(out / "tracking.json", logger.metadata)

            def check_state(updates: int) -> None:
                if (
                    sha(out / "initial_model.pt") != initial_sha
                    or model.get_extra_state() != inputs.provenance["rollout_flags"]
                ):
                    raise RuntimeError("Initial artifact or ordinary rollout flags changed")
                if any(
                    p.requires_grad or p.grad is not None for p in model.target_encoder.parameters()
                ):
                    raise RuntimeError("Target encoder acquired gradients")
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise FloatingPointError("Nonfinite dense model weights")
                state = optimizer.state_dict()
                if state["param_groups"] != [inputs.provenance["optimizer_group"]] or set(
                    state["state"]
                ) != set(range(len(parameters))):
                    raise ValueError("AdamW parameter group/order or moment coverage changed")
                if any(
                    float(row["step"]) != inputs.optimizer_step + updates
                    for row in optimizer.state.values()
                ):
                    raise ValueError(
                        "AdamW moment counter differs from uninterrupted joint history"
                    )
                for parameter in parameters:
                    for name in ("exp_avg", "exp_avg_sq"):
                        moment = optimizer.state[parameter][name]
                        if (
                            moment.shape != parameter.shape
                            or moment.dtype != torch.float32
                            or not bool(torch.isfinite(moment).all())
                            or (name == "exp_avg_sq" and bool((moment < 0).any()))
                        ):
                            raise FloatingPointError("Invalid continued AdamW moments")

            def save(updates: int) -> None:
                check_state(updates)
                atomic_save(
                    out / "last.pt",
                    {
                        "identity": identity,
                        "step": source_step + updates,
                        "updates": updates,
                        "optimizer_steps": inputs.optimizer_step + updates,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng": _rng(device),
                        "records": records,
                        "raw": raw,
                        "tracking": logger.metadata,
                    },
                )
                write_json(
                    out / "last.receipt.json",
                    {
                        "identity": identity,
                        "sha256": sha(out / "last.pt"),
                        "step": source_step + updates,
                        "updates": updates,
                    },
                )

            def validate(updates: int) -> None:
                check_state(updates)
                report, measurements = monitor(
                    model,
                    inputs.evaluation,
                    selection,
                    bootstrap_samples=config.bootstrap_samples,
                    seed=config.seed,
                )
                if updates == 0:
                    verify_replay(report["response"], inputs.archived_validation["response"])
                    torch.testing.assert_close(
                        torch.tensor(report["retention"]["selected_validation_r2"]),
                        torch.tensor(
                            inputs.archived_validation["retention"]["selected_validation_r2"]
                        ),
                        rtol=1e-6,
                        atol=1e-8,
                    )
                    torch.testing.assert_close(
                        torch.tensor(report["state"]["mcc"]),
                        torch.tensor(inputs.archived_validation["state"]["mcc"]),
                        rtol=1e-6,
                        atol=1e-8,
                    )
                records.append({"step": source_step + updates, "updates": updates, **report})
                raw[source_step + updates] = measurements
                logger.log(
                    source_step + updates,
                    {
                        **measurement_metrics(report),
                        "validation/state_mcc": report["state"]["mcc"],
                        "validation/online_target_mse": report["state"]["online_target_mse"],
                    },
                )
                write_json(out / "validation.json", {"identity": identity, "records": records})

            def log_final(metrics: dict[str, float]) -> None:
                logger.log(final_step, metrics)
                print({"step": final_step, **metrics}, flush=True)

            try:
                check_state(start)
                if not records:
                    print({"phase": "source_original64_replay", "step": source_step}, flush=True)
                    validate(0)
                    save(0)
                loaded_epoch, batches = -1, []
                for update in range(start + 1, config.steps + 1):
                    global_step = source_step + update
                    epoch, position = divmod(global_step - 1, train_groups // inputs.batch_size)
                    if loaded_epoch != epoch:
                        batches = list(
                            GlobalBatchSampler(train_groups, inputs.batch_size, config.seed, epoch)
                        )
                        loaded_epoch = epoch
                    batch = {
                        key: value[batches[position]].to(device) for key, value in train.items()
                    }
                    torch.manual_seed(config.seed * 100000 + global_step)  # pyright: ignore[reportUnknownMemberType]
                    optimizer.zero_grad(set_to_none=True)
                    loss, terms = training_loss(model, batch, inputs.objective, "live_state")
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"Nonfinite loss at global step{global_step}")
                    loss.backward()  # pyright: ignore[reportUnknownMemberType]
                    norm = float(
                        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    )
                    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
                    after_update(model, "live_state")
                    if update == 1 or update % 100 == 0 or update == config.steps:
                        print(
                            {
                                "step": global_step,
                                "updates": update,
                                "loss": float(loss.detach()),
                                "gradient_norm": norm,
                            },
                            flush=True,
                        )
                        logger.log(
                            global_step,
                            {
                                **{
                                    f"train/{name}": float(value.detach())
                                    for name, value in terms.items()
                                },
                                "train/gradient_norm": norm,
                                "train/joint_optimizer_steps": float(
                                    inputs.optimizer_step + update
                                ),
                            },
                        )
                    if config.verification or update % 5000 == 0 or update == config.steps:
                        validate(update)
                        save(update)
                # Preserve raw last.pt and its training RNG/optimizer throughout post-fit work.
                checkpoint_sha = sha(out / "last.pt")
                print({"phase": "fixed_final_train_calibration", "step": final_step}, flush=True)
                with preserve_rng(device):
                    source, candidate, fit_summary, calibration_names = fit_final(
                        model,
                        train,
                        out / "calibration",
                        identity=identity,
                        checkpoint_sha256=checkpoint_sha,
                        chunk_size=config.chunk_size,
                        max_iterations=config.max_iterations,
                        max_evaluations=config.max_evaluations,
                        log=log_final,
                    )
                final_report: dict[str, Any]
                if (out / "endpoint_assessment.json").exists():
                    receipt = read_json(out / "endpoint_assessment.json")
                    if (
                        receipt["identity"] != identity
                        or receipt["checkpoint_sha256"] != checkpoint_sha
                        or receipt["raw_sha256"] != sha(out / "endpoint_assessment.pt")
                    ):
                        raise ValueError("Fixed endpoint assessment changed")
                    final_report = cast(dict[str, Any], receipt["metrics"])
                else:
                    print({"phase": "fixed_final_evaluation", "step": final_step}, flush=True)
                    final_test, test_raw = monitor(
                        model,
                        inputs.evaluation,
                        selection,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                        final=True,
                    )
                    final_precision, final_raw = precision_assessment(
                        model,
                        inputs.evaluation,
                        inputs.precision_data,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                        chunk_size=config.chunk_size,
                        progress=lambda n: log_final(
                            {"progress/endpoint_precision_groups": float(n)}
                        ),
                    )
                    reports: dict[str, Any] = {}
                    measurements: dict[str, Any] = {}
                    with preserve_rng(device):
                        original = refresh_evaluation(
                            model,
                            {
                                "metadata": inputs.evaluation["metadata"],
                                "validation": {
                                    key: value[: config.groups]
                                    for key, value in inputs.evaluation["validation"].items()
                                },
                            },
                            ("validation",),
                        )
                        precision: EvaluationData = refresh_evaluation(
                            model, inputs.precision_data, ("validation",)
                        )
                        for name, data in (("original64", original), ("precision8192", precision)):
                            if name == "precision8192":
                                reports[name + "/source"] = final_precision["response"]
                                measurements[name + "/source"] = {
                                    "raw": final_raw["response"],
                                    "vectors": final_raw["vectors"],
                                }
                            for label, evaluated in (("source", source), ("calibrated", candidate)):
                                if name == "precision8192" and label == "source":
                                    continue
                                report, measured, vectors = evaluate(
                                    evaluated,
                                    data,
                                    chunk_size=config.chunk_size,
                                    bootstrap_samples=config.bootstrap_samples,
                                    seed=config.seed,
                                    progress=lambda n: log_final(
                                        {"progress/calibrated_validation_groups": float(n)}
                                    ),
                                )
                                if label == "source":
                                    verify_replay(report, records[-1]["response"])
                                reports[name + "/" + label] = report
                                measurements[name + "/" + label] = {
                                    "raw": measured,
                                    "vectors": vectors,
                                }
                    previous_raw = cast(
                        dict[str, Any],
                        torch.load(
                            Path(config.endpoint_dir) / "endpoint_assessment.pt",
                            map_location="cpu",
                            weights_only=False,
                        ),
                    )["precision"]
                    previous_state = inputs.archived_endpoint["final_test"]["state"]
                    original_state = inputs.archived_endpoint["source_test_state"]
                    final_report = {
                        "precision_validation": final_precision,
                        "final_test": final_test,
                        "source_test_state": previous_state,
                        "original_source_test_state": original_state,
                        "state_test_mcc_change": final_test["state"]["mcc"] - previous_state["mcc"],
                        "state_test_mcc_change_from_original": final_test["state"]["mcc"]
                        - original_state["mcc"],
                        "paired_precision_validation_change": paired_changes(
                            previous_raw,
                            final_raw,
                            bootstrap_samples=config.bootstrap_samples,
                            seed=config.seed,
                        ),
                        "calibration": {
                            "fit": fit_summary,
                            "reports": reports,
                            "retention": (
                                "Invertible translation preserves complete-code "
                                "information; no calibrated probes selected or test "
                                "inference performed."
                            ),
                        },
                    }
                    atomic_save(
                        out / "endpoint_assessment.pt",
                        {"precision": final_raw, "test": test_raw, "calibration": measurements},
                    )
                    write_json(
                        out / "endpoint_assessment.json",
                        {
                            "identity": identity,
                            "checkpoint_sha256": checkpoint_sha,
                            "raw_sha256": sha(out / "endpoint_assessment.pt"),
                            "metrics": final_report,
                        },
                    )
                check_state(config.steps)
                verify_files(inputs.hashes)
                if sha(out / "last.pt") != checkpoint_sha:
                    raise RuntimeError(
                        "Post-fit evaluation changed the raw final training checkpoint"
                    )
                result: dict[str, Any] = {
                    "identity": identity,
                    "step": final_step,
                    "updates": config.steps,
                    "optimizer_steps": inputs.optimizer_step + config.steps,
                    "records": records,
                    "endpoint": final_report,
                    "tracking": logger.metadata,
                    "scope": protocol["interpretation"],
                }
                write_json(out / "results.json", result)
                logger.log(
                    final_step,
                    {
                        **measurement_metrics(final_report["final_test"], final=True),
                        "test/state_mcc": final_report["final_test"]["state"]["mcc"],
                        "test/state_mcc_change": final_report["state_test_mcc_change"],
                        "diagnostic/complete": 1.0,
                    },
                )
                for name in ("protocol.json", "results.json"):
                    logger.log_artifact(
                        out / name,
                        name=f"joint-dense-continuation-{identity[:12]}-{Path(name).stem}",
                    )
                names = [
                    "protocol.json",
                    "initial_model.pt",
                    "last.pt",
                    "last.receipt.json",
                    "validation.json",
                    "endpoint_assessment.json",
                    "endpoint_assessment.pt",
                    "results.json",
                    "tracking.json",
                    *calibration_names,
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
    """Expose the pinned duration extension and explicit interrupted-run recovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("endpoint-dir", "source-dir", "precision-dir", "output-dir", "checkpoint-sha256"):
        parser.add_argument("--" + name, required=True)
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
