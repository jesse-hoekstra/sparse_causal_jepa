"""Conditionally test acquired-code survival under the original moving-state training package."""

import argparse
import copy
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
from diagnostics.joint_state_survival.data import check_train, refresh_evaluation, regenerate_train
from diagnostics.joint_state_survival.evaluation import (
    monitor,
    paired_changes,
    precision_assessment,
    state_metrics,
)
from diagnostics.joint_state_survival.forward import (
    Arm,
    after_update,
    configure_model,
    ordinary_objective,
    training_loss,
)
from diagnostics.joint_state_survival.source import (
    SourceGateMode,
    admission_gate,
    load_inputs,
    response_status,
    validate_source_mode,
)
from diagnostics.mlp_acquisition_control.evaluation import measurement_metrics
from diagnostics.multiquery_control.data import TensorSplit, module_fingerprint
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.predictor_transfer_calibration.run import code_identity as parent_code_identity
from diagnostics.table_settling_control.source import model_fingerprint
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng


@dataclass(frozen=True)
class Config:
    """Fixed10k paired state policies from one explicitly pinned calibrated SPARTAN endpoint."""

    source_dir: str
    endpoint_dir: str
    calibration_dir: str
    precision_dir: str
    output_dir: str
    checkpoint_sha256: str
    calibrated_sha256: str
    arm: Arm
    training_source_root: str | None = None
    data_record: str | None = None
    source_gate_mode: SourceGateMode = "all_six_use"
    seed: int = 202610261301
    device: str = "cpu"
    steps: int = 10000
    groups: int = 64
    bootstrap_samples: int = 1000
    chunk_size: int = 16
    threads: int = 1
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Keep source artifacts disjoint and prevent budget or source-policy selection."""
        validate_source_mode(
            self.source_gate_mode,
            self.checkpoint_sha256,
            self.calibrated_sha256,
            verification=self.verification,
        )
        if self.arm not in ("frozen_state", "live_state"):
            raise ValueError("Choose frozen_state or live_state")
        for value in (self.checkpoint_sha256, self.calibrated_sha256):
            if len(value) != 64 or any(x not in "0123456789abcdef" for x in value):
                raise ValueError("Require full explicit source checkpoint SHA256 pins")
        if self.verification:
            if not 1 <= self.steps <= 3 or not 2 <= self.groups <= 4:
                raise ValueError("Verification permits at most3 updates and2..4 groups")
        elif (self.steps, self.groups, self.bootstrap_samples, self.seed) != (
            10000,
            64,
            1000,
            202610261301,
        ):
            raise ValueError(
                "Scientific survival fixes10k/64response groups/1000bootstrap/canonical seed"
            )
        if min(self.threads, self.chunk_size) < 1 or self.bootstrap_samples < 2 or self.seed < 0:
            raise ValueError("Positive execution settings required")
        output = Path(self.output_dir).resolve()
        roots = [self.source_dir, self.endpoint_dir, self.calibration_dir, self.precision_dir]
        if self.training_source_root:
            roots.append(self.training_source_root)
        if self.data_record:
            roots.append(str(Path(self.data_record).parent))
        for name in roots:
            source = Path(name).resolve()
            if output == source or output in source.parents or source in output.parents:
                raise ValueError("Output must be disjoint from immutable source directories")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def code_identity() -> str:
    """Bind this independent continuation to unchanged calibration and ordinary training code."""
    digest = hashlib.sha256(parent_code_identity().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Authenticate the conditional source, train label-free, and test only the fixed endpoint."""
    device, out = torch.device(config.device), Path(config.output_dir)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        inputs = load_inputs(
            Path(config.source_dir),
            Path(config.endpoint_dir),
            Path(config.calibration_dir),
            Path(config.precision_dir),
            checkpoint_sha256=config.checkpoint_sha256,
            calibrated_sha256=config.calibrated_sha256,
            device=device,
            training_source_root=Path(config.training_source_root)
            if config.training_source_root
            else None,
            data_record=Path(config.data_record) if config.data_record else None,
            verification=config.verification,
            source_gate_mode=config.source_gate_mode,
        )
        bundle, source_step = inputs.bundle, inputs.source_step
        if bundle.seed != config.seed or (
            not config.verification
            and (source_step, bundle.train_groups, bundle.batch_size) != (100000, 8192, 16)
        ):
            raise ValueError("Require original seed/fixed100k source/8192TRAIN/B16")
        if any(
            out.resolve() == Path(name).resolve().parent
            or out.resolve() in Path(name).resolve().parents
            or Path(name).resolve().parent in out.resolve().parents
            for name in inputs.hashes
        ):
            raise ValueError("Output overlaps an immutable ancestor source")
        with file_lock(out / ".lock"):
            model = copy.deepcopy(inputs.model)
            configure_model(model, config.arm)
            initial_hash = model_fingerprint(model)
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.0)
            objective = ordinary_objective(
                ObjectiveConfig(**bundle.protocol["objective_config"])
            ).to(device)
            selection = DiagnosticSelection(
                tuple(range(config.groups)), tuple(range(config.groups))
            )
            settings = asdict(config)
            for key in (
                "source_dir",
                "endpoint_dir",
                "calibration_dir",
                "precision_dir",
                "output_dir",
                "training_source_root",
                "data_record",
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
                    "An acquired six-code SPARTAN model may retain its information and use "
                    "when original state learning/EMA resumes."
                ),
                "controlled_change": (
                    "Same calibrated E/P and distinct online/target initial weights; "
                    "frozen_state trains E/P with both state maps fixed and EMA off; "
                    "live_state additionally trains online state and applies original "
                    "EMA.996 after each accepted update. Both use full ordinary forecast + "
                    ".4 state SIGReg + .4 parameter SIGReg + consistency1 + .0001 "
                    "logit,256projections, all-open dense/noGECO."
                ),
                "source_gate_mode": config.source_gate_mode,
                "archived_response_status": inputs.provenance["archived_response_status"],
                "source_precondition": (
                    "Strict all_six_use mode retains the original six benefit/fidelity gates. "
                    "Prospective qualified_spartan mode is restricted to the specifically pinned "
                    "calibrated SPARTAN source: all-six fidelity, positive benefits1/3/4/5/6, "
                    "and positive theta2 estimate with lower95<=0<upper95, exactly the recorded "
                    "inconclusive pattern. "
                    "Both modes require fresh full512cal/256val retention R2>=.8 for every factor "
                    "and source0 re-encoded response admission before any update. The original "
                    "all-six gate and unmet benefit remain recorded. No source0 test prediction."
                ),
                "training": (
                    "Fresh matched AdamW.001/weight_decay0/clip1; E/P trained in both; "
                    "original8192 groups/B16/R10/K10/T12; original global "
                    "batch/donor/update indices100001..110000. Reconstructed "
                    "observations/actions only, authenticated against original TRAIN; no "
                    "labels, inverse maps, teacher loss, optimizer transplant or target "
                    "reset."
                ),
                "evaluation": (
                    "Original64cal/validation monitoring every1000; same "
                    "existing8192precisionVALIDATION physical trajectories re-encoded at "
                    "source0 and fixed10k, probes fit512cal/select256val only. Fixed10k "
                    "also receives original512/256/512retention and64testresponses. "
                    "Source0/final state MCC use same testgroups with calibration-only "
                    "assignments, both computed only after10k. No fresh32768 reuse; no "
                    "checkpoint selection/stopping."
                ),
                "interpretation": (
                    "Both arms retaining all six factors and response fidelity supports short-term "
                    "compatibility. Deterioration only in live_state implicates state learning/EMA "
                    "as a package; deterioration in both implicates the restored ordinary "
                    "E/P objective first. Compare benefit changes factorwise with the source and "
                    "paired intervals. Qualified admission does not establish all-six use: theta2 "
                    "was already inconclusive, so continued failure cannot show use forgetting. "
                    "A confidence bound crossing zero is not evidence of a negative change. "
                    "Final success still requires every factor's R2>=.8, benefit lower95>0 and "
                    "response-error upper95<1. No sparsity or fullExperiment1 success claim."
                ),
                "production_training_modified": False,
                "training_parameter_labels_absent": True,
                "source_step": source_step,
                "final_global_step": source_step + config.steps,
            }
            if (out / "protocol.json").exists():
                if not resume or read_json(out / "protocol.json")["identity"] != identity:
                    raise ValueError(
                        "Matching explicit resume required; settings/source/code changed"
                    )
            elif any(path.name != ".lock" for path in out.iterdir()):
                raise FileExistsError("Use a fresh output or matching explicit resume")
            else:
                write_json(out / "protocol.json", protocol)
                atomic_save(
                    out / "initial_model.pt", {"identity": identity, "model": model.state_dict()}
                )
            initial = cast(
                dict[str, Any],
                torch.load(out / "initial_model.pt", map_location="cpu", weights_only=False),
            )
            restored = copy.deepcopy(inputs.model)
            restored.load_state_dict(initial["model"], strict=True)
            if initial["identity"] != identity or model_fingerprint(restored) != initial_hash:
                raise ValueError("Persisted initial model differs")
            initial_sha = sha(out / "initial_model.pt")
            if (out / "completed.json").exists():
                completed = cast(dict[str, Any], read_json(out / "completed.json"))
                if completed["identity"] != identity or any(
                    sha(out / name) != value for name, value in completed["hashes"].items()
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
                start, records, raw, tracking = (
                    checkpoint["updates"],
                    checkpoint["records"],
                    checkpoint["raw"],
                    checkpoint["tracking"],
                )
                if checkpoint["step"] != source_step + start:
                    raise ValueError("Checkpoint global counter differs")
                _restore_rng(checkpoint["rng"], device)
            if not 0 <= start <= config.steps:
                raise ValueError("Invalid completed update count")
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
                resume_state=tracking if tracking and tracking.get("id") else None,
            )
            write_json(out / "tracking.json", logger.metadata)
            try:
                print({"phase": "prepare_observed_train", "arm": config.arm}, flush=True)
                if (out / "train_observations.pt").exists():
                    receipt = read_json(out / "train_receipt.json")
                    if receipt["identity"] != identity or receipt["sha256"] != sha(
                        out / "train_observations.pt"
                    ):
                        raise ValueError("Observed TRAIN cache changed")
                    train = cast(
                        TensorSplit,
                        torch.load(
                            out / "train_observations.pt", map_location="cpu", weights_only=True
                        ),
                    )
                else:
                    train, receipt = regenerate_train(
                        inputs.model,
                        bundle.train,
                        bundle.evaluation["metadata"],
                        inputs.data_config,
                        progress=lambda n: logger.log(
                            source_step, {"progress/regenerated_train_groups": float(n)}
                        ),
                    )
                    atomic_save(out / "train_observations.pt", train)
                    write_json(
                        out / "train_receipt.json",
                        {
                            "identity": identity,
                            "sha256": sha(out / "train_observations.pt"),
                            **receipt,
                        },
                    )
                check_train(train)
                if len(train["observations"]) != bundle.train_groups:
                    raise ValueError("Observed TRAIN cache group count differs")
                train_sha = sha(out / "train_observations.pt")

                def check_state() -> None:
                    if (
                        sha(out / "initial_model.pt") != initial_sha
                        or model.get_extra_state() != inputs.provenance["rollout_flags"]
                    ):
                        raise RuntimeError("Initial artifact or original rollout flags changed")
                    if any(
                        p.requires_grad or p.grad is not None
                        for p in model.target_encoder.parameters()
                    ):
                        raise RuntimeError("Target encoder acquired gradients")
                    if config.arm == "frozen_state":
                        for name, expected in inputs.provenance["initial_state_hashes"].items():
                            block = getattr(model, name)
                            if module_fingerprint(block) != expected or any(
                                p.requires_grad or p.grad is not None for p in block.parameters()
                            ):
                                raise RuntimeError("Frozen state chart changed")

                def save(step: int) -> None:
                    check_state()
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
                    check_state()
                    report, measurements = monitor(
                        model,
                        bundle.evaluation,
                        selection,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                    )
                    records.append({"step": source_step + step, "updates": step, **report})
                    raw[source_step + step] = measurements
                    logger.log(
                        source_step + step,
                        {
                            **measurement_metrics(report),
                            "validation/state_mcc": report["state"]["mcc"],
                            "validation/online_target_mse": report["state"]["online_target_mse"],
                        },
                    )
                    write_json(out / "validation.json", {"identity": identity, "records": records})

                print({"phase": "source_precision_validation", "arm": config.arm}, flush=True)
                if (out / "source_assessment.json").exists():
                    source_record = cast(dict[str, Any], read_json(out / "source_assessment.json"))
                    if source_record["identity"] != identity or source_record["raw_sha256"] != sha(
                        out / "source_assessment.pt"
                    ):
                        raise ValueError("Source assessment changed")
                    source_report = source_record["metrics"]
                    source_raw = cast(
                        dict[str, Any],
                        torch.load(
                            out / "source_assessment.pt", map_location="cpu", weights_only=False
                        ),
                    )
                else:
                    source_report, source_raw = precision_assessment(
                        inputs.model,
                        bundle.evaluation,
                        inputs.precision_data,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                        chunk_size=config.chunk_size,
                        progress=lambda n: logger.log(
                            source_step, {"progress/source_precision_groups": float(n)}
                        ),
                    )
                    atomic_save(out / "source_assessment.pt", source_raw)
                admission: dict[str, Any] = {
                    "mode": config.source_gate_mode,
                    "passed": source_report["original_full_validation_retention"][
                        "all_six_above_0_80"
                    ]
                    and admission_gate(source_report["response"], config.source_gate_mode),
                    **response_status(source_report["response"]),
                }
                source_report["admission"] = admission
                write_json(
                    out / "source_assessment.json",
                    {
                        "identity": identity,
                        "raw_sha256": sha(out / "source_assessment.pt"),
                        "metrics": source_report,
                    },
                )
                print(
                    {
                        "phase": "source_gate",
                        "strict_all_six_passed": source_report["all_six_source_gate"],
                        "admission_mode": config.source_gate_mode,
                        "admission_passed": admission["passed"],
                        "unmet_benefit_parameters": admission["unmet_benefit_parameters"],
                        "full_validation_min_r2": source_report[
                            "original_full_validation_retention"
                        ]["minimum_r2"],
                    },
                    flush=True,
                )
                if not config.verification and not admission["passed"]:
                    raise ValueError(
                        "Conditional freshly re-encoded source gate failed; no training allowed"
                    )
                logger.log(
                    source_step + start,
                    {
                        "source/full_validation_min_retention_r2": source_report[
                            "original_full_validation_retention"
                        ]["minimum_r2"],
                        "source/all_six_gate": float(source_report["all_six_source_gate"]),
                        "source/admission_passed": float(admission["passed"]),
                        "source/benefit_pass_count": float(
                            len(admission["benefit_pass_parameters"])
                        ),
                    },
                )
                if not records:
                    validate(0)
                    save(0)
                loaded_epoch, batches = -1, []
                for step in range(start + 1, config.steps + 1):
                    global_step = source_step + step
                    epoch, position = divmod(
                        global_step - 1, bundle.train_groups // bundle.batch_size
                    )
                    if loaded_epoch != epoch:
                        batches = list(
                            GlobalBatchSampler(
                                bundle.train_groups, bundle.batch_size, config.seed, epoch
                            )
                        )
                        loaded_epoch = epoch
                    batch = {
                        key: value[batches[position]].to(device) for key, value in train.items()
                    }
                    torch.manual_seed(config.seed * 100000 + global_step)  # pyright: ignore[reportUnknownMemberType]
                    optimizer.zero_grad(set_to_none=True)
                    loss, terms = training_loss(model, batch, objective, config.arm)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"Nonfinite loss at update{step}")
                    loss.backward()  # pyright: ignore[reportUnknownMemberType]
                    norm = float(
                        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    )
                    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
                    after_update(model, config.arm)
                    if step == 1 or step % 100 == 0 or step == config.steps:
                        print(
                            {
                                "step": global_step,
                                "updates": step,
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
                            },
                        )
                    if config.verification or step % 1000 == 0 or step == config.steps:
                        validate(step)
                        save(step)
                # This block is reached only at the fixed continuation endpoint.
                print({"phase": "fixed_endpoint_evaluation", "updates": config.steps}, flush=True)
                final_report: dict[str, Any]
                if (out / "endpoint_assessment.json").exists():
                    final_record = cast(dict[str, Any], read_json(out / "endpoint_assessment.json"))
                    if (
                        final_record["identity"] != identity
                        or final_record["checkpoint_sha256"] != sha(out / "last.pt")
                        or final_record["raw_sha256"] != sha(out / "endpoint_assessment.pt")
                    ):
                        raise ValueError("Fixed endpoint assessment changed")
                    final_report = final_record["metrics"]
                else:
                    final_precision, final_raw = precision_assessment(
                        model,
                        bundle.evaluation,
                        inputs.precision_data,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                        chunk_size=config.chunk_size,
                        progress=lambda n: logger.log(
                            source_step + config.steps,
                            {"progress/endpoint_precision_groups": float(n)},
                        ),
                    )
                    final_test, test_raw = monitor(
                        model,
                        bundle.evaluation,
                        selection,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                        final=True,
                    )
                    with preserve_rng(device):
                        source_mcc = state_metrics(
                            refresh_evaluation(
                                inputs.model, bundle.evaluation, ("calibration", "test")
                            ),
                            "test",
                        )
                    final_report = {
                        "precision_validation": final_precision,
                        "final_test": final_test,
                        "source_test_state": source_mcc,
                        "state_test_mcc_change": final_test["state"]["mcc"] - source_mcc["mcc"],
                        "paired_precision_validation_change": paired_changes(
                            source_raw,
                            final_raw,
                            bootstrap_samples=config.bootstrap_samples,
                            seed=config.seed,
                        ),
                    }
                    atomic_save(
                        out / "endpoint_assessment.pt", {"precision": final_raw, "test": test_raw}
                    )
                    write_json(
                        out / "endpoint_assessment.json",
                        {
                            "identity": identity,
                            "checkpoint_sha256": sha(out / "last.pt"),
                            "raw_sha256": sha(out / "endpoint_assessment.pt"),
                            "metrics": final_report,
                        },
                    )
                check_state()
                verify_files(inputs.hashes)
                if (
                    sha(out / "train_observations.pt") != train_sha
                    or model_fingerprint(inputs.model) != initial_hash
                ):
                    raise RuntimeError("Observed TRAIN cache or pristine source model changed")
                result: dict[str, Any] = {
                    "identity": identity,
                    "step": source_step + config.steps,
                    "updates": config.steps,
                    "arm": config.arm,
                    "source": source_report,
                    "records": records,
                    "endpoint": final_report,
                    "tracking": logger.metadata,
                    "scope": protocol["interpretation"],
                }
                write_json(out / "results.json", result)
                logger.log(
                    source_step + config.steps,
                    {
                        **measurement_metrics(final_report["final_test"], final=True),
                        "test/state_mcc": final_report["final_test"]["state"]["mcc"],
                        "test/state_mcc_change": final_report["state_test_mcc_change"],
                        "diagnostic/complete": 1.0,
                    },
                )
                for name in ("protocol.json", "results.json"):
                    logger.log_artifact(
                        out / name, name=f"joint-state-survival-{identity[:12]}-{Path(name).stem}"
                    )
                names = (
                    "protocol.json",
                    "initial_model.pt",
                    "train_observations.pt",
                    "train_receipt.json",
                    "source_assessment.json",
                    "source_assessment.pt",
                    "validation.json",
                    "last.pt",
                    "endpoint_assessment.json",
                    "endpoint_assessment.pt",
                    "results.json",
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
    """Expose explicit immutable source pins and one predeclared continuation arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-dir",
        "endpoint-dir",
        "calibration-dir",
        "precision-dir",
        "output-dir",
        "checkpoint-sha256",
        "calibrated-sha256",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--arm", choices=("frozen_state", "live_state"), required=True)
    parser.add_argument("--training-source-root")
    parser.add_argument("--data-record")
    parser.add_argument(
        "--source-gate-mode", choices=("all_six_use", "qualified_spartan"), default="all_six_use"
    )
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
