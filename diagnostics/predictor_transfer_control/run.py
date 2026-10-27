"""Fixed-budget fresh predictor fitting on one authenticated learned six-code chart."""

import argparse
import copy
import hashlib
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.group_latent_control.run import (
    _restore_rng,  # pyright: ignore[reportPrivateUsage]
    _rng,  # pyright: ignore[reportPrivateUsage]
)
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.predictor_transfer_control.evaluation import assess_model, measurement_metrics
from diagnostics.predictor_transfer_control.forward import training_loss
from diagnostics.predictor_transfer_control.model import audit_model_fingerprint
from diagnostics.predictor_transfer_control.source import (
    SEED,
    Arm,
    fixed_hashes,
    load_inputs,
    verify_sources,
)
from diagnostics.shared_feature_acquisition.run import code_identity as parent_code_identity
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig


@dataclass(frozen=True)
class Config:
    """One paired arm with an unchanged native PCA6 chart and fixed final test budget."""

    source_dir: str
    endpoint_dir: str
    pca_audit_dir: str
    output_dir: str
    arm: Arm
    training_source_root: str | None = None
    seed: int = SEED
    device: str = "cpu"
    steps: int = 100000
    groups: int = 64
    bootstrap_samples: int = 1000
    threads: int = 1
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Forbid source overwrites, scientific budget changes and undeclared arms."""
        if self.arm not in ("spartan", "mlp"):
            raise ValueError("Choose the predeclared SPARTAN or MLP arm")
        if self.verification:
            if not 1 <= self.steps <= 3 or not 2 <= self.groups <= 4:
                raise ValueError("Verification permits at most three updates and 2..4 groups")
        elif (self.seed, self.steps, self.groups, self.bootstrap_samples) != (
            SEED,
            100000,
            64,
            1000,
        ):
            raise ValueError("Scientific transfer fixes canonical seed, 100k steps and 64 groups")
        if self.threads < 1 or self.bootstrap_samples < 2 or self.seed < 0:
            raise ValueError("Invalid threads, bootstrap count or seed")
        if torch.device(self.device).type not in ("cpu", "cuda"):
            raise ValueError("Only CPU verification and CUDA scientific execution supported")
        output = Path(self.output_dir).resolve()
        for name in (
            self.source_dir,
            self.endpoint_dir,
            self.pca_audit_dir,
            self.training_source_root,
        ):
            if name is None:
                continue
            source = Path(name).resolve()
            if output == source or output in source.parents or source in output.parents:
                raise ValueError("Output must be disjoint from all immutable source roots")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def code_identity() -> str:
    """Bind this isolated runner and the full unchanged source/evaluation dependency chain."""
    digest = hashlib.sha256(parent_code_identity().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Train only fresh P; never select a checkpoint or update the learned code/state chart."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    device, out = torch.device(config.device), Path(config.output_dir)
    try:
        with file_lock(out / ".lock"):
            inputs = load_inputs(
                Path(config.source_dir),
                Path(config.endpoint_dir),
                Path(config.pca_audit_dir),
                config.seed,
                config.arm,
                training_source_root=Path(config.training_source_root)
                if config.training_source_root
                else None,
                verification=config.verification,
            )
            bundle = inputs.bundle
            if not config.verification and (bundle.train_groups, bundle.batch_size) != (8192, 16):
                raise ValueError("Require original 8192 TRAIN groups and B16")
            model = inputs.model.to(device)
            model.predictor.requires_grad_(True)
            parameters = list(model.predictor.parameters())
            optimizer = torch.optim.Adam(parameters, lr=0.001)
            objective = AcquisitionObjective(
                ObjectiveConfig(**bundle.protocol["objective_config"])
            ).to(device)
            selection = DiagnosticSelection(
                tuple(range(config.groups)), tuple(range(config.groups))
            )
            evidence: dict[str, Any] = {
                "config": asdict(config),
                "source_hashes": inputs.hashes,
                "provenance": inputs.provenance,
                "code_identity": code_identity(),
                "torch": str(torch.__version__),
            }
            identity = canonical_fingerprint(evidence)
            protocol: dict[str, Any] = {
                "identity": identity,
                **evidence,
                "command": shlex.join([sys.executable, *sys.argv]),
                "hypothesis": (
                    "An already informative frozen six-code encoder separates fresh predictor "
                    "learning from parameter acquisition. Original SPARTAN may learn usable "
                    "dynamics once all six factors are available."
                ),
                "controlled_change": (
                    "Fresh original dense SPARTAN versus fresh 28-211-211-6 ReLU MLP on the "
                    "same authenticated centered TRAIN-PCA6 chart. Fixed E/state maps, "
                    "original B16/R10/K10/T12 donor draws, attached two-step forecasts, "
                    "fresh Adam .001, global gradient clip1, fixed 100k updates."
                ),
                "objective": (
                    "SPARTAN forecast MSE + .0001 attention logit penalty; MLP forecast MSE. "
                    "The architecture-specific logit term is explicit. Frozen representation "
                    "penalties are constant; their original RNG draws remain paired."
                ),
                "reference": (
                    "The algebraically composed pretrained MLP6 is frozen and evaluated only "
                    "on validation. Neither trained predictor inherits its weights or optimizer."
                ),
                "evaluation": (
                    "Original64 calibration/validation monitoring every1000; no stopping or "
                    "selection. Only fixed100k trained arm gets full512/256/512 retention "
                    "and original64 test donor0/query5..9 positive support-clipped interventions."
                ),
                "decision": (
                    "SPARTAN success supports a graph-bearing predictor transfer; MLP-only "
                    "success localizes the remaining tested SPARTAN pipeline. If both fresh "
                    "arms fail while the composed reference works, fresh predictor learning "
                    "remains limiting. None proves a global capacity impossibility or "
                    "state-MCC/sparsification success."
                ),
                "training_parameter_labels_absent": True,
                "production_training_modified": False,
            }
            if (out / "protocol.json").exists():
                if not resume or read_json(out / "protocol.json")["identity"] != identity:
                    raise ValueError("Matching explicit resume required")
            elif any(path.name != ".lock" for path in out.iterdir()):
                raise FileExistsError("Require a fresh output directory")
            else:
                write_json(out / "protocol.json", protocol)
                atomic_save(
                    out / "initial_modules.pt",
                    {
                        "identity": identity,
                        "predictor": model.predictor.state_dict(),
                        "parameter_encoder": model.parameter_encoder.state_dict(),
                        "projection": inputs.projection,
                        "reference_model": inputs.reference_model.state_dict(),
                    },
                )
            initial = cast(
                dict[str, Any],
                torch.load(out / "initial_modules.pt", map_location="cpu", weights_only=False),
            )
            if initial["identity"] != identity:
                raise ValueError("Initial artifact identity changed")
            restored = copy.deepcopy(model).cpu()
            restored.predictor.load_state_dict(initial["predictor"], strict=True)
            restored.parameter_encoder.load_state_dict(initial["parameter_encoder"], strict=True)
            if (
                module_fingerprint(restored.predictor)
                != inputs.provenance["fresh_predictor_sha256"]
                or fixed_hashes(restored) != inputs.provenance["frozen_hashes"]
            ):
                raise ValueError("Saved initial predictor or frozen encoder changed")
            restored_reference = copy.deepcopy(inputs.reference_model)
            restored_reference.load_state_dict(initial["reference_model"], strict=True)
            if (
                audit_model_fingerprint(restored_reference)
                != inputs.provenance["reference_model_sha256"]
            ):
                raise ValueError("Saved composed reference changed")
            for name, value in inputs.projection.items():
                torch.testing.assert_close(initial["projection"][name], value, rtol=0, atol=0)
            initial_sha = sha(out / "initial_modules.pt")
            if (out / "completed.json").exists():
                completed = cast(dict[str, Any], read_json(out / "completed.json"))
                if completed["identity"] != identity or any(
                    sha(out / name) != value for name, value in completed["hashes"].items()
                ):
                    raise ValueError("Completed transfer artifacts changed")
                return cast(dict[str, Any], read_json(out / "results.json"))
            start = 0
            records: list[dict[str, Any]] = []
            raw: dict[int, Any] = {}
            tracking = (
                read_json(out / "tracking.json") if (out / "tracking.json").exists() else None
            )
            if resume and (out / "last.pt").exists():
                saved = cast(
                    dict[str, Any],
                    torch.load(out / "last.pt", map_location="cpu", weights_only=False),
                )
                if saved["identity"] != identity:
                    raise ValueError("Checkpoint identity differs")
                model.load_state_dict(saved["model"], strict=True)
                optimizer.load_state_dict(saved["optimizer"])
                start, records, raw = saved["step"], saved["records"], saved["raw"]
                _restore_rng(saved["rng"], device)
            if not 0 <= start <= config.steps:
                raise ValueError("Invalid checkpoint update")

            def check_fixed() -> None:
                if sha(out / "initial_modules.pt") != initial_sha:
                    raise RuntimeError("Initial artifact changed")
                if fixed_hashes(model) != inputs.provenance["frozen_hashes"]:
                    raise RuntimeError("Frozen parameter/state encoder changed")
                if any(
                    p.requires_grad or p.grad is not None
                    for module in (
                        model.online_encoder,
                        model.target_encoder,
                        model.parameter_encoder,
                    )
                    for p in module.parameters()
                ):
                    raise RuntimeError("Frozen encoder acquired gradients")

            check_fixed()
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=f"predictor-transfer-{config.arm}-{out.parent.name}",
                    job_type="diagnostic",
                ),
                resolved_config=protocol,
                dataset_identity=bundle.protocol["identity"],
                out_dir=out,
                resume_state=tracking if tracking and tracking.get("id") else None,
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
                report, measurements = assess_model(
                    model,
                    bundle.evaluation,
                    selection,
                    bootstrap_samples=config.bootstrap_samples,
                    seed=config.seed,
                )
                records.append({"step": step, **report})
                raw[step] = measurements
                logger.log(step, measurement_metrics(report))
                write_json(out / "validation.json", {"identity": identity, "records": records})

            step = start
            try:
                if not (out / "reference_validation.json").exists():
                    reference_report, reference_raw = assess_model(
                        inputs.reference_model.to(device),
                        bundle.evaluation,
                        selection,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                    )
                    atomic_save(out / "reference_measurements.pt", reference_raw)
                    write_json(
                        out / "reference_validation.json",
                        {
                            "identity": identity,
                            "metrics": reference_report,
                            "raw_sha256": sha(out / "reference_measurements.pt"),
                        },
                    )
                    logger.log(
                        0,
                        {
                            "reference/" + k: v
                            for k, v in measurement_metrics(reference_report).items()
                        },
                    )
                reference_record = read_json(out / "reference_validation.json")
                if reference_record["identity"] != identity or reference_record[
                    "raw_sha256"
                ] != sha(out / "reference_measurements.pt"):
                    raise ValueError("Reference validation artifact changed")
                if not records:
                    validate(0)
                    save(0)
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
                    indices = batches[position]
                    batch = {
                        name: value[indices].to(device) for name, value in bundle.train.items()
                    }
                    torch.manual_seed(config.seed * 100000 + step)  # pyright: ignore[reportUnknownMemberType]
                    model.train()
                    model.online_encoder.eval()
                    model.target_encoder.eval()
                    model.parameter_encoder.eval()
                    optimizer.zero_grad(set_to_none=True)
                    loss, terms = training_loss(model, batch, objective)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"Nonfinite diagnostic loss at update {step}")
                    loss.backward()  # pyright: ignore[reportUnknownMemberType]
                    norm = float(
                        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    )
                    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
                    if step == 1 or step % 100 == 0 or step == config.steps:
                        logger.log(
                            step,
                            {
                                **{f"train/{k}": float(v.detach()) for k, v in terms.items()},
                                "train/gradient_norm": norm,
                            },
                        )
                    if config.verification or step % 1000 == 0 or step == config.steps:
                        validate(step)
                        save(step)
                if (out / "final_test.json").exists():
                    final_record = cast(dict[str, Any], read_json(out / "final_test.json"))
                    if (
                        final_record["identity"] != identity
                        or final_record["checkpoint_sha256"] != sha(out / "last.pt")
                        or final_record["raw_sha256"] != sha(out / "final_test_measurements.pt")
                    ):
                        raise ValueError("Final test differs from fixed final checkpoint")
                    final = final_record["metrics"]
                    final_raw = torch.load(
                        out / "final_test_measurements.pt", map_location="cpu", weights_only=False
                    )
                else:
                    final, final_raw = assess_model(
                        model,
                        bundle.evaluation,
                        selection,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                        final=True,
                    )
                    atomic_save(out / "final_test_measurements.pt", final_raw)
                    write_json(
                        out / "final_test.json",
                        {
                            "identity": identity,
                            "checkpoint_sha256": sha(out / "last.pt"),
                            "raw_sha256": sha(out / "final_test_measurements.pt"),
                            "metrics": final,
                        },
                    )
                check_fixed()
                verify_sources(inputs)
                result = {
                    "identity": identity,
                    "step": step,
                    "arm": config.arm,
                    "records": records,
                    "reference_validation": reference_record["metrics"],
                    "final_test": final,
                    "tracking": logger.metadata,
                    "scope": "Frozen learned-code predictor transfer; no state-MCC claim",
                }
                write_json(out / "results.json", result)
                atomic_save(
                    out / "measurements.pt",
                    {
                        "identity": identity,
                        "validation": raw,
                        "test": final_raw,
                    },
                )
                logger.log(step, measurement_metrics(final, final=True))
                logger.log(step, {"diagnostic/complete": 1.0})
                for name in ("results.json", "protocol.json"):
                    logger.log_artifact(
                        out / name, name=f"predictor-transfer-{identity[:12]}-{Path(name).stem}"
                    )
                write_json(
                    out / "completed.json",
                    {
                        "identity": identity,
                        "hashes": {
                            name: sha(out / name)
                            for name in (
                                "protocol.json",
                                "initial_modules.pt",
                                "results.json",
                                "last.pt",
                                "validation.json",
                                "measurements.pt",
                                "reference_validation.json",
                                "reference_measurements.pt",
                                "final_test.json",
                                "final_test_measurements.pt",
                            )
                        },
                    },
                )
                logger.finish()
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
    """Run one independently named arm in a detached server session with online W&B."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "endpoint-dir", "pca-audit-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--arm", choices=("spartan", "mlp"), required=True)
    parser.add_argument("--training-source-root")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cpu")
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
