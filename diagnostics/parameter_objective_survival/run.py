"""Test whether an acquired six-code encoder survives restoring the parameter objective."""

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
from diagnostics.mlp_acquisition_control.evaluation import assess_model, measurement_metrics
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.overcomplete_use_precision.run import code_identity as precision_code_identity
from diagnostics.overcomplete_use_precision.run import verify_replay
from diagnostics.parameter_objective_survival.forward import Arm, training_loss
from diagnostics.parameter_objective_survival.source import load_inputs, verify_sources
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig


@dataclass(frozen=True)
class Config:
    """One fixed10k loss-arm continuation from the same learned MLP6 endpoint."""

    source_dir: str
    endpoint_dir: str
    output_dir: str
    arm: Arm = "forecast"
    training_source_root: str | None = None
    seed: int = 202610261301
    device: str = "cpu"
    steps: int = 10000
    groups: int = 64
    bootstrap_samples: int = 1000
    threads: int = 1
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Protect immutable sources and fix the scientific training/evaluation budget."""
        if self.verification:
            if not 1 <= self.steps <= 3 or not 2 <= self.groups <= 4:
                raise ValueError("Verification permits at most three updates and 2..4 groups")
        elif (self.steps, self.groups, self.bootstrap_samples, self.seed) != (
            10000,
            64,
            1000,
            202610261301,
        ):
            raise ValueError("Survival fixes 10k additional updates, 64 groups and canonical seed")
        if self.arm not in ("forecast", "parameter_objective"):
            raise ValueError("Choose forecast or restored parameter objective")
        if self.threads < 1 or self.bootstrap_samples < 2 or self.seed < 0:
            raise ValueError("Invalid threads, bootstrap count or seed")
        output = Path(self.output_dir).resolve()
        for source in (Path(self.source_dir).resolve(), Path(self.endpoint_dir).resolve()):
            if source == output or source in output.parents or output in source.parents:
                raise ValueError("Output must be separate from immutable sources")
        if self.training_source_root:
            archive = Path(self.training_source_root).resolve()
            if output == archive or output in archive.parents or archive in output.parents:
                raise ValueError("Output must be separate from archived training code")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def fixed_hashes(model: SCJEPA) -> dict[str, str]:
    """Only the state maps remain fixed; E and P both train."""
    return {
        name: module_fingerprint(getattr(model, name))
        for name in ("online_encoder", "target_encoder")
    }


def code_identity() -> str:
    """Bind this independent runner and every unchanged acquisition dependency."""
    from diagnostics.predictor_transfer_control.run import code_identity as parent_code_identity

    digest = hashlib.sha256((parent_code_identity() + precision_code_identity()).encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Train E/P without physical labels and evaluate only the fixed final test."""
    inputs = load_inputs(
        Path(config.source_dir),
        Path(config.endpoint_dir),
        device=torch.device(config.device),
        training_source_root=Path(config.training_source_root)
        if config.training_source_root
        else None,
        verification=config.verification,
    )
    bundle = inputs.bundle
    source_step = int(inputs.provenance["source_step"])
    if bundle.seed != config.seed:
        raise ValueError("Continuation seed differs from source")
    if not config.verification and (source_step, bundle.train_groups, bundle.batch_size) != (
        100000,
        8192,
        16,
    ):
        raise ValueError("Require fixed100k source and original8192 groups/B16")
    output = Path(config.output_dir).resolve()
    source_roots = {Path(name).resolve().parent for name in inputs.hashes}
    source_roots.add(bundle.source_dir.resolve())
    if any(
        output == root or output in root.parents or root in output.parents for root in source_roots
    ):
        raise ValueError("Output overlaps an immutable ancestor source directory")
    device, out = torch.device(config.device), Path(config.output_dir)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        with file_lock(out / ".lock"):
            model = inputs.model.requires_grad_(False)
            model.predictor.requires_grad_(True)
            model.parameter_encoder.requires_grad_(True)
            frozen, flags = fixed_hashes(model), model.get_extra_state()
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(parameters, lr=0.001)
            objective = AcquisitionObjective(
                ObjectiveConfig(**bundle.protocol["objective_config"])
            ).to(device)
            selection = DiagnosticSelection(
                tuple(range(config.groups)), tuple(range(config.groups))
            )
            initial_encoder_hash = module_fingerprint(model.parameter_encoder)
            initial_predictor_hash = module_fingerprint(model.predictor)
            settings = asdict(config)
            for name in (
                "source_dir",
                "endpoint_dir",
                "output_dir",
                "training_source_root",
                "wandb_mode",
                "wandb_entity",
            ):
                settings.pop(name)
            evidence: dict[str, Any] = {
                "settings": settings,
                "source_hashes": inputs.hashes,
                "source_provenance": inputs.provenance,
                "initial_encoder_sha256": initial_encoder_hash,
                "initial_predictor_sha256": initial_predictor_hash,
                "frozen_hashes": frozen,
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
                    "The acquired code may survive restoring parameter SIGReg .4 and "
                    "consistency1 when both E/P train, with state charts fixed."
                ),
                "controlled_change": (
                    "Only optimized parameter-auxiliary weights differ: forecast-only versus "
                    "forecast + .4 parameter SIGReg + consistency1. Both start the same fixed100k "
                    "MLP6 transfer weights with fresh Adam .001 and clip1; E/P train together, "
                    "both state encoders remain frozen and EMA is disabled. Continue original "
                    "batch/donor indices and per-update seeds100001..110000."
                ),
                "training_parameter_labels_absent": True,
                "optimized_loss_arm": config.arm,
                "production_training_modified": False,
                "optimizer_history": (
                    "Fresh identical Adam initialization in both arms; no parent moments"
                ),
                "source_step": source_step,
                "final_global_step": source_step + config.steps,
                "objective": (
                    "Compute the unchanged AcquisitionObjective in both arms to preserve "
                    "auxiliary RNG, then select the exact forecast tensor or complete restored "
                    "parameter objective. Frozen-state SIGReg is constant/omitted; MLP has no "
                    "attention logit term. No normalization, scaling, labels or distillation."
                ),
                "evaluation": (
                    "Every1000 additional updates: original64 calibration/validation retention "
                    "and matched responses. Only fixed110k endpoint: original full512/256/512 "
                    "calibration/validation/test retention and64test responses. No access to "
                    "the separate32768-group confirmatory test. Physical labels/inverse maps "
                    "remain evaluation-only. No stopping or checkpoint selection."
                ),
                "selection_scope": (
                    "Paired survival test on fresh label-free initializer ancestry; prior "
                    "assisted teacher controls are separate. Original tests remain unchanged."
                ),
                "decision": (
                    "Primary survival gate: every factor retains test R2>=.8. Both arms "
                    "passing supports retention compatibility only; forecast-only retention "
                    "success implicates auxiliary-induced retention loss, and both failing "
                    "implicates E/P continuation instability. Restored-objective-only retention "
                    "success supports a stabilizing effect. Response fidelity and predictive "
                    "benefit are reported separately and compared factorwise with the source "
                    "baseline: source all-six positive benefit is not established, so failing "
                    "that criterion alone cannot demonstrate forgetting of predictive use. "
                    "No state learning, sparse-graph or full Experiment1 success claim."
                ),
            }
            if resume:
                if read_json(out / "protocol.json")["identity"] != identity:
                    raise ValueError("Resume settings, code or source changed")
            elif any(path.name != ".lock" for path in out.iterdir()):
                raise FileExistsError("Use a fresh output directory or explicit resume")
            else:
                write_json(out / "protocol.json", protocol)
                atomic_save(
                    out / "initial_modules.pt",
                    {
                        "identity": identity,
                        "source_step": source_step,
                        "predictor": model.predictor.state_dict(),
                        "parameter_encoder": model.parameter_encoder.state_dict(),
                    },
                )
            initial_payload = cast(
                dict[str, Any],
                torch.load(out / "initial_modules.pt", map_location="cpu", weights_only=False),
            )
            if (
                initial_payload["identity"] != identity
                or initial_payload["source_step"] != source_step
            ):
                raise ValueError("Persisted continuation initialization identity changed")
            initial_copy = copy.deepcopy(model.predictor).cpu()
            initial_copy.load_state_dict(initial_payload["predictor"], strict=True)
            if module_fingerprint(initial_copy) != initial_predictor_hash:
                raise ValueError("Persisted continuation initialization weights changed")
            encoder_copy = copy.deepcopy(model.parameter_encoder).cpu()
            encoder_copy.load_state_dict(initial_payload["parameter_encoder"], strict=True)
            if module_fingerprint(encoder_copy) != initial_encoder_hash:
                raise ValueError("Persisted encoder initialization weights changed")
            initial_sha256 = sha(out / "initial_modules.pt")
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
            source_rng = dict(inputs.rng)
            if device.type != "cuda":
                source_rng["cuda"] = None
            _restore_rng(source_rng, device)
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
                start, records, raw = (
                    checkpoint["updates"],
                    checkpoint["records"],
                    checkpoint["raw"],
                )
                if checkpoint["step"] != source_step + start:
                    raise ValueError("Checkpoint global step differs from source plus updates")
                tracking = checkpoint["tracking"]
                _restore_rng(checkpoint["rng"], device)
            if not 0 <= start <= config.steps:
                raise ValueError("Invalid checkpoint update")

            def check_fixed() -> None:
                if sha(out / "initial_modules.pt") != initial_sha256:
                    raise RuntimeError("Persisted continuation initialization artifact changed")
                if fixed_hashes(model) != frozen or model.get_extra_state() != flags:
                    raise RuntimeError("Frozen encoders or rollout settings changed")
                if any(
                    p.requires_grad or p.grad is not None
                    for block in (model.online_encoder, model.target_encoder)
                    for p in block.parameters()
                ):
                    raise RuntimeError("Frozen state encoders acquired gradients")

            check_fixed()
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

            def save(step: int) -> None:
                check_fixed()
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
                check_fixed()
                report, measurements = assess_model(
                    model,
                    bundle.evaluation,
                    selection,
                    bootstrap_samples=config.bootstrap_samples,
                    seed=config.seed,
                )
                if step == 0:
                    verify_replay(report["response"], inputs.archived_validation["response"])
                    torch.testing.assert_close(
                        torch.tensor(report["retention"]["selected_validation_r2"]),
                        torch.tensor(
                            inputs.archived_validation["retention"]["selected_validation_r2"]
                        ),
                        atol=1e-7,
                        rtol=1e-6,
                    )
                raw[source_step + step] = measurements
                records.append({"step": source_step + step, "updates": step, **report})
                logger.log(source_step + step, measurement_metrics(report))
                write_json(out / "validation.json", {"identity": identity, "records": records})

            step = start
            try:
                if not records:
                    validate(0)
                    save(0)
                loaded_epoch, batches = -1, []
                for step in range(start + 1, config.steps + 1):
                    global_step = source_step + step
                    epoch, position = divmod(
                        global_step - 1, bundle.train_groups // bundle.batch_size
                    )
                    if epoch != loaded_epoch:
                        batches = list(
                            GlobalBatchSampler(
                                bundle.train_groups,
                                bundle.batch_size,
                                config.seed,
                                epoch,
                            )
                        )
                        loaded_epoch = epoch
                    selected = batches[position]
                    batch = {
                        name: value[selected].to(device) for name, value in bundle.train.items()
                    }
                    torch.manual_seed(config.seed * 100000 + global_step)  # pyright: ignore[reportUnknownMemberType]
                    model.train()
                    model.online_encoder.eval()
                    model.target_encoder.eval()
                    optimizer.zero_grad(set_to_none=True)
                    loss, terms = training_loss(model, batch, objective, config.arm)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"Nonfinite diagnostic loss at update{step}")
                    loss.backward()  # pyright: ignore[reportUnknownMemberType]
                    norm = float(
                        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    )
                    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
                    if step == 1 or step % 100 == 0 or step == config.steps:
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
                    "step": source_step + step,
                    "updates": step,
                    "records": records,
                    "final_test": final,
                    "tracking": logger.metadata,
                    "scope": (
                        "Paired restored-parameter-objective survival with fixed state charts; "
                        "not joint-state, SPARTAN or full Experiment1 success"
                    ),
                }
                write_json(out / "results.json", result)
                atomic_save(
                    out / "measurements.pt",
                    {"identity": identity, "validation": raw, "test": final_raw},
                )
                logger.log(source_step + step, measurement_metrics(final, final=True))
                logger.log(source_step + step, {"diagnostic/complete": 1.0})
                for name in ("results.json", "protocol.json"):
                    logger.log_artifact(
                        out / name,
                        name=f"parameter-objective-survival-{identity[:12]}-{Path(name).stem}",
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
                                "initial_modules.pt",
                                "results.json",
                                "last.pt",
                                "validation.json",
                                "measurements.pt",
                                "final_test.json",
                                "final_test_measurements.pt",
                            )
                        },
                    },
                )
                return cast(dict[str, Any], read_json(out / "results.json"))
            except BaseException as error:
                write_json(
                    out / "failure.json",
                    {
                        "identity": identity,
                        "step": source_step + step,
                        "updates": step,
                        "error": str(error),
                    },
                )
                logger.log(source_step + step, {"diagnostic/failed": 1.0})
                logger.log_artifact(
                    out / "failure.json",
                    name=f"parameter-objective-survival-{identity[:12]}-failure",
                )
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Launch one declared fixed10k survival arm with an independent output and W&B run."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "endpoint-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--arm", choices=("forecast", "parameter_objective"), required=True)
    parser.add_argument("--training-source-root")
    for name, default in (
        ("seed", 202610261301),
        ("steps", 10000),
        ("groups", 64),
        ("bootstrap-samples", 1000),
        ("threads", 1),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--verification", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    resume = bool(args.pop("resume"))
    run(Config(**args), resume=resume)


if __name__ == "__main__":
    main()
