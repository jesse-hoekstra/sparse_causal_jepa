"""Fit a matched dense MLP to exact codes in authenticated learned state coordinates."""

import argparse
import copy
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.group_latent_control.run import (
    _restore_rng,  # pyright: ignore[reportPrivateUsage]
    _rng,  # pyright: ignore[reportPrivateUsage]
)
from diagnostics.mlp_oracle_code_control.model import initialize_mlp, training_loss
from diagnostics.multiquery_control.data import TensorSplit, module_fingerprint
from diagnostics.multiquery_control.run import atomic_save, scalar_metrics, sha
from diagnostics.oracle_code_control.evaluation import PARAMETER_STD, assess, forecast_report
from diagnostics.oracle_code_control.run import Config, fixed_hashes, metrics
from diagnostics.oracle_code_control.run import code_identity as oracle_code_identity
from diagnostics.oracle_code_control.source import load_inputs
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.rng import preserve_rng


def code_identity() -> str:
    """Bind this independent runner and every unchanged oracle dependency."""
    digest = hashlib.sha256(oracle_code_identity().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
            model = inputs.model
            source_parameter_count = sum(
                parameter.numel() for parameter in model.predictor.parameters()
            )
            if not config.verification and source_parameter_count != 52289:
                raise ValueError(
                    "Source predictor parameter count differs from declared comparison"
                )
            mlp_initialization = initialize_mlp(model, config.seed)
            model = model.to(device)
            model.predictor.requires_grad_(True)
            frozen, flags = fixed_hashes(model), model.get_extra_state()
            optimizer = torch.optim.Adam(model.predictor.parameters(), lr=0.001)
            settings = asdict(config)
            for name in (
                "source_dir",
                "audit_dir",
                "output_dir",
                "training_source_root",
                "wandb_mode",
                "wandb_entity",
            ):
                settings.pop(name)
            evidence: dict[str, Any] = {
                "settings": settings,
                "source_hashes": inputs.hashes,
                "source_initialization": inputs.initialization,
                "mlp_initialization": mlp_initialization,
                "source_predictor_parameter_count": source_parameter_count,
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
                    "A parameter-count-matched dense MLP can learn all-six dynamics in the "
                    "same frozen learned coordinates where the original SPARTAN struggled."
                ),
                "controlled_change": (
                    "Replace fresh SPARTAN with fresh 28-211-211-6 ReLU MLP; exact code "
                    "theta/sqrt(.1875), 52123 versus 52289 parameters. "
                    "P-only fresh Adam .001 clip1, original B16/R10/K10/T12 two-step cached "
                    "forecasts; all encoders fixed."
                ),
                "diagnostic_physical_labels_in_training": True,
                "ordinary_training_modified": False,
                "parameter_population_std": PARAMETER_STD,
                "objective": (
                    "Mean two-step forecast MSE only. No attention logits exist; omitting "
                    "the SPARTAN logit regularizer is an explicit comparison confound. "
                    "No SIGReg, consistency gradients, or optimizer history."
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
                    " Training fit is separately evaluated on fixed first 256 training groups."
                ),
                "decision": (
                    "MLP success establishes bounded learnability in these learned coordinates, "
                    "implicating the SPARTAN-plus-regularizer training pipeline, not uniquely "
                    "attention bias. Both failing leaves target geometry and optimization "
                    "unresolved. One seed and matched parameter count do not equate capacity "
                    "or prove architectural superiority. Not label-free acquisition."
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
                    out / "initial_predictor.pt",
                    {
                        "identity": identity,
                        "initialization": mlp_initialization,
                        "predictor": model.predictor.state_dict(),
                    },
                )
            initial_payload = cast(
                dict[str, Any],
                torch.load(out / "initial_predictor.pt", map_location="cpu", weights_only=False),
            )
            if (
                initial_payload["identity"] != identity
                or initial_payload["initialization"] != mlp_initialization
            ):
                raise ValueError("Persisted MLP initialization identity changed")
            initial_copy = copy.deepcopy(model.predictor).cpu()
            initial_copy.load_state_dict(initial_payload["predictor"], strict=True)
            if module_fingerprint(initial_copy) != mlp_initialization["module_sha256"]:
                raise ValueError("Persisted MLP initialization weights changed")
            initial_sha256 = sha(out / "initial_predictor.pt")
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
                if sha(out / "initial_predictor.pt") != initial_sha256:
                    raise RuntimeError("Persisted MLP initialization artifact changed")
                if fixed_hashes(model) != frozen or model.get_extra_state() != flags:
                    raise RuntimeError("Frozen encoders or rollout settings changed")

            check_fixed()
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=f"mlp-oracle-code-{out.parent.name}",
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
                    candidate = copy.deepcopy(model).double().eval()
                    report = assess(
                        candidate,
                        inputs.bundle.evaluation,
                        "validation",
                        groups=config.groups,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                    )
                    report["cached_training_forecast"] = forecast_report(
                        candidate,
                        {
                            "states_online": inputs.bundle.train["online"][:256],
                            "states_target": inputs.bundle.train["target"][:256],
                            "actions": inputs.bundle.train["actions"][:256],
                            "theta": inputs.theta[:256],
                        },
                    )
                raw[step] = report.pop("raw")
                row = {"step": step, **report}
                records.append(row)
                logger.log(step, metrics(report, "validation"))
                logger.log(
                    step,
                    scalar_metrics(report["cached_training_forecast"], "cached_train/forecast"),
                )
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
                result = {
                    "identity": identity,
                    "step": step,
                    "records": records,
                    "final_test": final,
                    "tracking": logger.metadata,
                    "scope": "MLP exact-code architecture diagnostic, not label-free acquisition",
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
                        out / name, name=f"mlp-oracle-code-{identity[:12]}-{Path(name).stem}"
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
                                "initial_predictor.pt",
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
                    out / "failure.json", name=f"mlp-oracle-code-{identity[:12]}-failure"
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
