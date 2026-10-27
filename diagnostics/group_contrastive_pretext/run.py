"""Train fresh original E with group contrastive loss only; P stays frozen and unfitted."""

import argparse
import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.group_contrastive_pretext.evaluation import assess_model, measurement_metrics
from diagnostics.group_contrastive_pretext.forward import training_loss
from diagnostics.group_latent_control.run import (
    _restore_rng,  # pyright: ignore[reportPrivateUsage]
    _rng,  # pyright: ignore[reportPrivateUsage]
    initialize_control,
)
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.run import atomic_save, sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger


@dataclass(frozen=True)
class Config:
    """One bounded encoder-only pretext control with a validation-only followup gate."""

    source_dir: str
    output_dir: str
    training_source_root: str | None = None
    seed: int = 202610261301
    device: str = "cpu"
    steps: int = 30000
    groups: int = 64
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
        elif (self.steps, self.groups, self.seed) != (
            30000,
            64,
            202610261301,
        ):
            raise ValueError("Contrastive pretext fixes 30k updates, 64 groups and original seed")
        if self.threads < 1 or self.seed < 0:
            raise ValueError("Invalid threads or seed")
        source, output = Path(self.source_dir).resolve(), Path(self.output_dir).resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Output must be separate from immutable source")
        if self.training_source_root:
            archive = Path(self.training_source_root).resolve()
            if output == archive or output in archive.parents or archive in output.parents:
                raise ValueError("Output must be separate from the archived training source")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def fixed_hashes(model: SCJEPA) -> dict[str, str]:
    """Every module except the original parameter encoder remains fixed."""
    return {
        name: module_fingerprint(getattr(model, name))
        for name in ("online_encoder", "target_encoder", "predictor")
    }


def code_identity() -> str:
    """Bind this independent runner and every unchanged oracle dependency."""
    from diagnostics.mlp_acquisition_control.run import code_identity as mlp_code_identity

    digest = hashlib.sha256(mlp_code_identity().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Train E alone; use full validation to gate followup before fixed final test reporting."""
    source = Path(config.source_dir)
    bundle = load_primary_source(
        source,
        config.seed,
        expected_step=2 if config.verification else 30000,
        **({"expected_checkpoint_sha256": None} if config.verification else {}),
        training_source_root=Path(config.training_source_root)
        if config.training_source_root
        else None,
    )
    if not config.verification and (bundle.train_groups, bundle.batch_size) != (8192, 16):
        raise ValueError("Scientific acquisition requires original8192 groups and B16")
    device, out = torch.device(config.device), Path(config.output_dir)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        with file_lock(out / ".lock"):
            model, discarded_optimizer = build_source_model(bundle, torch.device("cpu"))
            del discarded_optimizer
            unused_table, initialization, source_hashes = initialize_control(
                model, bundle, "fresh_zero"
            )
            del unused_table
            model.requires_grad_(False)
            model = model.to(device)
            model.parameter_encoder.requires_grad_(True)
            frozen, flags = fixed_hashes(model), model.get_extra_state()
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(parameters, lr=0.001)
            selection = DiagnosticSelection(
                tuple(range(config.groups)), tuple(range(config.groups))
            )
            initial_encoder_hash = module_fingerprint(model.parameter_encoder)
            settings = asdict(config)
            for name in (
                "source_dir",
                "output_dir",
                "training_source_root",
                "wandb_mode",
                "wandb_entity",
            ):
                settings.pop(name)
            evidence: dict[str, Any] = {
                "settings": settings,
                "source_hashes": source_hashes,
                "source_initialization": initialization,
                "initial_encoder_sha256": initial_encoder_hash,
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
                    "Shared temporal information can be acquired by fresh original E without "
                    "predictor-dependent gradients, learned teachers or parameter labels."
                ),
                "controlled_change": (
                    "Original archived fresh E only, frozen original state maps and archived P. "
                    "Raw six-dimensional Euclidean multi-positive group discrimination replaces "
                    "all forecast/regularization training losses for30000 updates. No learned "
                    "projection head, code normalization, memory bank or hard-negative selection."
                ),
                "training_parameter_labels_absent": True,
                "production_training_modified": False,
                "optimized_loss_arm": "group_contrastive_only",
                "objective": (
                    "logits=-squaredEuclideanDistance/2; exclude self from denominator/positives; "
                    "all9 other contexts of samegroup positive, all othergroups negative. "
                    "Mean over anchors and positive log probabilities, subtractlog9 constant. "
                    "No SIGReg, separate consistency penalty, forecast or logits penalty."
                ),
                "optimizer": "Eonly freshAdam .001, globalnormclip1; originalB16/R10/K10/T12",
                "rng": (
                    "Original group batch schedule and seed*100000+step; "
                    "pretext uses no random projection draws"
                ),
                "evaluation": (
                    "Retention only; no predictive-use measurement with unfitted P. "
                    "Every1000: first64cal/val. At fixed30000: full512cal/256val first, "
                    "persist gate all6validationR2>=.8, then full512test retention. "
                    "Calibration fits and validation selects all original whole-code probes."
                ),
                "followup_gate": (
                    "Later predictor fitting allowed only by full-calibration/full-validation "
                    "all-sixR2>=.8; test never changes the gate. No automatic followup launch."
                ),
                "decision": (
                    "All-six heldout retention supports predictor-independent acquisition. "
                    "Low contrastive loss with partial retention shows discrimination can use "
                    "only a subset; finite continuous groups do not force allsix. Training-only "
                    "improvement suggests memorization. Failure is not intrinsic E incapacity. "
                    "No predictive-use, graph or original Experiment1 success claim."
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
                        "predictor": model.predictor.state_dict(),
                        "parameter_encoder": model.parameter_encoder.state_dict(),
                    },
                )
            initial_payload = cast(
                dict[str, Any],
                torch.load(out / "initial_modules.pt", map_location="cpu", weights_only=False),
            )
            if initial_payload["identity"] != identity:
                raise ValueError("Persisted pretext initialization identity changed")
            initial_copy = copy.deepcopy(model.predictor).cpu()
            initial_copy.load_state_dict(initial_payload["predictor"], strict=True)
            if module_fingerprint(initial_copy) != frozen["predictor"]:
                raise ValueError("Persisted pretext initialization weights changed")
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
                start, records = checkpoint["step"], checkpoint["records"]
                tracking = checkpoint["tracking"]
                _restore_rng(checkpoint["rng"], device)
            if not 0 <= start <= config.steps:
                raise ValueError("Invalid checkpoint update")

            def check_fixed() -> None:
                if sha(out / "initial_modules.pt") != initial_sha256:
                    raise RuntimeError("Persisted pretext initialization artifact changed")
                if fixed_hashes(model) != frozen or model.get_extra_state() != flags:
                    raise RuntimeError("Frozen encoders or rollout settings changed")

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
                        "tracking": logger.metadata,
                    },
                )

            def validate(step: int) -> None:
                check_fixed()
                report = assess_model(
                    model,
                    bundle.evaluation,
                    selection,
                )
                records.append({"step": step, **report})
                logger.log(step, measurement_metrics(report))
                write_json(out / "validation.json", {"identity": identity, "records": records})

            step = start
            try:
                if not records:
                    validate(0)
                    save(0)
                loaded_epoch, batches = -1, []
                for step in range(start + 1, config.steps + 1):
                    epoch, position = divmod(step - 1, bundle.train_groups // bundle.batch_size)
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
                    torch.manual_seed(config.seed * 100000 + step)  # pyright: ignore[reportUnknownMemberType]
                    model.train()
                    model.online_encoder.eval()
                    model.target_encoder.eval()
                    model.predictor.eval()
                    optimizer.zero_grad(set_to_none=True)
                    loss, terms = training_loss(model, batch)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"Nonfinite diagnostic loss at update{step}")
                    loss.backward()  # pyright: ignore[reportUnknownMemberType]
                    norm = float(
                        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
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
                checkpoint_sha = sha(out / "last.pt")
                if (out / "full_validation.json").exists():
                    full_record = cast(dict[str, Any], read_json(out / "full_validation.json"))
                    if (
                        full_record["identity"] != identity
                        or full_record["checkpoint_sha256"] != checkpoint_sha
                    ):
                        raise ValueError("Full-validation gate differs from fixed endpoint")
                    full_validation = full_record["metrics"]
                else:
                    full_validation = assess_model(model, bundle.evaluation, selection, full=True)
                    write_json(
                        out / "full_validation.json",
                        {
                            "identity": identity,
                            "checkpoint_sha256": checkpoint_sha,
                            "metrics": full_validation,
                        },
                    )
                gate = full_validation["full_validation_predictor_fit_gate"]
                logger.log(
                    step,
                    {
                        **{
                            f"validation/full/theta_{index}/retention_r2": float(value)
                            for index, value in enumerate(
                                full_validation["retention"]["selected_validation_r2"], 1
                            )
                        },
                        "validation/full/predictor_fit_gate": float(gate),
                    },
                )
                if (out / "final_test.json").exists():
                    final_record = cast(dict[str, Any], read_json(out / "final_test.json"))
                    if (
                        final_record["identity"] != identity
                        or final_record["checkpoint_sha256"] != checkpoint_sha
                    ):
                        raise ValueError("Final test differs from fixed endpoint")
                    final = final_record["metrics"]
                else:
                    final = assess_model(model, bundle.evaluation, selection, full=True, final=True)
                    write_json(
                        out / "final_test.json",
                        {
                            "identity": identity,
                            "checkpoint_sha256": checkpoint_sha,
                            "metrics": final,
                        },
                    )
                if final["full_validation_predictor_fit_gate"] != gate:
                    raise RuntimeError("Test evaluation changed the validation-only followup gate")
                check_fixed()
                for name, expected in source_hashes.items():
                    if sha(source / name) != expected:
                        raise RuntimeError("Immutable source changed")
                result = {
                    "identity": identity,
                    "step": step,
                    "records": records,
                    "final_test": final,
                    "full_validation": full_validation,
                    "predictor_fit_gate_from_validation_only": gate,
                    "initial_encoder_sha256": initial_encoder_hash,
                    "final_encoder_sha256": module_fingerprint(model.parameter_encoder),
                    "tracking": logger.metadata,
                    "scope": (
                        "Fresh label-free encoder-only contrastive diagnostic, "
                        "not original SPARTAN experiment success"
                    ),
                }
                write_json(out / "results.json", result)
                logger.log(step, measurement_metrics(final, final=True))
                logger.log(step, {"diagnostic/complete": 1.0})
                for name in ("results.json", "protocol.json"):
                    logger.log_artifact(
                        out / name,
                        name=f"group-contrastive-pretext-{identity[:12]}-{Path(name).stem}",
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
                                "full_validation.json",
                                "final_test.json",
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
                    out / "failure.json", name=f"group-contrastive-pretext-{identity[:12]}-failure"
                )
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Launch the fixed-budget encoder-only contrastive diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--training-source-root")
    for name, default in (
        ("seed", 202610261301),
        ("steps", 30000),
        ("groups", 64),
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
