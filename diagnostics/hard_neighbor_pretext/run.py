"""Train fresh E with uniform then hard-neighbor sampling and unchanged contrastive loss."""

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.group_contrastive_pretext.forward import training_loss
from diagnostics.group_latent_control.run import (
    _restore_rng,  # pyright: ignore[reportPrivateUsage]
    _rng,  # pyright: ignore[reportPrivateUsage]
    initialize_control,
)
from diagnostics.hard_neighbor_pretext.evaluation import assess_model, measurement_metrics
from diagnostics.hard_neighbor_pretext.sampler import (
    ANCHOR_NAMESPACE,
    Bank,
    hard_indices,
    refresh_bank,
    refresh_due,
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
    """A fixed fresh curriculum:5000uniform then25000hard-neighbor contrastive updates."""

    source_dir: str
    output_dir: str
    training_source_root: str | None = None
    seed: int = 202610261301
    device: str = "cpu"
    steps: int = 30000
    warmup_steps: int = 5000
    bank_refresh_interval: int = 512
    groups: int = 64
    threads: int = 1
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Protect immutable sources and fix the scientific training/evaluation budget."""
        if self.verification:
            if (
                not 1 <= self.steps <= 5
                or not 2 <= self.groups <= 4
                or not 0 <= self.warmup_steps < self.steps
                or not 1 <= self.bank_refresh_interval <= 3
            ):
                raise ValueError(
                    "Verification permits1..5updates,0..steps-1warmup,1..3refresh,2..4groups"
                )
        elif (
            self.steps,
            self.warmup_steps,
            self.bank_refresh_interval,
            self.groups,
            self.seed,
        ) != (
            30000,
            5000,
            512,
            64,
            202610261301,
        ):
            raise ValueError("Hard pretext fixes30k,5kuniform,512refresh,64groups and originalseed")
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
    from diagnostics.group_contrastive_pretext.run import code_identity as mlp_code_identity

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
                    "Confusable TRAIN-group neighborhoods expose missing shared factors beyond "
                    "the pair sufficient for ordinary small-batch group discrimination."
                ),
                "controlled_change": (
                    "Original archived fresh E only, frozen original state maps and archived P. "
                    "First5000originaluniform updates, then25000hard-neighbor batches; unchanged "
                    "raw-six-dimensional Euclidean multi-positive contrastive loss. "
                    "No bank logits, projection head, code normalization or teacher targets; "
                    "forecast loss, SIGReg or separate consistency penalty."
                ),
                "training_parameter_labels_absent": True,
                "production_training_modified": False,
                "optimized_loss_arm": "group_contrastive_only",
                "sampler": {
                    "warmup_steps": config.warmup_steps,
                    "refresh_interval": config.bank_refresh_interval,
                    "refresh_timing": "Before5001,5513,...; every512hardupdates",
                    "keys": "TRAINonly:mean10separatelyencoded rawcontextcodes, no whitening",
                    "neighbors": "Uniform anchor +15nearest distinct means; groupID tie-break",
                    "anchor_seed": f"(seed*1000003+global_step+{ANCHOR_NAMESPACE})mod(2**63-1)",
                    "gradient_boundary": "Keys select indices; recompute all160attached codes",
                },
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
                "hard_metric_scope": (
                    "TRAIN hard candidates8192, monitor validation64, final validation256/test512. "
                    "Hard loss gaps do not match neighborhood difficulty. Report candidate counts; "
                    "ordinary B16 heldout contrastive loss remains comparable."
                ),
                "followup_gate": (
                    "Later predictor fitting allowed only by full-calibration/full-validation "
                    "all-sixR2>=.8; test never changes the gate. No automatic followup launch."
                ),
                "decision": (
                    "All-six heldout retention supports confusable-group sampling. "
                    "Low contrastive loss with partial retention shows discrimination can use "
                    "only a subset; then deprioritize this objective family. Ordinary heldout "
                    "loss and retention assess generalization. TRAIN-hard versus heldout-hard "
                    "loss alone cannot prove memorization: neighbor candidate counts differ. "
                    "Failure is not intrinsic E incapacity. "
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
            bank: Bank | None = None
            refreshes: list[dict[str, Any]] = []
            exposure_counts = torch.zeros(bundle.train_groups, dtype=torch.long)
            hard_exposure_counts = torch.zeros_like(exposure_counts)
            anchor_counts = torch.zeros_like(exposure_counts)
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
                bank = (
                    Bank.from_payload(checkpoint["bank"])
                    if checkpoint["bank"] is not None
                    else None
                )
                refreshes = checkpoint["refreshes"]
                exposure_counts = checkpoint["exposure_counts"]
                hard_exposure_counts = checkpoint["hard_exposure_counts"]
                anchor_counts = checkpoint["anchor_counts"]
                tracking = checkpoint["tracking"]
                _restore_rng(checkpoint["rng"], device)
            if not 0 <= start <= config.steps:
                raise ValueError("Invalid checkpoint update")
            expected_refresh = (
                config.warmup_steps
                + 1
                + ((start - config.warmup_steps - 1) // config.bank_refresh_interval)
                * config.bank_refresh_interval
            )
            if start <= config.warmup_steps:
                if bank is not None or refreshes:
                    raise ValueError("Uniform prefix cannot contain a hard bank")
            elif (
                bank is None
                or bank.refresh_step != expected_refresh
                or len(bank.means) != bundle.train_groups
                or not refreshes
                or refreshes[-1]
                != {key: value for key, value in bank.payload().items() if key != "means"}
            ):
                raise ValueError("Saved hard bank schedule or lineage differs")
            for counts, total in (
                (exposure_counts, start * bundle.batch_size),
                (hard_exposure_counts, max(0, start - config.warmup_steps) * bundle.batch_size),
                (anchor_counts, max(0, start - config.warmup_steps)),
            ):
                if (
                    counts.shape != (bundle.train_groups,)
                    or counts.dtype != torch.long
                    or bool((counts < 0).any())
                    or int(counts.sum()) != total
                ):
                    raise ValueError("Saved group exposure counts differ")

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
                payload = {
                    "identity": identity,
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": _rng(device),
                    "records": records,
                    "tracking": logger.metadata,
                    "bank": bank.payload() if bank is not None else None,
                    "refreshes": refreshes,
                    "exposure_counts": exposure_counts,
                    "hard_exposure_counts": hard_exposure_counts,
                    "anchor_counts": anchor_counts,
                }
                atomic_save(out / "last.pt", payload)
                if step == config.warmup_steps:
                    atomic_save(out / "warmup_final.pt", payload)
                    write_json(
                        out / "warmup_boundary.json",
                        {
                            "identity": identity,
                            "step": step,
                            "encoder_sha256": module_fingerprint(model.parameter_encoder),
                            "checkpoint_sha256": sha(out / "warmup_final.pt"),
                        },
                    )
                write_json(
                    out / "sampler_history.json", {"identity": identity, "refreshes": refreshes}
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
                    hard = step > config.warmup_steps
                    if hard:
                        if refresh_due(step, config.warmup_steps, config.bank_refresh_interval):
                            bank = refresh_bank(model, bundle.train, step)
                            receipt = {
                                key: value
                                for key, value in bank.payload().items()
                                if key != "means"
                            }
                            refreshes.append(receipt)
                            write_json(
                                out / "sampler_history.json",
                                {"identity": identity, "refreshes": refreshes},
                            )
                            print(json.dumps({"sampler_refresh": receipt}), flush=True)
                        if bank is None:
                            raise RuntimeError("Hard sampling has no initialized bank")
                        selected = hard_indices(bank, bundle.batch_size, config.seed, step)
                    else:
                        epoch, position = divmod(step - 1, bundle.train_groups // bundle.batch_size)
                        if epoch != loaded_epoch:
                            batches = list(
                                GlobalBatchSampler(
                                    bundle.train_groups, bundle.batch_size, config.seed, epoch
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
                    exposure_counts[selected] += 1
                    if hard:
                        hard_exposure_counts[selected] += 1
                        anchor_counts[selected[0]] += 1
                    if step == 1 or step % 100 == 0 or step == config.steps:
                        logger.log(
                            step,
                            {
                                **{
                                    f"train/{name}": float(value.detach())
                                    for name, value in terms.items()
                                },
                                "train/gradient_norm": norm,
                                "train/hard_sampling": float(hard),
                                "train/hard_candidate_groups": float(
                                    bundle.train_groups if hard else 0
                                ),
                                "train/min_hard_group_exposure": float(hard_exposure_counts.min()),
                                "train/max_hard_group_exposure": float(hard_exposure_counts.max()),
                                "train/bank_refresh_step": float(bank.refresh_step if bank else 0),
                                "train/groups_seen": float((exposure_counts > 0).sum()),
                                "train/hard_groups_seen": float((hard_exposure_counts > 0).sum()),
                                "train/anchors_seen": float((anchor_counts > 0).sum()),
                                "train/min_group_exposure": float(exposure_counts.min()),
                                "train/max_group_exposure": float(exposure_counts.max()),
                            },
                        )
                    if (
                        config.verification
                        or step % 1000 == 0
                        or step in (config.warmup_steps, config.steps)
                    ):
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
                    "sampler_exposure": {
                        "total_group_draws": int(exposure_counts.sum()),
                        "unique_groups": int((exposure_counts > 0).sum()),
                        "hard_group_draws": int(hard_exposure_counts.sum()),
                        "hard_unique_groups": int((hard_exposure_counts > 0).sum()),
                        "unique_anchors": int((anchor_counts > 0).sum()),
                        "min_exposure": int(exposure_counts.min()),
                        "max_exposure": int(exposure_counts.max()),
                        "counts_artifact": "sampler_counts.pt",
                    },
                    "scope": (
                        "Fresh label-free hard-neighbor encoder pretext, "
                        "not original SPARTAN experiment success"
                    ),
                }
                atomic_save(
                    out / "sampler_counts.pt",
                    {
                        "identity": identity,
                        "exposure_counts": exposure_counts,
                        "hard_exposure_counts": hard_exposure_counts,
                        "anchor_counts": anchor_counts,
                    },
                )
                write_json(out / "results.json", result)
                logger.log(step, measurement_metrics(final, final=True))
                logger.log(step, {"diagnostic/complete": 1.0})
                for name in ("results.json", "protocol.json"):
                    logger.log_artifact(
                        out / name,
                        name=f"hard-neighbor-pretext-{identity[:12]}-{Path(name).stem}",
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
                                "warmup_final.pt",
                                "warmup_boundary.json",
                                "sampler_history.json",
                                "sampler_counts.pt",
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
                    out / "failure.json", name=f"hard-neighbor-pretext-{identity[:12]}-failure"
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
        ("warmup-steps", 5000),
        ("bank-refresh-interval", 512),
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
