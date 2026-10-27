"""Bounded predictor-only refit; source parameter information stays fixed."""

import argparse
import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.frozen_predictor_control.source import (
    SourceBundle,
    build_fresh_model,
    frozen_module_hashes,
    load_source_bundle,
)
from diagnostics.joint_acquisition.baseline import (
    DiagnosticSelection,
    build_matched_batch,
    evaluate_responses,
    validation_retention,
)
from diagnostics.joint_acquisition.corrections import validate_saved_optimizer
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.multiquery_control.run import atomic_save, scalar_metrics, sha
from diagnostics.predictive_supervision_control.evaluation import evaluate
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective
from diagnostics.predictive_supervision_control.run import code_identity as shared_code_identity
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng

SOURCE_SHA256 = "b28c7308fad6163344050949dcc6fabc11b29604c0b247b2cafaf36c8c234544"
FIXED_MODULES = ("online_encoder", "target_encoder", "parameter_encoder")


@dataclass(frozen=True)
class Config:
    """One predeclared 10k refit, with tiny fixtures explicitly marked verification."""

    source_dir: str
    checkpoint: str
    output_dir: str
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
        """Reject unbounded work and changes to the predeclared scientific budget."""
        if self.verification:
            if not 1 <= self.steps <= 3 or not 2 <= self.groups <= 4:
                raise ValueError("Verification permits at most three updates and four groups")
        elif self.steps != 10000 or self.groups != 64 or self.bootstrap_samples != 1000:
            raise ValueError("Scientific control requires 10000 updates, 64 groups, 1000 resamples")
        if self.seed < 0 or self.threads < 1 or self.bootstrap_samples < 2:
            raise ValueError("Invalid seed, threads or bootstrap budget")
        out, source = Path(self.output_dir).resolve(), Path(self.source_dir).resolve()
        if out == source or source in out.parents or out in source.parents:
            raise ValueError("Output and source directories must be separate")
        if Path(self.checkpoint).resolve().is_relative_to(out):
            raise ValueError("Source checkpoint must be outside output directory")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def fixed_hashes(model: SCJEPA) -> dict[str, str]:
    """Fingerprint both state encoders and the complete single-context parameter encoder."""
    return {name: module_fingerprint(getattr(model, name)) for name in FIXED_MODULES}


def restore_source(
    bundle: SourceBundle, checkpoint: Path, device: torch.device, *, verification: bool = False
) -> tuple[SCJEPA, torch.optim.AdamW]:
    """Load actual source weights and exact AdamW moments before freezing the encoder."""
    expected = bundle.manifest["provenance"]["sha256"]["checkpoint"]
    if sha(checkpoint) != expected or (not verification and expected != SOURCE_SHA256):
        raise ValueError("Actual source checkpoint hash differs from all-six provenance")
    payload = cast(dict[str, Any], torch.load(checkpoint, map_location="cpu", weights_only=False))
    model = build_fresh_model(bundle, 0, torch.device("cpu"))
    model.load_state_dict(payload["model"], strict=True)
    if (
        frozen_module_hashes(model) != bundle.manifest["frozen_hashes"]
        or model.get_extra_state() != bundle.frozen_payload["extra_state"]
        or module_fingerprint(model.parameter_encoder)
        != bundle.manifest["provenance"]["source_q_sha256"]
        or payload["stage"] != "washout"
        or payload["step"] != 10000
    ):
        raise ValueError("Actual source modules or completed washout step differ")
    model.parameter_encoder.requires_grad_(True)
    model.predictor.requires_grad_(True)
    model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    saved = copy.deepcopy(payload["optimizer"])
    groups = saved["param_groups"]
    if len(groups) != 1 or groups[0]["params"] != list(range(len(parameters))):
        raise ValueError("Source optimizer parameter order is not the complete encoder/predictor")
    group = groups[0]
    if (
        group["lr"] != 0.001
        or group["weight_decay"] != 0
        or tuple(group["betas"]) != (0.9, 0.999)
        or group["eps"] != 1e-8
        or group.get("amsgrad", False)
        or group.get("maximize", False)
    ):
        raise ValueError("Source optimizer semantics differ from recorded AdamW")
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0)
    optimizer.load_state_dict(saved)
    validate_saved_optimizer(model, optimizer)
    if any(float(state["step"]) != 10000 for state in optimizer.state.values()):
        raise ValueError("Source optimizer moments do not match source step")
    model.parameter_encoder.requires_grad_(False).eval()
    model.eval()
    return model, optimizer


def assessment_metrics(row: dict[str, Any]) -> dict[str, float]:
    """Log measured response and retention scalars, including every physical parameter."""
    metrics: dict[str, float] = {}
    for horizon, value in enumerate(row["response"]["ordinary_mse_by_horizon"], 1):
        metrics[f"validation/ordinary_mse/horizon_{horizon}"] = value
    for item in row["response"]["per_parameter"]:
        for key in ("normalized_response_error", "predictive_benefit"):
            estimate = item[key]
            prefix = f"validation/theta_{item['parameter']}/{key}"
            if estimate["estimate"] is not None:
                metrics[prefix] = estimate["estimate"]
            if estimate["interval_95"] is not None:
                metrics[prefix + "/lower95"], metrics[prefix + "/upper95"] = estimate["interval_95"]
    for index, value in enumerate(row["retention"]["selected_validation_r2"], 1):
        metrics[f"validation/theta_{index}/retention_r2"] = value
    return metrics


def paired_changes(
    before: dict[str, Tensor], after: dict[str, Tensor], *, samples: int, seed: int
) -> dict[str, Any]:
    """Bootstrap final-minus-initial changes over identical independent validation groups."""
    for key in ("group_indices", "query_indices", "finite_steps", "exact_response_energy"):
        if not torch.equal(before[key].cpu(), after[key].cpu()):
            raise ValueError("Paired comparison requires identical queries and reference responses")
    groups = len(before["group_indices"])
    indices = torch.randint(
        groups, (samples, groups), generator=torch.Generator().manual_seed(seed)
    )

    def interval(delta: Tensor, denominator: Tensor) -> dict[str, Any]:
        delta, denominator = delta.double().cpu(), denominator.double().cpu()
        draws = delta[indices].sum(1) / denominator[indices].sum(1)
        bounds = draws.quantile(torch.tensor([0.025, 0.975], dtype=torch.float64))
        return {
            "estimate": float(delta.sum() / denominator.sum()),
            "interval_95": [float(value) for value in bounds],
        }

    rows: list[dict[str, Any]] = []
    for factor in range(6):
        rows.append(
            {
                "parameter": factor + 1,
                "normalized_response_error_change": interval(
                    after["response_error_energy"][:, factor]
                    - before["response_error_energy"][:, factor],
                    before["exact_response_energy"][:, factor],
                ),
                "predictive_benefit_change": interval(
                    after["predictive_benefit"][:, factor]
                    - before["predictive_benefit"][:, factor],
                    torch.ones(groups),
                ),
            }
        )
    return {
        "direction": "final minus initial; negative response/MSE and positive benefit favor refit",
        "per_parameter": rows,
        "ordinary_mse_change_by_horizon": [
            interval(
                after["ordinary_mse_by_horizon"][:, h] - before["ordinary_mse_by_horizon"][:, h],
                torch.ones(groups),
            )
            for h in range(2)
        ],
    }


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Refit the predictor with label-free training and matched held-out diagnostics."""
    source, checkpoint, out = map(Path, (config.source_dir, config.checkpoint, config.output_dir))
    bundle = load_source_bundle(source)
    device = torch.device(config.device)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        with file_lock(out / ".lock"):
            model, optimizer = restore_source(
                bundle, checkpoint, device, verification=config.verification
            )
            frozen = fixed_hashes(model)
            settings = asdict(config)
            for key in ("source_dir", "checkpoint", "output_dir", "wandb_mode", "wandb_entity"):
                settings.pop(key)
            digest = hashlib.sha256(shared_code_identity().encode())
            for path in sorted(Path(__file__).parent.glob("*.py")):
                digest.update(path.read_bytes())
            for path in sorted((Path(__file__).parents[1] / "joint_acquisition").glob("*.py")):
                digest.update(path.read_bytes())
            evidence = {
                "settings": settings,
                "source_identity": bundle.manifest["identity"],
                "source_checkpoint_sha256": sha(checkpoint),
                "frozen_hashes": frozen,
                "code_identity": digest.hexdigest(),
                "torch": str(torch.__version__),
            }
            identity = canonical_fingerprint(evidence)
            protocol: dict[str, Any] = {
                "identity": identity,
                **evidence,
                "config": asdict(config),
                "hypothesis": "The source predictor may inadequately use retained all-six codes.",
                "controlled_change": (
                    "Only predictor weights update; original objective and AdamW moments retained."
                ),
                "decision": (
                    "Better responses support predictor limitation; no gain is limited "
                    "to 10k updates. Neither establishes fresh acquisition."
                ),
                "selection": (
                    "Fixed final checkpoint; validation is monitoring only; "
                    "test read once at final."
                ),
                "sampling": (
                    "New deterministic sampler schedule from update 1; "
                    "source Adam moments retain count 10000."
                ),
                "objective": bundle.objective_config,
            }
            if resume:
                if read_json(out / "protocol.json")["identity"] != identity:
                    raise ValueError("Resume settings, source or code identity changed")
            elif any(path.name != ".lock" for path in out.iterdir()):
                raise FileExistsError("Use a fresh output directory or explicit resume")
            else:
                write_json(out / "protocol.json", protocol)
            if (out / "completed.json").exists():
                completed = cast(dict[str, Any], read_json(out / "completed.json"))
                if completed["identity"] != identity or any(
                    sha(out / name) != value for name, value in completed["hashes"].items()
                ):
                    raise ValueError("Completed artifacts changed")
                return cast(dict[str, Any], read_json(out / "results.json"))
            records: list[dict[str, Any]] = []
            raw_records: dict[int, dict[str, Tensor]] = {}
            start = 0
            saved_tracking = (
                read_json(out / "tracking.json") if (out / "tracking.json").exists() else None
            )
            if resume and (out / "last.pt").exists():
                saved = cast(
                    dict[str, Any],
                    torch.load(out / "last.pt", map_location=device, weights_only=False),
                )
                if saved["identity"] != identity or not 0 <= saved["step"] <= config.steps:
                    raise ValueError("Checkpoint identity or step changed")
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                start, records, raw_records = saved["step"], saved["records"], saved["raw_records"]
                saved_tracking = saved["tracking"]
            if fixed_hashes(model) != frozen:
                raise ValueError("Checkpoint changed frozen source encoders")
            objective_config = ObjectiveConfig(**bundle.objective_config)
            objective = PredictiveSupervisionObjective(objective_config, 1.0).to(device)
            selection = DiagnosticSelection(
                tuple(range(config.groups)), tuple(range(config.groups))
            )
            geometry = copy.deepcopy(model).double().eval()
            matched = build_matched_batch(
                geometry, bundle.evaluation, selection, "validation"
            ).select_queries("heldout")
            del geometry

            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=f"predictor-adequacy-{out.name}",
                    job_type="diagnostic",
                ),
                resolved_config=protocol,
                dataset_identity=bundle.manifest["identity"],
                out_dir=out,
                resume_state=saved_tracking,
            )
            write_json(out / "tracking.json", logger.metadata)

            def check_frozen() -> None:
                if fixed_hashes(model) != frozen:
                    raise RuntimeError("Frozen source encoders changed")

            def assess(step: int) -> None:
                check_frozen()
                with preserve_rng(device):
                    candidate = copy.deepcopy(model).double().eval()
                    response, raw = evaluate_responses(
                        candidate,
                        matched,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                    )
                    retention = validation_retention(candidate, bundle.evaluation, selection)
                row = {"step": step, "response": response, "retention": retention}
                records.append(row)
                raw_records[step] = raw
                write_json(out / "validation.json", {"identity": identity, "records": records})
                logger.log(step, assessment_metrics(row))

            def save(step: int) -> None:
                check_frozen()
                atomic_save(
                    out / "last.pt",
                    {
                        "identity": identity,
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "records": records,
                        "raw_records": raw_records,
                        "tracking": logger.metadata,
                    },
                )

            latest_step = start
            try:
                if not records:
                    assess(0)
                    save(0)
                milestones = {0, 1000, 5000, config.steps}
                batches_per_epoch = bundle.train_groups // bundle.batch_size
                epoch_loaded, batches = -1, []
                parameters = tuple(model.predictor.parameters())
                for step in range(start + 1, config.steps + 1):
                    latest_step = step
                    epoch, position = divmod(step - 1, batches_per_epoch)
                    if epoch != epoch_loaded:
                        batches = list(
                            GlobalBatchSampler(
                                bundle.train_groups, bundle.batch_size, config.seed, epoch
                            )
                        )
                        epoch_loaded = epoch
                    batch = {
                        key: value[batches[position]].to(device)
                        for key, value in bundle.train.items()
                    }
                    torch.manual_seed(config.seed * 100000 + step)  # pyright: ignore[reportUnknownMemberType]
                    model.train()
                    for name in FIXED_MODULES:
                        getattr(model, name).eval()
                    optimizer.zero_grad(set_to_none=True)
                    output = cached_multiquery_forward(model, batch, queries=1)
                    loss, terms = objective(output)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"Nonfinite training loss at update {step}")
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
                                    f"train/{key}": float(value.detach())
                                    for key, value in terms.items()
                                },
                                "train/predictor_gradient_norm": norm,
                            },
                        )
                    if step in milestones:
                        assess(step)
                    if step % 1000 == 0 or step == config.steps:
                        save(step)
                with preserve_rng(device):
                    final = evaluate(
                        copy.deepcopy(model).cpu().eval(),
                        bundle.evaluation,
                        objective_config,
                        batch_size=bundle.batch_size,
                        final=True,
                        prediction_weight=1.0,
                        prediction_weights=(1.0,),
                    )
                result = {
                    "identity": identity,
                    "status": "software verification only"
                    if config.verification
                    else "conditional predictor refit; acquisition unresolved",
                    "steps": config.steps,
                    "records": records,
                    "final_test": final,
                    "tracking": logger.metadata,
                    "paired_validation_changes": paired_changes(
                        raw_records[0],
                        raw_records[config.steps],
                        samples=config.bootstrap_samples,
                        seed=config.seed,
                    ),
                }
                write_json(out / "results.json", result)
                atomic_save(out / "measurements.pt", {"records": raw_records})
                logger.log(config.steps, scalar_metrics(final, "test"))
                logger.log(config.steps, {"diagnostic/complete": 1.0})
                for name in ("results.json", "protocol.json"):
                    logger.log_artifact(
                        out / name, name=f"predictor-adequacy-{identity[:12]}-{Path(name).stem}"
                    )
                check_frozen()
                if sha(checkpoint) != evidence["source_checkpoint_sha256"]:
                    raise RuntimeError("Source checkpoint changed")
                if load_source_bundle(source).manifest["identity"] != bundle.manifest["identity"]:
                    raise RuntimeError("Source bundle identity changed")
                logger.finish()
                write_json(
                    out / "completed.json",
                    {
                        "identity": identity,
                        "hashes": {
                            name: sha(out / name)
                            for name in (
                                "last.pt",
                                "results.json",
                                "measurements.pt",
                                "protocol.json",
                            )
                        },
                    },
                )
                return cast(dict[str, Any], read_json(out / "results.json"))
            except BaseException as error:
                write_json(
                    out / "failure.json",
                    {"identity": identity, "attempted_step": latest_step, "error": str(error)},
                )
                logger.log(latest_step, {"diagnostic/failed": 1.0})
                logger.log_artifact(
                    out / "failure.json", name=f"predictor-adequacy-{identity[:12]}-failure"
                )
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    """Run only the predeclared control or explicit tiny software verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "checkpoint", "output-dir"):
        parser.add_argument("--" + name, required=True)
    for name, default in (
        ("seed", 202610261301),
        ("steps", 10000),
        ("groups", 64),
        ("bootstrap-samples", 1000),
        ("threads", 1),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--wandb-mode", default="disabled", choices=("disabled", "offline", "online")
    )
    parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--verification", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = vars(parser.parse_args())
    resume = bool(arguments.pop("resume"))
    run(Config(**arguments), resume=resume)


if __name__ == "__main__":
    main()
