"""Fit selected TRAIN-system codes with a fixed or jointly continued teacher predictor."""

import argparse
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from diagnostics.group_latent_control.forward import clip_teacher_gradients, group_teacher_forward
from diagnostics.group_latent_control.run import (
    _restore_rng,  # pyright: ignore[reportPrivateUsage]
    _rng,  # pyright: ignore[reportPrivateUsage]
)
from diagnostics.multiquery_control.data import TensorSplit
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective
from diagnostics.predictive_supervision_control.run import code_identity as shared_code_identity
from diagnostics.table_settling_control.evaluation import (
    forecast_errors,
    new_episodes,
    paired_change,
    readout,
)
from diagnostics.table_settling_control.source import (
    fixed_encoder_fingerprint,
    load_inputs,
    model_fingerprint,
)
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng

ObjectiveArm = Literal["original", "without_parameter_sigreg"]


class TableSettlingObjective(PredictiveSupervisionObjective):
    """Omit only parameter SIGReg when selected, preserving its original RNG draw."""

    def __init__(self, config: ObjectiveConfig, arm: ObjectiveArm = "original") -> None:
        """Retain the fixed original configuration and validate the isolated loss arm."""
        if arm not in ("original", "without_parameter_sigreg"):
            raise ValueError("Unknown table-settling objective arm")
        super().__init__(config, 1.0)
        self.arm = arm

    def forward(self, output: ModelOutput) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute matched components and explicitly distinguish optimized from original loss."""
        original, terms = super().forward(output)
        optimized = original
        effective_weight = self.config.parameter_regularizer_weight
        if self.arm == "without_parameter_sigreg":
            # Do not subtract a possibly dominant SIGReg gradient from a summed
            # float32 gradient. This graph contains only the retained components.
            optimized = (
                terms["weighted_prediction"]
                + self.config.logit_weight * terms["logit"]
                + self.config.parameter_consistency_weight * terms["parameter_consistency"]
            )
            effective_weight = 0.0
        terms["original_objective"] = original
        terms["optimized_objective"] = optimized
        terms["effective_parameter_regularizer_weight"] = original.new_tensor(effective_weight)
        return optimized, terms


@dataclass(frozen=True)
class Config:
    """One bounded label-free teacher-settling arm with optional predictor continuation."""

    curriculum_dir: str
    audit_dir: str
    output_dir: str
    device: str = "cpu"
    seed: int = 2026102703
    steps: int = 4000
    groups: int = 256
    bootstrap_samples: int = 1000
    threads: int = 1
    objective_arm: ObjectiveArm = "original"
    train_predictor: bool = False
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Fix scientific budget and require isolated outputs."""
        if self.objective_arm not in ("original", "without_parameter_sigreg"):
            raise ValueError("Unknown table-settling objective arm")
        if self.verification:
            if not 1 <= self.steps <= 3 or self.groups != 4:
                raise ValueError("Verification requires four groups and at most three steps")
        elif (self.steps, self.groups, self.bootstrap_samples) != (4000, 256, 1000):
            raise ValueError("Scientific protocol fixes 4000 updates, 256 groups, 1000 bootstraps")
        if min(self.seed, self.threads, self.bootstrap_samples) < 1:
            raise ValueError("Seeds, threads and bootstrap count must be positive")
        output = Path(self.output_dir).resolve()
        for name in (self.curriculum_dir, self.audit_dir):
            path = Path(name).resolve()
            if output == path or output in path.parents or path in output.parents:
                raise ValueError("Output must be separate from immutable sources")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def train_update(
    model: SCJEPA,
    table: nn.Embedding,
    optimizer: torch.optim.SparseAdam,
    batch: TensorSplit,
    indices: Tensor,
    objective: PredictiveSupervisionObjective,
    *,
    predictor_optimizer: torch.optim.Adam | None = None,
) -> dict[str, float]:
    """Train selected codes and optionally P; state and parameter encoders remain fixed."""
    if predictor_optimizer is None and any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise ValueError("All predictor and encoder weights must be frozen")
    if predictor_optimizer is not None:
        if any(
            parameter.requires_grad
            for module in (model.online_encoder, model.target_encoder, model.parameter_encoder)
            for parameter in module.parameters()
        ):
            raise ValueError("All three encoders must remain frozen")
        predictor_parameters = list(model.predictor.parameters())
        optimized = [
            parameter for group in predictor_optimizer.param_groups for parameter in group["params"]
        ]
        if not all(parameter.requires_grad for parameter in predictor_parameters) or (
            [id(parameter) for parameter in optimized]
            != [id(parameter) for parameter in predictor_parameters]
        ):
            raise ValueError("Joint optimizer must update exactly the predictor parameters")
        predictor_optimizer.zero_grad(set_to_none=True)
    if set(batch) != {"online", "target", "actions"}:
        raise ValueError("Optimization accepts label-free training tensors only")
    model.train()
    model.online_encoder.eval()
    model.target_encoder.eval()
    model.parameter_encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    output = group_teacher_forward(model, batch, table(indices))
    loss, terms = objective(output)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite table objective")
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    norm = clip_teacher_gradients(
        model.predictor.parameters() if predictor_optimizer is not None else (), table, 1.0
    )
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    if predictor_optimizer is not None:
        predictor_optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    return {**{key: float(value.detach()) for key, value in terms.items()}, "gradient_norm": norm}


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Continue selected codes and optional P while preserving sources and all encoders."""
    inputs = load_inputs(
        Path(config.curriculum_dir),
        Path(config.audit_dir),
        verification=config.verification,
        groups=config.groups,
        selection_seed=config.seed,
    )
    out = Path(config.output_dir)
    source = inputs.bundle.source_dir.resolve()
    if (
        out.resolve() == source
        or source in out.resolve().parents
        or out.resolve() in source.parents
    ):
        raise ValueError("Output must be separate from primary source")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    try:
        with file_lock(out / ".lock"):
            model = inputs.model.to(device).requires_grad_(False)
            frozen = model_fingerprint(model)
            frozen_encoders = fixed_encoder_fingerprint(model)
            predictor_optimizer = None
            if config.train_predictor:
                model.predictor.requires_grad_(True)
                predictor_optimizer = torch.optim.Adam(model.predictor.parameters(), lr=0.001)
                predictor_optimizer.load_state_dict(inputs.predictor_optimizer)
            selected = inputs.group_indices
            train = {key: value[selected].clone() for key, value in inputs.bundle.train.items()}
            table = nn.Embedding.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                inputs.table[selected].clone().to(device), freeze=False, sparse=True
            )  # pyright: ignore[reportUnknownMemberType]
            initial_table = table.weight.detach().cpu().clone()
            optimizer = torch.optim.SparseAdam(table.parameters(), lr=0.01)
            optimizer.load_state_dict(inputs.latent_optimizer)
            settings = asdict(config)
            for key in ("output_dir", "wandb_mode", "wandb_entity"):
                settings.pop(key)
            digest = hashlib.sha256(shared_code_identity().encode())
            for folder in (
                Path(__file__).parent,
                Path(__file__).parents[1] / "group_latent_control",
                Path(__file__).parents[1] / "joint_acquisition",
                Path(__file__).parents[1] / "encoder_gradient_control",
            ):
                for path in sorted(folder.glob("*.py")):
                    digest.update(path.read_bytes())
            digest.update((Path(__file__).parents[1] / "group_latent_table_audit.py").read_bytes())
            episode_seed = int(
                hashlib.sha256(
                    f"table-settling-evaluation-only:{config.seed}".encode()
                ).hexdigest()[:15],
                16,
            )
            evidence: dict[str, Any] = {
                "settings": settings,
                "source_hashes": inputs.file_hashes,
                "selected_group_ids": selected.tolist(),  # pyright: ignore[reportUnknownMemberType]
                "frozen_model_hash": None if config.train_predictor else frozen,
                "initial_model_hash": frozen,
                "fixed_encoder_hash": frozen_encoders,
                "predictor_parameter_order": [
                    name for name, _ in model.predictor.named_parameters()
                ],
                "initial_table_hash": tensor_fingerprint(initial_table),
                "episode_seed": episode_seed,
                "code_identity": digest.hexdigest(),
                "torch": str(torch.__version__),
            }
            identity = canonical_fingerprint(evidence)
            protocol: dict[str, Any] = {
                "identity": identity,
                **evidence,
                "config": asdict(config),
                "hypothesis": (
                    "Joint code/predictor plasticity enables transfer unavailable under fixed P."
                    if config.train_predictor
                    else "Free train-group codes are under-settled relative to the fixed predictor."
                    if config.objective_arm == "original"
                    else "Parameter SIGReg impedes transferable settling under the fixed predictor."
                ),
                "controlled_change": (
                    "Selected train-code rows receive additional "
                    "SparseAdam .01 updates with source moment rows and global step retained; "
                    + (
                        "predictor Adam .001 also continues original moments/global step; "
                        "one combined predictor/table gradient clip1; all encoders fixed. "
                        if config.train_predictor
                        else "global clip1; all model weights fixed. "
                    )
                    + (
                        "Optimize the original objective unchanged."
                        if config.objective_arm == "original"
                        else "Omit only parameter SIGReg from the optimized loss; compute the "
                        "same original forward and preserve its projection RNG consumption."
                    )
                ),
                "decision": (
                    "Improvement on cached and new episodes supports transferable settling; "
                    "cached-only improvement suggests nuisance fitting; neither does not "
                    "exclude joint underoptimization. For the omission arm, compare paired "
                    "system forecasts and missing-factor readouts against the original arm "
                    "started from the same teacher checkpoint, moments and seeds. Forecast "
                    "improvement without retention does not establish acquisition."
                ),
                "scope": (
                    "Already-trained systems only. 128/64/64 readout is low-power diagnostic "
                    "evidence, not unseen-system all-six acceptance."
                ),
                "milestones": [0, 500, 2000, 4000]
                if not config.verification
                else list(range(config.steps + 1)),
                "evaluation": (
                    "New episodes from a domain-separated persisted seed never enter "
                    "optimization; no encoder training or early selection."
                ),
                "objective": inputs.bundle.protocol["objective_config"],
                "effective_parameter_regularizer_weight": (
                    inputs.bundle.protocol["objective_config"]["parameter_regularizer_weight"]
                    if config.objective_arm == "original"
                    else 0.0
                ),
            }
            if resume:
                if read_json(out / "protocol.json")["identity"] != identity:
                    raise ValueError("Resume settings, code or sources changed")
            elif any(path.name != ".lock" for path in out.iterdir()):
                raise FileExistsError("Use a fresh output or explicit resume")
            else:
                write_json(out / "protocol.json", protocol)
                with preserve_rng(device):
                    fresh = new_episodes(
                        model,
                        inputs.theta_evaluation_only[selected],
                        inputs.bundle.evaluation["metadata"]["observation_matrix"],
                        episode_seed,
                    )
                if torch.equal(fresh["actions"], train["actions"]):
                    raise ValueError("Evaluation episodes duplicate cached training episodes")
                atomic_save(
                    out / "evaluation_only.pt",
                    {
                        "identity": identity,
                        "episode_seed": episode_seed,
                        "group_indices": selected,
                        "new_episodes": fresh,
                        "theta_evaluation_only": inputs.theta_evaluation_only[selected],
                    },
                )
                write_json(
                    out / "evaluation_manifest.json",
                    {"identity": identity, "sha256": sha(out / "evaluation_only.pt")},
                )
            manifest = read_json(out / "evaluation_manifest.json")
            if manifest["identity"] != identity or manifest["sha256"] != sha(
                out / "evaluation_only.pt"
            ):
                raise ValueError("Persisted evaluation-only episodes changed")
            evaluation = cast(
                dict[str, Any], torch.load(out / "evaluation_only.pt", weights_only=False)
            )
            if evaluation["identity"] != identity or not torch.equal(
                evaluation["group_indices"], selected
            ):
                raise ValueError("Evaluation identity or group selection changed")
            if (out / "completed.json").exists():
                completion = cast(dict[str, Any], read_json(out / "completed.json"))
                if completion["identity"] != identity or any(
                    sha(out / name) != value for name, value in completion["hashes"].items()
                ):
                    raise ValueError("Completed diagnostic artifacts changed")
                return cast(dict[str, Any], read_json(out / "results.json"))
            start = 0
            records: list[dict[str, Any]] = []
            raw: dict[int, dict[str, Tensor]] = {}
            tracking = (
                read_json(out / "tracking.json") if (out / "tracking.json").exists() else None
            )
            if resume and (out / "last.pt").exists():
                checkpoint = cast(
                    dict[str, Any],
                    torch.load(out / "last.pt", map_location="cpu", weights_only=False),
                )
                if (
                    checkpoint["identity"] != identity
                    or checkpoint["evaluation_sha256"] != manifest["sha256"]
                ):
                    raise ValueError("Checkpoint source/evaluation mismatch")
                table.load_state_dict(checkpoint["table"], strict=True)
                optimizer.load_state_dict(checkpoint["optimizer"])
                if predictor_optimizer is not None:
                    if (
                        checkpoint.get("predictor") is None
                        or checkpoint.get("predictor_optimizer") is None
                    ):
                        raise ValueError("Joint checkpoint lacks predictor or Adam state")
                    model.predictor.load_state_dict(checkpoint["predictor"], strict=True)
                    predictor_optimizer.load_state_dict(checkpoint["predictor_optimizer"])
                elif (
                    checkpoint.get("predictor") is not None
                    or checkpoint.get("predictor_optimizer") is not None
                ):
                    raise ValueError("Frozen control cannot resume a joint checkpoint")
                start, records, raw = checkpoint["step"], checkpoint["records"], checkpoint["raw"]
                tracking = checkpoint["tracking"]
                _restore_rng(checkpoint["rng"], device)
            if not 0 <= start <= config.steps:
                raise ValueError("Invalid checkpoint step")
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=f"table-settling-{out.parent.name}",
                    job_type="diagnostic",
                ),
                resolved_config=protocol,
                dataset_identity=inputs.bundle.protocol["identity"],
                out_dir=out,
                resume_state=tracking,
            )
            write_json(out / "tracking.json", logger.metadata)
            objective = TableSettlingObjective(
                ObjectiveConfig(**inputs.bundle.protocol["objective_config"]), config.objective_arm
            ).to(device)
            batch_size = inputs.bundle.batch_size
            if not config.verification and batch_size != 16:
                raise ValueError("Scientific protocol requires B16")

            def save(step: int) -> None:
                if fixed_encoder_fingerprint(model) != frozen_encoders:
                    raise RuntimeError("Frozen encoders or rollout flags changed")
                if not config.train_predictor and model_fingerprint(model) != frozen:
                    raise RuntimeError("Frozen model changed")
                atomic_save(
                    out / "last.pt",
                    {
                        "identity": identity,
                        "step": step,
                        "table": table.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "predictor": model.predictor.state_dict()
                        if config.train_predictor
                        else None,
                        "predictor_optimizer": predictor_optimizer.state_dict()
                        if predictor_optimizer is not None
                        else None,
                        "records": records,
                        "raw": raw,
                        "rng": _rng(device),
                        "tracking": logger.metadata,
                        "evaluation_sha256": manifest["sha256"],
                    },
                )

            def assess(step: int) -> None:
                with preserve_rng(device):
                    cached = forecast_errors(model, table, train, batch_size)
                    fresh = forecast_errors(model, table, evaluation["new_episodes"], batch_size)
                    retention = readout(table, evaluation["theta_evaluation_only"])
                raw[step] = {"cached": cached, "new_episodes": fresh}
                changes = {
                    name: paired_change(
                        raw[0][name], values, samples=config.bootstrap_samples, seed=config.seed
                    )
                    for name, values in raw[step].items()
                }
                row: dict[str, Any] = {
                    "step": step,
                    "cached_horizon_mse": cached.mean(0).tolist(),  # pyright: ignore[reportUnknownMemberType]
                    "new_episode_horizon_mse": fresh.mean(0).tolist(),  # pyright: ignore[reportUnknownMemberType]
                    "paired_changes": changes,
                    "readout": retention,
                    "table_movement_rms": float(
                        (table.weight.detach().cpu() - initial_table).square().mean().sqrt()
                    ),
                }
                records.append(row)
                metrics = {
                    f"forecast/{name}/horizon_{h + 1}": float(values.mean(0)[h])
                    for name, values in raw[step].items()
                    for h in range(2)
                }
                for name, result in changes.items():
                    for h in range(2):
                        metrics[f"change/{name}/horizon_{h + 1}"] = result["mean"][h]
                        for i, bound in enumerate(("lower", "upper")):
                            metrics[f"change/{name}/horizon_{h + 1}_{bound}"] = result["ci95"][h][i]
                for i, value in enumerate(retention["selected_test_r2"]):
                    metrics[f"trained_groups/theta_{i + 1}/readout_r2"] = float(value)
                metrics["table/movement_rms"] = row["table_movement_rms"]
                logger.log(step, metrics)
                write_json(out / "measurements.json", {"identity": identity, "records": records})

            step = start
            try:
                if not records:
                    assess(0)
                    save(0)
                epoch_cached, batches = -1, []
                for step in range(start + 1, config.steps + 1):
                    epoch, position = divmod(step - 1, config.groups // batch_size)
                    if epoch != epoch_cached:
                        batches = list(
                            GlobalBatchSampler(config.groups, batch_size, config.seed, epoch)
                        )
                        epoch_cached = epoch
                    indices = torch.tensor(batches[position], device=device)
                    batch = {
                        key: value[batches[position]].to(device) for key, value in train.items()
                    }
                    torch.manual_seed(config.seed * 100000 + step)  # pyright: ignore[reportUnknownMemberType]
                    terms = train_update(
                        model,
                        table,
                        optimizer,
                        batch,
                        indices,
                        objective,
                        **(
                            {"predictor_optimizer": predictor_optimizer}
                            if predictor_optimizer is not None
                            else {}
                        ),
                    )
                    if step == 1 or step % 100 == 0 or step == config.steps:
                        logger.log(step, {f"train/{key}": value for key, value in terms.items()})
                    if step in protocol["milestones"]:
                        assess(step)
                    if config.verification or step % 500 == 0 or step == config.steps:
                        save(step)
                for name, expected in inputs.file_hashes.items():
                    if sha(Path(name)) != expected:
                        raise RuntimeError("Immutable source changed during table optimization")
                if sha(out / "evaluation_only.pt") != manifest["sha256"]:
                    raise RuntimeError("Evaluation-only episodes changed during optimization")
                result = {
                    "identity": identity,
                    "objective_arm": config.objective_arm,
                    "train_predictor": config.train_predictor,
                    "step": step,
                    "scope": protocol["scope"],
                    "records": records,
                    "tracking": logger.metadata,
                }
                write_json(out / "results.json", result)
                atomic_save(out / "group_errors.pt", {"identity": identity, "raw": raw})
                for name in ("results.json", "protocol.json"):
                    logger.log_artifact(
                        out / name, name=f"table-settling-{identity[:12]}-{Path(name).stem}"
                    )
                logger.log(step, {"diagnostic/complete": 1.0})
                logger.finish()
                write_json(
                    out / "completed.json",
                    {
                        "identity": identity,
                        "hashes": {
                            name: sha(out / name)
                            for name in (
                                "protocol.json",
                                "results.json",
                                "last.pt",
                                "group_errors.pt",
                                "evaluation_only.pt",
                                "evaluation_manifest.json",
                                "measurements.json",
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
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Run the isolated fixed-predictor teacher-code diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("curriculum-dir", "audit-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--objective-arm",
        choices=("original", "without_parameter_sigreg"),
        default="original",
    )
    for name, default in (
        ("seed", 2026102703),
        ("steps", 4000),
        ("groups", 256),
        ("bootstrap-samples", 1000),
        ("threads", 1),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--verification", action="store_true")
    parser.add_argument("--train-predictor", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    resume = bool(args.pop("resume"))
    run(Config(**args), resume=resume)


if __name__ == "__main__":
    main()
