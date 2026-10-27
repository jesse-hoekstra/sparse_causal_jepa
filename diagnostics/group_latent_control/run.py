"""Train-only latent teacher, single-context amortization, then ordinary joint washout."""

import argparse
import copy
import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn

from diagnostics.encoder_gradient_control.source import SourceBundle, build_source_model
from diagnostics.group_latent_control.forward import clip_teacher_gradients, group_teacher_forward
from diagnostics.joint_acquisition.baseline import (
    DiagnosticSelection,
    build_matched_batch,
    evaluate_responses,
    validation_retention,
)
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.multiquery_control.run import atomic_save, frozen_hashes, scalar_metrics, sha
from diagnostics.predictive_supervision_control.evaluation import evaluate
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective
from diagnostics.predictive_supervision_control.run import code_identity as shared_code_identity
from diagnostics.predictor_adequacy.run import assessment_metrics, paired_changes
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng

Phase = Literal["teacher", "amortization", "washout"]
Initialization = Literal["warm", "fresh_zero"]
TeacherObjectiveArm = Literal["original", "without_parameter_sigreg"]
PHASES: tuple[Phase, ...] = ("teacher", "amortization", "washout")


@dataclass(frozen=True)
class Config:
    """Fixed phase budgets; any scientific setting change requires a separate control."""

    source_dir: str
    output_dir: str
    training_source_root: str | None = None
    initialization: Initialization = "warm"
    teacher_objective_arm: TeacherObjectiveArm = "original"
    seed: int = 202610261301
    device: str = "cpu"
    teacher_steps: int = 30000
    amortization_steps: int = 10000
    washout_steps: int = 10000
    groups: int = 64
    bootstrap_samples: int = 1000
    threads: int = 1
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Protect source artifacts and reject ambiguous scientific budgets."""
        budgets = (self.teacher_steps, self.amortization_steps, self.washout_steps)
        if self.initialization not in ("warm", "fresh_zero"):
            raise ValueError("Unknown curriculum initialization")
        if self.teacher_objective_arm not in ("original", "without_parameter_sigreg"):
            raise ValueError("Unknown teacher objective arm")
        if self.verification:
            if not all(1 <= value <= 3 for value in budgets) or not 2 <= self.groups <= 4:
                raise ValueError(
                    "Verification permits one to three updates per phase and 2..4 groups"
                )
        elif (
            budgets != (30000, 10000, 10000) or self.groups != 64 or self.bootstrap_samples != 1000
        ):
            raise ValueError("Scientific control requires 30000/10000/10000 updates and 64 groups")
        if self.seed < 0 or self.threads < 1 or self.bootstrap_samples < 2:
            raise ValueError("Invalid seed, threads or bootstrap count")
        source, out = Path(self.source_dir).resolve(), Path(self.output_dir).resolve()
        if source == out or source in out.parents or out in source.parents:
            raise ValueError("Output must be separate from the immutable source directory")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)

    @property
    def total_steps(self) -> int:
        """Return the final fixed-budget checkpoint's cumulative update count."""
        return self.teacher_steps + self.amortization_steps + self.washout_steps


def initialize_table(model: SCJEPA, train: TensorSplit, batch_size: int) -> nn.Embedding:
    """Infer one donor-zero code per training group, without labels or context averaging."""
    if set(train) != {"online", "target", "actions"}:
        raise ValueError("Training cache must contain only label-free states and actions")
    device = next(model.parameters()).device
    codes: list[Tensor] = []
    with preserve_rng(device), torch.no_grad():
        model.eval()
        for start in range(0, len(train["online"]), batch_size):
            states = train["online"][start : start + batch_size, 0, :10].to(device)
            actions = train["actions"][start : start + batch_size, 0, :9].to(device)
            codes.append(model.parameter_encoder(states, actions).detach())
    values = torch.cat(codes)
    if values.shape != (len(train["online"]), 6) or not bool(torch.isfinite(values).all()):
        raise ValueError(
            "Teacher initialization must be one finite six-dimensional code per train group"
        )
    return nn.Embedding.from_pretrained(values.clone(), freeze=False, sparse=True)  # pyright: ignore[reportUnknownMemberType]


def initialize_control(
    model: SCJEPA, bundle: SourceBundle, initialization: Initialization
) -> tuple[nn.Embedding, dict[str, Any], dict[str, str]]:
    """Authenticate fresh learned modules while retaining the informative frozen states."""
    source_hashes = dict(bundle.hashes)
    initial_file: dict[str, str] | None = None
    if initialization == "fresh_zero":
        record = bundle.protocol["initializations"][str(bundle.seed)]
        name = record["path"]
        path = bundle.source_dir / name
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or (bundle.source_dir.resolve() not in path.resolve().parents)
        ):
            raise ValueError("Initialization must be inside the immutable source directory")
        actual = sha(path)
        if actual != record["sha256"] or actual != bundle.protocol["artifacts"].get(name):
            raise ValueError("Initialization artifact hash differs from source protocol")
        payload = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))
        if (
            set(payload) != {"model", "seed", "source_sha256"}
            or payload["seed"] != bundle.seed
            or payload["source_sha256"] != bundle.protocol["source"]["sha256"]
        ):
            raise ValueError("Initialization provenance differs from source protocol")
        candidate = copy.deepcopy(model)
        candidate.load_state_dict(payload["model"], strict=True)
        if (
            frozen_hashes(candidate) != frozen_hashes(model)
            or candidate.get_extra_state() != model.get_extra_state()
        ):
            raise ValueError("Initialization changes frozen states or rollout settings")
        model.parameter_encoder.load_state_dict(
            candidate.parameter_encoder.state_dict(), strict=True
        )
        model.predictor.load_state_dict(candidate.predictor.state_dict(), strict=True)
        values = next(model.parameters()).new_zeros((bundle.train_groups, 6))
        table = nn.Embedding.from_pretrained(values, freeze=False, sparse=True)  # pyright: ignore[reportUnknownMemberType]
        source_hashes[name] = actual
        initial_file = {"path": name, "sha256": actual}
    elif initialization == "warm":
        table = initialize_table(model, bundle.train, bundle.batch_size)
    else:
        raise ValueError("Unknown curriculum initialization")
    return (
        table,
        {
            "mode": initialization,
            "source_initialization": initial_file,
            "parameter_encoder_hash": module_fingerprint(model.parameter_encoder),
            "predictor_hash": module_fingerprint(model.predictor),
        },
        source_hashes,
    )


def phase_optimizers(
    model: SCJEPA, phase: Phase, table: nn.Embedding | None
) -> tuple[torch.optim.Adam, torch.optim.SparseAdam | None]:
    """Start fresh, explicit phase optimizers and train exactly the requested blocks."""
    if phase not in PHASES:
        raise ValueError("Unknown training phase")
    if (phase == "washout") != (table is None):
        raise ValueError("Only washout discards the train-group code table")
    model.requires_grad_(False)
    model.parameter_encoder.requires_grad_(phase != "teacher")
    model.predictor.requires_grad_(phase != "amortization")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    if table is not None:
        table.weight.requires_grad_(phase == "teacher")
        table.weight.grad = None
    latent_optimizer = (
        torch.optim.SparseAdam(table.parameters(), lr=0.01)
        if phase == "teacher" and table is not None
        else None
    )
    return optimizer, latent_optimizer


def training_loss(
    model: SCJEPA,
    batch: TensorSplit,
    indices: Tensor,
    phase: Phase,
    table: nn.Embedding | None,
    objective: PredictiveSupervisionObjective,
    *,
    teacher_objective_arm: TeacherObjectiveArm = "original",
) -> tuple[Tensor, dict[str, Tensor]]:
    """Use group IDs only for train codes; washout has no auxiliary target or table."""
    if set(batch) != {"online", "target", "actions"}:
        raise ValueError("Training receives no physical parameters or reference maps")
    if teacher_objective_arm not in ("original", "without_parameter_sigreg"):
        raise ValueError("Unknown teacher objective arm")
    if phase == "teacher":
        if table is None:
            raise ValueError("Teacher phase needs train-group codes")
        output = group_teacher_forward(model, batch, table(indices))
    else:
        output = cached_multiquery_forward(model, batch, queries=1)
    loss, terms = objective(output)
    original = loss
    effective_weight = objective.config.parameter_regularizer_weight
    if phase == "teacher" and teacher_objective_arm == "without_parameter_sigreg":
        # Evaluate the original objective for matched RNG and reporting, but do not
        # subtract its potentially dominant SIGReg gradient from a float32 sum.
        loss = (
            terms["weighted_prediction"]
            + objective.config.logit_weight * terms["logit"]
            + objective.config.parameter_consistency_weight * terms["parameter_consistency"]
        )
        effective_weight = 0.0
    if phase == "amortization":
        if table is None:
            raise ValueError("Amortization needs detached train-only targets")
        targets = table(indices).detach()[:, None]
        terms["code_amortization"] = (output.parameters - targets).square().mean()
        loss = loss + terms["code_amortization"]
        original = original + terms["code_amortization"]
    elif phase == "washout" and table is not None:
        raise ValueError("Washout must discard the teacher table")
    terms["optimized_total"] = loss
    terms["original_total"] = original
    terms["effective_parameter_regularizer_weight"] = loss.new_tensor(effective_weight)
    return loss, terms


@torch.no_grad()
def scale_metrics(model: SCJEPA, table: nn.Embedding | None) -> dict[str, float]:
    """Read full TRAIN-table scale and predictor input scale without RNG or state mutation."""
    metrics = {
        "scale/predictor_param_project_weight_norm": float(
            model.predictor.param_project.weight.detach().double().square().sum().sqrt()
        ),
        "scale/teacher_table_available": float(table is not None),
    }
    if table is not None:
        values = table.weight.detach().double().cpu()
        centered = values - values.mean(0)
        covariance = centered.T @ centered / (len(values) - 1)
        metrics["scale/teacher_table_rms"] = float(values.square().mean().sqrt())
        eigenvalues = cast(Tensor, torch.linalg.eigvalsh(covariance))  # pyright: ignore[reportUnknownMemberType]
        for index, value in enumerate(eigenvalues):
            metrics[f"scale/teacher_table_covariance_eigenvalue_{index + 1}"] = float(value)
    return metrics


def assess_model(
    model: SCJEPA,
    evaluation: EvaluationData,
    selection: DiagnosticSelection,
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Assess the original single-context encoder only; no table lookup is accepted."""
    device = next(model.parameters()).device
    with preserve_rng(device):
        candidate = copy.deepcopy(model).double().eval()
        matched = build_matched_batch(
            candidate, evaluation, selection, "validation"
        ).select_queries("heldout")
        response, raw = evaluate_responses(
            candidate, matched, bootstrap_samples=bootstrap_samples, seed=seed
        )
        retention = validation_retention(candidate, evaluation, selection)
        # Detailed family fits are reproducible from the saved checkpoint; retain selected scores.
        retention.pop("families")
        return {"response": response, "retention": retention}, raw


def _rng(device: torch.device) -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Execute the predeclared teacher, amortization and washout phases sequentially."""
    source, out = Path(config.source_dir), Path(config.output_dir)
    device = torch.device(config.device)
    source_step = 2 if config.verification else 30000
    bundle = load_primary_source(
        source,
        config.seed,
        expected_step=source_step,
        **({"expected_checkpoint_sha256": None} if config.verification else {}),
        training_source_root=Path(config.training_source_root)
        if config.training_source_root
        else None,
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        with file_lock(out / ".lock"):
            model, discarded_optimizer = build_source_model(bundle, device)
            del discarded_optimizer
            frozen = frozen_hashes(model)
            extra_state = model.get_extra_state()
            table: nn.Embedding | None
            table, initialization, source_hashes = initialize_control(
                model, bundle, config.initialization
            )
            initial_table_hash = module_fingerprint(table)
            initial_encoder_hash = module_fingerprint(model.parameter_encoder)
            settings = asdict(config)
            for key in (
                "source_dir",
                "output_dir",
                "training_source_root",
                "wandb_mode",
                "wandb_entity",
            ):
                settings.pop(key)
            digest = hashlib.sha256(shared_code_identity().encode())
            for folder in (
                Path(__file__).parent,
                Path(__file__).parents[1] / "joint_acquisition",
                Path(__file__).parents[1] / "predictor_adequacy",
            ):
                for path in sorted(folder.glob("*.py")):
                    digest.update(path.read_bytes())
            evidence = {
                "settings": settings,
                "source_hashes": source_hashes,
                "initialization": initialization,
                "source_step": source_step,
                "initial_table_hash": initial_table_hash,
                "initial_encoder_hash": initial_encoder_hash,
                "frozen_hashes": frozen,
                "code_identity": digest.hexdigest(),
                "torch": str(torch.__version__),
            }
            identity = canonical_fingerprint(evidence)
            protocol: dict[str, Any] = {
                "identity": identity,
                **evidence,
                "config": asdict(config),
                "hypothesis": (
                    "Removing teacher parameter SIGReg from update1 may permit transferable "
                    "system-code learning instead of early nuisance-code variation."
                    if config.teacher_objective_arm == "without_parameter_sigreg"
                    else "Train-only latent optimization can bypass the encoder learning path; "
                    "one-context amortization can transfer the result. Fresh-zero mode "
                    "removes the trained encoder/predictor basin and random table nuisance."
                ),
                "controlled_change": (
                    "Teacher group-code fitting, then label-free auxiliary code "
                    "amortization, then original joint objective; architecture/DGP/state "
                    "maps unchanged."
                ),
                "teacher_objective_arm": config.teacher_objective_arm,
                "teacher_effective_parameter_regularizer_weight": (
                    0.0
                    if config.teacher_objective_arm == "without_parameter_sigreg"
                    else bundle.protocol["objective_config"]["parameter_regularizer_weight"]
                ),
                "teacher_objective_change": (
                    "Teacher only: omit parameter SIGReg from optimized loss from update1; "
                    "compute original forward/projection RNG, optimize forecast+logit+consistency "
                    "directly. Restore original .4 parameter SIGReg in amortization/washout."
                    if config.teacher_objective_arm == "without_parameter_sigreg"
                    else "Original objective in all phases."
                ),
                "decision": (
                    "Held-out single-context all-six retention/use supports transfer; low "
                    "teacher training error alone is insufficient and may reflect group "
                    "memorization."
                ),
                "training_labels_absent": True,
                "source_optimizer_retained": False,
                "algorithm": {
                    "teacher": (
                        "One 6D sparse embedding per train group; "
                        + (
                            "verified fresh encoder/predictor and zero codes; "
                            if config.initialization == "fresh_zero"
                            else "source encoder/predictor and donor0 code initialization; "
                        )
                        + "fresh predictor Adam .001 and SparseAdam .01, combined gradient clip1; "
                        "forecast+logit+selected teacher SIGReg; consistency zero by construction "
                        "up to floating-point roundoff."
                    ),
                    "amortization": (
                        "Original encoder only, fresh Adam .001 clip1; mean square to detached "
                        "learned group code plus full original objective; predictor fixed."
                    ),
                    "washout": (
                        "Discard table and auxiliary loss; original encoder+predictor, fresh "
                        "Adam .001 clip1."
                    ),
                    "sampling": (
                        "New deterministic shuffled train-group schedule per phase; per-global- "
                        "update RNG seed; no held-out group lookup or inference-time "
                        "optimization."
                    ),
                    "evaluation": (
                        "Original encoder only at every milestone; calibration fits probes, "
                        "validation monitors; test only at final total budget, no early "
                        "stopping."
                    ),
                },
                "objective": bundle.protocol["objective_config"],
            }
            if resume:
                if read_json(out / "protocol.json")["identity"] != identity:
                    raise ValueError("Resume source, settings or code changed")
            elif any(path.name != ".lock" for path in out.iterdir()):
                raise FileExistsError("Use a new output directory or explicit resume")
            else:
                write_json(out / "protocol.json", protocol)
                atomic_save(
                    out / "initial_group_codes.pt",
                    {"weight": table.weight.detach().cpu(), "identity": identity},
                )
            if (out / "completed.json").exists():
                completed = cast(dict[str, Any], read_json(out / "completed.json"))
                if completed["identity"] != identity or any(
                    sha(out / name) != value for name, value in completed["hashes"].items()
                ):
                    raise ValueError("Completed artifacts changed")
                return cast(dict[str, Any], read_json(out / "results.json"))
            saved: dict[str, Any] | None = None
            records: list[dict[str, Any]] = []
            raw_records: dict[int, dict[str, Tensor]] = {}
            start_phase, start_step, global_step = 0, 0, 0
            tracking = (
                read_json(out / "tracking.json") if (out / "tracking.json").exists() else None
            )
            if resume and (out / "last.pt").exists():
                saved = cast(
                    dict[str, Any],
                    torch.load(out / "last.pt", map_location=device, weights_only=False),
                )
                if saved["identity"] != identity:
                    raise ValueError("Checkpoint belongs to another control")
                start_phase = PHASES.index(saved["phase"])
                start_step, global_step = saved["phase_step"], saved["global_step"]
                model.load_state_dict(saved["model"], strict=True)
                if start_phase == 2:
                    if saved["table"] is not None:
                        raise ValueError("Washout checkpoint must not contain teacher codes")
                    table = None
                else:
                    assert table is not None
                    table.load_state_dict(saved["table"], strict=True)
                records, raw_records, tracking = (
                    saved["records"],
                    saved["raw_records"],
                    saved["tracking"],
                )
                _restore_rng(saved["rng"], device)
            if frozen_hashes(model) != frozen or model.get_extra_state() != extra_state:
                raise ValueError("Checkpoint changed frozen states or rollout settings")
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=f"group-latent-{out.name}",
                    job_type="diagnostic",
                ),
                resolved_config=protocol,
                dataset_identity=bundle.protocol["identity"],
                out_dir=out,
                resume_state=tracking,
            )
            write_json(out / "tracking.json", logger.metadata)
            objective_config = ObjectiveConfig(**bundle.protocol["objective_config"])
            objective = PredictiveSupervisionObjective(objective_config, 1.0).to(device)
            selection = DiagnosticSelection(
                tuple(range(config.groups)), tuple(range(config.groups))
            )

            def check_states() -> None:
                if frozen_hashes(model) != frozen or model.get_extra_state() != extra_state:
                    raise RuntimeError("Frozen state encoders or rollout settings changed")

            def assess(phase: Phase, phase_step: int) -> None:
                check_states()
                measured, raw = assess_model(
                    model,
                    bundle.evaluation,
                    selection,
                    bootstrap_samples=config.bootstrap_samples,
                    seed=config.seed,
                )
                row: dict[str, Any] = {
                    "global_step": global_step,
                    "phase": phase,
                    "phase_step": phase_step,
                    **measured,
                    "scale": scale_metrics(model, table),
                }
                records.append(row)
                raw_records[global_step] = raw
                logger.log(
                    global_step,
                    {
                        **assessment_metrics(row),
                        **row["scale"],
                        "phase/index": float(PHASES.index(phase)),
                        "phase/update": float(phase_step),
                    },
                )
                write_json(out / "validation.json", {"identity": identity, "records": records})

            try:
                for phase_index in range(start_phase, len(PHASES)):
                    phase = PHASES[phase_index]
                    budget = int(getattr(config, phase + "_steps"))
                    phase_start = start_step if phase_index == start_phase else 0
                    if phase == "washout":
                        table = None
                    optimizer, latent_optimizer = phase_optimizers(model, phase, table)
                    if saved is not None and phase_index == start_phase:
                        optimizer.load_state_dict(saved["optimizer"])
                        if latent_optimizer is not None:
                            latent_optimizer.load_state_dict(saved["latent_optimizer"])
                    if (
                        not 0 <= phase_start <= budget
                        or global_step
                        != sum(int(getattr(config, p + "_steps")) for p in PHASES[:phase_index])
                        + phase_start
                    ):
                        raise ValueError("Checkpoint phase counters are inconsistent")
                    phase_frozen = (
                        module_fingerprint(
                            model.parameter_encoder if phase == "teacher" else model.predictor
                        )
                        if phase != "washout"
                        else None
                    )
                    table_frozen = (
                        module_fingerprint(table)
                        if phase == "amortization" and table is not None
                        else None
                    )

                    def save(
                        phase_step: int,
                        total_step: int,
                        *,
                        boundary: bool = False,
                        saved_phase: Phase = phase,
                        phase_optimizer: torch.optim.Adam = optimizer,
                        phase_latent_optimizer: torch.optim.SparseAdam | None = latent_optimizer,
                        phase_table: nn.Embedding | None = table,
                    ) -> None:
                        check_states()
                        atomic_save(
                            out / "last.pt",
                            {
                                "identity": identity,
                                "phase": saved_phase,
                                "phase_step": phase_step,
                                "global_step": total_step,
                                "model": model.state_dict(),
                                "optimizer": phase_optimizer.state_dict(),
                                "latent_optimizer": phase_latent_optimizer.state_dict()
                                if phase_latent_optimizer
                                else None,
                                "table": phase_table.state_dict()
                                if phase_table is not None
                                else None,
                                "rng": _rng(device),
                                "records": records,
                                "raw_records": raw_records,
                                "tracking": logger.metadata,
                            },
                        )
                        if boundary:
                            atomic_save(
                                out / (saved_phase + "_final.pt"),
                                cast(
                                    dict[str, Any],
                                    torch.load(
                                        out / "last.pt", map_location="cpu", weights_only=False
                                    ),
                                ),
                            )

                    if not records:
                        assess(phase, 0)
                        save(0, global_step)
                    batches_per_epoch = bundle.train_groups // bundle.batch_size
                    loaded_epoch, batches = -1, []
                    for phase_step in range(phase_start + 1, budget + 1):
                        epoch, position = divmod(phase_step - 1, batches_per_epoch)
                        if epoch != loaded_epoch:
                            batches = list(
                                GlobalBatchSampler(
                                    bundle.train_groups,
                                    bundle.batch_size,
                                    config.seed + phase_index,
                                    epoch,
                                )
                            )
                            loaded_epoch = epoch
                        selected = batches[position]
                        batch = {
                            key: value[selected].to(device) for key, value in bundle.train.items()
                        }
                        indices = torch.tensor(selected, device=device, dtype=torch.long)
                        global_step += 1
                        torch.manual_seed(config.seed * 100000 + global_step)  # pyright: ignore[reportUnknownMemberType]
                        model.train()
                        model.online_encoder.eval()
                        model.target_encoder.eval()
                        if phase == "teacher":
                            model.parameter_encoder.eval()
                        if phase == "amortization":
                            model.predictor.eval()
                        model.zero_grad(set_to_none=True)
                        optimizer.zero_grad(set_to_none=True)
                        if latent_optimizer:
                            latent_optimizer.zero_grad(set_to_none=True)
                        objective_options: dict[str, TeacherObjectiveArm] = (
                            {"teacher_objective_arm": config.teacher_objective_arm}
                            if config.teacher_objective_arm != "original"
                            else {}
                        )
                        loss, terms = training_loss(
                            model,
                            batch,
                            indices,
                            phase,
                            table,
                            objective,
                            **objective_options,
                        )
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(
                                f"Nonfinite {phase} loss at update {phase_step}"
                            )
                        loss.backward()  # pyright: ignore[reportUnknownMemberType]
                        if phase == "teacher":
                            assert table is not None
                            assert latent_optimizer is not None
                            norm = clip_teacher_gradients(model.predictor.parameters(), table, 1.0)
                            latent_optimizer.step()  # pyright: ignore[reportUnknownMemberType]
                        else:
                            norm = float(
                                torch.nn.utils.clip_grad_norm_(
                                    [p for p in model.parameters() if p.requires_grad],
                                    1.0,
                                    error_if_nonfinite=True,
                                )
                            )
                        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
                        if phase_step == 1 or phase_step % 100 == 0 or phase_step == budget:
                            logger.log(
                                global_step,
                                {
                                    **{
                                        f"train/{key}": float(value.detach())
                                        for key, value in terms.items()
                                    },
                                    "train/gradient_norm": norm,
                                    "phase/index": float(phase_index),
                                    "phase/update": float(phase_step),
                                },
                            )
                        interval = 1 if config.verification else 1000
                        if phase_step % interval == 0 or phase_step == budget:
                            if (
                                phase_frozen is not None
                                and module_fingerprint(
                                    model.parameter_encoder
                                    if phase == "teacher"
                                    else model.predictor
                                )
                                != phase_frozen
                            ):
                                raise RuntimeError("A phase-frozen learned module changed")
                            if table_frozen is not None and (
                                table is None or module_fingerprint(table) != table_frozen
                            ):
                                raise RuntimeError(
                                    "Amortization changed the detached teacher targets"
                                )
                            assess(phase, phase_step)
                            save(phase_step, global_step, boundary=phase_step == budget)
                    saved = None
                if global_step != config.total_steps or table is not None:
                    raise RuntimeError("Final budget incomplete or teacher table retained")
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
                    else "label-free staged acquisition control; inspect all-six held-out criteria",
                    "global_step": global_step,
                    "records": records,
                    "final_test": final,
                    "paired_validation_changes": paired_changes(
                        raw_records[0],
                        raw_records[global_step],
                        samples=config.bootstrap_samples,
                        seed=config.seed,
                    ),
                    "tracking": logger.metadata,
                }
                write_json(out / "results.json", result)
                atomic_save(out / "measurements.pt", {"records": raw_records})
                logger.log(global_step, scalar_metrics(final, "test"))
                logger.log(global_step, {"diagnostic/complete": 1.0})
                for filename in ("results.json", "protocol.json"):
                    logger.log_artifact(
                        out / filename, name=f"group-latent-{identity[:12]}-{Path(filename).stem}"
                    )
                check_states()
                for name, expected in source_hashes.items():
                    if sha(source / name) != expected:
                        raise RuntimeError(f"Source artifact changed: {name}")
                logger.finish()
                filenames = (
                    "last.pt",
                    "teacher_final.pt",
                    "amortization_final.pt",
                    "washout_final.pt",
                    "initial_group_codes.pt",
                    "results.json",
                    "measurements.pt",
                    "protocol.json",
                )
                write_json(
                    out / "completed.json",
                    {"identity": identity, "hashes": {name: sha(out / name) for name in filenames}},
                )
                return cast(dict[str, Any], read_json(out / "results.json"))
            except BaseException as error:
                write_json(
                    out / "failure.json",
                    {"identity": identity, "global_step": global_step, "error": str(error)},
                )
                logger.log(global_step, {"diagnostic/failed": 1.0})
                logger.log_artifact(
                    out / "failure.json", name=f"group-latent-{identity[:12]}-failure"
                )
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    """Launch only this predeclared staged diagnostic, leaving ordinary defaults untouched."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--training-source-root")
    parser.add_argument("--initialization", choices=("warm", "fresh_zero"), default="warm")
    parser.add_argument(
        "--teacher-objective-arm",
        choices=("original", "without_parameter_sigreg"),
        default="original",
    )
    for name, default in (
        ("seed", 202610261301),
        ("teacher-steps", 30000),
        ("amortization-steps", 10000),
        ("washout-steps", 10000),
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
    arguments = vars(parser.parse_args())
    resume = bool(arguments.pop("resume"))
    run(Config(**arguments), resume=resume)


if __name__ == "__main__":
    main()
