"""Resumable, explicitly supervised shared-correction trajectories."""

import copy
import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from diagnostics.joint_acquisition.corrections import (
    CASES,
    AssessmentFunction,
    CorrectionCase,
    LossFunction,
    block_parameters,
    vector_norm,
)
from diagnostics.multiquery_control.data import module_fingerprint
from scjepa.models.model import SCJEPA
from scjepa.training.rng import preserve_rng


@dataclass(frozen=True)
class TrajectorySettings:
    """Fixed fitting policy; the 1,000-update ceiling is not an early-stop rule."""

    updates: int = 1000
    learning_rate: float = 0.001
    clip_norm: float = 1.0
    milestones: tuple[int, ...] = (0, 1, 10, 100, 1000)
    seed: int = 202610261301

    def __post_init__(self) -> None:
        """Require a bounded budget and deterministic, ordered report milestones."""
        if not 1 <= self.updates <= 1000:
            raise ValueError("Diagnostic budget must be within 1..1000 updates")
        if any(not math.isfinite(x) or x <= 0 for x in (self.learning_rate, self.clip_norm)):
            raise ValueError("Learning rate and clip norm must be finite and positive")
        if (
            not self.milestones
            or self.milestones[0] != 0
            or tuple(sorted(set(self.milestones))) != self.milestones
            or min(self.milestones) < 0
        ):
            raise ValueError("Milestones must start at zero and increase strictly")


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def run_trajectory(
    model: SCJEPA,
    fit_loss: LossFunction,
    assess: AssessmentFunction,
    code_values: Callable[[SCJEPA], Tensor],
    *,
    case: CorrectionCase,
    checkpoint_path: Path,
    identity: str,
    local_checks: Mapping[str, Any],
    settings: TrajectorySettings | None = None,
    resume: bool = False,
    stop_after: int | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fit calibration targets only and persist a separately resumable case.

    Assessment supplies validation response/use, ordinary error and whole-code
    retention. code_values supplies unaveraged calibration context codes. An
    operational stop_after does not change trajectory identity or fitting budget.
    Every update is checkpointed atomically; assessment and logging preserve RNG.
    """
    settings = settings or TrajectorySettings()
    if not local_checks.get("numerically_valid", False):
        raise ValueError("Local numerical checks must pass before trajectories")
    if case not in CASES:
        raise ValueError("Unknown correction case")
    if not identity:
        raise ValueError("Source/data/code identity is required")
    if stop_after is not None and not 0 <= stop_after <= settings.updates:
        raise ValueError("stop_after must lie within the fixed update budget")
    if checkpoint_path.exists() and not resume:
        raise ValueError("Existing trajectory requires explicit resume")
    if resume and not checkpoint_path.exists():
        raise ValueError("Resume checkpoint does not exist")
    source_blocks = block_parameters(model)
    device = next(model.parameters()).device
    protocol = {
        "version": 1,
        "source_model": {
            name: module_fingerprint(module)
            for name, module in (
                ("online", model.online_encoder),
                ("target", model.target_encoder),
                ("encoder", model.parameter_encoder),
                ("predictor", model.predictor),
            )
        },
        "model_extra_state": model.get_extra_state(),
        "parent_identity": identity,
        "settings": asdict(settings),
        "case": case,
        "optimizer": "fresh diagnostic Adam; no production objective change",
        "dtype": str(next(model.parameters()).dtype),
        "schedule": "Full selected calibration fit-query batch on every update",
        "target": "Caller-supplied exact finite parameter-response loss",
    }
    trajectory_identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    with preserve_rng(device):
        candidate = copy.deepcopy(model).eval()
        blocks = block_parameters(candidate)
        active = ("encoder", "predictor") if case == "joint" else (case,)
        for block, parameters in blocks.items():
            for parameter in parameters.values():
                parameter.requires_grad_(block in active)
        parameters = [p for p in candidate.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=settings.learning_rate)
        with torch.no_grad(), preserve_rng(device):
            original_codes = code_values(model).detach().clone()
        if not bool(torch.isfinite(original_codes).all()) or original_codes.numel() == 0:
            raise FloatingPointError("Invalid initial calibration codes")
        previous_codes = original_codes.clone()
        cumulative = {block: 0.0 for block in blocks}
        cumulative_code = 0.0
        records: list[dict[str, Any]] = []
        step = 0
        status = "interrupted"
        failure: str | None = None
        if resume:
            checkpoint = cast(
                dict[str, Any], torch.load(checkpoint_path, map_location=device, weights_only=False)
            )
            if checkpoint["identity"] != trajectory_identity:
                raise ValueError("Trajectory identity changed; refusing incompatible resume")
            candidate.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            step = int(checkpoint["step"])
            if not 0 <= step <= settings.updates:
                raise ValueError("Invalid saved trajectory step")
            cumulative = checkpoint["cumulative_weight_path"]
            cumulative_code = checkpoint["cumulative_code_path"]
            previous_codes = checkpoint["previous_codes"].to(original_codes)
            records = checkpoint["records"]
            status, failure = checkpoint["status"], checkpoint["failure"]
            _restore_rng(checkpoint["rng"], device)
            if failure is not None:
                raise ValueError("A numerically failed trajectory cannot be resumed")
        else:
            torch.default_generator.manual_seed(settings.seed)
            random.seed(settings.seed)
            np.random.seed(settings.seed % (2**32))  # noqa: NPY002
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(settings.seed)

        def report_record() -> dict[str, Any]:
            with preserve_rng(device):
                with torch.no_grad():
                    current_codes = code_values(candidate).detach()
                    displacement = {
                        block: vector_norm(
                            {
                                name: p.detach() - source_blocks[block][name].detach()
                                for name, p in values.items()
                            }
                        )
                        for block, values in blocks.items()
                    }
                row: dict[str, Any] = {
                    "step": step,
                    "fit_loss": float(fit_loss(candidate).detach()),
                    "assessment": assess(candidate),
                    "weight_displacement": displacement,
                    "total_weight_displacement": math.sqrt(
                        sum(x * x for x in displacement.values())
                    ),
                    "cumulative_weight_path": cumulative.copy(),
                    "code_displacement_rms": float(
                        (current_codes - original_codes).square().mean().sqrt()
                    ),
                    "cumulative_code_path_rms": cumulative_code,
                }
                if progress is not None:
                    progress(row)
                return row

        def save() -> None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save(
                {
                    "identity": trajectory_identity,
                    "protocol": protocol,
                    "model": candidate.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": _rng_state(device),
                    "step": step,
                    "records": records,
                    "cumulative_weight_path": cumulative,
                    "cumulative_code_path": cumulative_code,
                    "previous_codes": previous_codes,
                    "status": status,
                    "failure": failure,
                },
                temporary,
            )
            temporary.replace(checkpoint_path)

        milestones = {value for value in settings.milestones if value <= settings.updates} | {
            settings.updates
        }
        if not records:
            records.append(report_record())
            save()
        end = settings.updates if stop_after is None else stop_after
        while step < end:
            before = {
                block: {name: p.detach().clone() for name, p in values.items()}
                for block, values in blocks.items()
            }
            try:
                optimizer.zero_grad(set_to_none=True)
                loss = fit_loss(candidate)
                if loss.numel() != 1 or not bool(torch.isfinite(loss).all()):
                    raise FloatingPointError("Nonfinite diagnostic fitting loss")
                loss.backward()  # pyright: ignore[reportUnknownMemberType]
                torch.nn.utils.clip_grad_norm_(
                    parameters, settings.clip_norm, error_if_nonfinite=True
                )
                optimizer.step()  # pyright: ignore[reportUnknownMemberType]
                if any(not bool(torch.isfinite(p).all()) for p in parameters):
                    raise FloatingPointError("Nonfinite diagnostic parameters")
                with torch.no_grad(), preserve_rng(device):
                    new_codes = code_values(candidate).detach().clone()
                if not bool(torch.isfinite(new_codes).all()):
                    raise FloatingPointError("Nonfinite diagnostic codes")
                for block, values in blocks.items():
                    cumulative[block] += vector_norm(
                        {name: p.detach() - before[block][name] for name, p in values.items()}
                    )
                cumulative_code += float((new_codes - previous_codes).square().mean().sqrt())
                previous_codes = new_codes
                step += 1
                if step in milestones:
                    records.append(report_record())
            except (FloatingPointError, RuntimeError) as error:
                status, failure = "numerical_failure", str(error)
                save()
                break
            status = "complete" if step == settings.updates else "interrupted"
            save()
        return {
            "identity": trajectory_identity,
            "protocol": protocol,
            "status": status,
            "failure": failure,
            "updates": step,
            "records": records,
            "interpretation": (
                "Diagnostic checkpoint repair only; no improvement means "
                "no improvement within this tested budget, not global incapacity."
            ),
        }
