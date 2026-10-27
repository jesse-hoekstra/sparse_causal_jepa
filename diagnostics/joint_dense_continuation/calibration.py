"""One durable fixed-budget TRAIN-only offset fit after dense training finishes."""

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from torch import Tensor

from diagnostics.global_code_calibration.run import compact_fit
from diagnostics.joint_state_calibration.source import encode_observations
from diagnostics.multiquery_control.data import TensorSplit
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.predictor_transfer_calibration.objective import (
    TrainObjective,
    compose_translation,
    encode_train,
)
from diagnostics.predictor_transfer_calibration.optimization import (
    EmbeddedObjective,
    active_state,
    fit,
    verify_embedding,
)
from diagnostics.shared_confirmatory_control.run import checked_payload, save_payload
from diagnostics.table_settling_control.source import model_fingerprint
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import read_json, write_json
from scjepa.models.model import SCJEPA


def fit_final(
    model: SCJEPA,
    observations: TensorSplit,
    output: Path,
    *,
    identity: str,
    checkpoint_sha256: str,
    chunk_size: int,
    max_iterations: int,
    max_evaluations: int,
    log: Callable[[dict[str, float]], None],
) -> tuple[SCJEPA, SCJEPA, dict[str, Any], list[str]]:
    """Fit six new offsets on TRAIN, reusing terminal state without reopening its budget."""
    source = copy.deepcopy(model).requires_grad_(False).double().eval()
    fixed = model_fingerprint(source)
    protocol = {
        "continuation_identity": identity,
        "checkpoint_sha256": checkpoint_sha256,
        "source_model_sha256": fixed,
        "max_iterations": max_iterations,
        "max_evaluations": max_evaluations,
        "scope": (
            "Final fixed model, six new zero-initialized offsets, TRAIN only; monitoring "
            "does not enter fitting"
        ),
    }
    stage_identity = canonical_fingerprint(protocol)
    protocol["identity"] = stage_identity
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        if read_json(output / "protocol.json")["identity"] != stage_identity:
            raise ValueError("Final calibration source or fixed budget differs")
    else:
        write_json(output / "protocol.json", protocol)
    if (output / "completed.json").exists():
        complete = read_json(output / "completed.json")
        if complete["identity"] != stage_identity:
            raise ValueError("Completed final calibration identity changed")
        hashes = cast(dict[str, str], complete["hashes"])
        required = {
            "protocol.json",
            "train.pt",
            "train.receipt.json",
            "codes.pt",
            "codes.receipt.json",
            "fit_state.pt",
            "fit_state.receipt.json",
            "fit.json",
            "translation.pt",
            "composed_model.pt",
        }
        if set(hashes) != required:
            raise ValueError("Completed final calibration artifacts differ")
        for name, expected in hashes.items():
            if sha(output / name) != expected:
                raise ValueError("Completed final calibration artifact changed")
        saved = checked_payload(output, "fit_state", stage_identity)["state"]
        verify_embedding(saved)
        if saved["stop_reason"] == "running":
            raise ValueError("Completed final calibration is not terminal")
        delta = cast(Tensor, active_state(saved)["translation"]).to(next(source.parameters()))
        candidate = compose_translation(source, delta)
        summary = cast(dict[str, Any], read_json(output / "fit.json"))
        summary.pop("identity")
        return (
            source,
            candidate,
            summary,
            ["calibration/" + name for name in (*hashes, "completed.json")],
        )
    if (output / "train.pt").exists():
        train = cast(TensorSplit, checked_payload(output, "train", stage_identity))
    else:
        train = encode_observations(
            source,
            observations,
            chunk_size=chunk_size,
            progress=lambda n: log({"calibration/train_state_groups": float(n)}),
        )
        save_payload(output, "train", train, stage_identity)
    if (output / "codes.pt").exists():
        codes = cast(Tensor, checked_payload(output, "codes", stage_identity)["codes"])
    else:
        codes = encode_train(
            source, train, chunk_size, lambda n: log({"calibration/train_code_groups": float(n)})
        )
        save_payload(output, "codes", {"codes": codes}, stage_identity)
    state: dict[str, Any] | None = None
    if (output / "fit_state.pt").exists():
        state = checked_payload(output, "fit_state", stage_identity)["state"]
        assert state is not None
        verify_embedding(state)
    terminal = {
        "gradient_tolerance",
        "evaluation_budget_rollback",
        "no_accepted_displacement",
        "accepted_iteration_budget",
    }
    if state is not None and (
        (state["max_iterations"], state["max_evaluations"]) != (max_iterations, max_evaluations)
        or not 0 <= state["accepted_iterations"] <= max_iterations
        or not 0 <= state["full_evaluations"] <= max_evaluations
        or state["stop_reason"] not in terminal | {"running"}
    ):
        raise ValueError("Final calibration state budget or phase differs")

    def save(value: dict[str, Any]) -> None:
        verify_embedding(value)
        if model_fingerprint(source) != fixed:
            raise RuntimeError("Final calibration changed frozen source weights")
        save_payload(output, "fit_state", {"state": value}, stage_identity)
        metrics = {
            "calibration/full_evaluations": float(value["full_evaluations"]),
            "calibration/accepted_iterations": float(value["accepted_iterations"]),
        }
        if value["accepted_loss"] is not None:
            metrics["calibration/accepted_loss"] = float(value["accepted_loss"])
        log(metrics)

    if state is None or state["stop_reason"] == "running":
        state = fit(
            EmbeddedObjective(TrainObjective(source, train, codes, chunk_size)),
            device=next(source.parameters()).device,
            max_iterations=max_iterations,
            max_evaluations=max_evaluations,
            state=state,
            on_state=save,
        )
    active = active_state(state)
    summary = compact_fit(active, codes)
    summary["active_parameters"] = 6
    summary["inactive_optimizer_slots_exactly_zero"] = True
    write_json(output / "fit.json", {"identity": stage_identity, **summary})
    delta = cast(Tensor, active["translation"]).to(next(source.parameters()))
    candidate = compose_translation(source, delta)
    atomic_save(
        output / "translation.pt",
        {
            "identity": stage_identity,
            "translation": delta.detach().cpu(),
            "source_model_sha256": fixed,
        },
    )
    atomic_save(
        output / "composed_model.pt",
        {
            "identity": stage_identity,
            "model": candidate.state_dict(),
            "source_model_sha256": fixed,
            "composed_model_sha256": model_fingerprint(candidate),
        },
    )
    if model_fingerprint(source) != fixed or any(
        p.requires_grad or p.grad is not None for p in source.parameters()
    ):
        raise RuntimeError("TRAIN-only calibration changed source weights or accumulated gradients")
    names = [
        "protocol.json",
        "train.pt",
        "train.receipt.json",
        "codes.pt",
        "codes.receipt.json",
        "fit_state.pt",
        "fit_state.receipt.json",
        "fit.json",
        "translation.pt",
        "composed_model.pt",
    ]
    write_json(
        output / "completed.json",
        {"identity": stage_identity, "hashes": {name: sha(output / name) for name in names}},
    )
    return (
        source,
        candidate,
        summary,
        ["calibration/" + name for name in (*names, "completed.json")],
    )
