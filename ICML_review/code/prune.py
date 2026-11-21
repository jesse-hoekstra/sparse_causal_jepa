"""Post-fit greedy edge deletion under a frozen calibration forecast tolerance."""

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.vision_dimer import data, evaluate
from diagnostics.vision_dimer.analysis import endpoint, load_model
from diagnostics.vision_dimer.model import LaggedMaskedPredictor
from diagnostics.vision_dimer.train import rendered_batches
from scjepa.models.model import SCJEPA, ModelOutput, cross_episode_parameters

from .donors import DATA, ROOT


@dataclass(frozen=True)
class ForecastBatch:
    """Fixed initial histories, donor codes, actions and EMA targets for grouped episodes."""

    histories: Tensor
    donors: Tensor
    actions: Tensor
    targets: Tensor
    views: int


@torch.no_grad()
def cache(model: SCJEPA, batches: Iterable[dict[str, Tensor]]) -> list[ForecastBatch]:
    """Cache original eval-mode model inputs and confirm the first cached forecast."""
    if model.training:
        raise ValueError("Caching requires deterministic evaluation gates and donor assignment")
    cached: list[ForecastBatch] = []
    for batch in batches:
        observations, actions = batch["observations"], batch["actions"]
        output = cast(ModelOutput, model(observations, actions))
        start = model.context_len - model.predictor.history_len
        entry = ForecastBatch(
            output.online_states[:, :, start : model.context_len].flatten(0, 1).contiguous(),
            cross_episode_parameters(output.parameters, training=False).flatten(0, 1),
            actions[:, :, start:].flatten(0, 1).contiguous(),
            output.targets.flatten(0, 1),
            observations.shape[1],
        )
        if not cached:
            predicted = model.rollout(
                entry.histories, entry.donors, entry.targets.shape[1], entry.actions
            )
            torch.testing.assert_close(predicted, output.predictions.flatten(0, 1))
        cached.append(entry)
    if not cached:
        raise ValueError("No groups in forecast cache")
    return cached


@torch.no_grad()
def group_errors(model: SCJEPA, batches: list[ForecastBatch]) -> Tensor:
    """Return two-step latent mean squared errors, one per system."""
    errors: list[Tensor] = []
    for batch in batches:
        predicted = model.rollout(
            batch.histories, batch.donors, batch.targets.shape[1], batch.actions
        )
        error = (predicted.double() - batch.targets.double()).square()
        errors.append(error.reshape(-1, batch.views, *error.shape[1:]).mean((1, 2, 3)))
    return torch.cat(errors)


@torch.no_grad()
def greedy(
    logits: dict[str, Tensor], score: Callable[[], float], tau: float
) -> list[dict[str, Any]]:
    """Try every open edge each round, committing the least-loss feasible removal in RAM."""
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("Invalid calibration tolerance")
    trials: list[dict[str, Any]] = []
    while True:
        proposals: list[tuple[float, str, int, int]] = []
        for name, block in logits.items():
            for row, col in cast(list[list[int]], (block > 0).nonzero().tolist()):  # pyright: ignore[reportUnknownMemberType]
                old = block[row, col].clone()
                try:
                    block[row, col] = -1
                    mse = score()
                finally:
                    block[row, col] = old
                if not math.isfinite(mse):
                    raise FloatingPointError("Nonfinite MSE for a proposed deletion")
                proposals.append((mse, name, row, col))
        if not proposals:
            return trials
        mse, name, row, col = min(proposals)
        trials.append(
            {
                "graph": name,
                "row": row,
                "column": col,
                "calibration_mse": mse,
                "proposals_evaluated": len(proposals),
                "accepted": mse <= tau,
            }
        )
        if mse > tau:
            return trials
        logits[name][row, col] = -1


@torch.no_grad()
def run(seed: int) -> dict[str, Any]:
    """Select deletions on calibration and assess the frozen mask on validation."""
    if seed not in range(5):
        raise ValueError("Expected a reported seed")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = endpoint(ROOT, seed, "full")
    model, phase, payload = load_model(checkpoint, device)
    model.eval()
    predictor = model.predictor
    if phase != "full" or not isinstance(predictor, LaggedMaskedPredictor):
        raise ValueError("Expected the reported full lagged predictor")
    cfg = payload["config"]
    tau = float(cfg["tau"])
    tau_path = ROOT / f"seed{seed}/tau.json"
    dense_tau = float(json.loads(tau_path.read_text())["tau"])
    if cfg["seed"] != seed or not math.isclose(tau, 2 * dense_tau, rel_tol=1e-12):
        raise ValueError("Checkpoint configuration differs from its calibrated tolerance")

    logits: dict[str, Tensor] = dict(predictor.learned_logits())
    original_logits = {k: v.clone() for k, v in logits.items()}
    cal = cache(model, rendered_batches(data.RenderedInputs(DATA, "calibration", device)))

    def cal_score() -> float:
        return float(group_errors(model, cal).mean())

    original_mse = cal_score()
    path = greedy(logits, cal_score, tau)
    final_mse = cal_score()

    # Only the completed policy is exposed to validation and physical graph information.
    stored_path = checkpoint.parent / "final.json"
    stored: dict[str, Any] = json.loads(stored_path.read_text())
    val = cache(model, rendered_batches(data.RenderedInputs(DATA, "validation", device)))

    def assess() -> dict[str, Any]:
        errors = group_errors(model, val)
        return {
            "validation_mse": float(errors.mean()),
            "validation_group_mse": errors.tolist(),  # pyright: ignore[reportUnknownMemberType]
            "graphs": evaluate.graph_metrics(
                model, stored["assignment"], stored["code_parameters"]
            ),
            "learned_masks": {
                k: v.tolist()  # pyright: ignore[reportUnknownMemberType]
                for k, v in predictor.thresholded_graphs().items()
            },
        }

    final = assess()
    for name, block in logits.items():
        block.copy_(original_logits[name])
    original = assess()
    if any(original["graphs"][k] != stored[k] for k in original["graphs"]):
        raise ValueError("Original mask disagrees with recorded graph metrics")
    return {
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": data.sha256(checkpoint),
        "assignment_receipt_sha256": data.sha256(stored_path),
        "dense_calibration_receipt_sha256": data.sha256(tau_path),
        "data_manifest_sha256": data.sha256(DATA / "manifest.json"),
        "dense_calibration_mse": dense_tau,
        "tau": tau,
        "assignment": stored["assignment"],
        "code_parameters": stored["code_parameters"],
        "original": original
        | {"calibration_mse": original_mse, "calibration_feasible": original_mse <= tau},
        "final": final | {"calibration_mse": final_mse, "calibration_feasible": final_mse <= tau},
        "rounds": path,
        "accepted_deletions": sum(t["accepted"] for t in path),
        "stored_original_calibration_mse": stored["calibration_prediction_mse"],
        "stored_original_validation_mse": stored["prediction_mse"],
    }
