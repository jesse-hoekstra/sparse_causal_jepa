"""Evaluate scalar and joint affine calibration for the vision hold queries."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.vision_dimer import data, dgp, downstream
from diagnostics.vision_dimer.train import load_checkpoint, model_from_checkpoint
from scjepa.models.model import SCJEPA

_REPO = Path(__file__).resolve().parents[2]
DATA = _REPO / "outputs/data/vision-dimer-rgb-v1"
RUNS = _REPO / "outputs/diagnostics/vision-dimer-rgb-v1-s05all"


@dataclass(frozen=True)
class AffineChart:
    """Map complete latent vectors to physical coordinates and invert physical edits."""

    weight: Tensor
    offset: Tensor
    inverse: Tensor

    @classmethod
    def fit(cls, latent: Tensor, physical: Tensor) -> "AffineChart":
        """Fit an intercept and square OLS map on the calibration split."""
        x, y = latent.double(), physical.double()
        if x.shape != y.shape or x.ndim != 2:
            raise ValueError("expected corresponding rows of equal dimension")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("calibration inputs must be finite")
        xc, yc = x - x.mean(0), y - y.mean(0)
        weight, _, design_rank, _ = cast(
            tuple[Tensor, Tensor, Tensor, Tensor],
            torch.linalg.lstsq(xc, yc, driver="gelsd"),  # pyright: ignore[reportUnknownMemberType]
        )
        rank = cast(Tensor, torch.linalg.matrix_rank(weight))  # pyright: ignore[reportUnknownMemberType]
        if int(design_rank) != x.shape[1] or int(rank) != x.shape[1]:
            raise ValueError("the fitted affine chart has no unique inverse")
        inverse = cast(Tensor, torch.linalg.inv(weight))  # pyright: ignore[reportUnknownMemberType]
        return cls(weight, y.mean(0) - x.mean(0) @ weight, inverse)

    def decode(self, latent: Tensor) -> Tensor:
        """Read out the predicted physical state."""
        return latent @ self.weight + self.offset

    def clamp(self, latent: Tensor, coordinate: int, value: float) -> Tensor:
        """Set one physical coordinate while preserving all other decoded coordinates."""
        correction = value - self.decode(latent)[..., coordinate]
        return latent + correction.unsqueeze(-1) * self.inverse[coordinate]


def _r2(estimate: Tensor, truth: Tensor) -> list[float]:
    per_dimension = 1 - (estimate - truth).square().mean(0) / truth.var(0, correction=0)
    return [float(v) for v in per_dimension]


def _affine_query(
    model: SCJEPA, lat: dict[str, Tensor], chart: AffineChart, j: int, value: float
) -> Tensor:
    t = downstream.T
    history = chart.clamp(lat["online"][:, t - 1 : t + 1], j, value)
    actions, codes = lat["actions"].float(), lat["code"].float()
    for lag in range(2):
        acts = actions[:, t + lag - 1 : t + lag + 1]
        predicted = model.predictor(history.float(), codes, acts).prediction.double()
        held = chart.clamp(predicted, j, value)
        history = torch.stack((history[:, 1], held), 1)
    return chart.decode(history[:, 1])


def _slopes(grids: Tensor, query: Callable[[int, float], Tensor]) -> Tensor:
    return torch.stack(
        [
            downstream.slope(grid, torch.stack([query(j, float(a)).mean(0) for a in grid]))
            for j, grid in enumerate(grids)
        ]
    )


def _score(slopes: Tensor, truth: Tensor) -> dict[str, Any]:
    indices = torch.arange(truth.shape[0])
    peers = (indices[:, None] // 4 == indices[None, :] // 4) & (
        indices[:, None] != indices[None, :]
    )
    cross = indices[:, None] // 4 != indices[None, :] // 4
    delta = (slopes - truth).abs()
    return {
        "slopes_held_by_outcome": downstream.as_list(slopes),
        "mean_within_dimer_off_diagonal_absolute_error": float(delta[peers].mean()),
        "max_cross_dimer_absolute_error": float(delta[cross].max()),
        "u2a_to_u2b": float(slopes[4, 6]),
    }


def encode_endpoint(
    checkpoint: Path,
    arm: str,
    device: torch.device,
    calibration: data.RenderedInputs,
    validation: data.RenderedInputs,
) -> tuple[SCJEPA, Tensor, dict[str, Tensor]]:
    """Return the frozen model, calibration frames and validation windows on CPU."""
    payload = load_checkpoint(checkpoint)
    model = model_from_checkpoint(payload, phase=arm).to(device).eval()
    cal = (
        downstream.encode(model, calibration, (0,), len(calibration))["online"].reshape(-1, 8).cpu()
    )
    encoded = downstream.encode(model, validation, downstream.STARTS, len(validation))
    lat = {k: v.cpu() for k, v in encoded.items()}
    model.cpu()
    return model, cal, lat


def _arm(
    seed: int,
    arm: str,
    device: torch.device,
    calibration: data.RenderedInputs,
    validation: data.RenderedInputs,
    cal_truth: Tensor,
    val_current: Tensor,
    grids: Tensor,
    truth_slopes: Tensor,
) -> dict[str, Any]:
    checkpoint = RUNS / f"seed{seed}/sparse-500k" / arm / "final.pt"
    model, cal, lat = encode_endpoint(checkpoint, arm, device, calibration, validation)
    scalar = downstream.Model(model, lat, cal, cal_truth)
    affine = AffineChart.fit(cal, cal_truth)
    current = lat["online"][:, downstream.T]
    rms = (affine.decode(cal) - cal_truth).square().mean().sqrt()
    stored = json.loads((checkpoint.parent / "final.json").read_text())
    first = slice(0, len(validation) * validation.views)
    nmse = downstream.forecast_nmse(scalar.forecast(), lat["target"], first)
    scalar_slopes = _slopes(
        grids, lambda j, a: scalar.decode(scalar.rollout(scalar.hold_value(j, a), None))
    )
    affine_slopes = _slopes(grids, lambda j, a: _affine_query(model, lat, affine, j, a))
    if not torch.isfinite(affine_slopes).all() or not torch.isfinite(scalar_slopes).all():
        raise ValueError("nonfinite query result")
    if not torch.allclose(affine_slopes.diag(), torch.ones_like(grids[:, 0]), atol=1e-7, rtol=0):
        raise ValueError("affine held-coordinate readback failed")
    condition = cast(Tensor, torch.linalg.cond(affine.weight))  # pyright: ignore[reportUnknownMemberType]
    return {
        "checkpoint": str(checkpoint.relative_to(_REPO)),
        "checkpoint_sha256": data.sha256(checkpoint),
        "recomputed_first_window_nmse": nmse,
        "stored_first_window_nmse": stored["nmse"],
        "spearman_assignment_matches_stored": list(scalar.match) == stored["spearman_assignment"],
        "affine_condition_number": float(condition),
        "affine_calibration_rmse": float(rms),
        "affine_calibration_r2": _r2(affine.decode(cal), cal_truth),
        "scalar_calibration_r2": _r2(scalar.decode(cal), cal_truth),
        "affine_validation_r2": _r2(affine.decode(current), val_current),
        "scalar_validation_r2": _r2(scalar.decode(current), val_current),
        "scalar": _score(scalar_slopes, truth_slopes),
        "affine": _score(affine_slopes, truth_slopes),
    }


@torch.inference_mode()
def run(seed: int) -> dict[str, Any]:
    """Evaluate one matched pair on all validation contexts, using calibration labels for units."""
    if seed not in range(5):
        raise ValueError("expected a reported seed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = data.load_config(DATA)
    calibration = data.RenderedInputs(DATA, "calibration", device)
    validation = data.RenderedInputs(DATA, "validation", device)
    cal_truth = data.load_labels(DATA, "calibration")["states"][:, :, : config.frames]
    cal_truth = cal_truth.reshape(-1, dgp.STATE_DIM).double()
    val_data = data._load(DATA, "validation")  # pyright: ignore[reportPrivateUsage]
    ctx = downstream.truth_contexts(val_data, downstream.STARTS, len(validation))
    grids = torch.quantile(
        cal_truth, torch.tensor(downstream.QUANTILES, dtype=torch.float64), dim=0
    ).T
    simulator = downstream.Truth(config.physics, ctx)
    truth = _slopes(grids, lambda j, a: simulator.rollout(simulator.hold_value(j, a), None))
    return {
        "seed": seed,
        "encoding_device": str(device),
        "calibration_groups": len(calibration),
        "validation_groups": len(validation),
        "validation_contexts": len(ctx["cur"]),
        "data_sha256": {s: data.sha256(DATA / f"{s}.pt") for s in ("calibration", "validation")},
        "hold_grids": downstream.as_list(grids),
        "truth_slopes_held_by_outcome": downstream.as_list(truth),
        "arms": {
            arm: _arm(
                seed, arm, device, calibration, validation, cal_truth, ctx["cur"], grids, truth
            )
            for arm in ("baseline", "full")
        },
    }
