"""Evaluate held and free kick responses with scalar and joint calibration."""

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Protocol, cast

import torch
from torch import Tensor

from diagnostics.vision_dimer import data, downstream
from scjepa.models.model import SCJEPA

from .chart import DATA, AffineChart, encode_endpoint, endpoints


class KickWorld(Protocol):
    """Interface for the existing Q2 simulator and calibrated predictors."""

    def current(self, bead: int) -> downstream.Hold:
        """Hold a bead at its two current coordinates."""
        ...

    def rollout(self, hold: downstream.Hold, kick: Tensor | None) -> Tensor:
        """Predict two steps with the requested hold and optional first-step kick."""
        ...

    def decode(self, z: Tensor) -> Tensor:
        """Return physical coordinates."""
        ...


@dataclass
class AffineWorld:
    """Apply two-coordinate holds to an invertible affine readout."""

    model: SCJEPA
    lat: dict[str, Tensor]
    chart: AffineChart

    def current(self, bead: int) -> downstream.Hold:
        """Fix the physical bead at its decoded position in each context."""
        physical = self.decode(self.lat["online"][:, downstream.T])
        return {j: physical[:, j] for j in downstream.BEADS[bead]}

    def decode(self, z: Tensor) -> Tensor:
        """Read all physical coordinates jointly."""
        return self.chart.decode(z)

    def _clamp(self, latent: Tensor, hold: downstream.Hold) -> Tensor:
        if not hold:
            return latent
        indices = list(hold)
        values = torch.stack([hold[j] for j in indices], dim=-1)
        if latent.ndim == 3:
            values = values.unsqueeze(1)
        delta = values - self.decode(latent)[..., indices]
        return latent + delta @ self.chart.inverse[indices]

    def rollout(self, hold: downstream.Hold, kick: Tensor | None) -> Tensor:
        """Set both initial lags to the same held values and reapply them after each step."""
        t = downstream.T
        history = self._clamp(self.lat["online"][:, t - 1 : t + 1], hold)
        actions = self.lat["actions"][:, t - 1 : t + 2].clone()
        if kick is not None:
            actions[:, 1] += kick
        for step in range(2):
            output = self.model.predictor(
                history.float(), self.lat["code"].float(), actions[:, step : step + 2].float()
            )
            updated = self._clamp(output.prediction.double(), hold)
            history = torch.stack((history[:, 1], updated), dim=1)
        return history[:, 1]


def responses(world: KickWorld) -> dict[str, dict[str, Tensor]]:
    """Return partner response matrices; columns are +1 x/y kicks minus zero added kick."""
    result: dict[str, dict[str, Tensor]] = {}
    for bead, (u, _) in enumerate(downstream.BEADS):
        partner = list(downstream.BEADS[downstream.PARTNER[bead]])
        matrices: dict[str, Tensor] = {}
        for label, hold in (("held", world.current(bead)), ("free", {})):
            base = world.decode(world.rollout(hold, None))[:, partner]
            columns = [
                (world.decode(world.rollout(hold, kick))[:, partner] - base).mean(0)
                for kick in downstream.kicks(bead // 2)
            ]
            matrices[label] = torch.stack(columns, dim=1)
        matrices["held_minus_free"] = matrices["held"] - matrices["free"]
        result[downstream.NAMES[u][1:]] = matrices
    return result


def score(
    predicted: dict[str, dict[str, Tensor]],
    truth: dict[str, dict[str, Tensor]],
    *,
    allow_zero_prediction: bool = False,
) -> dict[str, Any]:
    """Score each bead's matrix and then average the four relative errors and cosines."""
    by_bead: dict[str, dict[str, dict[str, float | None]]] = {}
    for bead, matrices in predicted.items():
        by_bead[bead] = {}
        for label, m in matrices.items():
            t = truth[bead][label]
            mt, tt = m.square().sum().sqrt(), t.square().sum().sqrt()
            if float(tt) == 0 or (float(mt) == 0 and not allow_zero_prediction):
                raise ValueError(f"undefined relative matrix score for {bead} {label}")
            by_bead[bead][label] = {
                "relative_error": float((m - t).square().sum().sqrt() / tt),
                "cosine": None if float(mt) == 0 else float((m * t).sum() / (mt * tt)),
            }

    def average(label: str, metric: str) -> float | None:
        values = [row[label][metric] for row in by_bead.values()]
        observed = [value for value in values if value is not None]
        return mean(observed) if len(observed) == len(values) else None

    return {
        "matrices": _serialize(predicted),
        "per_bead": by_bead,
        "mean_across_beads": {
            label: {metric: average(label, metric) for metric in ("relative_error", "cosine")}
            for label in ("held", "free", "held_minus_free")
        },
    }


def _serialize(values: dict[str, dict[str, Tensor]]) -> dict[str, dict[str, list[Any]]]:
    return {
        bead: {label: downstream.as_list(matrix) for label, matrix in conditions.items()}
        for bead, conditions in values.items()
    }


@torch.inference_mode()
def run(seed: int, checkpoints: dict[str, Path] | None = None) -> dict[str, Any]:
    """Compare two calibrators on both frozen sparse arms and the fixed validation queries."""
    selected = endpoints(seed, checkpoints)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = data.load_config(DATA)
    cal = data.RenderedInputs(DATA, "calibration", device)
    val = data.RenderedInputs(DATA, "validation", device)
    phys = (
        data.load_labels(DATA, "calibration")["states"][:, :, : cfg.frames].reshape(-1, 8).double()
    )
    val_tensors = data._load(DATA, "validation")  # pyright: ignore[reportPrivateUsage]
    ctx = downstream.truth_contexts(val_tensors, downstream.STARTS, len(val))
    truth = responses(downstream.Truth(cfg.physics, ctx))
    arms: dict[str, Any] = {}
    for arm, ckpt in selected.items():
        model, cal_latent, lat = encode_endpoint(ckpt, arm, device, cal, val)
        scalar = downstream.Model(model, lat, cal_latent, phys)
        chart = AffineChart.fit(cal_latent, phys)
        affine = AffineWorld(model, lat, chart)
        first = slice(0, len(val) * val.views)
        forecast = scalar.forecast()
        torch.testing.assert_close(affine.rollout({}, None), forecast[:, 1], atol=1e-6, rtol=1e-6)
        stored = json.loads((ckpt.parent / "final.json").read_text())
        condition = cast(Tensor, torch.linalg.cond(chart.weight))  # pyright: ignore[reportUnknownMemberType]
        arms[arm] = {
            "checkpoint": str(ckpt.relative_to(DATA.parents[2])),
            "checkpoint_sha256": data.sha256(ckpt),
            "recomputed_first_window_nmse": downstream.forecast_nmse(
                forecast, lat["target"], first
            ),
            "stored_first_window_nmse": stored["nmse"],
            "assignment_matches_stored": stored["spearman_assignment"] == list(scalar.match),
            "affine_condition_number": float(condition),
            "scalar": score(
                responses(scalar), truth, allow_zero_prediction=checkpoints is not None
            ),
            "affine": score(
                responses(affine), truth, allow_zero_prediction=checkpoints is not None
            ),
        }
    return {
        "seed": seed,
        "encoding_device": str(device),
        "calibration_groups": len(cal),
        "validation_groups": len(val),
        "validation_contexts": len(ctx["cur"]),
        "data_sha256": {s: data.sha256(DATA / f"{s}.pt") for s in ("calibration", "validation")},
        "truth": _serialize(truth),
        "arms": arms,
    }
