"""Measure sensitivity of a named hold query to limited physical calibration."""

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from diagnostics.vision_dimer import data, downstream

from .chart import DATA, RUNS, AffineChart, encode_endpoint
from .chart import _affine_query as affine_query  # pyright: ignore[reportPrivateUsage]


def panel_indices(groups: int, budget: int) -> Tensor:
    """Partition systems into disjoint calibration panels at a fixed random permutation."""
    if budget <= 0 or groups % budget != 0:
        raise ValueError("Panel size must divide the system count")
    return torch.randperm(groups, generator=torch.Generator().manual_seed(2026)).reshape(-1, budget)


@torch.inference_mode()
def run(seed: int) -> dict[str, Any]:
    """Fit both chart interfaces on one labelled frame per selected calibration system."""
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    full_record = Path(f"ICML_review/outputs/chart-paired-s{seed}/metrics.json")
    full: dict[str, Any] = json.loads(full_record.read_text())
    # These are user-facing hold settings fixed by the reference query.
    # The reduced panel never fits a chart or assignment using extra frame labels.
    grid = torch.tensor(full["hold_grids"][4], dtype=torch.double)
    cal = data.RenderedInputs(DATA, "calibration", device)
    val = data.RenderedInputs(DATA, "validation", device)
    cal_z = data.load_labels(DATA, "calibration")["states"][:, 0, downstream.T].double()
    panels = panel_indices(len(cal), 32)
    result: dict[str, Any] = {
        "seed": seed,
        "truth_slope": full["truth_slopes_held_by_outcome"][4][6],
        "fixed_physical_hold_grid": grid.tolist(),  # pyright: ignore[reportUnknownMemberType]
        "panel_systems": panels.tolist(),  # pyright: ignore[reportUnknownMemberType]
        "reference_sha256": data.sha256(full_record),
        "validation_groups": len(val),
    }
    for arm in ("full", "baseline"):
        checkpoint = RUNS / f"seed{seed}/sparse-500k" / arm / "final.pt"
        model, all_frames, lat = encode_endpoint(checkpoint, arm, device, cal, val)
        lat_frame = all_frames.reshape(len(cal), cal.views, cal.frames, -1)[:, 0, downstream.T]
        fits: list[dict[str, Any]] = []
        for panel in panels:
            x, y = lat_frame[panel], cal_z[panel]
            scalar = downstream.Model(model, lat, x, y)
            affine = AffineChart.fit(x, y)
            scalar_means = torch.stack(
                [
                    scalar.decode(scalar.rollout(scalar.hold_value(4, float(a)), None))[:, 6].mean()
                    for a in grid
                ]
            )
            affine_means = torch.stack(
                [affine_query(model, lat, affine, 4, float(a))[:, 6].mean() for a in grid]
            )
            fits.append(
                {
                    "scalar": float(downstream.slope(grid, scalar_means[:, None])[0]),
                    "affine": float(downstream.slope(grid, affine_means[:, None])[0]),
                    "assignment": scalar.match,
                }
            )
        result[arm] = {
            "checkpoint": str(checkpoint.relative_to(DATA.parents[2])),
            "sha256": data.sha256(checkpoint),
            "fits": fits,
            "reference_scalar": full["arms"][arm]["scalar"]["u2a_to_u2b"],
            "reference_affine": full["arms"][arm]["affine"]["u2a_to_u2b"],
        }
    return result
