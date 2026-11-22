"""Evaluate RGB within-system variation, forecast transfer and native physical controls."""

import math
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.vision_dimer.analysis import load_model, spread
from diagnostics.vision_dimer.data import RenderedInputs
from diagnostics.vision_dimer.train import rendered_batches
from scjepa.models.model import ModelOutput

from .consensus_gradient import context_intervals
from .donors import risks
from .vision_acquisition import read, sha

SPLITS = ("calibration", "validation")
DONORS = ("single_mse", "pooled_mse", "alien_mse", "donor_variance")


def spectrum(codes: Tensor) -> dict[str, Any]:
    """Report the unsmoothed total covariance and whether all chart directions resolve."""
    rows = codes.detach().double().reshape(-1, codes.shape[-1])
    rows = rows - rows.mean(0)
    values = cast(Tensor, torch.linalg.eigvalsh(rows.T @ rows / len(rows)))  # pyright: ignore[reportUnknownMemberType]
    good = bool(torch.isfinite(values).all() and values[0] > (values[-1] * 1e-12).clamp_min(1e-300))
    normalized = values.clamp_min(0) / values.sum().clamp_min(1e-300)
    return {
        "full_rank": good,
        "eigenvalues": values.tolist(),  # pyright: ignore[reportUnknownMemberType]
        "condition": float(values[-1] / values[0]) if good else None,
        "effective_rank": float((-(normalized * normalized.clamp_min(1e-300).log()).sum()).exp()),
    }


@torch.no_grad()
def endpoint(path: Path, data: Path, device: torch.device) -> dict[str, Any]:
    """Measure held-out first-window hard forecasts on fixed donors and distinct donor panels."""
    model, _, _ = load_model(path, device)
    model.eval()
    semantic = read(path.parent / "final.json")
    result: dict[str, Any] = {"checkpoint": str(path), "sha256": sha(path), "semantic": semantic}
    for split, key in zip(SPLITS, ("calibration_prediction_mse", "prediction_mse"), strict=True):
        groups: list[Tensor] = []
        forecasts: list[Tensor] = []
        support: list[Tensor] = []
        for batch in rendered_batches(RenderedInputs(data, split, device)):
            actions = batch["actions"]
            out = cast(ModelOutput, model(batch["observations"], actions))
            forecasts.append(
                (out.predictions - out.targets).square().mean((1, 2, 3)).double().cpu()
            )
            groups.append(out.parameters.cpu())
            support.append(
                risks(model, out.online_states, out.parameters, actions, out.targets).double().cpu()
            )
        codes, errors, donors = torch.cat(groups), torch.cat(forecasts), torch.cat(support)
        chart = spectrum(codes)
        result[split] = {
            "chart": chart,
            "reported_spread": spread(codes) if chart["full_rank"] else None,
            "mse": float(errors.mean()),
            "mse_matched": math.isclose(
                float(errors.mean()), semantic[key], rel_tol=1e-4, abs_tol=1e-8
            ),
            "group_order": "contiguous release order, first window of each episode",
            "codes": codes.tolist(),  # pyright: ignore[reportUnknownMemberType]
            "group_mse": errors.tolist(),  # pyright: ignore[reportUnknownMemberType]
            "donor_columns": DONORS,
            "donor_mean": donors.mean(0).tolist(),  # pyright: ignore[reportUnknownMemberType]
            "group_donors": donors.tolist(),  # pyright: ignore[reportUnknownMemberType]
        }
    return result


def decision(records: dict[str, dict[str, Any]], *, draws: int = 2000) -> dict[str, Any]:
    """Evaluate the predeclared comparisons and bootstrap common held-out systems."""
    control, changed = records["native"]["semantic"], records["other"]["semantic"]
    shd_keys = ("state_shd", "action_shd", "parameter_shd")
    checks: dict[str, bool] = {
        "drag_retention": changed["parameter_r2_min"]
        >= max(0.95, control["parameter_r2_min"] - 0.01),
        "state_mcc": changed["mcc"] >= control["mcc"] - 0.01,
        "graph": sum(changed[k] for k in shd_keys) <= sum(control[k] for k in shd_keys),
    }
    intervals: dict[str, Any] = {}
    for split, bootstrap_seed in zip(SPLITS, (20261123, 20261124), strict=True):
        parts = {arm: data[split] for arm, data in records.items()}
        checks[f"{split}_rank"] = all(p["chart"]["full_rank"] for p in parts.values())
        checks[f"{split}_hard_mse_reproduced"] = all(p["mse_matched"] for p in parts.values())
        checks[f"{split}_mse"] = parts["other"]["mse"] <= 1.05 * parts["native"]["mse"]
        if not checks[f"{split}_rank"]:
            continue
        worst = {key: p["reported_spread"]["worst_direction_share"] for key, p in parts.items()}
        for ref in ("native", "same"):
            checks[f"{split}_spread_vs_{ref}"] = worst["other"] <= 0.8 * worst[ref]
        codes = {arm: torch.tensor(p["codes"], dtype=torch.float64) for arm, p in parts.items()}
        errors = {
            arm: torch.tensor(p["group_mse"], dtype=torch.float64) for arm, p in parts.items()
        }
        try:
            paired = context_intervals(codes, errors, bootstrap_seed, draws=draws)
        except ValueError as failure:
            checks[f"{split}_bootstrap_support"] = False
            intervals[split] = {"reason": str(failure)}
            continue
        checks[f"{split}_bootstrap_support"] = True
        gen = torch.Generator().manual_seed(bootstrap_seed)
        groups = len(errors["other"])
        idx = torch.randint(groups, (draws, groups), generator=gen)
        panels = {
            arm: torch.tensor(p["group_donors"], dtype=torch.float64) for arm, p in parts.items()
        }
        donors: dict[str, Any] = {}
        for ref in ("native", "same"):
            delta = panels["other"] - panels[ref]
            estimated = delta[idx].mean(1)
            donors[f"other_minus_{ref}"] = {
                k: {
                    "estimate": float(delta[:, i].mean()),
                    "ci95": torch.quantile(  # pyright: ignore[reportUnknownMemberType]
                        estimated[:, i], estimated.new_tensor([0.025, 0.975])
                    ).tolist(),
                }
                for i, k in enumerate(DONORS)
            }
        intervals[split] = {
            "seed": bootstrap_seed,
            "draws": draws,
            "paired": paired,
            "donor_differences": donors,
        }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "intervals": intervals,
        "semantic": {arm: item["semantic"] for arm, item in records.items()},
        "interval_definition": (
            "Common system resamples, all episodes together. Population covariance in bootstrap. "
            "Released spread uses finite-sample corrections; their common split-specific "
            "factor cancels in point ratios. Within-model intervals exclude seed variation."
        ),
    }
