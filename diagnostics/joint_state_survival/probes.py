"""Fit the existing probe families on original calibration/validation groups."""

import math
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.multiquery_control.evaluation import _r2  # pyright: ignore[reportPrivateUsage]
from diagnostics.overcomplete_acquisition.evaluation import cubic_features


def feature_map(values: Tensor, family: str, width: float, omega: Tensor, phase: Tensor) -> Tensor:
    """Use the exact original linear, cubic and fixed random-Fourier feature maps."""
    if family == "linear":
        return values
    if family == "cubic":
        return cubic_features(values)
    if family == "rbf_256_random_features":
        return math.sqrt(2) * torch.cos(values @ omega / width + phase)
    raise ValueError("Unknown predeclared readout family")


@torch.no_grad()
def fit_probes(
    cal_codes: Tensor, val_codes: Tensor, cal_theta: Tensor, val_theta: Tensor
) -> dict[str, Any]:
    """Freeze calibration-fitted coefficients selected only on original validation groups."""
    if cal_codes.ndim != 3 or val_codes.ndim != 3 or cal_codes.shape[-1] != 6:
        raise ValueError("Require individual-context six-code tensors")
    cal, val = (x.flatten(0, 1).double().cpu() for x in (cal_codes, val_codes))
    labels = [
        theta.double().cpu().repeat_interleave(code.shape[1], 0)
        for code, theta in ((cal_codes, cal_theta), (val_codes, val_theta))
    ]
    mean = cal.mean(0)
    eigenvalues, vectors = cast(tuple[Tensor, Tensor], torch.linalg.eigh(torch.cov(cal.T)))  # pyright: ignore[reportUnknownMemberType]
    floor = max(float(eigenvalues.max()) * 1e-12, 1e-20)
    transform = vectors / eigenvalues.clamp_min(floor).sqrt()
    cal, val = ((x - mean) @ transform for x in (cal, val))
    generator = torch.Generator().manual_seed(20261026072)
    omega = torch.randn(6, 256, dtype=torch.float64, generator=generator)
    phase = 2 * math.pi * torch.rand(256, dtype=torch.float64, generator=generator)
    candidates: list[dict[str, Any]] = []
    for family, width in [
        ("linear", 1.0),
        ("cubic", 1.0),
        *[("rbf_256_random_features", width) for width in (0.5, 1.0, 2.0, 4.0)],
    ]:
        features = [feature_map(x, family, width, omega, phase) for x in (cal, val)]
        center, scale = features[0].mean(0), features[0].std(0).clamp_min(1e-10)
        matrices = [torch.cat(((x - center) / scale, x.new_ones(len(x), 1)), 1) for x in features]
        covariance = matrices[0].T @ matrices[0] / len(cal)
        rhs = matrices[0].T @ labels[0] / len(cal)
        identity = torch.eye(covariance.shape[0], dtype=torch.float64)
        identity[-1, -1] = 0
        for strength in (1e-5, 1e-3, 0.1, 1.0, 10.0):
            weights = cast(Tensor, torch.linalg.solve(covariance + strength * identity, rhs))  # pyright: ignore[reportUnknownMemberType]
            candidates.append(
                {
                    "family": family,
                    "width": width,
                    "ridge": strength,
                    "center": center,
                    "scale": scale,
                    "weights": weights,
                    "validation_r2": _r2(labels[1], matrices[1] @ weights),
                }
            )
    best = torch.stack([x["validation_r2"] for x in candidates]).argmax(0)
    selected: list[dict[str, Any]] = []
    for factor, index in enumerate(best):
        row = candidates[int(index)]
        selected.append(
            {
                **{key: row[key] for key in ("family", "width", "ridge", "center", "scale")},
                "weights": row["weights"][:, factor].clone(),
                "validation_r2": float(row["validation_r2"][factor]),
            }
        )
    return {
        "mean": mean,
        "transform": transform,
        "omega": omega,
        "phase": phase,
        "selected": selected,
        "calibration_groups": len(cal_codes),
        "validation_groups": len(val_codes),
        "selection_excludes_precision_validation": True,
    }


@torch.no_grad()
def predict(probe: dict[str, Any], codes: Tensor, translation: Tensor | None = None) -> Tensor:
    """Apply immutable readouts to each context, undoing a known global code translation."""
    if codes.ndim != 3 or codes.shape[-1] != 6:
        raise ValueError("Require grouped individual-context six-code inputs")
    values = codes.double().cpu().flatten(0, 1)
    if translation is not None:
        if translation.shape != (6,):
            raise ValueError("Global translation must have six coordinates")
        values = values - translation.double().cpu()
    values = (values - probe["mean"]) @ probe["transform"]
    columns: list[Tensor] = []
    for row in probe["selected"]:
        pieces: list[Tensor] = []
        for start in range(0, len(values), 4096):
            x = feature_map(
                values[start : start + 4096],
                row["family"],
                row["width"],
                probe["omega"],
                probe["phase"],
            )
            x = torch.cat(((x - row["center"]) / row["scale"], x.new_ones(len(x), 1)), 1)
            pieces.append(x @ row["weights"])
        columns.append(torch.cat(pieces))
    return torch.stack(columns, 1).reshape(*codes.shape[:2], 6)


def score(predictions: Tensor, theta: Tensor) -> tuple[dict[str, Any], Tensor]:
    """Score every context independently; retain group errors for uncertainty analyses."""
    if predictions.shape != (len(theta), 10, 6):
        raise ValueError("Require all ten individual-context readouts per physical group")
    truth = theta.double().cpu()[:, None].expand_as(predictions)
    r2 = _r2(truth.flatten(0, 1), predictions.flatten(0, 1))
    errors = (predictions - truth).square().mean(1)
    return {
        "r2": r2.numpy().tolist(),
        "minimum_r2": float(r2.min()),
        "all_six_above_0_80": bool((r2 >= 0.8).all()),
        "groups": len(theta),
        "contexts_per_group": 10,
        "protocol": (
            "Frozen original calibration/validation-selected probes; one context per code; "
            "no precision-validation fitting or selection."
        ),
    }, errors
