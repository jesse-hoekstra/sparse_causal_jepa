"""Scale-invariant episode consistency for the frozen-state acquisition stage."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import Tensor

from diagnostics.masked_mlp_learned import run as runner
from diagnostics.shared.io import sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.loop import TrainingObjective


def within_shares(codes: Tensor) -> Tensor:
    """Within/total empirical variance per coordinate, with constant coordinates scored one."""
    if codes.ndim != 3 or min(codes.shape[:2]) < 2:
        raise ValueError("Expected at least two groups and episodes")
    shifted = codes - codes[0, 0]
    centered = shifted - shifted.mean((0, 1))
    # Both variances acquire the same factor, so division cancels it even in the gradient.
    magnitude = centered.detach().abs().amax((0, 1))
    centered = centered / torch.where(magnitude > 0, magnitude, 1.0)
    within = (centered - centered.mean(1, keepdim=True)).square().mean((0, 1))
    total = centered.square().mean((0, 1))
    return torch.where(total > 0, within / torch.where(total > 0, total, 1.0), 1.0)


def code_quality(codes: Tensor) -> dict[str, Any]:
    """Describe single-context consistency, scale, and informative shared modes without labels."""
    x = codes.double()
    groups, views, width = x.shape
    if min(groups, width) < 6:
        raise ValueError("Measuring six shared modes requires at least six groups and coordinates")
    mu = x.mean(1)
    within = (x - mu[:, None]).reshape(-1, width)
    w = within.T @ within / (groups * (views - 1))
    shared = torch.cov(mu.T) - w / views
    total_trace = float(x.flatten(0, 1).var(0).sum())
    vals = cast(Tensor, torch.linalg.eigvalsh(shared)).flip(0)  # pyright: ignore[reportUnknownMemberType]
    # Work on the resolved support of total covariance, including rank-deficient codes.
    rows = x.flatten(0, 1)
    centered = rows - rows.mean(0)
    total = centered.T @ centered / len(rows)
    spectrum, basis = cast(tuple[Tensor, Tensor], torch.linalg.eigh(total))  # pyright: ignore[reportUnknownMemberType]
    cutoff = float(spectrum[-1]) * max(rows.shape) * torch.finfo(x.dtype).eps
    positive = spectrum > cutoff
    directional: list[float] = []
    if bool(positive.any()):
        whitened = (within @ basis[:, positive]) / spectrum[positive].sqrt()
        normalized_within = whitened.T @ whitened / len(rows)
        eig = cast(Tensor, torch.linalg.eigvalsh(normalized_within))  # pyright: ignore[reportUnknownMemberType]
        directional = [float(v) for v in eig]
    shares = within_shares(x)
    return {
        "total_covariance_eigenvalues": [float(v) for v in spectrum],
        "total_rank_cutoff": cutoff,
        "total_numerical_rank": len(directional),
        "within_fraction_generalized_eigenvalues": directional,
        "worst_within_fraction": max(directional) if directional else None,
        "within_share": float(shares.mean()),
        "within_share_per_coordinate": [float(x) for x in shares],
        "total_trace": total_trace,
        "shared_eigenvalues": [float(x) for x in vals],
        "sixth_share": float(vals[5]) / total_trace if total_trace else 0.0,
    }


def group_moments(samples: dict[str, Tensor]) -> Tensor:
    """Collect per-system sufficient statistics for the two paired bootstrap estimates."""
    codes = samples["codes"].double()
    codes = codes - codes[0, 0]
    means = codes.mean(1)
    within = (codes - means[:, None]).square().mean(1)
    errors = (samples["predictions"].double() - samples["targets"].double()).square()
    return torch.cat((means, codes.square().mean(1), within, errors.flatten(1).mean(1)[:, None]), 1)


def paired_intervals(current: Tensor, historical: Tensor) -> dict[str, Any]:
    """Resample whole systems in common, including every episode of each drawn system."""
    if current.shape != historical.shape:
        raise ValueError("Paired acquisition evaluation requires the same groups and code width")
    groups, cols = current.shape
    width = (cols - 1) // 3
    seed = 5729
    g = torch.Generator().manual_seed(seed)
    indices = torch.randint(groups, (1000, groups), generator=g)

    def summarize(moments: Tensor, rows: Tensor) -> Tensor:
        row = moments[rows].mean(1)
        mean, second = row[:, :width], row[:, width : 2 * width]
        within, mse = row[:, 2 * width : 3 * width], row[:, -1]
        total = second - mean.square()
        ratio = torch.where(total > 0, within / torch.where(total > 0, total, 1.0), 1.0)
        return torch.column_stack((ratio.mean(1), mse))

    blocks: list[Tensor] = []
    for start in range(0, len(indices), 32):
        subset = indices[start : start + 32]
        blocks.append(summarize(current, subset) - summarize(historical, subset))
    differences = torch.cat(blocks)
    intervals = torch.quantile(differences, differences.new_tensor([0.025, 0.975]), dim=0)
    return {
        "estimator": "treatment minus original a0, paired percentile group bootstrap",
        "resamples": len(indices),
        "generator_seed": seed,
        "within_share_95": [float(x) for x in intervals[:, 0]],
        "prediction_mse_95": [float(x) for x in intervals[:, 1]],
    }


@dataclass(frozen=True, kw_only=True)
class AcquisitionConfig(runner.TrainConfig):
    """Include the actual calibration scale and starting-weight digest in the optimizer identity."""

    joint_calibration_prediction: float
    initialization_sha256: str


@contextmanager
def acquisition_runtime(
    original: Path, coefficient: float, *, restore_initial: bool
) -> Generator[None, None, None]:
    """Use the native stage machinery with a separately authenticated numerical extension."""
    base_identity = runner.code_identity()
    extension = Path(__file__)
    native_spec = runner._scientific_config  # pyright: ignore[reportPrivateUsage]

    def identity() -> str:
        return canonical_fingerprint({"base": base_identity, "loss_source": sha(extension)})

    def specification(config: runner.TrainConfig) -> dict[str, Any]:
        spec = native_spec(config)
        if isinstance(config, AcquisitionConfig):
            spec.update(
                initialization="verified historical first acquisition, copied at step zero",
                parameter_training="forecast plus normalized within-system code variance",
                objective=spec["objective"] | {"within_total_weight": coefficient},
            )
        return spec

    class Objective(TrainingObjective):
        def forward(self, output: ModelOutput) -> tuple[Tensor, dict[str, Tensor]]:
            loss, terms = super().forward(output)
            if self.training:
                ratio = within_shares(output.parameters).mean()
                terms["within_share"] = ratio
                terms["base"] = terms["base"] + coefficient * ratio
                loss = loss + coefficient * ratio
            return loss, terms

    def initialize(model: SCJEPA, _data: object, _device: torch.device) -> dict[str, Any]:
        saved = cast(
            dict[str, Any],
            torch.load(original / "initial.pt", map_location="cpu", weights_only=False),
        )
        model.load_state_dict(saved["model"], strict=True)
        return cast(dict[str, Any], saved["initialization"])

    # The native runner already supports distinct numerical identities for transferred states
    # and projections. Scope the two numerical substitutions to this single-process stage.
    with (
        patch.object(runner, "code_identity", identity),
        patch.object(runner, "_scientific_config", specification),
        patch.object(runner, "TrainingObjective", Objective),
    ):
        if restore_initial:
            with patch.object(runner, "initialize_shared_output", initialize):
                yield
        else:
            yield
