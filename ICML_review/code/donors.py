"""Evaluate transfer between disjoint support and query episodes at fixed weights."""

from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from diagnostics.vision_dimer.analysis import endpoint, load_model, spread
from diagnostics.vision_dimer.data import RenderedInputs
from diagnostics.vision_dimer.train import rendered_batches
from scjepa.models.model import SCJEPA

DATA = Path("outputs/data/vision-dimer-rgb-v1")
ROOT = Path("outputs/diagnostics/vision-dimer-rgb-v1-s05all")


def risks(model: SCJEPA, states: Tensor, codes: Tensor, actions: Tensor, target: Tensor) -> Tensor:
    """Return group-level MSE for single, averaged, alien and repeated-donor contrasts."""
    groups, views, _ = codes.shape
    if views < 4 or views % 2:
        raise ValueError("Each group needs two equal panels with at least two episodes each")
    scores: list[Tensor] = []
    for support, query in (
        (slice(0, views // 2), slice(views // 2, views)),
        (slice(views // 2, views), slice(0, views // 2)),
    ):
        c = codes[:, support]
        h = states[:, query, model.context_len - model.predictor.history_len : model.context_len]
        a = actions[:, query, model.context_len - model.predictor.history_len :]
        y = target[:, query]

        def forecast(value: Tensor, history: Tensor, acts: Tensor, truth: Tensor) -> Tensor:
            prediction = model.rollout(
                history.flatten(0, 1),
                value[:, None].expand(-1, history.shape[1], -1).flatten(0, 1),
                truth.shape[2],
                acts.flatten(0, 1),
            )
            return prediction.reshape_as(truth)

        single = torch.stack([forecast(donor, h, a, y) for donor in c.unbind(1)])
        pooled = forecast(c.mean(1), h, a, y)
        alien = forecast(c.mean(1).roll(1, 0), h, a, y)
        # Keep the system as the sampling unit. Variance is over the finite donor panel.
        independent = (single - y).square().mean(dim=(0, 2, 3, 4))
        dispersion = single.var(0, unbiased=False).mean(dim=(1, 2, 3))
        scores.append(
            torch.stack(
                (
                    independent,
                    (pooled - y).square().mean(dim=(1, 2, 3)),
                    (alien - y).square().mean(dim=(1, 2, 3)),
                    dispersion,
                ),
                1,
            )
        )
    return torch.stack(scores).mean(0).reshape(groups, 4)


@torch.no_grad()
def run(seed: int, arm: str) -> dict[str, object]:
    """Evaluate a fixed reported endpoint on both development splits."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = endpoint(ROOT, seed, arm)
    model, _, _ = load_model(path, device)
    model.eval()
    record: dict[str, object] = {"checkpoint": str(path)}
    for split in ("calibration", "validation"):
        rows: list[Tensor] = []
        all_codes: list[Tensor] = []
        for batch in rendered_batches(RenderedInputs(DATA, split, device), size=8):
            x, actions = batch["observations"], batch["actions"]
            flat = x.flatten(0, 1)
            z = model.online_encoder.encode_sequence(flat)
            c = cast(
                Tensor,
                model.parameter_encoder(
                    z[:, : model.context_len], actions.flatten(0, 1)[:, : model.context_len - 1]
                ),
            )
            target = model.target_encoder.encode_sequence(flat[:, model.context_len :])
            shape = x.shape[:2]
            grouped_codes = c.reshape(*shape, -1)
            all_codes.append(grouped_codes.cpu())
            rows.append(
                risks(
                    model,
                    z.reshape(*shape, *z.shape[1:]),
                    grouped_codes,
                    actions,
                    target.reshape(*shape, *target.shape[1:]),
                )
                .double()
                .cpu()
            )
        metrics = torch.cat(rows)
        record[split] = {
            "spread": spread(torch.cat(all_codes)),
            "systems": metrics.shape[0],
            "columns": ["single_mse", "pooled_mse", "alien_mse", "donor_variance"],
            "mean": metrics.mean(0).tolist(),  # pyright: ignore[reportUnknownMemberType]
            "per_system": metrics.tolist(),  # pyright: ignore[reportUnknownMemberType]
        }
    return record
