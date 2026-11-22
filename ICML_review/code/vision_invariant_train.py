"""Cache frozen TRAIN teacher codes and add a bounded gradient to the RGB code encoder."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import Tensor, nn

from diagnostics.vision_dimer import data as vdata
from diagnostics.vision_dimer import train as release
from scjepa.models.model import SCJEPA, ModelOutput

from .consensus_gradient import add_capped_gradient, anchor


@torch.no_grad()
def teacher_table(model: SCJEPA, source: vdata.RenderedInputs, batch: int = 8) -> Tensor:
    """Encode each admissible TRAIN episode/window, keeping its original group and view order."""
    if source.split != "train" or model.training:
        raise ValueError("require a frozen, evaluation-mode TRAIN teacher")
    k = model.context_len
    n = source.episode_frames - source.frames + 1
    rows: list[Tensor] = []
    for begin in range(0, len(source), batch):
        item = source.episodes(torch.arange(begin, min(begin + batch, len(source))))
        x, actions = item["observations"], item["actions"].flatten(0, 1)
        z = model.online_encoder.encode_sequence(x.flatten(0, 1))

        def contexts(series: Tensor, length: int) -> Tensor:
            wins = series.unfold(1, length, 1)[:, :n].transpose(-1, -2)
            return wins.reshape(-1, length, series.shape[-1])

        code = cast(Tensor, model.parameter_encoder(contexts(z, k), contexts(actions, k - 1)))
        rows.append(code.reshape(*x.shape[:2], n, -1).cpu())
    return torch.cat(rows)


def table_batch(table: Tensor, indices: Tensor, starts: Tensor | None) -> Tensor:
    """Look up the frozen teacher at the exact starts drawn by the native sampler."""
    return table[
        indices.cpu()[:, None], torch.arange(table.shape[1]), 0 if starts is None else starts.cpu()
    ]


@torch.no_grad()
def teacher_alignment(
    model: SCJEPA, source: vdata.RenderedInputs, table: Tensor
) -> dict[str, object]:
    """Compare the lookup against native rendered forwards on fixed TRAIN groups/start offsets."""
    if source.split != "train" or model.training:
        raise ValueError("expected an evaluation-mode TRAIN teacher")
    indices = torch.linspace(0, len(source) - 1, min(16, len(source))).round().long()
    starts = torch.arange(len(indices) * source.views).reshape(-1, source.views) % table.shape[2]
    item = source.batch(indices, starts)
    out = cast(ModelOutput, model(item["observations"], item["actions"]))
    direct = out.parameters.detach().cpu()
    cached = table_batch(table, indices, starts)
    return {
        "groups": indices.tolist(),  # pyright: ignore[reportUnknownMemberType]
        "starts": starts.tolist(),  # pyright: ignore[reportUnknownMemberType]
        "max_abs": float((direct - cached).abs().max()),
        "rms": float((direct - cached).square().mean().sqrt()),
        "rtol": 1e-4,
        "atol": 1e-5,
        "matched": torch.allclose(direct, cached, rtol=1e-4, atol=1e-5),
    }


def continue_training(
    cfg: release.StageConfig,
    table: Tensor,
    chart: Tensor,
    mode: str,
    check_head: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Resume the native full stage with only the code encoder receiving an auxiliary gradient."""
    if cfg.stage != "full" or mode not in ("native", "same", "other"):
        raise ValueError("expected a full sparse continuation and a registered target")
    outdir = Path(cfg.output_dir)
    step = int(release.load_checkpoint(outdir / "last.pt")["step"])
    history = outdir / "anchor-history.jsonl"
    if history.exists():
        rows = [
            line for line in history.read_text().splitlines() if json.loads(line)["step"] <= step
        ]
        history.write_text("".join(row + "\n" for row in rows))
    teachers: list[Tensor] = []
    graphs: list[tuple[Tensor, Tensor, tuple[nn.Parameter, ...]]] = []
    original_factory = release.model_from_checkpoint
    original_zero = torch.optim.AdamW.zero_grad
    chart = chart.to(cfg.device)
    sums: dict[str, Tensor] = {}
    count = 0

    class CapturedInputs(vdata.RenderedInputs):
        def batch(self, indices: Tensor, starts: Tensor | None = None) -> dict[str, Tensor]:
            if self.split == "train":
                if teachers:
                    raise RuntimeError("unused teacher batch")
                teachers.append(table_batch(table, indices, starts).to(cfg.device))
            return super().batch(indices, starts)

    def capture(model: nn.Module, _args: tuple[Any, ...], output: ModelOutput) -> None:
        if not model.training or not torch.is_grad_enabled():
            return
        if not teachers or graphs:
            raise RuntimeError("incomplete native forward/optimizer pairing")
        y = teachers.pop()
        params = tuple(cast(SCJEPA, model).parameter_encoder.parameters())
        if not params or not all(p.requires_grad for p in params):
            raise ValueError("code encoder is frozen")
        forecast = (output.predictions - output.targets).square().mean()
        aux = anchor(output.parameters, y, chart, "same" if mode == "native" else mode)
        graphs.append((forecast, aux, params))

    def factory(payload: dict[str, Any], *, phase: str = "dense") -> SCJEPA:
        model = original_factory(payload, phase=phase)
        model.register_forward_hook(capture)
        return model

    def zero(optimizer: torch.optim.AdamW, set_to_none: bool = True) -> None:
        nonlocal step, count
        original_zero(optimizer, set_to_none=set_to_none)
        forecast, aux, params = graphs.pop()
        if mode == "native":
            del forecast, aux, params
            return
        stats = add_capped_gradient(forecast, aux, params)
        nf, na = stats["consensus_forecast_grad_norm"], stats["consensus_raw_grad_norm"]
        stats["cap_bound"] = (na > nf).float()
        stats["added_to_native_grad_norm"] = torch.where(
            nf > 0,
            stats["consensus_gradient_scale"] * na / nf.clamp_min(torch.finfo(nf.dtype).tiny),
            0.0,
        )
        stats["forecast_mse"] = forecast.detach()
        for key, value in stats.items():
            sums[key] = sums.get(key, torch.zeros_like(value)) + value.detach()
        count += 1
        step += 1
        if step % cfg.log_every == 0 or step == cfg.steps:
            if check_head is not None:
                check_head()
            record = {
                "step": step,
                "updates": count,
                **{k: float(v) / count for k, v in sums.items()},
            }
            with history.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            sums.clear()
            count = 0

    with (
        patch.object(vdata, "RenderedInputs", CapturedInputs),
        patch.object(release, "model_from_checkpoint", factory),
        patch.object(torch.optim.AdamW, "zero_grad", zero),
    ):
        result = release.train(cfg, resume=True)
    if teachers or graphs or count:
        raise RuntimeError("training ended mid-batch")
    return result
