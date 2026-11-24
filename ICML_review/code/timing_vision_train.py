"""Apply timing ablations inside the native RGB stage without changing its sampling or optimizer."""

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor, nn

from diagnostics.vision_dimer import data as vdata
from diagnostics.vision_dimer import train as release
from diagnostics.vision_dimer.analysis import load_model
from scjepa.experiments.storage import write_json
from scjepa.models.model import SCJEPA, ModelOutput

from .consensus_gradient import add_capped_gradient, anchor, whitening
from .timing_loss import add_sibling_gradient
from .vision_acquisition import read, sha
from .vision_invariant_eval import spectrum
from .vision_invariant_train import table_batch, teacher_alignment, teacher_table


def frozen_teacher(
    parent: Path, data: Path, device: torch.device, out: Path
) -> tuple[Tensor, Tensor]:
    """Freeze the adapted encoder and fit the archived sibling metric over all TRAIN windows."""
    model, _, _ = load_model(parent, device)
    model.eval().requires_grad_(False)
    source = vdata.RenderedInputs(data, "train", device)
    table = teacher_table(model, source)
    probe = teacher_alignment(model, source, table)
    chart = spectrum(table)
    write_json(
        out / "teacher.json",
        {
            "parent": str(parent),
            "sha256": sha(parent),
            "data_manifest_sha256": sha(data / "manifest.json"),
            "split": "train",
            "shape": list(table.shape),
            "spectrum": chart,
            "alignment": probe,
        },
    )
    if not probe["matched"] or not chart["full_rank"]:
        raise ValueError("adapted TRAIN teacher lacks the required chart or lookup alignment")
    return table, whitening(table.flatten(1, 2))


def restore_log(path: Path, step: int) -> None:
    """Archive and trim rows past the resumable state, including a torn last append."""
    if not path.exists():
        return
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    kept: list[bytes] = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise
            break
        if row["step"] <= step:
            kept.append(line if line.endswith(b"\n") else line + b"\n")
    trimmed = b"".join(kept)
    if trimmed != raw:
        archive = path.parent / "interrupted-history" / f"{sha(path)}-{path.name}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, archive)
        temporary = path.with_suffix(".repair.tmp")
        temporary.write_bytes(trimmed)
        temporary.replace(path)


def train_stage(
    cfg: release.StageConfig,
    schedule: str,
    teacher: tuple[Tensor, Tensor] | None = None,
    check_head: Callable[[], None] | None = None,
) -> None:
    """Add one bounded gradient to the native code encoder immediately before native backward."""
    from unittest.mock import patch

    if schedule not in {"entire", "sparse"} or cfg.stage not in {
        "joint",
        "acquire",
        "adapt",
        "full",
    }:
        raise ValueError("unexpected timing or RGB stage")
    if (cfg.stage == "full") != (teacher is not None):
        raise ValueError("full sparse training requires its own frozen, adapted teacher")
    out = Path(cfg.output_dir)
    resume = (out / "last.pt").is_file()
    step = int(release.load_checkpoint(out / "last.pt")["step"]) if resume else 0
    for stream in ("history.jsonl", "evaluations.jsonl", "sibling-history.jsonl"):
        restore_log(out / stream, step)
    if schedule == "sparse" and cfg.stage != "full":
        release.train(cfg, resume=resume)
        return
    table, invsqrt = (None, None) if teacher is None else teacher
    if invsqrt is not None:
        invsqrt = invsqrt.to(cfg.device)
    history = out / "sibling-history.jsonl"
    count = 0
    sums: dict[str, Tensor] = {}
    captured_target: Tensor | None = None
    graphs: list[tuple[Tensor, Tensor, tuple[nn.Parameter, ...]]] = []
    original_forward = SCJEPA.forward
    original_zero = torch.optim.AdamW.zero_grad

    class FrozenCache(release.FrozenCache):
        def batch(
            self, split: str, indices: Tensor, starts: Tensor | None = None
        ) -> dict[str, Tensor]:
            nonlocal captured_target
            batch = super().batch(split, indices, starts)
            if split == "train" and torch.is_grad_enabled():
                if captured_target is not None:
                    raise RuntimeError("unused cached EMA target")
                captured_target = batch["targets"]
            return batch

    class RenderedInputs(vdata.RenderedInputs):
        def batch(self, indices: Tensor, starts: Tensor | None = None) -> dict[str, Tensor]:
            nonlocal captured_target
            if table is not None and self.split == "train" and torch.is_grad_enabled():
                if captured_target is not None:
                    raise RuntimeError("unused frozen teacher batch")
                captured_target = table_batch(table, indices, starts).to(cfg.device)
            return super().batch(indices, starts)

    def forward(
        model: SCJEPA,
        observations: Tensor,
        actions: Tensor | None = None,
        prediction_start: int | None = None,
        *,
        oracle_parameters: Tensor | None = None,
    ) -> ModelOutput:
        nonlocal captured_target
        result = original_forward(
            model, observations, actions, prediction_start, oracle_parameters=oracle_parameters
        )
        if model.training and torch.is_grad_enabled():
            if graphs:
                raise RuntimeError("unused training graph")
            target = result.targets
            auxiliary = result.parameters
            if cfg.stage in release.FROZEN:
                if captured_target is None:
                    raise RuntimeError("missing native cached EMA forecast targets")
                target = captured_target
            elif table is not None:
                if captured_target is None or invsqrt is None:
                    raise RuntimeError("missing frozen sibling target")
                auxiliary = anchor(result.parameters, captured_target, invsqrt, "other")
            captured_target = None
            forecast = (result.predictions - target).square().mean()
            graphs.append((forecast, auxiliary, tuple(model.parameter_encoder.parameters())))
        return result

    def zero(optimizer: torch.optim.AdamW, set_to_none: bool = True) -> None:
        nonlocal step, count
        original_zero(optimizer, set_to_none=set_to_none)
        forecast, auxiliary, params = graphs.pop()
        add = add_sibling_gradient if table is None else add_capped_gradient
        stats = add(forecast, auxiliary, params)
        stats["forecast_mse"] = forecast.detach()
        for key, value in stats.items():
            sums[key] = value.detach() if key not in sums else sums[key] + value.detach()
        step += 1
        count += 1
        if step % cfg.log_every == 0 or step == cfg.steps:
            if check_head is not None:
                check_head()
            record = {
                "step": step,
                "updates": count,
                **{k: float(v) / count for k, v in sums.items()},
            }
            with history.open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            count = 0
            sums.clear()

    with (
        patch.object(release, "FrozenCache", FrozenCache),
        patch.object(vdata, "RenderedInputs", RenderedInputs),
        patch.object(SCJEPA, "forward", forward),
        patch.object(torch.optim.AdamW, "zero_grad", zero),
    ):
        release.train(cfg, resume=resume)
    if captured_target is not None or graphs or count or step != cfg.steps:
        raise RuntimeError("unpaired RGB update")
    if read(out / "final.json")["step"] != step:
        raise RuntimeError("incomplete RGB stage")
