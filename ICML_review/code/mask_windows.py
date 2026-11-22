"""Separate gate and window effects in frozen full vision checkpoints on held-out contexts."""

import argparse
import math
import shlex
import subprocess
from typing import Any, Literal, cast

import torch
from torch import Tensor

from diagnostics.vision_dimer import data, train
from diagnostics.vision_dimer.model import LaggedMaskedPredictor
from scjepa.experiments.storage import file_lock, write_json
from scjepa.models.model import SCJEPA, ModelOutput, cross_episode_parameters

from .single_acquisition import _head, _idle_gpu  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import DATA, OUTPUT, PYTHON, ROOT, checkpoint_hashes, declared, read, sha
from .vision_loss import hard_forecast

CELLS = ("first_hard", "first_sampled", "uniform_hard", "uniform_sampled")
PAIRS = {
    "uniform_minus_first_hard": ("uniform_hard", "first_hard"),
    "uniform_minus_first_sampled": ("uniform_sampled", "first_sampled"),
    "hard_minus_sampled_first": ("first_hard", "first_sampled"),
    "hard_minus_sampled_uniform": ("uniform_hard", "uniform_sampled"),
}
N_BOOT = 2000
REPEATS = 4
RTOL, ATOL = 1e-4, 1e-8
PROTOCOL = (
    "Hypothesis: changing the gate policy and/or the episode-window distribution can change "
    "which side of the reported tolerance a held-out forecast occupies. Measure MSE/tau "
    "on all calibration and validation contexts of the original full endpoint, separately. "
    "Cross deterministic and sampled gates with first-window and uniform-window evaluation. "
    "Keep the cyclic evaluation donor shift, model, EMA targets, two-step horizon, original "
    "batch size and tolerance. Use four independent offset/gate schedules and common gate "
    "noise for each first/uniform pair. Within a window both gates share targets and donors. "
    "Report signed paired differences in MSE/tau, simple cell ratios, and the interaction. "
    "Resample entire contexts together across both factors and all repeats for 95% percentile "
    "intervals, conditional on those schedules. Report schedule SD separately. Positive "
    "uniform-minus-first values mean offsets increase forecast error. Positive hard-minus-sampled "
    "values mean deployment increases error. An interval spanning zero leaves the direction "
    "unresolved. A crossing of one identifies a change in feasibility under the measured "
    "conditions. Offset effects on fixed held-out contexts cannot estimate TRAIN generalization "
    "or establish overfitting. Proceed from seed zero only if the released first-window "
    "deterministic CAL and VAL MSE both replicate within rtol=1e-4, atol=1e-8 and the fixed-window "
    "hard error of every group is stable to that tolerance across schedules. No fitting."
)
type Split = Literal["calibration", "validation"]


def _sequence(x: Tensor) -> list[float]:
    return [float(v) for v in x.cpu().unbind()]


def _rng(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _interval(values: Tensor) -> list[float]:
    return _sequence(torch.quantile(values, torch.tensor([0.025, 0.975], dtype=values.dtype)))


def summarize(cells: dict[str, Tensor], tau: float, bootstrap_seed: int) -> dict[str, Any]:
    """Bootstrap entire contexts, retaining all paired gates, windows, views and schedules."""
    draws, groups = cells["first_hard"].shape
    if not tau > 0 or groups < 2 or draws < 2:
        raise ValueError("expected positive tau, multiple contexts and multiple schedules")
    indices = torch.randint(groups, (N_BOOT, groups), generator=_rng(bootstrap_seed))
    scaled = {key: value.double() / tau for key, value in cells.items()}

    def stats(values: Tensor) -> dict[str, Any]:
        by_group = values.mean(0)
        by_schedule = values.mean(1)
        return {
            "estimate": float(by_group.mean()),
            "context_bootstrap_95": _interval(by_group[indices].mean(1)),
            "schedule_means": _sequence(by_schedule),
            "schedule_sd": float(by_schedule.std(correction=1)),
        }

    estimates = {
        key: stats(val) | {"raw_mse": float(cells[key].double().mean())}
        for key, val in scaled.items()
    }
    contrasts: dict[str, Any] = {}
    for name, (a, b) in PAIRS.items():
        ga, gb = scaled[a].mean(0), scaled[b].mean(0)
        ratios = ga[indices].mean(1) / gb[indices].mean(1)
        contrasts[name] = stats(scaled[a] - scaled[b]) | {
            "ratio_of_mean_mse": float(ga.mean() / gb.mean()),
            "ratio_context_bootstrap_95": _interval(ratios),
        }
    contrasts["hard_minus_sampled_uniform_minus_first"] = stats(
        (scaled["uniform_hard"] - scaled["uniform_sampled"])
        - (scaled["first_hard"] - scaled["first_sampled"])
    )
    return {
        "units": "MSE / original tau",
        "cells": estimates,
        "contrasts": contrasts,
        "context_bootstrap_resamples": N_BOOT,
        "context_bootstrap_seed": bootstrap_seed,
    }


def _group_errors(model: SCJEPA, batch: dict[str, Tensor], gate_seed: int) -> tuple[Tensor, Tensor]:
    # Both window arms reset the generator immediately before the same released stochastic graph.
    torch.manual_seed(gate_seed)  # pyright: ignore[reportUnknownMemberType]
    actions = batch["actions"]
    out = cast(ModelOutput, model(batch["observations"], actions))
    donor = cross_episode_parameters(out.parameters, training=False).flatten(0, 1)
    deployed = hard_forecast(cast(LaggedMaskedPredictor, model.predictor), out, actions, donor)

    def errors(predictions: Tensor) -> Tensor:
        return (predictions - out.targets).square().double().mean((1, 2, 3)).cpu()

    return errors(deployed), errors(out.predictions)


@torch.no_grad()
def measure(
    model: SCJEPA,
    source: data.RenderedInputs,
    split: Split,
    batch_groups: int,
    tau: float,
    reference: float,
) -> dict[str, Any]:
    """Apply the two-by-two design to every group, storing paired raw errors by draw and batch."""
    if source.split != split or batch_groups < 1:
        raise ValueError("unexpected source split or batch size")
    model.eval()
    model.predictor.train()  # Encoder and donor policy stay in evaluation mode.
    rows: dict[str, list[Tensor]] = {k: [] for k in CELLS}
    batches: list[dict[str, Any]] = []
    available = source.episode_frames - source.frames + 1
    salt = (1 if split == "validation" else 0) * 1_000_000
    for repeat in range(REPEATS):
        window_seed = salt + repeat
        starts = torch.randint(available, (len(source), source.views), generator=_rng(window_seed))
        parts: dict[str, list[Tensor]] = {k: [] for k in CELLS}
        for group in range(0, len(source), batch_groups):
            end = min(group + batch_groups, len(source))
            ids = torch.arange(group, end)
            gate_seed = 10_000_000 + salt + repeat * len(source) + group
            batch_means: dict[str, float] = {}
            for window, offsets in (("first", None), ("uniform", starts[group:end])):
                batch = source.batch(ids, offsets)
                hard, sampled = _group_errors(model, batch, gate_seed)
                for graph, value in (("hard", hard), ("sampled", sampled)):
                    key = f"{window}_{graph}"
                    parts[key].append(value)
                    batch_means[key] = float(value.mean())
            batches.append(
                {
                    "repeat": repeat,
                    "group_start": group,
                    "group_stop": end,
                    "window_seed": window_seed,
                    "gate_seed": gate_seed,
                    "uniform_offset_counts": [
                        int(v)
                        for v in torch.bincount(starts[group:end].flatten(), minlength=available)
                    ],
                    "mean_raw_mse": batch_means,
                }
            )
        for key in CELLS:
            rows[key].append(torch.cat(parts[key]))
    model.eval()
    matrices = {k: torch.stack(values) for k, values in rows.items()}
    result = summarize(matrices, tau, salt + 331)
    first = matrices["first_hard"]
    measured = float(first.mean())
    return result | {
        "split": split,
        "groups": len(source),
        "views": source.views,
        "frames": source.frames,
        "episode_frames": source.episode_frames,
        "schedules": REPEATS,
        "fixed_donor_shift": 1,
        "raw_mse_by_repeat_then_group": {
            k: [_sequence(row) for row in m.unbind()] for k, m in matrices.items()
        },
        "batches": batches,
        "first_window_reference": {
            "stored_mse": reference,
            "measured_mse": measured,
            "absolute_difference": abs(measured - reference),
            "rtol": RTOL,
            "atol": ATOL,
            "bitwise_constant_across_schedules": bool(
                torch.equal(first, first[:1].expand_as(first))
            ),
            "stable_across_schedules": torch.allclose(first, first[:1], rtol=RTOL, atol=ATOL),
            "matches": math.isclose(measured, reference, rel_tol=RTOL, abs_tol=ATOL),
        },
    }


def main() -> None:
    """Evaluate a pinned archive after checkpoint verification and a successful technical pilot."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("commit")
    cli.add_argument("seed", type=int, choices=range(5))
    args = cli.parse_args()
    rev, seed = str(args.commit), int(args.seed)
    _head(rev)
    folder = OUTPUT / f"mask-windows-s{seed}"
    archive = ROOT / f"seed{seed}/sparse-500k/full"
    checkpoint = archive / "final.pt"
    identity = checkpoint_hashes()[str(checkpoint)]
    manifest = read(DATA / "manifest.json")
    archived_data = read(ROOT / "release-check-20261120/data-manifest-with-test.json")
    splits = {s: manifest["splits"][s]["sha256"] for s in ("calibration", "validation")}
    if any(h != archived_data["splits"][s]["sha256"] for s, h in splits.items()):
        raise ValueError("data identities differ from the archived release")
    config = train.StageConfig.from_saved(read(archive / "config.json")["config"])
    if config.seed != seed or config.stage != "full" or config.steps != 500_000:
        raise ValueError("expected the same-seed full sparse endpoint")
    command = "PYTHONPATH=$PWD:$PWD/src " + shlex.join(
        [PYTHON, "-m", __spec__.name, rev, str(seed)]
    )
    metadata = {
        "command": command,
        "commit": rev,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": identity,
        "split_sha256": splits,
        "tau": config.tolerance(),
        "reference_summary_sha256": sha(archive / "final.json"),
    }
    with file_lock(folder / ".lock"):
        declared(
            folder / "NOTE.md",
            f"# Gate policy and time window, seed {seed}\n\n{PROTOCOL}\n\n"
            f"Commit `{rev}`, checkpoint `{checkpoint}`, SHA-256 `{identity}`. "
            f"Data `{DATA}`, split SHA-256 `{splits}`, tau `{config.tolerance()}`.\n\n"
            f"    {command}\n",
        )
        if seed:
            pilot = read(OUTPUT / "mask-windows-s0/metrics.json")
            if pilot["commit"] != rev or not pilot["verified"]:
                raise RuntimeError("expansion requires a passing seed-zero technical check")
        subprocess.run(
            ["sha256sum", "-c", "--quiet", "reported-checkpoints.sha256"], cwd="outputs", check=True
        )
        if sha(checkpoint) != identity:
            raise RuntimeError("checkpoint digest differs from archive")
        result_path = folder / "metrics.json"
        if result_path.exists():
            existing = read(result_path)
            if any(existing[k] != v for k, v in metadata.items()):
                raise RuntimeError("stored result metadata changed")
            return
        _idle_gpu()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda")
        model = train.model_from_checkpoint(train.load_checkpoint(checkpoint), phase="full").to(
            device
        )
        original = read(archive / "final.json")
        results: dict[str, Any] = {}
        for split, key in (
            ("calibration", "calibration_prediction_mse"),
            ("validation", "prediction_mse"),
        ):
            _head(rev)
            results[split] = measure(
                model,
                data.RenderedInputs(DATA, split, device),
                cast(Split, split),
                config.batch_groups,
                config.tolerance(),
                original[key],
            )
        checks = [r["first_window_reference"] for r in results.values()]
        _head(rev)
        write_json(
            result_path,
            metadata
            | {
                "results": results,
                "gpu": torch.cuda.get_device_name(),
                "torch": str(torch.__version__),
                "tf32": False,
                "verified": all(x["matches"] and x["stable_across_schedules"] for x in checks),
            },
        )


if __name__ == "__main__":
    main()
