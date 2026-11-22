"""Run the paired RGB episode-code teacher continuation on original sparse checkpoints."""

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.vision_dimer import data as vdata
from diagnostics.vision_dimer import train as release
from diagnostics.vision_dimer.analysis import load_model
from scjepa.experiments.storage import file_lock, write_json

from .consensus_gradient import whitening
from .single_acquisition import _head, _idle_gpu  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import OUTPUT, checkpoint_hashes, declared, read, sha
from .vision_invariant_eval import SPLITS, decision, endpoint, spectrum
from .vision_invariant_train import continue_training, teacher_alignment, teacher_table
from .vision_loss import (
    ARCHIVE,
    _control_interval,  # pyright: ignore[reportPrivateUsage]
    _fork,  # pyright: ignore[reportPrivateUsage]
    _runtime,  # pyright: ignore[reportPrivateUsage]
)

PHASES = ("prepare", "own", "sibling", "assess")
MODES = {"own": "same", "sibling": "other"}
METHOD = (
    "Hypothesis: reducing episode-specific variation in the code preserves forecast, physical "
    "retention and graph recovery. Resume the original full sparse optimizer, EMA, sampler, "
    "window and model RNG, and GECO at 300000; continue to 500000. "
    "Both treatments add a label-free frozen-teacher loss only to the trainable code encoder. "
    "Use the same 300000 model and whiten codes with its full TRAIN population covariance "
    "across all feasible native episode windows. Each student predicts the same-window teacher "
    "(own), or the mean of the nine other teacher episodes of the same system at their sampled "
    "windows (sibling). The single paired difference is the target. Cap auxiliary gradient norm "
    "at the native forecast norm on this parameter group, before the original global clip. "
    "Forecast, GECO, gates, SIGReg, EMA, updates, rates and stochastic draws remain native. "
    "Physical labels enter only the original held-out evaluation. Record raw auxiliary loss, "
    "cap frequency, native and auxiliary gradient norms, realized ratio and throughput.\n\n"
    "A separate unchanged first logging interval must match the archived numeric trace within "
    "rtol=1e-4 and atol=1e-8. On first-window calibration and validation require at least 20% "
    "lower worst-direction within/total code share versus BOTH original and own endpoints, "
    "hard raw forecast MSE at most 1.05 times original on both splits, minimum validation drag "
    "R² at least 0.95 and at most 0.01 below original, Pearson state MCC at most 0.01 below "
    "original, and total validation SHD no larger. Keep every original code direction resolved. "
    "Require cached teacher codes to match native renders on a fixed TRAIN probe at "
    "rtol=1e-4 and atol=1e-5. Reproduce each stored hard MSE at rtol=1e-4 and atol=1e-8. "
    "Use whole-system bootstrap intervals. Apply the gate to point estimates. A pass "
    "licenses seeds 1-4. A scientific miss leaves cross-episode alignment an open training "
    "problem. A rank or replay failure prevents this comparison from supporting an improvement. "
    "Never select a checkpoint or alter the criterion based on these measurements.\n"
)


def commands(commit: str, seed: int) -> list[str]:
    """Return exact phase commands for the preregistration."""
    return [
        "PYTHONPATH=$PWD:$PWD/src .venv/bin/python -m ICML_review.code.vision_invariant "
        + f"{commit} {seed} {p}"
        for p in PHASES
    ]


def verify_files(files: dict[str, str]) -> None:
    """Check every recorded dependency before reuse."""
    for name, value in files.items():
        if sha(Path(name)) != value:
            raise ValueError(f"input digest changed at {name}")


def artifacts(paths: list[Path]) -> dict[str, str]:
    """Hash checkpoint, evidence and declaration files."""
    return {str(p): sha(p) for p in paths}


def performance(path: Path, start: int) -> dict[str, Any]:
    """Summarize native logged wall time only over continuation updates."""
    rows = [r for r in map(json.loads, path.read_text().splitlines()) if r["step"] > start]
    if not rows or rows[-1]["step"] != 500_000:
        raise ValueError("incomplete continuation history")
    seconds = 0.0
    prev = start
    for row in rows:
        seconds += (row["step"] - prev) * row["seconds_per_step"]
        prev = row["step"]
    return {
        "logged_seconds": seconds,
        "seconds_per_update": seconds / (prev - start),
        "updates": prev - start,
    }


def run(commit: str, seed: int, phase: str) -> None:
    """Run an independently locked phase of a declared scientific pilot or admitted replication."""
    _head(commit)
    if seed:
        pilot = read(OUTPUT / "vision-invariant-s0/metrics.json")
        if (
            pilot.get("commit") != commit
            or pilot.get("seed") != 0
            or pilot.get("passed") is not True
        ):
            raise ValueError("a complete passing seed-zero trial under this commit is required")
    output = OUTPUT / f"vision-invariant-s{seed}"
    reference = ARCHIVE / f"seed{seed}/sparse-500k/full"
    parent = ARCHIVE / f"seed{seed}/sparse-300k/full/last.pt"
    stage = output / phase if phase in MODES else output
    stage.mkdir(parents=True, exist_ok=True)
    with file_lock(output / f".{phase}.lock"):
        cfg = release.StageConfig.from_saved(read(reference / "config.json")["config"])
        if cfg.stage != "full" or cfg.steps != 500_000 or cfg.seed != seed or cfg.device != "cuda":
            raise ValueError("unexpected archived configuration")
        cfg = replace(
            cfg,
            output_dir=str(stage.resolve()),
            wandb_group=f"icml-review-vision-invariant-s{seed}-{phase}",
        )
        data = Path(cfg.data_dir)
        split_record = read(ARCHIVE / "release-check-20261120/data-manifest-with-test.json")[
            "splits"
        ]
        actual = read(data / "manifest.json")["splits"]
        splits = {s: split_record[s]["sha256"] for s in ("train", *SPLITS)}
        if any(splits[s] != actual[s]["sha256"] for s in splits):
            raise ValueError("input split declaration changed")
        fork = read(reference / "fork.json")
        source_hash = fork["source_sha256"]
        starter = Path(cfg.parent or "")
        auth = checkpoint_hashes()
        identities = {
            str(parent): source_hash,
            str(reference / "final.pt"): auth[str(reference / "final.pt")],
            str(starter): auth[str(starter.resolve().relative_to(Path.cwd()))],
        }
        if fork["source_step"] != 300_000 or Path(fork["source"]).resolve() != parent.resolve():
            raise ValueError("common parent path or step differs from archived fork")
        note = (
            f"# Episode-code target experiment, RGB seed {seed}\n\n{METHOD}\n"
            f"Commit {commit}. Numeric tau {cfg.tolerance()}. Data {data.name}. "
            f"Split SHA256 {json.dumps(splits, sort_keys=True)}. Archived checkpoint identities "
            f"{json.dumps(identities, sort_keys=True)}. Training and evaluation commands\n\n"
            + "\n".join(commands(commit, seed))
            + "\n"
        )
        if phase == "prepare":
            declared(output / "NOTE.md", note)
        elif (output / "NOTE.md").read_text() != note:
            raise ValueError("run declaration differs from the prepared trial")
        subprocess.run(
            ["sha256sum", "-c", "--quiet", "reported-checkpoints.sha256"], cwd="outputs", check=True
        )
        verify_files(identities)
        base: dict[str, Any] = {
            "commit": commit,
            "seed": seed,
            "source_sha256": source_hash,
            "inputs": identities,
            "splits": splits,
            "tolerance": cfg.tolerance(),
            "note_sha256": sha(output / "NOTE.md"),
        }
        pre = output / "prepare.json"
        metrics = output / "metrics.json"
        receipt = (
            pre if phase == "prepare" else (metrics if phase == "assess" else stage / "result.json")
        )

        def close_unready(record: dict[str, Any]) -> None:
            if not record["ready"] and not metrics.exists():
                write_json(
                    metrics,
                    {
                        **base,
                        "prepare_sha256": sha(pre),
                        "passed": False,
                        "checks": record["checks"],
                        "files": artifacts([pre]),
                    },
                )

        if receipt.exists():
            existing = read(receipt)
            if any(existing[k] != v for k, v in base.items()):
                raise ValueError("completed phase identity differs")
            verify_files(existing["files"])
            if phase == "prepare":
                close_unready(existing)
            return
        if phase != "prepare":
            preparation = read(pre)
            if any(preparation[k] != v for k, v in base.items()):
                raise ValueError("prepared trial identity differs")
            verify_files(preparation["files"])
            if not preparation["ready"]:
                raise ValueError("matching control replay and resolved teacher chart required")
            base["prepare_sha256"] = sha(pre)
        if phase != "assess":
            _idle_gpu()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            base["runtime"] = _runtime()
        _head(commit)
        if phase == "prepare":
            rec = _prepare(base, output, cfg, parent, reference, note, commit)
            _head(commit)
            write_json(pre, rec)
            close_unready(rec)
        elif phase in MODES:
            _arm(base, output, cfg, parent, phase, commit)
        else:
            records = {"native": read(output / "native.json")}
            files = [pre, output / "native.json"]
            for arm, mode in MODES.items():
                stage_record = read(output / arm / "result.json")
                if any(stage_record[k] != v for k, v in base.items() if k != "runtime"):
                    raise ValueError("completed arm has a different lineage")
                verify_files(stage_record["files"])
                files.extend([output / arm / "result.json", output / arm / "endpoint.json"])
                records[mode] = read(output / arm / "endpoint.json")
            assessment = decision(records)
            _head(commit)
            write_json(
                metrics,
                {
                    **base,
                    **assessment,
                    "checkpoint_sha256": {a: r["sha256"] for a, r in records.items()},
                    "files": artifacts(files),
                    "commands": commands(commit, seed),
                },
            )


def _prepare(
    base: dict[str, Any],
    out: Path,
    cfg: release.StageConfig,
    parent: Path,
    reference: Path,
    note: str,
    commit: str,
) -> dict[str, Any]:
    replay = _control_interval(parent, cfg, reference, base["source_sha256"], note, commit, 300_000)
    files = [
        out / "control-replay/replay.json",
        out / "control-replay/last.pt",
        out / "control-replay/final.pt",
        out / "control-replay/NOTE.md",
    ]
    checks = {"native_replay": bool(replay["numerically_matched"])}
    t0 = time.perf_counter()
    if checks["native_replay"]:
        _head(commit)
        device = torch.device(cfg.device)
        data = Path(cfg.data_dir)
        model, _, _ = load_model(parent, device)
        model.eval().requires_grad_(False)
        train = vdata.RenderedInputs(data, "train", device)
        table = teacher_table(model, train)
        alignment = teacher_alignment(model, train, table)
        base["teacher_lookup_probe"] = alignment
        checks["native_teacher_lookup"] = alignment["matched"] is True
        chart = spectrum(table)
        checks["resolved_train_teacher"] = chart["full_rank"]
        base.update({"teacher_spectrum": chart, "teacher_shape": list(table.shape)})
        if checks["resolved_train_teacher"] and checks["native_teacher_lookup"]:
            invsqrt = whitening(table.flatten(1, 2))
            temp = out / ".teacher.partial.pt"
            torch.save({"table": table, "invsqrt": invsqrt}, temp)
            temp.replace(out / "teacher.pt")
            base["teacher_seconds"] = time.perf_counter() - t0
            base["invsqrt"] = invsqrt.tolist()  # pyright: ignore[reportUnknownMemberType]
            files.append(out / "teacher.pt")
            _head(commit)
            original = endpoint(reference / "final.pt", data, device)
            write_json(out / "native.json", original)
            files.append(out / "native.json")
            checks["native_first_window"] = all(original[s]["mse_matched"] for s in SPLITS)
            checks["native_heldout_rank"] = all(original[s]["chart"]["full_rank"] for s in SPLITS)
    return {
        **base,
        "checks": checks,
        "ready": all(checks.values()),
        "files": artifacts(files),
        "native_performance": performance(reference / "history.jsonl", 300_000),
    }


def _arm(
    base: dict[str, Any], out: Path, cfg: release.StageConfig, parent: Path, phase: str, commit: str
) -> None:
    stage = Path(cfg.output_dir)
    declared(
        stage / "NOTE.md",
        (out / "NOTE.md").read_text() + f"\nRunning {phase}. "
        f"Preparation SHA256 {base['prepare_sha256']}.\n",
    )
    _fork(parent, cfg, base["source_sha256"])
    t0 = time.perf_counter()
    if not (stage / "final.json").exists():
        cached = cast(
            dict[str, Tensor], torch.load(out / "teacher.pt", map_location="cpu", weights_only=True)
        )
        continue_training(
            cfg, cached["table"], cached["invsqrt"], MODES[phase], lambda: _head(commit)
        )
    if (
        release.load_checkpoint(stage / "last.pt")["step"] != cfg.steps
        or read(stage / "final.json")["step"] != cfg.steps
    ):
        raise ValueError("incomplete sparse endpoint")
    _head(commit)
    endpoint_path = stage / "endpoint.json"
    write_json(
        endpoint_path, endpoint(stage / "final.pt", Path(cfg.data_dir), torch.device(cfg.device))
    )
    files = [
        stage / k
        for k in (
            "NOTE.md",
            "fork.json",
            "history.jsonl",
            "anchor-history.jsonl",
            "final.pt",
            "last.pt",
            "final.json",
            "endpoint.json",
        )
    ]
    rows = [json.loads(line) for line in (stage / "anchor-history.jsonl").read_text().splitlines()]
    updates = sum(r["updates"] for r in rows)
    if updates != cfg.steps - 300_000:
        raise ValueError("incomplete auxiliary gradient telemetry")
    means = {
        k: sum(r[k] * r["updates"] for r in rows) / updates
        for k in rows[0]
        if k not in ("step", "updates")
    }
    _head(commit)
    write_json(
        stage / "result.json",
        {
            **base,
            "mode": MODES[phase],
            "files": artifacts(files),
            "checkpoint_sha256": sha(stage / "final.pt"),
            "performance": performance(stage / "history.jsonl", 300_000),
            "invocation_training_and_evaluation_seconds": time.perf_counter() - t0,
            "mean_gradient": means,
        },
    )


def main() -> None:
    """Run one phase at a pinned revision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("seed", type=int, choices=range(5))
    parser.add_argument("phase", choices=PHASES)
    a = parser.parse_args()
    run(a.commit, a.seed, a.phase)


if __name__ == "__main__":
    main()
