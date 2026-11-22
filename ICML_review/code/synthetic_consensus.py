"""Compare matched single-episode and disjoint-episode guidance on a late synthetic fork."""

import argparse
import json
import math
import shlex
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import Tensor

from diagnostics.masked_mlp_learned import run as runner
from diagnostics.masked_mlp_learned.continuation import continue_training
from diagnostics.shared.io import read_json, sha
from scjepa.experiments.storage import file_lock, write_json

from . import consensus_gradient, synthetic_consensus_runtime
from .consensus_gradient import context_intervals
from .single_acquisition import DATA, PARENT, PRIVATE, PYTHON
from .single_acquisition import _head as check_head  # pyright: ignore[reportPrivateUsage]
from .single_acquisition import _idle_gpu as idle_gpu  # pyright: ignore[reportPrivateUsage]
from .synthetic_loss_math import code_quality
from .vision_acquisition import declared

MODULE = "ICML_review.code.synthetic_consensus"
STEPS = 50_000
SPLITS = {"calibration": 20261123, "validation": 20261124}
MODES = ("same", "other")
PROTOCOL = (
    "Hypothesis: a fixed predictive teacher's other-episode consensus can remove episode noise "
    "from single-context codes during the final sparse continuation without losing forecast or "
    "recovery. Use the native full-arm parent, train-only teacher covariance and raw observations "
    "and actions. Fit two children on identical training streams; the sole difference is one "
    "own-episode teacher target vs the mean of the nine other teacher episodes. A full-rank "
    "frozen teacher gives distinct group targets; collapse is tested on held-out data. Bound "
    "auxiliary gradient norm on the parameter encoder by its native forecast gradient norm. "
    "The native objective, GECO update, global clipping, AdamW state and budgets are preserved. "
    "No extra inference inputs. Before training reproduce the first 500 native continuation "
    "updates' five archived log points at rtol 1e-4, atol 1e-8. A mismatch stops both children. "
    "Predeclare seed zero. Require CAL and VAL worst within/total generalized eigenvalue at "
    "most 0.80 times BOTH own-episode and native controls; full six-dimensional rank and sixth "
    "shared eigenvalue/total trace at least 0.80 times native on both. On validation require "
    "raw MSE <=1.10 times native, minimum single-episode parameter R2 >=max(0.80,native-0.02), "
    "state MCC >=native-0.01 and aligned total graph SHD <= native. Report original tau "
    "separately. "
    "If the pilot fails, do not expand. A passing pilot licenses matching seeds one through four. "
    "Labels enter only terminal TRAIN-independent evaluation. Preserve all outcomes and hashes."
)


def roots(seed: int) -> tuple[Path, Path, Path]:
    """Return the authenticated 300k parent, released 350k comparator and new workspace."""
    if seed not in range(5):
        raise ValueError("unknown seed")
    release = Path("outputs/diagnostics") / (
        "masked-mlp-gelu-s0-350k-20261102" if seed == 0 else "masked-mlp-gelu-350k-20261102"
    )
    return (
        PARENT / f"full-s{seed}/results",
        release / f"full-s{seed}/results",
        PRIVATE / f"synthetic-consensus-s{seed}",
    )


def source_identity() -> dict[str, str]:
    """Hash extension code separately from the immutable historical trainer fingerprint."""
    return {
        p.name: sha(p)
        for p in (
            Path(__file__),
            Path(consensus_gradient.__file__),
            Path(synthetic_consensus_runtime.__file__),
        )
    }


def control_inputs(parent: Path, native: Path) -> dict[str, Any]:
    """Verify each source completion and tie the historical continuation to its exact parent."""
    manifest = Path("outputs/reported-checkpoints.sha256")
    source = runner.verify_completion(parent, kind="train", require_current_source=True)
    ref = runner.verify_completion(native, kind="train", require_current_source=True)
    release = read_json(native / "continuation.json")
    digests = {
        str(Path("outputs") / key): digest
        for digest, key in (line.split(maxsplit=1) for line in manifest.read_text().splitlines())
    }
    for path, data in ((parent, source), (native, ref)):
        if data["hashes"]["last.pt"] != digests[str(path / "last.pt")]:
            raise ValueError("source continuation differs from the release manifest")
    if (
        release["parent_checkpoint_sha256"] != source["hashes"]["last.pt"]
        or release["parent_identity"] != source["identity"]
        or ref["data_identity"] != source["data_identity"]
        or ref["seed"] != source["seed"]
        or (source["steps"], ref["steps"]) != (300000, 350000)
    ):
        raise ValueError("native and historical parents are unmatched")
    return {
        "parent": str(parent),
        "native": str(native),
        "parent_sha256": source["hashes"]["last.pt"],
        "native_sha256": ref["hashes"]["last.pt"],
        "data_identity": source["data_identity"],
        "parent_completion_sha256": sha(parent / "completion.json"),
        "native_completion_sha256": sha(native / "completion.json"),
        "continuation_sha256": sha(native / "continuation.json"),
        "manifest_sha256": sha(manifest),
        "code_sha256": source_identity(),
    }


class ReplayDone(Exception):
    """The five predeclared native numerical checkpoints have been observed."""


class SilentReplay:
    """Local-only W&B placeholder for the short interrupted native numerical prefix."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Expose the empty logger state expected by the native loop."""
        self.metadata: dict[str, object] = {}

    def finish(self, *, exit_code: int = 0) -> None:
        """The review replay receipt carries success or mismatch."""


def replay(
    parent: Path, native: Path, out: Path, commit: str, inputs: dict[str, Any], command: str
) -> bool:
    """Compare the actual inherited native updates before enabling either controlled child."""
    with file_lock(out / ".replay.lock", blocking=True):
        path = out / "replay.json"
        if path.is_file():
            previous = read_json(path)
            if previous["commit"] != commit or previous["inputs"] != inputs:
                raise ValueError("replay identity differs")
            return bool(previous["matched"])
        dest = out / "native-prefix/results"
        replay_note = dest.parent / "NOTE.md"
        commands = [command.rsplit(" ", 1)[0] + " " + mode for mode in MODES]
        declared(
            replay_note,
            f"# Native continuation replay\n\n{PROTOCOL}\n\nCheckout {commit}\n"
            f"Data {DATA} identity {inputs['data_identity']}\n"
            f"Native parent {inputs['parent_sha256']}\n\n"
            + "\n".join(f"    {c}" for c in commands)
            + "\n",
        )
        if dest.exists():
            dest.rename(dest.with_name(f"incomplete-{time.time_ns()}"))
        expected: dict[int, dict[str, float]] = {}
        for line in (native / "history.jsonl").read_text().splitlines():
            row = json.loads(line)
            if 300000 < row["step"] <= 300500 and "loss/prediction" in row:
                expected[row["step"]] = row
        if sorted(expected) != list(range(300100, 300501, 100)):
            raise ValueError("missing native log block")
        actual: dict[str, dict[str, float]] = {}
        differences: list[str] = []

        def capture(_logger: object, _path: Path, step: int, values: dict[str, float]) -> None:
            old = expected[step]
            keys = set(old) - {"step", "train/elapsed_seconds"}
            if keys != set(values) - {"train/elapsed_seconds"}:
                raise ValueError("native numerical fields differ")
            actual[str(step)] = {k: values[k] for k in sorted(keys)}
            differences.extend(
                f"{step}/{k}"
                for k in sorted(keys)
                if not math.isclose(values[k], old[k], rel_tol=1e-4, abs_tol=1e-8)
            )
            if step == max(expected):
                raise ReplayDone

        check_head(commit)
        with (
            patch.object(runner, "_log", capture),
            patch.object(runner, "WandbLogger", SilentReplay),
            suppress(ReplayDone),
        ):
            continue_training(parent, dest, additional_steps=STEPS, device="cuda", threads=1)
        check_head(commit)
        if len(actual) != len(expected):
            raise RuntimeError("native replay incomplete")
        write_json(
            path,
            {
                "commit": commit,
                "inputs": inputs,
                "matched": not differences,
                "differences": differences,
                "measured": actual,
                "declaration_sha256": sha(replay_note),
                "invocation": command,
                "reference": {str(k): v for k, v in expected.items()},
            },
        )
        return not differences


def endpoint(path: Path, seed: int) -> tuple[dict[str, Any], dict[str, Tensor], dict[str, Tensor]]:
    """Read the saved, split-identified samples and original alignment-based evaluation."""
    from diagnostics.masked_mlp_oracle.paper_analysis import load_run

    code: dict[str, Tensor] = {}
    mse: dict[str, Tensor] = {}
    results: dict[str, Any] = {}
    for s in SPLITS:
        source = path / f"{s}.pt"
        z = cast(dict[str, Any], torch.load(source, map_location="cpu", weights_only=False))
        samples = z["samples"]
        code[s] = cast(Tensor, samples["codes"]).double()
        mse[s] = (
            (samples["predictions"].double() - samples["targets"].double())
            .square()
            .flatten(1)
            .mean(1)
        )
        raw = read_json(path / f"{s}.json")["metrics"]
        if not math.isclose(float(mse[s].mean()), raw["prediction"], rel_tol=1e-7, abs_tol=1e-11):
            raise ValueError("stored per-system and aggregate forecast disagree")
        results[s] = code_quality(code[s]) | {
            "prediction": raw["prediction"],
            "native_tau_ratio": raw["geco/relative_prediction"],
            "tau": raw["geco/tau_prediction"],
            "code_r2_min": raw["code_r2_min"],
            "state_mcc": raw["state_mcc"],
            "bootstrap_samples_sha256": sha(source),
        }
    metrics = load_run(path, "full", seed).metrics
    keys = [
        "masks/state_aligned_shd",
        "masks/action_aligned_shd",
        "masks/parameter_code_aligned_shd",
    ]
    results["validation"].update(
        graph={k: metrics[k] for k in keys}, graph_shd=sum(metrics[k] for k in keys)
    )
    return results, code, mse


def run(commit: str, seed: int, phase: str) -> None:
    """Declare, authenticate and run a single controlled stage with a pilot-first expansion."""
    if phase not in (*MODES, "assess"):
        raise ValueError("unknown stage")
    check_head(commit)
    parent, native, out = roots(seed)
    if seed:
        pilot = read_json(roots(0)[2] / "metrics.json")
        if (
            pilot["commit"] != commit
            or pilot["seed"] != 0
            or pilot["passed"] is not True
            or not all(pilot["checks"].values())
        ):
            raise ValueError("same-revision seed-zero pilot must qualify expansion")
    subprocess.run(
        ["sha256sum", "-c", "--quiet", "reported-checkpoints.sha256"], cwd="outputs", check=True
    )
    inputs = control_inputs(parent, native)
    out.mkdir(parents=True, exist_ok=True)
    command = "PYTHONPATH=$PWD:$PWD/src " + shlex.join(
        [PYTHON, "-m", MODULE, commit, str(seed), phase]
    )
    identity = f"Checkout {commit}\nData {DATA} identity {inputs['data_identity']}"
    note = (
        f"# Frozen synthetic teacher, seed {seed}, phase {phase}\n\n{PROTOCOL}\n\n{identity}\n"
        f"Parent SHA-256 {inputs['parent_sha256']}\nNative SHA-256 {inputs['native_sha256']}\n"
        f"\n    {command}\n"
    )
    with file_lock(out / f".phase-{phase}.lock"):
        directory = out / phase
        directory.mkdir(exist_ok=True)
        declared(directory / "NOTE.md", note)
        with file_lock(out / ".initial-note.lock", blocking=True):
            declared(
                out / "NOTE.md",
                f"# Frozen synthetic teacher, seed {seed}\n\n{PROTOCOL}\n\n{identity}\n"
                + "\n".join(
                    f"    PYTHONPATH=$PWD:$PWD/src {PYTHON} -m {MODULE} {commit} {seed} {p}"
                    for p in (*MODES, "assess")
                )
                + "\n",
            )
        if phase == "assess" and not (out / "replay.json").is_file():
            raise ValueError("assessment requires a previously attempted numerical replay")
        if phase in MODES and not (directory / "metrics.json").exists():
            idle_gpu()
        if not replay(parent, native, out, commit, inputs, command):
            if phase == "assess":
                write_json(
                    out / "metrics.json",
                    {
                        "verified": False,
                        "passed": False,
                        "reason": "native_replay",
                        "seed": seed,
                        "commit": commit,
                        "checks": {"native_replay": False},
                        "inputs": inputs,
                        "replay_sha256": sha(out / "replay.json"),
                    },
                )
            return
        if phase != "assess":
            train_phase(commit, seed, phase, parent, directory, inputs, command)
        else:
            assess(commit, seed, out, native, inputs, command)


def train_phase(
    commit: str,
    seed: int,
    mode: str,
    parent: Path,
    directory: Path,
    inputs: dict[str, Any],
    command: str,
) -> None:
    """Train or resume a fork, preserving its input hashes and full native output receipt."""
    path = directory / "results"
    receipt = directory / "metrics.json"
    if receipt.exists():
        old = read_json(receipt)
        complete = runner.verify_completion(path, require_current_source=True)
        if (
            old["checkpoint_sha256"] != complete["hashes"]["last.pt"]
            or old["inputs"] != inputs
            or old["commit"] != commit
            or old["command"] != command
        ):
            raise ValueError("finished child integrity differs")
        return
    if path.exists() and not all((path / n).is_file() for n in ("last.pt", "continuation.json")):
        path.rename(directory / f"incomplete-{time.time_ns()}")
    resume = (path / "last.pt").is_file()
    check_head(commit)
    result = synthetic_consensus_runtime.train_child(
        parent, path, mode, steps=STEPS, device="cuda", resume=resume
    )
    check_head(commit)
    end = read_json(path / "continuation-completion.json")
    if (
        end["parent_checkpoint_sha256"] != inputs["parent_sha256"]
        or end["checkpoint_sha256"] != result["hashes"]["last.pt"]
        or result["steps"] != 350000
    ):
        raise ValueError("numerical child fork changed")
    write_json(
        receipt,
        {
            "commit": commit,
            "seed": seed,
            "mode": mode,
            "command": command,
            "inputs": inputs,
            "checkpoint_sha256": result["hashes"]["last.pt"],
            "steps": result["steps"],
            "note_sha256": sha(directory / "NOTE.md"),
            "completion_sha256": sha(path / "completion.json"),
            "continuation_sha256": sha(path / "continuation-completion.json"),
        },
    )


def assess(
    commit: str, seed: int, out: Path, native: Path, inputs: dict[str, Any], command: str
) -> None:
    """Quantify within-system stability and physical recovery on held-out development systems."""
    paths = {"native": native} | {m: out / m / "results" for m in MODES}
    records: dict[str, Any] = {}
    codes: dict[str, dict[str, Tensor]] = {}
    errors: dict[str, dict[str, Tensor]] = {}
    hashes = {"native": inputs["native_sha256"]}
    receipts: dict[str, str] = {}
    for mode, path in paths.items():
        cp = runner.verify_completion(path, require_current_source=True)
        if mode in MODES:
            entry = out / mode / "metrics.json"
            r = read_json(entry)
            if (
                r["commit"] != commit
                or r["seed"] != seed
                or r["inputs"] != inputs
                or r["mode"] != mode
                or r["checkpoint_sha256"] != cp["hashes"]["last.pt"]
                or r["note_sha256"] != sha(entry.with_name("NOTE.md"))
            ):
                raise ValueError("comparison inputs differ")
            hashes[mode], receipts[mode] = cp["hashes"]["last.pt"], sha(entry)
        records[mode], codes[mode], errors[mode] = endpoint(path, seed)
    checks: dict[str, bool] = {}
    intervals = {}
    for split, bootstrap_seed in SPLITS.items():
        old, same, other = (records[m][split] for m in ("native", "same", "other"))
        for name, control in (("native", old), ("same", same)):
            checks[f"{split}_worst_vs_{name}"] = (
                other["worst_within_fraction"] is not None
                and control["worst_within_fraction"] is not None
                and other["worst_within_fraction"] <= 0.8 * control["worst_within_fraction"]
            )
        checks[f"{split}_rank"] = (
            other["total_numerical_rank"] == 6
            and other["sixth_share"] > 0
            and other["sixth_share"] >= 0.8 * old["sixth_share"]
        )
        try:
            intervals[split] = context_intervals(
                {m: codes[m][split] for m in paths},
                {m: errors[m][split] for m in paths},
                bootstrap_seed,
            )
        except ValueError as exc:
            checks[f"{split}_bootstrap_defined"] = False
            intervals[split] = {"undefined": str(exc)}
    old, other = records["native"]["validation"], records["other"]["validation"]
    checks.update(
        forecast=other["prediction"] <= 1.10 * old["prediction"],
        retention=other["code_r2_min"] >= max(0.80, old["code_r2_min"] - 0.02),
        state_mcc=other["state_mcc"] >= old["state_mcc"] - 0.01,
        total_shd=other["graph_shd"] <= old["graph_shd"],
    )
    check_head(commit)
    write_json(
        out / "metrics.json",
        {
            "commit": commit,
            "command": command,
            "seed": seed,
            "verified": True,
            "checks": checks,
            "passed": all(checks.values()),
            "inputs": inputs,
            "checkpoint_sha256": hashes,
            "stage_receipt_sha256": receipts,
            "replay_sha256": sha(out / "replay.json"),
            "assess_note_sha256": sha(out / "assess/NOTE.md"),
            "initial_declaration_sha256": sha(out / "NOTE.md"),
            "records": records,
            "paired_intervals": intervals,
            "bootstrap": {"draws": 2000, "unit": "whole systems", "seeds": SPLITS},
        },
    )


def main() -> None:
    """Run one pinned synthetic full-arm frozen-teacher stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("seed", type=int)
    parser.add_argument("phase", choices=(*MODES, "assess"))
    args = parser.parse_args()
    run(args.commit, args.seed, args.phase)


if __name__ == "__main__":
    main()
