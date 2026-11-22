"""Condition the sibling-teacher continuation on prospectively replicated synthetic constituents."""

import argparse
import json
import math
import shlex
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from diagnostics.masked_mlp_learned import run as trainer
from diagnostics.masked_mlp_learned.continuation import continue_training
from diagnostics.shared.io import read_json, sha
from scjepa.experiments.storage import file_lock, write_json

from . import single_acquisition as schedule
from . import synthetic_consensus as consensus
from . import synthetic_warm as warm
from .consensus_gradient import context_intervals
from .vision_acquisition import checkpoint_hashes, declared

MODULE = "ICML_review.code.synthetic_combined"
FAMILIES = (warm.POLICY, "synthetic-consensus")
MANIFEST = Path("outputs/reported-checkpoints.sha256")
METHOD = (
    "Hypothesis: the fixed other-episode teacher improves stability after one warm acquisition "
    "while preserving the registered recovery. Require both constituent seed-zero pilots and "
    "at least four passing original seeds per constituent among all five terminal decisions. "
    "Replace only the final 50000 full-arm steps after the same warm 300000-step parent. "
    "Use its original joint, single warm acquisition, projection, adaptation, warm baseline, "
    "tolerance, GELU model, training streams, optimizer, GECO, EMA and independent contexts. "
    "Replay five archived native warm log points through the first 500 continuation updates "
    "on the warm native worker GPU model, rtol=1e-4 and atol=1e-8. Use only TRAIN observations, "
    "actions and same-system grouping for the frozen sibling teacher. Labels enter only "
    "held-out evaluation. Require CAL and VAL worst-direction W/T at most 0.80 times native "
    "warm, resolved rank six, sixth shared eigenvalue/total trace at least 0.80 times native "
    "warm, and raw forecast MSE at most 1.10 times native warm on both. Require validation "
    "minimum parameter R2 >= max(0.80,native warm minus 0.02), plus every original warm "
    "endpoint gate against the reported configuration (including its paired baseline). "
    "Only a passing combined seed zero licenses the remaining four. Include failed constituent "
    "seeds if they have warm parents. Record unavailable warm parents separately. A pass supports "
    "combining the changes at this budget; a failed gate leaves that joint claim unestablished."
)


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _command(commit: str, seed: int, phase: str) -> str:
    return "PYTHONPATH=$PWD:$PWD/src " + shlex.join(
        [schedule.PYTHON, "-m", MODULE, commit, str(seed), phase]
    )


def _note(path: Path, commit: str, command: str, identity: str, extra: str = "") -> None:
    declared(
        path,
        f"# Warm acquisition and frozen sibling teacher\n\n{METHOD}\n\n"
        f"Commit {commit}\nData {schedule.DATA} identity {identity}\n{extra}\n\n    {command}\n",
    )


def _bound(path: Path, digest: str) -> dict[str, Any]:
    _need(sha(path) == digest, f"Digest changed at {path}")
    return read_json(path)


def _fixed(path: Path, row: dict[str, Any]) -> None:
    if path.exists():
        _need(read_json(path) == row, f"Existing receipt differs at {path}")
    else:
        write_json(path, row)


def _constituent(family: str, folder: Path, seed: int, data_id: str) -> dict[str, Any]:
    path = folder / "metrics.json"
    r = read_json(path)
    _need(r["seed"] == seed and type(r["passed"]) is bool, str(path))
    if family == warm.POLICY:
        _need(r["policy"] == family and r["data_identity"] == data_id, str(path))
        a = _bound(folder / "admission.json", r["acquisition_receipt_sha256"])
        prep = _bound(folder / "prepare.json", r["prepare_sha256"])
        _need(
            a["commit"] == prep["commit"] == r["commit"]
            and a["data_identity"] == data_id
            and prep["admission_sha256"] == r["acquisition_receipt_sha256"]
            and prep["ready"] == a["admitted"]
            and prep["seed"] == seed,
            str(folder),
        )
        if a["admitted"]:
            ref = read_json(schedule.COMPARISON)["per_run"]
            ref = {k: ref[k][str(seed)] for k in ("baseline", "full")}
            _need(
                r["reference_summary_sha256"] == sha(schedule.COMPARISON)
                and r["reference_validation"] == ref,
                str(path),
            )
            gates = schedule.decision(r["validation"], ref)
            _need(r["checks"] == gates and r["passed"] == all(gates.values()), str(path))
        else:
            _need(r.get("reason") == "train_admission" and not r["passed"], str(path))
    else:
        replay = _bound(folder / "replay.json", r["replay_sha256"])
        _need(
            r["inputs"]["data_identity"] == data_id
            and replay["inputs"] == r["inputs"]
            and replay["commit"] == r["commit"],
            str(path),
        )
        if r["verified"]:
            _need(replay["matched"] and r["passed"] == all(r["checks"].values()), str(path))
            _need(r["initial_declaration_sha256"] == sha(folder / "NOTE.md"), str(path))
            expected = {"forecast", "retention", "state_mcc", "total_shd"}
            expected.update(
                f"{split}_{g}"
                for split in consensus.SPLITS
                for g in ("rank", "worst_vs_native", "worst_vs_same")
            )
            _need(expected <= r["checks"].keys(), str(path))
            for arm in consensus.MODES:
                stage = _bound(folder / arm / "metrics.json", r["stage_receipt_sha256"][arm])
                _need(
                    stage["checkpoint_sha256"] == r["checkpoint_sha256"][arm]
                    and stage["inputs"] == r["inputs"]
                    and stage["mode"] == arm
                    and stage["commit"] == r["commit"]
                    and stage["seed"] == seed,
                    str(path),
                )
        else:
            _need(
                not replay["matched"] and not r["passed"] and r.get("reason") == "native_replay",
                str(path),
            )
    return {
        "seed": seed,
        "passed": r["passed"],
        "reason": r.get("reason"),
        "commit": r["commit"],
        "path": str(path),
        "sha256": sha(path),
        "note_sha256": sha(folder / "NOTE.md"),
    }


def eligibility(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Distinguish pending evidence, two independently qualified families, and terminal failure."""
    counts: dict[str, int] = {}
    pilots: dict[str, bool | None] = {}
    for family in FAMILIES:
        by_seed = {r["seed"]: r for r in rows[family]}
        _need(len(by_seed) == len(rows[family]) and by_seed.keys() <= set(range(5)), family)
        _need(
            len({r["commit"] for r in rows[family]}) <= 1, f"Mixed constituent revisions: {family}"
        )
        counts[family] = sum(r["passed"] for r in rows[family])
        pilots[family] = by_seed.get(0, {}).get("passed")
    complete = all(len(rows[f]) == 5 for f in FAMILIES)
    passed = complete and all(pilots[f] and counts[f] >= 4 for f in FAMILIES)
    return {
        "status": ("eligible" if passed else "ineligible") if complete else "pending",
        "complete": complete,
        "passed": bool(passed),
        "passing_counts": counts,
        "pilots": pilots,
    }


def _cohort(commit: str, identity: str) -> dict[str, Any]:
    root = schedule.PRIVATE / "synthetic-combined-cohort"
    root.mkdir(parents=True, exist_ok=True)
    command = _command(commit, 0, "cohort")
    with file_lock(root / ".lock", blocking=True):
        _note(root / "NOTE.md", commit, command, identity)
        rows = {
            family: [
                _constituent(family, p, seed, identity)
                for seed in range(5)
                if ((p := schedule.PRIVATE / f"{family}-s{seed}") / "metrics.json").is_file()
            ]
            for family in FAMILIES
        }
        result = eligibility(rows) | {
            "commit": commit,
            "command": command,
            "constituents": rows,
            "data_identity": identity,
            "note_sha256": sha(root / "NOTE.md"),
        }
        consensus.check_head(commit)
        if result["complete"]:
            _fixed(root / "metrics.json", result)
        else:
            _need(
                not (root / "metrics.json").exists(),
                "Previously complete constituent receipts disappeared",
            )
            # Immutable distinct snapshots keep the admission history if this command is polled.
            found = "-".join("".join(str(x["seed"]) for x in rows[f]) or "none" for f in FAMILIES)
            _fixed(root / f"pending-{found}.json", result)
        return result


def _warm_inputs(root: Path, seed: int, row: dict[str, Any], data_id: str) -> dict[str, Any]:
    from diagnostics.masked_mlp_oracle.paper_analysis import load_run

    r = _bound(root / "metrics.json", row["sha256"])
    a = _bound(root / "admission.json", r["acquisition_receipt_sha256"])
    prep = _bound(root / "prepare.json", r["prepare_sha256"])
    tau = read_json(schedule.PARENT / "calibration/calibration.json")["tau_prediction"]
    _need(
        a["admitted"] and prep["ready"] and r["tau"] == tau and r["data_identity"] == data_id,
        str(root),
    )
    stages: dict[str, dict[str, Any]] = {}
    for name in ("acquisition", *schedule.STAGES):
        p = root / name / "results"
        rec = trainer.verify_completion(p, require_current_source=True)
        _need(
            rec["data_identity"] == data_id
            and rec["seed"] == seed
            and rec["hashes"]["last.pt"] == r["checkpoint_sha256"][name],
            str(p),
        )
        if name in schedule.TRAIN_STEPS:
            _need(rec["steps"] == rec["successful_updates"] == schedule.TRAIN_STEPS[name], str(p))
        stages[name] = rec
    joint = schedule.PARENT / f"states-s{seed}/results"
    _need(
        a["source_sha256"] == sha(joint / "last.pt")
        and a["acquisition_completion_sha256"] == sha(root / "acquisition/results/completion.json")
        and a["checkpoint_sha256"] == stages["acquisition"]["hashes"]["last.pt"]
        and a["source_completion_sha256"] == sha(joint / "completion.json"),
        str(root),
    )
    warm.verify_transfer(root / "acquisition/results", joint)
    _need(
        stages["project"]["source"]["checkpoint_sha256"] == a["checkpoint_sha256"]
        and stages["adapt"]["adaptation_source"]["identity"] == stages["project"]["identity"],
        "Projection/adaptation lineage differs",
    )
    for arm in ("full", "baseline"):
        source, end = f"{arm}-300k", f"{arm}-350k"
        p = root / end / "results"
        release = read_json(p / "continuation-completion.json")
        _need(
            stages[source]["condition"] == stages[end]["condition"] == arm
            and stages[source]["tau_prediction"] == stages[end]["tau_prediction"] == tau
            and stages[source]["dense_endpoint"]["identity"] == stages["adapt"]["identity"]
            and release["parent_checkpoint_sha256"] == stages[source]["hashes"]["last.pt"]
            and release["checkpoint_sha256"] == stages[end]["hashes"]["last.pt"]
            and release["parent_completion_sha256"]
            == sha(root / source / "results/completion.json")
            and release["completion_sha256"] == sha(p / "completion.json")
            and load_run(p, arm, seed).metrics == r["validation"][arm],
            str(p),
        )
    parent, native = root / "full-300k/results", root / "full-350k/results"
    native_cfg = read_json(native / "config.json")
    _need(
        a["selection_used_parameter_labels"] is False
        and a["initial_tensors_match"] is True
        and a["frozen_tensors_match"] is True
        and stages["acquisition"]["steps"] == stages["acquisition"]["successful_updates"] == 100000
        and stages["acquisition"]["state_source"]["checkpoint_sha256"] == sha(joint / "last.pt")
        and a["selection_sha256"] == sha(schedule.PARENT / f"selection-s{seed}.json")
        and prep["project_sha256"] == stages["project"]["hashes"]["last.pt"]
        and prep["adapt_sha256"] == stages["adapt"]["hashes"]["last.pt"]
        and native_cfg["physical_state_training"] is False
        and native_cfg["true_parameter_training"] is False
        and Path(native_cfg["config"]["data_dir"]).resolve() == schedule.DATA.resolve()
        and native_cfg["config"]["predictor_activation"] == "gelu",
        "Warm TRAIN-only design or published data lineage differs",
    )
    gpu = native_cfg["runtime"]["gpu_name"]
    manifest = checkpoint_hashes()
    reported_paths = (
        consensus.roots(seed)[1] / "last.pt",
        schedule.COMPARISON.parent.parent / f"baseline-s{seed}/results/last.pt",
    )
    return {
        "parent": str(parent),
        "native": str(native),
        "gpu_model": gpu,
        "tau": tau,
        "parent_sha256": stages["full-300k"]["hashes"]["last.pt"],
        "native_sha256": stages["full-350k"]["hashes"]["last.pt"],
        "data_identity": data_id,
        "baseline_sha256": stages["baseline-350k"]["hashes"]["last.pt"],
        "warm_metrics_sha256": row["sha256"],
        "warm_stage_sha256": r["checkpoint_sha256"],
        "warm_stage_completion_sha256": {
            k: sha(root / k / "results/completion.json") for k in stages
        },
        "native_continuation_sha256": sha(native / "continuation-completion.json"),
        "manifest_sha256": sha(MANIFEST),
        "reported_validation_sha256": sha(schedule.COMPARISON),
        "reported_checkpoints": {str(p): manifest[str(p)] for p in reported_paths},
        "code_sha256": consensus.source_identity() | {Path(__file__).name: sha(Path(__file__))},
    }


def numerical_replay(
    parent: Path, native: Path, dest: Path, *, steps: int, budget: int, device: str
) -> dict[str, Any]:
    """Replay the matched warm native prefix and compare every logged numerical field."""
    start = read_json(parent / "completion.json")["steps"]
    stride = read_json(native / "config.json")["config"]["log_every"]
    expected: dict[int, dict[str, float]] = {}
    for line in (native / "history.jsonl").read_text().splitlines():
        row = json.loads(line)
        if start < row["step"] <= start + steps and "loss/prediction" in row:
            expected[row["step"]] = row
    _need(
        list(expected) == list(range(start + stride, start + steps + 1, stride)) and bool(expected),
        "Missing warm native replay logs",
    )
    measured: dict[str, dict[str, float]] = {}
    differences: list[str] = []

    def capture(_logger: object, _path: Path, step: int, values: dict[str, float]) -> None:
        ref = expected[step]
        keys = set(ref) - {"step", "train/elapsed_seconds"}
        _need(keys == set(values) - {"train/elapsed_seconds"}, "Replay recorder fields changed")
        measured[str(step)] = {k: values[k] for k in sorted(keys)}
        differences.extend(
            f"{step}/{k}"
            for k in sorted(keys)
            if not math.isclose(values[k], ref[k], rel_tol=1e-4, abs_tol=1e-8)
        )
        if step == start + steps:
            raise consensus.ReplayDone

    if dest.exists():
        dest.rename(dest.with_name(f"incomplete-{time.time_ns()}"))
    with (
        patch.object(trainer, "_log", capture),
        patch.object(trainer, "WandbLogger", consensus.SilentReplay),
        suppress(consensus.ReplayDone),
    ):
        continue_training(parent, dest, additional_steps=budget, device=device, threads=1)
    _need(len(measured) == len(expected), "Warm replay never reached its comparison endpoint")
    return {
        "matched": not differences,
        "differences": differences,
        "measured": measured,
        "reference": {str(k): v for k, v in expected.items()},
    }


def combined_checks(
    records: dict[str, Any], measured: dict[str, Any], reference: dict[str, Any]
) -> dict[str, bool]:
    """Apply the preregistered invariance, retention and original paired warm recovery gates."""
    checks = schedule.decision(measured, reference)
    for split in consensus.SPLITS:
        old, new = (records[k][split] for k in ("native", "other"))
        checks[f"{split}_worst"] = (
            old["worst_within_fraction"] is not None
            and new["worst_within_fraction"] is not None
            and new["worst_within_fraction"] <= 0.80 * old["worst_within_fraction"]
        )
        checks[f"{split}_rank"] = (
            new["total_numerical_rank"] == 6
            and new["sixth_share"] > 0
            and new["sixth_share"] >= 0.80 * old["sixth_share"]
        )
        checks[f"{split}_forecast"] = new["prediction"] <= 1.10 * old["prediction"]
    checks["warm_parameter_retention"] = records["other"]["validation"]["code_r2_min"] >= max(
        0.80, records["native"]["validation"]["code_r2_min"] - 0.02
    )
    return checks


def _assess(out: Path, source: Path, seed: int, inputs: dict[str, Any]) -> dict[str, Any]:
    from diagnostics.masked_mlp_oracle.paper_analysis import load_run

    r = read_json(source / "metrics.json")
    child = out / "other/results"
    endpoint = _bound(
        child / "completion.json", read_json(out / "other/metrics.json")["completion_sha256"]
    )
    _need(
        endpoint["data_identity"] == inputs["data_identity"]
        and endpoint["tau_prediction"] == inputs["tau"]
        and endpoint["hashes"]["last.pt"] == sha(child / "last.pt"),
        str(child),
    )
    records: dict[str, Any] = {}
    codes: dict[str, dict[str, torch.Tensor]] = {}
    errors: dict[str, dict[str, torch.Tensor]] = {}
    for arm, path in (("native", Path(inputs["native"])), ("other", child)):
        records[arm], codes[arm], errors[arm] = consensus.endpoint(path, seed)
    measured = {
        "full": load_run(child, "full", seed).metrics,
        "baseline": r["validation"]["baseline"],
    }
    checks = combined_checks(records, measured, r["reference_validation"])
    intervals = {}
    for split, bs in consensus.SPLITS.items():
        # The existing three-arm bootstrap is paired. Alias its unused second comparator
        # to the native warm array and emit only the actual two-arm contrast.
        try:
            intervals[split] = context_intervals(
                {
                    k: codes["native" if k == "same" else k][split]
                    for k in ("native", "other", "same")
                },
                {
                    k: errors["native" if k == "same" else k][split]
                    for k in ("native", "other", "same")
                },
                bs,
            )["other_minus_native"]
        except ValueError as exc:
            intervals[split] = {"undefined": str(exc)}
    return {
        "verified": True,
        "passed": all(checks.values()),
        "checks": checks,
        "records": records,
        "validation": measured,
        "reference_validation": r["reference_validation"],
        "bootstrap": {"unit": "whole systems", "draws": 2000, "seeds": consensus.SPLITS},
        "paired_intervals": intervals,
        "checkpoint_sha256": {
            "native": inputs["native_sha256"],
            "warm_baseline": inputs["baseline_sha256"],
            "other": endpoint["hashes"]["last.pt"],
        },
    }


def run(commit: str, seed: int, phase: str) -> dict[str, Any]:
    """Archive cohort eligibility or run a single licensed warm sibling continuation."""
    consensus.check_head(commit)
    identity = read_json(schedule.DATA / "completion.json")["identity"]
    if phase == "cohort" and seed == 0:
        return _cohort(commit, identity)
    _need(phase == "run" and seed in range(5), "Expected seed zero cohort, or a valid seed and run")
    ledger = schedule.PRIVATE / "synthetic-combined-cohort/metrics.json"
    _need(ledger.is_file(), "Await a complete, independently declared constituent cohort decision")
    cohort = read_json(ledger)
    _need(
        cohort == _cohort(commit, identity) and cohort["status"] == "eligible",
        "Both registered constituent families must qualify",
    )
    pilot_digest = None
    if seed:
        pilot_file = schedule.PRIVATE / "synthetic-combined-s0/metrics.json"
        pilot = read_json(pilot_file)
        _need(
            pilot["commit"] == commit
            and pilot["cohort_sha256"] == sha(ledger)
            and pilot["seed"] == 0
            and pilot["passed"]
            and all(pilot["checks"].values()),
            "A complete passing combined seed-zero pilot is required",
        )
        pilot_digest = sha(pilot_file)
    out = schedule.PRIVATE / f"synthetic-combined-s{seed}"
    source = schedule.PRIVATE / f"{warm.POLICY}-s{seed}"
    row = cohort["constituents"][warm.POLICY][seed]
    _need(row["seed"] == seed, "Unordered constituent seed receipt")
    warm_receipt = _bound(source / "metrics.json", row["sha256"])
    command = _command(commit, seed, phase)
    out.mkdir(parents=True, exist_ok=True)
    with file_lock(out / ".run.lock"):
        common = {
            "seed": seed,
            "commit": commit,
            "command": command,
            "cohort_sha256": sha(ledger),
            "pilot_sha256": pilot_digest,
            "data_identity": identity,
            "warm_metrics_sha256": row["sha256"],
        }
        _note(
            out / "NOTE.md",
            commit,
            command,
            identity,
            f"Cohort SHA-256 {sha(ledger)}\nWarm receipt {row['sha256']}",
        )
        common["note_sha256"] = sha(out / "NOTE.md")
        terminal = out / "metrics.json"
        if warm_receipt.get("reason") == "train_admission":
            result = common | {
                "passed": False,
                "verified": True,
                "reason": "unavailable_warm_parent",
                "checks": {"warm_parent_available": False},
            }
            consensus.check_head(commit)
            _fixed(terminal, result)
            return result
        subprocess.run(
            ["sha256sum", "-c", "--quiet", MANIFEST.name], cwd=MANIFEST.parent, check=True
        )
        inputs = _warm_inputs(source, seed, row, identity)
        common["inputs"] = inputs
        phases = ("native-prefix", "other", "assess")
        for label in phases:
            _note(
                out / label / "NOTE.md",
                commit,
                command,
                identity,
                f"Stage {label}\nInputs {json.dumps(inputs, sort_keys=True)}",
            )
        parent, native = Path(inputs["parent"]), Path(inputs["native"])
        replay_path = out / "replay.json"
        if not replay_path.is_file() or (
            not (out / "other/metrics.json").is_file() and not terminal.is_file()
        ):
            consensus.idle_gpu()
            _need(
                torch.cuda.get_device_name(0) == inputs["gpu_model"],
                "Requires the warm 350k native GPU model",
            )
        prefix = {
            "commit": commit,
            "inputs": inputs,
            "command": command,
            "note_sha256": sha(out / "native-prefix/NOTE.md"),
        }
        if not replay_path.is_file():
            consensus.check_head(commit)
            report = numerical_replay(
                parent,
                native,
                out / "native-prefix/results",
                steps=500,
                budget=consensus.STEPS,
                device="cuda",
            )
            consensus.check_head(commit)
            _fixed(replay_path, prefix | report)
        replay = read_json(replay_path)
        _need(all(replay[k] == v for k, v in prefix.items()), str(replay_path))
        common["replay_sha256"] = sha(replay_path)
        if not replay["matched"]:
            result = common | {
                "verified": False,
                "passed": False,
                "reason": "native_replay",
                "checks": {"native_replay": False},
            }
        else:
            consensus.check_head(commit)
            consensus.train_phase(commit, seed, "other", parent, out / "other", inputs, command)
            consensus.check_head(commit)
            arm = out / "other/metrics.json"
            kid = read_json(arm)
            _need(
                kid["seed"] == seed
                and kid["mode"] == "other"
                and kid["note_sha256"] == sha(out / "other/NOTE.md"),
                str(arm),
            )
            common["stage_receipt_sha256"] = sha(arm)
            common["assess_note_sha256"] = sha(out / "assess/NOTE.md")
            if terminal.is_file():
                previous = read_json(terminal)
                _need(
                    all(previous[k] == v for k, v in common.items())
                    and previous["passed"] == all(previous["checks"].values())
                    and previous["checkpoint_sha256"]["other"] == kid["checkpoint_sha256"],
                    str(terminal),
                )
                return previous
            result = common | _assess(out, source, seed, inputs)
        consensus.check_head(commit)
        _fixed(terminal, result)
        return result


def main() -> None:
    """Run the cohort decision on CPU or one gated combined seed on its warm native GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("seed", type=int, choices=range(5))
    parser.add_argument("phase", choices=("cohort", "run"))
    args = parser.parse_args()
    result = run(args.commit, args.seed, args.phase)
    print(json.dumps({k: result.get(k) for k in ("status", "seed", "passed", "reason", "checks")}))


if __name__ == "__main__":
    main()
