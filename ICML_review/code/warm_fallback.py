"""Replay at most two synthetic acquisitions, with a TRAIN-only warm-to-cold fallback."""

import argparse
import json
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from diagnostics.masked_mlp_learned import run as trainer
from diagnostics.masked_mlp_learned.selection import select
from diagnostics.shared.io import read_json, sha
from scjepa.experiments.storage import file_lock

from . import single_acquisition as schedule
from .single_acquisition import _head as check_head  # pyright: ignore[reportPrivateUsage]
from .synthetic_combined import MANIFEST
from .synthetic_combined import _fixed as archive  # pyright: ignore[reportPrivateUsage]
from .synthetic_combined import _need as require  # pyright: ignore[reportPrivateUsage]
from .synthetic_consensus import roots
from .synthetic_warm import verify_transfer
from .vision_acquisition import checkpoint_hashes, declared

METHOD = (
    "Hypothesis: one cold acquisition can rescue a warm acquisition's TRAIN admission failure "
    "at a hard cap of two attempts. Reuse the original joint and attempt-zero acquisition seed, "
    "original 100000-update budget per attempt, terminal TRAIN-loss window and cutoff, projection, "
    "50000-step adaptation, fixed tau and both 350000-step sparse arms. Warm first; only a failed "
    "TRAIN admission permits one cold a0 attempt. Once admitted, always inherit that candidate's "
    "original outcome, including failed validation. Two admission failures end the procedure. "
    "Freeze route.json before inspecting labelled endpoint results or computing the comparisons. "
    "Use CAL/VAL for audit, never test. A passing paired pilot allows the four other fixed seeds. "
    "A failed endpoint or admission remains a failed policy result. This is a counterfactual "
    "reuse of pre-exposed candidate trajectories and the original pilot, not new training."
)


def choose(candidate: Callable[[str], dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Commit to the first TRAIN-admitted candidate without requesting another after success."""
    attempts: list[dict[str, Any]] = []
    for name in ("warm", "cold_a0"):
        row = candidate(name)
        attempts.append(row)
        if row["admitted"]:
            return name, attempts
    return None, attempts


def budget(attempts: int, admitted: bool) -> dict[str, int]:
    """Charge full from-scratch budgets for deployment and the paired study."""
    require(attempts in (1, 2) and (admitted or attempts == 2), "Invalid attempt/stop pattern")
    acquisition = 100_000 * attempts
    adapt, sparse = (50_000, 350_000) if admitted else (0, 0)
    total = 100_000 + acquisition + adapt + sparse
    return {
        "joint": 100_000,
        "acquisition": acquisition,
        "projection_fits": int(admitted),
        "adapt": adapt,
        "full_sparse": sparse,
        "baseline_sparse": sparse,
        "deployment_updates": total,
        "paired_study_updates": total + sparse,
        "new_optimizer_updates": 0,
    }


def _source(seed: int, name: str) -> tuple[Path, Path]:
    warm = name == "warm"
    root = schedule.PRIVATE / f"{'synthetic-warm' if warm else 'single-acquisition'}-s{seed}"
    return (
        root,
        root / "acquisition/results"
        if warm
        else schedule.PARENT / f"acquisition-0-s{seed}/results",
    )


def _candidate(
    seed: int, name: str, selector: dict[str, Any], joint: dict[str, Any]
) -> dict[str, Any]:
    _, path = _source(seed, name)
    record = trainer.verify_completion(path, kind="train", require_current_source=True)
    selected = select([path], window=selector["window"], maximum_loss=selector["maximum_loss"])
    attempt = selected["selected"]
    require(
        selected["common"] == selector["common"]
        and selected["selection_used_parameter_labels"] is False
        and record["steps"] == record["successful_updates"] == 100000
        and record["state_source"]["checkpoint_sha256"] == joint["hashes"]["last.pt"]
        and record["seed"] == seed
        and attempt["acquisition_seed"] == 1000 + seed
        and record["data_identity"] == joint["data_identity"],
        str(path),
    )
    if name == "warm":
        verify_transfer(path, schedule.PARENT / f"states-s{seed}/results")
    else:
        original = [r for r in selector["candidates"] if r["acquisition_seed"] == 1000 + seed]
        require(original == [attempt], str(path))
    return {
        "method": name,
        "path": str(path),
        "admitted": selected["admitted"],
        "train_prediction_mean": attempt["train_prediction_mean"],
        "selection_identity": selected["identity"],
        "acquisition_seed": attempt["acquisition_seed"],
        "checkpoint_sha256": record["hashes"]["last.pt"],
        "completion_sha256": sha(path / "completion.json"),
        "history_sha256": record["hashes"]["history.jsonl"],
    }


def _comparison(
    seed: int, attempt: dict[str, Any], tau: float, data_id: str, reference: dict[str, Any]
) -> dict[str, Any]:
    from diagnostics.masked_mlp_oracle.paper_analysis import load_run

    root, _ = _source(seed, attempt["method"])
    source = read_json(root / "metrics.json")
    admission = read_json(root / "admission.json")
    require(
        admission["admitted"] == attempt["admitted"]
        and admission["identity"] == attempt["selection_identity"]
        and admission["checkpoint_sha256"] == attempt["checkpoint_sha256"]
        and admission["data_identity"] == data_id
        and admission["commit"] == source["commit"]
        and source["seed"] == seed,
        str(root),
    )
    info: dict[str, Any] = {
        "source_run": str(root),
        "source_commit": source["commit"],
        "metrics_sha256": sha(root / "metrics.json"),
        "admission_sha256": sha(root / "admission.json"),
        "admitted": attempt["admitted"],
        "acquisition": attempt,
        "passed": source["passed"],
        "source_note_sha256": sha(root / "NOTE.md"),
    }
    if not attempt["admitted"]:
        require(not source["passed"] and source["reason"] == "train_admission", str(root))
        return info | {"reason": "train_admission"}
    require(
        source["tau"] == tau
        and source["data_identity"] == data_id
        and source["reference_validation"] == reference
        and source["reference_summary_sha256"] == sha(schedule.COMPARISON)
        and source["acquisition_receipt_sha256"] == info["admission_sha256"],
        str(root),
    )
    lineage: dict[str, dict[str, Any]] = {}
    completions: dict[str, str] = {}
    for name in schedule.STAGES:
        p = root / name / "results"
        rec = trainer.verify_completion(p, require_current_source=True)
        require(
            rec["hashes"]["last.pt"] == source["checkpoint_sha256"][name]
            and rec["data_identity"] == data_id
            and rec["seed"] == seed,
            str(p),
        )
        if name in schedule.TRAIN_STEPS:
            require(rec["steps"] == rec["successful_updates"] == schedule.TRAIN_STEPS[name], str(p))
        lineage[name] = rec
        completions[name] = sha(p / "completion.json")
    require(
        lineage["project"]["source"]["checkpoint_sha256"] == attempt["checkpoint_sha256"]
        and lineage["adapt"]["adaptation_source"]["identity"] == lineage["project"]["identity"],
        str(root),
    )
    for arm in ("baseline", "full"):
        start, end = f"{arm}-300k", f"{arm}-350k"
        p = root / end / "results"
        fork = read_json(p / "continuation-completion.json")
        require(
            lineage[start]["dense_endpoint"]["identity"] == lineage["adapt"]["identity"]
            and lineage[start]["tau_prediction"] == lineage[end]["tau_prediction"] == tau
            and fork["parent_checkpoint_sha256"] == lineage[start]["hashes"]["last.pt"]
            and fork["checkpoint_sha256"] == lineage[end]["hashes"]["last.pt"]
            and (fork["parent_step"], fork["final_step"]) == (300000, 350000)
            and fork["parent_completion_sha256"] == completions[start]
            and fork["completion_sha256"] == completions[end]
            and load_run(p, arm, seed).metrics == source["validation"][arm],
            str(p),
        )
    checks = schedule.decision(source["validation"], reference)
    require(checks == source["checks"] and all(checks.values()) == source["passed"], str(root))
    return info | {
        "checks": checks,
        "validation": source["validation"],
        "stage_completion_sha256": completions,
        "checkpoint_sha256": source["checkpoint_sha256"],
        "reason": None if source["passed"] else "endpoint",
    }


def _reported(seed: int, reference: dict[str, Any], data_id: str, tau: float) -> dict[str, Any]:
    from diagnostics.masked_mlp_oracle.paper_analysis import load_run

    registry = checkpoint_hashes()
    calibration_folder = schedule.PARENT / "calibration"
    bound = trainer.verify_completion(
        calibration_folder, kind="calibration", require_current_source=True
    )
    require(
        bound["data_identity"] == data_id and bound["tau_prediction"] == tau,
        str(calibration_folder),
    )
    folders = {
        "full": roots(seed)[1],
        "baseline": schedule.COMPARISON.parent.parent / f"baseline-s{seed}/results",
    }
    digests = {}
    for arm, path in folders.items():
        rec = trainer.verify_completion(path, kind="train", require_current_source=True)
        cp = path / "last.pt"
        original = load_run(path, arm, seed)
        values = original.metrics | {
            "start_state_mcc": None if original.start is None else original.start["state_mcc"]
        }
        require(
            rec["data_identity"] == data_id
            and rec["tau_prediction"] == tau
            and rec["calibration_identity"] == bound["identity"]
            and rec["steps"] == 350000
            and rec["hashes"]["last.pt"] == registry[str(cp)]
            and all(values[k] == v for k, v in reference[arm].items()),
            str(path),
        )
        digests[str(cp)] = registry[str(cp)]
    return {
        "validation": reference,
        "checkpoint_sha256": digests,
        "summary_sha256": sha(schedule.COMPARISON),
        "calibration_identity": bound["identity"],
        "calibration_completion_sha256": sha(calibration_folder / "completion.json"),
        "acquisition_attempts": 16,
        "deployment_updates": 2_100_000,
        "paired_study_updates": 2_450_000,
    }


def run(commit: str, seed: int) -> dict[str, Any]:
    """Freeze the TRAIN-only route, audit the controls and charge the selected policy."""
    require(seed in range(5), "Unknown seed")
    check_head(commit)
    out = schedule.PRIVATE / f"warm-fallback-s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    declaration = schedule.PRIVATE / "consensus-queue/warm-fallback-declaration.md"
    prior = {str(declaration): sha(declaration)} if declaration.is_file() else {}
    pilot_sha = None
    if seed:
        pilot = schedule.PRIVATE / "warm-fallback-s0/metrics.json"
        p = read_json(pilot)
        require(
            p["seed"] == 0
            and p["commit"] == commit
            and p["passed"]
            and all(p["checks"].values())
            and sha(pilot.with_name("route.json")) == p["route_sha256"],
            "A passing same-revision pilot is required",
        )
        pilot_sha = sha(pilot)
        require(
            p["prior_declaration_sha256"] == prior
            and p["checks"] == schedule.decision(p["validation"], p["reported"]["validation"]),
            "The pilot's declared policy or original endpoint checks differ",
        )
    command = "PYTHONPATH=$PWD:$PWD/src " + shlex.join(
        [schedule.PYTHON, "-m", "ICML_review.code.warm_fallback", commit, str(seed)]
    )
    data = read_json(schedule.DATA / "completion.json")["identity"]
    selector = schedule.PARENT / f"selection-s{seed}.json"
    bound = schedule.PARENT / "calibration/calibration.json"
    with file_lock(out / ".lock"):
        declared(
            out / "NOTE.md",
            f"# Warm then cold acquisition, seed {seed}\n\n{METHOD}\n\n{schedule.GATE}\n\n"
            f"Commit {commit}\nData {schedule.DATA} identity {data}\n"
            f"Original rule {selector} SHA256 {sha(selector)}\n"
            f"Bound {bound} SHA256 {sha(bound)}\n"
            f"Prior declaration {json.dumps(prior, sort_keys=True)}\n\n    {command}\n",
        )
        subprocess.run(
            ["sha256sum", "-c", "--quiet", MANIFEST.name], cwd=MANIFEST.parent, check=True
        )
        joint_dir = schedule.PARENT / f"states-s{seed}/results"
        joint = trainer.verify_completion(joint_dir, kind="train", require_current_source=True)
        rule = read_json(selector)
        require(
            joint["data_identity"] == data
            and joint["seed"] == seed
            and joint["steps"] == joint["successful_updates"] == 100000,
            str(joint_dir),
        )

        def fetch(name: str) -> dict[str, Any]:
            return _candidate(seed, name, rule, joint)

        chosen, attempts = choose(fetch)
        route = {
            "seed": seed,
            "commit": commit,
            "command": command,
            "data_identity": data,
            "pilot_sha256": pilot_sha,
            "note_sha256": sha(out / "NOTE.md"),
            "prior_declaration_sha256": prior,
            "manifest_sha256": sha(MANIFEST),
            "window": rule["window"],
            "maximum_loss": rule["maximum_loss"],
            "selector_sha256": sha(selector),
            "joint_checkpoint_sha256": joint["hashes"]["last.pt"],
            "tau_sha256": sha(bound),
            "attempts": attempts,
            "selected": chosen,
            "selection_used_parameter_labels": False,
        }
        check_head(commit)
        archive(out / "route.json", route)
        tau = read_json(bound)["tau_prediction"]
        summary = read_json(schedule.COMPARISON)["per_run"]
        ref = {k: summary[k][str(seed)] for k in ("baseline", "full")}
        # Any unattempted cold route is read solely as a disclosed scientific comparator.
        audit = {x["method"]: x for x in attempts}
        if "cold_a0" not in audit:
            audit["cold_a0"] = fetch("cold_a0")
        controls = {k: _comparison(seed, v, tau, data, ref) for k, v in audit.items()}
        source = controls[chosen] if chosen else {}
        result = route | {
            "verified": True,
            "route_sha256": sha(out / "route.json"),
            "tau": tau,
            "comparators": controls,
            "reported": _reported(seed, ref, data, tau),
            "budget": budget(len(attempts), chosen is not None),
            "passed": source.get("passed", False),
            "checks": source.get("checks", {"train_admission": False}),
            "reason": source.get("reason", "train_admission"),
            "validation": source.get("validation"),
            "checkpoint_sha256": source.get("checkpoint_sha256"),
        }
        check_head(commit)
        archive(out / "metrics.json", result)
        return result


def main() -> None:
    """Evaluate the fixed archival policy on a single seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("seed", type=int, choices=range(5))
    args = parser.parse_args()
    r = run(args.commit, args.seed)
    print(json.dumps({k: r[k] for k in ("seed", "selected", "passed", "reason", "budget")}))


if __name__ == "__main__":
    main()
