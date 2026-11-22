"""Test the frozen sibling-teacher continuation on the prospectively fixed four-head RGB policy."""

import argparse
import hashlib
import math
import shlex
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from diagnostics.vision_dimer import train as release
from scjepa.experiments.storage import file_lock, write_json

from . import chart, kicks, rgb_four
from .changed_queries import Inputs, controls, paired, require
from .consensus_gradient import context_intervals
from .single_acquisition import _head, _idle_gpu  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import (
    DATA,
    MANIFEST,
    OUTPUT,
    PYTHON,
    ROOT,
    checkpoint_hashes,
    declared,
    read,
    sha,
    sparse_checks,
)
from .vision_invariant import (
    _arm,  # pyright: ignore[reportPrivateUsage]
    _prepare,  # pyright: ignore[reportPrivateUsage]
    verify_files,
)
from .vision_invariant_eval import SPLITS, decision
from .vision_loss import _runtime  # pyright: ignore[reportPrivateUsage]

MODULE = "ICML_review.code.vision_combined"
FAMILIES = ("rgb-four", "vision-invariant")
MARGINS = {"chart": 0.01, "kicks": 0.05}
PROTOCOL = """Hypothesis: the same model can retain the four-acquisition scientific
endpoint and named intervention accuracy while reducing worst-direction episode disagreement.
Before any training require both scientific pilots to pass, all five original-seed decisions
and physical query dispositions per method, at least four passing scientific gates per method
and at least four of the same five seeds satisfying BOTH scalar intervention margins within
each method. Count the four-head full model and the original sibling model. Unavailable
checkpoints contribute no successes. Keep the selected four-head trajectory fixed even on
individual nonpassing seeds. Verify its TRAIN-only winner, native 300k resume state, native
500k endpoint, original tolerance, data, and paired baseline. If the four-head decision
retains the original winner and exactly the same 300k resume state and 500k native control,
inherit the original sibling trajectory and query measurements. Record that deterministic
replay separately. Otherwise replace only its final 200000 native sparse updates with the
same TRAIN-only frozen, whitened, other-nine-episode target and forecast-gradient cap after
a numerical native replay at the same GPU model. Hold budgets, state updates, optimizer,
EMA, gates, RNG/sampler streams and GECO fixed. Physical labels enter held-out evaluation.
On CAL and VAL require full four-dimensional numerical rank, reproduced hard first-window
MSE, at most 0.80x native four-head worst within/total share and at most 1.05x its raw MSE.
Retain every original four-head scientific endpoint criterion against reported full/baseline,
using the actual four-head baseline for the changed gain. For the scalar interface require
Q1 partner-error increase <=0.01 and Q2 held-kick relative-error increase <=0.05 versus
the archived reported full model. Report both interfaces' paired differences also versus
the native four-head endpoint. Use the pilot's combined scientific and physical decision
to admit remaining seeds. A technical replay or measurement mismatch cannot certify an
improvement. Report missing parents, failed criteria, and inheritance separately."""


def _command(rev: str, seed: int, phase: str) -> str:
    return "PYTHONPATH=$PWD:$PWD/src " + shlex.join([PYTHON, "-m", MODULE, rev, str(seed), phase])


def _fixed(path: Path, record: dict[str, Any]) -> None:
    if path.exists():
        require(read(path) == record, f"Existing receipt changed at {path}")
    else:
        write_json(path, record)


def _record(files: Inputs, path: Path, digest: str) -> dict[str, Any]:
    files.bind(path, digest)
    return read(path)


def _split_ids() -> dict[str, str]:
    splits = read(DATA / "manifest.json")["splits"]
    auth = checkpoint_hashes()
    result = {s: splits[s]["sha256"] for s in ("train", *SPLITS)}
    require(all(v == auth[str(DATA / f"{s}.pt")] for s, v in result.items()), "Wrong RGB cohort")
    return result


def _verified_queries(
    records: dict[str, Any], cp: Path, digest: str, refs: dict[str, Any]
) -> dict[str, bool]:
    """Recreate the physical scoring from the bound cohort and independent simulator targets."""
    checks: dict[str, bool] = {}
    forecasts: list[float] = []
    for name, margin in MARGINS.items():
        item = records[name]
        result = item["measurement"]
        full = result["arms"]["full"]
        assignment = (
            "spearman_assignment_matches_stored" if name == "chart" else "assignment_matches_stored"
        )
        require(
            Path(full["checkpoint"]).resolve() == cp.resolve()
            and full["checkpoint_sha256"] == digest
            and full[assignment] is True,
            f"Wrong query model or calibration assignment for {name}",
        )
        forecasts.append(full["recomputed_first_window_nmse"])
        deltas = paired(name, result, refs[name], "full")
        checks[name] = deltas["scalar"]["trial_minus_control"] <= margin
        require(
            deltas == item["paired"] and checks[name] == item["scalar_within_margin"],
            f"Physical margin or paired measurement differs for {name}",
        )
    require(math.isclose(*forecasts, rel_tol=1e-6), "Held and kicked unperturbed forecasts differ")
    return checks


def _constituent(family: str, seed: int, splits: dict[str, str]) -> dict[str, Any]:
    root = OUTPUT / f"{family}-s{seed}"
    qp = OUTPUT / f"consensus-queries-{family}-s{seed}"
    files = Inputs()
    row = files.record(root / "metrics.json")
    q = files.record(qp / "metrics.json")
    rev = row["commit"]
    require(
        row["seed"] == q["seed"] == seed
        and q["commit"] == rev
        and q["method"] == family
        and type(row["passed"]) is bool
        and q["training_passed"] == row["passed"]
        and q["input_sha256"][str(root / "metrics.json")] == files[str(root / "metrics.json")],
        f"Wrong constituent or query receipt at {root}",
    )
    files.bind(root / "NOTE.md", row["note_sha256"])
    files.bind(qp / "NOTE.md", q["note_sha256"])
    verify_files(q["input_sha256"])
    # The causal interface and the deployed loss must have the same source in both revisions.
    source_names = ["chart.py", "kicks.py", "changed_queries.py"]
    if family == "vision-invariant":
        source_names += [
            "consensus_gradient.py",
            "vision_invariant_train.py",
            "vision_invariant.py",
        ]
    for name in source_names:
        source = Path(__file__).with_name(name).resolve().relative_to(Path.cwd())
        contents = subprocess.check_output(["git", "show", f"{rev}:{source.as_posix()}"])
        files.bind(source, hashlib.sha256(contents).hexdigest())
    alias = "full" if family == "rgb-four" else "sibling"
    if family == "rgb-four":
        rgb_four._verify_row(row, rev, seed, root)  # pyright: ignore[reportPrivateUsage]
        has_weights = bool(row["terminal_paths"])
        reason = row["reason"]
    else:
        prep = _record(files, root / "prepare.json", row["prepare_sha256"])
        require(
            row["splits"] == prep["splits"] == splits
            and prep["source_sha256"] == row["source_sha256"]
            and prep["commit"] == rev
            and prep["seed"] == seed
            and prep["ready"] == all(prep["checks"].values())
            and type(row.get("checks")) is dict
            and row["passed"] == all(row["checks"].values()),
            "Wrong sibling validation or replay",
        )
        verify_files(row["files"])
        verify_files(prep["files"])
        verify_files(row["inputs"])
        has_weights = prep["ready"]
        reason = "sparse_endpoint" if has_weights else "preparation"
        if has_weights:
            require(
                all(k in row["checks"] for k in ("graph", "state_mcc", "drag_retention")),
                "Incomplete original sibling gates",
            )
            existing = {
                arm: read(root / directory)
                for arm, directory in (
                    ("native", "native.json"),
                    ("same", "own/endpoint.json"),
                    ("other", "sibling/endpoint.json"),
                )
            }
            replay = decision(existing)
            require(
                row["checks"] == replay["checks"] and row["passed"] == replay["passed"],
                "Stored primary sibling endpoint gates differ",
            )
            cp = root / "sibling/final.pt"
            digest = row["checkpoint_sha256"]["other"]
            arm = _record(
                files, cp.parent / "result.json", row["files"][str(cp.parent / "result.json")]
            )
            verify_files(arm["files"])
            fork = files.record(cp.parent / "fork.json")
            expected_parent = ROOT / f"seed{seed}/sparse-300k/full/last.pt"
            require(
                Path(fork["source"]).resolve() == expected_parent.resolve()
                and fork["source_step"] == 300000
                and fork["source_sha256"] == files.bind(expected_parent, row["source_sha256"]),
                "Original sibling's 300k resume state differs",
            )
            require(
                arm["mode"] == "other"
                and arm["prepare_sha256"] == row["prepare_sha256"]
                and arm["source_sha256"] == row["source_sha256"]
                and arm["checkpoint_sha256"] == digest
                and arm["commit"] == rev
                and arm["seed"] == seed,
                "Wrong sibling trajectory",
            )
    margins = dict.fromkeys(MARGINS, False)
    if has_weights:
        cp = (
            Path(row["terminal_paths"]["full"])
            if family == "rgb-four"
            else root / "sibling/final.pt"
        )
        digest = row["checkpoint_sha256"]["500k/full" if family == "rgb-four" else "other"]
        files.bind(cp, digest)
        refs = controls(seed, splits, checkpoint_hashes(), files)
        margins = _verified_queries(q["arms"][alias], cp, digest, refs)
    else:
        require(
            not row["passed"] and not q["arms"], "Missing weights at a claimed measured endpoint"
        )
    return {
        "seed": seed,
        "commit": rev,
        "passed": row["passed"],
        "reason": reason,
        "has_endpoint": has_weights,
        "query_checks": margins,
        "input_sha256": dict(files),
        "reported_winner_replay": row.get("route") == "reported" if family == "rgb-four" else None,
    }


def eligibility(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Require primary replication and both causal margins on four of the same five seeds."""
    counts: dict[str, dict[str, int]] = {}
    pilots: dict[str, bool | None] = {}
    complete = True
    for f in FAMILIES:
        indices = {r["seed"]: r for r in rows[f]}
        require(len(indices) == len(rows[f]) and indices.keys() <= set(range(5)), f)
        require(len({r["commit"] for r in rows[f]}) <= 1, f"Mixed revisions for {f}")
        pilots[f] = indices.get(0, {}).get("passed")
        complete = complete and (len(indices) == 5 or pilots[f] is False)
        counts[f] = {
            "primary": sum(r["passed"] for r in rows[f]),
            **{q: sum(r["query_checks"][q] for r in rows[f]) for q in MARGINS},
            "both_queries": sum(all(r["query_checks"][q] for q in MARGINS) for r in rows[f]),
        }
    eligible = complete and all(
        pilots[f]
        and len(rows[f]) == 5
        and min(counts[f]["primary"], counts[f]["both_queries"]) >= 4
        for f in FAMILIES
    )
    return {
        "complete": complete,
        "passed": bool(eligible),
        "status": ("eligible" if eligible else "ineligible") if complete else "pending",
        "pilots": pilots,
        "passing_counts": counts,
    }


def _cohort(rev: str, splits: dict[str, str]) -> dict[str, Any]:
    root = OUTPUT / "vision-combined-cohort"
    command = _command(rev, 0, "cohort")
    with file_lock(root / ".lock", blocking=True):
        declared(
            root / "NOTE.md",
            f"# Combined RGB admission\n\n{PROTOCOL}\n\nCommit {rev}\n"
            f"Data {DATA}: {splits}\n\n    {command}\n",
        )
        rows = {
            f: [
                _constituent(f, seed, splits)
                for seed in range(5)
                if (OUTPUT / f"{f}-s{seed}/metrics.json").is_file()
                and (OUTPUT / f"consensus-queries-{f}-s{seed}/metrics.json").is_file()
            ]
            for f in FAMILIES
        }
        result = eligibility(rows) | {
            "commit": rev,
            "command": command,
            "splits": splits,
            "constituents": rows,
            "note_sha256": sha(root / "NOTE.md"),
        }
        _head(rev)
        if result["complete"]:
            _fixed(root / "metrics.json", result)
        else:
            require(
                not (root / "metrics.json").exists(), "Formerly complete constituents disappeared"
            )
            observed = "-".join(
                "".join(str(r["seed"]) for r in rows[f]) or "none" for f in FAMILIES
            )
            _fixed(root / f"pending-{observed}.json", result)
        return result


def native_gpu(reference: Path, files: Inputs) -> str:
    """Bind W&B hardware of the native four-head 300k-to-500k full continuation."""
    names: set[str] = set()
    for p in sorted(reference.glob("wandb/run-*/files/wandb-metadata.json")):
        item = files.record(p)
        require(
            Path(item["root"]).resolve() == reference.resolve(),
            "Native GPU metadata from wrong stage",
        )
        names.add(item["gpu"])
    require(len(names) == 1 and bool(next(iter(names))), "Need one native GPU model")
    return names.pop()


def _route(row: dict[str, Any], seed: int, files: Inputs) -> tuple[Path, Path, Path, bool]:
    root = Path(row["stage_root"])
    base = ROOT / f"seed{seed}"
    sel = files.record(OUTPUT / f"rgb-four-s{seed}/selection.json")
    candidate = files.record(OUTPUT / f"rgb-four-s{seed}/candidate.json")
    chosen = base / f"acquire/a{row['selected']}/final.pt"
    require(
        row["selected"] == rgb_four.select_four(sel["train_scores"])
        and candidate["verified"]
        and Path(candidate["candidate_path"]).resolve() == chosen.resolve()
        and candidate["historical_train_score"] == sel["train_scores"][row["selected"]]
        and candidate["final_sha256"] == row["checkpoint_sha256"]["selected"],
        "Wrong TRAIN-selected candidate",
    )
    files.bind(chosen, candidate["final_sha256"])
    parent = root / "sparse-300k/full/last.pt"
    native = root / "sparse-500k/full/final.pt"
    baseline = root / "sparse-500k/baseline/final.pt"
    reported = row["route"] == "reported"
    require(
        reported == (row["selected"] == row["reported_selected"]) == (root == base),
        "Ambiguous winner inheritance",
    )
    previous = (
        None
        if reported
        else (
            read(root / "metrics.json")["commit"]
            if row["route"] == "previous_native_one_head"
            else row["commit"]
        )
    )
    upstream = rgb_four._native_chain(  # pyright: ignore[reportPrivateUsage]
        root, base, chosen, seed, files, review_rev=previous
    )
    require(
        all(row["checkpoint_sha256"][k] == v for k, v in upstream.items()),
        "Candidate projection or adaptation differs",
    )
    for arm in rgb_four.ARMS:
        hashes, _ = rgb_four._sparse_lineage(root, base, seed, arm, files, review_rev=previous)  # pyright: ignore[reportPrivateUsage]
        require(
            all(found == row["checkpoint_sha256"][name] for name, found in hashes.items()),
            "Sparse parent or endpoint differs",
        )
    cfg = release.StageConfig.from_saved(read(native.parent / "config.json")["config"])
    require(Path(cfg.data_dir).resolve() == DATA.resolve(), "Wrong continuation cohort")
    return parent, native, baseline, reported


def _evaluate(
    seed: int,
    p: Path,
    digest: str,
    refs: dict[str, Any],
    four: dict[str, Any],
    inherited: dict[str, Any] | None,
    rev: str,
) -> tuple[dict[str, Any], dict[str, bool]]:
    results: dict[str, Any] = {}
    for q, evaluate in (("chart", chart.run), ("kicks", kicks.run)):
        _head(rev)
        m = inherited[q]["measurement"] if inherited is not None else evaluate(seed, {"full": p})
        original = paired(q, m, refs[q], "full")
        results[q] = {
            "measurement": m,
            "paired": original,
            "scalar_within_margin": original["scalar"]["trial_minus_control"] <= MARGINS[q],
            "versus_native_four": paired(q, m, four[q]["measurement"], "full"),
        }
    return results, _verified_queries(results, p, digest, refs)


def endpoint_checks(
    native: dict[str, Any], sibling: dict[str, Any], four: dict[str, Any]
) -> dict[str, bool]:
    """Retain the exact full-minus-four-head-baseline criterion and check native-relative spread."""
    checks = sparse_checks(
        {"full": sibling["semantic"], "baseline": four["validation"]["baseline"]},
        four["reference_validation"],
    )
    for split in SPLITS:
        old, new = native[split], sibling[split]
        require(old["mse_matched"] and new["mse_matched"], f"Unreproduced hard MSE on {split}")
        rank = all(
            r["chart"]["full_rank"] and len(r["chart"]["eigenvalues"]) == 4 for r in (old, new)
        )
        checks[f"{split}_rank"] = rank
        checks[f"{split}_worst"] = (
            rank
            and new["reported_spread"]["worst_direction_share"]
            <= 0.80 * old["reported_spread"]["worst_direction_share"]
        )
        checks[f"{split}_mse"] = new["mse"] <= 1.05 * old["mse"]
    return checks


def _intervals(native: dict[str, Any], sibling: dict[str, Any]) -> dict[str, Any]:
    intervals: dict[str, Any] = {}
    for s, bs in zip(SPLITS, (20261123, 20261124), strict=True):
        parts = {"native": native[s], "same": native[s], "other": sibling[s]}
        try:
            intervals[s] = context_intervals(
                {a: torch.tensor(r["codes"], dtype=torch.float64) for a, r in parts.items()},
                {a: torch.tensor(r["group_mse"], dtype=torch.float64) for a, r in parts.items()},
                bs,
            )["other_minus_native"]
        except ValueError as exc:
            intervals[s] = {"unresolved": str(exc)}
    return intervals


def _fresh(
    rev: str,
    seed: int,
    out: Path,
    parent: Path,
    native: Path,
    files: Inputs,
    note: str,
    splits: dict[str, str],
    four: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    gpu = native_gpu(native.parent, files)
    if four["route"] == "fresh":
        worker = files.record(OUTPUT / f"four-queue/rgb-four-s{seed}/full.ok.json")
        require(
            worker["commit"] == four["commit"]
            and worker["seed"] == seed
            and worker["phase"] == "full"
            and worker["exit_code"] == 0
            and worker["gpu"] == gpu,
            "Native worker differs from W&B",
        )
    _idle_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = _runtime()
    require(runtime["gpu_model"] == gpu, f"Required native GPU model: {gpu}")
    cfg = release.StageConfig.from_saved(read(native.parent / "config.json")["config"])
    cfg = replace(
        cfg, output_dir=str(out.resolve()), wandb_group=f"icml-review-vision-combined-s{seed}"
    )
    source_sha = files.bind(parent)
    base = {
        "seed": seed,
        "commit": rev,
        "source_sha256": source_sha,
        "splits": splits,
        "runtime": runtime,
        "tolerance": cfg.tolerance(),
        "note_sha256": sha(out / "NOTE.md"),
    }
    pre = out / "prepare.json"
    if not pre.exists():
        prep = _prepare(dict(base), out, cfg, parent, native.parent, note, rev)
        _head(rev)
        write_json(pre, prep)
    prep = files.record(pre)
    require(
        all(prep[k] == v for k, v in base.items() if k != "runtime")
        and prep["runtime"]["gpu_model"] == gpu,
        "Previous preparation differs",
    )
    verify_files(prep["files"])
    if not prep["ready"]:
        return {
            "prepare_sha256": files[str(pre)],
            "reason": "preparation",
            "checks": prep["checks"],
            "verified": prep["checks"]["native_replay"]
            and prep["checks"].get("native_teacher_lookup") is True
            and prep["checks"].get("native_first_window", True),
            "passed": False,
        }, False
    base["prepare_sha256"] = files[str(pre)]
    directory = out / "sibling"
    directory.mkdir(exist_ok=True)
    result = directory / "result.json"
    if not result.exists():
        _arm(base, out, replace(cfg, output_dir=str(directory.resolve())), parent, "sibling", rev)
    record = files.record(result)
    verify_files(record["files"])
    require(
        all(record[k] == v for k, v in base.items() if k != "runtime")
        and record["mode"] == "other"
        and record["performance"]["updates"] == cfg.steps - 300_000,
        "Sibling arm changed",
    )
    return {
        "prepare_sha256": files[str(pre)],
        "native_gpu_model": gpu,
        "training_result_sha256": files[str(result)],
        "new_sibling_updates": record["performance"]["updates"],
        "training_performance": record["performance"],
    }, True


def run(rev: str, seed: int, phase: str) -> dict[str, Any]:
    """Archive conditional admission, a reused trajectory, or a matched new continuation."""
    _head(rev)
    require(
        seed in range(5) and (phase == "run" or (phase == "cohort" and seed == 0)),
        "Invalid seed or phase",
    )
    subprocess.run(["sha256sum", "-c", "--quiet", MANIFEST.name], cwd=MANIFEST.parent, check=True)
    splits = _split_ids()
    if phase == "cohort":
        return _cohort(rev, splits)
    cohort_path = OUTPUT / "vision-combined-cohort/metrics.json"
    files = Inputs()
    cohort = files.record(cohort_path)
    require(
        cohort["commit"] == rev
        and cohort["splits"] == splits
        and cohort["status"] == "eligible"
        and cohort["passed"]
        and all(cohort[key] == val for key, val in eligibility(cohort["constituents"]).items()),
        "Constituents have not jointly qualified",
    )
    for records in cohort["constituents"].values():
        for row in records:
            verify_files(row["input_sha256"])
    files.bind(cohort_path.parent / "NOTE.md", cohort["note_sha256"])
    if seed:
        pilot = files.record(OUTPUT / "vision-combined-s0/metrics.json")
        require(
            pilot["commit"] == rev
            and pilot["seed"] == 0
            and pilot["verified"]
            and pilot["passed"]
            and all(pilot["checks"].values())
            and pilot["cohort_sha256"] == files[str(cohort_path)],
            "Combined pilot has not qualified",
        )
        verify_files(pilot["input_sha256"])
    root = OUTPUT / f"rgb-four-s{seed}"
    old = OUTPUT / f"vision-invariant-s{seed}"
    four = files.record(root / "metrics.json")
    original = files.record(old / "metrics.json")
    out = OUTPUT / f"vision-combined-s{seed}"
    command = _command(rev, seed, phase)
    with file_lock(out / ".lock"):
        dest = out / "metrics.json"
        note = (
            f"# Combined RGB seed {seed}\n\n{PROTOCOL}\n\nCommit {rev}\nData {DATA}: {splits}\n"
            f"Cohort SHA256 {files[str(cohort_path)]}\n"
            f"Four-head declaration {four['note_sha256']}, "
            f"receipt {files[str(root / 'metrics.json')]}\n"
            f"Sibling declaration {original['note_sha256']}, "
            f"receipt {files[str(old / 'metrics.json')]}\n"
            f"Selected a{four['selected']}, route {four['route']}. "
            f"Native endpoint digests {four['checkpoint_sha256']}.\n\n    {command}\n"
        )
        if four["terminal_paths"]:
            reference = Path(four["terminal_paths"]["full"]).parent
            fork = files.record(reference / "fork.json")
            cfg = files.record(reference / "config.json")["config"]
            note += (
                f"\nNative 300k resumable {fork['source']}, SHA-256 {fork['source_sha256']}. "
                f"Numeric tau {cfg['tau']}.\n"
            )
        declared(out / "NOTE.md", note)
        base: dict[str, Any] = {
            "seed": seed,
            "commit": rev,
            "cohort_sha256": files[str(cohort_path)],
            "command": command,
            "splits": splits,
            "note_sha256": files.bind(out / "NOTE.md"),
            "selected": four["selected"],
            "reported_selected": four["reported_selected"],
            "four_route": four["route"],
            "new_sibling_updates": 0,
        }
        if dest.exists():
            prev = read(dest)
            require(
                all(prev[k] == v for k, v in base.items() if k != "new_sibling_updates"),
                "Existing combined identity differs",
            )
            verify_files(prev["input_sha256"])
            _head(rev)
            return prev

        def finish(extra: dict[str, Any]) -> dict[str, Any]:
            value = {**base, **extra, "input_sha256": dict(files)}
            _head(rev)
            _fixed(dest, value)
            return value

        if not four["terminal_paths"]:
            return finish(
                {
                    "reason": "no_four_head_endpoint",
                    "verified": True,
                    "passed": False,
                    "checks": {},
                    "route": "unavailable",
                    "four_reason": four["reason"],
                }
            )
        parent, native_path, baseline, inherited = _route(four, seed, files)
        base.update(
            {
                "route": "reported_winner_replay" if inherited else "fresh_changed_winner",
                "native_parent": str(parent),
                "parent_sha256": files.bind(parent),
                "native_endpoint": str(native_path),
                "baseline": str(baseline),
                "tolerance": read(native_path.parent / "config.json")["config"]["tau"],
            }
        )
        fq = files.record(OUTPUT / f"consensus-queries-rgb-four-s{seed}/metrics.json")["arms"][
            "full"
        ]
        inherited_q = None
        if inherited:
            require(
                original["source_sha256"] == files[str(parent)]
                and four["checkpoint_sha256"]["500k/full"] == files[str(native_path)],
                "Deterministic inheritance parent or native endpoint differs",
            )
            if "checkpoint_sha256" not in original:
                return finish(
                    {
                        "reason": "no_original_sibling_endpoint",
                        "verified": False,
                        "passed": False,
                        "checks": {},
                    }
                )
            require(
                original["checkpoint_sha256"]["native"] == files[str(native_path)],
                "Original native endpoint differs",
            )
            sibling_path = old / "sibling/final.pt"
            native = _record(
                files, old / "native.json", original["files"][str(old / "native.json")]
            )
            sibling = _record(
                files,
                old / "sibling/endpoint.json",
                original["files"][str(old / "sibling/endpoint.json")],
            )
            q = files.record(OUTPUT / f"consensus-queries-vision-invariant-s{seed}/metrics.json")
            inherited_q = q["arms"]["sibling"]
            files.bind(sibling_path, original["checkpoint_sha256"]["other"])
        else:
            extra, ready = _fresh(rev, seed, out, parent, native_path, files, note, splits, four)
            if not ready:
                return finish(extra)
            base.update(extra)
            native = files.record(out / "native.json")
            sibling = files.record(out / "sibling/endpoint.json")
            sibling_path = out / "sibling/final.pt"
        require(
            native["sha256"] == files[str(native_path)]
            and sibling["sha256"] == files.bind(sibling_path)
            and native["semantic"] == four["validation"]["full"]
            and sibling["semantic"] == files.record(sibling_path.parent / "final.json"),
            "Compared endpoints differ",
        )
        checks = endpoint_checks(native, sibling, four)
        refs = controls(seed, splits, checkpoint_hashes(), files)
        q, margins = _evaluate(
            seed, sibling_path, files[str(sibling_path)], refs, fq, inherited_q, rev
        )
        qp = out / "queries.json"
        _fixed(qp, q)
        primary = all(checks.values())
        checks.update({f"scalar_{name}": ok for name, ok in margins.items()})
        return finish(
            {
                "verified": True,
                "passed": all(checks.values()),
                "primary_passed": primary,
                "reason": "complete" if all(checks.values()) else "endpoint_or_queries",
                "checks": checks,
                "query_sha256": files.bind(qp),
                "checkpoint_sha256": {
                    "native": files[str(native_path)],
                    "other": files[str(sibling_path)],
                    "baseline": files[str(baseline)],
                },
                "terminal_paths": {
                    "sibling": str(sibling_path),
                    "native": str(native_path),
                    "baseline": str(baseline),
                },
                "semantic": {"native": native["semantic"], "other": sibling["semantic"]},
                "paired_intervals": _intervals(native, sibling),
                "interval_description": (
                    "2000 paired whole-system percentile resamples, CAL seed 20261123 and "
                    "VAL seed 20261124, conditional on model weights."
                ),
                "comparisons": {
                    split: {
                        arm: {
                            "mse": rec[split]["mse"],
                            "chart": rec[split]["chart"],
                            "spread": rec[split]["reported_spread"],
                        }
                        for arm, rec in (("native", native), ("other", sibling))
                    }
                    for split in SPLITS
                },
            }
        )


def main() -> None:
    """Run either the prerequisite cohort audit or one combined seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("seed", type=int, choices=range(5))
    parser.add_argument("phase", choices=("cohort", "run"))
    args = parser.parse_args()
    run(args.commit, args.seed, args.phase)


if __name__ == "__main__":
    main()
