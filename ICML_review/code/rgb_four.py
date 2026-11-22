"""Cap frozen RGB acquisition at four predetermined native starts."""

import argparse
import math
import shlex
import subprocess
from pathlib import Path
from typing import Any

from scjepa.experiments.storage import file_lock, write_json

from . import halving
from .single_acquisition import _head  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import (
    DATA,
    MANIFEST,
    OUTPUT,
    PYTHON,
    ROOT,
    adaptation_checks,
    checkpoint_hashes,
    declared,
    project_adapt,
    read,
    sha,
    sparse_arm,
    sparse_checks,
)
from .vision_one_head import (
    _candidate_receipt,  # pyright: ignore[reportPrivateUsage]
    _stage_identity,  # pyright: ignore[reportPrivateUsage]
    admission_checks,
)

MODULE = "ICML_review.code.rgb_four"
ARMS = ("full", "baseline")
PHASES = ("prepare", "full", "baseline", "assess")
PROTOCOL = """Hypothesis: the first four full-length RGB acquisitions suffice for the reported
pipeline at its original tolerances. The sole change to the released pipeline is acquisition
count. Use the original indices 0,1,2,3 in their original order, replay the native mean of
TRAIN forecast logs over the last 5000 of 100000 updates, and choose the minimum with the
original candidate order breaking ties. Reconstruct all sixteen archived scores to authenticate
the comparator. No calibration/validation ranking, failed candidate replacement or cap search.
The earlier audit disclosed original winner identities before this cap was declared. Identical
weights therefore give an archive replay, not independent evidence for new initializations.
A four-head reproduction would use 400000 acquisition updates instead of 1600000. The
experiment reuses recorded sequences and reports newly executed work separately.

Carry an exactly matching released-winner branch or the verified native one-head branch through
its original verdict, including any rejection. For a genuinely different winner authenticate
the recorded weights, frozen state encoders, native sampler/seed, TRAIN score and archived
first-window CAL/VAL forecasts. Apply native TRAIN shared projection and 50000-update frozen
adaptation. Require both adapted CAL and VAL forecasts to be at most 1.10 times the reported
best-of-sixteen initializer and minimum validation drag R-squared at least .95. Run both native
sparse arms to 300000 and 500000 updates at the published same-seed numeric tau only after
admission. Require full MCC within .01 of reported or higher, at least .80 of reported
full-minus-baseline MCC gain, no larger total full SHD, full NMSE at most 1.10 times reported
and minimum drag R-squared at least .95. Stop on a failed gate; only a passing seed-zero
endpoint qualifies seeds 1--4. Fixed physical queries are evaluated separately for each
available terminal endpoint. Training and ranking use no physical labels or test data."""


def select_four(scores: list[float]) -> int:
    """Select the first minimum among indices zero through three, after a full finite replay."""
    if len(scores) != 16 or not all(math.isfinite(s) and s >= 0 for s in scores):
        raise ValueError("expected sixteen finite trailing training scores")
    return min(range(4), key=scores.__getitem__)


def _bind(files: dict[str, str], p: Path, expected: str | None = None) -> str:
    digest = sha(p)
    if (expected is not None and digest != expected) or files.setdefault(str(p), digest) != digest:
        raise ValueError(f"input digest changed at {p}")
    return digest


def _verify(files: dict[str, str]) -> None:
    for name, h in files.items():
        if sha(Path(name)) != h:
            raise ValueError(f"bound input changed: {name}")


def _trigger(files: dict[str, str]) -> None:
    p = OUTPUT / "rgb-predictive-s0"
    _bind(files, p / "metrics.json")
    row = read(p / "metrics.json")
    checks = adaptation_checks(row["adaptation"], row["reference_adaptation"])
    if (
        row["seed"] != 0
        or row["reason"] != "adaptation"
        or row["passed"]
        or row["adaptation_checks"] != checks
        or all(checks.values())
    ):
        raise ValueError(
            "conditional four-head fallback requires the verified predictive adaptation miss"
        )
    if read(p / "adapt/final.json") != row["adaptation"]:
        raise ValueError("predictive endpoint differs")
    _bind(files, p / "candidate.json", row["candidate_sha256"])
    _stage_identity(
        p / "adapt",
        row["commit"],
        row["checkpoint_sha256"]["adapt"],
        row["checkpoint_sha256"]["projected"],
    )


def _selection(seed: int, rev: str, out: Path) -> dict[str, Any]:
    replay = halving.run("vision", seed)
    scores = list(replay["endpoint_scores"])
    index = select_four(scores)
    if min(range(16), key=scores.__getitem__) != replay["original_attempt"]:
        raise ValueError("sixteen-way training selection differs")
    files = dict(replay["input_sha256"])
    audit_path = OUTPUT / f"vision-head-audit-s{seed}/metrics.json"
    _bind(files, audit_path)
    audit = read(audit_path)
    if (
        audit["seed"] != seed
        or not audit["passed"]
        or audit["selected"] != f"a{replay['original_attempt']}"
    ):
        raise ValueError("original selection differs from frozen audit")
    # Bind each candidate history to the earlier replay if that hash is available.
    older = OUTPUT / f"halving-vision-s{seed}/metrics.json"
    _bind(files, older)
    prev = read(older)
    if prev["endpoint_scores"] != scores or prev["input_sha256"] != replay["input_sha256"]:
        raise ValueError("previous independent 16-way replay differs")
    record = {
        "seed": seed,
        "commit": rev,
        "train_scores": scores,
        "candidate_indices": list(range(4)),
        "selected": index,
        "reported_selected": replay["original_attempt"],
        "input_sha256": files,
        "window": 5000,
        "updates_per_head": 100000,
        "acquisition_updates_reproduction": 400000,
        "acquisition_updates_reference": 1600000,
        "new_acquisition_updates": 0,
        "selection_used_labels": False,
    }
    dest = out / "selection.json"
    if dest.exists() and read(dest) != record:
        raise ValueError("prior selection receipt changed")
    write_json(dest, record)
    return record


def _reference(seed: int, base: Path, files: dict[str, str]) -> dict[str, Any]:
    ap = OUTPUT / f"vision-head-audit-s{seed}/metrics.json"
    refs: dict[str, Any] = {}
    audit = read(ap)
    _bind(files, ap)
    for arm in ARMS:
        p = base / f"sparse-500k/{arm}/final.json"
        _bind(files, p, audit["input_sha256"][str(p)])
        refs[arm] = read(p)
        if refs[arm]["step"] != 500000:
            raise ValueError("wrong archived sparse comparison step")
    return refs


def _native_chain(
    root: Path,
    base: Path,
    source: Path,
    seed: int,
    files: dict[str, str],
    *,
    review_rev: str | None,
) -> dict[str, str]:
    """Authenticate the projection and adaptation from the chosen native acquisition."""
    hashes: dict[str, str] = {}
    manifest = checkpoint_hashes()
    prev = source
    for stage in ("projected", "adapt"):
        directory = root / stage
        cp = directory / "final.pt"
        h = _bind(files, cp, manifest[str(cp)] if review_rev is None else None)
        _bind(files, directory / "final.json")
        if review_rev is not None:
            _stage_identity(directory, review_rev, h, sha(prev))
            _bind(files, directory / "review-completion.json")
        if stage == "projected":
            desc = read(directory / "final.json")
            if (
                Path(desc["source"]).resolve() != source.parent.resolve()
                or desc["labels_used"] is not False
            ):
                raise ValueError("projection was not fitted from the selected native head")
        else:
            cfg = read(directory / "config.json")["config"]
            _bind(files, directory / "config.json")
            expected = read(base / "adapt/config.json")["config"]
            if (
                cfg["stage"] != "adapt"
                or cfg["steps"] != expected["steps"]
                or cfg["seed"] != seed
                or Path(cfg["parent"]).resolve() != prev.resolve()
            ):
                raise ValueError("wrong adapted initializer or stage")
        hashes[stage] = h
        prev = cp
    return hashes


def _sparse_lineage(
    root: Path,
    base: Path,
    seed: int,
    arm: str,
    files: dict[str, str],
    *,
    review_rev: str | None,
    budgets: tuple[int, int] = (300000, 500000),
) -> tuple[dict[str, str], Path]:
    manifest = checkpoint_hashes()
    first = root / f"sparse-300k/{arm}"
    end = root / f"sparse-500k/{arm}"
    reference = read(base / f"sparse-500k/{arm}/config.json")["config"]
    hashes: dict[str, str] = {}
    for label, p, steps in (("300k", first, budgets[0]), ("500k", end, budgets[1])):
        cp = p / "final.pt"
        summary = p / "final.json"
        cfg_path = p / "config.json"
        h = _bind(files, cp, manifest[str(cp)] if review_rev is None else None)
        _bind(files, summary)
        _bind(files, cfg_path)
        cfg = read(cfg_path)["config"]
        if (
            (cfg["stage"], cfg["steps"], cfg["seed"]) != (arm, steps, seed)
            or cfg["tau"] != reference["tau"]
            or read(summary)["step"] != steps
            or Path(cfg["parent"]).resolve() != (root / "adapt/final.pt").resolve()
        ):
            raise ValueError("unexpected sparse source, tolerance, update, seed or arm")
        parent = root / "adapt/final.pt" if label == "300k" else first / "last.pt"
        parent_hash = _bind(files, parent)
        if review_rev is not None:
            _stage_identity(p, review_rev, h, parent_hash)
            _bind(files, p / "review-completion.json")
        if label == "500k":
            fork = read(p / "fork.json")
            _bind(files, p / "fork.json")
            if (Path(fork["source"]).resolve(), fork["source_step"], fork["source_sha256"]) != (
                parent.resolve(),
                budgets[0],
                parent_hash,
            ):
                raise ValueError("sparse fork lineage differs")
        hashes[label + "/" + arm] = h
    return hashes, end / "final.pt"


def _terminal(
    base_row: dict[str, Any],
    out: Path,
    stage: Path,
    base: Path,
    seed: int,
    files: dict[str, str],
    *,
    review_rev: str | None,
) -> dict[str, Any]:
    row = dict(base_row)
    hashes = dict(row["checkpoint_sha256"])
    paths = {}
    measured: dict[str, Any] = {}
    for arm in ARMS:
        found, cp = _sparse_lineage(stage, base, seed, arm, files, review_rev=review_rev)
        hashes.update(found)
        paths[arm] = str(cp)
        measured[arm] = read(cp.with_name("final.json"))
        if stage == out:
            arm_path = out / f"{arm}.json"
            rec = read(arm_path)
            if (
                rec["commit"] != row["commit"]
                or rec["seed"] != seed
                or rec["arm"] != arm
                or rec["prepare_sha256"] != sha(out / "prepare.json")
                or rec["checkpoint_sha256"] != found
            ):
                raise ValueError("fresh sparse arm differs from prepare or terminal lineage")
            _bind(files, arm_path)
    refs = _reference(seed, base, files)
    gates = sparse_checks(measured, refs)
    return row | {
        "validation": measured,
        "reference_validation": refs,
        "checks": gates,
        "passed": all(gates.values()),
        "reason": "complete" if all(gates.values()) else "sparse_endpoints",
        "terminal_paths": paths,
        "checkpoint_sha256": hashes,
        "input_sha256": files,
    }


def _prepare(rev: str, seed: int, out: Path, base: Path, note: str) -> None:
    sel = _selection(seed, rev, out)
    index = sel["selected"]
    source = base / f"acquire/a{index}/final.pt"
    files = dict(sel["input_sha256"])
    _trigger(files)
    _bind(files, out / "selection.json")
    candidate = _candidate_receipt(rev, seed, out / "candidate.json", index)
    _bind(files, out / "candidate.json")
    if (
        not candidate["verified"]
        or candidate["candidate_path"] != str(source)
        or candidate["historical_train_score"] != sel["train_scores"][index]
    ):
        raise ValueError(
            "chosen head failed identity, exact TRAIN-score, or forecast corroboration"
        )
    for field, p in (
        ("final_sha256", source),
        ("last_sha256", source.with_name("last.pt")),
        ("joint_sha256", base / "joint/final.pt"),
    ):
        _bind(files, p, candidate[field])
    row: dict[str, Any] = {
        "seed": seed,
        "commit": rev,
        "selected": index,
        "reported_selected": sel["reported_selected"],
        "note_sha256": sha(out / "NOTE.md"),
        "selection_sha256": sha(out / "selection.json"),
        "candidate_sha256": sha(out / "candidate.json"),
        "data_manifest_sha256": sha(DATA / "manifest.json"),
        "terminal_paths": {},
        "checkpoint_sha256": {
            "selected": candidate["final_sha256"],
            "joint": candidate["joint_sha256"],
        },
    }
    old_dir = OUTPUT / f"vision-one-head-s{seed}"
    earlier: dict[str, Any] | None = None
    native_rev = None
    if index == sel["reported_selected"]:
        stage = base
        kind = "reported"
    elif index == 0 and old_dir.is_dir():
        if not (old_dir / "metrics.json").is_file():
            raise ValueError(
                "prior native one-head has no terminal verdict; do not duplicate its stages"
            )
        stage = old_dir
        kind = "previous_native_one_head"
        earlier = read(stage / "metrics.json")
        native_rev = earlier["commit"]
        _bind(files, stage / "metrics.json")
        _bind(files, stage / "candidate.json", earlier["candidate_sha256"])
        if (
            earlier["seed"] != seed
            or earlier["checkpoint_sha256"]["a0"] != candidate["final_sha256"]
            or earlier["data_manifest_sha256"] != row["data_manifest_sha256"]
        ):
            raise ValueError("prior one-head did not use this candidate and cohort")
    else:
        stage = out
        kind = "fresh"
        native_rev = rev
    row.update(route=kind, stage_root=str(stage), new_acquisition_updates=0)
    if earlier is not None and earlier.get("reason") == "candidate":
        rejected = admission_checks(read(stage / "candidate.json"))
        if earlier["passed"] or earlier["candidate_checks"] != rejected or all(rejected.values()):
            raise ValueError("prior candidate rejection inconsistent")
        _head(rev)
        write_json(
            out / "metrics.json",
            row
            | {
                "reason": "inherited_candidate",
                "passed": False,
                "candidate_checks": earlier["candidate_checks"],
                "input_sha256": files,
            },
        )
        return
    if kind == "fresh":
        project_adapt(
            seed,
            rev,
            out,
            note,
            source,
            base=base,
            data_dir=DATA,
            group=f"icml-review-rgb-four-s{seed}",
        )
    hashes = _native_chain(stage, base, source, seed, files, review_rev=native_rev)
    row["checkpoint_sha256"].update(hashes)
    desc = read(stage / "adapt/final.json")
    ref = read(base / "adapt/final.json")
    _bind(files, base / "adapt/final.json")
    admission = adaptation_checks(desc, ref)
    row.update(
        adaptation=desc, reference_adaptation=ref, adaptation_checks=admission, input_sha256=files
    )
    if earlier is not None and (
        earlier["adaptation_checks"] != admission
        or earlier["adaptation"] != desc
        or earlier["reference_adaptation"] != ref
    ):
        raise ValueError("inherited development gate differs")
    _head(rev)
    if not all(admission.values()):
        if earlier is not None and (earlier["passed"] or earlier.get("reason") != "adaptation"):
            raise ValueError("prior adaptation outcome differs")
        write_json(out / "metrics.json", row | {"reason": "adaptation", "passed": False})
    elif kind == "fresh":
        write_json(out / "prepare.json", row)
    else:
        final = _terminal(row, out, stage, base, seed, files, review_rev=native_rev)
        if earlier is not None and (
            earlier["passed"] != final["passed"] or earlier["checks"] != final["checks"]
        ):
            raise ValueError("prior sparse verdict differs")
        _head(rev)
        write_json(out / "metrics.json", final)


def _verify_row(row: dict[str, Any], rev: str, seed: int, out: Path) -> None:
    if (
        row["commit"] != rev
        or row["seed"] != seed
        or row["note_sha256"] != sha(out / "NOTE.md")
        or row["selection_sha256"] != sha(out / "selection.json")
        or row["candidate_sha256"] != sha(out / "candidate.json")
    ):
        raise ValueError("wrong scientific run or declaration")
    _verify(row["input_sha256"])
    selection = read(out / "selection.json")
    if (
        row["selected"] != select_four(selection["train_scores"])
        or row["selected"] != selection["selected"]
        or row["reported_selected"] != selection["reported_selected"]
        or row["data_manifest_sha256"] != sha(DATA / "manifest.json")
    ):
        raise ValueError("run, selection or cohort differs")
    routes = {
        "reported": ROOT / f"seed{seed}",
        "previous_native_one_head": OUTPUT / f"vision-one-head-s{seed}",
        "fresh": out,
    }
    stage = routes.get(row["route"])
    if stage is None or stage != Path(row["stage_root"]):
        raise ValueError("terminal stage root differs")
    expected_arms: set[str] = set(ARMS) if "validation" in row else set()
    if set(row["terminal_paths"]) != expected_arms:
        raise ValueError("terminal paths differ from attained scientific stage")
    for arm in expected_arms:
        path = str(stage / f"sparse-500k/{arm}/final.pt")
        if (
            row["terminal_paths"][arm] != path
            or row["input_sha256"].get(path) != row["checkpoint_sha256"]["500k/" + arm]
            or read(Path(path).with_name("final.json")) != row["validation"][arm]
        ):
            raise ValueError("inconsistent terminal source, summary or digest")
    if "adaptation" in row and row["adaptation_checks"] != adaptation_checks(
        row["adaptation"], row["reference_adaptation"]
    ):
        raise ValueError("stored adaptation scientific verdict differs")
    if "validation" in row:
        checks = sparse_checks(row["validation"], row["reference_validation"])
        if (
            row["checks"] != checks
            or row["passed"] != all(checks.values())
            or not all(row["adaptation_checks"].values())
        ):
            raise ValueError("stored sparse scientific verdict differs")
    elif "passed" in row and (
        row["passed"]
        or row["terminal_paths"]
        or (row["reason"] == "adaptation" and all(row["adaptation_checks"].values()))
        or (row["reason"] == "inherited_candidate" and all(row["candidate_checks"].values()))
    ):
        raise ValueError("inconsistent pre-sparse scientific stop")


def run(rev: str, seed: int, phase: str) -> None:
    """Run one hash-pinned phase, reusing only demonstrably identical native stages."""
    _head(rev)
    out = OUTPUT / f"rgb-four-s{seed}"
    base = ROOT / f"seed{seed}"
    if seed not in range(5) or phase not in PHASES:
        raise ValueError("invalid phase or seed")
    commands = [
        "PYTHONPATH=$PWD:$PWD/src " + shlex.join([PYTHON, "-m", MODULE, rev, str(seed), p])
        for p in PHASES
    ]
    note = (
        f"# Four RGB acquisitions, seed {seed}\n\n{PROTOCOL}\n\nCheckout {rev}\n"
        f"Data {DATA}, manifest SHA256 {sha(DATA / 'manifest.json')}\n\n"
        + "\n".join(commands)
        + "\n"
    )
    if phase == "prepare":
        declared(out / "NOTE.md", note)
    elif (out / "NOTE.md").read_text() != note:
        raise ValueError("declaration differs")
    subprocess.run(["sha256sum", "-c", "--quiet", MANIFEST.name], cwd=MANIFEST.parent, check=True)
    if seed:
        pilot = OUTPUT / "rgb-four-s0"
        d = read(pilot / "metrics.json")
        _verify_row(d, rev, 0, pilot)
        if d["passed"] is not True or not d["terminal_paths"] or not all(d["checks"].values()):
            raise ValueError("seed-zero gate did not license expansion")
    with file_lock(out / f".{phase}.lock"):
        final = out / "metrics.json"
        prepared = out / "prepare.json"
        if final.exists():
            _verify_row(read(final), rev, seed, out)
        elif phase == "prepare":
            if prepared.is_file():
                _verify_row(read(prepared), rev, seed, out)
            else:
                _prepare(rev, seed, out, base, note)
        else:
            row = read(prepared)
            _verify_row(row, rev, seed, out)
            if row["route"] != "fresh" or not all(row["adaptation_checks"].values()):
                raise ValueError("no admitted fresh sparse route")
            if phase == "assess":
                files = dict(row["input_sha256"])
                _bind(files, prepared)
                result = _terminal(row, out, out, base, seed, files, review_rev=rev)
                _head(rev)
                write_json(final, result)
            else:
                hashes = sparse_arm(
                    seed,
                    rev,
                    out,
                    note,
                    phase,
                    base=base,
                    data_dir=DATA,
                    group=f"icml-review-rgb-four-s{seed}",
                )
                _head(rev)
                write_json(
                    out / f"{phase}.json",
                    {
                        "commit": rev,
                        "seed": seed,
                        "arm": phase,
                        "prepare_sha256": sha(prepared),
                        "checkpoint_sha256": hashes,
                    },
                )
        _head(rev)


def main() -> None:
    """Run the requested seed and immutable native phase."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("commit")
    cli.add_argument("seed", type=int, choices=range(5))
    cli.add_argument("phase", choices=PHASES)
    args = cli.parse_args()
    run(args.commit, args.seed, args.phase)


if __name__ == "__main__":
    main()
