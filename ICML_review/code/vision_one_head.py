"""Conditionally test one archived, unranked RGB acquisition through both sparse branches."""

import argparse
import json
import math
import multiprocessing
import shlex
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from diagnostics.shared.sampling import GroupSampler
from diagnostics.vision_dimer import data, evaluate, train
from scjepa.experiments.storage import file_lock, write_json

from .single_acquisition import _head, _idle_gpu  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import (
    DATA,
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

MODULE = "ICML_review.code.vision_one_head"
PHASES = ("prepare", "full", "baseline", "assess")
SCORES = {"calibration": "calibration_prediction_mse", "validation": "prediction_mse"}
RTOL, ATOL = 1e-4, 1e-8
PROTOCOL = (
    "Hypothesis: one independent frozen-state acquisition can preserve the paired recovery of "
    "best-of-sixteen at unchanged numerical tolerance and sparse update budgets. Attempt zero "
    "is fixed in advance. Reuse its archived full-length weights, freshly hash both files, replay "
    "the historical training score and compare its frozen encoders with the manifest-listed joint "
    "initializer. Its independently archived summaries corroborate the bytes; the original "
    "manifest does not list the a0 checkpoint. Validate the original deterministic first-window "
    "calibration and validation MSE for both heads at rtol=1e-4, atol=1e-8. A failed identity "
    "or numerical replay is a technical failure. Admit a0 only if its reproduced calibration "
    "MSE is at most the joint head's. On scientific rejection stop without retries. Apply the "
    "reported TRAIN projection and adaptation. Require both adapted deterministic CAL and VAL "
    "MSE at most 1.10 times the reported initializer and minimum validation drag R-squared at "
    "least 0.95. Then run baseline and full for 300k plus 200k sparse updates with the published "
    "same-seed tau. Require full validation MCC within 0.01 of reported or higher, at least 80% "
    "of the original baseline-to-full gain, no higher full total SHD, at most 1.10 times its "
    "NMSE, and minimum drag R-squared at least 0.95. A passing seed-zero endpoint licenses "
    "seeds one through four. A failing pilot leaves the one-head route unestablished. The "
    "direct route is descriptive only. Learning is label-free. Retention and sparse recovery "
    "criteria use calibration/validation physical evaluation labels."
)


def _equal_states(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    if left.keys() != right.keys() or not left:
        raise ValueError("state dict has missing or extra entries")
    for name, value in left.items():
        other = right[name]
        if isinstance(value, Tensor):
            if (
                not isinstance(other, Tensor)
                or value.dtype != other.dtype
                or not torch.equal(value, other)
            ):
                raise ValueError(f"checkpoint tensor differs: {name}")
        elif value != other:
            raise ValueError(f"checkpoint metadata differs: {name}")


def corroborate_model(
    joint: dict[str, Any], final: dict[str, Any], last: dict[str, Any]
) -> dict[str, int]:
    """Check final/resumable weights and both frozen encoders against the manifest-listed joint."""
    _equal_states(final["model"], last["model"])
    if ("head" in last) != ("head" in final):
        raise ValueError("auxiliary head keys differ")
    if final.get("head") or last.get("head"):
        _equal_states(final["head"], last["head"])
    elif final.get("head") != last.get("head"):
        raise ValueError("auxiliary head state differs")
    counts: dict[str, int] = {}
    for prefix in ("online_encoder.", "target_encoder."):
        trial = {k: v for k, v in final["model"].items() if k.startswith(prefix)}
        original = {k: v for k, v in joint["model"].items() if k.startswith(prefix)}
        _equal_states(trial, original)
        counts[prefix[:-1]] = len(trial)
    return counts


def verify_streams(
    last: dict[str, Any], cfg: train.StageConfig, cohort: data.DataConfig, device: torch.device
) -> dict[str, Any]:
    """Reconstruct the acquisition sampler/window streams and verify saved global RNG seeds."""
    if cfg.acquisition_seed is None or int(last["step"]) != cfg.steps:
        raise ValueError("acquisition seed or terminal update is missing")
    seed = cfg.acquisition_seed
    sampler_seed = cfg.seed + 7919 * seed
    window_seed = cfg.seed + 104651 * seed + 17
    global_seed = cfg.seed + 31 * seed
    sampler = GroupSampler(cohort.train_groups, cfg.batch_groups, sampler_seed)
    windows = torch.Generator().manual_seed(window_seed)
    available = cohort.episode_frames - cohort.frames + 1
    for _ in range(cfg.steps):
        sampler.next()
        if available > 1:
            torch.randint(0, available, (cfg.batch_groups, cohort.views), generator=windows)
    _equal_states(sampler.state_dict(), last["sampler"])
    if not torch.equal(windows.get_state(), last["windows"]):
        raise ValueError("acquisition time-window stream differs")
    for name, dev in (("cpu_rng", torch.device("cpu")), ("cuda_rng", device)):
        if name == "cuda_rng" and dev.type != "cuda":
            if last[name] is not None:
                raise ValueError("unexpected GPU RNG in a CPU fixture")
            continue
        stream = torch.Generator(device=dev)
        stream.set_state(last[name])
        if stream.initial_seed() != global_seed:
            raise ValueError(f"{name} seed differs from the acquisition protocol")
    return {
        "acquisition_seed": seed,
        "updates": cfg.steps,
        "sampler_seed": sampler_seed,
        "window_seed": window_seed,
        "cpu_and_device_seed": global_seed,
        "sampler_offset": int(last["sampler"]["offset"]),
        "window_stream_reproduced": True,
        "sampler_stream_reproduced": True,
    }


@torch.no_grad()
def replay_forecasts(
    payload: dict[str, Any], inputs: dict[str, data.RenderedInputs], device: torch.device
) -> dict[str, float]:
    """Replay the released first-window scorer without opening physical labels."""
    model = train.model_from_checkpoint(payload, phase="dense").to(device).eval()
    return {
        split: evaluate.prediction_mse(evaluate.collect(model, train.rendered_batches(source)))
        for split, source in inputs.items()
    }


def authenticate_candidate(seed: int) -> dict[str, Any]:
    """Verify the unlisted candidate against its parent, metadata and archived forecasts."""
    base = ROOT / f"seed{seed}"
    d = base / "acquire/a0"
    j = base / "joint"
    a0, joint, last = d / "final.pt", j / "final.pt", d / "last.pt"
    manifest = checkpoint_hashes()
    audit_path = OUTPUT / f"vision-head-audit-s{seed}/metrics.json"
    audit = read(audit_path)
    data_sha = sha(DATA / "manifest.json")
    if (
        not audit["passed"]
        or audit["scope"] != "audit_complete"
        or audit["seed"] != seed
        or audit["data_manifest_sha256"] != data_sha
        or sha(joint) != manifest[str(joint)]
        or audit["joint_checkpoint_sha256"] != manifest[str(joint)]
        or str(a0) in manifest
    ):
        raise ValueError("head audit, data, or manifest joint provenance differs")
    summaries = {"a0": d / "final.json", "joint": j / "final.json"}
    selection_path = base / "selection.json"
    for p in [*summaries.values(), selection_path]:
        if audit["input_sha256"][str(p)] != sha(p):
            raise ValueError(f"prior head-audit input digest differs: {p}")
    initial_hashes = (sha(a0), sha(last), manifest[str(joint)])
    saved = train.load_checkpoint(last)
    a = train.load_checkpoint(a0)
    parent = train.load_checkpoint(joint)
    record = read(d / "config.json")
    cfg = train.StageConfig.from_saved(record["config"])
    if (
        cfg.stage != "acquire"
        or cfg.seed != seed
        or cfg.acquisition_seed != 1000 + seed
        or cfg.steps != 100000
        or cfg.code_dim != 8
        or cfg.parent is None
        or Path(cfg.parent).resolve() != joint.resolve()
        or Path(cfg.output_dir).resolve() != d.resolve()
        or Path(cfg.data_dir).resolve() != DATA.resolve()
        or cfg.device != "cuda"
        or saved["config"] != record["config"]
        or a["config"] != record["config"]
        or saved["meta"] != record["meta"]
        or a["meta"] != record["meta"]
        or parent["meta"]["seed"] != seed
        or parent["meta"]["stage"] != "joint"
    ):
        raise ValueError("a0 stage, parent, dimension, budget or archived metadata differ")
    cohort = data.load_config(DATA)
    required_meta = {
        "schema": train.SCHEMA,
        "stage": "acquire",
        "seed": seed,
        "code_dim": cfg.code_dim,
        "episode_frames": cohort.episode_frames,
        "context": cohort.context,
        "shape": list(cohort.renderer.shape),
        "initial_logit": cfg.initial_logit,
    }
    if (
        record["meta"] != required_meta
        or record["data"] != cohort.as_record()
        or saved["geco"] is not None
    ):
        raise ValueError("acquisition geometry, manifest or optimizer condition differs")
    counts = corroborate_model(parent, a, saved)
    device = torch.device(cfg.device)
    rng = verify_streams(saved, cfg, cohort, device)
    selection = read(selection_path)
    history = d / "history.jsonl"
    tail = [
        row
        for row in map(json.loads, history.read_text().splitlines())
        if cfg.steps - int(selection["window"]) < row["step"] <= cfg.steps
    ]
    if (
        selection["labels_used"]
        or selection["window"] != 5000
        or [row["step"] for row in tail]
        != list(range(cfg.steps - 5000 + cfg.log_every, cfg.steps + 1, cfg.log_every))
    ):
        raise ValueError("invalid historical training-score window")
    score = sum(float(row["prediction"]) for row in tail) / len(tail)
    if not math.isclose(score, selection["scores"][cfg.output_dir], rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("a0 historical trailing training score differs")
    inputs = {split: data.RenderedInputs(DATA, split, device) for split in SCORES}
    references: dict[str, Any] = {}
    for name, payload in (("joint", parent), ("a0", a)):
        archive = read(summaries[name])
        if archive["step"] != cfg.steps:
            raise ValueError("wrong archived summary update")
        truth = {s: archive[k] for s, k in SCORES.items()}
        if any(truth[s] != audit["heads"][name][k] for s, k in SCORES.items()):
            raise ValueError("archived forecast values disagree with the head audit")
        actual = replay_forecasts(payload, inputs, device)
        matches = {s: math.isclose(actual[s], truth[s], rel_tol=RTOL, abs_tol=ATOL) for s in SCORES}
        references[name] = {
            "released": truth,
            "measured": actual,
            "matches": matches,
            "summary_sha256": sha(summaries[name]),
        }
    if (sha(a0), sha(last), sha(joint)) != initial_hashes:
        raise ValueError("archive changed during candidate corroboration")
    return {
        "seed": seed,
        "final_sha256": initial_hashes[0],
        "last_sha256": initial_hashes[1],
        "joint_sha256": manifest[str(joint)],
        "candidate_path": str(a0),
        "candidate_manifest_listed": False,
        "archive_audit_sha256": sha(audit_path),
        "archive_audit_commit": audit["commit"],
        "config_sha256": sha(d / "config.json"),
        "selection_sha256": sha(selection_path),
        "history_sha256": sha(history),
        "manifest_sha256": data_sha,
        "final_and_last_model_identical": True,
        "frozen_encoder_tensors": counts,
        "normalized_stage_config": asdict(cfg),
        "streams": rng,
        "historical_a0_train_score": score,
        "first_window": references,
        "rtol": RTOL,
        "atol": ATOL,
        "verified": all(all(r["matches"].values()) for r in references.values()),
    }


def _candidate_worker(rev: str, seed: int, record: Path) -> None:
    """Release the candidate-audit CUDA context before the native training subprocesses start."""
    _head(rev)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = authenticate_candidate(seed) | {"commit": rev}
    _head(rev)
    write_json(record, result)


def _candidate_receipt(rev: str, seed: int, record: Path) -> dict[str, Any]:
    if not record.exists():
        _idle_gpu()
        proc = multiprocessing.get_context("spawn").Process(
            target=_candidate_worker, args=(rev, seed, record)
        )
        proc.start()
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"candidate identity worker failed with exit code {proc.exitcode}")
    result = read(record)
    base = ROOT / f"seed{seed}"
    sources = {
        "final_sha256": base / "acquire/a0/final.pt",
        "last_sha256": base / "acquire/a0/last.pt",
        "joint_sha256": base / "joint/final.pt",
        "config_sha256": base / "acquire/a0/config.json",
        "selection_sha256": base / "selection.json",
        "history_sha256": base / "acquire/a0/history.jsonl",
        "archive_audit_sha256": OUTPUT / f"vision-head-audit-s{seed}/metrics.json",
        "manifest_sha256": DATA / "manifest.json",
    }
    if (
        result["commit"] != rev
        or result["seed"] != seed
        or any(result[k] != sha(path) for k, path in sources.items())
    ):
        raise ValueError("candidate receipt or archived source changed")
    for name, directory in (("joint", base / "joint"), ("a0", base / "acquire/a0")):
        if result["first_window"][name]["summary_sha256"] != sha(directory / "final.json"):
            raise ValueError("archived evaluation summary changed")
    return result


def scientific_trigger(output: Path) -> dict[str, Any]:
    """Require closed original queues and a validated scientific failure of direct projection."""
    statuses: dict[str, int] = {}
    for suite in ("synthetic-loss", "vision-loss", "vision-direct"):
        statuses[suite] = int((output / f"loss-queues/{suite}.exit_code").read_text().strip())
    if any(statuses.values()):
        raise ValueError("required original queues did not finish successfully")
    failed: list[dict[str, Any]] = []
    pilot = read(output / "vision-direct-s0/metrics.json")
    for n in range(5) if pilot["passed"] else range(1):
        path = output / f"vision-direct-s{n}/metrics.json"
        row = read(path)
        seed = int(path.parent.name[-1])
        if (
            row["seed"] != seed
            or row["commit"] != pilot["commit"]
            or row["commit"] not in (path.parent / "NOTE.md").read_text()
        ):
            raise ValueError("direct receipt identity differs")
        init = adaptation_checks(row["adaptation"], row["reference_adaptation"])
        if init != row["adaptation_checks"]:
            raise ValueError("direct adaptation decision differs")
        if all(init.values()):
            checks = sparse_checks(row["validation"], row["reference_validation"])
            if row["checks"] != checks:
                raise ValueError("direct sparse decision differs")
            passed = all(checks.values())
        else:
            if row["reason"] != "adaptation":
                raise ValueError("direct adaptation status differs")
            passed = False
        if row["passed"] is not passed:
            raise ValueError("inconsistent direct decision")
        if not passed:
            failed.append(
                {"seed": seed, "path": str(path), "sha256": sha(path), "commit": row["commit"]}
            )
    if not failed:
        raise ValueError("one head requires a prior scientific failure of direct projection")
    return {"failed_direct": failed, "queue_exits": statuses}


def _stage_identity(directory: Path, rev: str, final_digest: str, parent_digest: str) -> None:
    expected = {
        "commit": rev,
        "parent_sha256": parent_digest,
        "checkpoint_sha256": final_digest,
        "summary_sha256": sha(directory / "final.json"),
    }
    if (
        read(directory / "review-completion.json") != expected
        or sha(directory / "final.pt") != final_digest
    ):
        raise ValueError(f"checkpoint, summary or source receipt changed in {directory}")


def admission_checks(candidate: dict[str, Any]) -> dict[str, bool]:
    """Admit this fixed candidate on reproduced, label-free calibration forecasts only."""
    if not candidate["verified"]:
        raise RuntimeError("a0 or joint raw CAL/VAL replay differed; see candidate.json")
    f = candidate["first_window"]
    return {
        "a0_calibration_at_most_joint": f["a0"]["measured"]["calibration"]
        <= f["joint"]["measured"]["calibration"]
    }


def _hashes_prepared(out: Path, rev: str, seed: int) -> dict[str, Any]:
    row = read(out / "prepare.json")
    candidate = out / "candidate.json"
    if (
        row["commit"] != rev
        or row["seed"] != seed
        or sha(candidate) != row["candidate_sha256"]
        or not all(row["candidate_checks"].values())
        or not all(row["adaptation_checks"].values())
    ):
        raise ValueError("prepare did not license these sparse stages")
    hashes = row["checkpoint_sha256"]
    _stage_identity(out / "projected", rev, hashes["projected"], hashes["a0"])
    _stage_identity(out / "adapt", rev, hashes["adapt"], hashes["projected"])
    if row["adaptation"] != read(out / "adapt/final.json"):
        raise ValueError("the prepared adaptation report changed")
    return row


def _base(rev: str, seed: int, trigger: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": seed,
        "commit": rev,
        "trigger": trigger,
        "data_manifest_sha256": sha(DATA / "manifest.json"),
    }


@dataclass(frozen=True)
class OneHeadTrial:
    """Configure a one-head trial using the archived admission and sparse gates."""

    folder: str = "vision-one-head"
    module: str = MODULE
    protocol: str = PROTOCOL
    trigger: Callable[[Path], dict[str, Any]] = scientific_trigger
    project_command: tuple[str, ...] | None = None


ORIGINAL_TRIAL = OneHeadTrial()


def run_phase(rev: str, seed: int, phase: str, *, trial: OneHeadTrial = ORIGINAL_TRIAL) -> None:
    """Execute one independently lockable phase against the predeclared original comparator."""
    _head(rev)
    out = OUTPUT / f"{trial.folder}-s{seed}"
    base = ROOT / f"seed{seed}"
    trigger = trial.trigger(OUTPUT)
    if seed:
        pilot = read(OUTPUT / f"{trial.folder}-s0/metrics.json")
        if (
            pilot["commit"] != rev
            or pilot["seed"] != 0
            or not pilot["passed"]
            or not pilot.get("checks")
            or not all(sparse_checks(pilot["validation"], pilot["reference_validation"]).values())
        ):
            raise ValueError("expansion requires the same-revision complete passing pilot")
    commands = [
        "PYTHONPATH=$PWD:$PWD/src " + shlex.join([PYTHON, "-m", trial.module, rev, str(seed), p])
        for p in PHASES
    ]
    manifest = read(DATA / "manifest.json")
    split_hashes = {
        s: manifest["splits"][s]["sha256"] for s in ("train", "calibration", "validation")
    }
    note = (
        f"# One fixed acquisition, vision seed {seed}\n\n{trial.protocol}\n\nCommit: {rev}\n"
        f"Data manifest SHA-256: {sha(DATA / 'manifest.json')}\n"
        f"Used split SHA-256: {json.dumps(split_hashes, sort_keys=True)}\n"
        f"Data: {DATA}. Fixed candidate: {base / 'acquire/a0/final.pt'}.\n\n    "
        + "\n    ".join(commands)
        + "\n"
    )
    root_note = out / "NOTE.md"
    if phase == "prepare":
        declared(root_note, note)
    elif root_note.read_text() != note:
        raise ValueError("scientific declaration differs")
    group = (
        f"icml-review-one-head-s{seed}"
        if trial.folder == "vision-one-head"
        else f"icml-review-{trial.folder}-s{seed}"
    )
    final = out / "metrics.json"
    subprocess.run(
        ["sha256sum", "-c", "--quiet", "reported-checkpoints.sha256"], cwd="outputs", check=True
    )
    with file_lock(out / f".{phase}.lock"):
        if final.exists():
            row = read(final)
            if (
                row["commit"] != rev
                or row["seed"] != seed
                or row["trigger"] != trigger
                or row["candidate_sha256"] != sha(out / "candidate.json")
                or ("prepare_sha256" in row and row["prepare_sha256"] != sha(out / "prepare.json"))
            ):
                raise ValueError("existing terminal decision changed")
            return
        if phase == "prepare":
            if (out / "prepare.json").exists():
                _hashes_prepared(out, rev, seed)
                return
            record = out / "candidate.json"
            candidate = _candidate_receipt(rev, seed, record)
            gate = admission_checks(candidate)
            hashes = {"joint": candidate["joint_sha256"], "a0": candidate["final_sha256"]}
            root: dict[str, Any] = _base(rev, seed, trigger) | {
                "candidate_checks": gate,
                "candidate_sha256": sha(record),
                "checkpoint_sha256": hashes,
            }
            if not all(gate.values()):
                _head(rev)
                write_json(final, root | {"reason": "candidate", "passed": False})
                return
            result = project_adapt(
                seed,
                rev,
                out,
                note,
                base / "acquire/a0/final.pt",
                base=base,
                data_dir=DATA,
                group=group,
                project_command=trial.project_command,
            )
            result["checkpoint_sha256"].update(hashes)
            root.update(result)
            _head(rev)
            if not all(root["adaptation_checks"].values()):
                write_json(final, root | {"reason": "adaptation", "passed": False})
            else:
                write_json(out / "prepare.json", root)
            return
        prep = _hashes_prepared(out, rev, seed)
        prep_sha = sha(out / "prepare.json")
        if phase in ("full", "baseline"):
            hashes = sparse_arm(seed, rev, out, note, phase, base=base, data_dir=DATA, group=group)
            _head(rev)
            write_json(
                out / f"{phase}.json",
                {
                    "commit": rev,
                    "seed": seed,
                    "arm": phase,
                    "prepare_sha256": prep_sha,
                    "checkpoint_sha256": hashes,
                },
            )
            return
        if phase != "assess":
            raise ValueError("unknown one-head phase")
        sparse: dict[str, dict[str, Any]] = {}
        ref: dict[str, dict[str, Any]] = {}
        for arm in ("full", "baseline"):
            receipt = read(out / f"{arm}.json")
            if (
                receipt["commit"] != rev
                or receipt["seed"] != seed
                or receipt["arm"] != arm
                or receipt["prepare_sha256"] != prep_sha
            ):
                raise ValueError("sparse phase identity differs")
            first = out / f"sparse-300k/{arm}"
            endpoint = out / f"sparse-500k/{arm}"
            _stage_identity(
                first,
                rev,
                receipt["checkpoint_sha256"][f"300k/{arm}"],
                prep["checkpoint_sha256"]["adapt"],
            )
            resumed = sha(first / "last.pt")
            _stage_identity(endpoint, rev, receipt["checkpoint_sha256"][f"500k/{arm}"], resumed)
            fork = read(endpoint / "fork.json")
            if (
                fork["source_sha256"] != resumed
                or fork["source"] != str(first / "last.pt")
                or fork["source_step"] != read(first / "final.json")["step"]
            ):
                raise ValueError("archive fork lineage changed")
            prep["checkpoint_sha256"].update(receipt["checkpoint_sha256"])
            sparse[arm] = read(out / f"sparse-500k/{arm}/final.json")
            reference_path = base / f"sparse-500k/{arm}/final.json"
            audit = read(OUTPUT / f"vision-head-audit-s{seed}/metrics.json")
            if audit["input_sha256"][str(reference_path)] != sha(reference_path):
                raise ValueError("reported comparator changed since the head audit")
            ref[arm] = read(reference_path)
        checks = sparse_checks(sparse, ref)
        direct = OUTPUT / f"vision-direct-s{seed}/metrics.json"
        descriptive = None
        if direct.is_file():
            other = read(direct)
            descriptive = {
                "path": str(direct),
                "sha256": sha(direct),
                "passed": other["passed"],
                "adaptation": other["adaptation"],
                "validation": other.get("validation"),
            }
        _head(rev)
        write_json(
            final,
            prep
            | {
                "checks": checks,
                "passed": all(checks.values()),
                "validation": sparse,
                "reference_validation": ref,
                "direct_descriptive": descriptive,
                "prepare_sha256": prep_sha,
            },
        )


def main() -> None:
    """Run the requested one-head phase with a pinned checkout."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("commit")
    cli.add_argument("seed", type=int, choices=range(5))
    cli.add_argument("phase", choices=PHASES)
    args = cli.parse_args()
    run_phase(args.commit, args.seed, args.phase)


if __name__ == "__main__":
    main()
