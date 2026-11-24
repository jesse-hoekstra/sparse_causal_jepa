"""Run a fixed-first-initialization RGB timing pilot at the original sparse tolerance."""

import argparse
import shlex
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from diagnostics.vision_dimer import train as release
from scjepa.experiments.storage import file_lock, write_json

from . import chart, kicks
from .changed_queries import Inputs, controls, paired, require
from .consensus_gradient import context_intervals
from .single_acquisition import _head  # pyright: ignore[reportPrivateUsage]
from .timing_vision_train import frozen_teacher, train_stage
from .vision_acquisition import DATA, OUTPUT, PYTHON, ROOT, checkpoint_hashes, declared, read, sha
from .vision_invariant_eval import SPLITS, endpoint
from .vision_one_head import authenticate_candidate


def diagnose(
    commit: str,
    seed: int,
    terminal: Path,
    out: Path,
    ledger: Inputs,
    tau: float,
    split_ids: dict[str, str],
) -> dict[str, Any]:
    """Compare the endpoint with the reported and late-loss endpoints on matched systems."""
    _head(commit)
    trial = endpoint(terminal, DATA, torch.device("cuda"))
    write_json(out / "endpoint.json", trial)
    old = OUTPUT / f"vision-invariant-s{seed}"
    receipt = ledger.record(old / "metrics.json")
    require(receipt["splits"] == split_ids, "comparison cohorts differ")
    raw = {"native": old / "native.json", "same": old / "sibling/endpoint.json"}
    rows = {"other": trial}
    for key, file in raw.items():
        ledger.bind(file, receipt["files"][str(file)])
        rows[key] = read(file)
        digest = receipt["checkpoint_sha256"]["native" if key == "native" else "other"]
        require(rows[key]["sha256"] == digest, "wrong comparison checkpoint")
        ledger.bind(Path(rows[key]["checkpoint"]), digest)
    parts: dict[str, Any] = {}
    for index, split in enumerate(SPLITS):
        block = trial[split]
        parts[split] = {
            k: block[k] for k in ("mse", "chart", "mse_matched", "reported_spread", "donor_mean")
        }
        parts[split]["mse_over_tau"] = block["mse"] / tau
        codes = {
            arm: torch.tensor(r[split]["codes"], dtype=torch.float64) for arm, r in rows.items()
        }
        errors = {
            arm: torch.tensor(r[split]["group_mse"], dtype=torch.float64) for arm, r in rows.items()
        }
        try:
            intervals = context_intervals(codes, errors, 20261123 + index)
            parts[split].update(
                vs_reported=intervals["other_minus_native"], vs_late=intervals["other_minus_same"]
            )
        except ValueError as error:
            parts[split]["bootstrap_failure"] = str(error)
    _head(commit)
    reference = controls(seed, split_ids, checkpoint_hashes(), ledger)
    queries: dict[str, Any] = {}
    for name, evaluate in (("chart", chart.run), ("kicks", kicks.run)):
        _head(commit)
        try:
            measurement = evaluate(seed, {"full": terminal})
        except ValueError as error:
            # A singular fitted physical chart or undefined score is an experimental outcome.
            failure = {"status": "undefined", "reason": str(error), "sha256": trial["sha256"]}
            write_json(out / f"{name}.json", failure)
            queries[name] = failure
            _head(commit)
            continue
        arm = measurement["arms"]["full"]
        require(arm["checkpoint_sha256"] == trial["sha256"], "wrong query endpoint")
        alignment = (
            "spearman_assignment_matches_stored" if name == "chart" else "assignment_matches_stored"
        )
        write_json(out / f"{name}.json", measurement)
        queries[name] = {
            "calibration_assignment_matched": arm[alignment],
            "paired": paired(name, measurement, reference[name], "full"),
        }
        _head(commit)
    return {"semantic": trial["semantic"], "splits": parts, "queries": queries}


def run(commit: str, schedule: str, seed: int) -> None:
    """Train through every stage and evaluate one timing condition on the registered seed."""
    _head(commit)
    out = OUTPUT / f"timing-rgb-{schedule}-s{seed}"
    base = ROOT / f"seed{seed}"
    cmd = "PYTHONPATH=$PWD:$PWD/src " + shlex.join(
        [PYTHON, "-m", "ICML_review.code.timing_vision", commit, schedule, "--seed", str(seed)]
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with file_lock(out / ".timing.lock"):
        ledger = Inputs()
        manifest = ledger.record(DATA / "manifest.json")
        split_ids = {
            s: ledger.bind(DATA / f"{s}.pt", manifest["splits"][s]["sha256"])
            for s in ("train", *SPLITS)
        }
        root_note = (out / "NOTE.md").read_text()
        require(commit in root_note, "root declaration must identify the checkout")
        ledger.bind(out / "NOTE.md")
        checkpoints: dict[str, str] = {}
        stages: dict[str, Any] = {}
        archived = {
            "joint": base / "joint",
            "acquire": base / "acquire/a0",
            "adapt": base / "adapt",
            "full": base / "sparse-500k/full",
        }
        paths = {
            "joint": out / "joint",
            "acquire": out / "acquire/a0",
            "project": out / "projected",
            "adapt": out / "adapt",
            "full": out / "sparse-500k/full",
        }
        tau = 2.0 * float(ledger.record(base / "tau.json")["tau"])

        def note(directory: Path, stage: str, parent: Path | None, settings: str) -> None:
            declared(
                directory / "NOTE.md",
                f"{root_note.rstrip()}\n\nStage {stage}\n{settings}\nParent {parent}\n"
                f"Parent SHA-256 {ledger.bind(parent) if parent else 'none'}\n"
                f"Data manifest SHA-256 {ledger[str(DATA / 'manifest.json')]}\n"
                f"Exact driver command\n\n    {cmd}\n",
            )

        parent: Path | None = None
        for stage in ("joint", "acquire", "project", "adapt", "full"):
            _head(commit)
            directory = paths[stage]
            if schedule == "sparse" and stage in {"joint", "acquire"}:
                parent = archived[stage] / "final.pt"
                if stage == "joint":
                    checkpoints[stage] = ledger.bind(parent, checkpoint_hashes()[str(parent)])
                else:
                    candidate = authenticate_candidate(seed, 0)
                    require(
                        candidate["verified"] and candidate["candidate_index"] == 0,
                        "first candidate could not be corroborated",
                    )
                    write_json(out / "candidate.json", candidate)
                    checkpoints[stage] = ledger.bind(parent, candidate["final_sha256"])
                stages[stage] = {"source": str(parent), "reused": True}
                _head(commit)
                continue
            label = (
                "native TRAIN covariance projection"
                if stage == "project"
                else "frozen adapted sibling"
                if stage == "full"
                else "online sibling"
                if schedule == "entire"
                else "native"
            )
            cfg: release.StageConfig | None = None
            if stage != "project":
                original = release.StageConfig.from_saved(
                    ledger.record(archived[stage] / "config.json")["config"]
                )
                require(
                    original.stage == stage and original.seed == seed,
                    "wrong archived stage configuration",
                )
                if stage == "acquire":
                    require(
                        original.acquisition_seed == 1000 + seed, "wrong acquisition initialization"
                    )
                if stage == "full":
                    require(original.tau == tau, "sparse tolerance differs from the reported seed")
                cfg = replace(
                    original,
                    data_dir=str(DATA),
                    output_dir=str(directory),
                    parent=str(parent) if parent else None,
                    wandb_group=f"icml-review-timing-rgb-{schedule}-s{seed}",
                )
            settings = asdict(cfg) if cfg else "project to four coordinates"
            note(directory, stage, parent, f"Objective {label}\nConfig {settings}")
            final = directory / "final.pt"
            completion = directory / "completion.json"
            if completion.exists():
                previous = read(completion)
                require(
                    previous["commit"] == commit
                    and previous["note_sha256"] == sha(directory / "NOTE.md")
                    and previous["checkpoint_sha256"] == sha(final),
                    "completed stage identity differs",
                )
            else:
                # The native checkpoint may have finished just before an interrupted receipt write.
                if not (final.is_file() and (directory / "final.json").is_file()):
                    if stage == "project":
                        assert parent is not None
                        release.project(parent.parent, DATA, directory, device="cuda")
                    else:
                        assert cfg is not None
                        if cfg.stage == "full":
                            assert parent is not None
                            teacher = frozen_teacher(parent, DATA, torch.device("cuda"), directory)
                        else:
                            teacher = None
                        _head(commit)
                        train_stage(cfg, schedule, teacher, lambda: _head(commit))
                _head(commit)
                summary = read(directory / "final.json")
                if cfg is not None:
                    require(summary["step"] == cfg.steps, "incomplete stage")
                write_json(
                    completion,
                    {
                        "commit": commit,
                        "note_sha256": sha(directory / "NOTE.md"),
                        "checkpoint_sha256": sha(final),
                        "parent_sha256": ledger.bind(parent) if parent else None,
                    },
                )
            checkpoints[stage] = ledger.bind(final)
            if (directory / "last.pt").exists():
                checkpoints[f"{stage}/last"] = ledger.bind(directory / "last.pt")
            stages[stage] = {
                "source": str(final),
                "reused": False,
                "objective": label,
                "summary_sha256": ledger.bind(directory / "final.json"),
            }
            parent = final
            _head(commit)
        assert parent is not None
        note(
            out / "validation",
            "validation",
            parent,
            "Calibration-fitted probes, context-paired validation, holds and kicks",
        )
        metrics = diagnose(commit, seed, parent, out, ledger, tau, split_ids)
        _head(commit)
        write_json(
            out / "metrics.json",
            {
                "commit": commit,
                "seed": seed,
                "schedule": schedule,
                "tolerance": tau,
                "command": cmd,
                "input_sha256": ledger,
                "checkpoint_sha256": checkpoints,
                "stages": stages,
                "diagnostic_sha256": {
                    n: sha(out / n) for n in ("endpoint.json", "chart.json", "kicks.json")
                },
                **metrics,
            },
        )


def main() -> None:
    """Run one fixed-first-head RGB arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("schedule", choices=("entire", "sparse"))
    parser.add_argument("--seed", type=int, choices=range(5), default=3)
    args = parser.parse_args()
    run(args.commit, args.schedule, args.seed)


if __name__ == "__main__":
    main()
