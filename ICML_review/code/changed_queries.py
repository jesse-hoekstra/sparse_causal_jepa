"""Pair validation hold and kick responses of changed vision endpoints with released controls."""

import argparse
import subprocess
from pathlib import Path
from typing import Any

import torch

from diagnostics.vision_dimer.train import load_checkpoint
from scjepa.experiments.storage import file_lock, write_json

from . import chart, kicks
from .single_acquisition import _head, _idle_gpu  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import DATA, MANIFEST, OUTPUT, ROOT, checkpoint_hashes, declared, read, sha

TRAINING = "9869591144da171dbe558ef0d5930f4ef59ebde4"
CONTROLS = {
    "chart": "0a1ef7aa79c7aa53b1d6db1dbf1eb588d6d67917",
    "kicks": "3a344ecf0ebb056cc5b6b63c5aae5f9c8769c135",
}
ARMS = {"vision-direct": ("baseline", "full"), "vision-loss": ("full",)}
ONE_HEAD = "vision-one-head"
PROTOCOL = (
    "Hypothesis: changing the training procedure may change intervention accuracy even at matched "
    "forecast and representation quality. Change only the frozen checkpoint within each seed and "
    "arm. Fit each scalar or affine physical chart on the original labelled calibration systems. "
    "Keep physical hold levels, held and added kick, validation systems, contexts, action traces, "
    "simulator, horizon and aggregation fixed. Primary comparisons are the absolute error in the "
    "u2a-to-u2b partner slope (Q1) and the four-bead mean held-kick relative Frobenius error (Q2). "
    "Report trial minus archived same-seed, same-arm error for each chart. Negative means a "
    "smaller error, positive a larger error, zero equal errors. Retain the raw slopes, matrices "
    "and secondary scores. Outcomes are descriptive; no result qualifies another training run. "
    "Include every authenticated terminal endpoint, including failed scientific pilots. Record "
    "unavailable descendants explicitly. Validation labels enter only evaluation."
)


def require(condition: bool, reason: str) -> None:
    """Reject an inconsistent receipt before it can enter a comparison."""
    if not condition:
        raise ValueError(reason)


class Inputs(dict[str, str]):
    """Record the bytes of every input and verify prior identities where available."""

    def bind(self, path: Path, expected: str | None = None) -> str:
        """Hash the file and check its expected SHA-256."""
        key = str(path)
        if key not in self:
            self[key] = sha(path)
        digest = self[key]
        require(expected is None or expected == digest, f"SHA-256 mismatch: {path}")
        return digest

    def record(self, path: Path) -> dict[str, Any]:
        """Read a JSON receipt that is included in the run ledger."""
        self.bind(path)
        return read(path)

    def note(self, path: Path, *contents: str) -> None:
        """Verify the recorded command and identities inside a preregistration."""
        self.bind(path)
        note = path.read_text()
        require(all(item in note for item in contents), f"Inconsistent declaration: {path}")


def _command(module: str, rev: str, seed: int) -> str:
    return f"PYTHONPATH=$PWD:$PWD/src .venv/bin/python -m ICML_review.code.{module} {rev} {seed}"


def _bind_trigger(record: dict[str, Any], ledger: Inputs, manifest: str) -> None:
    failed = record["trigger"]["failed_direct"]
    require(bool(failed), "One-head training lacks a failed direct-projection decision")
    for row in failed:
        seed = row["seed"]
        path = OUTPUT / f"vision-direct-s{seed}/metrics.json"
        require(
            seed in range(5) and row["path"] == str(path) and row["commit"] == TRAINING, str(path)
        )
        ledger.bind(path, row["sha256"])
        gate = ledger.record(path)
        require(
            gate["seed"] == seed
            and gate["commit"] == TRAINING
            and gate["passed"] is False
            and gate["data_manifest_sha256"] == manifest,
            str(path),
        )


def sources(
    method: str,
    seed: int,
    split_ids: dict[str, str],
    reported: dict[str, str],
    ledger: Inputs,
    *,
    training_commit: str = TRAINING,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Find eligible terminal checkpoints and authenticate their source training histories."""
    staged = method in ("vision-direct", ONE_HEAD)
    arms = ARMS["vision-direct"] if method == ONE_HEAD else ARMS[method]
    root = OUTPUT / f"{method}-s{seed}"
    original = ROOT / f"seed{seed}"
    selected: dict[str, Path] = {}
    receipt_path = root / "metrics.json"
    receipt = ledger.record(receipt_path) if receipt_path.is_file() else None
    final_dirs = {arm: root / "sparse-500k" / arm if staged else root for arm in arms}
    state: dict[str, Any] = {"training_commit": training_commit, "skipped": {}}
    if receipt is not None:
        require(receipt["commit"] == training_commit and receipt["seed"] == seed, str(receipt_path))
        state.update(
            scientific_gate_passed=receipt.get("scientific_gate_passed", receipt["passed"]),
            training_gate_passed=receipt["passed"],
            training_checks=receipt.get(
                "checks", receipt.get("adaptation_checks", receipt.get("candidate_checks"))
            ),
            reason=receipt.get("reason"),
        )
    else:
        state.update(scientific_gate_passed=None, training_gate_passed=None)
    if seed:
        pilot_path = OUTPUT / f"{method}-s0/metrics.json"
        if pilot_path.is_file():
            pilot = ledger.record(pilot_path)
            require(pilot["commit"] == training_commit and pilot["seed"] == 0, str(pilot_path))
            state["pilot_passed"] = pilot["passed"]
        if root.joinpath("NOTE.md").is_file():
            require(state.get("pilot_passed") is True, f"Expansion lacks a passing pilot: {root}")
    if root.joinpath("NOTE.md").is_file() or receipt is not None:
        module, suffix = {
            "vision-direct": ("vision_acquisition", " direct"),
            "vision-loss": ("vision_loss", ""),
            ONE_HEAD: ("vision_one_head", " prepare"),
        }[method]
        command = _command(module, training_commit, seed) + suffix
        data_ids = [ledger.bind(DATA / "manifest.json")] if staged else list(split_ids.values())
        ledger.note(root / "NOTE.md", command, *data_ids)
        if receipt is not None and staged:
            require(receipt["data_manifest_sha256"] == data_ids[0], str(receipt_path))
            if method == ONE_HEAD:
                _bind_trigger(receipt, ledger, data_ids[0])
    for arm, directory in final_dirs.items():
        final = directory / "final.pt"
        if not final.is_file():
            if receipt is not None:
                require(
                    staged
                    and receipt.get("reason")
                    in ({"candidate", "adaptation"} if method == ONE_HEAD else {"adaptation"}),
                    f"Missing completed checkpoint: {final}",
                )
            state["skipped"][arm] = (
                f"{receipt['reason']} gate stopped training"
                if receipt is not None
                else "pilot did not qualify expansion"
                if state.get("pilot_passed") is False
                else "no terminal checkpoint"
            )
            continue
        if staged:
            last = root / "sparse-300k" / arm / "last.pt"
            parent = original / "joint/final.pt"
            ledger.bind(parent, reported[str(parent)])
            if method == ONE_HEAD:
                parent = original / "acquire/a0/final.pt"
                prepare = ledger.record(root / "prepare.json")
                require(prepare["commit"] == training_commit and prepare["seed"] == seed, str(root))
                if receipt is None:
                    _bind_trigger(prepare, ledger, ledger[str(DATA / "manifest.json")])
                candidate_path = root / "candidate.json"
                ledger.bind(candidate_path, prepare["candidate_sha256"])
                candidate = ledger.record(candidate_path)
                require(
                    candidate["verified"] is True
                    and candidate["final_and_last_model_identical"] is True
                    and candidate["commit"] == training_commit
                    and candidate["seed"] == seed
                    and candidate["candidate_path"] == str(parent)
                    and candidate["manifest_sha256"] == ledger[str(DATA / "manifest.json")],
                    str(candidate_path),
                )
                ledger.bind(
                    OUTPUT / f"vision-head-audit-s{seed}/metrics.json",
                    candidate["archive_audit_sha256"],
                )
                digest = ledger.bind(parent, candidate["final_sha256"])
                ledger.bind(parent.with_name("last.pt"), candidate["last_sha256"])
                require(
                    candidate["joint_sha256"] == ledger[str(original / "joint/final.pt")],
                    str(parent),
                )
                if receipt is not None:
                    require(receipt["checkpoint_sha256"]["a0"] == digest, str(parent))
                    ledger.bind(root / "prepare.json", receipt["prepare_sha256"])
            for stage, key in (
                (root / "projected", "projected"),
                (root / "adapt", "adapt"),
                (last.parent, f"300k/{arm}"),
                (directory, f"500k/{arm}"),
            ):
                if stage == directory:
                    parent = last
                completion = ledger.record(stage / "review-completion.json")
                require(completion["commit"] == training_commit, str(stage))
                ledger.bind(parent, completion["parent_sha256"])
                ledger.note(
                    stage / "NOTE.md", training_commit, f"Parent: {parent}\n", ledger[str(parent)]
                )
                parent = stage / "final.pt"
                digest = ledger.bind(parent, completion["checkpoint_sha256"])
                ledger.bind(stage / "final.json", completion["summary_sha256"])
                if receipt is not None:
                    require(receipt["checkpoint_sha256"][key] == digest, str(parent))
        else:
            require(receipt is not None, f"Terminal tensor awaits training receipt: {final}")
            assert receipt is not None
            require(receipt["splits"] == split_ids, str(receipt_path))
            require(receipt["command"] == _command("vision_loss", training_commit, seed), str(root))
            last = original / "sparse-300k/full/last.pt"
            control = original / "sparse-500k/full/final.pt"
            require(
                Path(receipt["source"]) == last and Path(receipt["reference"]) == control, str(root)
            )
            ledger.bind(last, receipt["source_sha256"])
            ledger.bind(control, receipt["reference_sha256"])
            require(ledger[str(control)] == reported[str(control)], str(control))
            archive_fork = ledger.record(control.parent / "fork.json")
            require(archive_fork["source_sha256"] == ledger[str(last)], str(last))
            ledger.bind(final, receipt["sha256"])
            state["control_replay_matched"] = receipt["control_replay_matched"]
        fork = ledger.record(directory / "fork.json")
        require(
            fork["source_step"] == 300_000
            and Path(fork["source"]).resolve() == last.resolve()
            and fork["source_sha256"] == ledger[str(last)],
            str(directory),
        )
        summary = ledger.record(directory / "final.json")
        if receipt is not None:
            original_summary = receipt["validation"]
            require(
                summary == (original_summary[arm] if staged else original_summary),
                str(final),
            )
        payload = load_checkpoint(final)
        cfg, meta = payload["config"], payload["meta"]
        require(
            summary["step"] == cfg["steps"] == 500_000
            and cfg["stage"] == meta["stage"] == arm
            and cfg["seed"] == meta["seed"] == seed
            and Path(cfg["data_dir"]).resolve() == DATA.resolve(),
            str(final),
        )
        selected[arm] = final
    return selected, state


def controls(
    seed: int, split_ids: dict[str, str], reported: dict[str, str], ledger: Inputs
) -> dict[str, dict[str, Any]]:
    """Bind the original same-seed query receipts, calibration units and model identities."""
    records: dict[str, dict[str, Any]] = {}
    for query, revision in CONTROLS.items():
        directory = OUTPUT / f"{query}-paired-s{seed}"
        ledger.note(
            directory / "NOTE.md",
            f"Checkout {revision}\n",
            f"ICML_review.code.run {query} {seed} paired {directory}",
            *split_ids.values(),
            ledger.bind(DATA / "manifest.json"),
        )
        ledger.bind(directory / "exit_code")
        require((directory / "exit_code").read_text().strip() == "0", str(directory))
        record = ledger.record(directory / "metrics.json")
        require(
            record["seed"] == seed
            and record["data_sha256"] == {s: split_ids[s] for s in ("calibration", "validation")},
            str(directory),
        )
        for arm, path in chart.endpoints(seed).items():
            relative = str(path.relative_to(Path.cwd()))
            ledger.bind(Path(relative), reported[relative])
            control = record["arms"][arm]
            require(
                control["checkpoint"] == relative
                and control["checkpoint_sha256"] == reported[relative],
                relative,
            )
        records[query] = record
    return records


def paired(
    query: str, result: dict[str, Any], reference: dict[str, Any], arm: str
) -> dict[str, dict[str, float]]:
    """Compare errors only after checking identical simulator targets and evaluation cohorts."""
    shared = [
        "seed",
        "data_sha256",
        "validation_groups",
        "validation_contexts",
        "calibration_groups",
    ]
    require(all(result[k] == reference[k] for k in shared), "Different evaluation cohort")
    truths = ("hold_grids", "truth_slopes_held_by_outcome") if query == "chart" else ("truth",)
    for key in truths:
        torch.testing.assert_close(result[key], reference[key], atol=1e-11, rtol=1e-10)
    scores: dict[str, dict[str, float]] = {}
    for interface in ("scalar", "affine"):
        changed, control = (r["arms"][arm][interface] for r in (result, reference))
        if query == "chart":
            target = reference["truth_slopes_held_by_outcome"][4][6]
            new, old = (abs(row["u2a_to_u2b"] - target) for row in (changed, control))
        else:
            new, old = (
                row["mean_across_beads"]["held"]["relative_error"] for row in (changed, control)
            )
        scores[interface] = {
            "trial_error": new,
            "control_error": old,
            "trial_minus_control": new - old,
        }
    return scores


def run(commit: str, seed: int, out: Path, *, one_head: bool = False) -> None:
    """Evaluate the available endpoints under the recorded protocol."""
    _head(commit)
    subprocess.run(["sha256sum", "-c", "--quiet", MANIFEST.name], cwd="outputs", check=True)
    reported = checkpoint_hashes()
    ledger = Inputs()
    ledger.bind(MANIFEST)
    manifest = ledger.record(DATA / "manifest.json")
    split_ids = {s: manifest["splits"][s]["sha256"] for s in ("train", "calibration", "validation")}
    for split, digest in split_ids.items():
        require(digest == reported[str(DATA / f"{split}.pt")], split)
    command = _command("changed_queries", commit, seed) + (" --one-head" if one_head else "")
    training_commit = commit if one_head else TRAINING
    title = "One-head vision query endpoints" if one_head else "Changed vision query endpoints"
    declared(
        out / "NOTE.md",
        f"# {title}, seed {seed}\n\n{PROTOCOL}\n\n"
        f"Checkout {commit}\nTraining checkout {training_commit}\nData manifest SHA-256 "
        f"{ledger[str(DATA / 'manifest.json')]}\n\n    {command}\n",
    )
    reference = controls(seed, split_ids, reported, ledger)
    methods: dict[str, Any] = {}
    gpu_checked = False
    for method in (ONE_HEAD,) if one_head else tuple(ARMS):
        paths, state = sources(
            method, seed, split_ids, reported, ledger, training_commit=training_commit
        )
        methods[method] = state | {"evaluated_arms": list(paths), "queries": {}}
        if not paths:
            continue
        if not gpu_checked:
            _idle_gpu()
            gpu_checked = True
        for query, evaluate in (("chart", chart.run), ("kicks", kicks.run)):
            _head(commit)
            result = evaluate(seed, paths)
            deltas: dict[str, Any] = {}
            for arm, path in paths.items():
                require(result["arms"][arm]["checkpoint_sha256"] == ledger[str(path)], str(path))
                deltas[arm] = paired(query, result, reference[query], arm)
            methods[method]["queries"][query] = {"measurement": result, "paired": deltas}
    _head(commit)
    write_json(
        out / "metrics.json",
        {
            "verified": True,
            "seed": seed,
            "commit": commit,
            "command": command,
            "input_sha256": ledger,
            "methods": methods,
            "data_sha256": split_ids,
            "note_sha256": sha(out / "NOTE.md"),
        },
    )


def main() -> None:
    """Run on a free GPU from the checkout under a per-seed lock."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("commit")
    cli.add_argument("seed", type=int, choices=range(5))
    cli.add_argument(
        "--one-head", action="store_true", help="measure only the one-head training route"
    )
    args = cli.parse_args()
    prefix = "one-head-queries" if args.one_head else "changed-queries"
    out = OUTPUT / f"{prefix}-s{args.seed}"
    with file_lock(out / ".lock"):
        run(args.commit, args.seed, out, one_head=args.one_head)


if __name__ == "__main__":
    main()
