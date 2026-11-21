"""Rebuild a synthetic warm start from its first acquisition and compare both sparse arms."""

import argparse
import math
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from diagnostics.masked_mlp_learned import run as runner
from diagnostics.masked_mlp_learned.selection import select
from diagnostics.shared.io import read_json, sha
from scjepa.experiments.storage import file_lock, write_json

DATA = Path("outputs/data/masked-mlp-gelu-20261101")
PARENT = Path("outputs/diagnostics/masked-mlp-gelu-20261101")
COMPARISON = Path("outputs/diagnostics/masked-mlp-gelu-350k-20261102/paper/summary.json")
PRIVATE = Path("ICML_review/outputs")
PYTHON = ".venv/bin/python"
MODULE = "diagnostics.masked_mlp_learned"
STAGES = ("project", "adapt", "baseline-300k", "full-300k", "baseline-350k", "full-350k")
TRAIN_STEPS = {
    "adapt": 50_000,
    **{f"{a}-{n}k": n * 1000 for a in ("baseline", "full") for n in (300, 350)},
}
GATE = (
    "The original TRAIN-loss acquisition admission rule must pass. At matched sparse budgets "
    "require full validation MCC within 0.01 of the reported full seed or higher, at least "
    "80% of the reported paired MCC gain, no larger total full SHD, full validation NMSE "
    "at most 1.10 times its reference and minimum full parameter R² at least 0.80. "
    "Passing seed zero licenses all remaining seeds. Failure keeps the cheaper protocol "
    "unestablished. A five-seed result reports every outcome."
)


def _head(expected: str) -> None:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise RuntimeError(f"Expected commit {expected}, found {actual}")


def _idle_gpu() -> None:
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    )
    free = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    if apps.strip() or int(free.strip()) < 6144:
        raise RuntimeError("GPU occupied or has insufficient available memory")


def _note(commit: str, data_id: str, command: list[str], first_admitted: bool = False) -> str:
    if first_admitted:
        return (
            "# First-admitted acquisition comparison\n\n"
            "Hypothesis: sequential retry after training-only admission failure recovers a "
            "usable starting point at lower acquisition cost. Follow the original acquisition "
            "order, stopping at the first eligible trailing TRAIN forecast score, at most sixteen "
            "attempts. Hold joint weights, stage budgets, data, seed, numeric tolerance and "
            "endpoint criteria fixed. An identical previously assessed starting point inherits "
            "its verified outcome, including any failed endpoint checks. When all candidates "
            "fail admission, end without sparse training.\n\n"
            f"{GATE}\n\nGit commit: {commit}\nData identity: {data_id}\n\n"
            f"    PYTHONPATH=$PWD:$PWD/src {shlex.join(command)}\n"
        )
    return (
        "# Single-acquisition synthetic comparison\n\n"
        "Hypothesis: one standard-budget acquisition preserves the reported recovery. "
        "Only the number of candidate acquisitions changes. Use historical attempt zero, "
        "all original per-stage budgets, paired seeds and the reported numeric tolerance.\n\n"
        f"{GATE}\n\nGit commit: {commit}\nData identity: {data_id}\n\n"
        "From the server checkout with GPU 0 visible and the project venv active:\n\n"
        f"    PYTHONPATH=$PWD:$PWD/src {shlex.join(command)}\n"
    )


def _stage(
    directory: Path, args: list[str], commit: str, data_id: str, first_admitted: bool = False
) -> None:
    _head(commit)
    directory.mkdir(parents=True, exist_ok=True)
    results = directory / "results"
    args = [PYTHON, "-m", *args]
    note = _note(commit, data_id, args, first_admitted)
    path = directory / "NOTE.md"
    if path.exists() and path.read_text() != note:
        raise RuntimeError(f"Different declared command in {directory}")
    path.write_text(note)
    status = directory / "exit_code"
    if (results / "completion.json").exists():
        if (
            args[2].endswith(".continuation")
            and not (results / "continuation-completion.json").exists()
        ):
            raise RuntimeError("Continuation lacks its parent verification receipt")
        runner.verify_completion(results, require_current_source=True)
        status.write_text("0\n")
        return
    if (results / "last.pt").exists() and "project" not in args:
        args.append("--resume")
        (directory / "resume-command.txt").write_text(shlex.join(args) + "\n")
    _idle_gpu()
    with (directory / "log.txt").open("a") as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=False)
    status.write_text(f"{result.returncode}\n")
    result.check_returncode()
    _head(commit)
    runner.verify_completion(results, require_current_source=True)


def decision(
    measured: dict[str, dict[str, float]], reference: dict[str, dict[str, float]]
) -> dict[str, bool]:
    """Apply the predeclared paired validation requirements to a fixed endpoint."""

    def gain(rows: dict[str, dict[str, float]]) -> float:
        return rows["full"]["state_mcc"] - rows["baseline"]["state_mcc"]

    def shd(full: dict[str, float]) -> float:
        return sum(
            full[key]
            for key in (
                "masks/state_aligned_shd",
                "masks/action_aligned_shd",
                "masks/parameter_code_aligned_shd",
            )
        )

    full, old = measured["full"], reference["full"]
    return {
        "full_mcc": full["state_mcc"] >= old["state_mcc"] - 0.01,
        "paired_mcc_gain": gain(measured) >= 0.80 * gain(reference),
        "full_shd": shd(full) <= shd(old),
        "full_nmse": full["normalized_prediction_mse"] <= 1.10 * old["normalized_prediction_mse"],
        "parameter_retention": full["code_r2_min"] >= 0.80,
    }


def _admission(seed: int, commit: str, data_id: str, first_admitted: bool) -> dict[str, Any]:
    """Check candidates in initialization order until the fixed training-only cutoff passes."""
    path = PARENT / f"selection-s{seed}.json"
    historical = read_json(path)
    originals = {row["acquisition_seed"]: row for row in historical["candidates"]}
    attempts: list[dict[str, Any]] = []
    for index in range(16 if first_admitted else 1):
        directory = PARENT / f"acquisition-{index}-s{seed}/results"
        original = originals[1000 + 100 * index + seed]
        if original["completion_sha256"] != sha(directory / "completion.json"):
            raise RuntimeError("Acquisition has changed since the original selection")
        admission = select(
            [directory], maximum_loss=historical["maximum_loss"], window=historical["window"]
        )
        source = runner.verify_completion(directory, kind="train", require_current_source=True)
        if (
            admission["common"] != historical["common"]
            or source["data_identity"] != data_id
            or original["identity"] != source["identity"]
            or admission["selected"]["acquisition_seed"] != original["acquisition_seed"]
            or not math.isclose(
                admission["selected"]["train_prediction_mean"], original["train_prediction_mean"]
            )
        ):
            raise RuntimeError("Acquisition inputs or TRAIN score differ from the original receipt")
        record = admission | {
            "commit": commit,
            "data_identity": data_id,
            "checkpoint_sha256": source["hashes"]["last.pt"],
            "completion_sha256": original["completion_sha256"],
        }
        if not first_admitted:
            return record
        attempts.append(
            admission["selected"]
            | {
                "index": index,
                "admitted": admission["admitted"],
                "selection_identity": admission["identity"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "history_sha256": source["hashes"]["history.jsonl"],
            }
        )
        if admission["admitted"] or index == 15:
            return {
                "schema": "first-admitted-acquisition-v1",
                "policy": "first eligible original candidate in index order",
                "attempts": attempts,
                "selected_index": index if admission["admitted"] else None,
                "admitted": admission["admitted"],
                "selected": attempts[-1] if admission["admitted"] else None,
                "data_identity": data_id,
                "commit": commit,
                "common": historical["common"],
                "window": historical["window"],
                "maximum_loss": historical["maximum_loss"],
                "checkpoint_sha256": record["checkpoint_sha256"] if admission["admitted"] else None,
                "historical_selection_sha256": sha(path),
                "selection_used_parameter_labels": False,
            }
    raise RuntimeError("Acquisition sequence is empty")


def _reuse_planned(root: Path, seed: int, index: int, phase: str, commit: str) -> bool:
    """Freeze the reuse decision before any stages start, including an independent fresh run."""
    path = root / "execution.json"
    receipt = sha(root / "admission.json")
    if not path.exists():
        if phase != "prepare":
            raise RuntimeError("Prepare must select the execution route first")
        previous = PRIVATE / f"single-acquisition-s{seed}/metrics.json"
        write_json(
            path,
            {
                "commit": commit,
                "admission_sha256": receipt,
                "selected_index": index,
                "reuse_prior_result": index == 0 and previous.is_file(),
            },
        )
    plan = read_json(path)
    if (
        plan["commit"] != commit
        or plan["admission_sha256"] != receipt
        or plan["selected_index"] != index
    ):
        raise RuntimeError("The stored execution route belongs to a different selection")
    return bool(plan["reuse_prior_result"])


def _carry_forward(
    root: Path, seed: int, commit: str, admission: dict[str, Any], tau: float
) -> None:
    """Authenticate the completed identical route and carry its whole paired result forward."""
    from diagnostics.masked_mlp_oracle.paper_analysis import load_run

    old = PRIVATE / f"single-acquisition-s{seed}"
    metrics = read_json(old / "metrics.json")
    original = read_json(old / "admission.json")
    summary = read_json(COMPARISON)["per_run"]
    reference = {a: summary[a][str(seed)] for a in ("baseline", "full")}
    checks = decision(metrics["validation"], reference)
    if (
        not original["admitted"]
        or original["checkpoint_sha256"] != admission["checkpoint_sha256"]
        or original["selected"]["identity"] != admission["selected"]["identity"]
        or original["commit"] != metrics["commit"]
        or original["data_identity"] != admission["data_identity"]
        or metrics["seed"] != seed
        or metrics["data_identity"] != admission["data_identity"]
        or metrics["tau"] != tau
        or metrics["acquisition_receipt_sha256"] != sha(old / "admission.json")
        or metrics["reference_summary_sha256"] != sha(COMPARISON)
        or metrics["reference_validation"] != reference
        or metrics["checks"] != checks
        or metrics["passed"] != all(checks.values())
    ):
        raise RuntimeError("Previous result differs in inputs, tolerance or endpoint criteria")
    stages = {}
    for stage in STAGES:
        path = old / stage / "results"
        record = runner.verify_completion(path, require_current_source=True)
        if (
            record["data_identity"] != admission["data_identity"]
            or record["seed"] != seed
            or record["hashes"]["last.pt"] != metrics["checkpoint_sha256"][stage]
        ):
            raise RuntimeError(f"Previous endpoint changed at {path}")
        if stage in TRAIN_STEPS and (
            record["steps"] != TRAIN_STEPS[stage]
            or record["successful_updates"] != TRAIN_STEPS[stage]
        ):
            raise RuntimeError(f"Previous update budget differs at {path}")
        stages[stage] = record
    if stages["project"]["source"]["checkpoint_sha256"] != admission["checkpoint_sha256"]:
        raise RuntimeError("Previous projection has a different acquisition")
    if stages["adapt"]["adaptation_source"]["identity"] != stages["project"]["identity"]:
        raise RuntimeError("Previous adaptation has a different projection")
    for arm in reference:
        parent, end = f"{arm}-300k", f"{arm}-350k"
        folder = old / end / "results"
        continuation = read_json(folder / "continuation-completion.json")
        if (
            stages[parent]["tau_prediction"] != tau
            or stages[end]["tau_prediction"] != tau
            or stages[parent]["dense_endpoint"]["identity"] != stages["adapt"]["identity"]
            or continuation["parent_checkpoint_sha256"] != stages[parent]["hashes"]["last.pt"]
            or continuation["checkpoint_sha256"] != stages[end]["hashes"]["last.pt"]
            or load_run(folder, arm, seed).metrics != metrics["validation"][arm]
        ):
            raise RuntimeError(f"Previous {arm} training or assessment differs")
    _head(commit)
    write_json(
        root / "metrics.json",
        metrics
        | {
            "commit": commit,
            "policy": "first_admitted",
            "source_commit": metrics["commit"],
            "source_run": str(old),
            "source_metrics_sha256": sha(old / "metrics.json"),
            "source_admission_sha256": sha(old / "admission.json"),
            "acquisition_receipt_sha256": sha(root / "admission.json"),
            "reused": True,
            "selected_index": 0,
        },
    )


def run(commit: str, seed: int, phase: str, *, first_admitted: bool = False) -> None:
    """Run a CPU assessment or a GPU phase at fixed HEAD, with separate output locks."""
    _head(commit)
    policy = "single-admitted" if first_admitted else "single-acquisition"
    root = PRIVATE / f"{policy}-s{seed}"
    pilot = PRIVATE / f"{policy}-s0/metrics.json"
    if seed:
        if not pilot.is_file():
            raise RuntimeError("Await the seed-zero pilot assessment")
        gate = read_json(pilot)
        if gate["commit"] != commit or not gate["passed"]:
            raise RuntimeError("Expansion requires a passing same-commit pilot")
    data_id = read_json(DATA / "completion.json")["identity"]
    calibration = PARENT / "calibration"
    project = root / "project"
    adapt = root / "adapt"
    receipt = root / "admission.json"
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / f".{phase}.lock"):
        args = [PYTHON, "-m", "ICML_review.code.single_acquisition", commit, str(seed), phase]
        if first_admitted:
            args.append("--first-admitted")
        stage_note = _note(commit, data_id, args, first_admitted)
        if phase == "prepare":
            note = root / "NOTE.md"
            if note.exists() and note.read_text() != stage_note:
                raise RuntimeError("The run already has a different declaration")
            note.write_text(stage_note)
            subprocess.run(
                ["sha256sum", "-c", "--quiet", "reported-checkpoints.sha256"],
                cwd="outputs",
                check=True,
            )
            write_json(receipt, _admission(seed, commit, data_id, first_admitted))
        admission = read_json(receipt)
        if admission["commit"] != commit or admission["data_identity"] != data_id:
            raise RuntimeError("Acquisition admission belongs to a different run")
        if not admission["admitted"]:
            failure: dict[str, Any] = {
                "commit": commit,
                "seed": seed,
                "passed": False,
                "reason": "train_admission",
            }
            if first_admitted:
                failure.update(
                    data_identity=data_id,
                    acquisition_receipt_sha256=sha(receipt),
                    policy="first_admitted",
                    attempted=len(admission["attempts"]),
                )
            _head(commit)
            write_json(root / "metrics.json", failure)
            return
        index = int(admission["selected_index"]) if first_admitted else 0
        acquisition = PARENT / f"acquisition-{index}-s{seed}/results"
        if first_admitted and _reuse_planned(root, seed, index, phase, commit):
            if phase == "prepare":
                _carry_forward(
                    root,
                    seed,
                    commit,
                    admission,
                    read_json(calibration / "calibration.json")["tau_prediction"],
                )
            elif not (root / "metrics.json").is_file():
                raise RuntimeError("Prepare must verify the prior endpoint first")
            _head(commit)
            return
        wandb_group = "icml-review-" + policy
        wandb = read_json(acquisition / "wandb.json")
        flags = [
            "--device",
            "cuda",
            "--predictor-activation",
            "gelu",
            "--wandb-mode",
            "online",
            "--wandb-project",
            wandb["project"],
            "--wandb-entity",
            wandb["entity"],
            "--wandb-group",
            wandb_group,
        ]
        if phase == "prepare":
            _stage(
                project,
                [
                    f"{MODULE}.run",
                    "project",
                    "--data-dir",
                    str(DATA),
                    "--dense-dir",
                    str(acquisition),
                    "--output-dir",
                    str(project / "results"),
                    "--device",
                    "cuda",
                ],
                commit,
                data_id,
                first_admitted,
            )
            _stage(
                adapt,
                [
                    f"{MODULE}.run",
                    "train",
                    "--data-dir",
                    str(DATA),
                    "--output-dir",
                    str(adapt / "results"),
                    "--condition",
                    "dense",
                    "--seed",
                    str(seed),
                    "--steps",
                    "50000",
                    "--adapt-from",
                    str(project / "results"),
                    *flags,
                ],
                commit,
                data_id,
                first_admitted,
            )
        elif phase in ("full", "baseline"):
            first, final = root / f"{phase}-300k", root / f"{phase}-350k"
            _stage(
                first,
                [
                    f"{MODULE}.budget_control",
                    "--data-dir",
                    str(DATA),
                    "--output-dir",
                    str(first / "results"),
                    "--condition",
                    phase,
                    "--seed",
                    str(seed),
                    "--init-dir",
                    str(adapt / "results"),
                    "--calibration-dir",
                    str(calibration),
                    *flags,
                ],
                commit,
                data_id,
                first_admitted,
            )
            _stage(
                final,
                [
                    f"{MODULE}.continuation",
                    "--source-dir",
                    str(first / "results"),
                    "--output-dir",
                    str(final / "results"),
                    "--additional-steps",
                    "50000",
                    "--device",
                    "cuda",
                    "--threads",
                    "1",
                    "--wandb-group",
                    wandb_group,
                ],
                commit,
                data_id,
                first_admitted,
            )
        else:
            from diagnostics.masked_mlp_oracle.paper_analysis import load_run

            assess = root / "assess"
            assess.mkdir(exist_ok=True)
            (assess / "NOTE.md").write_text(stage_note)
            original_summary = read_json(COMPARISON)["per_run"]
            reference = {a: original_summary[a][str(seed)] for a in ("baseline", "full")}
            measured = {a: load_run(root / f"{a}-350k/results", a, seed).metrics for a in reference}
            checks = decision(measured, reference)
            hashes = {s: sha(root / s / "results/last.pt") for s in STAGES}
            write_json(
                root / "metrics.json",
                {
                    "commit": commit,
                    "seed": seed,
                    "data_identity": data_id,
                    "acquisition_receipt_sha256": sha(receipt),
                    "reference_summary_sha256": sha(COMPARISON),
                    "tau": read_json(calibration / "calibration.json")["tau_prediction"],
                    "reference_validation": reference,
                    "validation": measured,
                    "checks": checks,
                    "passed": all(checks.values()),
                    "checkpoint_sha256": hashes,
                    **(
                        {"policy": "first_admitted", "selected_index": index, "reused": False}
                        if first_admitted
                        else {}
                    ),
                },
            )
        _head(commit)
        if (
            phase in ("prepare", "assess")
            and sha(acquisition / "last.pt") != admission["checkpoint_sha256"]
        ):
            raise RuntimeError("The archived acquisition checkpoint changed")


def main() -> None:
    """Dispatch one phase from the server checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("seed", type=int, choices=range(5))
    parser.add_argument("phase", choices=("prepare", "baseline", "full", "assess"))
    parser.add_argument("--first-admitted", action="store_true")
    args = parser.parse_args()
    os.environ["PYTHONPATH"] = os.pathsep.join((str(Path.cwd()), str(Path.cwd() / "src")))
    run(args.commit, args.seed, args.phase, first_admitted=args.first_admitted)


if __name__ == "__main__":
    main()
