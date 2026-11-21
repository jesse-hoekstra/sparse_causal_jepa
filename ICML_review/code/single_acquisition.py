"""Rebuild a synthetic warm start from its first acquisition and compare both sparse arms."""

import argparse
import math
import os
import shlex
import subprocess
from pathlib import Path

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


def _note(commit: str, data_id: str, command: list[str]) -> str:
    return (
        "# Single-acquisition synthetic comparison\n\n"
        "Hypothesis: one standard-budget acquisition preserves the reported recovery. "
        "Only the number of candidate acquisitions changes. Use historical attempt zero, "
        "all original per-stage budgets, paired seeds and the reported numeric tolerance.\n\n"
        f"{GATE}\n\nGit commit: {commit}\nData identity: {data_id}\n\n"
        "From the server checkout with GPU 0 visible and the project venv active:\n\n"
        f"    PYTHONPATH=$PWD:$PWD/src {shlex.join(command)}\n"
    )


def _stage(directory: Path, args: list[str], commit: str, data_id: str) -> None:
    _head(commit)
    directory.mkdir(parents=True, exist_ok=True)
    results = directory / "results"
    args = [PYTHON, "-m", *args]
    note = _note(commit, data_id, args)
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


def run(commit: str, seed: int, phase: str) -> None:
    """Run a CPU assessment or a GPU phase at fixed HEAD, with separate output locks."""
    _head(commit)
    root = PRIVATE / f"single-acquisition-s{seed}"
    pilot = PRIVATE / "single-acquisition-s0/metrics.json"
    if seed:
        if not pilot.is_file():
            raise RuntimeError("Await the seed-zero pilot assessment")
        gate = read_json(pilot)
        if gate["commit"] != commit or not gate["passed"]:
            raise RuntimeError("Expansion requires a passing same-commit pilot")
    data_id = read_json(DATA / "completion.json")["identity"]
    acquisition = PARENT / f"acquisition-0-s{seed}/results"
    calibration = PARENT / "calibration"
    project = root / "project"
    adapt = root / "adapt"
    receipt = root / "admission.json"
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / f".{phase}.lock"):
        args = [PYTHON, "-m", "ICML_review.code.single_acquisition", commit, str(seed), phase]
        stage_note = _note(commit, data_id, args)
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
            historical = read_json(PARENT / f"selection-s{seed}.json")
            original = next(
                row for row in historical["candidates"] if row["acquisition_seed"] == 1000 + seed
            )
            if original["completion_sha256"] != sha(acquisition / "completion.json"):
                raise RuntimeError("First acquisition has changed since the original selection")
            admission = select(
                [acquisition], maximum_loss=historical["maximum_loss"], window=historical["window"]
            )
            source = runner.verify_completion(
                acquisition, kind="train", require_current_source=True
            )
            if source["data_identity"] != data_id or original["identity"] != source["identity"]:
                raise RuntimeError("First acquisition cohort or model identity mismatch")
            if not math.isclose(
                admission["selected"]["train_prediction_mean"], original["train_prediction_mean"]
            ):
                raise RuntimeError("First acquisition TRAIN loss differs from its original receipt")
            write_json(
                receipt,
                admission
                | {
                    "commit": commit,
                    "checkpoint_sha256": source["hashes"]["last.pt"],
                    "completion_sha256": original["completion_sha256"],
                    "data_identity": data_id,
                },
            )
        admission = read_json(receipt)
        if admission["commit"] != commit or admission["data_identity"] != data_id:
            raise RuntimeError("Acquisition admission belongs to a different run")
        if not admission["admitted"]:
            write_json(
                root / "metrics.json",
                {"commit": commit, "seed": seed, "passed": False, "reason": "train_admission"},
            )
            return
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
            "icml-review-single-acquisition",
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
                    "icml-review-single-acquisition",
                ],
                commit,
                data_id,
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
            names = ["project", "adapt", "baseline-300k", "full-300k", "baseline-350k", "full-350k"]
            hashes = {s: sha(root / s / "results/last.pt") for s in names}
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
    args = parser.parse_args()
    os.environ["PYTHONPATH"] = os.pathsep.join((str(Path.cwd()), str(Path.cwd() / "src")))
    run(args.commit, args.seed, args.phase)


if __name__ == "__main__":
    main()
