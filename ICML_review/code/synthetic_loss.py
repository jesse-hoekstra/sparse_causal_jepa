"""Test a scale-normalized within-system loss on the first synthetic acquisition."""

import argparse
import math
import shlex
import subprocess
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.masked_mlp_learned import run as runner
from diagnostics.masked_mlp_learned.budget_control import BudgetConfig
from diagnostics.masked_mlp_learned.continuation import continue_training
from diagnostics.masked_mlp_learned.selection import training_score
from diagnostics.shared.io import read_json, sha
from scjepa.experiments.storage import file_lock, write_json

from .single_acquisition import COMPARISON, DATA, PARENT, PRIVATE, PYTHON, decision
from .single_acquisition import _head as check_head  # pyright: ignore[reportPrivateUsage]
from .single_acquisition import _idle_gpu as idle_gpu  # pyright: ignore[reportPrivateUsage]
from .synthetic_loss_math import (
    AcquisitionConfig,
    acquisition_runtime,
    code_quality,
    group_moments,
    paired_intervals,
)

PILOT = 3
MODULE = "ICML_review.code.synthetic_loss"
CHILD_STAGES = {
    "prepare": ("project", "adapt"),
    "full": ("full-300k", "full-350k"),
    "baseline": ("baseline-300k", "baseline-350k"),
}
GATES = (
    "Seed 3 is a retrospectively selected stress pilot. Acquire runs "
    "one matched 100k a0. Require original TRAIN admission, CAL mean within/total <=0.80 times "
    "old a0, a positive CAL sixth shared eigenvalue / total trace >=0.80 times old a0, validation "
    "minimum single-context parameter R² >=0.80, and validation raw MSE <=1.10 times the original "
    "first eligible acquisition. Physical R² is an explicitly labelled post-training check. "
    "An acquisition failure ends this seed before sparse work. Passing qualifies projection, "
    "adaptation and both full-budget forks. Assess the downstream gate against each of the "
    "reported best-of-sixteen and the first-admitted policies. Only passing both comparisons "
    "on the pilot qualifies other seeds. A failed endpoint leaves the simpler procedure unproven. "
)


def _note(path: Path, commit: str, seed: int, phase: str, data: str, scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = shlex.join([PYTHON, "-m", MODULE, commit, str(seed), phase])
    note = (
        "# Controlled acquisition consistency experiment\n\n"
        "Hypothesis: directly reducing the fraction of each code coordinate's variation due to "
        "episode choice improves the first acquisition without weakening transfer to the two "
        "sparse arms. Only the frozen-state acquisition loss changes. Its starting tensors come "
        "from the original a0 initialization. Baselines are original a0 at acquisition, the "
        "first-eligible policy and the reported final models. Original group minibatches, AdamW, "
        "prediction loss, TF32-off training, SIGReg and stage lengths are preserved.\n\n"
        f"Acquisition adds {scale:.17g} times the mean within/total population variance ratio. "
        "The coefficient is the frozen joint initializer's raw calibration forecast MSE. "
        "The variance decomposition T = W + between shows that the penalty is the fraction of "
        "variation occurring within groups. It is invariant to a nonzero rescaling or offset of "
        "an individual coordinate. A constant coordinate receives penalty one. Rotating or "
        "duplicating coordinates can alter the criterion. Calibration shared-rank, raw total "
        "spread and independent validation retention diagnose such failures.\n\n"
        f"{GATES} Against each reference require full MCC no more than 0.01 lower, at least "
        "80 percent of the paired MCC gain, no larger total full SHD, full validation NMSE "
        "at most 1.10 times reference, and minimum full parameter R² at least 0.80.\n\n"
        f"Git commit: {commit}\nCohort identity: {data}\nSeed: {seed}\n\n"
        f"    PYTHONPATH=$PWD:$PWD/src {cmd}\n"
    )
    if path.is_file() and path.read_text() != note:
        raise RuntimeError(f"Incompatible declaration at {path}")
    path.write_text(note)


def _eligible(seed: int) -> tuple[int, Path]:
    history = read_json(PARENT / f"selection-s{seed}.json")
    ordered = sorted(history["candidates"], key=lambda row: row["acquisition_seed"])
    if [r["acquisition_seed"] for r in ordered] != [1000 + 100 * a + seed for a in range(16)]:
        raise RuntimeError("Unexpected acquisition identifiers")
    for index, row in enumerate(ordered):
        if row["train_prediction_mean"] <= history["maximum_loss"]:
            path = PARENT / f"acquisition-{index}-s{seed}/results"
            if sha(path / "completion.json") != row["completion_sha256"]:
                raise RuntimeError("Eligible acquisition receipt changed")
            return index, path
    raise RuntimeError("Historical first-eligible comparator is undefined")


def _measures(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    result: dict[str, Any] = {
        "checkpoint_sha256": read_json(path / "completion.json")["hashes"]["last.pt"]
    }
    moments: dict[str, torch.Tensor] = {}
    for split in ("calibration", "validation"):
        file = path / f"{split}.pt"
        saved = cast(dict[str, Any], torch.load(file, map_location="cpu", weights_only=False))
        code = cast(torch.Tensor, saved["samples"]["codes"])
        moments[split] = group_moments(saved["samples"])
        result[split] = code_quality(code) | {
            "groups": code.shape[0],
            "prediction": saved["metrics"]["prediction"],
            "labelled_min_parameter_r2": saved["metrics"]["code_r2_min"],
            "split_sha256": sha(file),
        }
    return result, moments


def _config(path: Path, output: Path) -> dict[str, Any]:
    values = read_json(path / "config.json")["config"]
    args = {f.name: values[f.name] for f in fields(runner.TrainConfig) if f.init}
    return args | {
        "output_dir": str(output),
        "data_dir": str(DATA),
        "wandb_group": "icml-review-synthetic-loss",
    }


def _gate(root: Path, seed: int, commit: str, data_id: str, scale: float) -> None:
    current = root / "acquire/results"
    baseline = PARENT / f"acquisition-0-s{seed}/results"
    history = read_json(PARENT / f"selection-s{seed}.json")
    eligible_index, eligible_path = _eligible(seed)
    a0 = runner.verify_completion(baseline, kind="train", require_current_source=True)
    eligible = runner.verify_completion(eligible_path, kind="train", require_current_source=True)
    trained = runner.verify_completion(current, kind="train")
    if trained["initial_model_identity"] != a0["initial_model_identity"] or (
        trained["data_identity"] != data_id or eligible["data_identity"] != data_id
    ):
        raise RuntimeError("Initial tensors or comparison cohort differ")
    treatment, treatment_moments = _measures(current)
    control, control_moments = _measures(baseline)
    eligible_prediction = read_json(eligible_path / "validation.json")["metrics"]["prediction"]
    log_every = read_json(current / "config.json")["config"]["log_every"]
    scores = {
        n: training_score(p, steps=trained["steps"], window=history["window"], log_every=log_every)
        for n, p in (("treatment", current), ("a0", baseline), ("first_eligible", eligible_path))
    }
    old, new = control["calibration"], treatment["calibration"]
    checks = {
        "training_admission": scores["treatment"] <= history["maximum_loss"],
        "calibration_within_share": new["within_share"] <= 0.8 * old["within_share"],
        "calibration_shared_rank": new["sixth_share"] > 0
        and new["sixth_share"] >= 0.8 * old["sixth_share"],
        "validation_parameter_retention": treatment["validation"]["labelled_min_parameter_r2"]
        >= 0.8,
        "validation_prediction": treatment["validation"]["prediction"] <= 1.1 * eligible_prediction,
    }
    receipt = {
        "commit": commit,
        "seed": seed,
        "data_identity": data_id,
        "passed": all(checks.values()),
        "checks": checks,
        "coefficient": scale,
        "matched_original_a0": control,
        "paired_system_bootstrap": {
            s: paired_intervals(treatment_moments[s], control_moments[s])
            for s in ("calibration", "validation")
        },
        "treatment": treatment,
        "origin_initial_checkpoint_sha256": a0["hashes"]["initial.pt"],
        "initial_model_identity": trained["initial_model_identity"],
        "joint_calibration_sha256": sha(PARENT / f"states-s{seed}/results/calibration.json"),
        "acquisition_source_identity": trained["code_identity"],
        "historical_selection_sha256": sha(PARENT / f"selection-s{seed}.json"),
        "window": history["window"],
        "maximum_loss": history["maximum_loss"],
        "training_scores": scores,
        "first_eligible_index": eligible_index,
        "first_eligible_validation_prediction": eligible_prediction,
        "first_eligible_checkpoint_sha256": eligible["hashes"]["last.pt"],
        "first_eligible_validation_sha256": sha(eligible_path / "validation.json"),
    }
    check_head(commit)
    write_json(root / "acquisition-gate.json", receipt)
    if not receipt["passed"]:
        write_json(
            root / "metrics.json",
            receipt | {"reason": "acquisition_gate", "downstream_trained": False},
        )


def _train_stage(root: Path, seed: int, stage: str, coefficient: float) -> None:
    original = PARENT / f"acquisition-0-s{seed}/results"
    out = root / stage / "results"
    finished = out / "completion.json"
    resume = (out / "last.pt").is_file()
    continuation_missing = stage.endswith("350k") and not (out / "continuation.json").is_file()
    if (
        out.exists()
        and not finished.exists()
        and (stage == "project" or not resume or continuation_missing)
    ):
        # A snapshot interrupted before its restart receipt cannot be resumed honestly.
        out.rename(out.parent / f"incomplete-{time.time_ns()}")
        resume = False
    if stage in ("acquire", "project"):
        with acquisition_runtime(original, coefficient, restore_initial=stage == "acquire"):
            if finished.is_file():
                runner.verify_completion(out, require_current_source=True)
                return
            idle_gpu()
            if stage == "acquire":
                args = _config(original, out)
                args["states_from"] = str(PARENT / f"states-s{seed}/results")
                config = AcquisitionConfig(
                    **args,
                    joint_calibration_prediction=coefficient,
                    initialization_sha256=sha(original / "initial.pt"),
                )
                runner.train(config, resume=resume)
            else:
                runner.project(
                    runner.ProjectConfig(
                        str(out), str(root / "acquire/results"), str(DATA), device="cuda"
                    )
                )
        return
    if finished.is_file() and not (
        stage.endswith("350k") and not (out / "continuation-completion.json").is_file()
    ):
        runner.verify_completion(out, require_current_source=True)
        return
    idle_gpu()
    if stage == "adapt":
        args = _config(PARENT / f"dense-s{seed}/results", out)
        args["adapt_from"] = str(root / "project/results")
        runner.train(runner.TrainConfig(**args), resume=resume)
    elif stage.endswith("300k"):
        arm = stage.split("-")[0]
        args = _config(PARENT / f"{arm}-s{seed}/results", out)
        args.update(
            init_dir=str(root / "adapt/results"), calibration_dir=str(PARENT / "calibration")
        )
        cfg = BudgetConfig(**args)
        runner.train(cfg, resume=resume)
    else:
        arm = stage.split("-")[0]
        continue_training(
            root / f"{arm}-300k/results",
            out,
            additional_steps=50_000,
            device="cuda",
            threads=1,
            wandb_group="icml-review-synthetic-loss",
            resume=resume,
        )


def _assess(root: Path, seed: int, commit: str, data: str) -> None:
    from diagnostics.masked_mlp_oracle.paper_analysis import load_run

    source = PRIVATE / f"single-admitted-s{seed}/metrics.json"
    comparator = read_json(source)
    gate = read_json(root / "acquisition-gate.json")
    if (
        comparator.get("selected_index") != gate["first_eligible_index"]
        or comparator.get("data_identity") != data
        or "validation" not in comparator
    ):
        raise RuntimeError("Need the verified, completed first-eligible comparison")
    measured = {
        a: load_run(root / f"{a}-350k/results", a, seed).metrics for a in ("full", "baseline")
    }
    summary = read_json(COMPARISON)["per_run"]
    reported = {a: summary[a][str(seed)] for a in measured}
    checks = {
        "reported": decision(measured, reported),
        "first_eligible": decision(measured, comparator["validation"]),
    }
    tau = read_json(PARENT / "calibration/calibration.json")["tau_prediction"]
    if comparator["tau"] != tau:
        raise RuntimeError("Paired sparse tolerance mismatch")
    hashes = {
        k: sha(root / k / "results/last.pt")
        for k in (
            "acquire",
            "project",
            "adapt",
            "full-300k",
            "full-350k",
            "baseline-300k",
            "baseline-350k",
        )
    }
    check_head(commit)
    write_json(
        root / "metrics.json",
        {
            "commit": commit,
            "seed": seed,
            "data_identity": data,
            "acquisition_gate_sha256": sha(root / "acquisition-gate.json"),
            "downstream_trained": True,
            "checkpoint_sha256": hashes,
            "tau": tau,
            "validation": measured,
            "reported_validation": reported,
            "first_eligible_validation": comparator["validation"],
            "first_eligible_run": str(source.parent),
            "first_eligible_metrics_sha256": sha(source),
            "first_eligible_checkpoint_sha256": comparator["checkpoint_sha256"],
            "reported_summary_sha256": sha(COMPARISON),
            "checks": checks,
            "passed": all(all(c.values()) for c in checks.values()),
        },
    )


def run(commit: str, seed: int, phase: str) -> None:
    """Run immutable declared stages and stop a seed at its first failed gate."""
    check_head(commit)
    if phase in CHILD_STAGES:
        for child in CHILD_STAGES[phase]:
            # Each native GPU job releases its CUDA context before the next idle-device check.
            subprocess.run([PYTHON, "-m", MODULE, commit, str(seed), child], check=True)
        return
    if seed != PILOT:
        pilot = read_json(PRIVATE / f"synthetic-loss-s{PILOT}/metrics.json")
        if not (pilot["commit"] == commit and pilot["passed"] and pilot.get("downstream_trained")):
            raise RuntimeError("Require a successful full same-commit pilot")
    root = PRIVATE / f"synthetic-loss-s{seed}"
    data_id = read_json(DATA / "completion.json")["identity"]
    joint = PARENT / f"states-s{seed}/results"
    scale = float(read_json(joint / "calibration.json")["metrics"]["prediction"])
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("Invalid calibration forecast scale")
    with file_lock(root / f".{phase}.lock"):
        if phase == "acquire":
            _note(root / "NOTE.md", commit, seed, phase, data_id, scale)
        _note(root / phase / "NOTE.md", commit, seed, phase, data_id, scale)
        if phase == "acquire":
            subprocess.run(
                ["sha256sum", "-c", "--quiet", "reported-checkpoints.sha256"],
                cwd="outputs",
                check=True,
            )
            original = PARENT / f"acquisition-0-s{seed}/results"
            history = read_json(PARENT / f"selection-s{seed}.json")
            original_row = next(
                r for r in history["candidates"] if r["acquisition_seed"] == 1000 + seed
            )
            if sha(original / "completion.json") != original_row["completion_sha256"]:
                raise RuntimeError("The original a0 differs from its historical selection receipt")
            for source in (joint, original):
                if (
                    runner.verify_completion(source, kind="train", require_current_source=True)[
                        "data_identity"
                    ]
                    != data_id
                ):
                    raise RuntimeError("Source checkpoint uses a different cohort")
            _train_stage(root, seed, phase, scale)
            _gate(root, seed, commit, data_id, scale)
        else:
            gate = read_json(root / "acquisition-gate.json")
            if gate["commit"] != commit or gate["data_identity"] != data_id or not gate["passed"]:
                raise RuntimeError("Require a passing same-run acquisition gate")
            if phase == "assess":
                _assess(root, seed, commit, data_id)
            else:
                _train_stage(root, seed, phase, scale)
        check_head(commit)


def main() -> None:
    """Dispatch the fixed pilot or one qualified follow-up."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("seed", type=int, choices=range(5))
    parser.add_argument(
        "phase",
        choices=[
            "acquire",
            "assess",
            *CHILD_STAGES,
            *[c for v in CHILD_STAGES.values() for c in v],
        ],
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.commit, args.seed, args.phase)


if __name__ == "__main__":
    main()
