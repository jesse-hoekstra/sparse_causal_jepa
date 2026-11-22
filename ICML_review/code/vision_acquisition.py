"""Audit the acquisition budget and train a direct projection of the reported joint encoder."""

import argparse
import hashlib
import json
import math
import shlex
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Any

from diagnostics.vision_dimer.train import StageConfig
from scjepa.experiments.storage import file_lock, write_json

from .single_acquisition import _head, _idle_gpu  # pyright: ignore[reportPrivateUsage]

ROOT = Path("outputs/diagnostics/vision-dimer-rgb-v1-s05all")
DATA = Path("outputs/data/vision-dimer-rgb-v1")
OUTPUT = Path("ICML_review/outputs")
PYTHON = ".venv/bin/python"
MODULE = "ICML_review.code.vision_acquisition"
TRAIN = [PYTHON, "-m", "diagnostics.vision_dimer.train"]
MANIFEST = Path("outputs/reported-checkpoints.sha256")
AUDIT = (
    "Compare the joint head, the first independent head, the published winner, and the first "
    "head in index order whose raw calibration forecast MSE does not exceed the joint head's. "
    "All share frozen state encoders at this point. Selection uses no physical labels. Compare "
    "validation forecast error, minimum parameter retention and shuffle sensitivity afterwards. "
    "A pre-projection alternative qualifies descriptively if validation prediction MSE is at "
    "most 1.10 times the selected head's and minimum physical parameter R² is at least 0.95. "
    "Also measure the final-window TRAIN and deterministic CAL/VAL error divided by the original "
    "tolerance in both sparse arms at both budgets. A sign mismatch around one shows why a "
    "training statistic cannot certify deployed feasibility. Agreement is descriptive. "
    "This audit trains no descendants. Replicate a complete seed-zero audit in all seeds."
)
DIRECT = (
    "Test whether the overcomplete joint code itself can be projected and adapted. Omit all "
    "acquisitions. Use the original train-only projection, frozen-state adaptation, original "
    "same-seed numerical tolerance and both 300k+200k sparse budgets. At adaptation require "
    "both deterministic CAL and VAL prediction MSE at most 1.10 times the reported adapted "
    "initializer and minimum VAL parameter R² at least 0.95. A failure stops expansion of this "
    "route for that seed. For the two sparse endpoints require full MCC within 0.01 of the "
    "reported seed or higher, at least 80% of its baseline-to-full MCC gain, full SHD at most "
    "the reference, full validation NMSE at most 1.10 times the reference and minimum full "
    "parameter R² at least 0.95. Only a passing full seed-zero pilot licenses more seeds. "
    "A pass supports a cheaper warm start at this tolerance. A failure preserves the evidence "
    "for the reported procedure and identifies the first failed checkpoint."
)


def read(path: Path) -> dict[str, Any]:
    """Read one frozen receipt."""
    return json.loads(path.read_bytes())


def sha(path: Path) -> str:
    """Digest without loading the tensor into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def declared(path: Path, value: str) -> None:
    """Require restarts to retain the actual declaration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != value:
        raise ValueError(f"Conflicting declaration at {path}")
    path.write_text(value)


def checkpoint_hashes() -> dict[str, str]:
    """Read the original immutable input list."""
    return {
        str(Path("outputs") / f): h
        for h, f in (line.split(maxsplit=1) for line in MANIFEST.read_text().splitlines())
    }


def stage_record(path: Path, inputs: dict[str, str]) -> dict[str, Any]:
    """Read and bind one summary used in the retrospective audit."""
    inputs[str(path)] = sha(path)
    return read(path)


def qualification(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, bool]:
    """Apply the pre-projection diagnostic comparisons."""
    return {
        "forecast": candidate["prediction_mse"] <= 1.10 * reference["prediction_mse"],
        "retention": candidate["parameter_r2_min"] >= 0.95,
    }


def audit(seed: int) -> dict[str, Any]:
    """Inspect immutable training, calibration and validation logs without opening tensors."""
    base = ROOT / f"seed{seed}"
    inputs: dict[str, str] = {}
    receipt = stage_record(base / "selection.json", inputs)
    if receipt["labels_used"] or len(receipt["scores"]) != 16:
        raise ValueError("Unexpected reference acquisition protocol")
    selected = Path(receipt["selected"]).name
    heads = {"joint": base / "joint", **{f"a{i}": base / "acquire" / f"a{i}" for i in range(16)}}
    summaries = {name: stage_record(p / "final.json", inputs) for name, p in heads.items()}
    eligible = next(
        (
            f"a{i}"
            for i in range(16)
            if summaries[f"a{i}"]["calibration_prediction_mse"]
            <= summaries["joint"]["calibration_prediction_mse"]
        ),
        None,
    )
    reference = summaries[selected]
    projections = {
        name: {
            key: record[key]
            for key in (
                "calibration_prediction_mse",
                "prediction_mse",
                "parameter_r2_min",
                "code_shuffle_ratio",
            )
        }
        | {"checks": qualification(record, reference)}
        for name, record in summaries.items()
    }
    hashes = checkpoint_hashes()
    sparse: dict[str, Any] = {}
    for budget in (300_000, 500_000):
        for arm in ("baseline", "full"):
            stage = base / f"sparse-{budget // 1000}k" / arm
            end = stage_record(stage / "final.json", inputs)
            tau = stage_record(stage / "config.json", inputs)["config"]["tau"]
            history = stage / "history.jsonl"
            inputs[str(history)] = sha(history)
            tail = [
                r
                for line in history.read_text().splitlines()
                if budget - 5000 < (r := json.loads(line))["step"] <= budget
            ]
            if [r["step"] for r in tail] != list(range(budget - 4500, budget + 1, 500)) or end[
                "step"
            ] != budget:
                raise ValueError(f"Invalid final update window in {stage}")
            measured = {
                name: value / tau
                for name, value in (
                    ("train", mean(r["prediction"] for r in tail)),
                    ("calibration", end["calibration_prediction_mse"]),
                    ("validation", end["prediction_mse"]),
                )
            }
            if not all(math.isfinite(v) and v > 0 for v in measured.values()):
                raise ValueError(f"Invalid forecast errors in {stage}")
            sparse[f"{budget}/{arm}"] = measured | {
                "tau": tau,
                "state_shd": end["state_shd"],
                "action_shd": end["action_shd"],
                "parameter_shd": end["parameter_shd"],
                "checkpoint_sha256": hashes[str(stage / "final.pt")],
            }
    return {
        "seed": seed,
        "input_sha256": inputs,
        "selected": selected,
        "first_eligible": eligible,
        "heads": projections,
        "sparse": sparse,
        "joint_checkpoint_sha256": hashes[str(heads["joint"] / "final.pt")],
        "selected_checkpoint_sha256": hashes[str(heads[selected] / "final.pt")],
    }


def sparse_checks(
    arms: dict[str, dict[str, Any]], ref: dict[str, dict[str, Any]]
) -> dict[str, bool]:
    """Apply matched endpoint criteria on validation."""
    f, b = arms["full"], arms["baseline"]
    rf, rb = ref["full"], ref["baseline"]
    keys = ("state_shd", "action_shd", "parameter_shd")
    return {
        "mcc": f["mcc"] >= rf["mcc"] - 0.01,
        "gain": f["mcc"] - b["mcc"] >= 0.8 * (rf["mcc"] - rb["mcc"]),
        "shd": sum(f[k] for k in keys) <= sum(rf[k] for k in keys),
        "nmse": f["nmse"] <= 1.10 * rf["nmse"],
        "retention": f["parameter_r2_min"] >= 0.95,
    }


def stage_args(config: StageConfig) -> list[str]:
    """Serialize the complete original stage configuration for reproducible child processes."""
    return [
        part
        for key, val in asdict(config).items()
        if val is not None
        for part in ("--" + key.replace("_", "-"), str(val))
    ]


def stage_run(rev: str, directory: Path, cmd: list[str], note: str, parent: Path) -> str:
    """Run the release entry point with pinned inputs and resumable training."""
    _head(rev)
    parent_hash = sha(parent)
    declared(
        directory / "NOTE.md",
        note
        + f"\nParent: {parent}\nSHA-256: {parent_hash}\n\n"
        + "    PYTHONPATH=$PWD:$PWD/src "
        + shlex.join(cmd)
        + "\n",
    )
    done = directory / "review-completion.json"
    final = directory / "final.pt"
    if done.is_file():
        stored = read(done)
        if stored != {
            "commit": rev,
            "checkpoint_sha256": sha(final),
            "summary_sha256": sha(directory / "final.json"),
            "parent_sha256": parent_hash,
        }:
            raise RuntimeError("Prior stage has changed")
        return stored["checkpoint_sha256"]
    if not (final.is_file() and (directory / "final.json").is_file()):
        if "train" in cmd and (directory / "last.pt").is_file():
            cmd = [*cmd, "--resume"]
            (directory / "resume-command.txt").write_text(shlex.join(cmd) + "\n")
        _idle_gpu()
        with (directory / "log.txt").open("a") as log:
            child = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        (directory / "exit_code").write_text(f"{child.returncode}\n")
        child.check_returncode()
    _head(rev)
    if sha(parent) != parent_hash:
        raise RuntimeError("The parent changed during training")
    digest = sha(final)
    write_json(
        done,
        {
            "commit": rev,
            "checkpoint_sha256": digest,
            "summary_sha256": sha(directory / "final.json"),
            "parent_sha256": parent_hash,
        },
    )
    return digest


def adaptation_checks(measured: dict[str, Any], reference: dict[str, Any]) -> dict[str, bool]:
    """Evaluate held-out forecast and physical retention at the dense initializer."""
    checks = {
        k: measured[k] <= 1.10 * reference[k]
        for k in ("prediction_mse", "calibration_prediction_mse")
    }
    checks["retention"] = measured["parameter_r2_min"] >= 0.95
    return checks


def _config(
    seed: int, stage: str, source: Path, destination: Path, parent: Path, data_dir: Path, group: str
) -> StageConfig:
    old = StageConfig.from_saved(read(source / "config.json")["config"])
    if old.seed != seed or old.stage != stage:
        raise RuntimeError("Wrong archived config")
    return replace(
        old,
        data_dir=str(data_dir),
        output_dir=str(destination),
        parent=str(parent),
        wandb_group=group,
    )


def project_adapt(
    seed: int,
    rev: str,
    out: Path,
    note: str,
    source: Path,
    *,
    base: Path,
    data_dir: Path,
    group: str,
) -> dict[str, Any]:
    """Apply the reported projection and frozen adaptation to a specific overcomplete source."""
    projected, adapted = out / "projected", out / "adapt"
    cfg = _config(seed, "adapt", base / "adapt", adapted, projected / "final.pt", data_dir, group)
    projected_sha = stage_run(
        rev,
        projected,
        [
            *TRAIN,
            "project",
            "--source",
            str(source.parent),
            "--data-dir",
            str(data_dir),
            "--output",
            str(projected),
            "--device",
            cfg.device,
        ],
        note,
        source,
    )
    adapted_sha = stage_run(
        rev, adapted, [*TRAIN, "train", *stage_args(cfg)], note, projected / "final.pt"
    )
    measured, reference = read(adapted / "final.json"), read(base / "adapt/final.json")
    return {
        "seed": seed,
        "adaptation_checks": adaptation_checks(measured, reference),
        "adaptation": measured,
        "reference_adaptation": reference,
        "checkpoint_sha256": {"projected": projected_sha, "adapt": adapted_sha},
    }


def sparse_arm(
    seed: int, rev: str, out: Path, note: str, arm: str, *, base: Path, data_dir: Path, group: str
) -> dict[str, str]:
    """Run one released sparse path and resume its optimizer to the archived second budget."""
    first, final = out / f"sparse-300k/{arm}", out / f"sparse-500k/{arm}"
    adapted = out / "adapt/final.pt"
    cfg = _config(seed, arm, base / f"sparse-300k/{arm}", first, adapted, data_dir, group)
    endpoint = _config(seed, arm, base / f"sparse-500k/{arm}", final, adapted, data_dir, group)
    if replace(cfg, output_dir=endpoint.output_dir, steps=endpoint.steps) != endpoint:
        raise ValueError("reference sparse arm changes numerical settings between budgets")
    first_sha = stage_run(rev, first, [*TRAIN, "train", *stage_args(cfg)], note, adapted)
    last = first / "last.pt"
    final_sha = stage_run(
        rev,
        final,
        [
            *TRAIN,
            "fork",
            "--source",
            str(last),
            "--output-dir",
            str(final),
            "--steps",
            str(endpoint.steps),
            "--wandb-mode",
            cfg.wandb_mode,
            "--wandb-group",
            group,
        ],
        note,
        last,
    )
    return {f"300k/{arm}": first_sha, f"500k/{arm}": final_sha}


def direct(seed: int, rev: str, out: Path, note: str) -> dict[str, Any]:
    """Run only qualifying stages of the zero-acquisition route."""
    base = ROOT / f"seed{seed}"
    subprocess.run(
        ["sha256sum", "-c", "--quiet", "reported-checkpoints.sha256"], cwd="outputs", check=True
    )
    joint = base / "joint/final.pt"
    joint_sha = checkpoint_hashes()[str(joint)]
    if sha(joint) != joint_sha:
        raise RuntimeError("Wrong joint initializer")
    group = "icml-review-direct-joint"
    result = project_adapt(seed, rev, out, note, joint, base=base, data_dir=DATA, group=group)
    result["checkpoint_sha256"]["joint"] = joint_sha
    if not all(result["adaptation_checks"].values()):
        return result | {"passed": False, "reason": "adaptation"}
    arms: dict[str, dict[str, Any]] = {}
    reference: dict[str, dict[str, Any]] = {}
    for arm in ("full", "baseline"):
        result["checkpoint_sha256"].update(
            sparse_arm(seed, rev, out, note, arm, base=base, data_dir=DATA, group=group)
        )
        arms[arm] = read(out / f"sparse-500k/{arm}/final.json")
        reference[arm] = read(base / f"sparse-500k/{arm}/final.json")
    checks = sparse_checks(arms, reference)
    return result | {
        "validation": arms,
        "reference_validation": reference,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    """Audit locally or run the direct training comparison on a free server GPU."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("commit")
    cli.add_argument("seed", type=int, choices=range(5))
    cli.add_argument("mode", choices=("audit", "direct"))
    args = cli.parse_args()
    rev, seed, mode = args.commit, args.seed, args.mode
    _head(rev)
    prefix = "vision-head-audit" if mode == "audit" else "vision-direct"
    out = OUTPUT / f"{prefix}-s{seed}"
    manifest = sha(DATA / "manifest.json")
    if seed:
        pilot = read(OUTPUT / f"{prefix}-s0/metrics.json")
        if pilot["commit"] != rev or not pilot["passed"]:
            raise RuntimeError("Expansion requires a successful same-commit pilot")
    note = (
        f"# Vision {mode}, seed {seed}\n\n"
        + (AUDIT if mode == "audit" else DIRECT)
        + f"\n\nGit: {rev}\nData manifest SHA-256: {manifest}\n\n    PYTHONPATH=$PWD:$PWD/src "
        + shlex.join([PYTHON, "-m", MODULE, rev, str(seed), mode])
        + "\n"
    )
    with file_lock(out / ".lock"):
        declared(out / "NOTE.md", note)
        if mode == "audit":
            result = audit(seed) | {"passed": True, "scope": "audit_complete"}
        else:
            result = direct(seed, rev, out, note)
        _head(rev)
        write_json(out / "metrics.json", result | {"commit": rev, "data_manifest_sha256": manifest})


if __name__ == "__main__":
    main()
