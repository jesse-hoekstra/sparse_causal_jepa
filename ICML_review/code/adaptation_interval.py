"""Estimate paired context uncertainty for the frozen seed-zero RGB adaptation gate."""

import argparse
import json
import math
import shlex
import subprocess
from typing import Any

import torch
from torch import Tensor

from diagnostics.vision_dimer import data, evaluate, train
from scjepa.experiments.storage import file_lock, write_json
from scjepa.models.model import SCJEPA

from .changed_queries import Inputs, require
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
)

MARGIN = 1.10
DRAWS = 10000
RTOL, ATOL = 1e-4, 1e-8
SEEDS = {"calibration": 20261101, "validation": 20261102}
COUNTS = {"calibration": 512, "validation": 256}
KEYS = {"calibration": "calibration_prediction_mse", "validation": "prediction_mse"}
FOLDERS = {
    "reported": ROOT / "seed0",
    "direct": OUTPUT / "vision-direct-s0",
    "one_head": OUTPUT / "vision-one-head-s0",
}
PROTOCOL = (
    "Hypothesis: sampling uncertainty across held-out systems may span the original adaptation "
    "margin. Freeze the published, direct and one-head seed-zero adapted models; use the original "
    "first-window two-step predictions, fixed evaluation donors and raw squared errors. Use all "
    "512 calibration and 256 validation systems and each system's ten episodes. Require all six "
    "raw checkpoint/split means to match their original stored values at rtol=1e-4, atol=1e-8 "
    "before reporting interval estimates. Resample whole systems with replacement 10000 times "
    "within each split, using the same indices for every method. Report two-sided 95% percentile "
    "intervals for the trial/reported ratio of mean MSE and for mean trial MSE minus 1.10 times "
    "mean reported MSE. An excess interval containing zero leaves the sign of the population "
    "margin unresolved. An interval wholly above or below zero supports the corresponding "
    "direction for these fixed checkpoints under system resampling. The original observed "
    "admission and stopping decisions remain fixed. No optimizer, labels, test data or fitting."
)


def group_errors(model: SCJEPA, inputs: data.RenderedInputs, *, batch_size: int = 16) -> Tensor:
    """Average native forecast errors over views, steps and coordinates within each system."""
    collected = evaluate.collect(model, train.rendered_batches(inputs, size=batch_size))
    predicted, target = collected["predictions"], collected["targets"]
    if predicted.shape != target.shape or predicted.shape[:3] != (len(inputs), inputs.views, 2):
        raise ValueError("unexpected grouped two-step forecast shape")
    return (predicted - target).square().mean((1, 2, 3))


def paired_intervals(errors: dict[str, Tensor], seed: int, *, draws: int = DRAWS) -> dict[str, Any]:
    """Apply common paired context resamples to the margin excess and ratio of mean errors."""
    original = errors["reported"].double().cpu()
    if (
        original.ndim != 1
        or len(original) < 2
        or not torch.isfinite(original).all()
        or not (original > 0).all()
    ):
        raise ValueError("expected positive reported errors for multiple contexts")
    ids = torch.randint(
        len(original), (draws, len(original)), generator=torch.Generator().manual_seed(seed)
    )
    average = original.mean()
    resampled = original[ids].mean(1)

    def interval(values: Tensor) -> list[float]:
        return [
            float(v)
            for v in torch.quantile(values, torch.tensor([0.025, 0.975], dtype=torch.float64))
        ]

    result: dict[str, Any] = {}
    for name in ("direct", "one_head"):
        trial = errors[name].double().cpu()
        if (
            trial.shape != original.shape
            or not torch.isfinite(trial).all()
            or not (trial >= 0).all()
        ):
            raise ValueError("trial and comparator must have aligned finite errors")
        mean = trial.mean()
        boot = trial[ids].mean(1)
        lo, hi = interval(boot / resampled)
        lower, upper = interval(boot - MARGIN * resampled)
        result[name] = {
            "ratio": float(mean / average),
            "ratio_ci95": [lo, hi],
            "ratio_contains_margin": lo <= MARGIN <= hi,
            "excess": float(mean - MARGIN * average),
            "excess_ci95": [lower, upper],
            "excess_contains_zero": lower <= 0 <= upper,
        }
    return result


def _sources(ledger: Inputs, reported: dict[str, str], manifest: str) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    original = FOLDERS["reported"]
    joint = original / "joint/final.pt"
    joint_sha = ledger.bind(joint, reported[str(joint)])
    summary = ledger.record(original / "adapt/final.json")
    for name, root in FOLDERS.items():
        path = root / "adapt/final.pt"
        final = root / "adapt/final.json"
        metrics = ledger.record(final)
        source = {
            "checkpoint": str(path),
            "summary_sha256": ledger[str(final)],
            "raw_references": {s: metrics[key] for s, key in KEYS.items()},
        }
        if name == "reported":
            source["checkpoint_sha256"] = ledger.bind(path, reported[str(path)])
        else:
            terminal = root / "metrics.json"
            prior = ledger.record(terminal)
            rev = prior["commit"]
            require(
                prior["seed"] == 0
                and prior["data_manifest_sha256"] == manifest
                and prior["adaptation"] == metrics
                and prior["reference_adaptation"] == summary
                and prior["checkpoint_sha256"]["joint"] == joint_sha,
                str(terminal),
            )
            ledger.note(root / "NOTE.md", rev, manifest)
            parent_sha = joint_sha
            if name == "one_head":
                record = root / "candidate.json"
                ledger.bind(record, prior["candidate_sha256"])
                candidate = ledger.record(record)
                require(
                    candidate["verified"] is True
                    and candidate["commit"] == rev
                    and candidate["joint_sha256"] == joint_sha
                    and candidate["manifest_sha256"] == manifest
                    and candidate["final_and_last_model_identical"] is True,
                    str(record),
                )
                parent_sha = ledger.bind(
                    original / "acquire/a0/final.pt", candidate["final_sha256"]
                )
                require(prior["checkpoint_sha256"]["a0"] == parent_sha, str(terminal))
            for stage in ("projected", "adapt"):
                folder = root / stage
                done = ledger.record(folder / "review-completion.json")
                require(
                    done["commit"] == rev
                    and done["parent_sha256"] == parent_sha
                    and done["checkpoint_sha256"] == prior["checkpoint_sha256"][stage],
                    str(folder),
                )
                ledger.note(folder / "NOTE.md", rev, parent_sha)
                ledger.bind(folder / "final.json", done["summary_sha256"])
                parent_sha = ledger.bind(folder / "final.pt", done["checkpoint_sha256"])
            source.update(
                checkpoint_sha256=parent_sha,
                training_commit=rev,
                receipt_sha256=ledger[str(terminal)],
                root_note_sha256=ledger[str(root / "NOTE.md")],
                stage_receipt_sha256=ledger[str(root / "adapt/review-completion.json")],
                stage_note_sha256=ledger[str(root / "adapt/NOTE.md")],
                original_adaptation_checks=prior["adaptation_checks"],
            )
        sources[name] = source
    return sources


def replay(raw: Tensor, reference: float) -> dict[str, Any]:
    """Compare a measured group mean to the released stage's archived raw error."""
    measured = float(raw.double().mean())
    return {
        "stored": reference,
        "measured": measured,
        "absolute_difference": abs(measured - reference),
        "matches": math.isclose(measured, reference, rel_tol=RTOL, abs_tol=ATOL),
    }


def compare(
    raw: dict[str, dict[str, Tensor]], sources: dict[str, Any], *, views: int
) -> tuple[bool, dict[str, Any]]:
    """Release paired intervals only after all six group-mean forecast replays agree."""
    if set(raw) != set(SEEDS) or any(set(arms) != set(FOLDERS) for arms in raw.values()):
        raise ValueError("both splits and all three frozen models are required")
    reports: dict[str, Any] = {}
    for s, arms in raw.items():
        checks = {
            name: replay(values, sources[name]["raw_references"][s])
            for name, values in arms.items()
        }
        reports[s] = {
            "groups": len(arms["reported"]),
            "views": views,
            "bootstrap_seed": SEEDS[s],
            "replay": checks,
            "group_mse": {name: [float(v) for v in values] for name, values in arms.items()},
        }
    verified = all(check["matches"] for r in reports.values() for check in r["replay"].values())
    if verified:
        for s in SEEDS:
            reports[s]["paired"] = paired_intervals(raw[s], SEEDS[s])
    return verified, reports


def run(rev: str) -> None:
    """Verify frozen inputs and report uncertainty without reopening any scientific gate."""
    out = OUTPUT / "adaptation-interval-s0"
    _head(rev)
    command = "PYTHONPATH=$PWD:$PWD/src " + shlex.join(
        [PYTHON, "-m", "ICML_review.code.adaptation_interval", rev]
    )
    reported = checkpoint_hashes()
    ledger = Inputs()
    manifest = ledger.bind(DATA / "manifest.json")
    splits = {s: read(DATA / "manifest.json")["splits"][s]["sha256"] for s in SEEDS}
    expected = {
        name: (
            reported[str(root / "adapt/final.pt")]
            if name == "reported"
            else read(root / "metrics.json")["checkpoint_sha256"]["adapt"]
        )
        for name, root in FOLDERS.items()
    }
    note = (
        f"# Seed-zero paired adaptation margin\n\n{PROTOCOL}\n\nCommit `{rev}`. "
        f"Data `{DATA}`, manifest SHA-256 `{manifest}`, split SHA-256 "
        f"`{json.dumps(splits, sort_keys=True)}`. Expected adapted checkpoint SHA-256 "
        f"`{json.dumps(expected, sort_keys=True)}`. Bootstrap seeds `{SEEDS}`.\n\n    {command}\n"
    )
    with file_lock(out / ".lock"):
        declared(out / "NOTE.md", note)
        note_sha = sha(out / "NOTE.md")
        subprocess.run(
            ["sha256sum", "-c", "--quiet", "reported-checkpoints.sha256"], cwd="outputs", check=True
        )
        ledger.bind(MANIFEST)
        archive = ledger.record(ROOT / "release-check-20261120/data-manifest-with-test.json")
        for s, digest in splits.items():
            require(archive["splits"][s]["sha256"] == digest, s)
            ledger.bind(DATA / f"{s}.pt", digest)
        sources = _sources(ledger, reported, manifest)
        result_path = out / "metrics.json"
        if result_path.exists():
            previous = read(result_path)
            require(
                previous["commit"] == rev
                and previous["note_sha256"] == note_sha
                and previous["sources"] == sources
                and previous["input_sha256"] == ledger,
                str(result_path),
            )
            return
        _idle_gpu()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda")
        datasets = {s: data.RenderedInputs(DATA, s, device) for s in SEEDS}
        for s, inputs in datasets.items():
            require(len(inputs) == COUNTS[s] and inputs.views == 10 and inputs.frames == 12, s)
        raw: dict[str, dict[str, Tensor]] = {s: {} for s in SEEDS}
        for name, source in sources.items():
            _head(rev)
            model = train.model_from_checkpoint(
                train.load_checkpoint(source["checkpoint"]), phase="dense"
            )
            model.to(device).eval()
            for s, inputs in datasets.items():
                raw[s][name] = group_errors(model, inputs)
        verified, reports = compare(raw, sources, views=10)
        _head(rev)
        write_json(
            result_path,
            {
                "verified": verified,
                "seed": 0,
                "commit": rev,
                "command": command,
                "note_sha256": note_sha,
                "sources": sources,
                "results": reports,
                "input_sha256": ledger,
                "split_sha256": splits,
                "margin": MARGIN,
                "bootstrap_draws": DRAWS,
                "rtol": RTOL,
                "atol": ATOL,
                "gpu": torch.cuda.get_device_name(),
                "torch": str(torch.__version__),
                "tf32": False,
            },
        )


def main() -> None:
    """Read the expected Git revision and run the frozen diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    run(parser.parse_args().commit)


if __name__ == "__main__":
    main()
