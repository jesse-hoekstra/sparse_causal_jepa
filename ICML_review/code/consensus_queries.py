"""Score eligible changed sparse RGB checkpoints on the paired validation interventions."""

import argparse
import math
import shlex
import subprocess
from pathlib import Path
from typing import Any

from scjepa.experiments.storage import file_lock, write_json

from . import chart, kicks
from .changed_queries import Inputs, controls, paired, require
from .single_acquisition import _head, _idle_gpu  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import (
    DATA,
    MANIFEST,
    OUTPUT,
    PYTHON,
    checkpoint_hashes,
    declared,
    read,
    sha,
)

METHODS = ("linear-consensus-vision", "rgb-predictive", "vision-invariant")
PROTOCOL = """Hypothesis: a changed encoder can improve forecast or consistency while
degrading the named physical intervention. Within each reported seed and sparse condition,
replace only the evaluated endpoint. Refit the declared scalar and joint affine physical
readouts on calibration. Keep the physical query and the validation systems, episodes,
windows, simulator, substrate and aggregation fixed. Report partner-slope absolute error
(Q1) and mean held-kick relative Frobenius error (Q2) for both interfaces. For the named
single-coordinate interface, flag deterioration of more than 0.01 in Q1 or 0.05 in Q2
compared with the original matching seed and arm. These absolute margins are practical
review criteria. Evaluate every available end checkpoint including failed pilots and both
teacher arms. A downstream failure precludes a recommendation for the causal application.
A trial ending before the sparse endpoint yields no downstream estimate."""


def sources(method: str, root: Path, record: dict[str, Any]) -> dict[str, tuple[str, Path, str]]:
    """Bind every actually produced terminal checkpoint to its training receipt."""
    hashes = record.get("checkpoint_sha256")
    if method == "linear-consensus-vision":
        return {"sibling": ("full", root / "final.pt", str(hashes))}
    if method == "vision-invariant" and hashes and "other" in hashes:
        return {
            a: ("full", root / a / "final.pt", hashes[m])
            for a, m in (("own", "same"), ("sibling", "other"))
        }
    if method == "rgb-predictive" and hashes and "validation" in record:
        return {
            a: (a, root / "sparse-500k" / a / "final.pt", hashes[f"500k/{a}"])
            for a in ("full", "baseline")
        }
    require(not record["passed"], "Passing trial lacks an endpoint")
    return {}


def run(commit: str, method: str, seed: int) -> None:
    """Compare each completed arm with its independently archived control."""
    root = OUTPUT / f"{method}-s{seed}"
    out = OUTPUT / f"consensus-queries-{method}-s{seed}"
    _head(commit)
    cmd = "PYTHONPATH=$PWD:$PWD/src " + shlex.join(
        [PYTHON, "-m", "ICML_review.code.consensus_queries", commit, method, str(seed)]
    )
    with file_lock(out / ".lock"):
        ledger = Inputs()
        manifest = ledger.record(DATA / "manifest.json")
        split_ids = {
            s: manifest["splits"][s]["sha256"] for s in ("train", "calibration", "validation")
        }
        record = ledger.record(root / "metrics.json")
        require(record["commit"] == commit and record["seed"] == seed, "Wrong training lineage")
        declared(
            out / "NOTE.md",
            f"# Physical intervention audit, {method} seed {seed}\n\n"
            f"{PROTOCOL}\n\nCheckout {commit}\nSplit digests {split_ids}\n"
            f"Training receipt SHA256 {ledger[str(root / 'metrics.json')]}\n\n    {cmd}\n",
        )
        subprocess.run(
            ["sha256sum", "-c", "--quiet", MANIFEST.name], cwd=MANIFEST.parent, check=True
        )
        existing = out / "metrics.json"
        if existing.exists():
            prior = read(existing)
            require(
                prior["commit"] == commit and prior["note_sha256"] == sha(out / "NOTE.md"),
                "Stored causal receipt changed",
            )
            return
        paths = sources(method, root, record)
        reference = controls(seed, split_ids, checkpoint_hashes(), ledger)
        results: dict[str, Any] = {}
        if paths:
            _idle_gpu()
        for alias, (arm, path, digest) in paths.items():
            ledger.bind(path, digest)
            ledger.bind(path.parent / "final.json")
            result: dict[str, Any] = {}
            for q, evaluate, limit in (("chart", chart.run, 0.01), ("kicks", kicks.run, 0.05)):
                _head(commit)
                r = evaluate(seed, {arm: path})
                require(r["arms"][arm]["checkpoint_sha256"] == digest, "Query checkpoint differs")
                endpoint = r["arms"][arm]
                match = (
                    "spearman_assignment_matches_stored"
                    if q == "chart"
                    else "assignment_matches_stored"
                )
                require(endpoint[match] is True, "CAL readout assignment did not reproduce")
                require(
                    math.isfinite(endpoint["recomputed_first_window_nmse"]),
                    "Nonfinite own-episode forecast",
                )
                if q == "kicks":
                    hold = result["chart"]["measurement"]["arms"][arm]
                    require(
                        math.isclose(
                            hold["recomputed_first_window_nmse"],
                            endpoint["recomputed_first_window_nmse"],
                            rel_tol=1e-6,
                        ),
                        "Hold and kick unperturbed own-episode forecasts differ",
                    )
                delta = paired(q, r, reference[q], arm)
                result[q] = {
                    "paired": delta,
                    "measurement": r,
                    "scalar_within_margin": delta["scalar"]["trial_minus_control"] <= limit,
                }
            results[alias] = result
        _head(commit)
        write_json(
            existing,
            {
                "commit": commit,
                "seed": seed,
                "method": method,
                "command": cmd,
                "input_sha256": ledger,
                "note_sha256": sha(out / "NOTE.md"),
                "training_passed": record["passed"],
                "arms": results,
            },
        )


def main() -> None:
    """Evaluate one completed seed on the validation cohort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("seed", type=int, choices=range(5))
    args = parser.parse_args()
    run(args.commit, args.method, args.seed)


if __name__ == "__main__":
    main()
