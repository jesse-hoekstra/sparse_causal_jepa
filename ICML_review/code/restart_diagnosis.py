"""Relate synthetic acquisition admission to held-out forecast and parameter recovery."""

import argparse
import math
import shlex
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr  # pyright: ignore[reportMissingTypeStubs]

from scjepa.experiments.storage import write_json

from . import halving
from .single_acquisition import DATA, PARENT, PRIVATE, PYTHON
from .single_acquisition import _head as check_head  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import declared, read, sha

PLAN = """Hypothesis: among heads with the same frozen state encoder, high TRAIN forecast
error is associated with failure to encode at least one parameter. Inspect all archived heads
without refitting any model. Cross TRAIN admission at the documented threshold with minimum
labelled validation R2 at 0.80, record the weakest physical parameter and both CAL/VAL errors.
Report within-seed Spearman associations, the midpoint to endpoint admission transition,
the joint-stage reference, first admitted index, and exact without-replacement success
probabilities for a random subset of the fixed archive. A strong association would explain
why repeated initializations can help. Weak association would indicate the selector measures
another property. Neither outcome identifies a causal mechanism. Labels here are evaluation
only. This descriptive audit cannot select an acquisition or score an untrained method."""


def hit_probability(size: int, good: int, draws: int) -> float:
    """Chance of at least one hit in a uniform subset of a fixed finite archive."""
    if not 0 <= good <= size or not 0 < draws <= size:
        raise ValueError("invalid finite population")
    return 1 - math.comb(size - good, draws) / math.comb(size, draws)


def read_scores(root: Path, hashes: dict[str, str]) -> dict[str, Any]:
    """Read only CAL and VAL receipts from one archived candidate."""
    out: dict[str, Any] = {}
    for split in ("calibration", "validation"):
        path = root / f"{split}.json"
        hashes[str(path)] = sha(path)
        m = read(path)["metrics"]
        out[split] = {
            "prediction": m["prediction"],
            "minimum_r2": m["code_r2_min"],
            "parameter_r2": [m[f"code_r2/theta_{j}"] for j in range(1, 7)],
        }
    return out


def collect(seed: int, hashes: dict[str, str]) -> dict[str, Any]:
    """Rebuild TRAIN scores and preserve matching system/seed membership."""
    history = halving.run("synthetic", seed)
    hashes.update(history["input_sha256"])
    threshold = read(PARENT / f"selection-s{seed}.json")["maximum_loss"]
    rows: list[dict[str, Any]] = []
    for i, (mid, final) in enumerate(
        zip(history["midpoint_scores"], history["endpoint_scores"], strict=True)
    ):
        metrics = read_scores(PARENT / f"acquisition-{i}-s{seed}/results", hashes)
        rows.append(
            {
                "attempt": i,
                "midpoint": mid,
                "final": final,
                "mid_admitted": mid <= threshold,
                "admitted": final <= threshold,
                "retained": metrics["validation"]["minimum_r2"] >= 0.8,
                **metrics,
            }
        )
    good = sum(r["admitted"] for r in rows)
    both = sum(r["admitted"] and r["retained"] for r in rows)
    return {
        "seed": seed,
        "cutoff": threshold,
        "counts": {
            "admitted": good,
            "admitted_and_retained": both,
            "rejected_and_retained": sum(not r["admitted"] and r["retained"] for r in rows),
            "mid_fail_final_pass": sum(not r["mid_admitted"] and r["admitted"] for r in rows),
            "mid_pass_final_fail": sum(r["mid_admitted"] and not r["admitted"] for r in rows),
        },
        "first_admitted": next((r["attempt"] for r in rows if r["admitted"]), None),
        "random_subset": {
            k: {str(d): hit_probability(len(rows), n, d) for d in (1, 2, 4, 8, 16)}
            for k, n in (("admitted", good), ("admitted_and_retained", both))
        },
        "spearman": {
            k: float(spearmanr(history["endpoint_scores"], [r[s][m] for r in rows])[0])  # pyright: ignore[reportUnknownArgumentType]
            for k, s, m in (
                ("train_vs_cal", "calibration", "prediction"),
                ("train_vs_val_min_r2", "validation", "minimum_r2"),
            )
        },
        "joint": read_scores(PARENT / f"states-s{seed}/results", hashes),
        "winner": history["original_attempt"],
        "heads": rows,
    }


def main() -> None:
    """Write the declaration before reading any scientific scores."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("commit")
    rev: str = cli.parse_args().commit
    check_head(rev)
    out = PRIVATE / "restart-archive-synthetic"
    cmd = "PYTHONPATH=$PWD:$PWD/src " + shlex.join(
        [PYTHON, "-m", "ICML_review.code.restart_diagnosis", rev]
    )
    identity = DATA / "completion.json"
    declared(
        out / "NOTE.md",
        f"# Synthetic acquisition archive\n\n{PLAN}\n\nCheckout {rev}\n"
        f"Data {identity} SHA256 {sha(identity)}\n\n    {cmd}\n",
    )
    hashes = {str(identity): sha(identity)}
    seeds = [collect(seed, hashes) for seed in range(5)]
    check_head(rev)
    write_json(
        out / "metrics.json",
        {
            "commit": rev,
            "command": cmd,
            "seeds": seeds,
            "note_sha256": sha(out / "NOTE.md"),
            "input_sha256": hashes,
        },
    )


if __name__ == "__main__":
    main()
