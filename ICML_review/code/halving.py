"""Replay a fixed two-round acquisition policy on recorded TRAIN prediction losses."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

OUTPUTS = Path("outputs")
ATTEMPTS = 16
MIDPOINT = 50_000
ENDPOINT = 100_000
WINDOW = 5_000


def window_scores(
    history: bytes, key: str, interval: int, *, compensated_sum: bool
) -> tuple[float, float]:
    """Check the complete logging grids and reproduce both trailing-window means."""
    scores: dict[int, float] = {}
    for line in history.splitlines():
        row: dict[str, Any] = json.loads(line)
        if key not in row:
            continue
        step = row["step"]
        if not (MIDPOINT - WINDOW < step <= MIDPOINT or ENDPOINT - WINDOW < step <= ENDPOINT):
            continue
        loss = float(row[key])
        if step in scores or not math.isfinite(loss) or loss < 0:
            raise ValueError(f"Invalid or duplicate training loss at step {step}")
        scores[step] = loss

    def average(end: int) -> float:
        steps = list(range(end - WINDOW + interval, end + 1, interval))
        actual = sorted(s for s in scores if end - WINDOW < s <= end)
        if actual != steps:
            raise ValueError(f"Training log grid differs from the expected window ending at {end}")
        values = [scores[s] for s in steps]
        return (math.fsum(values) if compensated_sum else sum(values)) / len(values)

    return average(MIDPOINT), average(ENDPOINT)


def run(experiment: str, seed: int) -> dict[str, Any]:
    """Compare survivor and original winners for one reported seed from the checkout."""
    if experiment not in {"synthetic", "vision"} or seed not in range(5):
        raise ValueError("Choose a reported experiment and a seed in 0..4")
    synthetic = experiment == "synthetic"
    root = OUTPUTS / "diagnostics"
    if synthetic:
        batch = root / "masked-mlp-gelu-20261101"
        folders = [batch / f"acquisition-{i}-s{seed}" / "results" for i in range(ATTEMPTS)]
        selection = batch / f"selection-s{seed}.json"
    else:
        batch = root / "vision-dimer-rgb-v1-s05all" / f"seed{seed}"
        folders = [batch / "acquire" / f"a{i}" for i in range(ATTEMPTS)]
        selection = batch / "selection.json"

    hashes: dict[str, str] = {}

    def read(path: Path) -> bytes:
        data = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        return data

    receipt: dict[str, Any] = json.loads(read(selection))
    if receipt["window"] != WINDOW:
        raise ValueError("Recorded selection used a different window")
    reported: dict[str, float]
    if synthetic:
        if receipt["selection_used_parameter_labels"] or not receipt["admitted"]:
            raise ValueError("Expected an admitted TRAIN-only selection")
        reported = {r["directory"]: r["train_prediction_mean"] for r in receipt["candidates"]}
        recorded_winner = receipt["selected"]["directory"]
    else:
        if receipt["labels_used"]:
            raise ValueError("Expected a TRAIN-only selection")
        reported = receipt["scores"]
        recorded_winner = receipt["selected"]
    if len(reported) != ATTEMPTS:
        raise ValueError("Acquisition receipt contains an unexpected number of attempts")

    mid: list[float] = []
    final: list[float] = []
    display_paths: list[str] = []
    for folder in folders:
        half_loss, full_loss = window_scores(
            read(folder / "history.jsonl"),
            "loss/prediction" if synthetic else "prediction",
            100 if synthetic else 500,
            compensated_sum=synthetic,
        )
        suffix = folder.relative_to(OUTPUTS).parts
        matches = [path for path in reported if Path(path).parts[-len(suffix) :] == suffix]
        if len(matches) != 1 or not math.isclose(full_loss, reported[matches[0]], rel_tol=1e-13):
            raise ValueError(f"Replayed endpoint score does not match the receipt for {folder}")
        display_paths.append(matches[0])
        mid.append(half_loss)
        final.append(full_loss)

    original = min(range(ATTEMPTS), key=lambda i: (final[i], i))
    if display_paths[original] != recorded_winner:
        raise ValueError("Replayed reference winner differs from the selection receipt")
    survivors = sorted(range(ATTEMPTS), key=lambda i: (mid[i], i))[: ATTEMPTS // 2]
    winner = min(survivors, key=lambda i: (final[i], i))
    admitted = not synthetic or final[winner] <= float(receipt["maximum_loss"])
    manifest = read(OUTPUTS / "reported-checkpoints.sha256").decode()
    checkpoints = {
        path: digest for digest, path in (line.split(maxsplit=1) for line in manifest.splitlines())
    }
    checkpoint_name = "last.pt" if synthetic else "final.pt"
    reference = (folders[original] / checkpoint_name).relative_to(OUTPUTS).as_posix()
    candidate = (folders[winner] / checkpoint_name).relative_to(OUTPUTS).as_posix()
    return {
        "experiment": experiment,
        "seed": seed,
        "original_attempt": original,
        "surviving_attempts": survivors,
        "selected_attempt": winner,
        "admitted": admitted,
        "matched_winner": admitted and winner == original,
        "midpoint_scores": mid,
        "endpoint_scores": final,
        "original_checkpoint": reference,
        "original_checkpoint_sha256": checkpoints[reference],
        "selected_checkpoint": candidate if admitted else None,
        "selected_checkpoint_sha256": checkpoints.get(candidate) if admitted else None,
        "acquisition_updates_reference": ATTEMPTS * ENDPOINT,
        "acquisition_updates_policy": ATTEMPTS * MIDPOINT + len(survivors) * (ENDPOINT - MIDPOINT),
        "input_sha256": hashes,
    }
