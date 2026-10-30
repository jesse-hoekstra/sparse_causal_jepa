"""Declare a tighter forecast bound without changing or rehashing the trainer.

Ordinary calibration remains fixed at 1.1 times seed zero's dense calibration
loss. This explicit diagnostic creates a separate, authenticated calibration
artifact from that same loss, recording the hypothesis and its original source.
It reads no parameter/state labels, validation scores or model tensors.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from diagnostics.masked_mlp_learned.run import verify_completion
from diagnostics.oracle_state_learning.data import read_json, sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, write_json


def tighten_bound(
    calibration_dir: Path, output_dir: Path, *, tau_factor: float, hypothesis: str
) -> dict[str, Any]:
    """Authenticate the original bound and dense reference, then write a new bound."""
    if not math.isfinite(tau_factor) or not 0 < tau_factor < 1.1:
        raise ValueError("a tighter control requires 0 < tau_factor < 1.1")
    if not hypothesis.strip():
        raise ValueError("record the prospectively declared hypothesis")
    source = calibration_dir.resolve()
    original = verify_completion(source, kind="calibration", require_current_source=True)
    record = read_json(source / "calibration.json")
    identity = canonical_fingerprint({k: v for k, v in record.items() if k != "identity"})
    if record.get("identity") != identity or original["identity"] != identity:
        raise ValueError("original calibration identity differs")
    if record.get("tau_factor") != 1.1 or "bound_control" in record:
        raise ValueError("start from the original 1.1 calibration, not another control")
    for key in ("schema", "kind", "profile", "data_identity", "code_identity"):
        if record[key] != original[key]:
            raise ValueError(f"original calibration {key} differs")

    reference = record["reference"]
    directory = Path(reference["directory"])
    dense = verify_completion(directory, require_current_source=True)
    if sha(directory / "completion.json") != reference["completion_sha256"]:
        raise ValueError("dense reference completion differs")
    if (
        dense["condition"] != "dense"
        or dense["seed"] != 0
        or dense["code_dim"] != 6
        or dense["kind"] not in {"train", "projection"}
        or dense["profile"] != record["profile"]
        or dense["data_identity"] != record["data_identity"]
        or any(dense[key] != reference[key] for key in ("identity", "kind", "steps"))
    ):
        raise ValueError("dense reference is not the original seed-zero six-code endpoint")
    prediction = float(read_json(directory / "calibration.json")["metrics"]["prediction"])
    if (
        not math.isfinite(prediction)
        or prediction <= 0
        or prediction != reference["prediction_loss"]
        or record["tau_prediction"] != 1.1 * prediction
    ):
        raise ValueError("original calibration loss or formula differs")

    result = {k: v for k, v in record.items() if k != "identity"}
    result.update(
        tau_factor=tau_factor,
        tau_prediction=tau_factor * prediction,
        policy="Explicit tighter-bound control on the same dense calibration loss",
        bound_control={
            "schema": "masked-mlp-prediction-bound-control-v1",
            "hypothesis": hypothesis.strip(),
            "generator_sha256": sha(Path(__file__)),
            "original_directory": str(source),
            "original_identity": identity,
            "original_completion_sha256": sha(source / "completion.json"),
        },
    )
    result["identity"] = canonical_fingerprint(result)
    output = output_dir.resolve()
    with file_lock(output.parent / f".{output.name}.lock"):
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"output is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "calibration.json", result)
        completion = dict(result, hashes={"calibration.json": sha(output / "calibration.json")})
        write_json(output / "completion.json", completion)
    return completion


def main() -> None:
    """Create a separate bound artifact for an explicitly declared control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tau-factor", required=True, type=float)
    parser.add_argument("--hypothesis", required=True)
    arguments = parser.parse_args()
    result = tighten_bound(
        arguments.calibration_dir,
        arguments.output_dir,
        tau_factor=arguments.tau_factor,
        hypothesis=arguments.hypothesis,
    )
    print(json.dumps({key: result[key] for key in ("identity", "tau_factor", "tau_prediction")}))


if __name__ == "__main__":
    main()
