"""Publish a completed diagnostic to its original W&B run without recomputation."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from diagnostics.joint_acquisition.tracking import publish_results
from scjepa.training.logging import WandbConfig, WandbLogger


def verified_results(output: Path) -> dict[str, Any]:
    """Require an intact completion manifest and matching saved run identity."""
    completed = json.loads((output / "completed.json").read_text())
    for name, expected in completed["outputs"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError("Completion artifact must stay within the result directory")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Completed artifact changed: {name}")
    result = cast(dict[str, Any], json.loads((output / "results.json").read_text()))
    tracking = json.loads((output / "tracking.json").read_text())
    if result["identity"] != completed["identity"] or result["tracking"] != tracking:
        raise ValueError("Saved result or tracking identity changed")
    if tracking["mode"] != "online":
        raise ValueError("Backfill requires the original online W&B run")
    return result


def main() -> None:
    """Append measurements to the existing run using a separate local logging directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--logging-dir", required=True, type=Path)
    args = parser.parse_args()
    output = cast(Path, args.output_dir).resolve()
    logging_dir = cast(Path, args.logging_dir).resolve()
    if logging_dir == output or logging_dir.is_relative_to(output):
        raise ValueError("Backfill logs must be outside the original result directory")
    result = verified_results(output)
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["identity"] != result["identity"]:
        raise ValueError("Protocol and results differ")
    source_protocol = json.loads(
        (Path(protocol["config"]["source_dir"]) / "protocol.json").read_text()
    )
    tracking = result["tracking"]
    logging_dir.mkdir(parents=True, exist_ok=False)
    logger = WandbLogger(
        WandbConfig(mode="online", project=tracking["project"], entity=tracking["entity"]),
        resolved_config={"diagnostic_identity": result["identity"], **protocol["config"]},
        dataset_identity=source_protocol["identity"],
        out_dir=logging_dir,
        resume_state=tracking,
    )
    try:
        publish_results(logger, result, output)
        logger.finish()
    except BaseException:
        logger.finish(exit_code=1)
        raise
    verified_results(output)
    print(f"Published saved diagnostic to W&B run {tracking['id']}; original hashes unchanged.")


if __name__ == "__main__":
    main()
