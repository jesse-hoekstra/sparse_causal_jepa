"""Prepare the publication DGP, preserving its original numerical construction."""

import argparse
import json
from pathlib import Path

from diagnostics.shared import data, source_data


def main() -> None:
    """Create physical data, then express the same controls in the Paley chart."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--wandb-mode", default="disabled")
    parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="experiment1-data")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    physical = output.with_name(output.name + "-physical")
    source_data.prepare(
        source_data.PrepareConfig(
            str(physical),
            threads=args.threads,
            wandb_mode=args.wandb_mode,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group,
        ),
        resume=args.resume,
    )
    completed = data.prepare(
        data.PrepareConfig(
            str(output),
            str(physical),
            threads=args.threads,
            wandb_mode=args.wandb_mode,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group,
        ),
        resume=args.resume,
    )
    print(json.dumps({"directory": str(output), "identity": completed["identity"]}))


if __name__ == "__main__":
    main()
