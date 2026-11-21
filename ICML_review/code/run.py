"""Run a single predeclared review diagnostic from the checkout."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    """Write outputs into the private directory created by the launch script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("donors", "chart", "halving", "prune"))
    parser.add_argument("seed", type=int, choices=range(5))
    parser.add_argument("detail")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record: dict[str, object]
    if args.kind == "halving":
        from .halving import run

        record = run(args.detail, args.seed)
    elif args.kind == "prune":
        from .prune import run as prune

        record = prune(args.seed)
    elif args.kind == "chart":
        from .chart import run as chart

        record = chart(args.seed)
    else:
        import torch

        from .donors import run as donors

        torch.set_num_threads(2)
        record = donors(args.seed, args.detail)
        path = Path(str(record["checkpoint"]))
        with path.open("rb") as stream:
            record["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    args.output.joinpath("metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    print("Completed", args.kind, args.seed, args.detail, flush=True)


if __name__ == "__main__":
    main()
