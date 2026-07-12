"""Identifiability evaluation: ``python scripts/eval_identifiability.py <run_dir>``.

Loads a training run's ``resolved_config.yaml`` + ``last.pt``, rebuilds the
model through the same factory the trainer used, evaluates SHD / MMC /
prediction error on a held-out split (same generator, offset seed), prints the
report, and saves the marginal-recovery scatter next to the checkpoint.

Only meaningful for the GT-embedding regime (``model.type: states``), where
slot i ≡ object i by construction; refuses the vision regime until a
slot↔object alignment step exists (see scjepa/eval/harness.py).
"""

# pyright: reportUnknownMemberType=false
# (matplotlib's Axes API is partially typed; this file is a thin plotting shell)

import argparse
import json
from pathlib import Path

import matplotlib
import torch
from omegaconf import OmegaConf

from scjepa.eval import evaluate_identifiability
from scjepa.models.jepa import SCJepa
from scjepa.models.state_jepa import StateJepa
from scjepa.training.factory import build_dataset, build_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Load run, evaluate, print report, save recovery scatter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path, help="Hydra run dir with resolved_config.yaml + last.pt"
    )
    parser.add_argument("--episodes", type=int, default=512, help="held-out eval episodes")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run_dir / "resolved_config.yaml")
    if cfg.model.type != "states":
        raise SystemExit(
            "eval_identifiability currently supports model.type=states only: in the "
            "vision regime learned slots are not aligned to objects, so SHD/MMC "
            "computed naively would be meaningless (see scjepa/eval/harness.py)."
        )
    model = build_model(cfg.model)
    assert isinstance(model, SCJepa | StateJepa)
    payload = torch.load(args.run_dir / "last.pt", weights_only=False)
    model.load_state_dict(payload["model"])

    eval_cfg = OmegaConf.merge(cfg.data, {"num_clips": args.episodes})
    dataset = build_dataset(eval_cfg, seed_offset=17)  # pyright: ignore[reportArgumentType]
    report = evaluate_identifiability(
        model,
        dataset,
        input_key=cfg.train.input_key,
        batch_size=args.batch_size,
        context_len=cfg.train.get("context_len", None),
        rollout_horizon=cfg.train.get("rollout_horizon", None),
        lambda_logit=cfg.train.get("lambda_logit", 0.0),
    )

    print(f"identifiability report for {args.run_dir} (step {payload['step']}):")
    for key, value in report.metrics.items():
        print(f"  {key:>14}: {value:.4f}")

    # Machine-readable copy for cross-seed aggregation (scripts/aggregate_runs.py).
    record = dict(report.metrics)
    record.update(
        {
            "step": int(payload["step"]),
            "seed": int(cfg.train.seed),
            "recovery_best_dim": report.recovery_best_dim,
        }
    )
    metrics_path = args.run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2))
    print(f"  metrics saved to {metrics_path}")

    # Baumgartner Fig. 5/12-style grid: true mass of ball i (rows) vs slot j's
    # learned parameter (columns, at the globally best dim). Identification of
    # each mass IN ITS OWN slot = sharp diagonal, blobby off-diagonals.
    num_slots = report.per_slot_true.shape[1]
    dim = report.recovery_best_dim
    grid_fig, grid_axes = plt.subplots(
        num_slots, num_slots, figsize=(2.0 * num_slots, 2.0 * num_slots), squeeze=False
    )
    for i in range(num_slots):  # row: true mass of ball i
        for j in range(num_slots):  # column: learned param of slot j
            cell = grid_axes[i][j]
            cell.scatter(
                report.per_slot_learned[:, j, dim].numpy(),
                report.per_slot_true[:, i].numpy(),
                s=2,
                alpha=0.3,
            )
            cell.set_xticks([])
            cell.set_yticks([])
            if j == 0:
                cell.set_ylabel(f"$m_{{{i + 1}}}$")
            if i == num_slots - 1:
                cell.set_xlabel(f"slot {j + 1}")
    grid_fig.suptitle(f"per-slot recovery grid (learned dim {dim}); diagonal should be sharp")
    grid_fig.tight_layout()
    grid_path = args.run_dir / "recovery_grid.png"
    grid_fig.savefig(grid_path, dpi=150)
    print(f"  recovery grid saved to {grid_path}")


if __name__ == "__main__":
    main()
