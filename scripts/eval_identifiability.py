"""Identifiability evaluation: ``python scripts/eval_identifiability.py <run_dir>``.

Loads a training run's ``resolved_config.yaml`` + ``last.pt``, rebuilds the
model through the same factory the trainer used, evaluates prediction, sparse
graphs and strict one-to-one mass recovery on a held-out split, and saves the
full pairwise recovery matrix next to the checkpoint.

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
from omegaconf import DictConfig, OmegaConf

from scjepa.eval import evaluate_identifiability
from scjepa.models.jepa import SCJepa
from scjepa.models.state_jepa import StateJepa
from scjepa.training.factory import build_dataset, build_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Load run, evaluate, print report, and save recovery artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path, help="Hydra run dir with resolved_config.yaml + last.pt"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="held-out trajectories (5000 matches Baumgartner App. F.1)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="evaluation device (defaults to CUDA when available)",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=17,
        help="held-out split offset (use a different value for validation and final test)",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run_dir / "resolved_config.yaml")
    if cfg.model.type != "states":
        raise SystemExit(
            "eval_identifiability currently supports model.type=states only: in the "
            "vision regime learned slots are not aligned to objects, so graph/recovery "
            "computed naively would be meaningless (see scjepa/eval/harness.py)."
        )
    model = build_model(cfg.model)
    assert isinstance(model, SCJepa | StateJepa)
    # Always deserialize through CPU so checkpoints written on a GPU machine
    # remain inspectable elsewhere; the harness moves the model to --device.
    payload = torch.load(args.run_dir / "last.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])

    eval_cfg = OmegaConf.merge(cfg.data, {"num_clips": args.episodes})
    assert isinstance(eval_cfg, DictConfig)
    dataset = build_dataset(eval_cfg, seed_offset=args.seed_offset)
    report = evaluate_identifiability(
        model,
        dataset,
        input_key=cfg.train.input_key,
        batch_size=args.batch_size,
        device=args.device,
        context_len=cfg.train.get("context_len", None),
        rollout_horizon=cfg.train.get("rollout_horizon", None),
        lambda_logit=cfg.train.get("lambda_logit", 0.0),
        prediction_matching=str(cfg.train.get("prediction_matching", "auto")),
        constraint_normalization=str(cfg.train.get("constraint_normalization", "auto")),
    )

    print(f"identifiability report for {args.run_dir} (step {payload['step']}):")
    for key, value in report.metrics.items():
        print(f"  {key:>22}: {value:.10g}")
    for key, value in report.diagnostics.items():
        print(f"  {key:>22}: {value:.10g}")

    # Machine-readable copy for cross-seed aggregation (scripts/aggregate_runs.py).
    record = dict(report.metrics)
    record.update(report.diagnostics)
    record.update(
        {
            "step": int(payload["step"]),
            "seed": int(cfg.train.seed),
            "eval_seed_offset": int(args.seed_offset),
        }
    )
    metrics_path = args.run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2))
    print(f"  metrics saved to {metrics_path}")

    alignment_record: dict[str, object] = {
        "target_to_learned": [int(value) for value in report.target_to_learned],
        "nonlinear_r2": [[float(value) for value in row] for row in report.recovery_matrix],
        "absolute_pearson": [
            [float(value) for value in row] for row in report.recovery_linear_matrix
        ],
    }
    alignment_path = args.run_dir / "recovery_alignment.json"
    alignment_path.write_text(json.dumps(alignment_record, indent=2))
    print(f"  recovery alignment saved to {alignment_path}")

    # Full global-coordinate recovery grid. Rows are physical masses, columns
    # are anonymous learned coordinates. The green outline shows the ONE
    # dataset-level Hungarian assignment; no diagonal correspondence is
    # assumed or rewarded.
    num_slots = report.true_parameters.shape[1]
    grid_fig, grid_axes = plt.subplots(
        num_slots, num_slots, figsize=(1.75 * num_slots, 1.75 * num_slots), squeeze=False
    )
    for i in range(num_slots):  # row: true mass of ball i
        for j in range(num_slots):  # column: learned global coordinate j
            cell = grid_axes[i][j]
            cell.scatter(
                report.learned_coordinates[:, j].numpy(),
                report.true_parameters[:, i].numpy(),
                s=2,
                alpha=0.2,
            )
            cell.set_xticks([])
            cell.set_yticks([])
            cell.text(
                0.04,
                0.94,
                f"$R^2={report.recovery_matrix[j, i]:.2f}$",
                transform=cell.transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )
            if int(report.target_to_learned[i]) == j:
                for spine in cell.spines.values():
                    spine.set_color("tab:green")
                    spine.set_linewidth(2.0)
            if j == 0:
                cell.set_ylabel(f"$m_{{{i + 1}}}$")
            if i == num_slots - 1:
                cell.set_xlabel(f"$\\hat\\theta_{{{j + 1}}}$")
    grid_fig.suptitle(
        "global one-to-one mass recovery "
        f"(green = frozen assignment, MCC={report.metrics['mass_mcc']:.3f})"
    )
    grid_fig.tight_layout()
    grid_path = args.run_dir / "recovery_grid.png"
    grid_fig.savefig(grid_path, dpi=150)
    print(f"  recovery grid saved to {grid_path}")


if __name__ == "__main__":
    main()
