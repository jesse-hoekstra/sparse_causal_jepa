"""Identifiability evaluation: ``python scripts/eval_identifiability.py <run_dir>``.

Loads a training run's ``resolved_config.yaml`` + ``last.pt``, rebuilds the
model through the same factory the trainer used, evaluates prediction, sparse
graphs and Baumgartner App. F.1 mass-recovery MCC on a held-out split, and saves
the full pairwise R² matrix next to the checkpoint.

Experiment 1: slot i ≡ tracked object i by construction (ζ = id).
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
    model = build_model(cfg.model)
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
        batch_size=args.batch_size,
        device=args.device,
        context_len=cfg.train.get("context_len", None),
        lambda_logit=cfg.train.get("lambda_logit", 0.0),
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

    # Rows are true masses, columns are learned coordinates (App. F.1's I x J).
    matrix_record: dict[str, object] = {
        "orientation": "nonlinear_r2[true_mass][learned_coordinate]",
        "nonlinear_r2": [[float(value) for value in row] for row in report.recovery_matrix],
    }
    matrix_path = args.run_dir / "mcc_matrix.json"
    matrix_path.write_text(json.dumps(matrix_record, indent=2))
    print(f"  MCC R^2 matrix saved to {matrix_path}")

    # Recovery grid: rows are physical masses, columns are learned parameter
    # coordinates. The green outline marks each row's argmax — the cell that
    # actually enters MCC = mean_i max_j R^2_ij. An off-diagonal green cell
    # means the mass is recoverable, but not from its own track's coordinate.
    num_slots = report.true_parameters.shape[1]
    best_for_mass = report.recovery_matrix.argmax(dim=1)  # per true mass: argmax_j
    grid_fig, grid_axes = plt.subplots(
        num_slots, num_slots, figsize=(1.75 * num_slots, 1.75 * num_slots), squeeze=False
    )
    for i in range(num_slots):  # row: true mass of ball i
        for j in range(num_slots):  # column: learned tracked parameter slot j
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
                f"$R^2={report.recovery_matrix[i, j]:.2f}$",
                transform=cell.transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )
            if int(best_for_mass[i]) == j:
                for spine in cell.spines.values():
                    spine.set_color("tab:green")
                    spine.set_linewidth(2.0)
            if j == 0:
                cell.set_ylabel(f"$m_{{{i + 1}}}$")
            if i == num_slots - 1:
                cell.set_xlabel(f"$\\hat\\theta_{{{j + 1}}}$")
    grid_fig.suptitle(f"mass recovery (green = row argmax, F.1 MCC={report.metrics['mcc']:.3f})")
    grid_fig.tight_layout()
    grid_path = args.run_dir / "recovery_grid.png"
    grid_fig.savefig(grid_path, dpi=150)
    print(f"  recovery grid saved to {grid_path}")


if __name__ == "__main__":
    main()
