"""Evaluate an Experiment-1 checkpoint on a fixed held-out split.

Loads ``resolved_config.yaml`` and ``last.pt``, evaluates the final
teacher-forcing-plus-T=2 constraint, graph recovery, MCC, and the no-gradient
K=30 sampled observational-equivalence diagnostic. The resulting tolerance
rate estimates approximate agreement on held-out trajectories; it is not a
proof of population observational equivalence.
"""

# pyright: reportUnknownMemberType=false

import argparse
import json
import math
from pathlib import Path
from typing import Any, TypedDict, cast

import matplotlib
import torch
from omegaconf import DictConfig, OmegaConf

from scjepa.eval import IdentifiabilityReport, evaluate_identifiability
from scjepa.training.factory import build_dataset, build_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ProtocolProvenance(TypedDict):
    """Fixed objective/evaluation values attached to every final report."""

    total_skips: int
    lambda_rollout_t2: float
    num_rollout_t2_anchors: int
    rollout_t2_horizon: int
    oe_eval_horizon: int
    oe_tolerance_nrmse: float
    oe_coordinate_std: list[float]


def _protocol_provenance(cfg: DictConfig, checkpoint: dict[str, Any]) -> ProtocolProvenance:
    """Extract the fixed protocol without any schedule-dependent state."""
    scales = cfg.train.get("oe_coordinate_std", None)
    if scales is None:
        raise ValueError("resolved config has no fixed train.oe_coordinate_std")
    values = [float(value) for value in scales]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("train.oe_coordinate_std must contain finite positive values")
    return {
        "total_skips": int(checkpoint.get("total_skips", 0)),
        "lambda_rollout_t2": float(cfg.train.lambda_rollout_t2),
        "num_rollout_t2_anchors": int(cfg.train.num_rollout_t2_anchors),
        "rollout_t2_horizon": int(cfg.train.rollout_t2_horizon),
        "oe_eval_horizon": int(cfg.train.oe_eval_horizon),
        "oe_tolerance_nrmse": float(cfg.train.oe_tolerance_nrmse),
        "oe_coordinate_std": values,
    }


def _require_complete_protocol(
    cfg: DictConfig,
    checkpoint: dict[str, Any],
    provenance: ProtocolProvenance,
) -> None:
    """Reject incomplete or non-primary checkpoints used for tau/final reports."""
    problems: list[str] = []
    if provenance["lambda_rollout_t2"] != 1.0:
        problems.append("lambda_rollout_t2 must equal 1.0")
    if provenance["num_rollout_t2_anchors"] != 8:
        problems.append("num_rollout_t2_anchors must equal 8")
    if provenance["rollout_t2_horizon"] != 2:
        problems.append("rollout_t2_horizon must equal 2")
    if provenance["oe_eval_horizon"] != 30:
        problems.append("oe_eval_horizon must equal 30")
    if int(checkpoint["step"]) != int(cfg.train.steps):
        problems.append(
            f"checkpoint step {int(checkpoint['step'])} != configured steps {int(cfg.train.steps)}"
        )
    obsolete = {
        "rollout_curriculum",
        "rollout_len",
        "lambda_roll",
    }.intersection(cfg.train.keys())
    if obsolete:
        problems.append(f"resolved config retains obsolete state rollout keys {sorted(obsolete)}")
    if problems:
        raise SystemExit("not a complete fixed T=2 protocol checkpoint: " + "; ".join(problems))


def main() -> None:
    """Load a run, evaluate it, save artifacts, and optionally update W&B."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path, help="Hydra run dir with resolved_config.yaml and last.pt"
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
        help="held-out split offset (use distinct validation and final-test values)",
    )
    parser.add_argument(
        "--require-complete-protocol",
        action="store_true",
        help="require the fixed 8-anchor T=2 protocol and a completed checkpoint",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run_dir / "resolved_config.yaml")
    if not isinstance(cfg, DictConfig):
        raise SystemExit("resolved_config.yaml must contain a mapping")
    model = build_model(cfg.model)
    payload = cast(
        dict[str, Any],
        torch.load(args.run_dir / "last.pt", map_location="cpu", weights_only=False),
    )
    provenance = _protocol_provenance(cfg, payload)
    if args.require_complete_protocol:
        _require_complete_protocol(cfg, payload, provenance)
    model.load_state_dict(payload["model"])

    eval_cfg = OmegaConf.merge(cfg.data, {"num_clips": args.episodes})
    assert isinstance(eval_cfg, DictConfig)
    dataset = build_dataset(eval_cfg, seed_offset=args.seed_offset)
    report = evaluate_identifiability(
        model,  # pyright: ignore[reportArgumentType]
        dataset,
        batch_size=args.batch_size,
        device=args.device,
        context_len=cfg.train.get("context_len", None),
        lambda_logit=float(cfg.train.lambda_logit),
        lambda_rollout_t2=provenance["lambda_rollout_t2"],
        num_rollout_t2_anchors=provenance["num_rollout_t2_anchors"],
        rollout_t2_horizon=provenance["rollout_t2_horizon"],
        oe_eval_horizon=provenance["oe_eval_horizon"],
        oe_tolerance_nrmse=provenance["oe_tolerance_nrmse"],
        oe_coordinate_std=provenance["oe_coordinate_std"],
    )

    print(f"identifiability report for {args.run_dir} (step {payload['step']}):")
    for key, value in report.metrics.items():
        print(f"  {key:>38}: {value:.10g}")
    for key, value in report.diagnostics.items():
        print(f"  {key:>38}: {value:.10g}")

    record: dict[str, object] = dict(report.metrics)
    record.update(report.diagnostics)
    record.update(
        {
            "step": int(payload["step"]),
            "seed": int(cfg.train.seed),
            "eval_seed_offset": int(args.seed_offset),
        }
    )
    record.update(provenance)
    metrics_path = args.run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2))
    print(f"  metrics saved to {metrics_path}")

    matrix_record: dict[str, object] = {
        "orientation": "nonlinear_r2[true_mass][learned_coordinate]",
        "nonlinear_r2": [[float(value) for value in row] for row in report.recovery_matrix],
    }
    matrix_path = args.run_dir / "mcc_matrix.json"
    matrix_path.write_text(json.dumps(matrix_record, indent=2))
    print(f"  MCC R^2 matrix saved to {matrix_path}")

    num_slots = report.true_parameters.shape[1]
    best_for_mass = report.recovery_matrix.argmax(dim=1)
    grid_fig, grid_axes = plt.subplots(
        num_slots,
        num_slots,
        figsize=(1.75 * num_slots, 1.75 * num_slots),
        squeeze=False,
    )
    for i in range(num_slots):
        for j in range(num_slots):
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

    _log_to_wandb(
        args.run_dir,
        cfg,
        report,
        grid_path,
        int(payload["step"]),
        args.seed_offset,
        provenance,
    )


def _log_to_wandb(
    run_dir: Path,
    cfg: DictConfig,
    report: IdentifiabilityReport,
    grid_path: Path,
    step: int,
    seed_offset: int,
    provenance: ProtocolProvenance,
) -> None:
    """Attach final metrics to the original W&B run when one exists."""
    id_file = run_dir / "wandb_run_id.txt"
    if not cfg.get("wandb", {}).get("enabled", False) or not id_file.exists():
        return
    try:
        import wandb
    except ImportError:
        print("  wandb not installed — skipping upload")
        return
    try:
        run = wandb.init(
            project=cfg.wandb.project,
            id=id_file.read_text().strip(),
            resume="allow",
            mode=cfg.wandb.get("mode", "online"),
        )
        payload: dict[str, object] = {
            f"final/{key}": value for key, value in {**report.metrics, **report.diagnostics}.items()
        }
        payload["final/recovery_grid"] = wandb.Image(str(grid_path))
        payload["final/eval_seed_offset"] = seed_offset
        for key, value in provenance.items():
            payload[f"final/{key}"] = value
        run.log(payload, step=step)
        run.finish()
        print(f"  recovery grid + final metrics logged to W&B run {run.id}")
    except Exception as error:  # logging must never fail evaluation
        print(f"  W&B upload skipped ({type(error).__name__}: {error})")


if __name__ == "__main__":
    main()
