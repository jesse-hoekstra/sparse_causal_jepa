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
from typing import Any, TypedDict, cast

import matplotlib
import torch
from omegaconf import DictConfig, OmegaConf

from scjepa.eval import IdentifiabilityReport, evaluate_identifiability
from scjepa.training.factory import build_dataset, build_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPECTED_ROLLOUT_CURRICULUM: tuple[tuple[int, int | None], ...] = (
    (0, None),
    (10_000, 2),
    (15_000, 5),
    (25_000, 10),
    (40_000, 20),
    (60_000, 30),
)


class CurriculumProvenance(TypedDict):
    """Scalar and structured evidence tying final metrics to the trained horizon."""

    successful_updates: int
    total_skips: int
    successful_updates_checkpointed: bool
    rollout_curriculum: list[dict[str, int | None]]
    curriculum_current_rollout_len: int | None
    curriculum_terminal_start_update: int
    curriculum_terminal_reached: bool
    terminal_rollout_updates: int
    evaluation_rollout_len: int | None
    lambda_roll: float


def _rollout_curriculum(cfg: DictConfig) -> tuple[tuple[int, int | None], ...]:
    """Read the accepted-update curriculum from a resolved run config."""
    raw = cfg.train.get("rollout_curriculum", None)
    if raw is None:
        return ()
    container_value = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(container_value, list):
        raise ValueError("train.rollout_curriculum must be a list")
    container = cast(list[object], container_value)
    stages: list[tuple[int, int | None]] = []
    for index, stage in enumerate(container):
        if not isinstance(stage, dict):
            raise ValueError(f"rollout curriculum stage {index} must be a mapping")
        stage_mapping = cast(dict[str, object], stage)
        if set(stage_mapping) != {"start_update", "rollout_len"}:
            raise ValueError(
                f"rollout curriculum stage {index} must contain start_update and rollout_len"
            )
        start_update = stage_mapping["start_update"]
        raw_horizon = stage_mapping["rollout_len"]
        if not isinstance(start_update, int) or isinstance(start_update, bool):
            raise ValueError(f"rollout curriculum start_update {index} must be an integer")
        if raw_horizon is not None and (
            not isinstance(raw_horizon, int) or isinstance(raw_horizon, bool)
        ):
            raise ValueError(f"rollout curriculum rollout_len {index} must be an integer or null")
        stages.append((start_update, raw_horizon))
    return tuple(stages)


def _curriculum_provenance(cfg: DictConfig, checkpoint: dict[str, Any]) -> CurriculumProvenance:
    """Describe which accepted-update stage the evaluated checkpoint reached."""
    stages = _rollout_curriculum(cfg)
    step = int(checkpoint["step"])
    total_skips = int(checkpoint.get("total_skips", 0))
    counter_checkpointed = "successful_updates" in checkpoint
    successful_updates = int(checkpoint.get("successful_updates", max(step - total_skips, 0)))
    terminal_rollout = cfg.train.get("rollout_len", None)
    terminal_rollout = None if terminal_rollout is None else int(terminal_rollout)

    current_rollout: int | None = terminal_rollout
    if stages:
        current_rollout = None
        for start_update, rollout_len in stages:
            if start_update > successful_updates:
                break
            current_rollout = rollout_len
    terminal_start = stages[-1][0] if stages else 0
    terminal_stage_matches = bool(stages and stages[-1][1] == terminal_rollout)
    terminal_reached = terminal_stage_matches and successful_updates >= terminal_start
    terminal_updates = max(successful_updates - terminal_start, 0) if terminal_reached else 0
    return {
        "successful_updates": successful_updates,
        "total_skips": total_skips,
        "successful_updates_checkpointed": counter_checkpointed,
        "rollout_curriculum": [
            {"start_update": start_update, "rollout_len": rollout_len}
            for start_update, rollout_len in stages
        ],
        "curriculum_current_rollout_len": current_rollout,
        "curriculum_terminal_start_update": terminal_start,
        "curriculum_terminal_reached": terminal_reached,
        "terminal_rollout_updates": terminal_updates,
        "evaluation_rollout_len": terminal_rollout,
        "lambda_roll": float(cfg.train.get("lambda_roll", 0.0)),
    }


def _require_reportable_terminal_curriculum(
    cfg: DictConfig, checkpoint: dict[str, Any], provenance: CurriculumProvenance
) -> None:
    """Reject tau/final reports that never trained under the terminal K=30 objective."""
    stages = _rollout_curriculum(cfg)
    problems: list[str] = []
    if stages != EXPECTED_ROLLOUT_CURRICULUM:
        problems.append(f"curriculum is {stages!r}, expected {EXPECTED_ROLLOUT_CURRICULUM!r}")
    if cfg.train.get("rollout_len", None) != 30:
        problems.append(
            f"terminal rollout_len is {cfg.train.get('rollout_len', None)!r}, expected 30"
        )
    if float(cfg.train.get("lambda_roll", 0.0)) != 1.0:
        problems.append(f"lambda_roll is {cfg.train.get('lambda_roll', None)!r}, expected 1.0")
    if not bool(provenance["successful_updates_checkpointed"]):
        problems.append("checkpoint has no explicit successful_updates counter")
    if provenance["successful_updates"] + provenance["total_skips"] != int(checkpoint["step"]):
        problems.append("successful_updates + total_skips does not equal checkpoint step")
    if not bool(provenance["curriculum_terminal_reached"]):
        problems.append(
            "terminal curriculum stage was not reached "
            f"(successful_updates={provenance['successful_updates']})"
        )
    elif int(provenance["terminal_rollout_updates"]) < 1:
        problems.append("checkpoint reached the K=30 boundary but completed no K=30 update")
    if int(checkpoint["step"]) != int(cfg.train.steps):
        problems.append(
            f"checkpoint step {int(checkpoint['step'])} != configured train.steps "
            f"{int(cfg.train.steps)}"
        )
    if problems:
        raise SystemExit("not a reportable terminal-curriculum checkpoint: " + "; ".join(problems))


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
    parser.add_argument(
        "--require-terminal-curriculum",
        action="store_true",
        help=(
            "require the exact accepted-update curriculum, at least one successful K=30 "
            "update, and a completed checkpoint before emitting reportable metrics"
        ),
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run_dir / "resolved_config.yaml")
    if not isinstance(cfg, DictConfig):
        raise SystemExit("resolved_config.yaml must contain a mapping")
    model = build_model(cfg.model)
    # Always deserialize through CPU so checkpoints written on a GPU machine
    # remain inspectable elsewhere; the harness moves the model to --device.
    payload = cast(
        dict[str, Any],
        torch.load(args.run_dir / "last.pt", map_location="cpu", weights_only=False),
    )
    provenance = _curriculum_provenance(cfg, payload)
    if args.require_terminal_curriculum:
        _require_reportable_terminal_curriculum(cfg, payload, provenance)
    model.load_state_dict(payload["model"])

    eval_cfg = OmegaConf.merge(cfg.data, {"num_clips": args.episodes})
    assert isinstance(eval_cfg, DictConfig)
    dataset = build_dataset(eval_cfg, seed_offset=args.seed_offset)
    # Final evaluation deliberately uses the configured TERMINAL horizon, not
    # the live stage at the checkpoint. Reportable callers additionally require
    # proof above that training reached and updated under this exact K=30 stage.
    report = evaluate_identifiability(
        model,  # pyright: ignore[reportArgumentType]
        dataset,
        batch_size=args.batch_size,
        device=args.device,
        context_len=cfg.train.get("context_len", None),
        lambda_logit=cfg.train.get("lambda_logit", 0.0),
        rollout_len=cfg.train.get("rollout_len", None),
        lambda_roll=float(cfg.train.get("lambda_roll", 0.0)),
    )

    print(f"identifiability report for {args.run_dir} (step {payload['step']}):")
    print(f"  {'successful_updates':>22}: {provenance['successful_updates']}")
    print(f"  {'terminal_rollout_updates':>22}: {provenance['terminal_rollout_updates']}")
    print(f"  {'evaluation_rollout_len':>22}: {provenance['evaluation_rollout_len']}")
    for key, value in report.metrics.items():
        print(f"  {key:>22}: {value:.10g}")
    for key, value in report.diagnostics.items():
        print(f"  {key:>22}: {value:.10g}")

    # Machine-readable copy for cross-seed aggregation (scripts/aggregate_runs.py).
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
    report: object,
    grid_path: Path,
    step: int,
    seed_offset: int,
    provenance: CurriculumProvenance,
) -> None:
    """Attach the final eval + recovery grid to the TRAINING run's W&B page.

    ``scripts/train.py`` writes ``wandb_run_id.txt`` next to the checkpoint, so
    this resumes that run rather than creating a detached second entry. Silently
    does nothing when the run was not tracked (``wandb.enabled=false``, CI, or
    an older run dir without the id file) — evaluation must never fail because
    of logging.
    """
    id_file = run_dir / "wandb_run_id.txt"
    if not cfg.get("wandb", {}).get("enabled", False) or not id_file.exists():
        return
    try:
        import wandb
    except ImportError:
        print("  wandb not installed — skipping upload")
        return
    assert isinstance(report, IdentifiabilityReport)
    try:
        run = wandb.init(
            project=cfg.wandb.project,
            id=id_file.read_text().strip(),
            resume="allow",
            mode=cfg.wandb.get("mode", "online"),
        )
        # Prefix distinguishes the 5000-episode final numbers from the periodic
        # eval/* curves logged during training on a much smaller sample.
        payload: dict[str, object] = {
            f"final/{key}": value for key, value in {**report.metrics, **report.diagnostics}.items()
        }
        payload["final/recovery_grid"] = wandb.Image(str(grid_path))
        payload["final/eval_seed_offset"] = seed_offset
        for key in (
            "successful_updates",
            "total_skips",
            "curriculum_terminal_start_update",
            "curriculum_terminal_reached",
            "terminal_rollout_updates",
            "evaluation_rollout_len",
            "lambda_roll",
        ):
            payload[f"final/{key}"] = provenance[key]
        run.log(payload, step=step)
        run.finish()
        print(f"  recovery grid + final metrics logged to W&B run {run.id}")
    except Exception as error:  # logging must never fail the evaluation
        print(f"  W&B upload skipped ({type(error).__name__}: {error})")


if __name__ == "__main__":
    main()
