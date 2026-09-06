"""Evaluate Experiment 2 and save concise metrics, slot maps, and mass recovery.

The state probe is fitted on TRAINING episodes and scored on held-out episodes.
True states, identities, and masses never affect encoder training or tau
selection. The normalized latent constraint is specific to a learned target
space; a calibrated tau is exploratory until physical recovery and tracking
also succeed. Existing W&B runs receive the two images and final metrics.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset

from scjepa.eval.visual_artifacts import save_visual_artifacts
from scjepa.eval.visual_to_visual import evaluate_visual_to_visual
from scjepa.models.visual_to_visual import VisualToVisualModel
from scjepa.training.factory import build_dataset, build_model


def require_noncollapsed(metrics: dict[str, float], variance_floor: float) -> None:
    """Reject only obvious label-free degeneracy before dense tau calibration.

    These are numerical screening thresholds, not evidence of object discovery.
    No simulator-labelled recovery or tracking metric is used to accept tau.
    """
    keys = (
        "constraint_loss",
        "target_variance",
        "target_temporal_variance",
        "target_effective_rank",
    )
    if any(not math.isfinite(metrics[key]) for key in keys):
        raise ValueError("non-finite dense predictive/collapse diagnostics; cannot calibrate tau")
    if metrics["constraint_loss"] <= 0 or metrics["target_variance"] <= variance_floor:
        raise ValueError(
            "dense target variance is at the normalization floor; cannot calibrate tau"
        )
    if metrics["target_temporal_variance"] <= 1e-10:
        raise ValueError("dense target is effectively frozen in time; cannot calibrate tau")
    if metrics["target_effective_rank"] <= 1.01:
        raise ValueError("dense target is effectively rank one; cannot calibrate tau")


def _log_to_wandb(
    run_dir: Path, cfg: DictConfig, metrics: dict[str, Any], images: tuple[Path, Path]
) -> None:
    """Append final metrics and two images to the training run, when configured."""
    id_file = run_dir / "wandb_run_id.txt"
    if not cfg.get("wandb", {}).get("enabled", False) or not id_file.exists():
        return
    try:
        import wandb

        run = wandb.init(
            project=cfg.wandb.project,
            id=id_file.read_text().strip(),
            resume="allow",
            mode=cfg.wandb.get("mode", "online"),
        )
        values: dict[str, object] = {f"final/{key}": value for key, value in metrics.items()}
        values["final/slot_tracks"] = wandb.Image(str(images[0]))
        values["final/recovery_grid"] = wandb.Image(str(images[1]))
        run.log(values, commit=True)
        run.finish()
    except Exception as error:
        print(f"W&B upload skipped ({type(error).__name__}: {error})")


def main() -> None:
    """Load the completed run, evaluate it, and write reproducible artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument(
        "--batch-size", type=int, default=None, help="defaults to training batch size"
    )
    parser.add_argument(
        "--probe-episodes",
        type=int,
        default=128,
        help="training episodes to fit the frozen state probe",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--seed-offset", type=int, default=17, help="17 for calibration, 29 for final test"
    )
    parser.add_argument(
        "--require-noncollapsed",
        action="store_true",
        help="screen gross label-free collapse before tau calibration",
    )
    parser.add_argument(
        "--require-complete-protocol",
        action="store_true",
        help="require completed checkpoint and eight-anchor T=2 objective",
    )
    args = parser.parse_args()
    if args.seed_offset == 0 or args.episodes < 2 or args.probe_episodes < 2:
        parser.error("use a held-out nonzero seed offset and at least two test/probe episodes")
    cfg = OmegaConf.load(args.run_dir / "resolved_config.yaml")
    if not isinstance(cfg, DictConfig):
        raise ValueError("resolved_config.yaml must contain a mapping")
    model = build_model(cfg.model)
    if not isinstance(model, VisualToVisualModel):
        raise ValueError("checkpoint must use the visual_to_visual regime")
    payload = cast(
        dict[str, Any], torch.load(args.run_dir / "last.pt", map_location="cpu", weights_only=False)
    )
    model.load_state_dict(payload["model"])
    batch_size = int(cfg.train.batch_size) if args.batch_size is None else args.batch_size
    if batch_size < 1:
        parser.error("batch size must be positive")
    if args.require_complete_protocol:
        if int(payload["step"]) != int(cfg.train.steps):
            raise ValueError("checkpoint has not completed the configured training steps")
        if (
            float(cfg.train.lambda_rollout_t2),
            int(cfg.train.num_rollout_t2_anchors),
            int(cfg.train.rollout_t2_horizon),
            int(cfg.train.oe_eval_horizon),
        ) != (1.0, 8, 2, 30):
            raise ValueError(
                "expected fixed eight-anchor T=2 training and K=30 evaluation protocol"
            )
        if "visual_rollout_len" in cfg.train or "lambda_visual_rollout" in cfg.train:
            raise ValueError("checkpoint uses the retired visual full-horizon training objective")
    eval_cfg = OmegaConf.merge(
        cfg.data, {"num_clips": args.episodes, "preload": None, "cache": False}
    )
    train_cfg = OmegaConf.merge(cfg.data, {"cache": False})
    assert isinstance(eval_cfg, DictConfig)
    assert isinstance(train_cfg, DictConfig)
    dataset = build_dataset(eval_cfg, seed_offset=args.seed_offset)
    training = build_dataset(train_cfg)
    probe_count = min(args.probe_episodes, int(cfg.data.num_clips))
    probe_dataset = Subset(training, range(probe_count))
    report = evaluate_visual_to_visual(
        model,
        dataset,
        batch_size=batch_size,
        max_batches=None,
        device=args.device,
        context_len=cfg.train.get("context_len"),
        lambda_logit=float(cfg.train.lambda_logit),
        lambda_rollout_t2=float(cfg.train.lambda_rollout_t2),
        num_rollout_t2_anchors=int(cfg.train.num_rollout_t2_anchors),
        rollout_t2_horizon=int(cfg.train.rollout_t2_horizon),
        oe_eval_horizon=int(cfg.train.oe_eval_horizon),
        resolution=int(cfg.data.resolution),
        probe_dataset=probe_dataset,
        max_probe_batches=math.ceil(probe_count / batch_size),
    )
    metrics: dict[str, Any] = dict(report.metrics)
    metrics.update(
        step=int(payload.get("step", 0)),
        total_skips=int(payload.get("total_skips", 0)),
        eval_seed_offset=args.seed_offset,
        eval_batch_size=batch_size,
        seed=int(cfg.train.seed),
        lambda_rollout_t2=float(cfg.train.lambda_rollout_t2),
        num_rollout_t2_anchors=int(cfg.train.num_rollout_t2_anchors),
        rollout_t2_horizon=int(cfg.train.rollout_t2_horizon),
        oe_eval_horizon=int(cfg.train.oe_eval_horizon),
        probe_fit_episodes=probe_count,
        target_matching="detached_context_slot_trajectory",
        objective_version="visual_tf_t2_v1",
        constraint_interpretation=(
            "per-model latent scale; does not establish shared physical fidelity"
        ),
    )
    (args.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.run_dir / "mcc_matrix.json").write_text(
        json.dumps(
            {
                "orientation": "nonlinear_r2[true_mass][learned_coordinate]",
                "nonlinear_r2": [[float(value) for value in row] for row in report.recovery_matrix],
            },
            indent=2,
        )
    )
    images = save_visual_artifacts(report, args.run_dir)
    _log_to_wandb(args.run_dir, cfg, metrics, images)
    print(json.dumps(metrics, indent=2))
    print(
        f"wrote metrics.json, mcc_matrix.json, slot_tracks.png, recovery_grid.png in {args.run_dir}"
    )
    if args.require_noncollapsed:
        require_noncollapsed(report.metrics, model.variance_floor)


if __name__ == "__main__":
    main()
