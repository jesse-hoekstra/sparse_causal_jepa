"""Visual-to-visual evaluation: ``python scripts/eval_visual_to_visual.py <run_dir>``.

Loads a run's ``resolved_config.yaml`` + ``last.pt``, rebuilds the model through
the same factory the trainer used, and evaluates on a held-out split with the
evaluation-only geometric track alignment of Eqs. 137-138.

Writes ``metrics.json`` in the same key namespace as the state-to-state regime, so the
pipeline can read ``constraint_loss`` for tau calibration exactly as it does for
The state-to-state regime, and so ``mcc``/``shd`` sit on one axis across the ladder.
"""

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from scjepa.eval.visual_to_visual import evaluate_visual_to_visual
from scjepa.training.factory import build_dataset, build_model


def main() -> None:
    """Load run, evaluate, print report, save metrics.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Hydra run dir with resolved_config.yaml")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=17,
        help="held-out split offset; use a DIFFERENT value for the final test split",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run_dir / "resolved_config.yaml")
    model = build_model(cfg.model)  # pyright: ignore[reportArgumentType]
    payload = torch.load(args.run_dir / "last.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])

    eval_cfg = OmegaConf.merge(cfg.data, {"num_clips": args.episodes, "preload": None})
    dataset = build_dataset(eval_cfg, seed_offset=args.seed_offset)  # pyright: ignore[reportArgumentType]
    report = evaluate_visual_to_visual(
        model,  # pyright: ignore[reportArgumentType]
        dataset,
        batch_size=args.batch_size,
        max_batches=None,
        device=args.device,
        context_len=cfg.train.get("context_len"),
        lambda_logit=float(cfg.train.get("lambda_logit", 0.0)),
        # From the run's OWN config: tau_3 is calibrated on the constraint this
        # reports, so a mismatch with training silently rescales the bound.
        rollout_len=cfg.train.get("visual_rollout_len", None),
        lambda_roll=float(cfg.train.get("lambda_visual_rollout", 0.0)),
        resolution=int(cfg.data.resolution),
    )
    metrics = dict(report.metrics)
    metrics["step"] = float(payload.get("step", 0))
    metrics["eval_seed_offset"] = float(args.seed_offset)
    (args.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
