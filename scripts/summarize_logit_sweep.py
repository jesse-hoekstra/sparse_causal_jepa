"""Summarize dense lambda-logit runs without selecting on mass labels.

The dense sweep is a feasibility/Pareto screen, not a complete test of the
logit regularizer: its purpose is to reject coefficients that damage dense
prediction or fail to control attention logits.  The subsequent gated run is
the only test of whether a candidate preserves plasticity during pruning.

Usage:
    python scripts/summarize_logit_sweep.py outputs/lambda_logit_sweep_<tag>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _load_record(run_dir: Path) -> dict[str, Any]:
    """Load the coefficient and final evaluation from one dense run."""
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "resolved_config.yaml"
    if not metrics_path.exists() or not config_path.exists():
        missing = [str(path.name) for path in (metrics_path, config_path) if not path.exists()]
        raise FileNotFoundError(f"{run_dir}: missing {', '.join(missing)}")
    metrics = json.loads(metrics_path.read_text())
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"{config_path}: expected a mapping config")
    coefficient = float(cfg.train.lambda_logit)
    raw_logit = float(metrics["logit_penalty"])
    constraint = float(metrics["constraint_loss"])
    weighted_logit = float(metrics.get("logit_weighted", coefficient * raw_logit))
    teacher_forcing = float(metrics["pred_loss"])
    rollout_t2_weighted = float(metrics["loss_rollout_t2_weighted"])
    record: dict[str, Any] = {
        "run_dir": str(run_dir),
        "lambda_logit": coefficient,
        # Select against the complete predictive objective, not TF alone.
        "pred_loss": teacher_forcing + rollout_t2_weighted,
        "teacher_forcing_loss": teacher_forcing,
        "rollout_t2_raw": float(metrics["loss_rollout_t2_raw"]),
        "rollout_t2_weighted": rollout_t2_weighted,
        "logit_penalty": raw_logit,
        # Baumgartner's exp(z)+exp(-z) penalty has its minimum at 2.
        "logit_excess": max(raw_logit - 2.0, 0.0),
        "weighted_logit": weighted_logit,
        "weighted_logit_fraction": float(
            metrics.get("logit_fraction", weighted_logit / constraint if constraint else 0.0)
        ),
        "constraint_loss": constraint,
        # Ground-truth mass recovery is a diagnostic only.  It must not be the
        # hyperparameter-selection objective for an unsupervised claim.
        "mcc": float(metrics["mcc"]),
        "mean_abs_logit": float(metrics["mean_abs_logit"]),
        "gate_entropy": float(metrics["gate_entropy"]),
        "mean_gate_probability": float(metrics["mean_gate_probability"]),
        "step": int(metrics["step"]),
        "total_skips": int(metrics["total_skips"]),
        "lambda_rollout_t2": float(metrics["lambda_rollout_t2"]),
        "num_rollout_t2_anchors": int(metrics["num_rollout_t2_anchors"]),
        "rollout_t2_horizon": int(metrics["rollout_t2_horizon"]),
        "oe_eval_horizon": int(metrics["oe_eval_horizon"]),
        "oe_tolerance_nrmse": float(metrics["oe_tolerance_nrmse"]),
        "oe_coordinate_std": tuple(float(value) for value in metrics["oe_coordinate_std"]),
        "num_samples": int(metrics["num_samples"]),
        "eval_seed_offset": int(metrics["eval_seed_offset"]),
        "git_sha": str(cfg.get("git_sha", "unknown")),
        "train_seed": int(cfg.train.seed),
        "data_seed": int(cfg.data.seed),
        "train_steps": int(cfg.train.steps),
        "protocol_300k_steps": int(cfg.train.steps) == 300_000,
        "batch_size": int(cfg.train.batch_size),
        "learning_rate": float(cfg.train.lr),
        "num_clips": int(cfg.data.num_clips),
        "clip_len": int(cfg.data.clip_len),
        "context_len": int(cfg.train.context_len),
        "num_slots": int(cfg.model.num_slots),
        "param_encoder_dim": int(cfg.model.param_encoder_dim),
        "spartan_layers": int(cfg.model.spartan_layers),
        "spartan_embed_dim": int(cfg.model.spartan_embed_dim),
        "dense": bool(cfg.model.spartan_dense),
        "sparsity_enabled": bool(cfg.train.sparsity_enabled),
    }
    numeric = [value for value in record.values() if isinstance(value, int | float)]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError(f"{run_dir}: metrics contain non-finite values")
    if record["step"] != record["train_steps"]:
        raise ValueError(
            f"{run_dir}: checkpoint step {record['step']} != configured {record['train_steps']}"
        )
    if not record["dense"] or record["sparsity_enabled"]:
        raise ValueError(f"{run_dir}: expected a dense run with sparsity disabled")
    expected_protocol = (1.0, 8, 2, 30)
    actual_protocol = (
        record["lambda_rollout_t2"],
        record["num_rollout_t2_anchors"],
        record["rollout_t2_horizon"],
        record["oe_eval_horizon"],
    )
    if actual_protocol != expected_protocol:
        raise ValueError(
            f"{run_dir}: fixed T=2/OE protocol is {actual_protocol}, expected {expected_protocol}"
        )
    obsolete = {"rollout_curriculum", "rollout_len", "lambda_roll"}.intersection(cfg.train.keys())
    if obsolete:
        raise ValueError(f"{run_dir}: obsolete rollout config keys remain: {sorted(obsolete)}")
    return record


def _pareto(records: list[dict[str, Any]]) -> None:
    """Mark prediction/logit non-dominated runs (both quantities minimized)."""
    for candidate in records:
        candidate["pareto"] = not any(
            other["pred_loss"] <= candidate["pred_loss"]
            and other["logit_penalty"] <= candidate["logit_penalty"]
            and (
                other["pred_loss"] < candidate["pred_loss"]
                or other["logit_penalty"] < candidate["logit_penalty"]
            )
            for other in records
        )


def main() -> None:
    """Print and save a compact Pareto report for all completed sweep runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument(
        "--prediction-tolerance",
        type=float,
        default=0.05,
        help="maximum prediction degradation relative to lambda=0",
    )
    parser.add_argument(
        "--logit-reduction-fraction",
        type=float,
        default=0.9,
        help="fraction of the best admissible logit-excess reduction required",
    )
    args = parser.parse_args()
    if not 0.0 <= args.prediction_tolerance < 1.0:
        raise SystemExit("--prediction-tolerance must be in [0, 1)")
    if not 0.0 < args.logit_reduction_fraction <= 1.0:
        raise SystemExit("--logit-reduction-fraction must be in (0, 1]")

    run_dirs = sorted(args.sweep_dir.glob("lambda_*/dense"))
    if not run_dirs:
        raise SystemExit(f"no lambda_*/dense runs found under {args.sweep_dir}")
    try:
        records = [_load_record(run_dir) for run_dir in run_dirs]
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid or incomplete sweep: {error}") from error
    records.sort(key=lambda record: record["lambda_logit"])
    if len({record["lambda_logit"] for record in records}) != len(records):
        raise SystemExit("duplicate numeric lambda_logit values found")
    baselines = [record for record in records if record["lambda_logit"] == 0.0]
    if len(baselines) != 1:
        raise SystemExit("the sweep must contain exactly one lambda_logit=0 control")
    baseline = baselines[0]
    if baseline["pred_loss"] <= 0:
        raise SystemExit("lambda_logit=0 prediction loss must be positive")
    if not baseline["protocol_300k_steps"]:
        print(
            "WARNING: non-300k sweep; selection is smoke/ablation provenance, "
            "not the reportable protocol"
        )

    provenance_keys = (
        "git_sha",
        "train_seed",
        "data_seed",
        "train_steps",
        "batch_size",
        "learning_rate",
        "num_clips",
        "clip_len",
        "context_len",
        "lambda_rollout_t2",
        "num_rollout_t2_anchors",
        "rollout_t2_horizon",
        "oe_eval_horizon",
        "oe_tolerance_nrmse",
        "oe_coordinate_std",
        "num_slots",
        "param_encoder_dim",
        "spartan_layers",
        "spartan_embed_dim",
        "num_samples",
        "eval_seed_offset",
    )
    for key in provenance_keys:
        values = {record[key] for record in records}
        if len(values) != 1:
            raise SystemExit(f"inconsistent {key} across sweep runs: {sorted(values)!r}")

    for record in records:
        record["prediction_relative_to_baseline"] = (
            record["pred_loss"] / baseline["pred_loss"] - 1.0
        )
        record["prediction_admissible"] = (
            record["prediction_relative_to_baseline"] <= args.prediction_tolerance
        )
    _pareto(records)

    print(
        "lambda_logit | pred_loss  rel_zero | logit excess mean|z| entropy | "
        "     mcc | admissible pareto"
    )
    print("-" * 112)
    for record in records:
        print(
            f"{record['lambda_logit']:12.4g} | "
            f"{record['pred_loss']:9.6f} {record['prediction_relative_to_baseline']:8.2%} | "
            f"{record['logit_penalty']:5.2f} {record['logit_excess']:6.3f} "
            f"{record['mean_abs_logit']:7.3f} {record['gate_entropy']:7.3f} | "
            f"{record['mcc']:8.4f} | "
            f"{record['prediction_admissible']!s:>10} {record['pareto']!s:>6}"
        )

    admissible = [
        record
        for record in records
        if record["prediction_admissible"] and record["lambda_logit"] > 0
    ]
    if not admissible:
        raise SystemExit("no non-zero coefficient stayed within the prediction tolerance")
    best_excess = min(record["logit_excess"] for record in admissible)
    achievable_reduction = baseline["logit_excess"] - best_excess
    if achievable_reduction <= 0:
        raise SystemExit("no admissible non-zero coefficient reduced logit excess below lambda=0")
    target_excess = baseline["logit_excess"] - (
        args.logit_reduction_fraction * achievable_reduction
    )
    candidates = [
        record
        for record in admissible
        if record["pareto"] and record["logit_excess"] <= target_excess
    ]
    if not candidates:
        raise SystemExit("no Pareto coefficient reached the requested logit-reduction target")
    selected = min(candidates, key=lambda record: record["lambda_logit"])
    payload = {
        "selection_rule": {
            "prediction_tolerance": args.prediction_tolerance,
            "prediction_reference": "lambda_logit=0",
            "logit_penalty_floor": 2.0,
            "logit_reduction_fraction": args.logit_reduction_fraction,
            "target_logit_excess": target_excess,
            "uses_mass_labels": False,
            "protocol_300k_steps": bool(selected["protocol_300k_steps"]),
            "description": (
                "Among Pareto coefficients within the prediction tolerance, choose the smallest "
                "one achieving the requested fraction of the best logit-excess reduction. "
                "mcc is shown only as a validation diagnostic."
            ),
        },
        "pareto_candidates": [record["lambda_logit"] for record in candidates],
        "selected_lambda_logit": selected["lambda_logit"],
        "runs": records,
    }
    out_path = args.sweep_dir / "sweep_summary.json"
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(f"\nsummary saved to {out_path}")
    print(
        f"Selected lambda_logit={selected['lambda_logit']:g}: it is the smallest Pareto "
        f"coefficient reaching {args.logit_reduction_fraction:.0%} of the best admissible "
        "reduction in logit excess without exceeding the prediction tolerance. The full "
        "gated run must still cross tau and prune—dense runs cannot establish that."
    )


if __name__ == "__main__":
    main()
