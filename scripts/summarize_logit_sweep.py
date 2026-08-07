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
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

RolloutStage = tuple[int, int | None, tuple[int, ...], tuple[int, ...]]

EXPECTED_ROLLOUT_CURRICULUM: tuple[RolloutStage, ...] = (
    (0, None, (), ()),
    (10_000, 2, (0, 10, 20), ()),
    (20_000, 5, (0, 10, 20), ()),
    (30_000, 10, (0, 10, 20), ()),
    (50_000, 30, (0,), (10, 20)),
    (70_000, 30, (0,), (15,)),
    (85_000, 30, (0,), (20,)),
    (100_000, 30, (0,), (25,)),
    (115_000, 30, (0,), ()),
)


def _integer_list(value: object, *, field: str, stage_index: int) -> tuple[int, ...]:
    """Validate and freeze one stage's starts or gradient cuts."""
    if not isinstance(value, list):
        raise ValueError(f"rollout curriculum {field} {stage_index} must be a list")
    integers: list[int] = []
    for item_index, item in enumerate(cast(list[object], value)):
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(
                f"rollout curriculum {field} {stage_index}[{item_index}] must be an integer"
            )
        integers.append(item)
    return tuple(integers)


def _curriculum(value: object) -> tuple[RolloutStage, ...]:
    """Normalize a JSON/OmegaConf curriculum for validation and comparison."""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, list):
        raise ValueError("rollout_curriculum must be a list")
    stages_value = cast(list[object], value)
    stages: list[RolloutStage] = []
    for index, stage in enumerate(stages_value):
        if not isinstance(stage, dict):
            raise ValueError(f"rollout curriculum stage {index} must be a mapping")
        stage_mapping = cast(dict[str, object], stage)
        expected_fields = {"start_update", "rollout_len", "rollout_starts", "gradient_cuts"}
        if set(stage_mapping) != expected_fields:
            raise ValueError(
                f"rollout curriculum stage {index} must contain exactly "
                "start_update, rollout_len, rollout_starts, and gradient_cuts"
            )
        start_update = stage_mapping["start_update"]
        raw_horizon = stage_mapping["rollout_len"]
        if not isinstance(start_update, int) or isinstance(start_update, bool):
            raise ValueError(f"rollout curriculum start_update {index} must be an integer")
        if raw_horizon is not None and (
            not isinstance(raw_horizon, int) or isinstance(raw_horizon, bool)
        ):
            raise ValueError(f"rollout curriculum rollout_len {index} must be an integer or null")
        starts = _integer_list(
            stage_mapping["rollout_starts"], field="rollout_starts", stage_index=index
        )
        cuts = _integer_list(
            stage_mapping["gradient_cuts"], field="gradient_cuts", stage_index=index
        )
        stages.append((start_update, raw_horizon, starts, cuts))
    return tuple(stages)


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
    metric_curriculum = _curriculum(metrics["rollout_curriculum"])
    config_curriculum = _curriculum(cfg.train.rollout_curriculum)
    record: dict[str, Any] = {
        "run_dir": str(run_dir),
        "lambda_logit": coefficient,
        "pred_loss": float(metrics["pred_loss"]),
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
        "successful_updates": int(metrics["successful_updates"]),
        "total_skips": int(metrics["total_skips"]),
        "terminal_rollout_updates": int(metrics["terminal_rollout_updates"]),
        "terminal_curriculum_reached": bool(metrics["curriculum_terminal_reached"]),
        "successful_updates_checkpointed": bool(metrics["successful_updates_checkpointed"]),
        "rollout_curriculum": metric_curriculum,
        "current_rollout_len": int(metrics["curriculum_current_rollout_len"]),
        "current_rollout_starts": tuple(
            int(value) for value in metrics["curriculum_current_rollout_starts"]
        ),
        "current_gradient_cuts": tuple(
            int(value) for value in metrics["curriculum_current_gradient_cuts"]
        ),
        "terminal_stage_start_update": int(metrics["curriculum_terminal_start_update"]),
        "evaluation_rollout_len": int(metrics["evaluation_rollout_len"]),
        "evaluation_rollout_starts": tuple(
            int(value) for value in metrics["evaluation_rollout_starts"]
        ),
        "evaluation_gradient_cuts": tuple(
            int(value) for value in metrics["evaluation_gradient_cuts"]
        ),
        "lambda_roll": float(metrics["lambda_roll"]),
        "num_samples": int(metrics["num_samples"]),
        "eval_seed_offset": int(metrics["eval_seed_offset"]),
        "git_sha": str(cfg.get("git_sha", "unknown")),
        "train_seed": int(cfg.train.seed),
        "data_seed": int(cfg.data.seed),
        "train_steps": int(cfg.train.steps),
        "protocol_355k_steps": int(cfg.train.steps) == 355_000,
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
    if metric_curriculum != config_curriculum:
        raise ValueError(f"{run_dir}: metrics/config rollout curricula differ")
    if metric_curriculum != EXPECTED_ROLLOUT_CURRICULUM:
        raise ValueError(
            f"{run_dir}: curriculum is {metric_curriculum!r}, "
            f"expected {EXPECTED_ROLLOUT_CURRICULUM!r}"
        )
    if record["evaluation_rollout_len"] != 30 or record["lambda_roll"] != 1.0:
        raise ValueError(f"{run_dir}: final evaluation must use K=30 and lambda_roll=1")
    if record["evaluation_rollout_starts"] != (0,) or record["evaluation_gradient_cuts"]:
        raise ValueError(
            f"{run_dir}: final evaluation must use one rollout from start 0 with no gradient cuts"
        )
    if metric_curriculum[-1][2] != (0,) or metric_curriculum[-1][3]:
        raise ValueError(
            f"{run_dir}: terminal training stage must use one rollout from start 0 "
            "with no gradient cuts"
        )
    if record["terminal_stage_start_update"] != metric_curriculum[-1][0]:
        raise ValueError(f"{run_dir}: terminal-stage boundary disagrees with the curriculum")
    if (
        record["current_rollout_len"] != 30
        or record["current_rollout_starts"] != (0,)
        or record["current_gradient_cuts"]
    ):
        raise ValueError(f"{run_dir}: checkpoint did not finish in the uncut one-window K=30 stage")
    if not record["successful_updates_checkpointed"]:
        raise ValueError(f"{run_dir}: checkpoint did not persist successful_updates")
    if not record["terminal_curriculum_reached"] or record["terminal_rollout_updates"] < 1:
        raise ValueError(f"{run_dir}: checkpoint completed no update at terminal K=30")
    expected_terminal_updates = max(
        record["successful_updates"] - record["terminal_stage_start_update"], 0
    )
    if record["terminal_rollout_updates"] != expected_terminal_updates:
        raise ValueError(f"{run_dir}: terminal rollout update count is inconsistent")
    if record["successful_updates"] + record["total_skips"] != record["step"]:
        raise ValueError(
            f"{run_dir}: successful_updates + total_skips does not equal attempted step"
        )
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
    if not baseline["protocol_355k_steps"]:
        print(
            "WARNING: non-355k sweep; selection is smoke/ablation provenance, "
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
        "rollout_curriculum",
        "terminal_stage_start_update",
        "evaluation_rollout_len",
        "evaluation_rollout_starts",
        "evaluation_gradient_cuts",
        "lambda_roll",
        "num_slots",
        "param_encoder_dim",
        "spartan_layers",
        "spartan_embed_dim",
        "num_samples",
        "eval_seed_offset",
        # D36 phases advance on accepted updates, not attempted batches. A
        # coefficient comparison must therefore have equal terminal exposure.
        "successful_updates",
        "total_skips",
        "terminal_rollout_updates",
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
            "protocol_355k_steps": bool(selected["protocol_355k_steps"]),
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
