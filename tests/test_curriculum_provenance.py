"""Reportability checks for the documented full-rollout curriculum."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_identifiability: Any = _load_script("eval_identifiability")
summarize_logit_sweep: Any = _load_script("summarize_logit_sweep")
aggregate_runs: Any = _load_script("aggregate_runs")


def _stage_records() -> list[dict[str, object]]:
    return [
        {
            "start_update": start,
            "rollout_len": horizon,
            "rollout_starts": list(starts),
            "gradient_cuts": list(cuts),
        }
        for start, horizon, starts, cuts in eval_identifiability.EXPECTED_ROLLOUT_CURRICULUM
    ]


def _config(*, steps: int = 355_000) -> DictConfig:
    cfg = OmegaConf.create(
        {
            "train": {
                "steps": steps,
                "rollout_len": 30,
                "lambda_roll": 1.0,
                "rollout_curriculum": _stage_records(),
            }
        }
    )
    assert isinstance(cfg, DictConfig)
    return cfg


def test_both_reportability_scripts_use_the_documented_curriculum() -> None:
    cfg = _config()
    expected = eval_identifiability.EXPECTED_ROLLOUT_CURRICULUM
    assert expected == summarize_logit_sweep.EXPECTED_ROLLOUT_CURRICULUM
    assert eval_identifiability._rollout_curriculum(cfg) == expected
    assert summarize_logit_sweep._curriculum(_stage_records()) == expected


def test_terminal_provenance_means_one_uncut_k30_update() -> None:
    cfg = _config(steps=115_001)
    checkpoint: dict[str, Any] = {
        "step": 115_001,
        "successful_updates": 115_001,
        "total_skips": 0,
    }
    provenance = eval_identifiability._curriculum_provenance(cfg, checkpoint)

    assert provenance["curriculum_current_rollout_len"] == 30
    assert provenance["curriculum_current_rollout_starts"] == [0]
    assert provenance["curriculum_current_gradient_cuts"] == []
    assert provenance["curriculum_terminal_reached"]
    assert provenance["terminal_rollout_updates"] == 1
    assert provenance["evaluation_rollout_len"] == 30
    assert provenance["evaluation_rollout_starts"] == [0]
    assert provenance["evaluation_gradient_cuts"] == []
    eval_identifiability._require_reportable_terminal_curriculum(cfg, checkpoint, provenance)


def test_reaching_terminal_boundary_without_update_is_not_reportable() -> None:
    cfg = _config(steps=115_000)
    checkpoint: dict[str, Any] = {
        "step": 115_000,
        "successful_updates": 115_000,
        "total_skips": 0,
    }
    provenance = eval_identifiability._curriculum_provenance(cfg, checkpoint)

    assert provenance["curriculum_terminal_reached"]
    assert provenance["terminal_rollout_updates"] == 0
    with pytest.raises(SystemExit, match="completed no K=30 update"):
        eval_identifiability._require_reportable_terminal_curriculum(cfg, checkpoint, provenance)


def test_old_two_field_stage_schema_is_rejected() -> None:
    legacy = [{"start_update": 0, "rollout_len": None}]
    with pytest.raises(ValueError, match="gradient_cuts"):
        summarize_logit_sweep._curriculum(legacy)


def test_aggregate_preserves_structured_curriculum_provenance() -> None:
    """List-valued D36 provenance is compared, not cast to a scalar."""
    record: dict[str, object] = {
        "step": 355_000,
        "successful_updates": 355_000,
        "total_skips": 0,
        "terminal_rollout_updates": 240_000,
        "rollout_curriculum": _stage_records(),
        "curriculum_current_rollout_starts": [0],
        "curriculum_current_gradient_cuts": [],
        "evaluation_rollout_starts": [0],
        "evaluation_gradient_cuts": [],
    }
    provenance = aggregate_runs.shared_provenance([record, dict(record)])
    assert provenance["rollout_curriculum"] == _stage_records()
    assert provenance["evaluation_rollout_starts"] == [0]


def test_aggregate_rejects_unequal_terminal_exposure() -> None:
    first: dict[str, object] = {
        "successful_updates": 355_000,
        "total_skips": 0,
        "terminal_rollout_updates": 240_000,
    }
    second = dict(first)
    second["successful_updates"] = 354_999
    second["total_skips"] = 1
    second["terminal_rollout_updates"] = 239_999
    with pytest.raises(ValueError, match="successful_updates"):
        aggregate_runs.shared_provenance([first, second])


def test_sweep_record_requires_uncut_terminal_training_and_evaluation(tmp_path: Path) -> None:
    run_dir = tmp_path / "dense"
    run_dir.mkdir()
    cfg = OmegaConf.create(
        {
            "git_sha": "test",
            "train": {
                "lambda_logit": 1e-3,
                "rollout_curriculum": _stage_records(),
                "rollout_len": 30,
                "lambda_roll": 1.0,
                "seed": 3,
                "steps": 355_000,
                "batch_size": 16,
                "lr": 5e-5,
                "context_len": 30,
                "sparsity_enabled": False,
            },
            "data": {"seed": 3, "num_clips": 100, "clip_len": 60},
            "model": {
                "num_slots": 5,
                "param_encoder_dim": 128,
                "spartan_layers": 2,
                "spartan_embed_dim": 64,
                "spartan_dense": True,
            },
        }
    )
    (run_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    metrics: dict[str, object] = {
        "logit_penalty": 2.1,
        "constraint_loss": 0.2,
        "pred_loss": 0.1,
        "mcc": 0.8,
        "mean_abs_logit": 0.1,
        "gate_entropy": 0.6,
        "mean_gate_probability": 0.5,
        "step": 355_000,
        "successful_updates": 355_000,
        "total_skips": 0,
        "terminal_rollout_updates": 240_000,
        "curriculum_terminal_reached": True,
        "successful_updates_checkpointed": True,
        "rollout_curriculum": _stage_records(),
        "curriculum_current_rollout_len": 30,
        "curriculum_current_rollout_starts": [0],
        "curriculum_current_gradient_cuts": [],
        "curriculum_terminal_start_update": 115_000,
        "evaluation_rollout_len": 30,
        "evaluation_rollout_starts": [0],
        "evaluation_gradient_cuts": [],
        "lambda_roll": 1.0,
        "num_samples": 5000,
        "eval_seed_offset": 23,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))

    record = summarize_logit_sweep._load_record(run_dir)
    assert record["protocol_355k_steps"]

    metrics["evaluation_gradient_cuts"] = [20]
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="final evaluation must use one rollout"):
        summarize_logit_sweep._load_record(run_dir)
