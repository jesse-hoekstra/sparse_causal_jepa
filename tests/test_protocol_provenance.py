"""Reportability checks for the fixed T=2 state-to-state protocol."""

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


def _config(*, steps: int = 300_000) -> DictConfig:
    cfg = OmegaConf.create(
        {
            "train": {
                "steps": steps,
                "lambda_rollout_t2": 1.0,
                "num_rollout_t2_anchors": 8,
                "rollout_t2_horizon": 2,
                "oe_eval_horizon": 30,
                "oe_tolerance_nrmse": 0.1,
                "oe_coordinate_std": [0.25, 0.25, 0.52, 0.52],
            }
        }
    )
    assert isinstance(cfg, DictConfig)
    return cfg


def test_fixed_protocol_provenance_contains_no_schedule_state() -> None:
    cfg = _config()
    checkpoint: dict[str, Any] = {"step": 300_000, "total_skips": 2}
    provenance = eval_identifiability._protocol_provenance(cfg, checkpoint)
    assert provenance == {
        "total_skips": 2,
        "lambda_rollout_t2": 1.0,
        "num_rollout_t2_anchors": 8,
        "rollout_t2_horizon": 2,
        "oe_eval_horizon": 30,
        "oe_tolerance_nrmse": 0.1,
        "oe_coordinate_std": [0.25, 0.25, 0.52, 0.52],
    }
    assert not any("curriculum" in key or "stage" in key for key in provenance)
    eval_identifiability._require_complete_protocol(cfg, checkpoint, provenance)


def test_incomplete_checkpoint_is_not_reportable() -> None:
    cfg = _config()
    checkpoint: dict[str, Any] = {"step": 299_999}
    provenance = eval_identifiability._protocol_provenance(cfg, checkpoint)
    with pytest.raises(SystemExit, match="checkpoint step"):
        eval_identifiability._require_complete_protocol(cfg, checkpoint, provenance)


def test_reportability_rejects_obsolete_rollout_config() -> None:
    cfg = _config()
    cfg.train.rollout_curriculum = []
    checkpoint: dict[str, Any] = {"step": 300_000}
    provenance = eval_identifiability._protocol_provenance(cfg, checkpoint)
    with pytest.raises(SystemExit, match="obsolete"):
        eval_identifiability._require_complete_protocol(cfg, checkpoint, provenance)


def test_aggregate_compares_fixed_protocol_but_allows_health_to_vary() -> None:
    first: dict[str, object] = {
        "step": 300_000,
        "eval_seed_offset": 29,
        "num_samples": 5000,
        "lambda_rollout_t2": 1.0,
        "num_rollout_t2_anchors": 8,
        "rollout_t2_horizon": 2,
        "oe_eval_horizon": 30,
        "oe_tolerance_nrmse": 0.1,
        "total_skips": 0,
    }
    second = dict(first)
    second["total_skips"] = 3
    provenance = aggregate_runs.shared_provenance([first, second])
    assert provenance["rollout_t2_horizon"] == 2

    second["num_rollout_t2_anchors"] = 7
    with pytest.raises(ValueError, match="num_rollout_t2_anchors"):
        aggregate_runs.shared_provenance([first, second])


def _write_sweep_run(run_dir: Path) -> None:
    run_dir.mkdir()
    cfg = OmegaConf.create(
        {
            "git_sha": "test",
            "train": {
                "lambda_logit": 1e-3,
                "lambda_rollout_t2": 1.0,
                "num_rollout_t2_anchors": 8,
                "rollout_t2_horizon": 2,
                "oe_eval_horizon": 30,
                "oe_tolerance_nrmse": 0.1,
                "oe_coordinate_std": [0.25, 0.25, 0.52, 0.52],
                "seed": 3,
                "steps": 300_000,
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
        "pred_loss": 0.08,
        "loss_rollout_t2_raw": 0.02,
        "loss_rollout_t2_weighted": 0.02,
        "mcc": 0.8,
        "mean_abs_logit": 0.1,
        "gate_entropy": 0.6,
        "mean_gate_probability": 0.5,
        "step": 300_000,
        "total_skips": 0,
        "lambda_rollout_t2": 1.0,
        "num_rollout_t2_anchors": 8,
        "rollout_t2_horizon": 2,
        "oe_eval_horizon": 30,
        "oe_tolerance_nrmse": 0.1,
        "oe_coordinate_std": [0.25, 0.25, 0.52, 0.52],
        "num_samples": 5000,
        "eval_seed_offset": 23,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))


def test_sweep_record_requires_fixed_t2_protocol(tmp_path: Path) -> None:
    run_dir = tmp_path / "dense"
    _write_sweep_run(run_dir)
    record = summarize_logit_sweep._load_record(run_dir)
    assert record["protocol_300k_steps"]
    assert record["pred_loss"] == pytest.approx(0.10)

    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    cfg.train.rollout_len = 30
    (run_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    with pytest.raises(ValueError, match="obsolete"):
        summarize_logit_sweep._load_record(run_dir)
