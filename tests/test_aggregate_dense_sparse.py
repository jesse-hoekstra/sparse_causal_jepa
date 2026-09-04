"""Tests for the paired eight-seed Experiment-1 figure builder."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from omegaconf import OmegaConf


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_dense_sparse.py"
    spec = importlib.util.spec_from_file_location("aggregate_dense_sparse", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


aggregate_dense_sparse: Any = _load_script()
TAU = 0.05


def _write_run(root: Path, seed: int, phase: str) -> Path:
    run_dir = root / f"seed{seed}" / phase
    run_dir.mkdir(parents=True, exist_ok=True)
    sparse = phase == "sparse"
    coordinate_std = [0.25 + seed * 1e-4, 0.25, 0.52, 0.52]
    pred_loss = 0.01 + seed * 0.001 + (0.002 if sparse else 0.0)
    rollout_raw = 0.02 + seed * 0.001 + (0.003 if sparse else 0.0)
    logit_weighted = 0.001 if sparse else 0.0001
    metrics: dict[str, object] = {
        "pred_loss": pred_loss,
        "trajectory_reconstruction_mse_k30": 0.04 + seed * 0.002 + 0.004 * sparse,
        "loss_rollout_t2_raw": rollout_raw,
        "loss_rollout_t2_weighted": rollout_raw,
        "logit_weighted": logit_weighted,
        "constraint_loss": pred_loss + rollout_raw + logit_weighted,
        "mcc": 0.80 + seed * 0.01 + (0.02 if sparse else 0.0),
        "step": 300_000,
        "seed": seed,
        "eval_seed_offset": 29,
        "num_samples": 5_000.0,
        "total_skips": seed if sparse else 0,
        "lambda_rollout_t2": 1.0,
        "num_rollout_t2_anchors": 8,
        "rollout_t2_horizon": 2,
        "oe_eval_horizon": 30,
        "oe_tolerance_nrmse": 0.1,
        "oe_coordinate_std": coordinate_std,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))

    config = OmegaConf.create(
        {
            "model": {
                "spartan_dense": not sparse,
                "spartan_identity": False,
                "spartan_embed_dim": 512,
                "spartan_layers": 3,
            },
            "data": {
                "name": "bounce",
                "seed": seed,
                "num_clips": 100_000,
                "preload": f"data/bounce_train_v2_100000_seed{seed}.pt",
            },
            "train": {
                "seed": seed,
                "steps": 300_000,
                "batch_size": 16,
                "sparsity_enabled": sparse,
                "sparsity_tau": TAU if sparse else "???",
                "lambda_logit": 1e-5,
                "lambda_rollout_t2": 1.0,
                "num_rollout_t2_anchors": 8,
                "rollout_t2_horizon": 2,
                "oe_eval_horizon": 30,
                "oe_tolerance_nrmse": 0.1,
                "oe_coordinate_std": coordinate_std,
            },
            "wandb": {"enabled": True, "project": "test", "run_tag": f"seed{seed}"},
            # A dirty suffix is valid provenance as long as it is common.
            "git_sha": "abc1234-dirty",
        }
    )
    OmegaConf.save(config, run_dir / "resolved_config.yaml")
    return run_dir


def _write_eight_seed_root(root: Path) -> None:
    # Deliberately create directories in reverse order; pairing must use seed.
    for seed in reversed(range(8)):
        _write_run(root, seed, "dense")
        _write_run(root, seed, "sparse")


def _update_metrics(run_dir: Path, **updates: object) -> None:
    path = run_dir / "metrics.json"
    loaded: object = json.loads(path.read_text())
    assert isinstance(loaded, dict)
    metrics = cast(dict[str, object], loaded)
    metrics.update(updates)
    path.write_text(json.dumps(metrics))


def test_writes_paired_constraint_and_paper_analogue_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "eight_seed"
    _write_eight_seed_root(root)

    comparison = aggregate_dense_sparse.build_comparison(root, TAU)
    assert [pair.seed for pair in comparison.pairs] == list(range(8))
    assert comparison.git_sha == "abc1234-dirty"
    assert abs(comparison.pairs[4].sparse.values["tf_t2_predictive_loss"] - (0.016 + 0.027)) < 1e-12

    output_dir = root / "aggregate"
    artifacts = aggregate_dense_sparse.write_artifacts(comparison, output_dir)
    assert artifacts["constraint_png"].name == "dense_sparse_constraint_mcc_boxplots.png"
    assert artifacts["constraint_pdf"].name == "dense_sparse_constraint_mcc_boxplots.pdf"
    assert artifacts["mse_png"].name == "dense_sparse_mse_mcc_boxplots.png"
    assert artifacts["mse_pdf"].name == "dense_sparse_mse_mcc_boxplots.pdf"
    assert artifacts["summary"].name == "dense_sparse_loss_mcc_summary.json"
    for key in ("constraint_png", "mse_png"):
        assert artifacts[key].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for key in ("constraint_pdf", "mse_pdf"):
        assert artifacts[key].read_bytes().startswith(b"%PDF")

    payload = json.loads(artifacts["summary"].read_text())
    assert payload["seeds"] == list(range(8))
    assert payload["provenance"]["sparsity_tau"] == TAU
    assert payload["provenance"]["eval_seed_offset"] == 29
    assert payload["provenance"]["num_samples"] == 5_000
    assert payload["pairs"][4]["seed"] == 4
    assert payload["pairs"][4]["dense"]["run_dir"].endswith("seed4/dense")
    for key in (
        "pred_loss",
        "trajectory_reconstruction_mse_k30",
        "loss_rollout_t2_raw",
        "loss_rollout_t2_weighted",
        "logit_weighted",
        "tf_t2_predictive_loss",
        "constraint_loss",
        "mcc",
    ):
        assert key in payload["metrics"]


def test_rejects_dense_calibration_split_instead_of_plotting_it(tmp_path: Path) -> None:
    root = tmp_path / "eight_seed"
    _write_eight_seed_root(root)
    _update_metrics(root / "seed3" / "dense", eval_seed_offset=17, num_samples=256.0)

    with pytest.raises(ValueError, match="required final evaluation"):
        aggregate_dense_sparse.build_comparison(root, TAU)


def test_rejects_unpaired_or_duplicate_recorded_seeds(tmp_path: Path) -> None:
    root = tmp_path / "eight_seed"
    _write_eight_seed_root(root)
    run_dir = root / "seed7" / "sparse"
    _update_metrics(run_dir, seed=6)
    config_path = run_dir / "resolved_config.yaml"
    config = OmegaConf.load(config_path)
    config.train.seed = 6
    config.data.seed = 6
    OmegaConf.save(config, config_path)

    with pytest.raises(ValueError, match="duplicate sparse seed 6"):
        aggregate_dense_sparse.build_comparison(root, TAU)


def test_rejects_unexpected_seed_directory(tmp_path: Path) -> None:
    root = tmp_path / "eight_seed"
    _write_eight_seed_root(root)
    (root / "seed8").mkdir()

    with pytest.raises(ValueError, match=r"unexpected seed directories: \[8\]"):
        aggregate_dense_sparse.build_comparison(root, TAU)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("phase", "not a sparse run"),
        ("tau", "!= expected tau"),
        ("coordinate_std", "oe_coordinate_std mismatch"),
        ("config", "resolved configuration mismatch"),
        ("git", "git_sha differs"),
    ],
)
def test_rejects_phase_and_paired_provenance_mismatches(
    tmp_path: Path, mutation: str, message: str
) -> None:
    root = tmp_path / "eight_seed"
    _write_eight_seed_root(root)
    run_dir = root / "seed2" / "sparse"
    config_path = run_dir / "resolved_config.yaml"
    config = OmegaConf.load(config_path)
    if mutation == "phase":
        config.model.spartan_identity = True
    elif mutation == "tau":
        config.train.sparsity_tau = 0.06
    elif mutation == "coordinate_std":
        config.train.oe_coordinate_std[0] = 0.99
        _update_metrics(run_dir, oe_coordinate_std=[0.99, 0.25, 0.52, 0.52])
    elif mutation == "config":
        config.model.spartan_embed_dim = 256
    else:
        config.git_sha = "different-dirty"
    OmegaConf.save(config, config_path)

    with pytest.raises(ValueError, match=message):
        aggregate_dense_sparse.build_comparison(root, TAU)


def test_rejects_nonfinite_metrics_and_inconsistent_constraint(tmp_path: Path) -> None:
    root = tmp_path / "eight_seed"
    _write_eight_seed_root(root)
    sparse = root / "seed5" / "sparse"
    _update_metrics(sparse, mcc=float("nan"))
    with pytest.raises(ValueError, match="'mcc' must be finite"):
        aggregate_dense_sparse.build_comparison(root, TAU)

    _write_run(root, 5, "sparse")
    _update_metrics(sparse, constraint_loss=999.0)
    with pytest.raises(ValueError, match="does not match its components"):
        aggregate_dense_sparse.build_comparison(root, TAU)


def test_requires_logit_contribution(tmp_path: Path) -> None:
    root = tmp_path / "eight_seed"
    _write_eight_seed_root(root)
    path = root / "seed0" / "dense" / "metrics.json"
    metrics = json.loads(path.read_text())
    del metrics["logit_weighted"]
    path.write_text(json.dumps(metrics))

    with pytest.raises(ValueError, match="missing required field 'logit_weighted'"):
        aggregate_dense_sparse.build_comparison(root, TAU)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (float("nan"), "must be finite"),
        (-0.01, "must be non-negative"),
    ],
)
def test_rejects_invalid_trajectory_reconstruction_mse(
    tmp_path: Path, value: float, message: str
) -> None:
    root = tmp_path / "eight_seed"
    _write_eight_seed_root(root)
    _update_metrics(root / "seed1" / "dense", trajectory_reconstruction_mse_k30=value)

    with pytest.raises(ValueError, match=message):
        aggregate_dense_sparse.build_comparison(root, TAU)
