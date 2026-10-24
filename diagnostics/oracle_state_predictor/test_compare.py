"""Paired graph-comparison provenance, reference reuse and controller continuation."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from diagnostics.oracle_state_predictor.compare import (
    ComparisonConfig,
    predictor_config,
    read_checkpoint,
    run_comparison,
    sha256,
    sparsity_loss,
    state_identity,
    train_condition,
)
from diagnostics.oracle_state_predictor.data import build_data
from diagnostics.oracle_state_predictor.graphs import build_predictor
from diagnostics.oracle_state_predictor.run import run
from scjepa.models.spartan import GraphPenalties
from scjepa.training.logging import WandbLogger


def _config(path: Path, steps: int = 2) -> ComparisonConfig:
    return ComparisonConfig(
        output_dir=str(path),
        steps=steps,
        batch_size=2,
        train_groups=4,
        validation_groups=3,
        test_groups=3,
        calibration_groups=3,
        views=2,
        embed_dim=8,
        mlp_hidden=8,
        threads=1,
        log_every=1,
        evaluate_every=1,
        checkpoint_every=1,
    )


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def test_comparison_pairs_dense_checkpoint_tau_and_group_sampling(tmp_path: Path) -> None:
    """All conditions run, sparse stages share weights/bound, and reruns are safe."""
    config = _config(tmp_path / "comparison")
    summary = run_comparison(config)
    root = Path(config.output_dir)
    dense = _json(root / "dense" / "result.json")
    for name, expected in (("state_edges", 36), ("action_edges", 96), ("parameter_edges", 36)):
        assert dense["metrics"][name] == expected
    baseline = _json(root / "baseline" / "config.json")
    full = _json(root / "full" / "config.json")
    selected = read_checkpoint(root / "dense" / "best.pt")
    assert baseline["initial_state_fingerprint"] == full["initial_state_fingerprint"]
    assert baseline["initial_state_fingerprint"] == state_identity(selected["model"])
    assert baseline["tau_calibration"] == full["tau_calibration"]
    assert baseline["tau_calibration"]["checkpoint_sha256"] == sha256(root / "dense" / "best.pt")
    assert baseline["parameter_gates_open"] is True
    assert full["parameter_gates_open"] is False
    assert baseline["sparsity_weights"] == {"state": 1, "action": 1, "parameter": 0}
    assert full["sparsity_weights"] == {"state": 1, "action": 1, "parameter": 1}
    baseline_result = _json(root / "baseline" / "result.json")
    assert baseline_result["metrics"]["parameter_edges"] == 36
    for condition in ("baseline", "full"):
        checkpoint = read_checkpoint(root / condition / "last.pt")
        assert int(checkpoint["controller"]["num_updates"]) == config.steps
        assert (root / condition / "test_graphs.json").is_file()
    before = {
        condition: sha256(root / condition / "last.pt")
        for condition in ("dense", "baseline", "full")
    }
    assert run_comparison(config, resume=True) == summary
    assert before == {condition: sha256(root / condition / "last.pt") for condition in before}
    with pytest.raises(FileExistsError, match="not empty"):
        run_comparison(config)
    with pytest.raises(ValueError, match="changed"):
        run_comparison(replace(config, tau_multiplier=1.1), resume=True)


def test_existing_dense_is_reused_read_only_for_three_condition_results(tmp_path: Path) -> None:
    """Point 1 best initializes sparse stages while its last model supplies dense results."""
    config = _config(tmp_path / "comparison")
    reference = tmp_path / "point1"
    original = replace(predictor_config(config, reference), steps=1)
    run(original)
    before = {path.name: sha256(path) for path in reference.iterdir() if path.is_file()}
    requested = replace(config, dense_reference_dir=str(reference))
    summary = run_comparison(requested)
    root = Path(config.output_dir)
    dense = _json(root / "dense" / "result.json")
    assert dense["evaluation_only"] is True
    assert dense["evaluated_step"] == 1
    assert not (root / "dense" / "last.pt").exists()
    assert _json(root / "calibration.json")["selected_step"] == 1
    assert _json(root / "baseline" / "result.json")["evaluated_step"] == 2
    assert before == {path.name: sha256(path) for path in reference.iterdir() if path.is_file()}
    assert run_comparison(requested, resume=True) == summary


def test_sparse_restart_restores_optimizer_controller_and_gate_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupted stochastic sparse training must match uninterrupted CPU training exactly."""
    torch.set_num_threads(1)
    config = _config(tmp_path / "uninterrupted", steps=4)
    datasets = {
        split: build_data(split, 4 if split == "train" else 3, views=2)
        for split in ("train", "validation", "test")
    }
    initial = build_predictor(predictor_config(config, tmp_path), "dense").state_dict()
    calibration: dict[str, object] = {"tau": 2.0, "checkpoint_sha256": "test-reference"}
    train_condition(
        config, "full", datasets, "test-comparison", initial=initial, calibration=calibration
    )
    interrupted = replace(config, output_dir=str(tmp_path / "interrupted"))
    original_log = WandbLogger.log

    def fail(self: WandbLogger, step: int, metrics: dict[str, float]) -> None:
        if step == 2 and "loss/prediction" in metrics:
            raise RuntimeError("simulated interruption")
        original_log(self, step, metrics)

    with monkeypatch.context() as context:
        context.setattr(WandbLogger, "log", fail)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            train_condition(
                interrupted,
                "full",
                datasets,
                "test-comparison",
                initial=initial,
                calibration=calibration,
            )
    assert read_checkpoint(Path(interrupted.output_dir) / "full" / "last.pt")["step"] == 1
    train_condition(
        interrupted,
        "full",
        datasets,
        "test-comparison",
        initial=initial,
        calibration=calibration,
        resume=True,
    )
    expected = read_checkpoint(Path(config.output_dir) / "full" / "last.pt")
    actual = read_checkpoint(Path(interrupted.output_dir) / "full" / "last.pt")
    for key in ("model", "optimizer", "controller", "sampler", "torch_rng"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_condition_penalties_keep_parameter_weight_zero_only_in_baseline() -> None:
    """Dense has no path objective; baseline and full have the intended block means."""
    blocks = GraphPenalties(torch.tensor(2.0), torch.tensor(3.0), torch.tensor(7.0))
    assert float(sparsity_loss(blocks, "dense")) == 0.0
    assert float(sparsity_loss(blocks, "baseline")) == 5.0
    assert float(sparsity_loss(blocks, "full")) == 12.0
