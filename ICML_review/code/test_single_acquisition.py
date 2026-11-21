"""Gate checks and a tiny CPU rehearsal of first-acquisition training and continuation."""

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from diagnostics.masked_mlp_learned import run as runner
from diagnostics.masked_mlp_learned.budget_control import BudgetConfig
from diagnostics.masked_mlp_learned.continuation import continue_training
from diagnostics.masked_mlp_learned.selection import select
from diagnostics.masked_mlp_oracle.paper_analysis import load_run
from diagnostics.shared import data, source_data
from diagnostics.shared.io import read_json, sha
from scjepa.experiments.storage import write_json

from . import single_acquisition as policy
from .single_acquisition import decision


def test_pilot_requires_retention_and_paired_gain() -> None:
    """A strong full endpoint can still lose the paired contrast or a physical parameter."""
    full = {
        "state_mcc": 0.91,
        "normalized_prediction_mse": 0.02,
        "code_r2_min": 0.95,
        "masks/state_aligned_shd": 1.0,
        "masks/action_aligned_shd": 0.0,
        "masks/parameter_code_aligned_shd": 0.0,
    }
    original = {"full": full, "baseline": {"state_mcc": 0.65}}
    assert all(decision(original, original).values())
    control = {"full": full | {"code_r2_min": 0.7}, "baseline": {"state_mcc": 0.85}}
    checks = decision(control, original)
    assert not checks["parameter_retention"]
    assert not checks["paired_mcc_gain"]


def test_tiny_cpu_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run GELU projection, adaptation, both sparse arms and exact continuations on toy groups."""
    old = tmp_path / "single-acquisition-s0"
    new = tmp_path / "single-admitted-s0"
    new.mkdir()
    monkeypatch.setattr(policy, "PRIVATE", tmp_path)
    monkeypatch.setattr(policy, "PARENT", tmp_path)
    comparison = tmp_path / "comparison.json"
    monkeypatch.setattr(policy, "COMPARISON", comparison)
    monkeypatch.setattr(
        policy,
        "TRAIN_STEPS",
        {
            "adapt": 2,
            **{f"{a}-{n}k": v for a in ("full", "baseline") for n, v in ((300, 2), (350, 4))},
        },
    )
    physical, cohort = tmp_path / "physical", tmp_path / "cohort"
    source_data.prepare(
        source_data.PrepareConfig(
            str(physical),
            train_groups=4,
            calibration_groups=4,
            validation_groups=2,
            test_groups=2,
            profile="smoke",
        )
    )
    data.prepare(data.PrepareConfig(str(cohort), str(physical), profile="smoke"))
    joint = runner.TrainConfig(
        str(tmp_path / "joint"),
        str(cohort),
        "dense",
        profile="smoke",
        steps=2,
        batch_size=2,
        predictor_activation="gelu",
        code_dim=12,
        log_every=1,
        evaluate_every=2,
        checkpoint_every=1,
    )
    runner.train(joint)
    acquisition = replace(
        joint,
        output_dir=str(tmp_path / "acquisition-0-s0/results"),
        states_from=joint.output_dir,
        acquisition_seed=1000,
    )
    runner.train(acquisition)
    historical = select([Path(acquisition.output_dir)], window=2, maximum_loss=100.0)
    write_json(tmp_path / "selection-s0.json", historical)
    data_id = read_json(cohort / "completion.json")["identity"]
    old.mkdir()
    original = policy._admission(0, "old", data_id, False)  # pyright: ignore[reportPrivateUsage]
    admission = policy._admission(0, "new", data_id, True)  # pyright: ignore[reportPrivateUsage]
    assert len(admission["attempts"]) == 1
    assert admission["selected_index"] == 0
    assert admission["checkpoint_sha256"] == original["checkpoint_sha256"]
    write_json(old / "admission.json", original)
    write_json(new / "admission.json", admission)
    projected = str(old / "project/results")
    runner.project(
        runner.ProjectConfig(
            projected, acquisition.output_dir, str(cohort), profile="smoke", batch_size=2
        )
    )
    bound = str(tmp_path / "fixed-bound")
    runner.calibrate(runner.CalibrateConfig(bound, projected, tau_factor=1.0, profile="smoke"))
    dense = replace(
        joint, output_dir=str(old / "adapt/results"), code_dim=None, adapt_from=projected
    )
    runner.train(dense)
    for arm in ("baseline", "full"):
        parent = old / f"{arm}-300k/results"
        cfg = BudgetConfig(
            str(parent),
            str(cohort),
            arm,
            profile="smoke",
            steps=2,
            batch_size=2,
            predictor_activation="gelu",
            init_dir=dense.output_dir,
            calibration_dir=bound,
            log_every=1,
            evaluate_every=2,
            checkpoint_every=1,
        )
        runner.train(cfg)
        result = continue_training(parent, old / f"{arm}-350k/results", additional_steps=2)
        assert result["steps"] == 4
        assert result["predictor_activation"] == "gelu"

    measured = {a: load_run(old / f"{a}-350k/results", a, 0).metrics for a in ("full", "baseline")}
    reference = measured | {"full": measured["full"] | {"normalized_prediction_mse": 0.0}}
    write_json(comparison, {"per_run": {a: {"0": vals} for a, vals in reference.items()}})
    checks = decision(measured, reference)
    assert not checks["full_nmse"]
    tau = read_json(Path(bound) / "calibration.json")["tau_prediction"]
    metrics = {
        "commit": "old",
        "seed": 0,
        "passed": False,
        "checks": checks,
        "validation": measured,
        "data_identity": data_id,
        "tau": tau,
        "reference_validation": reference,
        "reference_summary_sha256": sha(comparison),
        "acquisition_receipt_sha256": sha(old / "admission.json"),
        "checkpoint_sha256": {s: sha(old / s / "results/last.pt") for s in policy.STAGES},
    }
    write_json(old / "metrics.json", metrics)
    with patch.object(policy, "_head") as head:
        policy._carry_forward(new, 0, "new", admission, tau)  # pyright: ignore[reportPrivateUsage]
        head.assert_called_once_with("new")
    reused = read_json(new / "metrics.json")
    assert reused["reused"]
    assert not reused["passed"]
    assert reused["source_commit"] == "old"
    assert reused["commit"] == "new"
    assert reused["checkpoint_sha256"] == metrics["checkpoint_sha256"]
    assert reused["checks"] == checks
    with pytest.raises(RuntimeError, match="tolerance"):
        policy._carry_forward(new, 0, "new", admission, tau / 2)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("first_pass", [2, None])
def test_retry_stops_at_first_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_pass: int | None
) -> None:
    """Reject early attempts, stop on the earliest pass, and explicitly represent exhaustion."""
    monkeypatch.setattr(policy, "PARENT", tmp_path)
    stored: list[dict[str, Any]] = []
    for index in range(16):
        folder = tmp_path / f"acquisition-{index}-s0/results"
        folder.mkdir(parents=True)
        row = {
            "identity": str(index),
            "data_identity": "cohort",
            "hashes": {"last.pt": str(index), "history.jsonl": str(index)},
        }
        write_json(folder / "completion.json", row)
        stored.append(
            {
                "acquisition_seed": 1000 + 100 * index,
                "identity": str(index),
                "completion_sha256": sha(folder / "completion.json"),
                "train_prediction_mean": 0.5 if index == first_pass else 2.0,
            }
        )
    write_json(
        tmp_path / "selection-s0.json",
        {"common": {}, "candidates": stored, "maximum_loss": 1.0, "window": 2},
    )

    def local_selector(paths: list[Path], *, maximum_loss: float, window: int) -> dict[str, Any]:
        assert window == 2
        index = int(paths[0].parent.name.split("-")[1])
        row = stored[index]
        return {
            "selected": row,
            "common": {},
            "identity": str(index),
            "admitted": row["train_prediction_mean"] <= maximum_loss,
        }

    def verify(path: Path, *, kind: str, require_current_source: bool) -> dict[str, Any]:
        assert kind == "train"
        assert require_current_source
        return read_json(path / "completion.json")

    monkeypatch.setattr(policy, "select", local_selector)
    monkeypatch.setattr(runner, "verify_completion", verify)
    result = policy._admission(0, "rev", "cohort", True)  # pyright: ignore[reportPrivateUsage]
    assert result["selected_index"] == first_pass
    assert result["admitted"] == (first_pass is not None)
    assert len(result["attempts"]) == (16 if first_pass is None else first_pass + 1)
    assert [r["index"] for r in result["attempts"]] == list(range(len(result["attempts"])))


def test_first_admitted_can_run_without_previous_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh eligible a0 starts training and keeps that route across later phase calls."""
    monkeypatch.setattr(policy, "PRIVATE", tmp_path)
    root = tmp_path / "single-admitted-s0"
    root.mkdir()
    write_json(root / "admission.json", {"admitted": True})
    with pytest.raises(RuntimeError, match="Prepare"):
        policy._reuse_planned(root, 0, 0, "full", "rev")  # pyright: ignore[reportPrivateUsage]
    assert not policy._reuse_planned(root, 0, 0, "prepare", "rev")  # pyright: ignore[reportPrivateUsage]
    prior = tmp_path / "single-acquisition-s0"
    prior.mkdir()
    write_json(prior / "metrics.json", {})
    assert not policy._reuse_planned(root, 0, 0, "baseline", "rev")  # pyright: ignore[reportPrivateUsage]
    write_json(root / "admission.json", {"admitted": False})
    with pytest.raises(RuntimeError, match="different"):
        policy._reuse_planned(root, 0, 0, "full", "rev")  # pyright: ignore[reportPrivateUsage]
