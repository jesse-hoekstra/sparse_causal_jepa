"""Calibration, sparse training, exact resume and protocol checks for the masked MLP runner."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.balanced_action_oracle.data import PrepareConfig, prepare
from diagnostics.isotropic_oracle.data import PrepareConfig as SourcePrepareConfig
from diagnostics.isotropic_oracle.data import prepare as prepare_source
from diagnostics.masked_mlp_oracle import run as runner
from diagnostics.oracle_state_learning.data import read_json
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.loop import TrainingObjective


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _prepare(root: Path) -> None:
    prepare_source(
        SourcePrepareConfig(
            str(root / "source"),
            train_groups=4,
            calibration_groups=2,
            validation_groups=2,
            test_groups=2,
            profile="smoke",
        )
    )
    prepare(PrepareConfig(str(root / "data"), str(root / "source"), profile="smoke"))


def _config(root: Path, condition: runner.Condition, seed: int = 0) -> runner.TrainConfig:
    return runner.TrainConfig(
        str(root / f"{condition}-{seed}"),
        str(root / "data"),
        condition,
        profile="smoke",
        seed=seed,
        steps=4,
        batch_size=2,
        log_every=1,
        checkpoint_every=1,
        evaluate_every=3,
        calibration_dir=str(root / "bound") if condition != "dense" else None,
    )


def _checkpoint(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], torch.load(path, weights_only=False, map_location="cpu"))


def _bound(root: Path) -> dict[str, Any]:
    runner.train(_config(root, "dense"))
    return runner.calibrate(
        runner.CalibrateConfig(str(root / "bound"), str(root / "dense-0"), profile="smoke")
    )


def test_prediction_bound_paired_sparse_arms_and_summary(tmp_path: Path) -> None:
    _prepare(tmp_path)
    bound = _bound(tmp_path)
    dense_calibration = read_json(tmp_path / "dense-0/calibration.json")["metrics"]
    assert bound["tau_prediction"] == 1.1 * dense_calibration["prediction"]
    dense = runner.verify_completion(tmp_path / "dense-0")
    runs = [tmp_path / "dense-0"]
    for condition in ("baseline", "full"):
        completion = runner.train(_config(tmp_path, condition))
        runs.append(tmp_path / f"{condition}-0")
        for key in ("initial_encoder_identity", "initial_trunk_identity"):
            assert completion[key] == dense[key]
        assert completion["tau_prediction"] == bound["tau_prediction"]
        metrics = read_json(tmp_path / f"{condition}-0/validation.json")["metrics"]
        for key in (
            "state_mcc",
            "block_purity",
            "masks/state_aligned_shd",
            "geco/relative_prediction",
            "geco/penalty_weight",
        ):
            assert key in metrics
        assert ("masks/parameter_open_probability" in metrics) == (condition == "full")
        config = read_json(tmp_path / f"{condition}-0/config.json")
        assert config["geco"]["tau"] == 1.0
        assert config["scientific_config"]["objective"]["parameter_sparsity_weight"] == float(
            condition == "full"
        )
    assert not (tmp_path / "full-0/test.pt").exists()
    other_seed = runner.train(_config(tmp_path, "full", seed=1))
    assert other_seed["initial_trunk_identity"] != dense["initial_trunk_identity"]
    rows = runner.summarize([str(path) for path in runs])
    assert [row["condition"] for row in rows] == ["baseline", "dense", "full"]
    assert all(row["steps"] == 4 for row in rows)


@pytest.mark.parametrize("condition", ["dense", "full"])
def test_exact_interruption_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: runner.Condition
) -> None:
    _prepare(tmp_path)
    if condition == "full":
        _bound(tmp_path)
    config = _config(tmp_path, condition, seed=2)
    complete = runner.train(config)
    interrupted = replace(config, output_dir=str(tmp_path / "interrupted"))
    actual = runner.forward_loss
    updates = 0

    def interrupt(
        model: SCJEPA,
        batch: dict[str, Tensor],
        objective: TrainingObjective,
        *,
        forward_seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], ModelOutput]:
        nonlocal updates
        updates += 1
        if updates == 3:
            raise RuntimeError("deliberate interruption")
        return actual(model, batch, objective, forward_seed=forward_seed)

    with monkeypatch.context() as patch:
        patch.setattr(runner, "forward_loss", interrupt)
        with pytest.raises(RuntimeError, match="deliberate"):
            runner.train(interrupted)
    assert _checkpoint(tmp_path / "interrupted/last.pt")["step"] == 2
    resumed = runner.train(interrupted, resume=True)
    assert complete["identity"] == resumed["identity"]
    left, right = (
        _checkpoint(Path(config.output_dir) / "last.pt"),
        _checkpoint(tmp_path / "interrupted/last.pt"),
    )
    for key in ("model", "optimizer", "objective", "sampler", "torch_rng", "cuda_rng"):
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_protocol_drift_and_source_changes_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="seed"):
        runner.TrainConfig("unused", "unused", "dense", seed=3)
    with pytest.raises(ValueError, match="100000"):
        runner.TrainConfig("unused", "unused", "dense", steps=10)
    with pytest.raises(ValueError, match="calibration directory"):
        runner.TrainConfig("unused", "unused", "full")
    with pytest.raises(ValueError, match="prespecified"):
        runner.CalibrateConfig("unused", "unused", tau_factor=2)
    _prepare(tmp_path)
    config = _config(tmp_path, "dense", seed=1)
    runner.train(config)
    with pytest.raises(ValueError, match="seed-zero"):
        runner.calibrate(
            runner.CalibrateConfig(str(tmp_path / "bound"), config.output_dir, profile="smoke")
        )
    monkeypatch.setattr(runner, "code_identity", lambda: "changed-source")
    with pytest.raises(ValueError, match="configuration, source"):
        runner.train(config, resume=True)
