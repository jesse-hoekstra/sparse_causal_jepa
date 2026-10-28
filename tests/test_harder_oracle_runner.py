"""Training isolation, paired initialization and exact resume for harder DGP screens."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.harder_oracle import run as runner
from diagnostics.harder_oracle.data import PrepareConfig, StateData, load_data, prepare
from diagnostics.harder_oracle.model import build_control
from diagnostics.oracle_state_learning.core import EvaluationResult, LiveStateData
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
    prepare(
        PrepareConfig(
            str(root / "data"),
            train_groups=4,
            calibration_groups=2,
            validation_groups=2,
            test_groups=2,
            profile="smoke",
        )
    )


def _config(root: Path, condition: runner.Condition) -> runner.TrainConfig:
    return runner.TrainConfig(
        str(root / condition),
        str(root / "data"),
        condition,
        profile="smoke",
        steps=4,
        batch_size=2,
        log_every=1,
        checkpoint_every=1,
        evaluate_every=3,
        calibration_dir=str(root / "bound") if condition in {"full", "baseline"} else None,
    )


def _checkpoint(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], torch.load(path, weights_only=False, map_location="cpu"))


def _bound(root: Path) -> dict[str, Any]:
    runner.train(_config(root, "dense"))
    return runner.calibrate(
        runner.CalibrateConfig(str(root / "bound"), (str(root / "dense"),), profile="smoke")
    )


def test_model_pairing_and_mlp_parameter_match() -> None:
    models = {
        condition: build_control(condition, 0) for condition in ("dense", "baseline", "full", "mlp")
    }
    for name in ("baseline", "full", "mlp"):
        torch.testing.assert_close(
            models[name].online_encoder.state_dict(),
            models["dense"].online_encoder.state_dict(),
            rtol=0,
            atol=0,
        )
    for name in ("baseline", "full"):
        torch.testing.assert_close(
            models[name].predictor.state_dict(),
            models["dense"].predictor.state_dict(),
            rtol=0,
            atol=0,
        )
    counts = {
        name: sum(p.numel() for p in model.predictor.parameters()) for name, model in models.items()
    }
    assert abs(counts["mlp"] / counts["dense"] - 1) < 0.005
    assert all(model.predictor.action_dim == 22 for model in models.values())


def test_cache_labels_addresses_and_validation_only_load(tmp_path: Path) -> None:
    _prepare(tmp_path)
    data, completion = load_data(tmp_path / "data")
    assert set(data) == {"train", "calibration", "validation"}
    assert data["train"].states is None
    assert not (tmp_path / "data/train_states.pt").exists()
    assert completion["settings"]["transport_chart"] == "mixed"
    assert data["train"].tensors["actions"].shape == (4, 10, 11, 22)
    assert set(data["train"].batch(torch.tensor([0]), "cpu")) == {
        "observations",
        "actions",
        "oracle_parameters",
        "episode_ids",
    }
    all_data, _ = load_data(tmp_path / "data", include_test=True)
    assert (
        len({int(i) for sample in all_data.values() for i in sample.tensors["episode_ids"]}) == 10
    )
    payload = _checkpoint(tmp_path / "data/train.pt")
    assert set(payload) == {"tensors", "metadata"}


def test_one_seed_calibration_fresh_sparse_start_and_separate_final_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(tmp_path)
    evaluated: list[str] = []
    actual = runner.evaluate

    def observe(
        model: SCJEPA,
        data: LiveStateData,
        batch_size: int,
        device: torch.device | str,
        objective: TrainingObjective,
    ) -> EvaluationResult:
        evaluated.append(cast(StateData, data).split)
        return actual(model, data, batch_size, device, objective)

    monkeypatch.setattr(runner, "evaluate", observe)
    bound = _bound(tmp_path)
    assert bound["tau"] == 1.1 * read_json(tmp_path / "dense/calibration.json")["metrics"]["base"]
    full = runner.train(_config(tmp_path, "full"))
    dense = runner.verify_completion(tmp_path / "dense")
    for key in ("initial_encoder_identity", "initial_predictor_identity"):
        assert dense[key] == full[key]
    assert "test" not in evaluated
    assert not (tmp_path / "full/test.pt").exists()
    tested = runner.score_test(
        runner.TestConfig(str(tmp_path / "tested"), str(tmp_path / "full"), str(tmp_path / "data"))
    )
    assert evaluated.count("test") == 1
    assert tested["parent_identity"] == full["identity"]
    assert read_json(tmp_path / "tested/matching.json") == read_json(
        tmp_path / "full/matching.json"
    )
    assert runner.verify_completion(tmp_path / "full") == full
    assert runner.verify_completion(tmp_path / "tested", kind="test") == tested
    assert runner.train(_config(tmp_path, "full"), resume=True) == full


@pytest.mark.parametrize("condition", ["dense", "mlp", "full"])
def test_exact_interruption_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: runner.Condition
) -> None:
    _prepare(tmp_path)
    if condition == "full":
        _bound(tmp_path)
    config = _config(tmp_path, condition)
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
    for split in ("calibration", "validation"):
        torch.testing.assert_close(
            _checkpoint(Path(config.output_dir) / f"{split}.pt"),
            _checkpoint(tmp_path / f"interrupted/{split}.pt"),
            rtol=0,
            atol=0,
        )


def test_source_changes_and_protocol_drift_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="fixed"):
        runner.TrainConfig("unused", "unused", "dense", seed=1)
    with pytest.raises(ValueError, match="30000 or 100000"):
        runner.TrainConfig("unused", "unused", "dense", steps=10)
    with pytest.raises(ValueError, match="prespecified"):
        runner.CalibrateConfig("unused", ("unused",), tau_factor=2)
    _prepare(tmp_path)
    config = _config(tmp_path, "mlp")
    runner.train(config)
    monkeypatch.setattr(runner, "code_identity", lambda: "changed-source")
    with pytest.raises(ValueError, match="configuration, source"):
        runner.train(config, resume=True)
