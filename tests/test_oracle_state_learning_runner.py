"""Protocol, provenance and exact-resume checks for the oracle state-learning runner."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.oracle_state_learning import run as runner
from diagnostics.oracle_state_learning.core import EvaluationResult, LiveStateData
from diagnostics.oracle_state_learning.data import PrepareConfig, prepare, read_json
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


def _config(root: Path, condition: runner.Condition, seed: int = 0) -> runner.TrainConfig:
    return runner.TrainConfig(
        output_dir=str(root / f"{condition}-{seed}"),
        data_dir=str(root / "data"),
        condition=condition,
        profile="smoke",
        seed=seed,
        steps=4,
        batch_size=2,
        log_every=1,
        checkpoint_every=1,
        evaluate_every=3,
        calibration_dir=str(root / "bound") if condition in {"baseline", "full"} else None,
    )


def _prepare(root: Path) -> None:
    prepare(
        PrepareConfig(
            output_dir=str(root / "data"),
            profile="smoke",
            train_groups=4,
            calibration_groups=2,
            validation_groups=2,
            test_groups=2,
        )
    )


def _checkpoint(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))


def test_scientific_config_rejects_protocol_drift_and_sparse_without_bound() -> None:
    with pytest.raises(ValueError, match="fixed"):
        runner.TrainConfig("unused", "unused", "dense", steps=10)
    with pytest.raises(ValueError, match="shared calibration"):
        runner.TrainConfig("unused", "unused", "full")
    with pytest.raises(ValueError, match="shared calibration"):
        runner.TrainConfig("unused", "unused", "mlp", calibration_dir="unused")
    with pytest.raises(ValueError, match="paired training seeds"):
        runner.TrainConfig("unused", "unused", "mlp", seed=2)


def test_dense_calibration_transfers_only_bound_and_final_sparse_test_is_authenticated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(tmp_path)
    actual_evaluate = runner.evaluate
    evaluated: list[str] = []

    def observe(
        model: SCJEPA,
        data: LiveStateData,
        batch_size: int,
        device: torch.device | str,
        objective: TrainingObjective,
    ) -> EvaluationResult:
        evaluated.append(data.config.split)
        return actual_evaluate(model, data, batch_size, device, objective)

    monkeypatch.setattr(runner, "evaluate", observe)
    dense = [runner.train(_config(tmp_path, "dense", seed)) for seed in (0, 1)]
    assert "test" not in evaluated
    assert not (tmp_path / "dense-0/test.pt").exists()
    bound = runner.calibrate(
        runner.CalibrateConfig(
            str(tmp_path / "bound"),
            tuple(str(tmp_path / f"dense-{seed}") for seed in (0, 1)),
            profile="smoke",
        )
    )
    assert bound["tau"] == max(
        read_json(tmp_path / f"dense-{seed}/calibration.json")["metrics"]["base"] for seed in (0, 1)
    )
    evaluated.clear()
    full_config = _config(tmp_path, "full")
    full = runner.train(full_config)
    assert evaluated.count("test") == 1
    for key in ("initial_encoder_identity", "initial_predictor_identity"):
        assert full[key] == dense[0][key]
    dense_weights = _checkpoint(tmp_path / "dense-0/last.pt")["model"]
    fresh_weights = _checkpoint(tmp_path / "full-0/initial.pt")["model"]
    encoder_weight = next(
        name
        for name in dense_weights
        if name.startswith("online_encoder.") and name.endswith("weight")
    )
    assert not torch.equal(
        dense_weights[encoder_weight],
        fresh_weights[encoder_weight],
    )
    final = _checkpoint(tmp_path / "full-0/last.pt")
    assert final["step"] == 4
    assert int(final["objective"]["geco.num_updates"]) == 4
    test = _checkpoint(tmp_path / "full-0/test.pt")
    calibration = _checkpoint(tmp_path / "full-0/calibration.pt")
    metrics, matching = runner.recovery_metrics(
        EvaluationResult(calibration["metrics"], calibration["samples"]),
        EvaluationResult(test["metrics"], test["samples"]),
    )
    assert metrics["state_mcc"] == test["metrics"]["state_mcc"]
    assert matching == read_json(tmp_path / "full-0/matching.json")
    assert runner.train(full_config, resume=True) == full
    assert runner.verify_completion(tmp_path / "full-0") == full
    (tmp_path / "full-0/matching.json").write_text("{}\n")
    with pytest.raises(ValueError, match="artifact hash differs"):
        runner.verify_completion(tmp_path / "full-0")


@pytest.mark.parametrize("condition", ["dense", "mlp", "full"])
def test_interrupted_run_resumes_exact_optimizer_rng_sampler_and_ema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: runner.Condition
) -> None:
    _prepare(tmp_path)
    if condition == "full":
        for seed in (0, 1):
            runner.train(_config(tmp_path, "dense", seed))
        runner.calibrate(
            runner.CalibrateConfig(
                str(tmp_path / "bound"),
                tuple(str(tmp_path / f"dense-{seed}") for seed in (0, 1)),
                profile="smoke",
            )
        )
    config = _config(tmp_path, condition)
    complete = runner.train(config)
    interrupted = replace(config, output_dir=str(tmp_path / "interrupted"))
    actual_forward = runner.forward_loss
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
            raise RuntimeError("deliberate interruption before third update")
        return actual_forward(model, batch, objective, forward_seed=forward_seed)

    with monkeypatch.context() as patch:
        patch.setattr(runner, "forward_loss", interrupt)
        with pytest.raises(RuntimeError, match="deliberate interruption"):
            runner.train(interrupted)
    assert _checkpoint(tmp_path / "interrupted/last.pt")["step"] == 2
    resumed = runner.train(interrupted, resume=True)
    assert complete["identity"] == resumed["identity"]
    left = _checkpoint(Path(config.output_dir) / "last.pt")
    right = _checkpoint(tmp_path / "interrupted/last.pt")
    for key in ("model", "optimizer", "objective", "sampler", "torch_rng", "cuda_rng"):
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    for split in (
        ("calibration", "validation")
        if condition == "dense"
        else ("calibration", "validation", "test")
    ):
        left_result = _checkpoint(Path(config.output_dir) / f"{split}.pt")
        right_result = _checkpoint(tmp_path / f"interrupted/{split}.pt")
        torch.testing.assert_close(left_result, right_result, rtol=0, atol=0)


def test_mismatched_source_refuses_checkpoint_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(tmp_path)
    config = _config(tmp_path, "mlp")
    runner.train(config)
    monkeypatch.setattr(runner, "code_identity", lambda: "changed-numerical-source")
    with pytest.raises(ValueError, match="configuration, source"):
        runner.train(config, resume=True)
