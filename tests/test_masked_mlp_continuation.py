"""Completed-run continuation preserves the full trajectory and immutable parent."""

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.masked_mlp_learned import continuation
from diagnostics.masked_mlp_learned import run as runner
from diagnostics.masked_mlp_oracle.model import Activation
from diagnostics.shared.data import PrepareConfig, prepare
from diagnostics.shared.io import read_json, sha
from diagnostics.shared.source_data import PrepareConfig as SourcePrepareConfig
from diagnostics.shared.source_data import prepare as prepare_source
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.loop import TrainingObjective


def _checkpoint(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))


@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_completed_continuation_and_interrupted_resume_match_uninterrupted_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, activation: Activation
) -> None:
    prepare_source(
        SourcePrepareConfig(
            str(tmp_path / "source"),
            train_groups=4,
            calibration_groups=4,
            validation_groups=2,
            test_groups=2,
            profile="smoke",
        )
    )
    prepare(PrepareConfig(str(tmp_path / "data"), str(tmp_path / "source"), profile="smoke"))
    dense = runner.TrainConfig(
        str(tmp_path / "dense"),
        str(tmp_path / "data"),
        "dense",
        profile="smoke",
        steps=2,
        batch_size=2,
        code_dim=6,
        initializer="gain",
        predictor_activation=activation,
        log_every=1,
        checkpoint_every=1,
        evaluate_every=3,
    )
    runner.train(dense)
    runner.calibrate(
        runner.CalibrateConfig(str(tmp_path / "bound"), dense.output_dir, profile="smoke")
    )
    parent = replace(
        dense,
        output_dir=str(tmp_path / "parent"),
        condition="full",
        init_dir=dense.output_dir,
        calibration_dir=str(tmp_path / "bound"),
        code_dim=None,
        initializer=None,
    )
    runner.train(parent)
    parent_dir = Path(parent.output_dir)
    original = {p.name: sha(p) for p in parent_dir.iterdir() if p.is_file()}
    reference = replace(parent, output_dir=str(tmp_path / "reference"), steps=6)
    runner.train(reference)
    child = tmp_path / "continued"
    result = continuation.continue_training(parent_dir, child, additional_steps=4)
    assert result["steps"] == result["successful_updates"] == 6
    assert result["tau_prediction"] == read_json(parent_dir / "completion.json")["tau_prediction"]
    assert sha(child / "parent.pt") == original["last.pt"]
    assert sha(child / "initial.pt") == original["initial.pt"]
    assert read_json(child / "continuation.json")["parent_step"] == 2
    assert (
        read_json(child / "config.json")["scientific_config"]["continuation_checkpoint_sha256"]
        == original["last.pt"]
    )
    expected = _checkpoint(Path(reference.output_dir) / "last.pt")
    actual = _checkpoint(child / "last.pt")
    for key in ("model", "optimizer", "objective", "sampler", "torch_rng"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert actual["python_rng"] == expected["python_rng"]
    assert actual["numpy_rng"][0] == expected["numpy_rng"][0]
    assert (actual["numpy_rng"][1] == expected["numpy_rng"][1]).all()
    assert actual["numpy_rng"][2:] == expected["numpy_rng"][2:]

    interrupted = tmp_path / "interrupted"
    calls = 0
    forward = runner.forward_loss

    def interrupt(
        model: SCJEPA,
        batch: dict[str, Tensor],
        objective: TrainingObjective,
        *,
        forward_seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], ModelOutput]:
        nonlocal calls
        if model.training:
            calls += 1
            if calls == 2:
                raise RuntimeError("deliberate interruption")
        return forward(model, batch, objective, forward_seed=forward_seed)

    with monkeypatch.context() as patch:
        patch.setattr(runner, "forward_loss", interrupt)
        with pytest.raises(RuntimeError, match="deliberate"):
            continuation.continue_training(parent_dir, interrupted, additional_steps=4)
    assert _checkpoint(interrupted / "last.pt")["step"] == 3
    with pytest.raises(ValueError, match="parent, source or budget"):
        continuation.continue_training(parent_dir, interrupted, additional_steps=5, resume=True)
    resumed = continuation.continue_training(
        parent_dir, interrupted, additional_steps=4, resume=True
    )
    assert resumed["identity"] == result["identity"]
    actual = _checkpoint(interrupted / "last.pt")
    for key in ("model", "optimizer", "objective", "sampler", "torch_rng"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert {p.name: sha(p) for p in parent_dir.iterdir() if p.is_file()} == original


def test_exploratory_continuation_has_one_explicit_budget_and_preserves_guards() -> None:
    config = continuation.ContinuationConfig(
        "unused",
        "unused",
        "full",
        steps=350_000,
        calibration_dir="unused",
        init_dir="unused",
        continuation_parent_identity="0" * 64,
        continuation_checkpoint_sha256="1" * 64,
        continuation_start_step=300_000,
    )
    with pytest.raises(ValueError, match="exactly 300000 to 350000"):
        replace(config, steps=400_000)
    with pytest.raises(ValueError, match="optimizer settings"):
        replace(config, lr=0.002)
    with pytest.raises(ValueError, match="completed sparse"):
        replace(config, condition="dense")
    with pytest.raises(ValueError, match="authenticated parent"):
        replace(config, continuation_checkpoint_sha256="unverified")
