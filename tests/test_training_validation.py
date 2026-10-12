"""Generic validation, selection, exact replay, and boundary-safe interruption."""

import copy
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from scjepa.data import TensorEpisodes
from scjepa.models.encoders import VectorEncoder
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import ParameterEncoder
from scjepa.models.spartan import Spartan
from scjepa.training import ObjectiveConfig, TrainConfig, Trainer

SELECTION = "validation/base_loss"


class _Logger:
    def __init__(self) -> None:
        self.records: list[tuple[int, dict[str, float]]] = []

    def log(self, step: int, metrics: dict[str, float]) -> None:
        self.records.append((step, dict(metrics)))


def _trainer(path: Path, steps: int = 6, accumulation: int = 2) -> Trainer:
    generator = torch.Generator().manual_seed(5)
    dataset = TensorEpisodes(torch.randn(8, 4, 4, generator=generator))
    torch.manual_seed(17)  # pyright: ignore[reportUnknownMemberType]
    model = SCJEPA(
        VectorEncoder(4, 3, (8,)),
        ParameterEncoder(3, 2, dim=8, num_heads=2, max_history=4),
        Spartan(3, 2, num_layers=1, embed_dim=8, mlp_hidden_size=8),
        context_len=2,
    )
    return Trainer(
        model,
        TrainConfig(
            steps=steps,
            batch_size=4,
            accumulation_steps=accumulation,
            out_dir=str(path),
            checkpoint_every=2,
        ),
        dataset,
        logger=_Logger(),
    )


def test_validation_defers_to_boundaries_and_selects_best_without_duplicate_final(
    tmp_path: Path,
) -> None:
    trainer = _trainer(tmp_path)
    calls: list[int] = []
    scores = iter((0.4, 0.2, 0.3))

    def evaluate(model: SCJEPA) -> dict[str, float]:
        assert model is trainer.model
        assert not model.training
        assert trainer.pending_microbatches == 0
        assert all(parameter.grad is None for parameter in model.parameters())
        calls.append(trainer.step)
        return {SELECTION: next(scores), "validation/generic_diagnostic": 1.0}

    metrics = trainer.train(evaluate, validation_every=1)
    assert calls == [2, 4, 6]
    assert metrics[SELECTION] == 0.3
    assert trainer.model.training
    assert not trainer.model.target_encoder.training
    assert trainer.validation_state == {
        "last_step": 6,
        "best_score": 0.2,
        "best_step": 4,
        "selection_metric": SELECTION,
    }
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    last = torch.load(tmp_path / "last.pt", weights_only=False)
    assert best["step"] == 4
    assert best["validation_state"]["best_score"] == 0.2
    assert last["step"] == 6
    assert last["validation_state"] == trainer.validation_state
    logger = cast(_Logger, trainer.logger)
    assert [step for step, item in logger.records if SELECTION in item] == [2, 4, 6]


def test_resume_restores_selection_state_and_final_evaluation_runs_once(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, steps=3)
    calls: list[int] = []

    def evaluate(_model: SCJEPA) -> dict[str, float]:
        calls.append(trainer.step)
        return {SELECTION: 0.4}

    trainer.train(evaluate, validation_every=5)
    assert calls == [3]
    assert trainer.discarded_microbatches == 1
    resumed = _trainer(tmp_path, steps=7)
    resumed.load_checkpoint(tmp_path / "last.pt")
    assert resumed.validation_state == trainer.validation_state

    def evaluate_resumed(_model: SCJEPA) -> dict[str, float]:
        calls.append(resumed.step)
        return {SELECTION: 0.5}

    resumed.train(evaluate_resumed, validation_every=5)
    assert calls == [3, 5, 7]
    assert resumed.validation_state["best_step"] == 3
    before = list(calls)
    resumed.train(evaluate_resumed, validation_every=5)
    assert calls == before
    with pytest.raises(ValueError, match="selection_metric differs"):
        resumed.train(evaluate_resumed, selection_metric="different")


def test_callback_rng_and_modes_cannot_change_training_replay(tmp_path: Path) -> None:
    reference = _trainer(tmp_path / "reference")
    reference.train()
    expected = copy.deepcopy(reference.model.state_dict())
    expected_torch = torch.get_rng_state().clone()
    expected_python = random.getstate()
    expected_numpy = cast(tuple[Any, ...], np.random.get_state())  # noqa: NPY002
    trainer = _trainer(tmp_path / "evaluated")

    def evaluate(model: SCJEPA) -> dict[str, float]:
        assert not model.target_encoder.training
        random.random()
        np.random.uniform(size=7)  # noqa: NPY002
        torch.randn(11)
        model.train()
        # Distinct child modes are restored even if evaluation changed them.
        model.online_encoder.eval()
        return {SELECTION: 1.0}

    trainer.train(evaluate, validation_every=2)
    torch.testing.assert_close(trainer.model.state_dict(), expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), expected_torch, rtol=0, atol=0)
    assert random.getstate() == expected_python
    actual_numpy = cast(tuple[Any, ...], np.random.get_state())  # noqa: NPY002
    assert actual_numpy[0] == expected_numpy[0]
    np.testing.assert_array_equal(actual_numpy[1], expected_numpy[1])
    assert actual_numpy[2:] == expected_numpy[2:]
    assert trainer.model.online_encoder.training
    assert not trainer.model.target_encoder.training
    assert all(parameter.grad is None for parameter in trainer.model.target_encoder.parameters())


@pytest.mark.parametrize("invalid", ["missing", "nonfinite", "exception"])
def test_failed_validation_restores_rng_and_model_modes(tmp_path: Path, invalid: str) -> None:
    trainer = _trainer(tmp_path)
    # Evaluate an already-complete run, without another optimizer update.
    trainer.step = trainer.config.steps
    trainer.model.online_encoder.eval()
    before = torch.get_rng_state().clone()

    def evaluate(model: SCJEPA) -> dict[str, float]:
        model.train()
        torch.randn(3)
        if invalid == "exception":
            raise ValueError("intentional callback failure")
        return {"wrong" if invalid == "missing" else SELECTION: float("nan")}

    with pytest.raises(RuntimeError, match="validation failed on rank zero"):
        trainer.train(evaluate)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    assert trainer.model.training
    assert not trainer.model.online_encoder.training
    assert not trainer.model.target_encoder.training
    assert trainer.validation_state["last_step"] is None


def test_interrupt_finishes_group_then_checkpoints_without_final_evaluation(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, steps=10, accumulation=3)
    calls: list[int] = []

    def evaluate(_model: SCJEPA) -> dict[str, float]:
        calls.append(trainer.step)
        return {SELECTION: 0.5}

    trainer.train(evaluate, validation_every=2, stop_requested=lambda: trainer.step >= 1)
    assert trainer.interrupted
    assert trainer.step == 3
    assert trainer.optimizer_steps == 1
    assert trainer.pending_microbatches == 0
    assert trainer.discarded_microbatches == 0
    assert calls == []
    saved = torch.load(tmp_path / "last.pt", weights_only=False)
    assert saved["step"] == 3
    resumed = _trainer(tmp_path, steps=6, accumulation=3)
    resumed.load_checkpoint(tmp_path / "last.pt")

    def resumed_evaluate(_model: SCJEPA) -> dict[str, float]:
        calls.append(resumed.step)
        return {SELECTION: 0.5}

    resumed.train(resumed_evaluate, validation_every=2)
    assert calls == [3, 6]
    assert not resumed.interrupted
    assert resumed.step == 6
    assert resumed.validation_state["last_step"] == 6
    assert resumed.validation_state["best_step"] == 3


def test_stop_before_first_batch_saves_exact_initial_state(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    before = copy.deepcopy(trainer.model.state_dict())
    trainer.train(stop_requested=lambda: True)
    assert trainer.interrupted
    assert trainer.step == trainer.optimizer_steps == 0
    torch.testing.assert_close(trainer.model.state_dict(), before, rtol=0, atol=0)
    assert (tmp_path / "last.pt").exists()


def test_old_checkpoint_without_validation_state_remains_supported(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    path = tmp_path / "older.pt"
    trainer.save_checkpoint(path)
    payload: dict[str, Any] = torch.load(path, weights_only=False)
    del payload["validation_state"]
    torch.save(payload, path)
    trainer.load_checkpoint(path)
    assert trainer.validation_state == {}


def test_configuration_rejects_visreg_and_invalid_validation_settings(tmp_path: Path) -> None:
    for name in ("state_regularizer", "parameter_regularizer"):
        with pytest.raises(ValueError, match="none or sigreg"):
            ObjectiveConfig(**{name: "visreg"})  # pyright: ignore[reportArgumentType]
    trainer = _trainer(tmp_path)
    with pytest.raises(ValueError, match="validation_every must be positive"):
        trainer.train(validation_every=0)
    with pytest.raises(ValueError, match="requires a validation_callback"):
        trainer.train(validation_every=1)
    with pytest.raises(ValueError, match="selection_metric must be nonempty"):
        trainer.train(selection_metric="")


@pytest.mark.parametrize("accumulation", [1, 3])
def test_interrupted_resume_preserves_every_validation_candidate_and_exact_weights(
    tmp_path: Path, accumulation: int
) -> None:
    reference = _trainer(tmp_path / "reference", steps=6, accumulation=accumulation)
    expected_calls: list[int] = []

    def reference_evaluate(_model: SCJEPA) -> dict[str, float]:
        expected_calls.append(reference.step)
        torch.randn(7)
        return {SELECTION: float(reference.step)}

    every = 1 if accumulation == 1 else 2
    reference.train(reference_evaluate, validation_every=every)
    reference_rng = torch.get_rng_state().clone()
    interrupted = _trainer(tmp_path / "resumed", steps=6, accumulation=accumulation)
    interrupted.train(
        lambda _model: {SELECTION: float(interrupted.step)},
        validation_every=every,
        stop_requested=lambda: interrupted.step >= 1,
    )
    assert interrupted.step == accumulation
    assert interrupted.validation_state["last_step"] is None
    resumed = _trainer(tmp_path / "resumed", steps=6, accumulation=accumulation)
    resumed.load_checkpoint(tmp_path / "resumed" / "last.pt")
    actual_calls: list[int] = []

    def evaluate(_model: SCJEPA) -> dict[str, float]:
        actual_calls.append(resumed.step)
        torch.randn(7)
        return {SELECTION: float(resumed.step)}

    resumed.train(evaluate, validation_every=every)
    assert actual_calls == expected_calls
    assert resumed.validation_state == reference.validation_state
    assert resumed.validation_state["best_step"] == accumulation
    torch.testing.assert_close(
        resumed.model.state_dict(), reference.model.state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        resumed.optimizer.state_dict(), reference.optimizer.state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(torch.get_rng_state(), reference_rng, rtol=0, atol=0)
    expected_best = torch.load(tmp_path / "reference" / "best.pt", weights_only=False)
    resumed_best = torch.load(tmp_path / "resumed" / "best.pt", weights_only=False)
    torch.testing.assert_close(resumed_best["model"], expected_best["model"], rtol=0, atol=0)


def test_signal_during_resumed_validation_checkpoints_before_another_update(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, steps=6, accumulation=1)
    trainer.train(
        lambda _model: {SELECTION: 0.5},
        validation_every=1,
        stop_requested=lambda: trainer.step >= 1,
    )
    resumed = _trainer(tmp_path, steps=6, accumulation=1)
    resumed.load_checkpoint(tmp_path / "last.pt")
    stopped = [False]
    evaluated: list[int] = []

    def evaluate(_model: SCJEPA) -> dict[str, float]:
        evaluated.append(resumed.step)
        stopped[0] = True
        return {SELECTION: 0.5}

    resumed.train(evaluate, validation_every=1, stop_requested=lambda: stopped[0])
    assert resumed.interrupted
    assert evaluated == [1]
    assert resumed.step == resumed.optimizer_steps == 1
    payload = torch.load(tmp_path / "last.pt", weights_only=False)
    assert payload["validation_state"]["last_step"] == 1
    assert payload["validation_state"]["best_step"] == 1
