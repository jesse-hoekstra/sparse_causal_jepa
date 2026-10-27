from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_encoder_gradient_source import write_source_fixture
from torch import Tensor, nn

from diagnostics.group_latent_control import run as runner
from diagnostics.group_latent_control.run import Config, Phase, run
from diagnostics.multiquery_control.data import TensorSplit
from diagnostics.multiquery_control.run import sha
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective
from scjepa.models.model import SCJEPA


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize(
    ("interrupted_phase", "interrupt_at"),
    [("teacher", 2), ("amortization", 1), ("amortization", 2)],
)
def test_staged_resume_matches_uninterrupted_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupted_phase: Phase, interrupt_at: int
) -> None:
    source = write_source_fixture(tmp_path / "source")
    hashes = {str(path): sha(path) for path in source.rglob("*") if path.is_file()}
    config = Config(
        source_dir=str(source),
        output_dir=str(tmp_path / "complete"),
        seed=7,
        teacher_steps=2,
        amortization_steps=2,
        washout_steps=2,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    complete = run(config)
    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    original = runner.training_loss
    calls = 0

    def interrupt(
        model: SCJEPA,
        batch: TensorSplit,
        indices: Tensor,
        phase: Phase,
        table: nn.Embedding | None,
        objective: PredictiveSupervisionObjective,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal calls
        if phase == interrupted_phase:
            calls += 1
            if calls == interrupt_at:
                raise RuntimeError("Operational interruption")
        return original(model, batch, indices, phase, table, objective)

    monkeypatch.setattr(runner, "training_loss", interrupt)
    with pytest.raises(RuntimeError, match="Operational interruption"):
        run(resumed_config)
    partial = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    expected_phase = "teacher" if interrupt_at == 1 else interrupted_phase
    assert partial["phase"] == expected_phase
    assert partial["phase_step"] == (2 if interrupt_at == 1 else 1)
    if interrupted_phase == "teacher":
        assert partial["latent_optimizer"]["state"]
    monkeypatch.setattr(runner, "training_loss", original)
    resumed = run(resumed_config, resume=True)
    full = cast(dict[str, Any], torch.load(tmp_path / "complete/last.pt", weights_only=False))
    final = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    torch.testing.assert_close(full["model"], final["model"], rtol=0, atol=0)
    torch.testing.assert_close(full["optimizer"], final["optimizer"], rtol=0, atol=0)
    assert final["table"] is None
    assert final["global_step"] == 6
    assert complete["records"] == resumed["records"]
    assert complete["final_test"] == resumed["final_test"]
    assert {str(path): sha(path) for path in source.rglob("*") if path.is_file()} == hashes
    assert run(resumed_config, resume=True) == resumed


def test_scientific_budget_cannot_be_silently_changed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="30000/10000/10000"):
        Config(
            source_dir=str(tmp_path / "source"),
            output_dir=str(tmp_path / "out"),
            teacher_steps=10000,
        )
