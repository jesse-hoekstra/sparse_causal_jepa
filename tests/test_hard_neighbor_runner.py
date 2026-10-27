from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from test_group_latent_runner import source_with_initialization
from torch import Tensor

from diagnostics.group_contrastive_pretext import run as ordinary_runner
from diagnostics.hard_neighbor_pretext import run as runner
from diagnostics.hard_neighbor_pretext.run import Config, run
from diagnostics.hard_neighbor_pretext.sampler import Bank, hard_indices
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.multiquery_control.data import TensorSplit
from scjepa.models.model import SCJEPA
from scjepa.training.distributed import GlobalBatchSampler


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_prefix_matches_uniform_then_hard_bank_and_exact_stale_bank_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_with_initialization(tmp_path / "source")
    bundle = load_primary_source(source, 7, expected_step=2, expected_checkpoint_sha256=None)
    config = Config(
        str(source),
        str(tmp_path / "full"),
        seed=7,
        steps=5,
        groups=2,
        warmup_steps=1,
        bank_refresh_interval=2,
        verification=True,
    )
    refresh_steps: list[int] = []
    active_bank: Bank | None = None
    original_refresh, original_loss = runner.refresh_bank, runner.training_loss
    step = 0

    def refresh(model: SCJEPA, train: TensorSplit, step: int) -> Bank:
        nonlocal active_bank
        assert set(train) == {"online", "target", "actions"}
        refresh_steps.append(step)
        active_bank = original_refresh(model, train, step)
        return active_bank

    def loss(model: SCJEPA, batch: TensorSplit) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal step
        step += 1
        if step == 1:
            indices = next(iter(GlobalBatchSampler(bundle.train_groups, bundle.batch_size, 7, 0)))
        else:
            assert active_bank is not None
            indices = hard_indices(active_bank, bundle.batch_size, 7, step)
        for name, value in batch.items():
            torch.testing.assert_close(value, bundle.train[name][indices], rtol=0, atol=0)
        torch.testing.assert_close(
            torch.get_rng_state(),
            torch.Generator().manual_seed(7 * 100000 + step).get_state(),
            rtol=0,
            atol=0,
        )
        return original_loss(model, batch)

    monkeypatch.setattr(runner, "refresh_bank", refresh)
    monkeypatch.setattr(runner, "training_loss", loss)
    full = run(config)
    assert refresh_steps == [2, 4]
    checkpoint = cast(dict[str, Any], torch.load(tmp_path / "full/last.pt", weights_only=False))
    assert checkpoint["bank"]["refresh_step"] == 4
    assert int(checkpoint["exposure_counts"].sum()) == 5 * bundle.batch_size
    assert int(checkpoint["hard_exposure_counts"].sum()) == 4 * bundle.batch_size
    assert int(checkpoint["anchor_counts"].sum()) == 4
    assert full["sampler_exposure"]["total_group_draws"] == 5 * bundle.batch_size
    assert full["records"][0]["hard_pretext"]["contrastive"] >= 0
    ordinary_runner.run(
        ordinary_runner.Config(
            str(source), str(tmp_path / "ordinary"), seed=7, steps=1, groups=2, verification=True
        )
    )
    ordinary = cast(dict[str, Any], torch.load(tmp_path / "ordinary/last.pt", weights_only=False))
    warmup = cast(dict[str, Any], torch.load(tmp_path / "full/warmup_final.pt", weights_only=False))
    for key in ("model", "optimizer"):
        torch.testing.assert_close(warmup[key], ordinary[key], rtol=0, atol=0)
    assert warmup["bank"] is None
    assert full["records"][1]["retention"] == ordinary["records"][1]["retention"]
    updates = 0

    def interrupt(model: SCJEPA, batch: TensorSplit) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal updates
        updates += 1
        if updates == 3:
            raise RuntimeError("interrupt inside stale bank interval")
        return original_loss(model, batch)

    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    monkeypatch.setattr(runner, "training_loss", interrupt)
    refresh_steps.clear()
    with pytest.raises(RuntimeError, match="stale bank"):
        run(resumed_config)
    interrupted = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    assert interrupted["step"] == 2
    assert interrupted["bank"]["refresh_step"] == 2
    assert refresh_steps == [2]
    assert not (tmp_path / "resumed/full_validation.json").exists()
    assert not (tmp_path / "resumed/final_test.json").exists()
    refresh_steps.clear()
    monkeypatch.setattr(runner, "training_loss", original_loss)
    resumed = run(resumed_config, resume=True)
    assert refresh_steps == [4]  # step3 must reuse saved bank, not recompute from updated E
    final = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    for key in ("model", "optimizer", "exposure_counts", "hard_exposure_counts", "anchor_counts"):
        torch.testing.assert_close(checkpoint[key], final[key], rtol=0, atol=0)
    torch.testing.assert_close(checkpoint["bank"]["means"], final["bank"]["means"], rtol=0, atol=0)
    assert checkpoint["bank"]["identity"] == final["bank"]["identity"]
    assert checkpoint["refreshes"] == final["refreshes"]
    torch.testing.assert_close(checkpoint["rng"]["torch"], final["rng"]["torch"], rtol=0, atol=0)
    assert checkpoint["rng"]["python"] == final["rng"]["python"]
    np.testing.assert_array_equal(checkpoint["rng"]["numpy"][1], final["rng"]["numpy"][1])
    assert full["records"] == resumed["records"]
    assert full["full_validation"] == resumed["full_validation"]
    assert full["final_test"] == resumed["final_test"]
    assert run(resumed_config, resume=True) == resumed
    with pytest.raises(ValueError, match="Resume settings"):
        run(replace(resumed_config, bank_refresh_interval=1), resume=True)
    with pytest.raises(ValueError, match="separate"):
        replace(config, output_dir=str(source / "bad"))
