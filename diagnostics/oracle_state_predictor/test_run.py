"""Check the disposable runner's output isolation and exact CPU continuation."""

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from diagnostics.oracle_state_predictor.run import Config, GroupSampler, resolve_device, run


def _config(path: Path, steps: int) -> Config:
    return Config(
        output_dir=str(path),
        steps=steps,
        batch_size=2,
        train_groups=5,
        validation_groups=3,
        test_groups=3,
        views=2,
        embed_dim=8,
        mlp_hidden=8,
        threads=1,
        log_every=1,
        evaluate_every=1,
        checkpoint_every=1,
    )


def test_resume_matches_uninterrupted_updates(tmp_path: Path) -> None:
    """A restart preserves parameters, optimizer trajectory and next sampled groups."""
    full = _config(tmp_path / "full", 4)
    split = _config(tmp_path / "split", 2)
    expected = run(full)
    run(split)
    actual = run(replace(split, steps=4), resume=True)
    assert actual == expected
    first = cast(dict[str, Any], torch.load(Path(full.output_dir) / "last.pt", weights_only=False))
    second = cast(
        dict[str, Any], torch.load(Path(split.output_dir) / "last.pt", weights_only=False)
    )
    for key, value in first["model"].items():
        torch.testing.assert_close(value, second["model"][key], rtol=0, atol=0)
    assert first["step"] == second["step"] == 4
    left, right = GroupSampler(5, 2, 0), GroupSampler(5, 2, 0)
    left.load_state_dict(first["sampler"])
    right.load_state_dict(second["sampler"])
    torch.testing.assert_close(left.next(), right.next(), rtol=0, atol=0)
    assert (Path(split.output_dir) / "best.pt").is_file()
    assert (Path(split.output_dir) / "test_metrics.json").is_file()


def test_existing_output_and_incompatible_resume_are_rejected(tmp_path: Path) -> None:
    """A one-off diagnostic must not overwrite other runs or silently change its loss."""
    config = _config(tmp_path / "run", 1)
    run(config)
    with pytest.raises(FileExistsError, match="not empty"):
        run(config)
    with pytest.raises(ValueError, match="code/data/config changed"):
        run(replace(config, logit_weight=0.0, steps=2), resume=True)
    with pytest.raises(FileNotFoundError, match="no diagnostic checkpoint"):
        run(replace(config, output_dir=str(tmp_path / "missing")), resume=True)


def test_group_sampler_restarts_without_repeating_groups_within_batch() -> None:
    """Epoch remainders are dropped and checkpointed positions are restored."""
    sampler = GroupSampler(5, 2, 3)
    first, second = sampler.next(), sampler.next()
    assert len({int(value) for value in torch.cat((first, second))}) == 4
    restored = GroupSampler(5, 2, 99)
    restored.load_state_dict(sampler.state_dict())
    for _ in range(5):
        actual = sampler.next()
        assert len({int(value) for value in actual}) == 2
        torch.testing.assert_close(restored.next(), actual, rtol=0, atol=0)


@pytest.mark.parametrize(("requested", "expected"), [("cuda", "cuda:3"), ("cuda:1", "cuda:1")])
def test_cuda_selection_always_receives_an_explicit_index(
    monkeypatch: pytest.MonkeyPatch, requested: str, expected: str
) -> None:
    """Bare CUDA uses the current visible GPU; an explicit choice remains unchanged."""
    selected: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    def select(device: torch.device) -> None:
        assert device.index is not None
        selected.append(device)

    monkeypatch.setattr(torch.cuda, "set_device", select)
    assert resolve_device(requested) == torch.device(expected)
    assert selected == [torch.device(expected)]


def test_cpu_selection_does_not_access_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU-only verification must not initialize or inspect the CUDA runtime."""

    def fail_cuda() -> bool:
        raise AssertionError("CPU selection accessed CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", fail_cuda)
    assert resolve_device("cpu") == torch.device("cpu")


def test_cuda_unavailable_fails_before_selecting_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the useful availability error when a CUDA run cannot start."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        resolve_device("cuda")
