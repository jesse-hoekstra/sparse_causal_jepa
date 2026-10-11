"""Auxiliary work protects host streams without touching another rank's GPU."""

import pickle
import random

import numpy as np
import pytest
import torch
from torch import Tensor

from scjepa.training.rng import preserve_rng


def _forbidden(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("auxiliary RNG preservation touched an unowned CUDA device")


def _forbid_global_cuda_access(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "device_count",
        "get_rng_state_all",
        "set_rng_state_all",
        "manual_seed_all",
        "init",
    ):
        monkeypatch.setattr(torch.cuda, name, _forbidden)


@pytest.mark.parametrize("failure", [False, True])
def test_host_streams_restore_on_success_and_failure_without_initializing_cuda(
    monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    _forbid_global_cuda_access(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_device", _forbidden)
    monkeypatch.setattr(torch.cuda, "get_rng_state", _forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state", _forbidden)
    before_python = random.getstate()
    before_numpy = pickle.dumps(np.random.get_state())  # noqa: NPY002
    before_torch = torch.get_rng_state().clone()

    def auxiliary() -> None:
        with preserve_rng(torch.device("cuda:3")):
            random.random()
            np.random.uniform(size=3)  # noqa: NPY002
            torch.randn(5)
            if failure:
                raise ValueError("test failure")

    if failure:
        with pytest.raises(ValueError, match="test failure"):
            auxiliary()
    else:
        auxiliary()
    assert random.getstate() == before_python
    assert pickle.dumps(np.random.get_state()) == before_numpy  # noqa: NPY002
    torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)


@pytest.mark.parametrize("device", [None, torch.device("cuda"), torch.device("cuda:5")])
def test_only_current_or_explicit_cuda_rng_is_snapshotted_and_restored(
    monkeypatch: pytest.MonkeyPatch, device: torch.device | None
) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    _forbid_global_cuda_access(monkeypatch)
    current = [3]
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    states = {
        3: torch.tensor([3, 7], dtype=torch.uint8),
        5: torch.tensor([5, 9], dtype=torch.uint8),
    }
    expected = 5 if device is not None and device.index == 5 else 3
    accesses: list[tuple[str, int]] = []

    def get_state(index: int) -> Tensor:
        accesses.append(("get", index))
        return states[index].clone()

    def set_state(state: Tensor, index: int) -> None:
        accesses.append(("set", index))
        states[index] = state.clone()

    monkeypatch.setattr(torch.cuda, "get_rng_state", get_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", set_state)
    original = states[expected].clone()
    other_index = 3 if expected == 5 else 5
    other_state = states[other_index].clone()
    with preserve_rng(device):
        states[expected] = torch.zeros(2, dtype=torch.uint8)
        # Restoration must target the captured device even if auxiliary code
        # changes the current-device selection before returning.
        current[0] = other_index
    assert accesses == [("get", expected), ("set", expected)]
    torch.testing.assert_close(states[expected], original, rtol=0, atol=0)
    torch.testing.assert_close(states[other_index], other_state, rtol=0, atol=0)


def test_explicit_cpu_work_does_not_read_any_initialized_cuda_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    _forbid_global_cuda_access(monkeypatch)
    for name in ("current_device", "get_rng_state", "set_rng_state"):
        monkeypatch.setattr(torch.cuda, name, _forbidden)
    before = torch.get_rng_state().clone()
    with preserve_rng(torch.device("cpu")):
        torch.randn(11)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


def test_cuda_rng_restores_after_an_auxiliary_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    _forbid_global_cuda_access(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_device", _forbidden)
    saved = torch.tensor([1, 2, 3], dtype=torch.uint8)
    restored: list[tuple[Tensor, int]] = []

    def get_state(device: int) -> Tensor:
        assert device == 2
        return saved

    def set_state(state: Tensor, device: int) -> None:
        restored.append((state, device))

    monkeypatch.setattr(torch.cuda, "get_rng_state", get_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", set_state)
    with pytest.raises(ValueError, match="intentional"), preserve_rng(torch.device("cuda:2")):
        raise ValueError("intentional")
    assert len(restored) == 1
    assert restored[0][1] == 2
    torch.testing.assert_close(restored[0][0], saved, rtol=0, atol=0)
