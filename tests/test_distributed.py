"""CPU verification of batch-preserving torchrun helpers and DDP arithmetic."""

import os
import socket
import time
from typing import cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

from scjepa.training.distributed import (
    GlobalBatchSampler,
    any_rank,
    destroy_distributed,
    distributed_barrier,
    gather_batch_tensor,
    gather_rank_objects,
    initialize_distributed,
    is_main_process,
    mean_metrics,
    mean_tensor,
    rank,
    world_size,
)


@pytest.mark.parametrize("processes", [1, 2, 4])
@pytest.mark.parametrize("epoch", [0, 1, 7])
def test_global_batch_sampler_preserves_legacy_batch_order(processes: int, epoch: int) -> None:
    seed, dataset_size, batch_size = 13, 23, 4
    generator = torch.Generator().manual_seed(seed * 100_003 + epoch)
    original = DataLoader(
        TensorDataset(torch.arange(dataset_size)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=True,
    )
    expected = [cast(list[int], batch[0].tolist()) for batch in original]
    samplers = [
        GlobalBatchSampler(dataset_size, batch_size, seed, epoch, worker, processes)
        for worker in range(processes)
    ]
    recovered = [
        [episode for local_batch in global_parts for episode in local_batch]
        for global_parts in zip(*samplers, strict=True)
    ]
    assert recovered == expected
    assert all(len(sampler) == len(expected) for sampler in samplers)
    assert len({episode for batch in recovered for episode in batch}) == len(expected) * batch_size


@pytest.mark.parametrize(
    ("batch_size", "worker", "processes"),
    [(4, 0, 3), (4, 0, 8), (0, 0, 1), (4, 0, 0), (4, 2, 2), (4, -1, 2)],
)
def test_global_batch_sampler_rejects_invalid_shards(
    batch_size: int, worker: int, processes: int
) -> None:
    with pytest.raises(ValueError, match="rank|batch_size"):
        GlobalBatchSampler(12, batch_size, 0, 0, worker, processes)


def test_single_process_helpers_do_not_initialize_a_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert initialize_distributed("cpu") == torch.device("cpu")
    assert not dist.is_initialized()
    assert world_size() == 1
    assert rank() == 0
    assert is_main_process()
    value = torch.arange(48, dtype=torch.float32).view(2, 4, 2, 3).requires_grad_()
    assert not mean_tensor(value).requires_grad
    assert not gather_batch_tensor(value).requires_grad
    assert gather_rank_objects({"step": 3}) == [{"step": 3}]
    assert mean_metrics({"loss": 2.0}, "cpu") == {"loss": 2.0}
    assert any_rank(True, "cpu")
    assert not any_rank(False, "cpu")
    assert any_rank(torch.tensor(True), "cpu")
    assert not any_rank(torch.tensor(False), "cpu")
    distributed_barrier()
    destroy_distributed()


def test_single_cuda_device_resolves_current_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    selected: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    assert initialize_distributed("cuda") == torch.device("cuda:3")
    assert initialize_distributed("cuda:1") == torch.device("cuda:1")
    assert selected == [torch.device("cuda:3"), torch.device("cuda:1")]


def _distributed_worker(worker: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(worker),
        LOCAL_RANK=str(worker),
        WORLD_SIZE="2",
    )
    torch.set_num_threads(1)
    try:
        assert initialize_distributed("cpu") == torch.device("cpu")
        assert dist.get_backend() == "gloo"
        assert rank() == worker
        assert world_size() == 2
        assert is_main_process() == (worker == 0)
        torch.testing.assert_close(mean_tensor(torch.tensor([worker * 2.0])), torch.tensor([1.0]))
        assert any_rank(worker == 1, "cpu")
        assert not any_rank(False, "cpu")
        rejection = torch.tensor(worker == 1)
        assert any_rank(rejection, "cpu")
        assert bool(rejection) == (worker == 1)  # reduction never mutates caller's flag
        assert not any_rank(torch.tensor(False), "cpu")
        assert mean_metrics({"loss": worker * 4.0, "step": 3.0}, "cpu") == {
            "loss": 2.0,
            "step": 3.0,
        }

        # Gather the episodes in rank order for global representation diagnostics.
        slot_offsets = torch.tensor([0.0, 1000.0]).view(1, 1, 2, 1)
        local_states = slot_offsets.expand(1, 3, 2, 3) + worker * 4.0
        global_states = torch.cat(
            (slot_offsets.expand(1, 3, 2, 3), slot_offsets.expand(1, 3, 2, 3) + 4)
        )
        torch.testing.assert_close(gather_batch_tensor(local_states), global_states)
        assert gather_rank_objects({"worker": worker}) == [{"worker": 0}, {"worker": 1}]

        # DDP local means with global batch 4 reproduce a full-batch gradient
        # and Adam update. This guards against a second world-size loss scaling.
        torch.manual_seed(7)  # pyright: ignore[reportUnknownMemberType]
        full_model = nn.Linear(3, 2)
        local_model = nn.Linear(3, 2)
        local_model.load_state_dict(full_model.state_dict())
        distributed_model = DistributedDataParallel(local_model)
        inputs = torch.arange(12, dtype=torch.float32).view(4, 3) / 10
        targets = torch.arange(8, dtype=torch.float32).view(4, 2) / 7
        full_optimizer = torch.optim.Adam(full_model.parameters(), lr=1e-3)
        local_optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
        full_loss = (full_model(inputs) - targets).square().mean()
        full_loss.backward()
        local_slice = slice(worker * 2, (worker + 1) * 2)
        local_loss = (distributed_model(inputs[local_slice]) - targets[local_slice]).square().mean()
        local_loss.backward()
        torch.testing.assert_close(mean_tensor(local_loss), full_loss.detach())
        for local_parameter, full_parameter in zip(
            local_model.parameters(), full_model.parameters(), strict=True
        ):
            assert local_parameter.grad is not None
            assert full_parameter.grad is not None
            torch.testing.assert_close(local_parameter.grad, full_parameter.grad)
        local_optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        full_optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        for local_parameter, full_parameter in zip(
            local_model.parameters(), full_model.parameters(), strict=True
        ):
            torch.testing.assert_close(local_parameter, full_parameter)
        distributed_barrier()
    finally:
        destroy_distributed()


def test_two_cpu_ranks_match_global_statistics_and_full_batch_gradients() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
    context = mp.spawn(_distributed_worker, args=(port,), nprocs=2, join=False)  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
    assert context is not None
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("distributed CPU workers did not finish within 60 seconds")
    finally:
        for process in context.processes:
            assert process is not None
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
