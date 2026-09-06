"""Exact bounded slot matching stays on-device and preserves the context cost."""

from itertools import permutations
from unittest.mock import patch

import pytest
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from scjepa.models.visual_to_visual import context_target_assignment


def _reference_cost(online: Tensor, target: Tensor) -> Tensor:
    online, target = online.detach().float(), target.detach().float()
    histories = torch.cat((online.flatten(1, 2), target.flatten(1, 2)), dim=1)
    variance = histories.var(dim=1, unbiased=False).clamp_min(1e-6)
    return (
        ((online.unsqueeze(3) - target.unsqueeze(2)) / variance.sqrt()[:, None, None, None])
        .square()
        .mean(dim=(1, 4))
    )


@pytest.mark.parametrize("num_slots", [1, 2, 5, 6, 7])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_assignment_matches_scipy_optimal_cost(num_slots: int, dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(39)
    online = torch.randn(4, 8, num_slots, 9, generator=generator).to(dtype)
    target = torch.randn(4, 8, num_slots, 9, generator=generator).to(dtype)
    costs = _reference_cost(online, target).double()
    reference = torch.tensor([linear_sum_assignment(cost.numpy())[1].tolist() for cost in costs])
    with patch(
        "scjepa.models.visual_to_visual.linear_sum_assignment", wraps=linear_sum_assignment
    ) as scipy_solver:
        assignment = context_target_assignment(online, target)
    assert scipy_solver.call_count == (4 if num_slots > 6 else 0)
    actual_cost = costs.gather(2, assignment.unsqueeze(-1)).sum(dim=(1, 2))
    reference_cost = costs.gather(2, reference.unsqueeze(-1)).sum(dim=(1, 2))
    torch.testing.assert_close(actual_cost, reference_cost, rtol=0, atol=0)


def test_nearly_duplicate_target_slots_retain_double_precision_cost_ordering() -> None:
    generator = torch.Generator().manual_seed(43)
    online = torch.randn(32, 4, 5, 8, generator=generator)
    target = torch.randn(32, 4, 1, 8, generator=generator).expand_as(online)
    target = target + 1e-6 * torch.randn(32, 4, 5, 8, generator=generator)
    costs = _reference_cost(online, target).double()
    expected = torch.tensor([linear_sum_assignment(cost.numpy())[1].tolist() for cost in costs])
    actual = context_target_assignment(online, target)
    torch.testing.assert_close(
        costs.gather(2, actual.unsqueeze(-1)).sum(dim=(1, 2)),
        costs.gather(2, expected.unsqueeze(-1)).sum(dim=(1, 2)),
        rtol=0,
        atol=0,
    )


def test_all_five_slot_permutations_are_recovered_without_host_transfer() -> None:
    candidates = torch.tensor(list(permutations(range(5))))
    generator = torch.Generator().manual_seed(40)
    online = torch.randn(1, 4, 5, 8, generator=generator).expand(120, -1, -1, -1)
    target = online.gather(2, candidates[:, None, :, None].expand_as(online))
    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected host copy")):
        assignment = context_target_assignment(online, target)
    torch.testing.assert_close(assignment, candidates.argsort(dim=1))


def test_equal_cost_optima_use_lexicographic_order() -> None:
    # Every permutation has identical total cost, despite unequal per-row costs.
    online = torch.arange(5).float()[None, None, :, None].expand(2, 3, 5, 4)
    target = torch.zeros_like(online)
    expected = torch.arange(5)[None].expand(2, -1)
    torch.testing.assert_close(context_target_assignment(online, target), expected)
    torch.testing.assert_close(context_target_assignment(target, target), expected)


def test_scaled_noncontiguous_histories_preserve_the_matching_objective() -> None:
    generator = torch.Generator().manual_seed(41)
    online = torch.randn(3, 8, 5, 4, generator=generator).transpose(1, 3)
    target = torch.randn(3, 8, 5, 4, generator=generator).transpose(1, 3)
    assert not online.is_contiguous()
    scales = torch.logspace(-7, 7, 8)
    for source, destination in ((online, target), (online * scales, target * scales)):
        costs = _reference_cost(source, destination).double()
        expected = torch.tensor([linear_sum_assignment(cost.numpy())[1].tolist() for cost in costs])
        torch.testing.assert_close(context_target_assignment(source, destination), expected)


def test_assignment_builds_no_autograd_graph() -> None:
    online = torch.randn(2, 4, 5, 8, requires_grad=True)
    target = torch.randn_like(online, requires_grad=True)
    saved: list[Tensor] = []

    def pack(tensor: Tensor) -> Tensor:
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        assignment = context_target_assignment(online, target)
    assert not saved
    assert not assignment.requires_grad
    assert assignment.grad_fn is None
    assert assignment.dtype == torch.long
    assert assignment.device == online.device


@pytest.mark.parametrize(
    ("online_shape", "target_shape"),
    [
        ((2, 5), (2, 5)),
        ((2, 4, 5, 8), (2, 3, 5, 8)),
        ((0, 4, 5, 8), (0, 4, 5, 8)),
        ((2, 0, 5, 8), (2, 0, 5, 8)),
        ((2, 4, 0, 8), (2, 4, 0, 8)),
        ((2, 4, 5, 0), (2, 4, 5, 0)),
    ],
)
def test_invalid_history_shapes_are_rejected(
    online_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="shape|nonempty"):
        context_target_assignment(torch.empty(online_shape), torch.empty(target_shape))


def test_mismatched_devices_are_rejected_before_tensor_operations() -> None:
    with pytest.raises(ValueError, match="device"):
        context_target_assignment(torch.empty(2, 4, 5, 8), torch.empty(2, 4, 5, 8, device="meta"))


def test_nonfloating_coordinates_are_rejected() -> None:
    online = torch.zeros(2, 4, 5, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="floating-point"):
        context_target_assignment(online, online.float())


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_tensor_costs_leave_numerical_training_guards_reachable(bad_value: float) -> None:
    online = torch.randn(2, 4, 5, 8)
    target = torch.full_like(online, bad_value)
    assignment = context_target_assignment(online, target)
    assert assignment.shape == (2, 5)
    torch.testing.assert_close(assignment.sort(dim=1).values, torch.arange(5)[None].expand(2, -1))
    torch.testing.assert_close(context_target_assignment(online, target), assignment)


def test_large_problem_retains_scipy_nonfinite_error() -> None:
    slots = torch.full((2, 4, 7, 8), float("nan"))
    with pytest.raises(ValueError, match="invalid numeric entries"):
        context_target_assignment(slots, slots)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_assignment_matches_cpu_optimum_and_stays_on_cuda() -> None:
    generator = torch.Generator().manual_seed(42)
    online = torch.randn(4, 30, 5, 32, generator=generator)
    target = torch.randn(4, 30, 5, 32, generator=generator)
    expected = context_target_assignment(online, target)
    assignment = context_target_assignment(online.cuda(), target.cuda())
    assert assignment.device.type == "cuda"
    assert not assignment.requires_grad
    torch.testing.assert_close(assignment.cpu(), expected)
