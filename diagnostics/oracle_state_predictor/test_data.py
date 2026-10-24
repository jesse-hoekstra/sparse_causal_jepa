"""Addressing, forecasting and group/view alignment for the diagnostic data."""

import pytest
import torch
from torch import Tensor

from diagnostics.oracle_state_predictor.data import build_data
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.simplified_gaussian import transition


@pytest.mark.parametrize("views", [1, 3])
def test_forecasts_reproduce_true_dynamics_with_float32_input_rounding(views: int) -> None:
    """The supplied true state, observed actions and raw theta determine both targets."""
    data = build_data("test", 3, views=views)
    batch = data.batch(torch.arange(3), "cpu")
    state = batch.initial_states[:, 0].double()
    predictions: list[Tensor] = []
    for step in range(2):
        state = transition(state, batch.parameters.double(), batch.actions[:, step].double())
        predictions.append(state)
    error = torch.stack(predictions, dim=1) - batch.targets.double()
    assert float(error.square().mean()) < 1e-12
    assert float(error.abs().max()) < 1e-6
    for value in (data.initial_states, data.actions, data.parameters, data.targets):
        assert value.device.type == "cpu"
        assert value.dtype == torch.float32
    assert data.config.root_seed == 0
    assert data.config.trajectory_length == 12
    assert data.metadata["source_state_index"] == 9
    assert data.metadata["action_indices"] == [9, 10]
    assert data.metadata["target_state_indices"] == [10, 11]
    source = build_dgp_dataset(data.config)
    assert data.source_fingerprint == source.fingerprint
    assert data.fingerprint != data.source_fingerprint
    assert set(source[0]) == {"observations", "actions", "episode_ids"}


def test_addressed_prefixes_reproduce_and_splits_are_independent() -> None:
    """Changing split count preserves addressed prefixes; split labels change the draws."""
    small = build_data("train", 2, views=2)
    larger = build_data("train", 3, views=2)
    repeated = build_data("train", 2, views=2)
    assert small.fingerprint == repeated.fingerprint
    assert small.fingerprint != larger.fingerprint
    for field in ("initial_states", "actions", "targets", "parameters", "episode_ids"):
        torch.testing.assert_close(getattr(small, field), getattr(repeated, field), rtol=0, atol=0)
        torch.testing.assert_close(
            getattr(small, field), getattr(larger, field)[:2], rtol=0, atol=0
        )
    splits = [small, build_data("validation", 2, views=2), build_data("test", 2, views=2)]
    ids = [{int(value) for value in data.episode_ids} for data in splits]
    assert len({data.fingerprint for data in splits}) == 3
    for first in range(3):
        for second in range(first + 1, 3):
            assert ids[first].isdisjoint(ids[second])
            assert not torch.equal(splits[first].parameters, splits[second].parameters)
            assert not torch.equal(splits[first].initial_states, splits[second].initial_states)
            assert not torch.equal(splits[first].actions, splits[second].actions)


def test_batch_preserves_selected_group_order_and_repeats_theta_across_views() -> None:
    """Flattening views must keep every trajectory paired with its own group's theta."""
    data = build_data("train", 3, views=3)
    indices = torch.tensor([2, 0])
    batch = data.batch(indices, torch.device("cpu"))
    assert batch.groups == 2
    assert batch.views == 3
    assert len(data) == 3
    assert data.views == 3
    assert batch.initial_states.shape == (6, 1, 6)
    assert batch.actions.shape == (6, 2, 16)
    assert batch.targets.shape == (6, 2, 6)
    assert batch.parameters.shape == (6, 6)
    torch.testing.assert_close(batch.group_ids, data.episode_ids[indices], rtol=0, atol=0)
    torch.testing.assert_close(
        batch.parameters, data.parameters[indices].repeat_interleave(3, dim=0), rtol=0, atol=0
    )
    for field in ("initial_states", "actions", "targets"):
        torch.testing.assert_close(
            getattr(batch, field), getattr(data, field)[indices].flatten(0, 1), rtol=0, atol=0
        )


def test_batches_reject_invalid_group_addresses() -> None:
    """Invalid addresses must not silently truncate or wrap onto a different group."""
    data = build_data("test", 2, views=2)
    with pytest.raises(ValueError, match="CPU int64"):
        data.batch(torch.tensor([0.5]), "cpu")
    with pytest.raises(ValueError, match="at least one"):
        data.batch(torch.empty(0, dtype=torch.int64), "cpu")
    with pytest.raises(IndexError, match="outside"):
        data.batch(torch.tensor([-1]), "cpu")
    with pytest.raises(IndexError, match="outside"):
        data.batch(torch.tensor([2]), "cpu")
