"""Frozen nested-context inference and noise-corrected variance contracts."""

from copy import deepcopy
from math import isclose
from typing import cast

import pytest
import torch
from torch import Tensor

from diagnostics.evaluate_context_variance import (
    encode_contexts,
    sample_contexts,
    variance_components,
)
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.training.factory import ModelConfig, build_model


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([[[0.0], [2.0]], [[2.0], [4.0]]], (2.0, 2.0, 1.0, 3.0, 2 / 3)),
        ([[[0.0], [2.0]], [[0.0], [2.0]]], (2.0, 0.0, -1.0, 1.0, 2.0)),
        ([[[0.0], [0.0]], [[0.0], [0.0]]], (0.0, 0.0, 0.0, 0.0, None)),
    ],
)
def test_variance_corrects_replica_noise_without_clamping(
    values: list[list[list[float]]], expected: tuple[float, float, float, float, float | None]
) -> None:
    actual = variance_components(torch.tensor(values))
    for key, value in zip(
        ("within", "group_mean_variance", "between", "total", "ratio"), expected, strict=True
    ):
        if value is None:
            assert actual[key] is None
        else:
            assert isclose(cast(float, actual[key]), value)


def test_context_draws_are_reproducible_and_leave_global_rng_untouched() -> None:
    observer = VectorObservationMap(dimension=6)
    before = torch.get_rng_state().clone()
    first = sample_contexts(observer, 3, 2, 10, 71)
    second = sample_contexts(observer, 3, 2, 10, 71)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    for left, right in zip(first, second, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert first[0].shape == (3, 6)
    assert first[1].shape == (6, 10, 6)
    assert first[2].shape == (6, 9, 16)
    assert not torch.equal(first[1][0], first[1][1])


def test_nested_prefix_inference_preserves_weights_and_trained_context() -> None:
    model = (
        build_model(
            ModelConfig(
                observation_dim=6,
                state_dim=6,
                param_dim=6,
                action_dim=16,
                context_len=10,
                encoder_hidden_dims=(8,),
                parameter_embed_dim=8,
                parameter_heads=2,
                spartan_embed_dim=8,
                spartan_mlp_hidden=8,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    before = deepcopy(model.state_dict())
    _, observations, actions = sample_contexts(VectorObservationMap(dimension=6), 3, 2, 10, 1)
    received: list[tuple[Tensor, Tensor]] = []

    def capture_context(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        received.append((cast(Tensor, inputs[0]), cast(Tensor, inputs[1])))

    handle = model.parameter_encoder.register_forward_pre_hook(capture_context)
    codes = encode_contexts(model, observations, actions, [2, 5, 10], 2)
    handle.remove()
    assert len(received) == 9
    for index, length in enumerate((2, 5, 10)):
        expected_states = model.online_encoder.encode_sequence(observations[:, :length])
        actual_states = torch.cat([states for states, _ in received[index * 3 : (index + 1) * 3]])
        actual_actions = torch.cat([action for _, action in received[index * 3 : (index + 1) * 3]])
        torch.testing.assert_close(actual_states, expected_states)
        torch.testing.assert_close(actual_actions, actions[:, : length - 1])
        assert codes[length].shape == (6, 6)
    for key, value in model.state_dict().items():
        if isinstance(value, Tensor):
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        else:
            assert value == before[key]
    assert model.context_len == 10
    with pytest.raises(ValueError, match="trained K"):
        encode_contexts(model, observations, actions, [11], 2)
