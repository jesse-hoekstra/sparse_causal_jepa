"""Contract tests for the state-to-state T=2 autoregressive auxiliary."""

from typing import cast

import pytest
import torch
from torch import Tensor, nn

from scjepa.losses import rollout_t2_endpoint_mse
from scjepa.models import (
    StateToStateModel,
    num_valid_rollout_t2_offsets,
    sample_rollout_t2_offsets,
)
from scjepa.models.parameter_encoder import ParameterEncoder
from scjepa.models.spartan import Spartan, SpartanOutput

NUM_OBJECTS = 2
STATE_DIM = 3


class _RecordingEncoder(nn.Module):
    """Deterministic episode encoder that exposes how often it was called."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, states: Tensor) -> Tensor:
        self.calls += 1
        return states.mean(dim=(1, 3)).unsqueeze(-1)


class _RecordingAffinePredictor(nn.Module):
    """Simple differentiable transition with inspectable inputs and outputs."""

    def __init__(self, scale: float = 1.5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.inputs: list[Tensor] = []
        self.params: list[Tensor] = []
        self.outputs: list[Tensor] = []

    def forward(self, state: Tensor, params: Tensor) -> SpartanOutput:
        self.inputs.append(state)
        self.params.append(params)
        prediction = self.scale * state + params
        if prediction.requires_grad:
            prediction.retain_grad()
        self.outputs.append(prediction)

        batch, objects = state.shape[:2]
        token_count = 2 * objects
        zero = state.new_zeros(())
        return SpartanOutput(
            prediction=prediction,
            path_matrix=state.new_zeros(batch, token_count, token_count),
            sparsity=zero,
            logit_penalty=zero,
            mean_abs_logit=zero,
            mean_gate_probability=zero,
            gate_entropy=zero,
        )


def _instrumented_model() -> tuple[
    StateToStateModel,
    _RecordingEncoder,
    _RecordingAffinePredictor,
]:
    encoder = _RecordingEncoder()
    predictor = _RecordingAffinePredictor()
    # StateToStateModel deliberately depends on the two production concrete
    # classes, but these test doubles implement their complete forward contracts.
    model = StateToStateModel(
        cast(ParameterEncoder, encoder),
        cast(Spartan, predictor),
    )
    return model, encoder, predictor


def test_samples_exactly_eight_distinct_valid_offsets_per_episode() -> None:
    valid = num_valid_rollout_t2_offsets(sequence_len=60, context_len=30)
    offsets = sample_rollout_t2_offsets(
        batch_size=16,
        num_valid_offsets=valid,
        num_anchors=8,
        device="cpu",
        generator=torch.Generator().manual_seed(7),
    )

    assert valid == 29
    assert offsets.shape == (16, 8)
    assert bool((offsets >= 0).all())
    assert bool((offsets < valid).all())
    sorted_offsets = offsets.sort(dim=1).values
    assert bool((sorted_offsets[:, 1:] != sorted_offsets[:, :-1]).all())


@pytest.mark.parametrize(("sequence_len", "context_len"), [(60, 30), (17, 6), (10, 3)])
def test_every_sampled_two_step_window_stays_inside_sequence(
    sequence_len: int,
    context_len: int,
) -> None:
    valid = num_valid_rollout_t2_offsets(sequence_len, context_len)
    anchors = min(8, valid)
    offsets = sample_rollout_t2_offsets(
        batch_size=12,
        num_valid_offsets=valid,
        num_anchors=anchors,
        device="cpu",
        generator=torch.Generator().manual_seed(11),
    )

    anchor_indices = context_len - 1 + offsets
    endpoint_indices = anchor_indices + 2
    assert bool((anchor_indices >= context_len - 1).all())
    assert bool((endpoint_indices < sequence_len).all())
    assert int(endpoint_indices.max()) <= sequence_len - 1


def test_offset_sampling_draws_independently_for_each_episode() -> None:
    batched_generator = torch.Generator().manual_seed(2026)
    sequential_generator = torch.Generator().manual_seed(2026)
    batched = sample_rollout_t2_offsets(
        5,
        29,
        8,
        device="cpu",
        generator=batched_generator,
    )
    sequential = torch.cat(
        [
            sample_rollout_t2_offsets(
                1,
                29,
                8,
                device="cpu",
                generator=sequential_generator,
            )
            for _ in range(5)
        ]
    )

    # A batched draw is five consecutive per-episode draws, not one sampled
    # row broadcast over all episodes.
    torch.testing.assert_close(batched, sequential)
    assert any(not torch.equal(batched[0], batched[index]) for index in range(1, 5))


def test_offset_sampling_is_reproducible_with_a_fixed_seed() -> None:
    first = sample_rollout_t2_offsets(
        8,
        29,
        8,
        device="cpu",
        generator=torch.Generator().manual_seed(19),
    )
    replay = sample_rollout_t2_offsets(
        8,
        29,
        8,
        device="cpu",
        generator=torch.Generator().manual_seed(19),
    )
    different_seed = sample_rollout_t2_offsets(
        8,
        29,
        8,
        device="cpu",
        generator=torch.Generator().manual_seed(20),
    )

    torch.testing.assert_close(first, replay)
    assert not torch.equal(first, different_seed)


def test_explicit_offsets_must_be_integer_indices() -> None:
    model, _, _ = _instrumented_model()
    states = torch.randn(2, 12, NUM_OBJECTS, STATE_DIM)
    with pytest.raises(ValueError, match="integer indices"):
        model(states, context_len=4, rollout_t2_offsets=torch.tensor([[0.0], [1.5]]))


def test_all_eight_windows_reuse_one_episode_level_theta() -> None:
    model, encoder, predictor = _instrumented_model()
    states = torch.randn(3, 60, NUM_OBJECTS, STATE_DIM)
    output = model(states, context_len=30, num_rollout_t2_anchors=8)

    assert encoder.calls == 1
    assert output.rollout_t2_offsets is not None
    assert output.rollout_t2_offsets.shape == (3, 8)
    # Predictor calls: all teacher-forced transitions, then AR step 1 and AR step 2.
    assert len(predictor.params) == 3
    expected = output.causal_params[:, None].expand(-1, 8, -1, -1).reshape(3 * 8, NUM_OBJECTS, 1)
    torch.testing.assert_close(predictor.params[1], expected)
    torch.testing.assert_close(predictor.params[2], expected)


def test_second_prediction_consumes_generated_first_prediction() -> None:
    model, _, predictor = _instrumented_model()
    states = torch.arange(2 * 12 * NUM_OBJECTS * STATE_DIM, dtype=torch.float32).reshape(
        2,
        12,
        NUM_OBJECTS,
        STATE_DIM,
    )
    offsets = torch.tensor([[0, 2, 5], [1, 3, 6]])
    output = model(states, context_len=4, rollout_t2_offsets=offsets)

    assert output.rollout_t2_intermediate is not None
    assert output.rollout_t2_prediction is not None
    generated_first = output.rollout_t2_intermediate.flatten(0, 1)
    torch.testing.assert_close(predictor.inputs[2], generated_first)
    torch.testing.assert_close(
        output.rollout_t2_prediction,
        predictor.outputs[2].reshape_as(output.rollout_t2_prediction),
    )

    episode_index = torch.arange(states.shape[0])[:, None]
    true_intermediate = states[episode_index, 4 + offsets].flatten(0, 1)
    assert not torch.allclose(predictor.inputs[2], true_intermediate)


def test_endpoint_gradients_traverse_both_transition_calls_without_detach() -> None:
    model, _, predictor = _instrumented_model()
    states = torch.zeros(2, 8, NUM_OBJECTS, STATE_DIM, requires_grad=True)
    theta = torch.ones(2, NUM_OBJECTS, 1, requires_grad=True)
    offsets = torch.tensor([[0, 2], [1, 3]])

    endpoint, target, _ = model.rollout_t2_from_offsets(
        states,
        context_len=3,
        causal_params=theta,
        offsets=offsets,
    )
    rollout_t2_endpoint_mse(endpoint, target).backward()  # pyright: ignore[reportUnknownMemberType]

    assert len(predictor.outputs) == 2
    first_prediction_grad = predictor.outputs[0].grad
    assert first_prediction_grad is not None
    assert float(first_prediction_grad.abs().sum()) > 0
    assert states.grad is not None
    # The endpoint target is detached; a nonzero true-anchor gradient can only
    # arrive through generated step 1 and then generated step 2.
    episode_index = torch.arange(states.shape[0])[:, None]
    anchor_indices = 2 + offsets
    assert float(states.grad[episode_index, anchor_indices].abs().sum()) > 0
    assert predictor.scale.grad is not None
    assert float(predictor.scale.grad.abs()) > 0


def test_rollout_endpoint_loss_is_mean_over_batch_windows_objects_and_coordinates() -> None:
    prediction = torch.tensor(
        [
            [[[[1.0, 2.0]], [[3.0, 4.0]]]],
            [[[[5.0, 6.0]], [[7.0, 8.0]]]],
        ]
    ).reshape(2, 2, 1, 2)
    target = torch.zeros_like(prediction)

    loss = rollout_t2_endpoint_mse(prediction, target)
    expected = prediction.square().sum() / prediction.numel()
    torch.testing.assert_close(loss, expected)


def test_duplicating_windows_does_not_multiply_rollout_loss() -> None:
    prediction = torch.randn(4, 3, NUM_OBJECTS, STATE_DIM)
    target = torch.randn_like(prediction)
    original = rollout_t2_endpoint_mse(prediction, target)
    duplicated = rollout_t2_endpoint_mse(
        torch.cat((prediction, prediction), dim=1),
        torch.cat((target, target), dim=1),
    )

    torch.testing.assert_close(duplicated, original)
