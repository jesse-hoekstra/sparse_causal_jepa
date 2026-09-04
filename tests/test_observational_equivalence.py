"""Tests for the deterministic, sampled observational-equivalence diagnostic."""

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from scjepa.eval import (
    evaluate_identifiability,
    oe_worst_step_nrmse,
    summarize_oe,
    training_coordinate_std,
)
from scjepa.models.state_to_state import StateToStateModel

N = 2
D = 4


class TensorEpisodeDataset(Dataset[dict[str, Tensor]]):
    """Small deterministic tensor-backed dataset."""

    def __init__(self, states: Tensor) -> None:
        """Store a finite set of trajectories."""
        self.states = states

    def __len__(self) -> int:
        """Return the episode count."""
        return self.states.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return one trajectory and synthetic evaluation labels."""
        length = self.states.shape[1]
        return {
            "states": self.states[index],
            "params": self.states[index, 0, :, :1],
            "contacts": torch.zeros(length - 1, N, N, dtype=torch.bool),
        }


class InitialValueEncoder(nn.Module):
    """Expose a varying, perfectly recoverable episode coordinate."""

    def forward(self, context: Tensor) -> Tensor:
        """Return an episode-varying scalar for each object."""
        return context[:, 0, :, :1]


class ExactIncrementPredictor(nn.Module):
    """Synthetic transition that exactly follows the generated trajectories."""

    def __init__(self, increment: Tensor) -> None:
        """Store the exact per-coordinate transition increment."""
        super().__init__()
        self.register_buffer("increment", increment)

    def forward(self, state: Tensor, causal_params: Tensor) -> SimpleNamespace:
        """Advance the synthetic state by one exact increment."""
        del causal_params
        assert not torch.is_grad_enabled()
        batch, slots = state.shape[:2]
        token_count = 2 * slots
        path = torch.eye(token_count, device=state.device).expand(batch, -1, -1)
        zero = state.new_zeros(())
        return SimpleNamespace(
            prediction=state + self.increment,
            path_matrix=path,
            sparsity=zero,
            logit_penalty=zero,
            mean_abs_logit=zero,
            mean_gate_probability=zero,
            gate_entropy=zero,
        )


def exact_dataset(num_episodes: int = 8, length: int = 60) -> TensorEpisodeDataset:
    increment = torch.tensor([0.01, -0.02, 0.03, -0.04])
    initial = torch.linspace(-1.0, 1.0, num_episodes * N * D).reshape(num_episodes, N, D)
    time = torch.arange(length).view(1, length, 1, 1)
    states = initial[:, None] + time * increment.view(1, 1, 1, D)
    return TensorEpisodeDataset(states)


def exact_model() -> StateToStateModel:
    increment = torch.tensor([0.01, -0.02, 0.03, -0.04]).view(1, 1, D)
    # The production constructor annotations are stricter than this synthetic
    # spy, but both modules implement the exact runtime contracts under test.
    return StateToStateModel(  # pyright: ignore[reportArgumentType]
        InitialValueEncoder(),  # pyright: ignore[reportArgumentType]
        ExactIncrementPredictor(increment),  # pyright: ignore[reportArgumentType]
    )


def test_exact_synthetic_k30_predictor_has_full_sample_satisfaction() -> None:
    report = evaluate_identifiability(
        exact_model(),
        exact_dataset(),
        batch_size=4,
        context_len=30,
        lambda_rollout_t2=1.0,
        num_rollout_t2_anchors=8,
        rollout_t2_horizon=2,
        oe_eval_horizon=30,
        oe_tolerance_nrmse=0.10,
        oe_coordinate_std=torch.ones(D),
    )
    assert report.metrics["oe_sample_satisfaction_k30"] == 1.0
    assert report.metrics["trajectory_reconstruction_mse_k30"] == pytest.approx(0.0, abs=1e-12)
    assert report.metrics["oe_k30_worst_step_nrmse_p50"] == pytest.approx(0.0, abs=1e-6)
    assert report.metrics["oe_k30_worst_step_nrmse_p95"] == pytest.approx(0.0, abs=1e-6)
    assert report.metrics["pred_loss"] == pytest.approx(0.0, abs=1e-12)
    assert report.metrics["loss_rollout_t2_raw"] == pytest.approx(0.0, abs=1e-12)


def test_worst_step_nrmse_and_controlled_errors_lower_satisfaction() -> None:
    target = torch.zeros(4, 5, N, D)
    prediction = target.clone()
    scales = torch.tensor([1.0, 2.0, 4.0, 8.0])
    normalized_maxima = torch.tensor([0.0, 0.49, 0.51, 1.0])
    # Put the complete normalized error into one late step. Every N,D entry
    # then has the requested standardized magnitude, so its step NRMSE is exact.
    prediction[:, -1] = normalized_maxima[:, None, None] * scales
    errors = oe_worst_step_nrmse(prediction, target, scales)
    torch.testing.assert_close(errors, normalized_maxima)
    summary = summarize_oe(errors, tolerance=0.50)
    assert summary.satisfaction == 0.5
    assert summarize_oe(errors * 2, tolerance=0.50).satisfaction < summary.satisfaction

    # A temporal mean would hide this isolated bad step; the required max does not.
    one_spike = torch.zeros(1, 30, N, D)
    one_spike[:, -1] = scales
    worst = oe_worst_step_nrmse(one_spike, torch.zeros_like(one_spike), scales)
    assert worst.item() == pytest.approx(1.0)


def test_training_coordinate_std_uses_all_states_and_population_correction() -> None:
    states = torch.tensor(
        [
            [[[[0.0, 2.0, 4.0, 6.0]]]],
            [[[[2.0, 4.0, 8.0, 10.0]]]],
        ]
    ).reshape(2, 1, 1, D)
    dataset = TensorEpisodeDataset(states.expand(-1, -1, N, -1).clone())
    expected = dataset.states.reshape(-1, D).std(dim=0, correction=0)
    torch.testing.assert_close(training_coordinate_std(dataset, batch_size=1), expected)


def test_oe_summary_uses_inclusive_threshold_and_global_quantiles() -> None:
    errors = torch.tensor([2.0, 1.0, 0.0])
    summary = summarize_oe(errors, tolerance=1.0)
    assert summary.satisfaction == pytest.approx(2 / 3)
    assert summary.p50 == pytest.approx(1.0)
    assert summary.p95 == pytest.approx(1.9)
