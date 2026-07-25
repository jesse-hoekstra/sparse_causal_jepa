"""Trajectory-level visual-track alignment (experiments.pdf Eqs. 98-100)."""

import pytest
import torch

from scjepa.data.bounce import BounceDataset
from scjepa.losses.alignment import (
    align_to_assignment,
    assignment_cost,
    coordinate_scales,
    trajectory_assignment,
)

SCALES = torch.ones(4)


def _episode(batch: int = 3, steps: int = 7, tracks: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(batch, steps, tracks, 4, generator=generator)


def test_recovers_the_permutation_that_was_applied() -> None:
    """A permuted copy of the targets must be matched straight back."""
    targets = _episode()
    permutation = torch.stack([torch.randperm(5) for _ in range(targets.shape[0])])
    predictions = align_to_assignment(targets, permutation, track_dim=2)
    assert torch.equal(trajectory_assignment(predictions, targets, SCALES), permutation)


def test_alignment_restores_the_targets() -> None:
    """Aligning targets by the recovered assignment reproduces the predictions."""
    targets = _episode()
    permutation = torch.stack([torch.randperm(5) for _ in range(targets.shape[0])])
    predictions = align_to_assignment(targets, permutation, track_dim=2)
    assignment = trajectory_assignment(predictions, targets, SCALES)
    assert torch.allclose(align_to_assignment(targets, assignment, track_dim=2), predictions)


def test_assignment_is_one_to_one() -> None:
    """Eq. 99 is a permutation: every physical track is claimed exactly once."""
    predictions, targets = _episode(), _episode(batch=3) + 0.5
    assignment = trajectory_assignment(predictions, targets, SCALES)
    for episode in assignment:
        assert sorted(episode.tolist()) == list(range(5))


def test_one_assignment_per_episode_not_per_frame() -> None:
    """A mid-episode identity switch must NOT be absorbed by rematching.

    Half the episode is emitted under one permutation and half under another.
    A per-frame matcher would score both halves perfectly; the trajectory-level
    rule must keep a positive residual, which is exactly why S6.4 forbids
    per-timestep rematching.
    """
    targets = _episode(batch=1, steps=8)
    first = torch.arange(5).unsqueeze(0)
    second = torch.tensor([[1, 0, 2, 3, 4]])
    switched = torch.cat(
        [
            align_to_assignment(targets[:, :4], first, track_dim=2),
            align_to_assignment(targets[:, 4:], second, track_dim=2),
        ],
        dim=1,
    )
    assignment = trajectory_assignment(switched, targets, SCALES)
    aligned = align_to_assignment(targets, assignment, track_dim=2)
    assert (switched - aligned).square().mean() > 0.1


def test_no_gradient_flows_through_the_assignment() -> None:
    """The discrete choice is detached; only the selected predictions carry grad."""
    predictions = _episode().requires_grad_(True)
    targets = _episode(batch=3) + 0.5
    assignment = trajectory_assignment(predictions, targets, SCALES)
    assert not assignment.is_floating_point()
    aligned = align_to_assignment(targets, assignment, track_dim=2)
    assert aligned.grad_fn is None
    (predictions - aligned).square().mean().backward()
    assert predictions.grad is not None
    assert bool(predictions.grad.abs().sum() > 0)


def test_cost_matches_equation_98_elementwise() -> None:
    """Spot-check D_e(i, j) against a literal transcription of Eq. 98."""
    predictions, targets = _episode(batch=2, steps=4), _episode(batch=2, steps=4) + 0.3
    scales = torch.tensor([1.0, 2.0, 0.5, 4.0])
    cost = assignment_cost(predictions, targets, scales)
    steps, coords = predictions.shape[1], predictions.shape[3]
    expected = sum(
        ((predictions[1, t, 2, a] - targets[1, t, 3, a]) / scales[a]) ** 2
        for t in range(steps)
        for a in range(coords)
    ) / (steps * coords)
    assert torch.allclose(cost[1, 2, 3], expected)


def test_scales_are_per_coordinate_and_positive() -> None:
    """sigma_a is one frozen number per physical coordinate."""
    dataset = BounceDataset(
        num_episodes=6, clip_len=12, num_balls=5, seed=1, render=False, radius_from_mass=True
    )
    scales = coordinate_scales(dataset, num_episodes=6, context_len=6)
    assert scales.shape == (4,)
    assert bool((scales > 0).all())


def test_rejects_mismatched_shapes() -> None:
    """Shape errors must be loud, not silently broadcast into a wrong matching."""
    with pytest.raises(ValueError, match="same shape"):
        assignment_cost(_episode(), _episode(steps=5), SCALES)
    with pytest.raises(ValueError, match="scales"):
        assignment_cost(_episode(), _episode(), torch.ones(3))
    with pytest.raises(ValueError, match="non-batch axis"):
        align_to_assignment(_episode(), torch.zeros(3, 5, dtype=torch.long), track_dim=0)


def test_aligns_masses_as_well_as_states() -> None:
    """One assignment serves the prediction loss AND parameter evaluation."""
    masses = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
    assignment = torch.tensor([[4, 3, 2, 1, 0]])
    aligned = align_to_assignment(masses, assignment, track_dim=1)
    assert aligned.squeeze(-1).tolist() == [[5.0, 4.0, 3.0, 2.0, 1.0]]
