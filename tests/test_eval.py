"""Tests for the identifiability diagnostics (graphs/SHD, correlations/MCC)."""

import pytest
import torch
from torch.utils.data import DataLoader

from scjepa.data import BounceDataset
from scjepa.eval import (
    correlation_matrix,
    gt_graphs_from_contacts,
    marginal_recovery,
    mean_max_correlation,
    read_learned_graphs,
    structural_hamming_distance,
)
from scjepa.models.jepa import build_scjepa

N = 3


def test_gt_graph_derivation() -> None:
    """State graph = last contacts + self; param diag = involved-in-contact."""
    contacts = torch.zeros(1, 2, N, N, dtype=torch.bool)
    contacts[0, 0, 0, 1] = contacts[0, 0, 1, 0] = True  # earlier transition: ignored
    contacts[0, 1, 1, 2] = contacts[0, 1, 2, 1] = True  # last transition: pair (1, 2)
    state_graph, param_graph = gt_graphs_from_contacts(contacts)
    expected_state = torch.eye(N, dtype=torch.bool)
    expected_state[1, 2] = expected_state[2, 1] = True
    assert torch.equal(state_graph[0], expected_state)
    expected_param = torch.zeros(N, N, dtype=torch.bool)
    expected_param[1, 2] = expected_param[2, 1] = True
    expected_param[1, 1] = expected_param[2, 2] = True  # own mass matters in contact
    assert torch.equal(param_graph[0], expected_param)  # ball 0: no mass edges at all


def test_learned_graph_readout_and_shd() -> None:
    path = torch.zeros(1, 2 * N, 2 * N)
    path[0, :, :] = 0.0
    path[0, 0, 0] = 1.0  # self path
    path[0, 0, 1] = 2.0  # two paths state 1 -> state 0
    path[0, 1, N + 2] = 1.0  # param 2 -> state 1
    state_graph, param_graph = read_learned_graphs(path, num_slots=N)
    assert state_graph[0, 0, 1]
    assert state_graph[0, 0, 0]
    assert param_graph[0, 1, 2]
    assert param_graph.sum() == 1
    # SHD: identical -> 0; one flip -> 1.
    assert structural_hamming_distance(state_graph, state_graph).item() == 0
    flipped = state_graph.clone()
    flipped[0, 2, 2] = ~flipped[0, 2, 2]
    assert structural_hamming_distance(state_graph, flipped).item() == 1


def test_correlation_recovers_planted_diffeomorphism() -> None:
    """ŝ = tanh(θ) in one dim + noise elsewhere ⇒ that dim wins, MCC high."""
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    theta = torch.rand(500, 1) * 2.5 + 0.5  # masses
    learned = torch.randn(500, 6)
    learned[:, 3] = torch.tanh(theta.squeeze(-1)) + 0.01 * torch.randn(500)
    best_dim, best_values, true_values = marginal_recovery(learned, theta)
    assert best_dim.item() == 3
    # tanh saturates over this mass range, so Pearson |corr| sits well below 1
    # even for a perfect diffeomorphism (the docstring caveat); the monotone
    # scatter below is the decisive check.
    assert mean_max_correlation(learned, theta) > 0.8
    # The scatter pairs are monotone (diffeomorphism shape), up to noise.
    order = true_values.argsort()
    diffs = best_values[order].diff()
    assert (diffs > -0.05).float().mean() > 0.95


def test_correlation_low_for_independent_noise() -> None:
    torch.manual_seed(1)  # pyright: ignore[reportUnknownMemberType]
    theta = torch.rand(500, 1)
    learned = torch.randn(500, 6)
    assert mean_max_correlation(learned, theta) < 0.2


def test_correlation_matrix_guards() -> None:
    with pytest.raises(ValueError, match="equal S"):
        correlation_matrix(torch.randn(4, 2), torch.randn(5, 1))
    with pytest.raises(ValueError, match="at least 2"):
        correlation_matrix(torch.randn(1, 2), torch.randn(1, 1))


def test_diagnostics_run_on_bounce_pipeline() -> None:
    """Untrained model on real bounce data: everything wires, values bounded."""
    torch.manual_seed(2)  # pyright: ignore[reportUnknownMemberType]
    dataset = BounceDataset(num_episodes=4, clip_len=4, num_balls=N, resolution=64, seed=5)
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    model = build_scjepa(
        resolution=64,
        num_slots=N,
        slot_size=16,
        slot_mlp_size=32,
        num_iterations=1,
        enc_channels=(3, 8, 8),
        enc_out_channels=16,
        pooling_heads=2,
        spartan_layers=1,
        spartan_embed_dim=None,
        spartan_mlp_hidden=32,
        spartan_mlp_layers=2,
    )
    model.eval()
    with torch.no_grad():
        out = model(batch["frames"])
    state_gt, param_gt = gt_graphs_from_contacts(batch["contacts"])
    state_learned, param_learned = read_learned_graphs(out.path_matrix, num_slots=N)
    for learned_graph, gt_graph in ((state_learned, state_gt), (param_learned, param_gt)):
        shd = structural_hamming_distance(learned_graph, gt_graph)
        assert 0 <= shd.item() <= N * N
    # (episode, ball) samples: flatten for the parameter metrics.
    learned_flat = out.causal_params.reshape(-1, out.causal_params.shape[-1])
    target_flat = batch["params"].reshape(-1, 1)
    mcc = mean_max_correlation(learned_flat, target_flat)
    assert 0 <= mcc.item() <= 1 + 1e-6


def test_nonlinear_mcc_beats_pearson_on_diffeomorphism() -> None:
    """F.1 metric: near-1 on a planted diffeomorphism where Pearson saturates."""
    from scjepa.eval import nonlinear_mcc

    torch.manual_seed(4)  # pyright: ignore[reportUnknownMemberType]
    theta = torch.rand(800, 1) * 2.5 + 0.5
    learned = torch.randn(800, 4)
    learned[:, 2] = torch.tanh(theta.squeeze(-1)) + 0.01 * torch.randn(800)
    score = nonlinear_mcc(learned, theta, epochs=200)
    assert score > 0.9
    assert score > mean_max_correlation(learned, theta)
    # And near-zero relationship stays near zero (no overfitting inflation).
    noise_score = nonlinear_mcc(torch.randn(800, 4), theta, epochs=200)
    assert noise_score < 0.3
