"""Tests for the identifiability diagnostics (graphs/SHD and the F.1 MCC)."""

import pytest
import torch
from torch.utils.data import DataLoader

from scjepa.data import BounceDataset
from scjepa.eval import (
    gt_causal_graph_from_contacts,
    nonlinear_mcc,
    read_learned_graph,
    structural_hamming_distance,
)
from scjepa.models import build_experiment1

N = 3


def test_gt_causal_graph_derivation() -> None:
    """Columns are [state | params]: state edge on contact or self; mass edge on contact."""
    contacts = torch.zeros(1, 2, N, N, dtype=torch.bool)
    contacts[0, 0, 0, 1] = contacts[0, 0, 1, 0] = True  # earlier transition: ignored
    contacts[0, 1, 1, 2] = contacts[0, 1, 2, 1] = True  # last transition: pair (1, 2)
    graph = gt_causal_graph_from_contacts(contacts)
    assert graph.shape == (1, N, 2 * N)
    expected_state = torch.eye(N, dtype=torch.bool)
    expected_state[1, 2] = expected_state[2, 1] = True
    assert torch.equal(graph[0, :, :N], expected_state)
    expected_param = torch.zeros(N, N, dtype=torch.bool)
    expected_param[1, 2] = expected_param[2, 1] = True
    expected_param[1, 1] = expected_param[2, 2] = True  # own mass matters in contact
    assert torch.equal(graph[0, :, N:], expected_param)  # ball 0: no mass edges at all


def test_learned_graph_readout_and_shd() -> None:
    path = torch.zeros(1, 2 * N, 2 * N)
    path[0, 0, 0] = 1.0  # self path
    path[0, 0, 1] = 2.0  # two paths state 1 -> state 0
    path[0, 1, N + 2] = 1.0  # param 2 -> state 1
    graph = read_learned_graph(path, num_slots=N)
    assert graph.shape == (1, N, 2 * N)
    assert graph[0, 0, 0]  # state self-path retained
    assert graph[0, 0, 1]  # state source retained
    assert graph[0, 1, N + 2]  # parameter source retained
    assert graph.sum() == 3
    # Undecoded parameter ROWS are excluded even when they carry paths.
    assert read_learned_graph(torch.ones(1, 2 * N, 2 * N), num_slots=N).shape == (1, N, 2 * N)
    # SHD: identical -> 0; one flip -> 1.
    assert structural_hamming_distance(graph, graph).item() == 0
    flipped = graph.clone()
    flipped[0, 2, 2] = ~flipped[0, 2, 2]
    assert structural_hamming_distance(graph, flipped).item() == 1


def test_mcc_matrix_is_true_by_learned() -> None:
    """R² rows are TRUE parameters, columns learned coordinates (App. F.1's I x J)."""
    torch.manual_seed(8)  # pyright: ignore[reportUnknownMemberType]
    target = torch.randn(600, 3)
    learned = target[:, [2, 0, 1]]  # learned j carries true parameter [2, 0, 1][j]
    report = nonlinear_mcc(learned, target, epochs=100)
    assert report.matrix.shape == (3, 3)
    # true 0 lives in learned 1, true 1 in learned 2, true 2 in learned 0.
    assert [int(value) for value in report.matrix.argmax(dim=1)] == [1, 2, 0]
    assert report.score > 0.98
    assert report.num_samples == 600


def test_mcc_is_permutation_insensitive() -> None:
    """The kept metric credits a mass to ANY learned coordinate (no bijection)."""
    torch.manual_seed(7)  # pyright: ignore[reportUnknownMemberType]
    theta = torch.randn(400, 3)
    permuted = nonlinear_mcc(theta[:, [2, 0, 1]], theta, epochs=100).score
    identity = nonlinear_mcc(theta, theta, epochs=100).score
    assert abs(float(permuted) - float(identity)) < 0.02


def test_mcc_credits_one_learned_coordinate_for_several_true_parameters() -> None:
    """No bijection is enforced: the same column may win several rows."""
    torch.manual_seed(11)  # pyright: ignore[reportUnknownMemberType]
    shared = torch.randn(400, 1)
    # True rows 0 and 1 are both determined by `shared`; only one learned
    # coordinate carries it, yet both rows score highly.
    target = torch.cat((shared, 2.0 * shared + 1.0, torch.randn(400, 1)), dim=1)
    learned = torch.cat((shared, torch.randn(400, 2)), dim=1)
    report = nonlinear_mcc(learned, target, epochs=200)
    assert [int(value) for value in report.matrix.argmax(dim=1)[:2]] == [0, 0]
    assert float(report.matrix[0, 0]) > 0.95
    assert float(report.matrix[1, 0]) > 0.95


def test_mcc_guards() -> None:
    with pytest.raises(ValueError, match="equal S"):
        nonlinear_mcc(torch.randn(4, 2), torch.randn(5, 1), epochs=2)
    with pytest.raises(ValueError, match="at least 2"):
        nonlinear_mcc(torch.randn(1, 2), torch.randn(1, 1), epochs=2)


def test_diagnostics_run_on_bounce_pipeline() -> None:
    """Untrained Experiment-1 model on real bounce data: everything wires."""
    torch.manual_seed(2)  # pyright: ignore[reportUnknownMemberType]
    dataset = BounceDataset(num_episodes=4, clip_len=4, num_balls=N, seed=5, render=False)
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    model = build_experiment1(
        num_slots=N, spartan_layers=1, spartan_embed_dim=32, spartan_mlp_hidden=32
    )
    model.eval()
    with torch.no_grad():
        out = model(batch["states"], context_len=3)  # K = 1
    graph_gt = gt_causal_graph_from_contacts(batch["contacts"])
    graph_learned = read_learned_graph(out.path_matrix, num_slots=N)
    assert 0 <= structural_hamming_distance(graph_learned, graph_gt).item() <= 2 * N * N
    # One row per episode for the parameter metric.
    mcc = nonlinear_mcc(out.causal_params.flatten(1), batch["params"].flatten(1), epochs=20).score
    assert 0 <= mcc.item() <= 1 + 1e-6


def test_nonlinear_mcc_recovers_planted_diffeomorphism() -> None:
    """F.1 metric: near-1 on a planted tanh diffeomorphism, near-0 on noise."""
    torch.manual_seed(4)  # pyright: ignore[reportUnknownMemberType]
    theta = torch.rand(800, 1) * 2.5 + 0.5
    learned = torch.randn(800, 4)
    learned[:, 2] = torch.tanh(theta.squeeze(-1)) + 0.01 * torch.randn(800)
    report = nonlinear_mcc(learned, theta, epochs=200)
    assert report.score > 0.9
    assert int(report.matrix.argmax(dim=1)[0]) == 2
    # And near-zero relationship stays near zero (no overfitting inflation).
    assert nonlinear_mcc(torch.randn(800, 4), theta, epochs=200).score < 0.3


def test_nonlinear_mcc_preserves_global_torch_rng() -> None:
    """Periodic evaluation must not change subsequent stochastic training draws."""
    torch.manual_seed(17)  # pyright: ignore[reportUnknownMemberType]
    target = torch.randn(40, 1)
    learned = torch.cat((target.square(), torch.randn(40, 1)), dim=1)
    before = torch.get_rng_state().clone()
    nonlinear_mcc(learned, target, epochs=2, seed=9)
    assert torch.equal(torch.get_rng_state(), before)
