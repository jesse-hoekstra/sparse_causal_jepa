"""GT-embedding regime: StateJepa contract, training, and the eval harness."""

from pathlib import Path

import torch

from scjepa.data import BounceDataset
from scjepa.eval import evaluate_identifiability
from scjepa.models.state_jepa import StateJepa, build_state_jepa
from scjepa.training import TrainConfig, Trainer

N = 3


def tiny_state_model() -> StateJepa:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    return build_state_jepa(
        state_dim=4,
        slot_size=16,
        pooling_heads=2,
        spartan_layers=1,
        spartan_embed_dim=None,
        spartan_mlp_hidden=32,
        spartan_mlp_layers=2,
    )


def test_state_jepa_forward_contract() -> None:
    model = tiny_state_model()
    states = torch.randn(2, 5, N, 4)  # Th=4 context + 1 target
    out = model(states)
    assert out.prediction.shape == (2, N, 16)
    assert out.target_slots.shape == (2, N, 16)
    assert out.context_slots.shape == (2, 4, N, 16)
    assert out.path_matrix.shape == (2, 2 * N, 2 * N)


def test_rollout_chains_shapes_and_anchors() -> None:
    """D16: K=6 transitions, Tp=3 -> 2 chains; flattened outputs stay (B*K, ...)."""
    model = tiny_state_model()
    states = torch.randn(2, 10, N, 4)  # Th=4 context, K=6 transitions
    out = model(states, context_len=4, rollout_horizon=3)
    assert out.prediction.shape == (2 * 6, N, 16)
    assert out.target_slots.shape == (2 * 6, N, 16)
    assert out.path_matrix.shape == (2 * 6, 2 * N, 2 * N)
    assert out.kinematic_state.shape == (2 * 2, N, 16)  # one anchor per chain
    assert out.causal_params.shape == (2, N, 16)


def test_rollout_is_autoregressive() -> None:
    """Chained predictions differ from teacher-forced ones (Tp=1 ≡ old D15)."""
    model = tiny_state_model().eval()  # eval: deterministic gates
    states = torch.randn(1, 8, N, 4)
    with torch.no_grad():
        chained = model(states, context_len=4, rollout_horizon=4)  # one chain of 4
        forced = model(states, context_len=4, rollout_horizon=1)  # 4 single steps
    # step 0 shares the same true anchor; later steps consume model outputs.
    torch.testing.assert_close(chained.prediction[0], forced.prediction[0])
    assert not torch.allclose(chained.prediction[1], forced.prediction[1])


def test_rollout_gradient_reaches_params_through_chain() -> None:
    """The pooled Ŝ^ph must receive gradient from LATE rollout steps too."""
    model = tiny_state_model()
    states = torch.randn(1, 6, N, 4)
    out = model(states, context_len=4, rollout_horizon=2)
    out.prediction[-1].sum().backward()  # pyright: ignore[reportUnknownMemberType]
    grads = [p.grad for p in model.pooling.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_rollout_horizon_must_divide_transitions() -> None:
    model = tiny_state_model()
    states = torch.randn(1, 9, N, 4)  # Th=4 -> K=5
    try:
        model(states, context_len=4, rollout_horizon=2)
    except ValueError as err:
        assert "divide" in str(err)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for K % Tp != 0")


def test_state_jepa_trains_on_bounce(tmp_path: Path) -> None:
    """input_key=states: the trainer runs the GT-embedding regime end to end."""
    dataset = BounceDataset(num_episodes=8, clip_len=4, num_balls=N, resolution=16, seed=2)
    config = TrainConfig(
        steps=3,
        batch_size=4,
        input_key="states",
        num_projections=16,
        sparsity_tau=0.5,
        log_every=1,
        checkpoint_every=1000,
        out_dir=str(tmp_path),
    )
    metrics = Trainer(tiny_state_model(), dataset, config).train()
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def test_identifiability_harness_end_to_end() -> None:
    """Untrained StateJepa on bounce: harness returns bounded, finite metrics."""
    dataset = BounceDataset(num_episodes=12, clip_len=4, num_balls=N, resolution=16, seed=3)
    report = evaluate_identifiability(
        tiny_state_model(), dataset, input_key="states", batch_size=6, max_batches=2
    )
    for key in ("pred_loss", "shd_state", "shd_param", "mcc", "path_density", "num_samples"):
        assert key in report.metrics
        assert torch.isfinite(torch.tensor(report.metrics[key])), key
    assert 0 <= report.metrics["mcc"] <= 1 + 1e-6
    assert report.metrics["num_samples"] == 12 * N
    assert report.recovery_learned.shape == report.recovery_true.shape
    assert 0 <= report.recovery_best_dim < 16
    assert report.per_slot_learned.shape == (12, N, 16)  # (episodes, N, d)
    assert report.per_slot_true.shape == (12, N)
