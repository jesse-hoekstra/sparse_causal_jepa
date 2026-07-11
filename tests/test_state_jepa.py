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
