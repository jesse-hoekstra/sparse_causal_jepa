"""Physical recovery, temporal graph alignment, and evaluation isolation."""

from pathlib import Path

import pytest
import torch
from torch import Tensor

from scjepa.data.bounce import BounceDataset
from scjepa.eval.parameters import MccReport
from scjepa.eval.state_probes import fit_state_probe, state_probe_metrics
from scjepa.eval.visual_alignment import tracking_metrics
from scjepa.eval.visual_artifacts import save_visual_artifacts
from scjepa.eval.visual_to_visual import evaluate_visual_to_visual, graph_in_slot_order
from scjepa.models.visual_to_visual import build_visual_to_visual
from scripts.eval_visual_to_visual import require_noncollapsed


def test_graph_uses_each_transitions_contacts_and_conjugates_both_axes() -> None:
    contacts = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    contacts[0, 1, 0, 0] = True  # only ball 0 bounces at the second transition
    graph = graph_in_slot_order(contacts, torch.tensor([[1, 0]]))
    expected = torch.tensor(
        [
            [[True, False, False, False], [False, True, False, False]],
            [[True, False, False, False], [False, True, False, True]],
        ]
    )
    torch.testing.assert_close(graph, expected)


def test_tracking_preserves_one_assignment_and_detects_identity_switches() -> None:
    truth = torch.tensor([[[[0.2, 0.3], [0.8, 0.7]]]]).expand(1, 3, 2, 2).clone()
    good = tracking_metrics(truth, truth, torch.tensor([[0, 1]]))
    assert good == {"slot_centroid_rmse": 0.0, "slot_switch_rate": 0.0}
    switched = truth.clone()
    switched[:, 1:] = switched[:, 1:, [1, 0]]
    bad = tracking_metrics(switched, truth, torch.tensor([[0, 1]]))
    assert bad["slot_switch_rate"] == 0.5
    assert bad["slot_centroid_rmse"] > 0.1


def test_probe_recovers_linear_state_on_unseen_episodes_and_uses_fit_statistics() -> None:
    generator = torch.Generator().manual_seed(7)
    train = torch.randn(1000, 4, generator=generator)
    test = torch.randn(300, 4, generator=generator) + 3
    transform = torch.randn(4, 12, generator=generator)
    features = train @ transform
    probe = fit_state_probe(features, train)
    torch.testing.assert_close(probe.feature_mean, features.double().mean(dim=0))
    metrics = state_probe_metrics(probe.predict(test @ transform), test)
    assert metrics["position_probe_r2"] > 0.999
    assert metrics["velocity_probe_r2"] > 0.999
    shuffled = test[torch.randperm(test.shape[0], generator=generator)]
    broken = state_probe_metrics(probe.predict(test @ transform), shuffled)
    assert broken["position_probe_r2"] < 0
    assert broken["velocity_probe_r2"] < 0


def test_probe_constant_features_have_finite_mean_baseline() -> None:
    labels = torch.arange(80).float().reshape(20, 4)
    probe = fit_state_probe(torch.ones(20, 8), labels)
    predicted = probe.predict(torch.full((3, 8), 7.0))
    torch.testing.assert_close(predicted, labels.double().mean(dim=0).expand(3, 4))


def _dataset(seed: int) -> BounceDataset:
    return BounceDataset(
        num_episodes=4,
        clip_len=4,
        num_balls=2,
        seed=seed,
        render=True,
        resolution=64,
        uniform_appearance=True,
        render_radius_from_mass=False,
        cache=False,
    )


def _fast_mcc(learned: Tensor, target: Tensor) -> MccReport:
    return MccReport(
        torch.tensor(0.0), torch.zeros(target.shape[1], learned.shape[1]), learned.shape[0]
    )


def test_visual_evaluation_preserves_rng_and_training_mode_and_writes_compact_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("scjepa.eval.visual_to_visual.nonlinear_mcc", _fast_mcc)
    model = build_visual_to_visual(
        num_slots=2,
        slot_size=8,
        state_dim=8,
        param_encoder_dim=8,
        param_encoder_heads=2,
        max_history=8,
        spartan_embed_dim=16,
        spartan_mlp_hidden=16,
    )
    training, test = _dataset(1), _dataset(29)
    rng = torch.get_rng_state().clone()
    report = evaluate_visual_to_visual(
        model,
        test,
        batch_size=2,
        context_len=2,
        num_rollout_t2_anchors=1,
        oe_eval_horizon=2,
        probe_dataset=training,
        max_probe_batches=1,
    )
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    for key in (
        "position_probe_r2",
        "velocity_probe_r2",
        "slot_centroid_rmse",
        "slot_switch_rate",
        "branch_slot_disagreement",
        "latent_rollout_k2_nrmse",
    ):
        assert key in report.metrics
        assert torch.isfinite(torch.tensor(report.metrics[key]))
    for path in save_visual_artifacts(report, tmp_path):
        assert path.stat().st_size > 1000
    with pytest.raises(ValueError, match="separate episode"):
        evaluate_visual_to_visual(model, test, probe_dataset=test)


def test_dense_calibration_rejects_label_free_collapse_without_using_recovery_labels() -> None:
    healthy = dict(
        constraint_loss=0.5,
        target_variance=0.1,
        target_temporal_variance=0.01,
        target_effective_rank=3.0,
    )
    require_noncollapsed(healthy, variance_floor=1e-4)
    for key, value in (
        ("target_variance", 1e-7),
        ("target_temporal_variance", 0.0),
        ("target_effective_rank", 1.0),
        ("constraint_loss", float("nan")),
    ):
        with pytest.raises(ValueError, match="cannot calibrate tau"):
            require_noncollapsed({**healthy, key: value}, variance_floor=1e-4)
