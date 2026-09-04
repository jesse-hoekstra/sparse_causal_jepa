"""Tests for the state-to-state model (experiments.pdf §6.2, Eqs. 15/38-40)."""

import pytest
import torch
from torch.utils.data import DataLoader

from scjepa.data import BounceDataset
from scjepa.eval import evaluate_identifiability
from scjepa.models import StateToStateModel, build_state_to_state

N = 3


def tiny_model(dense: bool = False, identity: bool = False) -> StateToStateModel:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    return build_state_to_state(
        num_slots=N,
        param_encoder_dim=16,
        param_encoder_heads=2,
        max_history=8,
        spartan_layers=2,
        spartan_embed_dim=32,
        spartan_mlp_hidden=32,
        spartan_mlp_layers=2,
        spartan_dense=dense,
        spartan_identity=identity,
    )


def test_forward_contract() -> None:
    """Tpar context -> theta; K = T - Tpar teacher-forced one-step transitions."""
    model = tiny_model()
    states = torch.randn(2, 6, N, 4)
    out = model(states, context_len=4)  # K = 2
    assert out.prediction.shape == (4, N, 4)
    assert out.target.shape == (4, N, 4)
    assert out.causal_params.shape == (2, N, 1)
    assert out.path_matrix.shape == (4, 2 * N, 2 * N)
    # Targets are the raw observed next states, in trajectory order.
    torch.testing.assert_close(out.target, states[:, 4:].flatten(0, 1))


def test_predictions_are_teacher_forced_at_true_states() -> None:
    """Eq. 38 anchors every transition at the observed Z_t.

    Corrupting one anchor must not affect the other transitions (no rollout).
    """
    model = tiny_model(dense=True).eval()
    states = torch.randn(1, 6, N, 4)
    out = model(states, context_len=3)  # K = 3, anchors Z_2, Z_3, Z_4
    corrupted = states.clone()
    corrupted[:, 3] += 100.0  # corrupt the SECOND anchor only
    out2 = model(corrupted, context_len=3)
    # First transition (anchor Z_2) unchanged; second changed.
    torch.testing.assert_close(out.prediction[0], out2.prediction[0])
    assert not torch.allclose(out.prediction[1], out2.prediction[1])


def test_same_theta_reused_for_every_transition() -> None:
    """The episode's theta is computed once from the context window only."""
    model = tiny_model(dense=True).eval()
    states = torch.randn(1, 6, N, 4)
    out = model(states, context_len=3)
    corrupted = states.clone()
    corrupted[:, 3:] += 100.0  # future frames must not affect theta
    out2 = model(corrupted, context_len=3)
    torch.testing.assert_close(out.causal_params, out2.causal_params)


def test_gradient_reaches_parameter_encoder() -> None:
    """The transition objective trains P_eta through SPARTAN's param tokens."""
    model = tiny_model(dense=True)
    out = model(torch.randn(2, 5, N, 4), context_len=3)
    (out.prediction - out.target).square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
    head_grad = model.parameter_encoder.head.weight.grad
    assert head_grad is not None
    assert head_grad.abs().sum() > 0


def test_identity_reference_predictions_ignore_theta() -> None:
    model = tiny_model(identity=True).eval()
    states = torch.randn(1, 4, N, 4)
    out = model(states, context_len=2)
    # Corrupt the context BEFORE the first anchor (Z_{Tpar-1} is the last
    # context frame): theta changes, token-local predictions must not.
    corrupted = states.clone()
    corrupted[:, 0] += 10.0
    out2 = model(corrupted, context_len=2)
    torch.testing.assert_close(out.prediction, out2.prediction)


def test_context_len_guards() -> None:
    model = tiny_model()
    with pytest.raises(ValueError, match="context_len"):
        model(torch.randn(1, 4, N, 4), context_len=4)
    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(1, 4, N))


def test_identifiability_harness_end_to_end() -> None:
    """Untrained model on real bounce data: harness returns bounded metrics."""
    dataset = BounceDataset(
        num_episodes=12,
        clip_len=4,
        num_balls=N,
        seed=3,
        render=False,
        mass_normal=(1.5, 0.5),
        radius_from_mass=True,
    )
    report = evaluate_identifiability(
        tiny_model(), dataset, batch_size=6, max_batches=2, context_len=2, lambda_logit=1e-3
    )
    for key in (
        "pred_loss",
        "loss_rollout_t2_raw",
        "loss_rollout_t2_weighted",
        "constraint_loss",
        "shd",
        "mcc",
        "path_density",
    ):
        assert key in report.metrics
        assert torch.isfinite(torch.tensor(report.metrics[key])), key
    # Eq. 13 raw units: constraint = pred + lambda_logit * logit_penalty.
    expected = report.metrics["pred_loss"] + 1e-3 * report.diagnostics["logit_penalty"]
    assert abs(report.metrics["constraint_loss"] - expected) < 1e-9
    assert 0 <= report.metrics["mcc"] <= 1 + 1e-6
    assert 0 <= report.metrics["shd"] <= 2 * N * N
    assert report.diagnostics["num_samples"] == 12
    assert report.learned_coordinates.shape == (12, N)
    assert report.recovery_matrix.shape == (N, N)


def test_harness_reports_t2_constraint_and_no_grad_oe_rollout() -> None:
    """The deterministic held-out constraint and OE diagnostic are separate."""
    dataset = BounceDataset(
        num_episodes=6,
        clip_len=8,
        num_balls=N,
        seed=4,
        render=False,
        mass_normal=(1.5, 0.5),
        radius_from_mass=True,
    )
    report = evaluate_identifiability(
        tiny_model(),
        dataset,
        batch_size=3,
        context_len=3,
        lambda_rollout_t2=1.0,
        num_rollout_t2_anchors=3,
        rollout_t2_horizon=2,
        oe_eval_horizon=5,
        oe_tolerance_nrmse=0.1,
        oe_coordinate_std=torch.ones(4),
    )
    for key in (
        "loss_rollout_t2_raw",
        "loss_rollout_t2_weighted",
        "trajectory_reconstruction_mse_k5",
        "oe_sample_satisfaction_k5",
        "oe_k5_worst_step_nrmse_p50",
        "oe_k5_worst_step_nrmse_p95",
    ):
        assert key in report.metrics
        assert torch.isfinite(torch.tensor(report.metrics[key])), key
    assert report.metrics["constraint_loss"] == pytest.approx(
        report.metrics["pred_loss"] + report.metrics["loss_rollout_t2_weighted"],
        rel=1e-7,
    )


def test_harness_batches_gt_graphs_per_transition() -> None:
    """K > 1: one ground-truth local graph per predicted transition."""
    dataset = BounceDataset(
        num_episodes=4,
        clip_len=5,
        num_balls=N,
        seed=5,
        render=False,
        mass_normal=(1.5, 0.5),
        radius_from_mass=True,
    )
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    model = tiny_model()
    out = model(batch["states"], context_len=2)  # K = 3
    assert out.path_matrix.shape[0] == 4 * 3
