"""GT-embedding regime: StateJepa contract, training, and the eval harness."""

from pathlib import Path

import pytest
import torch

from scjepa.data import BounceDataset
from scjepa.eval import evaluate_identifiability
from scjepa.models import TrackedSlotAttentionPooling
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
    assert grads
    assert any(g.abs().sum() > 0 for g in grads)


def test_rollout_horizon_must_divide_transitions() -> None:
    model = tiny_state_model()
    states = torch.randn(1, 9, N, 4)  # Th=4 -> K=5
    with pytest.raises(ValueError, match="divide"):
        model(states, context_len=4, rollout_horizon=2)


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


def tiny_gt_model(*, spartan_dense: bool = False, spartan_identity: bool = False) -> StateJepa:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    return build_state_jepa(
        state_dim=4,
        slot_size=16,
        pooling_heads=2,
        spartan_layers=1,
        spartan_embed_dim=None,
        spartan_mlp_hidden=32,
        spartan_mlp_layers=2,
        gt_states=True,
        spartan_dense=spartan_dense,
        spartan_identity=spartan_identity,
    )


def tiny_gt_tracked_slot_model() -> StateJepa:
    """Small version of Example 1's tracked iterative parameter encoder."""
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    return build_state_jepa(
        state_dim=4,
        num_slots=N,
        slot_size=16,
        pooling_heads=2,
        pooling_type="tracked_slot_attention",
        parameter_slot_iterations=2,
        param_dim=1,
        spartan_layers=1,
        spartan_embed_dim=None,
        spartan_mlp_hidden=32,
        spartan_mlp_layers=2,
        spartan_node_embeddings=True,
        gt_states=True,
    )


def test_gt_states_forward_contract() -> None:
    """D20: predictions/targets/anchors live in the raw 4-dim GT state space."""
    model = tiny_gt_model()
    states = torch.randn(2, 5, N, 4)
    out = model(states)
    assert out.prediction.shape == (2, N, 4)
    assert out.target_slots.shape == (2, N, 4)
    torch.testing.assert_close(out.target_slots, states[:, 4])  # RAW states, untouched
    torch.testing.assert_close(out.kinematic_state, states[:, 3])  # raw anchor
    assert out.context_slots.shape == (2, 4, N, 16)  # param path stays in slot space
    assert out.causal_params.shape == (2, N, 16)
    assert out.path_matrix.shape == (2, 2 * N, 2 * N)


def test_gt_states_tracked_slot_parameter_contract() -> None:
    """One competitively refined scalar per track feeds independent graph addresses."""
    model = tiny_gt_tracked_slot_model()
    states = torch.randn(2, 5, N, 4)
    out = model(states)
    assert out.causal_params.shape == (2, N, 1)
    assert isinstance(model.pooling, TrackedSlotAttentionPooling)
    assert model.predictor.param_size == 1
    assert model.predictor.node_embeddings
    assert out.prediction.shape == (2, N, 4)
    assert out.path_matrix.shape == (2, 2 * N, 2 * N)


def test_gt_states_rollout_is_autoregressive_in_state_space() -> None:
    model = tiny_gt_model().eval()
    states = torch.randn(1, 8, N, 4)
    with torch.no_grad():
        chained = model(states, context_len=4, rollout_horizon=4)
        forced = model(states, context_len=4, rollout_horizon=1)
    assert chained.prediction.shape == (4, N, 4)
    torch.testing.assert_close(chained.prediction[0], forced.prediction[0])
    assert not torch.allclose(chained.prediction[1], forced.prediction[1])


def test_gt_states_grads_reach_param_path_only_on_state_path() -> None:
    """The GT ruler must be fixed.

    No trainable module feeds anchors/targets, while the pooled Ŝ^ph still
    gets gradient from late rollout steps.
    """
    model = tiny_gt_model()
    states = torch.randn(1, 6, N, 4)
    out = model(states, context_len=4, rollout_horizon=2)
    hungarian = (out.prediction - out.target_slots).square().mean()
    hungarian.backward()  # pyright: ignore[reportUnknownMemberType]
    pool_grads = [p.grad for p in model.pooling.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in pool_grads)
    embed_grads = [p.grad for p in model.context_embed.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in embed_grads)
    assert not any(True for _ in model.target_embed.parameters())  # Identity: no params
    assert model.kinematic_head is None


def test_gt_states_teacher_forcing_anchors_every_transition_at_true_state() -> None:
    """D21: Tp=1 + gt_states — every transition's input is the raw TRUE state.

    The anchors must be states[th-1 .. L-2] verbatim (rolling ground truth,
    never a fed-back prediction), and each prediction targets the next state.
    """
    model = tiny_gt_model()
    states = torch.randn(2, 8, N, 4)  # Th=4, K=4
    out = model(states, context_len=4, rollout_horizon=1)
    expected_anchors = states[:, 3:7].reshape(2 * 4, N, 4)
    torch.testing.assert_close(out.kinematic_state, expected_anchors)
    torch.testing.assert_close(out.target_slots, states[:, 4:].reshape(2 * 4, N, 4))


def test_gt_states_dense_and_identity_references_build() -> None:
    """The τ-calibration (dense) and go/no-go (identity) variants compose with D20."""
    for model in (
        tiny_gt_model(spartan_dense=True),
        tiny_gt_model(spartan_identity=True),
    ):
        out = model(torch.randn(2, 5, N, 4))
        assert out.prediction.shape == (2, N, 4)


def test_gt_states_trains_on_bounce(tmp_path: Path) -> None:
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
    metrics = Trainer(tiny_gt_model(), dataset, config).train()
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def test_prediction_matching_auto_resolves_from_tracked_state_contract(tmp_path: Path) -> None:
    """Literal tracked states select aligned MSE and a raw dual constraint."""
    dataset = BounceDataset(num_episodes=4, clip_len=4, num_balls=N, resolution=16, seed=2)
    config = TrainConfig(steps=1, batch_size=2, input_key="states", out_dir=str(tmp_path))
    gt_trainer = Trainer(tiny_gt_model(), dataset, config)
    assert gt_trainer.prediction_matching == "aligned"
    assert gt_trainer.constraint_normalization == "raw"
    embedded_trainer = Trainer(tiny_state_model(), dataset, config)
    assert embedded_trainer.prediction_matching == "aligned"
    assert embedded_trainer.constraint_normalization == "target_variance"
    config.prediction_matching = "hungarian"
    assert Trainer(tiny_gt_model(), dataset, config).prediction_matching == "hungarian"


def test_gt_states_harness_reports_fixed_ruler() -> None:
    """target_var must be the GT data constant, not a trainable quantity."""
    dataset = BounceDataset(num_episodes=12, clip_len=4, num_balls=N, resolution=16, seed=3)
    model = tiny_gt_tracked_slot_model()
    rng_before = torch.get_rng_state().clone()
    report = evaluate_identifiability(
        model, dataset, input_key="states", batch_size=6, max_batches=2
    )
    assert torch.equal(torch.get_rng_state(), rng_before)
    states = torch.stack([dataset[i]["states"][-1] for i in range(6)])  # first eval batch targets
    gt_var = states.reshape(-1, 4).var(dim=0).mean()
    assert abs(report.diagnostics["target_var"] - gt_var.item()) < 0.05
    assert abs(report.metrics["constraint_loss"] - report.metrics["pred_loss"]) < 1e-9
    assert report.learned_coordinates.shape == (12, N)


def test_identifiability_harness_end_to_end() -> None:
    """Untrained StateJepa on bounce: harness returns bounded, finite metrics."""
    dataset = BounceDataset(num_episodes=12, clip_len=4, num_balls=N, resolution=16, seed=3)
    report = evaluate_identifiability(
        tiny_gt_tracked_slot_model(), dataset, input_key="states", batch_size=6, max_batches=2
    )
    for key in (
        "pred_loss",
        "constraint_loss",
        "shd_state",
        "shd_param_aligned",
        "mass_mcc",
        "path_density",
    ):
        assert key in report.metrics
        assert torch.isfinite(torch.tensor(report.metrics[key])), key
    # Literal GT states use Baumgartner's raw observation-space MSE as the
    # constraint (lambda_logit=0 here). The normalized value remains diagnostic.
    assert report.metrics["constraint_loss"] == report.metrics["pred_loss"]
    approx = report.metrics["pred_loss"] / report.diagnostics["target_var"]
    assert 0.5 * approx < report.diagnostics["pred_loss_normalized"] < 2.0 * approx
    assert 0 <= report.metrics["mass_mcc"] <= 1 + 1e-6
    assert report.diagnostics["num_samples"] == 12
    assert report.learned_coordinates.shape == (12, N)
    assert report.true_parameters.shape == (12, N)
    assert report.recovery_matrix.shape == (N, N)
    assert report.target_to_learned.shape == (N,)


def test_lambda_max_config_reaches_the_dual(tmp_path: Path) -> None:
    """D22: sparsity_lambda_max must actually clamp the Lagrangian."""
    import math

    dataset = BounceDataset(num_episodes=8, clip_len=4, num_balls=N, resolution=16, seed=2)
    config = TrainConfig(
        steps=1,
        batch_size=4,
        input_key="states",
        sparsity_lambda_init=10.0,
        sparsity_lambda_max=50.0,
        out_dir=str(tmp_path),
    )
    trainer = Trainer(tiny_gt_model(), dataset, config)
    assert math.isclose(trainer.lagrangian.log_lambda_max, math.log(50.0))
