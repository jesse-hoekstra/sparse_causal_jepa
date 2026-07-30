"""Visual-to-visual regime: frames in, EMA-encoded next frame out (experiments.pdf §6.6)."""

import copy

import pytest
import torch

from scjepa.data.bounce import BounceDataset
from scjepa.eval.visual_alignment import physical_assignment, slot_centroids
from scjepa.eval.visual_to_visual import evaluate_visual_to_visual
from scjepa.models.visual_to_visual import build_visual_to_visual
from scjepa.training.loop import TrainConfig
from scjepa.training.visual_to_visual import VisualToVisualTrainer, collapse_metrics


def _model(**overrides: object):  # noqa: ANN202 - tiny CPU fixture
    kwargs = dict(
        num_slots=5,
        slot_size=8,
        state_dim=8,
        param_encoder_dim=8,
        param_encoder_heads=2,
        max_history=8,
        spartan_embed_dim=16,
        spartan_mlp_hidden=16,
    )
    return build_visual_to_visual(**{**kwargs, **overrides})  # pyright: ignore[reportArgumentType]


def _dataset(episodes: int = 4, clip_len: int = 6) -> BounceDataset:
    return BounceDataset(
        num_episodes=episodes,
        clip_len=clip_len,
        num_balls=5,
        seed=3,
        render=True,
        resolution=64,
        mass_normal=(1.5, 0.5),
        radius_from_mass=True,
        speed=0.7,
        radius=0.08,
        render_radius_from_mass=False,
        uniform_appearance=True,
    )


def test_target_is_initialized_from_the_online_path() -> None:
    """Eq. 110: shared row ancestry is what makes matching unnecessary."""
    model = _model()
    for online, target in zip(model.online.parameters(), model.target.parameters(), strict=True):
        assert torch.equal(online, target)


def test_target_receives_no_gradient() -> None:
    """§6.6: the target is updated exclusively through Eq. 111."""
    model = _model()
    out = model(torch.rand(2, 5, 3, 64, 64), context_len=3)
    (out.prediction - out.target).square().mean().backward()
    assert all(p.grad is None for p in model.target.parameters())
    assert any(p.grad is not None for p in model.online.parameters())


def test_ema_update_moves_target_towards_online() -> None:
    """Eq. 111 is a convex combination, so the gap must shrink monotonically."""
    model = _model(ema_decay=0.5)
    with torch.no_grad():
        for parameter in model.online.parameters():
            parameter.add_(torch.ones_like(parameter))
    before = [t.clone() for t in model.target.parameters()]
    model.update_target()
    for old, new, online in zip(
        before, model.target.parameters(), model.online.parameters(), strict=True
    ):
        assert float((new - online).abs().sum()) < float((old - online).abs().sum())


def test_only_the_state_path_has_an_ema_copy() -> None:
    """§6.6: SPARTAN, P_eta, the keys and the gates are predictor-side, no EMA copy.

    Note SAVi's own recurrent slot predictor (Eq. 61) IS duplicated — it belongs
    to the visual state path chi. What must not be duplicated is the SPARTAN
    transition predictor and the parameter encoder, which sit beside the two
    branches rather than inside them.
    """
    model = _model()
    names = {name for name, _ in model.named_parameters()}
    online = {name.removeprefix("online.") for name in names if name.startswith("online.")}
    target = {name.removeprefix("target.") for name in names if name.startswith("target.")}
    assert online  # the state path exists...
    assert online == target  # ...and is duplicated exactly
    duplicated = ("target.predictor", "target.parameter_encoder")
    assert not any(name.startswith(duplicated) for name in names)
    assert sum(name.startswith("predictor.") for name in names) > 0
    assert sum(name.startswith("parameter_encoder.") for name in names) > 0


def test_parameter_encoder_sees_only_the_parameter_window() -> None:
    """Frames after Tpar-1 must not affect theta-hat (a §6.6 continuation gate)."""
    model = _model().eval()
    frames = torch.rand(2, 6, 3, 64, 64)
    baseline = model(frames, context_len=3).causal_params
    perturbed = frames.clone()
    perturbed[:, 4:] = torch.rand_like(perturbed[:, 4:])
    assert torch.allclose(baseline, model(perturbed, context_len=3).causal_params, atol=1e-6)


def test_decodes_into_the_learned_state_width() -> None:
    """Eq. 118: this regime's head outputs d_s, not the raw 4."""
    model = _model(state_dim=8)
    out = model(torch.rand(2, 5, 3, 64, 64), context_len=3)
    assert out.prediction.shape[-1] == 8
    assert out.target.shape == out.prediction.shape


def test_track_keys_are_permuted_per_episode() -> None:
    """§6.4: keys come from a fixed codebook under an episode-level permutation."""
    model = _model()
    keys = model.predictor.sample_track_keys(64)
    codebook = model.predictor.track_keys[0]
    # Every episode's keys are a permutation of the same codebook rows...
    for episode in keys:
        sums = sorted(float(row.sum()) for row in episode)
        assert sums == pytest.approx(sorted(float(row.sum()) for row in codebook), abs=1e-5)
    # ...and the assignment is not frozen to the identity across episodes.
    assert not all(torch.equal(episode, codebook) for episode in keys)


def test_constraint_is_variance_normalized() -> None:
    """Eq. 123 divides by the target variance; Eq. 121's gradient objective does not."""
    model = _model()
    config = TrainConfig(steps=1, batch_size=2, context_len=3, rollout_len=3, lambda_logit=0.0)
    trainer = VisualToVisualTrainer(model, _dataset(), config, eval_dataset=None)
    output = model(torch.rand(2, 5, 3, 64, 64), context_len=3)
    pred = torch.tensor(0.5)
    constraint = trainer._constraint(pred, torch.tensor(0.0), output)
    expected = 0.5 / max(float(output.target_variance), model.variance_floor)
    assert float(constraint) == pytest.approx(expected, rel=1e-5)


def test_variance_floor_bounds_the_constraint() -> None:
    """A collapsing target must not make the constraint look satisfiable."""
    model = _model(variance_floor=1e-2)
    config = TrainConfig(steps=1, batch_size=2, context_len=3, rollout_len=3)
    trainer = VisualToVisualTrainer(model, _dataset(), config, eval_dataset=None)
    output = model(torch.rand(2, 5, 3, 64, 64), context_len=3)._replace(
        target_variance=torch.tensor(0.0)
    )
    constraint = trainer._constraint(torch.tensor(1.0), torch.tensor(0.0), output)
    assert float(constraint) == pytest.approx(100.0, rel=1e-5)


def test_training_step_runs_and_steps_the_ema() -> None:
    """One end-to-end step: frames in, EMA advanced, collapse metrics logged."""
    model = _model()
    config = TrainConfig(
        steps=1, batch_size=2, context_len=3, rollout_len=3, device="cpu", out_dir="/tmp/e3"
    )
    trainer = VisualToVisualTrainer(model, _dataset(), config, eval_dataset=None)
    before = copy.deepcopy([t.clone() for t in model.target.parameters()])
    metrics = trainer._train_step(next(iter(trainer._epoch_loader(0))))
    assert "collapse/online/effective_rank" in metrics
    assert "collapse/target/content_var" in metrics
    assert metrics["health/skipped_steps"] == 0.0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.target.parameters(), strict=True)
    )


def test_collapse_metrics_detect_a_constant_representation() -> None:
    """The Eq. 124 diagnostics must bottom out exactly when the model collapses."""
    constant = torch.ones(2, 4, 5, 8)
    varied = torch.randn(2, 4, 5, 8)
    collapsed = collapse_metrics(constant, "x")
    healthy = collapse_metrics(varied, "x")
    assert collapsed["x/content_var"] == pytest.approx(0.0, abs=1e-9)
    assert collapsed["x/effective_rank"] == pytest.approx(1.0, abs=1e-6)
    assert healthy["x/content_var"] > 0.1
    assert healthy["x/effective_rank"] > 2.0


def test_evaluation_reports_the_shared_metric_keys() -> None:
    """This regime must land on the same mcc/shd axis as state-to-state."""
    report = evaluate_visual_to_visual(
        _model(), _dataset(episodes=4), batch_size=2, max_batches=2, context_len=3
    )
    for key in ("pred_loss", "constraint_loss", "path_density", "shd", "mcc"):
        assert key in report.metrics
    assert 0.0 <= report.metrics["mcc"] <= 1.0
    assert report.learned_params.shape == report.true_masses.shape


def test_geometric_alignment_is_blind_to_parameters_and_masses() -> None:
    """Eqs. 137-138 match on trajectory geometry alone."""
    dataset = _dataset(episodes=3, clip_len=5)
    centres = torch.stack([dataset[i]["states"][:, :, :2] for i in range(3)])
    permutation = torch.stack([torch.randperm(5) for _ in range(3)])
    shuffled = torch.gather(
        centres, 2, permutation[:, None, :, None].expand(3, centres.shape[1], 5, 2)
    )
    assert torch.equal(physical_assignment(shuffled, centres), permutation)


def test_slot_centroids_match_the_renderer_axis_convention() -> None:
    """A point mass at (row, col) must return (x from col, y from row)."""
    allocation = torch.zeros(1, 1, 1, 64 * 64)
    allocation[0, 0, 0, 10 * 64 + 20] = 1.0
    centroid = slot_centroids(allocation, 64)[0, 0, 0]
    assert float(centroid[0]) == pytest.approx((20 + 0.5) / 64)
    assert float(centroid[1]) == pytest.approx((10 + 0.5) / 64)


def test_rollout_is_autoregressive_and_anchored_on_the_online_path() -> None:
    """Eqs. 33-34 in the learned latent space.

    The anchor must be the ONLINE state at Tpar-1 and each later step must
    consume the previous PREDICTION. A rollout that silently re-read the online
    path every step would be K teacher-forced predictions wearing a chain's
    clothing, and would train and log identically.
    """
    torch.manual_seed(0)
    model = _model()
    model.eval()
    frames = _dataset()[0]["frames"].unsqueeze(0).repeat(2, 1, 1, 1, 1)
    tpar, horizon = 3, 3
    with torch.no_grad():
        out = model(frames, context_len=tpar, rollout_len=horizon)
        anchor = out.context_states[:, tpar - 1]
        assert out.rollout_prediction is not None
        torch.testing.assert_close(out.rollout_prediction[:, 0].shape, anchor.shape)
        # k=2 must differ from what a fresh online encoding would have produced.
        teacher_forced_k2 = out.context_states[:, tpar]
        assert not torch.allclose(out.rollout_prediction[:, 1], teacher_forced_k2)


def test_rollout_targets_come_from_the_ema_branch() -> None:
    """Eq. 114: targets are the EMA path's states, detached, at t+1..t+K."""
    torch.manual_seed(0)
    model = _model()
    model.eval()
    frames = _dataset()[0]["frames"].unsqueeze(0).repeat(2, 1, 1, 1, 1)
    tpar, horizon = 3, 3
    with torch.no_grad():
        out = model(frames, context_len=tpar, rollout_len=horizon)
    assert out.rollout_target is not None
    assert not out.rollout_target.requires_grad
    for k in range(1, horizon + 1):
        torch.testing.assert_close(out.rollout_target[:, k - 1], out.target_states[:, tpar - 1 + k])


def test_temporal_var_catches_a_time_frozen_representation() -> None:
    """The mode content_var and effective_rank are jointly blind to.

    A representation constant in time but varying across episodes keeps the
    pooled statistics healthy while making every prediction trivially
    satisfiable by the identity map.
    """
    torch.manual_seed(0)
    episodes, time, tracks, dim = 8, 6, 3, 4
    varied = torch.randn(episodes, time, tracks, dim)
    frozen = torch.randn(episodes, 1, tracks, dim).expand(-1, time, -1, -1).contiguous()

    healthy = collapse_metrics(varied, "x")
    degenerate = collapse_metrics(frozen, "x")

    # The pooled statistics cannot tell these apart...
    assert degenerate["x/content_var"] > 0.5 * healthy["x/content_var"]
    assert degenerate["x/effective_rank"] > 0.5 * healthy["x/effective_rank"]
    # ...but the temporal one does, unambiguously.
    assert degenerate["x/temporal_var"] == pytest.approx(0.0, abs=1e-9)
    assert healthy["x/temporal_var"] > 0.1
