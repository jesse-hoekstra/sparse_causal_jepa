"""Visual-to-visual regime: frames in, EMA-encoded next frame out (Experiment 2)."""

import copy

import pytest
import torch
from torch import Tensor, nn

from scjepa.data.bounce import BounceDataset
from scjepa.eval.visual_alignment import physical_assignment, slot_centroids
from scjepa.eval.visual_to_visual import evaluate_visual_to_visual
from scjepa.models.spartan import SpartanOutput
from scjepa.models.visual_to_visual import build_visual_to_visual, context_target_assignment
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
    """EMA starts in a shared feature basis; object tracking still needs testing."""
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
    config = TrainConfig(
        steps=1,
        batch_size=2,
        context_len=3,
        lambda_rollout_t2=0.0,
        lambda_logit=0.0,
    )
    trainer = VisualToVisualTrainer(model, _dataset(), config, eval_dataset=None)
    output = model(torch.rand(2, 5, 3, 64, 64), context_len=3)
    pred = torch.tensor(0.5)
    constraint = trainer._constraint(pred, torch.tensor(0.0), output)
    expected = 0.5 / max(float(output.target_variance), model.variance_floor)
    assert float(constraint) == pytest.approx(expected, rel=1e-5)


def test_variance_floor_bounds_the_constraint() -> None:
    """The variance floor bounds division; it is not an anti-collapse objective."""
    model = _model(variance_floor=1e-2)
    config = TrainConfig(
        steps=1,
        batch_size=2,
        context_len=3,
        lambda_rollout_t2=0.0,
    )
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
        steps=1,
        batch_size=2,
        context_len=3,
        lambda_rollout_t2=0.0,
        device="cpu",
        out_dir="/tmp/e2",
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
        _model(),
        _dataset(episodes=4),
        batch_size=2,
        max_batches=2,
        context_len=3,
        num_rollout_t2_anchors=2,
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


def test_evaluation_rollout_targets_and_guards() -> None:
    model = _model().eval()
    with torch.no_grad():
        out = model(torch.rand(2, 6, 3, 64, 64), context_len=3)
        prediction, target = model.rollout_for_evaluation(
            out.context_states, out.target_states, 3, 3, out.causal_params, out.episode_keys
        )
    assert prediction.shape == (2, 3, 5, 8)
    torch.testing.assert_close(target, out.target_states[:, 3:])
    assert not prediction.requires_grad
    with pytest.raises(RuntimeError, match="no_grad"):
        model.rollout_for_evaluation(
            out.context_states, out.target_states, 3, 3, out.causal_params, out.episode_keys
        )
    model.train()
    with torch.no_grad(), pytest.raises(RuntimeError, match="model.eval"):
        model.rollout_for_evaluation(
            out.context_states, out.target_states, 3, 3, out.causal_params, out.episode_keys
        )


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


def test_matching_recovers_one_per_episode_slot_permutation() -> None:
    torch.manual_seed(9)
    online = torch.randn(3, 4, 5, 8, requires_grad=True)
    permutations = torch.stack([torch.randperm(5) for _ in range(3)])
    target = online.detach().gather(2, permutations[:, None, :, None].expand_as(online))
    assignment = context_target_assignment(online, target)
    torch.testing.assert_close(assignment, permutations.argsort(dim=1))
    assert not assignment.requires_grad
    torch.testing.assert_close(
        target.gather(2, assignment[:, None, :, None].expand_as(target)), online
    )


def test_matching_does_not_depend_on_other_episodes_in_the_batch() -> None:
    torch.manual_seed(1)
    online = torch.randn(1, 4, 5, 8)
    target = torch.randn_like(online)
    alone = context_target_assignment(online, target)
    unrelated_online = torch.randn_like(online) * torch.logspace(-4, 6, 8)
    unrelated_target = torch.randn_like(online) * torch.logspace(-4, 6, 8)
    batched = context_target_assignment(
        torch.cat((online, unrelated_online)), torch.cat((target, unrelated_target))
    )
    torch.testing.assert_close(alone[0], batched[0])


def test_identical_ema_prefixes_remain_deterministic_during_training() -> None:
    model = _model().train()
    frames = torch.rand(2, 6, 3, 64, 64)
    out = model(frames, context_len=3)
    assert not model.target.training
    torch.testing.assert_close(out.context_states, out.target_states[:, :-1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out.target_assignment, torch.arange(5)[None].expand(2, -1))


def test_future_frames_do_not_change_assignment_parameters_or_past_states() -> None:
    model = _model().eval()
    frames = torch.rand(2, 6, 3, 64, 64)
    baseline = model(frames, context_len=3)
    changed = frames.clone()
    changed[:, 3:] = torch.rand_like(changed[:, 3:])
    altered = model(changed, context_len=3)
    torch.testing.assert_close(baseline.target_assignment, altered.target_assignment)
    torch.testing.assert_close(baseline.causal_params, altered.causal_params)
    torch.testing.assert_close(baseline.context_states[:, :3], altered.context_states[:, :3])
    torch.testing.assert_close(baseline.target_states[:, :3], altered.target_states[:, :3])


def test_allocations_are_optional_detached_and_reuse_the_encoder_pass() -> None:
    model = _model()
    frames = torch.rand(2, 5, 3, 64, 64)
    calls: list[int] = []
    handle = model.online.encoder._impl.slot_attention.register_forward_hook(
        lambda *_: calls.append(1)
    )
    try:
        output = model(frames, context_len=3, capture_allocations=True)
    finally:
        handle.remove()
    assert len(calls) == 4  # exactly one correction pass per online frame
    assert output.context_allocations is not None
    assert output.target_allocations is not None
    assert output.context_allocations.shape == (2, 4, 5, 64 * 64)
    assert output.target_allocations.shape == (2, 5, 5, 64 * 64)
    assert not output.context_allocations.requires_grad
    torch.testing.assert_close(output.context_allocations.sum(-1), torch.ones(2, 4, 5))
    assert model(frames, context_len=3).context_allocations is None


def test_t2_reuses_one_parameter_inference_and_episode_keys() -> None:
    model = _model()
    parameter_calls: list[int] = []
    predictor_calls: list[tuple[tuple[Tensor, ...], dict[str, Tensor], SpartanOutput]] = []
    encoder_hook = model.parameter_encoder.register_forward_hook(
        lambda *_: parameter_calls.append(1)
    )

    def record_call(
        _module: nn.Module,
        args: tuple[Tensor, ...],
        kwargs: dict[str, Tensor],
        output: SpartanOutput,
    ) -> None:
        predictor_calls.append((args, kwargs, output))

    predictor_hook = model.predictor.register_forward_hook(record_call, with_kwargs=True)
    try:
        out = model(torch.rand(2, 6, 3, 64, 64), context_len=3, num_rollout_t2_anchors=2)
    finally:
        encoder_hook.remove()
        predictor_hook.remove()
    assert len(parameter_calls) == 1
    assert len(predictor_calls) == 3  # TF, generated first step, generated second step
    assert out.rollout_t2_offsets is not None
    assert out.rollout_t2_offsets.shape == (2, 2)
    assert out.rollout_t2_prediction is not None
    assert out.rollout_t2_target is not None
    assert out.rollout_t2_intermediate is not None
    torch.testing.assert_close(predictor_calls[2][0][0], out.rollout_t2_intermediate.flatten(0, 1))
    for args, kwargs, _ in predictor_calls[1:]:
        torch.testing.assert_close(args[1], out.causal_params.repeat_interleave(2, dim=0))
        torch.testing.assert_close(
            kwargs["track_keys"], out.episode_keys.repeat_interleave(2, dim=0)
        )
    indices = 4 + out.rollout_t2_offsets
    torch.testing.assert_close(
        out.rollout_t2_target, out.target_states[torch.arange(2)[:, None], indices]
    )
    assert not out.rollout_t2_target.requires_grad


def test_t2_endpoint_gradient_reaches_both_predictions_and_online_anchor() -> None:
    model = _model()
    online_states = torch.randn(2, 5, 5, 8, requires_grad=True)
    target_states = torch.randn(2, 6, 5, 8, requires_grad=True)
    theta = torch.randn(2, 5, 1, requires_grad=True)
    generated: list[Tensor] = []

    def retain(_module: nn.Module, _args: tuple[Tensor, ...], output: SpartanOutput) -> None:
        output.prediction.retain_grad()
        generated.append(output.prediction)

    handle = model.predictor.register_forward_hook(retain)
    try:
        endpoint, target, _ = model.rollout_t2_from_offsets(
            online_states,
            target_states,
            3,
            theta,
            torch.tensor([[0, 1], [1, 0]]),
            model.predictor.sample_track_keys(2),
        )
        (endpoint - target).square().mean().backward()
    finally:
        handle.remove()
    assert len(generated) == 2
    assert generated[0].grad is not None
    assert float(generated[0].grad.abs().sum()) > 0
    assert generated[1].grad is not None
    assert float(generated[1].grad.abs().sum()) > 0
    assert online_states.grad is not None
    assert float(online_states.grad.abs().sum()) > 0
    assert theta.grad is not None
    assert float(theta.grad.abs().sum()) > 0
    assert target_states.grad is None


def test_t2_disabled_preserves_teacher_forcing_rng_and_calls() -> None:
    model = _model()
    frames = torch.rand(2, 6, 3, 64, 64)
    rng = torch.get_rng_state()
    baseline = model(frames, context_len=3)
    baseline_after = torch.get_rng_state()
    torch.set_rng_state(rng)
    disabled = model(frames, context_len=3, num_rollout_t2_anchors=0)
    torch.testing.assert_close(baseline.prediction, disabled.prediction)
    torch.testing.assert_close(torch.get_rng_state(), baseline_after)
    assert disabled.rollout_t2_prediction is None


def test_trainer_uses_only_frames_and_shared_t2_metric_names() -> None:
    model = _model()
    config = TrainConfig(steps=1, batch_size=2, context_len=3, num_rollout_t2_anchors=2)
    trainer = VisualToVisualTrainer(model, _dataset(), config)
    # Supplying no state, mass, or contact fields also rules out accidental
    # simulator supervision in either the model or the training objective.
    metrics = trainer._train_step({"frames": torch.rand(2, 6, 3, 64, 64)})
    assert metrics["health/skipped_steps"] == 0.0
    assert metrics["train/loss_rollout_t2_raw"] > 0
    for key in ("train/loss_teacher_forcing", "train/loss_rollout_t2_weighted", "train/loss_total"):
        assert key in metrics


def test_static_slot_identities_do_not_count_as_content_variance() -> None:
    fixed_slots = torch.randn(1, 1, 5, 8).expand(3, 4, -1, -1)
    metrics = collapse_metrics(fixed_slots, "x")
    assert metrics["x/content_var"] == 0.0
    assert metrics["x/temporal_var"] == 0.0


def test_collapse_metrics_exclude_context_initialization_transients() -> None:
    model = _model()
    trainer = VisualToVisualTrainer(
        model, _dataset(), TrainConfig(steps=1, batch_size=2, context_len=3)
    )
    out = model(torch.rand(2, 6, 3, 64, 64), context_len=3)
    online = torch.zeros_like(out.context_states)
    target = torch.zeros_like(out.target_states)
    online[:, :2] = torch.randn_like(online[:, :2])
    target[:, :3] = torch.randn_like(target[:, :3])
    metrics = trainer._extra_metrics(out._replace(context_states=online, target_states=target))
    assert metrics["collapse/online/temporal_var"] == 0.0
    assert metrics["collapse/target/temporal_var"] == 0.0
    assert metrics["collapse/online/content_var"] == 0.0
    assert metrics["collapse/target/content_var"] == 0.0


def test_fixed_context_mapping_does_not_rematch_a_future_identity_switch() -> None:
    model = _model().eval()
    permutation = torch.tensor([4, 1, 2, 3, 0])

    def swap_future(_module: nn.Module, _args: tuple[Tensor, ...], output):  # noqa: ANN001, ANN202
        slots, states = output.slots.clone(), output.states.clone()
        slots[:, 3:] = slots[:, 3:, permutation]
        states[:, 3:] = states[:, 3:, permutation]
        return output._replace(slots=slots, states=states)

    frames = torch.rand(2, 6, 3, 64, 64)
    baseline = model(frames, context_len=3)
    handle = model.target.register_forward_hook(swap_future)
    try:
        switched = model(frames, context_len=3)
    finally:
        handle.remove()
    torch.testing.assert_close(switched.target_assignment, baseline.target_assignment)
    torch.testing.assert_close(
        switched.target_states[:, 3:], baseline.target_states[:, 3:, permutation]
    )
