"""Visual-to-state regime: frames in, true next physical state out (experiments.pdf §6.5)."""

import pytest
import torch

from scjepa.data.bounce import BounceDataset
from scjepa.eval.visual_to_state import evaluate_visual_to_state
from scjepa.models.visual_to_state import build_visual_to_state
from scjepa.training.loop import TrainConfig
from scjepa.training.visual_to_state import VisualToStateTrainer


def _model(**overrides: object):  # noqa: ANN202 - tiny CPU fixture
    kwargs = dict(
        state_dim=4,
        num_slots=5,
        slot_size=8,
        latent_state_dim=8,
        param_encoder_dim=8,
        param_encoder_heads=2,
        max_history=8,
        spartan_embed_dim=16,
        spartan_mlp_hidden=16,
    )
    return build_visual_to_state(**{**kwargs, **overrides})  # pyright: ignore[reportArgumentType]


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


def test_decodes_into_the_raw_state_space() -> None:
    """Eq. 95: a latent d_s input, decoded into the raw 4-dimensional target."""
    model = _model()
    frames, states = torch.rand(2, 6, 3, 64, 64), torch.randn(2, 6, 5, 4)
    out = model(frames, states, context_len=3)
    assert out.prediction.shape == (2, 3, 5, 4)
    assert out.target.shape == out.prediction.shape
    assert model.predictor.state_dim == 8  # W_S in R^{D_sp x d_s}
    assert model.predictor.output_dim == 4  # W_out into the physical space


def test_has_no_target_encoder_and_no_ema() -> None:
    """§6.5 states this verbatim; it is the whole difference from visual-to-visual."""
    model = _model()
    names = {name for name, _ in model.named_modules()}
    assert not any(name.startswith("target") for name in names)
    assert not hasattr(model, "update_target")
    assert not hasattr(model, "ema_decay")


def _seeded(model, frames, states, context_len: int = 3):  # noqa: ANN001, ANN202
    """Forward with a fixed RNG so the §6.4 key permutation is held constant.

    Track keys are resampled on every forward by design, so two passes over the
    same input legitimately differ; seeding isolates whatever the test is
    actually varying.
    """
    torch.manual_seed(1234)
    return model(frames, states, context_len=context_len)


def test_track_keys_are_resampled_every_forward() -> None:
    """§6.4: an independently sampled episode-level permutation, not a fixed one."""
    model = _model().eval()
    frames, states = torch.rand(2, 6, 3, 64, 64), torch.randn(2, 6, 5, 4)
    torch.manual_seed(0)
    first = model(frames, states, context_len=3).prediction
    second = model(frames, states, context_len=3).prediction
    assert not torch.allclose(first, second, atol=1e-7)
    # ...but seeding pins them, which is what the other tests rely on.
    assert torch.allclose(
        _seeded(model, frames, states).prediction, _seeded(model, frames, states).prediction
    )


def test_true_states_never_reach_the_encoder_or_predictor() -> None:
    """States supply targets only; perturbing them must not move a prediction."""
    model = _model().eval()
    frames = torch.rand(2, 6, 3, 64, 64)
    states = torch.randn(2, 6, 5, 4)
    baseline = _seeded(model, frames, states)
    other = _seeded(model, frames, torch.randn_like(states))
    assert torch.allclose(baseline.prediction, other.prediction, atol=1e-6)
    assert torch.allclose(baseline.causal_params, other.causal_params, atol=1e-6)


def test_parameter_encoder_sees_only_the_parameter_window() -> None:
    """Frames after Tpar-1 must leave theta-hat unchanged (§6.5 gate)."""
    model = _model().eval()
    frames, states = torch.rand(2, 6, 3, 64, 64), torch.randn(2, 6, 5, 4)
    baseline = _seeded(model, frames, states).causal_params
    perturbed = frames.clone()
    perturbed[:, 4:] = torch.rand_like(perturbed[:, 4:])
    assert torch.allclose(baseline, _seeded(model, perturbed, states).causal_params, atol=1e-6)


def test_source_state_is_causal_in_the_frames() -> None:
    """The prediction for t -> t+1 may depend only on X_{0:t}, never on X_{t+1}."""
    model = _model().eval()
    frames, states = torch.rand(2, 6, 3, 64, 64), torch.randn(2, 6, 5, 4)
    baseline = _seeded(model, frames, states).prediction
    perturbed = frames.clone()
    perturbed[:, -1] = torch.rand_like(perturbed[:, -1])  # the final target frame
    assert torch.allclose(baseline, _seeded(model, perturbed, states).prediction, atol=1e-6)


def test_loss_uses_the_trajectory_assignment_and_stays_differentiable() -> None:
    """Eqs. 99-101: detached assignment, gradients through the predictions."""
    model = _model()
    config = TrainConfig(steps=1, batch_size=2, context_len=3)
    trainer = VisualToStateTrainer(model, _dataset(), config, eval_dataset=None, scale_episodes=4)
    frames, states = torch.rand(2, 6, 3, 64, 64), torch.randn(2, 6, 5, 4)
    loss = trainer._prediction_loss(model(frames, states, context_len=3))
    assert loss.requires_grad
    loss.backward()
    assert any(p.grad is not None for p in model.visual.parameters())


def test_perfect_predictions_under_a_permutation_give_zero_loss() -> None:
    """The assignment must absorb track relabelling, and nothing more."""
    model = _model()
    config = TrainConfig(steps=1, batch_size=2, context_len=3)
    trainer = VisualToStateTrainer(model, _dataset(), config, eval_dataset=None, scale_episodes=4)
    targets = torch.randn(2, 3, 5, 4)
    permutation = torch.stack([torch.randperm(5) for _ in range(2)])
    index = permutation[:, None, :, None].expand(2, 3, 5, 4)
    output = model(torch.rand(2, 6, 3, 64, 64), torch.randn(2, 6, 5, 4), context_len=3)._replace(
        prediction=torch.gather(targets, 2, index), target=targets
    )
    assert float(trainer._prediction_loss(output)) == pytest.approx(0.0, abs=1e-10)


def test_coordinate_scales_are_frozen_into_the_model() -> None:
    """sigma_a must ride along in the checkpoint, or held-out tau is a different number."""
    model = _model()
    assert torch.equal(model.coordinate_scales, torch.ones(4))
    config = TrainConfig(steps=1, batch_size=2, context_len=3)
    VisualToStateTrainer(model, _dataset(), config, eval_dataset=None, scale_episodes=4)
    assert not torch.equal(model.coordinate_scales, torch.ones(4))
    assert bool((model.coordinate_scales > 0).all())
    assert "coordinate_scales" in model.state_dict()


def test_constraint_is_raw_not_variance_normalized() -> None:
    """Eq. 103 matches Experiment 1's form; only Experiment 3 normalizes."""
    model = _model()
    config = TrainConfig(steps=1, batch_size=2, context_len=3, lambda_logit=0.5)
    trainer = VisualToStateTrainer(model, _dataset(), config, eval_dataset=None, scale_episodes=4)
    metrics = trainer._train_step(next(iter(trainer._epoch_loader(0))))
    assert metrics["sparsity/constraint"] == pytest.approx(
        metrics["loss/pred"] + metrics["loss/logit"], rel=1e-6
    )


def test_training_step_runs() -> None:
    """One end-to-end step on real frames."""
    model = _model()
    config = TrainConfig(steps=1, batch_size=2, context_len=3, device="cpu", out_dir="/tmp/e2")
    trainer = VisualToStateTrainer(model, _dataset(), config, eval_dataset=None, scale_episodes=4)
    metrics = trainer._train_step(next(iter(trainer._epoch_loader(0))))
    assert metrics["health/skipped_steps"] == 0.0
    assert metrics["health/latent_std"] >= 0.0
    assert metrics["loss/sparsity"] > 0.0
    assert trainer.successful_updates == 1
    assert metrics["schedule/successful_updates"] == 1.0


def test_evaluation_reports_the_shared_metric_keys() -> None:
    """Experiment 2 lands on the same mcc/shd axis as Experiments 1 and 3."""
    report = evaluate_visual_to_state(
        _model(), _dataset(episodes=4), batch_size=2, max_batches=2, context_len=3
    )
    for key in ("pred_loss", "constraint_loss", "path_density", "shd", "mcc"):
        assert key in report.metrics
    assert 0.0 <= report.metrics["mcc"] <= 1.0
    assert 0.0 <= report.metrics["assignment_disagreement"] <= 1.0
    assert report.learned_params.shape == report.true_masses.shape
