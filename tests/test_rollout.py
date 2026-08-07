"""Hybrid rollout branch (write-up §4.2, Eqs. 33-35).

The point of these tests is that the branch is genuinely AUTOREGRESSIVE — the
predictor consumes its own output for k >= 2 — and that Eq. 35's normalisation
and weights are what the code computes. A rollout that silently degenerated
into K teacher-forced steps would still train and still look healthy, so the
feedback itself needs an explicit test.
"""

import pytest
import torch

from scjepa.losses import rollout_weights, weighted_rollout_mse
from scjepa.models import StateToStateModel, build_state_to_state

N = 3
STATE_DIM = 4


def tiny_model() -> StateToStateModel:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    return build_state_to_state(
        num_slots=N,
        param_encoder_dim=16,
        param_encoder_heads=2,
        max_history=8,
        spartan_layers=1,
        spartan_embed_dim=32,
        spartan_mlp_hidden=32,
        spartan_mlp_layers=2,
    )


def episodes(batch: int = 4, length: int = 8) -> torch.Tensor:
    return torch.randn(batch, length, N, STATE_DIM)


def test_eq35_is_the_mean_over_k_2_to_K() -> None:
    """w_1 = 0 with the rest normalised makes L_roll the MEAN over k = 2..K."""
    pred = torch.randn(4, 5, N, STATE_DIM)
    target = torch.randn(4, 5, N, STATE_DIM)
    per_step = ((pred - target) ** 2).mean(dim=(0, 2, 3))
    assert weighted_rollout_mse(pred, target, rollout_weights(5)) == pytest.approx(
        float(per_step[1:].mean()), rel=1e-6
    )


def test_weights_drop_k1_and_normalise_to_mean_one() -> None:
    """w_1 = 0 always, and K⁻¹ Σ_k w_k = 1 so lambda_roll is K-invariant."""
    for horizon in (2, 5, 10):
        weights = rollout_weights(horizon)
        assert float(weights[0]) == 0.0
        assert float(weights.mean()) == pytest.approx(1.0, rel=1e-6)
        # Uniform over the retained steps: every k >= 2 carries K/(K-1).
        expected = horizon / (horizon - 1)
        assert all(float(w) == pytest.approx(expected, rel=1e-6) for w in weights[1:])


def test_horizon_of_one_is_rejected() -> None:
    """K=1 would be entirely duplicated by L_TF once w_1 = 0."""
    with pytest.raises(ValueError, match="horizon must be >= 2"):
        rollout_weights(1)


def test_weights_must_match_horizon() -> None:
    with pytest.raises(ValueError, match="one weight per horizon step"):
        weighted_rollout_mse(
            torch.zeros(2, 4, N, STATE_DIM), torch.zeros(2, 4, N, STATE_DIM), torch.ones(3)
        )


def test_rollout_is_autoregressive_not_teacher_forced() -> None:
    """Eq. 34: for k >= 2 the input is the PREVIOUS PREDICTION.

    Run in eval mode, where gates are the deterministic Eq. 34 thresholds, so
    the chain can be reproduced by hand.
    """
    model = tiny_model()
    model.eval()
    states = episodes(batch=4, length=8)
    tpar, horizon = 3, 2
    with torch.no_grad():
        out = model(states, context_len=tpar, rollout_len=horizon)
        # The chain is anchored at the fixed t = Tpar-1 (no sampling).
        anchor = states[:, tpar - 1]
        step1 = model.predictor(anchor, out.causal_params).prediction
        step2 = model.predictor(step1, out.causal_params).prediction

    assert out.rollout_prediction is not None
    torch.testing.assert_close(out.rollout_prediction[:, 0], step1)
    torch.testing.assert_close(out.rollout_prediction[:, 1], step2)
    # And the k=2 prediction must NOT be what teacher forcing would produce.
    teacher_forced = model.predictor(states[:, tpar], out.causal_params).prediction
    assert not torch.allclose(out.rollout_prediction[:, 1], teacher_forced)


def test_rollout_targets_are_t_plus_1_through_t_plus_K() -> None:
    """target[:, k-1] is S̄_{t+k} with t = Tpar-1 fixed."""
    model = tiny_model()
    model.eval()
    states = episodes(batch=4, length=8)
    tpar, horizon = 3, 2
    with torch.no_grad():
        out = model(states, context_len=tpar, rollout_len=horizon)
    assert out.rollout_target is not None
    for k in range(1, horizon + 1):
        torch.testing.assert_close(out.rollout_target[:, k - 1], states[:, tpar - 1 + k])


def test_multiple_true_anchored_windows_share_one_parameter_encoding() -> None:
    """Relative starts align values/targets and do not re-encode θ̂ per window."""
    model = tiny_model()
    model.eval()
    states = episodes(batch=2, length=9)
    tpar, horizon = 3, 2
    starts = (0, 2, 4)
    encoder_calls: list[None] = []
    handle = model.parameter_encoder.register_forward_hook(
        lambda _module, _inputs, _output: encoder_calls.append(None)
    )
    try:
        with torch.no_grad():
            out = model(
                states,
                context_len=tpar,
                rollout_len=horizon,
                rollout_starts=starts,
            )
    finally:
        handle.remove()

    assert len(encoder_calls) == 1
    assert out.rollout_prediction is not None
    assert out.rollout_target is not None
    assert out.rollout_prediction.shape == (2, 3, horizon, N, STATE_DIM)
    assert out.rollout_target.shape == out.rollout_prediction.shape

    expected_windows: list[torch.Tensor] = []
    with torch.no_grad():
        for start in starts:
            state = states[:, tpar - 1 + start]
            predictions: list[torch.Tensor] = []
            for _ in range(horizon):
                state = model.predictor(state, out.causal_params).prediction
                predictions.append(state)
            expected_windows.append(torch.stack(predictions, dim=1))
    torch.testing.assert_close(out.rollout_prediction, torch.stack(expected_windows, dim=1))
    for window, start in enumerate(starts):
        torch.testing.assert_close(
            out.rollout_target[:, window],
            states[:, tpar + start : tpar + start + horizon],
        )


def test_explicit_single_start_retains_the_window_axis() -> None:
    """Explicit starts have one stable (B, W, H, ...) interface even when W=1."""
    model = tiny_model()
    states = episodes()
    out = model(states, context_len=3, rollout_len=2, rollout_starts=(0,))
    assert out.rollout_prediction is not None
    assert out.rollout_target is not None
    assert out.rollout_prediction.shape == (4, 1, 2, N, STATE_DIM)
    assert out.rollout_target.shape == out.rollout_prediction.shape


def test_targets_are_identical_in_train_and_eval_mode() -> None:
    """τ is calibrated on the eval constraint, so the anchor must not move."""
    model = tiny_model()
    states = episodes(batch=8, length=10)
    model.eval()
    with torch.no_grad():
        evaluated = model(states, context_len=3, rollout_len=4)
    model.train()
    with torch.no_grad():
        trained = model(states, context_len=3, rollout_len=4)
    assert evaluated.rollout_target is not None
    assert trained.rollout_target is not None
    torch.testing.assert_close(evaluated.rollout_target, trained.rollout_target)


def test_rollout_start_range_respects_context_and_episode_end() -> None:
    """The horizon must fit: t = Tpar-1 and t+K <= T-1."""
    model = tiny_model()
    states = episodes(batch=4, length=8)
    # T=8, Tpar=3: the chain starts at t=2 and its last target is S̄_7 = S̄_{T-1}.
    out = model(states, context_len=3, rollout_len=5)
    assert out.rollout_prediction is not None
    assert out.rollout_prediction.shape == (4, 5, N, STATE_DIM)
    # One more step would need S̄_8, which is past the end of the episode.
    with pytest.raises(ValueError, match="runs past the episode"):
        model(states, context_len=3, rollout_len=6)

    # The same bound includes a local window's offset from the legacy anchor.
    with pytest.raises(ValueError, match="runs past the episode"):
        model(states, context_len=3, rollout_len=3, rollout_starts=(0, 3))


def test_rollout_disabled_reproduces_teacher_forced_output() -> None:
    """rollout_len=None must leave the original objective bit-identical."""
    model = tiny_model()
    model.eval()
    states = episodes()
    with torch.no_grad():
        without = model(states, context_len=3)
        with_rollout = model(states, context_len=3, rollout_len=2)
    assert without.rollout_prediction is None
    torch.testing.assert_close(without.prediction, with_rollout.prediction)
    torch.testing.assert_close(without.sparsity, with_rollout.sparsity)


def test_gradient_cuts_do_not_change_continuous_rollout_values() -> None:
    """Detach truncates backward only: it never resets the predicted state."""
    model = tiny_model()
    model.eval()
    states = episodes(batch=2, length=9)
    without_cuts = model(states, context_len=3, rollout_len=5)
    with_cuts = model(
        states,
        context_len=3,
        rollout_len=5,
        rollout_gradient_cuts=(2, 4),
    )
    assert without_cuts.rollout_prediction is not None
    assert without_cuts.rollout_target is not None
    assert with_cuts.rollout_prediction is not None
    assert with_cuts.rollout_target is not None
    torch.testing.assert_close(with_cuts.rollout_prediction, without_cuts.rollout_prediction)
    torch.testing.assert_close(with_cuts.rollout_target, without_cuts.rollout_target)


def test_gradient_cut_stops_later_losses_but_not_the_boundary_loss() -> None:
    """A cut after step 2 blocks L4 -> z0 while the saved z2 still trains z0."""
    model = tiny_model()
    model.eval()
    base_states = episodes(batch=2, length=8)
    base_params = torch.randn(2, N, 1)
    tpar, horizon = 3, 4

    full_states = base_states.detach().clone().requires_grad_(True)
    full_params = base_params.detach().clone().requires_grad_(True)
    full_prediction, _ = model._rollout(  # pyright: ignore[reportPrivateUsage]
        full_states, tpar, horizon, full_params
    )
    full_state_grad, full_param_grad = torch.autograd.grad(
        full_prediction[:, -1].square().sum(), (full_states, full_params)
    )
    assert float(full_state_grad[:, tpar - 1].abs().sum()) > 0
    assert float(full_param_grad.abs().sum()) > 0

    cut_states = base_states.detach().clone().requires_grad_(True)
    cut_params = base_params.detach().clone().requires_grad_(True)
    cut_prediction, _ = model._rollout(  # pyright: ignore[reportPrivateUsage]
        cut_states,
        tpar,
        horizon,
        cut_params,
        gradient_cuts=(2,),
    )
    # The step-2 prediction was saved before detach, so its own loss still has
    # the complete two-transition path into the true anchor.
    boundary_state_grad = torch.autograd.grad(
        cut_prediction[:, 1].square().sum(), cut_states, retain_graph=True
    )[0]
    assert float(boundary_state_grad[:, tpar - 1].abs().sum()) > 0

    cut_state_grad, cut_param_grad = torch.autograd.grad(
        cut_prediction[:, -1].square().sum(),
        (cut_states, cut_params),
        allow_unused=True,
    )
    assert cut_state_grad is None or float(cut_state_grad.abs().sum()) == 0.0
    assert cut_param_grad is not None
    assert float(cut_param_grad.abs().sum()) > 0


def test_rollout_gradients_reach_the_parameter_encoder_across_cuts() -> None:
    """θ̂ stays attached and receives direct gradients from every backward chunk."""
    model = tiny_model()
    states = episodes()
    out = model(
        states,
        context_len=3,
        rollout_len=3,
        rollout_gradient_cuts=(1, 2),
    )
    assert out.rollout_prediction is not None
    assert out.rollout_target is not None
    loss = weighted_rollout_mse(out.rollout_prediction, out.rollout_target, rollout_weights(3))
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    grads = [p.grad for p in model.parameter_encoder.parameters() if p.grad is not None]
    assert grads, "no gradient reached the parameter encoder"
    assert any(float(g.abs().sum()) > 0 for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
