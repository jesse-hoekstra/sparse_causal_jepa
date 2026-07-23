"""Invariant tests for the channel split (D4): AttnPooling + KinematicHead."""

import pytest
import torch

from scjepa.models import (
    AttnPooling,
    CrossSlotAttnPooling,
    KinematicHead,
    TrackAwareAttnPooling,
    TrackedSlotAttentionPooling,
)
from scjepa.models.channel_split import build_pooling

B, T, N, D = 2, 5, 4, 16


@pytest.fixture
def pooling() -> AttnPooling:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    return AttnPooling(slot_size=D, num_heads=2, max_history=8)


@pytest.fixture
def history() -> torch.Tensor:
    torch.manual_seed(1)  # pyright: ignore[reportUnknownMemberType]
    return torch.randn(B, T, N, D)


def test_pooling_shape(pooling: AttnPooling, history: torch.Tensor) -> None:
    """D4: (B, Th, N, d) → (B, N, d) — the time axis collapses inside the MHA."""
    out = pooling(history)
    assert out.shape == (B, N, D)
    assert torch.isfinite(out).all()


def test_attention_normalized_over_time(pooling: AttnPooling, history: torch.Tensor) -> None:
    attn = pooling.attention_over_time(history)
    assert attn.shape == (B, N, T)
    assert (attn >= 0).all()
    torch.testing.assert_close(attn.sum(dim=-1), torch.ones(B, N))


def test_slot_locality(pooling: AttnPooling, history: torch.Tensor) -> None:
    """Perturbing slot j's history must never change ŝ^ph_i for i != j."""
    pooling.eval()
    j = 1
    perturbed = history.clone()
    perturbed[:, :, j] += 1.0
    with torch.no_grad():
        base = pooling(history)
        after = pooling(perturbed)
    others = [i for i in range(N) if i != j]
    torch.testing.assert_close(base[:, others], after[:, others])
    assert not torch.allclose(base[:, j], after[:, j])


def test_slot_permutation_equivariance(pooling: AttnPooling, history: torch.Tensor) -> None:
    """Shared weights across slots: permuting slots permutes outputs identically."""
    pooling.eval()
    perm = torch.randperm(N)
    with torch.no_grad():
        out_perm = pooling(history[:, :, perm])
        perm_out = pooling(history)[:, perm]
    torch.testing.assert_close(out_perm, perm_out)


def test_pooling_is_time_order_aware(pooling: AttnPooling, history: torch.Tensor) -> None:
    """Temporal PE must make the pooling sensitive to frame order (D4 rationale)."""
    pooling.eval()
    with torch.no_grad():
        forward_order = pooling(history)
        reversed_order = pooling(history.flip(dims=[1]))
    assert not torch.allclose(forward_order, reversed_order)


def test_pooling_gradients(pooling: AttnPooling, history: torch.Tensor) -> None:
    pooling(history).square().mean().backward()
    for name, param in pooling.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
    assert pooling.query.grad is not None
    assert pooling.query.grad.abs().sum() > 0


def test_pooling_rejects_bad_input(pooling: AttnPooling) -> None:
    with pytest.raises(ValueError, match="expected"):
        pooling(torch.randn(B, T, D))
    with pytest.raises(ValueError, match="max_history"):
        pooling(torch.randn(B, 9, N, D))  # 9 > max_history=8
    with pytest.raises(ValueError, match="must divide"):
        AttnPooling(slot_size=15, num_heads=4)


def test_kinematic_head_shape_and_projection() -> None:
    torch.manual_seed(2)  # pyright: ignore[reportUnknownMemberType]
    head = KinematicHead(slot_size=D, state_size=8)
    out = head(torch.randn(B, T, N, D))
    assert out.shape == (B, N, 8)
    default_head = KinematicHead(slot_size=D)
    assert default_head(torch.randn(B, T, N, D)).shape == (B, N, D)


def test_kinematic_head_uses_only_last_step() -> None:
    torch.manual_seed(3)  # pyright: ignore[reportUnknownMemberType]
    head = KinematicHead(slot_size=D)
    head.eval()
    history = torch.randn(B, T, N, D)
    perturbed = history.clone()
    perturbed[:, :-1] += 1.0  # change everything EXCEPT the last step
    with torch.no_grad():
        torch.testing.assert_close(head(history), head(perturbed))
    last_changed = history.clone()
    last_changed[:, -1] += 1.0
    with torch.no_grad():
        assert not torch.allclose(head(history), head(last_changed))


@pytest.fixture
def cross_pooling() -> CrossSlotAttnPooling:
    torch.manual_seed(5)  # pyright: ignore[reportUnknownMemberType]
    return CrossSlotAttnPooling(slot_size=D, num_heads=2, max_history=8)


def test_cross_slot_shape_and_grads(
    cross_pooling: CrossSlotAttnPooling, history: torch.Tensor
) -> None:
    out = cross_pooling(history)
    assert out.shape == (B, N, D)
    assert torch.isfinite(out).all()
    out.square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
    for name, param in cross_pooling.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"


def test_cross_slot_information_flow(
    cross_pooling: CrossSlotAttnPooling, history: torch.Tensor
) -> None:
    """D14 raison d'être: perturbing slot j's history CAN change ŝ^ph_i, i != j."""
    cross_pooling.eval()
    j = 1
    perturbed = history.clone()
    perturbed[:, :, j] += 1.0
    with torch.no_grad():
        base = cross_pooling(history)
        after = cross_pooling(perturbed)
    others = [i for i in range(N) if i != j]
    assert not torch.allclose(base[:, others], after[:, others])


def test_cross_slot_permutation_equivariance(
    cross_pooling: CrossSlotAttnPooling, history: torch.Tensor
) -> None:
    """Anchored queries + slot-order-free key set ⇒ outputs permute with slots."""
    cross_pooling.eval()
    perm = torch.randperm(N)
    with torch.no_grad():
        out_perm = cross_pooling(history[:, :, perm])
        perm_out = cross_pooling(history)[:, perm]
    torch.testing.assert_close(out_perm, perm_out)


def test_cross_slot_time_order_aware(
    cross_pooling: CrossSlotAttnPooling, history: torch.Tensor
) -> None:
    cross_pooling.eval()
    with torch.no_grad():
        forward_order = cross_pooling(history)
        reversed_order = cross_pooling(history.flip(dims=[1]))
    assert not torch.allclose(forward_order, reversed_order)


def test_build_pooling_dispatch() -> None:
    assert isinstance(build_pooling("cross_slot", D, 2, 8), CrossSlotAttnPooling)
    assert isinstance(build_pooling("per_slot", D, 2, 8), AttnPooling)
    track_aware = build_pooling("track_aware", D, 2, 8, param_dim=1)
    assert isinstance(track_aware, TrackAwareAttnPooling)
    assert track_aware.param_dim == 1
    tracked_slots = build_pooling(
        "tracked_slot_attention",
        D,
        2,
        8,
        param_dim=1,
        num_slots=N,
        num_iterations=2,
    )
    assert isinstance(tracked_slots, TrackedSlotAttentionPooling)
    assert tracked_slots.num_slots == N
    assert tracked_slots.num_iterations == 2
    with pytest.raises(ValueError, match="pooling_type"):
        build_pooling("nope", D, 2, 8)


@pytest.fixture
def track_aware_pooling() -> TrackAwareAttnPooling:
    torch.manual_seed(7)  # pyright: ignore[reportUnknownMemberType]
    return TrackAwareAttnPooling(
        slot_size=D,
        num_heads=2,
        max_history=8,
        param_dim=1,
    )


def test_track_aware_scalar_shape_and_gradients(
    track_aware_pooling: TrackAwareAttnPooling, history: torch.Tensor
) -> None:
    out = track_aware_pooling(history)
    assert out.shape == (B, N, 1)
    assert torch.isfinite(out).all()
    out.square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
    for name, param in track_aware_pooling.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"


def test_track_aware_slot_permutation_equivariance(
    track_aware_pooling: TrackAwareAttnPooling, history: torch.Tensor
) -> None:
    """A joint object permutation must produce the same output permutation."""
    track_aware_pooling.eval()
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        expected = track_aware_pooling(history)[:, perm]
        actual = track_aware_pooling(history[:, :, perm])
    torch.testing.assert_close(actual, expected)


def test_track_aware_mixes_evidence_across_objects(
    track_aware_pooling: TrackAwareAttnPooling, history: torch.Tensor
) -> None:
    """Changing one trajectory can inform another object's parameter estimate."""
    track_aware_pooling.eval()
    changed_track = history.clone()
    changed_track[:, :, 1] += 5.0
    with torch.no_grad():
        original = track_aware_pooling(history)
        changed = track_aware_pooling(changed_track)
    other_slots = torch.tensor([0, 2, 3])
    assert not torch.allclose(original[:, other_slots], changed[:, other_slots])


def test_track_aware_retains_tracks_across_time(
    track_aware_pooling: TrackAwareAttnPooling, history: torch.Tensor
) -> None:
    """A one-frame slot shuffle changes trajectories and therefore parameters.

    Flattening ``time x slots`` into a set would be invariant to this operation;
    per-track temporal pooling must not be.
    """
    track_aware_pooling.eval()
    transient_shuffle = history.clone()
    transient_shuffle[:, 1] = transient_shuffle[:, 1, torch.tensor([1, 0, 3, 2])]
    with torch.no_grad():
        original = track_aware_pooling(history)
        shuffled = track_aware_pooling(transient_shuffle)
    assert not torch.allclose(original, shuffled)


def test_track_aware_has_no_final_state_residual_shortcut(
    track_aware_pooling: TrackAwareAttnPooling, history: torch.Tensor
) -> None:
    """The final state reaches the result only through temporal attention.

    Zeroing that attention makes the learned temporal query constant. Changing
    the last state must then have no effect; a projected-last-state residual,
    like the legacy cross-slot pooler uses, would fail this test.
    """
    track_aware_pooling.eval()
    with torch.no_grad():
        for parameter in track_aware_pooling.temporal_pool.mha.parameters():
            parameter.zero_()
        changed_last = history.clone()
        changed_last[:, -1] += 100.0
        original = track_aware_pooling(history)
        changed = track_aware_pooling(changed_last)
    torch.testing.assert_close(original, changed)


def test_track_aware_validates_param_dim() -> None:
    with pytest.raises(ValueError, match="param_dim"):
        TrackAwareAttnPooling(slot_size=D, num_heads=2, param_dim=0)
    with pytest.raises(ValueError, match="track_aware"):
        build_pooling("cross_slot", D, 2, 8, param_dim=1)


@pytest.fixture
def tracked_slot_pooling() -> TrackedSlotAttentionPooling:
    torch.manual_seed(11)  # pyright: ignore[reportUnknownMemberType]
    return TrackedSlotAttentionPooling(
        slot_size=D,
        num_slots=N,
        num_heads=2,
        max_history=8,
        param_dim=1,
        num_iterations=3,
    )


def test_tracked_slot_attention_shape_gradients_and_relational_evidence(
    tracked_slot_pooling: TrackedSlotAttentionPooling, history: torch.Tensor
) -> None:
    """Every tracked scalar can use contextualized collision-partner evidence."""
    tracked_slot_pooling.eval()
    changed_track = history.clone()
    changed_track[:, :, 1] += 5.0
    original = tracked_slot_pooling(history)
    with torch.no_grad():
        changed = tracked_slot_pooling(changed_track)
    assert original.shape == (B, N, 1)
    assert torch.isfinite(original).all()
    other_slots = torch.tensor([0, 2, 3])
    assert not torch.allclose(original[:, other_slots], changed[:, other_slots])

    original.square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
    for name, parameter in tracked_slot_pooling.named_parameters():
        assert parameter.grad is not None, f"no gradient for {name}"


def test_tracked_slot_attention_is_jointly_permutation_equivariant(
    tracked_slot_pooling: TrackedSlotAttentionPooling, history: torch.Tensor
) -> None:
    """Track identities and their parameter slots must permute together."""
    tracked_slot_pooling.eval()
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        expected = tracked_slot_pooling(history)[:, permutation]
        actual = tracked_slot_pooling(history[:, :, permutation])
    torch.testing.assert_close(actual, expected)


def test_tracked_slot_attention_competes_over_parameter_slots(
    tracked_slot_pooling: TrackedSlotAttentionPooling, history: torch.Tensor
) -> None:
    responsibilities = tracked_slot_pooling.assignment_matrix(history)
    assert responsibilities.shape == (B, N, N)
    assert (responsibilities >= 0).all()
    torch.testing.assert_close(responsibilities.sum(dim=-1), torch.ones(B, N))


def test_tracked_slot_attention_respects_time_and_track_continuation(
    tracked_slot_pooling: TrackedSlotAttentionPooling, history: torch.Tensor
) -> None:
    """Reversing time or swapping tracks transiently must change the inferred parameters."""
    tracked_slot_pooling.eval()
    transient_swap = history.clone()
    transient_swap[:, 1] = transient_swap[:, 1, torch.tensor([1, 0, 3, 2])]
    with torch.no_grad():
        original = tracked_slot_pooling(history)
        reversed_time = tracked_slot_pooling(history.flip(dims=[1]))
        broken_tracks = tracked_slot_pooling(transient_swap)
    assert not torch.allclose(original, reversed_time)
    assert not torch.allclose(original, broken_tracks)


def test_tracked_slot_attention_guards_and_removed_global_option(
    tracked_slot_pooling: TrackedSlotAttentionPooling, history: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="object slots"):
        tracked_slot_pooling(history[:, :, :-1])
    with pytest.raises(ValueError, match="requires num_slots"):
        build_pooling("tracked_slot_attention", D, 2, 8, param_dim=1)
    with pytest.raises(ValueError, match="pooling_type"):
        build_pooling("global_latent", D, 2, 8, param_dim=1, num_slots=N)
    with pytest.raises(ValueError, match="num_iterations"):
        TrackedSlotAttentionPooling(slot_size=D, num_slots=N, num_iterations=0)
    with pytest.raises(ValueError, match="param_dim"):
        TrackedSlotAttentionPooling(slot_size=D, num_slots=N, param_dim=0)
    with pytest.raises(ValueError, match="max_history"):
        tracked_slot_pooling(torch.randn(B, 9, N, D))
