"""Invariant tests for the channel split (D4): AttnPooling + KinematicHead."""

import pytest
import torch

from scjepa.models import AttnPooling, CrossSlotAttnPooling, KinematicHead

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
    from scjepa.models.channel_split import build_pooling

    assert isinstance(build_pooling("cross_slot", D, 2, 8), CrossSlotAttnPooling)
    assert isinstance(build_pooling("per_slot", D, 2, 8), AttnPooling)
    with pytest.raises(ValueError, match="pooling_type"):
        build_pooling("nope", D, 2, 8)
