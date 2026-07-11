"""Contract tests for the SAVi encoder wrapper (shapes, grads, determinism, D7)."""

import pytest
import torch

from scjepa.models import SAViEncoder

B, T, N, D = 2, 4, 3, 16
RES = 64


@pytest.fixture
def tiny_encoder() -> SAViEncoder:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    return SAViEncoder(
        resolution=(RES, RES),
        num_slots=N,
        slot_size=D,
        slot_mlp_size=32,
        num_iterations=1,
        enc_channels=(3, 8, 8),
        enc_out_channels=16,
        pred_num_layers=1,
        pred_num_heads=2,
        pred_ffn_dim=32,
    )


@pytest.fixture
def tiny_clip() -> torch.Tensor:
    torch.manual_seed(1)  # pyright: ignore[reportUnknownMemberType]
    return torch.randn(B, T, 3, RES, RES)


def test_slot_history_shape(tiny_encoder: SAViEncoder, tiny_clip: torch.Tensor) -> None:
    slots = tiny_encoder(tiny_clip)
    assert slots.shape == (B, T, N, D)
    assert torch.isfinite(slots).all()


def test_gradients_reach_all_submodules(tiny_encoder: SAViEncoder, tiny_clip: torch.Tensor) -> None:
    """Joint training (no stop-gradient, D7): every trainable part must get grads."""
    tiny_encoder(tiny_clip).square().mean().backward()
    impl = tiny_encoder._impl  # pyright: ignore[reportPrivateUsage]
    for part in ("encoder", "slot_attention", "predictor"):
        grads = [p.grad for p in getattr(impl, part).parameters() if p.requires_grad]
        assert grads, f"{part} has no trainable parameters"
        assert any(g is not None and g.abs().sum() > 0 for g in grads), (
            f"no gradient reached {part}"
        )


def test_deterministic(tiny_encoder: SAViEncoder, tiny_clip: torch.Tensor) -> None:
    """kld_method='none' must make the encoder a deterministic function."""
    tiny_encoder.eval()
    with torch.no_grad():
        first = tiny_encoder(tiny_clip)
        second = tiny_encoder(tiny_clip)
    torch.testing.assert_close(first, second)


def test_recurrence_state_reset_between_clips(
    tiny_encoder: SAViEncoder, tiny_clip: torch.Tensor
) -> None:
    """A second clip must not inherit recurrent state from the first."""
    tiny_encoder.eval()
    other = torch.randn_like(tiny_clip)
    with torch.no_grad():
        baseline = tiny_encoder(tiny_clip)
        tiny_encoder(other)  # pollute LSTM state
        again = tiny_encoder(tiny_clip)
    torch.testing.assert_close(baseline, again)


def test_no_decoder_built(tiny_encoder: SAViEncoder) -> None:
    """Encoder-only (D7): the reconstruction decoder must not exist at all."""
    impl = tiny_encoder._impl  # pyright: ignore[reportPrivateUsage]
    assert not hasattr(impl, "decoder")
    assert not hasattr(impl, "decoder_pos_embedding")


def test_temporal_recurrence_is_causal(tiny_encoder: SAViEncoder) -> None:
    """Slots at time t must depend only on frames ≤ t."""
    tiny_encoder.eval()
    torch.manual_seed(2)  # pyright: ignore[reportUnknownMemberType]
    clip = torch.randn(1, T, 3, RES, RES)
    perturbed = clip.clone()
    perturbed[:, -1] += 1.0  # change ONLY the last frame
    with torch.no_grad():
        slots_a = tiny_encoder(clip)
        slots_b = tiny_encoder(perturbed)
    torch.testing.assert_close(slots_a[:, :-1], slots_b[:, :-1])
    assert not torch.allclose(slots_a[:, -1], slots_b[:, -1])


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="64x64 or 128x128"):
        SAViEncoder(resolution=(32, 32))
    enc = SAViEncoder(
        num_slots=N,
        slot_size=D,
        slot_mlp_size=32,
        enc_channels=(3, 8, 8),
        enc_out_channels=16,
    )
    with pytest.raises(ValueError, match="expected"):
        enc(torch.randn(2, 3, 64, 64))  # missing time dim


@pytest.fixture
def target_encoder() -> SAViEncoder:
    """D9 target-branch encoder: single frame, no slot predictor."""
    torch.manual_seed(4)  # pyright: ignore[reportUnknownMemberType]
    return SAViEncoder(
        num_slots=N,
        slot_size=D,
        slot_mlp_size=32,
        enc_channels=(3, 8, 8),
        enc_out_channels=16,
        single_frame=True,
    )


def test_single_frame_shape_and_grads(target_encoder: SAViEncoder) -> None:
    """D9: one future frame in, (B, 1, N, d) out, grads to every live part.

    ``prior_slot_layer`` is exempt: documented dead weight in the vendored code
    (upstream keeps it only for checkpoint compatibility; never in the forward
    path — see PROVENANCE.md "known upstream quirks").
    """
    frame = torch.randn(B, 1, 3, RES, RES)
    slots = target_encoder(frame)
    assert slots.shape == (B, 1, N, D)
    slots.square().mean().backward()
    for name, param in target_encoder.named_parameters():
        if not param.requires_grad or "prior_slot_layer" in name:
            continue
        assert param.grad is not None, f"no gradient for {name}"
        assert param.grad.abs().sum() > 0, f"zero gradient for {name}"


def test_single_frame_builds_no_predictor(target_encoder: SAViEncoder) -> None:
    """D9: no dead weight — the never-invoked slot predictor must not exist."""
    impl = target_encoder._impl  # pyright: ignore[reportPrivateUsage]
    assert not hasattr(impl, "predictor")


def test_single_frame_rejects_clips(target_encoder: SAViEncoder) -> None:
    with pytest.raises(ValueError, match="expects T=1"):
        target_encoder(torch.randn(B, T, 3, RES, RES))
