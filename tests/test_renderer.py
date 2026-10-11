import pytest
import torch

from scjepa.data.renderer import SpringRenderer
from scjepa.data.types import RendererConfig


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_shape_gaussian_reference_and_fixed_appearance(dtype: torch.dtype) -> None:
    renderer = SpringRenderer()
    positions = torch.zeros(10, dtype=dtype)
    frame = renderer(positions)
    assert frame.shape == (3, 320, 64)
    assert frame.dtype == dtype
    assert frame[2].count_nonzero() == 0
    assert frame[0, 20, 32] == 1
    assert frame[1, 44, 32] == 1
    pixels = torch.arange(64, dtype=torch.float64)
    reference = torch.exp(-((pixels[:, None] - 20) ** 2 + (pixels - 32) ** 2) / 18).to(dtype)
    torch.testing.assert_close(frame[0, :64], reference, rtol=0, atol=0)
    for pair in range(1, 5):
        torch.testing.assert_close(frame[:, 64 * pair : 64 * (pair + 1)], frame[:, :64])
    assert list(renderer.parameters()) == []
    torch.testing.assert_close(renderer.render(positions), frame, rtol=0, atol=0)


def test_arbitrary_batch_axes_are_framewise_without_temporal_mixing() -> None:
    positions = torch.linspace(-0.6, 0.6, 60, dtype=torch.float64).reshape(2, 3, 10)
    renderer = SpringRenderer()
    frames = renderer(positions)
    assert frames.shape == (2, 3, 3, 320, 64)
    for episode in range(2):
        for time in range(3):
            torch.testing.assert_close(frames[episode, time], renderer(positions[episode, time]))
    modified = positions.clone()
    modified[0] += 5
    modified[1, 2] -= 8
    torch.testing.assert_close(renderer(modified)[1, :2], frames[1, :2])


def test_position_dependence_is_smooth_and_local_with_fixed_intensities() -> None:
    renderer = SpringRenderer()
    position = torch.linspace(-0.5, 0.5, 10, dtype=torch.float64)
    position.requires_grad_()
    frame = renderer(position)
    # Off-centre pixel: derivative is smooth and only its own coordinate contributes.
    gradient = torch.autograd.grad(frame[0, 20, 30], position)[0]
    assert gradient[0].abs() > 1e-4
    assert gradient[1:].count_nonzero() == 0
    epsilon = 1e-6
    delta = torch.zeros_like(position)
    delta[0] = epsilon
    finite_difference = (
        renderer(position + delta)[0, 20, 30] - renderer(position - delta)[0, 20, 30]
    ) / (2 * epsilon)
    torch.testing.assert_close(gradient[0], finite_difference, atol=1e-8, rtol=1e-7)
    changed = renderer(position + delta)
    assert not torch.equal(frame[0, :64], changed[0, :64])
    torch.testing.assert_close(frame[1:], changed[1:], rtol=0, atol=0)
    torch.testing.assert_close(frame[0, 64:], changed[0, 64:], rtol=0, atol=0)


def test_fractional_centers_are_not_rounded_or_normalized_or_clipped() -> None:
    renderer = SpringRenderer(RendererConfig(sigma=2.5, amplitude=2.0, track_y=(17.0, 45.0)))
    positions = torch.full((10,), 0.013, dtype=torch.float64)
    frame = renderer(positions)
    center = 32 + 20 * positions[0].tanh()
    expected = 2 * torch.exp(-((32 - center) ** 2) / (2 * 2.5**2))
    torch.testing.assert_close(frame[0, 17, 32], expected, atol=1e-15, rtol=1e-15)
    assert 1 < frame.max() < 2
    assert frame[0, 17, 33] != frame[0, 17, 31]


def test_float64_tanh_saturation_is_a_real_precision_limitation() -> None:
    renderer = SpringRenderer()
    position = torch.full((10,), 100.0, dtype=torch.float64)
    assert torch.equal(renderer(position), renderer(position + 1))


def test_invalid_renderer_inputs_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="last dimension 10"):
        SpringRenderer()(torch.zeros(3))
    with pytest.raises(ValueError, match="float32 or float64"):
        SpringRenderer()(torch.zeros(10, dtype=torch.int64))
    with pytest.raises(ValueError, match="finite"):
        SpringRenderer()(torch.full((10,), float("nan")))
    with pytest.raises(ValueError, match="sigma"):
        SpringRenderer(RendererConfig(sigma=0))
    with pytest.raises(ValueError, match="differ"):
        SpringRenderer(RendererConfig(track_y=(32, 32)))
