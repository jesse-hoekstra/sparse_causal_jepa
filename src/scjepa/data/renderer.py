"""Fixed smooth, framewise Gaussian rendering for Section 10.4 spring positions.

There is no per-frame normalization, clipping, rounding, quantization, resizing,
or dependence on masses/actions. Calculations use float64 internally, then return
the input dtype. Even float64 ``tanh`` saturates for sufficiently large positions;
underflow and the output cast can also erase distinctions. Finite precision is
therefore not a globally injective implementation of the ideal mathematical map.
"""

import torch
from torch import Tensor, nn

from scjepa.data.types import RendererConfig


class SpringRenderer(nn.Module):
    """Map positions ``(...,10)`` to native RGB frames ``(...,3,320,64)``.

    Five 64-by-64 panels are stacked vertically. The first mass in each pair is
    red and the second green; blue stays zero. Pixel coordinates are ``0..63``.
    The paper fixes panel geometry and ``rho(z)=32+20*tanh(z)``. Gaussian sigma,
    amplitude, and track heights are explicit implementation choices in config.
    """

    def __init__(self, config: RendererConfig | None = None) -> None:
        """Build a parameter-free renderer with reproducible fixed appearance."""
        super().__init__()
        self.config = RendererConfig() if config is None else config

    def forward(self, states: Tensor) -> Tensor:
        """Render each frame independently, preserving arbitrary leading axes."""
        if states.ndim < 1 or states.shape[-1] != 10:
            raise ValueError("renderer expects position vectors with last dimension 10")
        if states.dtype not in (torch.float32, torch.float64):
            raise ValueError("renderer positions must use float32 or float64")
        if not bool(torch.isfinite(states).all()):
            raise ValueError("renderer positions must be finite")
        coordinates = torch.arange(64, dtype=torch.float64, device=states.device)
        positions = states.to(torch.float64).reshape(*states.shape[:-1], 5, 2)
        centers = 32.0 + 20.0 * positions.tanh()
        track_y = torch.tensor(self.config.track_y, dtype=torch.float64, device=states.device)
        horizontal = (coordinates - centers[..., None]).square()
        vertical = (coordinates - track_y[:, None]).square()
        squared_distance = vertical[:, :, None] + horizontal[..., None, :]
        blobs = self.config.amplitude * torch.exp(-squared_distance / (2.0 * self.config.sigma**2))
        # (...,pair,colour,y,x) -> (...,colour,pair*y,x); one colour per mass.
        panels = torch.cat((blobs, torch.zeros_like(blobs[..., :1, :, :])), dim=-3)
        return panels.transpose(-4, -3).flatten(-3, -2).to(dtype=states.dtype)

    def render(self, states: Tensor) -> Tensor:
        """Render positions through the same fixed framewise observation map."""
        return self.forward(states)


__all__ = ["SpringRenderer"]
