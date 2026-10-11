"""Small state encoders that process each observation independently.

Sequence encoding folds only the batch and time axes. All normalization is
within one observation: no batch statistics, recurrence, or temporal attention.
The output coordinates have no prescribed physical interpretation.
"""

from collections.abc import Sequence

from torch import Tensor, nn


class FramewiseEncoder(nn.Module):
    """Base interface for an observation-to-state map with a complete output head."""

    state_dim: int
    observation_ndim: int

    def encode_sequence(self, observations: Tensor) -> Tensor:
        """Map ``(B, T, *observation_shape)`` to ``(B, T, state_dim)``."""
        if observations.ndim != self.observation_ndim + 2:
            raise ValueError("observations must have shape (B, T, *observation_shape)")
        batch, length = observations.shape[:2]
        if batch < 1 or length < 1:
            raise ValueError("batch and sequence length must be nonempty")
        encoded: Tensor = self(observations.flatten(0, 1))
        return encoded.reshape(batch, length, self.state_dim)


class VectorEncoder(FramewiseEncoder):
    """A configurable MLP for vector observations; LayerNorm stays within a frame."""

    observation_ndim = 1

    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
    ) -> None:
        """Build an MLP and unconstrained state head with independent dimensions."""
        super().__init__()
        if min(observation_dim, state_dim, *hidden_dims) < 1:
            raise ValueError("all encoder dimensions must be positive")
        self.observation_dim = observation_dim
        self.state_dim = state_dim
        layers: list[nn.Module] = []
        previous = observation_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.LayerNorm(width), nn.GELU()))
            previous = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(previous, state_dim)

    def forward(self, observations: Tensor) -> Tensor:
        """Encode a batch ``(B, observation_dim)`` without other frames or inputs."""
        if observations.ndim != 2 or observations.shape[-1] != self.observation_dim:
            raise ValueError(f"expected (B, {self.observation_dim}) vector observations")
        return self.head(self.backbone(observations))


class ImageEncoder(FramewiseEncoder):
    """A compact convolutional image encoder with within-frame GroupNorm."""

    observation_ndim = 3

    def __init__(
        self,
        in_channels: int,
        state_dim: int,
        channels: Sequence[int] = (16, 32, 64),
        pool_size: int = 2,
    ) -> None:
        """Build strided convolutions, spatial pooling, and a vector state head."""
        super().__init__()
        if not channels or min(in_channels, state_dim, pool_size, *channels) < 1:
            raise ValueError("image dimensions and the nonempty channel widths must be positive")
        self.in_channels = in_channels
        self.state_dim = state_dim
        layers: list[nn.Module] = []
        previous = in_channels
        for width in channels:
            layers.extend(
                (
                    nn.Conv2d(previous, width, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(1, width),
                    nn.GELU(),
                )
            )
            previous = width
        layers.extend((nn.AdaptiveAvgPool2d((pool_size, pool_size)), nn.Flatten(1)))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(previous * pool_size * pool_size, state_dim)

    def forward(self, observations: Tensor) -> Tensor:
        """Encode ``(B, C, H, W)``; neither normalization nor pooling spans frames."""
        if observations.ndim != 4 or observations.shape[1] != self.in_channels:
            raise ValueError(f"expected (B, {self.in_channels}, H, W) image observations")
        if observations.shape[2] < 1 or observations.shape[3] < 1:
            raise ValueError("image spatial dimensions must be positive")
        return self.head(self.backbone(observations))


__all__ = ["FramewiseEncoder", "ImageEncoder", "VectorEncoder"]
