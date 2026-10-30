"""Fixed invertible cubic observations for the six-state Experiment 1 DGP."""

from hashlib import sha256
from math import sqrt
from typing import cast

import torch
from torch import Tensor

DIMENSION = 6


class VectorObservationMap:
    """Fixed dense orthogonal mixing followed by ``u + 0.1*u**3``.

    A dedicated local CPU generator fixes W independently of trajectory draws.
    Supplying a persisted matrix avoids relying on QR reproducibility across
    library/platform versions. The returned matrix is a defensive copy.
    """

    def __init__(
        self, seed: int = 1729, matrix: Tensor | None = None, *, dimension: int = DIMENSION
    ) -> None:
        """Generate W once, or validate and copy a persisted orthogonal W."""
        if dimension < 1:
            raise ValueError("observation dimension must be positive")
        self.dimension = dimension
        if matrix is None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            raw = torch.randn(dimension, dimension, dtype=torch.float64, generator=generator)
            orthogonal, triangular = cast(
                tuple[Tensor, Tensor],
                torch.linalg.qr(raw),  # pyright: ignore[reportUnknownMemberType]
            )
            matrix = orthogonal * torch.where(triangular.diagonal() >= 0, 1.0, -1.0)
        if matrix.shape != (dimension, dimension) or not matrix.is_floating_point():
            raise ValueError(f"observation matrix must be floating point ({dimension},{dimension})")
        matrix = matrix.detach().to(device="cpu", dtype=torch.float64).clone()
        identity = torch.eye(dimension, dtype=torch.float64)
        if not torch.isfinite(matrix).all() or not torch.allclose(
            matrix.T @ matrix, identity, atol=1e-12, rtol=1e-12
        ):
            raise ValueError("observation matrix must be finite and orthogonal")
        self._matrix = matrix
        self.seed = seed

    @property
    def matrix(self) -> Tensor:
        """Return the actual fixed float64 CPU matrix for provenance storage."""
        return self._matrix.clone()

    @property
    def fingerprint(self) -> str:
        """Return SHA-256 over shape/dtype-tagged little-endian matrix bytes."""
        payload = (
            f"controlled-vector-W:<f8:{self.dimension},{self.dimension}:".encode("ascii")
            + self._matrix.numpy().astype("<f8", copy=False).tobytes()
        )
        return sha256(payload).hexdigest()

    def forward(self, states: Tensor) -> Tensor:
        """Observe states with the configured final dimension and fixed map."""
        states = self._coordinates(states, "states")
        mixed = states @ self._matrix.to(device=states.device).T
        return mixed + 0.1 * mixed.pow(3)

    def inverse(self, observations: Tensor) -> Tensor:
        """Invert the cubic stably using its hyperbolic closed form.

        This avoids subtraction of nearly equal Cardano roots near zero.
        The mathematical map is a diffeomorphism; finite precision still limits
        extreme inputs, including overflow in cubic observation generation.
        """
        observations = self._coordinates(observations, "observations")
        mixed = 2 * sqrt(10 / 3) * torch.sinh(torch.asinh(1.5 * sqrt(0.3) * observations) / 3)
        return mixed @ self._matrix.to(device=observations.device)

    def _coordinates(self, value: Tensor, name: str) -> Tensor:
        if value.ndim < 1 or value.shape[-1] != self.dimension or not value.is_floating_point():
            raise ValueError(f"{name} must be floating point with final dimension {self.dimension}")
        return value.to(dtype=torch.float64)
