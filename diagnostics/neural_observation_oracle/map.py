"""Fixed smooth observation mixing; its explicit inverse is evaluation-only."""

import math
from typing import Any, cast

import torch
from torch import Tensor

from scjepa.data.datasets import canonical_fingerprint
from scjepa.data.identity import tensor_digest

DIMENSION = 6
MAP_SEED = 20261028
ALPHA = 0.2
DELTA = 0.1
GAIN = math.sqrt(2 / (1 + ALPHA * ALPHA))
LAYERS = 4
SCHEMA = "smooth-leaky-orthogonal-observation-map-v1"
MAP_KIND = "smooth_leaky_orthogonal_mlp"


def settings() -> dict[str, Any]:
    """Return the immutable map definition fixed before scientific fits."""
    return {
        "schema": SCHEMA,
        "kind": MAP_KIND,
        "seed": MAP_SEED,
        "dimension": DIMENSION,
        "orthogonal_layers": LAYERS,
        "hidden_layers": LAYERS - 1,
        "alpha": ALPHA,
        "delta": DELTA,
        "gain": GAIN,
        "composition": "Q3 h(Q2 h(Q1 h(Q0 x_old)))",
        "activation": "gain*(.6*t+.4*(sqrt(t*t+.01)-.1))",
        "qr_convention": "independent Gaussian QR with positive diagonal of R",
    }


def smooth_leaky(value: Tensor) -> Tensor:
    """Apply the fixed smooth strictly increasing activation with attached gradients."""
    radius = torch.hypot(value, value.new_tensor(DELTA))
    return GAIN * ((1 + ALPHA) * value + (1 - ALPHA) * (radius - DELTA)) / 2


def inverse_smooth_leaky(value: Tensor) -> Tensor:
    """Invert the fixed scalar map analytically; no fitted or iterative inverse."""
    a, b = (1 + ALPHA) / 2, (1 - ALPHA) / 2
    shifted = value / GAIN + b * DELTA
    radius = torch.hypot(shifted, shifted.new_tensor(math.sqrt(ALPHA) * DELTA))
    return (a * shifted - b * radius) / ALPHA


class FrozenObservationMap:
    """Compose four orthogonal maps and three smooth bijections on six coordinates.

    Preparation draws from one dedicated CPU stream. Loading and resume always
    pass the persisted matrices, avoiding cross-platform QR regeneration. This
    object is not a learned module and never enters an optimizer.
    """

    def __init__(self, matrices: Tensor | None = None) -> None:
        """Draw once or validate a defensive copy of the realized float64 weights."""
        if matrices is None:
            generator = torch.Generator(device="cpu").manual_seed(MAP_SEED)
            values: list[Tensor] = []
            for _ in range(LAYERS):
                raw = torch.randn(DIMENSION, DIMENSION, dtype=torch.float64, generator=generator)
                orthogonal, triangular = cast(
                    tuple[Tensor, Tensor],
                    torch.linalg.qr(raw),  # pyright: ignore[reportUnknownMemberType]
                )
                values.append(orthogonal * torch.where(triangular.diagonal() >= 0, 1.0, -1.0))
            matrices = torch.stack(values)
        if matrices.shape != (LAYERS, DIMENSION, DIMENSION) or not matrices.is_floating_point():
            raise ValueError("Observation network requires four floating 6-by-6 matrices")
        matrices = matrices.detach().to(device="cpu", dtype=torch.float64).clone()
        identity = torch.eye(DIMENSION, dtype=torch.float64).expand_as(matrices)
        if not torch.isfinite(matrices).all() or not torch.allclose(
            matrices.transpose(-1, -2) @ matrices, identity, atol=1e-12, rtol=1e-12
        ):
            raise ValueError("Observation network matrices must be finite and orthogonal")
        self._matrices = matrices

    @property
    def matrices(self) -> Tensor:
        """Return a defensive copy of actual weights for provenance storage."""
        return self._matrices.clone()

    @property
    def fingerprint(self) -> str:
        """Bind the exact float64 weights and every fixed numerical setting."""
        return canonical_fingerprint(
            settings() | {"matrices_sha256": tensor_digest(self._matrices)}
        )

    def payload(self) -> dict[str, Any]:
        """Serialize weights and their fixed settings without a learned state."""
        return {"settings": settings(), "matrices": self.matrices, "fingerprint": self.fingerprint}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "FrozenObservationMap":
        """Restore only the recorded matrices, rejecting relabeled network definitions."""
        if (
            set(payload) != {"settings", "matrices", "fingerprint"}
            or payload["settings"] != settings()
        ):
            raise ValueError("Frozen observation network settings changed")
        result = cls(payload["matrices"])
        if payload["fingerprint"] != result.fingerprint:
            raise ValueError("Frozen observation network weights changed")
        return result

    @staticmethod
    def _coordinates(value: Tensor) -> Tensor:
        if value.ndim < 1 or value.shape[-1] != DIMENSION or not value.is_floating_point():
            raise ValueError("Observation network requires floating final dimension six")
        return value.to(dtype=torch.float64)

    def forward(self, observations: Tensor) -> Tensor:
        """Transform original cached observations, preserving their batch axes and gradients."""
        value = self._coordinates(observations)
        matrices = self._matrices.to(device=value.device)
        for index in range(LAYERS):
            value = value @ matrices[index].T
            if index < LAYERS - 1:
                value = smooth_leaky(value)
        return value

    def inverse(self, observations: Tensor) -> Tensor:
        """Recover the old observations for evaluation; never called during training."""
        value = self._coordinates(observations)
        matrices = self._matrices.to(device=value.device)
        for index in reversed(range(LAYERS)):
            value = value @ matrices[index]
            if index > 0:
                value = inverse_smooth_leaky(value)
        return value
