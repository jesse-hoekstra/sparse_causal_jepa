"""Anti-collapse regularizer on slot embeddings (D3): VISReg, SIGReg as ablation.

Joint training with no EMA target and no stop-gradient (D7) means this term is
the ONLY thing standing between the encoders and representational collapse.
D3 picks VISReg (its corrective gradient survives in the collapse regime, unlike
SIGReg's — Fig. 2 of sources/VISReg.pdf) and keeps SIGReg config-selectable as
an ablation/safety hatch. Both vendored losses share one contract:
``(G, B, D) → scalar`` with statistics over the sample axis ``dim=1``.

Placement per D3/Fig. 1: applied to BOTH branches' slot embeddings, flattening
slots (and any leading axes) into samples: ``(B, N, d) → (1, B·N, d)``.
Both losses draw fresh random projections per call, so values are stochastic.
"""

from typing import Literal, cast

from jaxtyping import Float
from torch import Tensor, nn

from scjepa.third_party.visreg import SIGReg, VISReg

RegularizerKind = Literal["visreg", "sigreg"]


class SlotRegularizer(nn.Module):
    """Config-selectable anti-collapse penalty over a batch of slot embeddings.

    Treats every slot vector as one sample: a batch of (B, N, d) slot embeddings
    is regularized as B·N samples in R^d, pushed toward an isotropic Gaussian
    (zero mean, unit per-dimension variance, Gaussian sliced shape).
    """

    def __init__(self, kind: RegularizerKind = "visreg", num_projections: int = 256) -> None:
        """Build the regularizer.

        Args:
            kind: ``"visreg"`` (D3 default) or ``"sigreg"`` (ablation).
            num_projections: Random projection directions per call (both kinds).
        """
        super().__init__()
        if kind == "visreg":
            self._impl: nn.Module = VISReg(num_projections=num_projections)
        elif kind == "sigreg":
            self._impl = SIGReg(num_projections=num_projections)
        else:  # pragma: no cover - Literal narrows, guards config typos at runtime
            raise ValueError(f"unknown regularizer kind: {kind!r}")
        self.kind: RegularizerKind = kind

    def forward(self, embeddings: Float[Tensor, "*lead d"]) -> Float[Tensor, ""]:
        """Penalize deviation of the embedding batch from an isotropic Gaussian.

        Args:
            embeddings: Slot embeddings with any leading axes, e.g. (B, N, d) or
                (B, T, N, d); everything but the last axis is flattened into
                samples. Needs at least 2 samples.

        Returns:
            Scalar penalty (large when the batch is collapsed).
        """
        if embeddings.ndim < 2:
            raise ValueError(f"expected (..., d), got shape {tuple(embeddings.shape)}")
        samples = embeddings.reshape(1, -1, embeddings.shape[-1])  # (1, S, d)
        if samples.shape[1] < 2:
            raise ValueError("regularizer needs at least 2 embedding samples")
        return cast(Tensor, self._impl(samples))


__all__ = ["RegularizerKind", "SlotRegularizer"]
