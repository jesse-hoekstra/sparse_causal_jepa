"""Typed adapter around the retained, attributed SIGReg loss."""

from typing import Literal, cast

from torch import Tensor, nn

from scjepa.third_party.visreg import SIGReg

RegularizerKind = Literal["none", "sigreg"]


class RepresentationRegularizer(nn.Module):
    """Regularize independent episodes along axis 1 of ``(groups, B, D)``.

    For state representations each time and episode view is a separate group.
    For time-constant parameters each view is a separate group. The independent
    sample axis never pools two episodes that share the same true parameter.
    These distributional penalties do not establish injectivity or identifiability.
    """

    def __init__(self, kind: RegularizerKind, num_projections: int = 256) -> None:
        """Select an existing regularizer without altering its upstream implementation."""
        super().__init__()
        if kind not in {"none", "sigreg"}:
            raise ValueError("regularizer must be none or sigreg")
        if num_projections < 1:
            raise ValueError("num_projections must be positive")
        self.kind = kind
        self.loss: nn.Module = (
            SIGReg(num_projections=num_projections) if kind == "sigreg" else nn.Identity()
        )

    def forward(self, samples: Tensor) -> Tensor:
        """Compute episode statistics, preserving the independent group dimension."""
        if samples.ndim != 3:
            raise ValueError("regularizer samples must have shape (groups, B, D)")
        if self.kind == "none":
            return samples.new_zeros(())
        if samples.shape[1] < 2 or samples.shape[0] < 1 or samples.shape[2] < 1:
            raise ValueError("regularization requires at least two episodes per local batch")
        return cast(Tensor, self.loss(samples))


__all__ = ["RegularizerKind", "RepresentationRegularizer"]
