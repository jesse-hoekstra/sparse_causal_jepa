"""Vendored VISReg/SIGReg anti-collapse losses (see PROVENANCE.md).

Input contract (both): ``(G, B, D)`` — statistics are computed over the SAMPLE
axis ``dim=1``; ``G`` is an arbitrary leading group axis. Returns a scalar.
"""

from .sigreg import SIGReg
from .visreg import VISReg

__all__ = ["SIGReg", "VISReg"]
