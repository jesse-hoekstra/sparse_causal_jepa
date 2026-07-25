"""Datasets: the Bounce v2 simulator (experiments.pdf §6.1.1).

Every episode carries frames (optional), true states, masses, and the
per-transition contact record from which the ground-truth local graphs
(Eqs. 8/9) are derived. Raw data lives under data/ (gitignored).
"""

from scjepa.data.bounce import BounceDataset

__all__ = ["BounceDataset"]
