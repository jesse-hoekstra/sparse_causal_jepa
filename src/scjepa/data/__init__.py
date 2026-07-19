"""Datasets and dataloaders: CLEVRER, Push-T, synthetic ground-truth systems.

Will contain: video clip datasets and reproducible dataloaders for CLEVRER and
Push-T, plus synthetic dynamical systems with known ground-truth causal
parameters for the SHD/MCC diagnostics. Raw data lives under data/ (gitignored,
never committed).

Owner: data-pipeline-engineer.
"""

from scjepa.data.bounce import BounceDataset
from scjepa.data.synthetic_smoke import RandomClipDataset

__all__ = ["BounceDataset", "RandomClipDataset"]
