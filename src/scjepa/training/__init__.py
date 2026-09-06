"""Training loops and optimization for the two active experiments.

Experiment 1 predicts true states; Experiment 2 learns online visual states and
stopped-gradient EMA targets. Both train teacher forcing and sampled T=2
endpoints, with shared gradient safeguards, checkpointing, seeding, and W&B
logging. Representation regularizers are not active in either current preset.
"""

from scjepa.training.lagrangian import SparsityLagrangian
from scjepa.training.loop import (
    MetricLogger,
    NoopLogger,
    TrainConfig,
    Trainer,
    seed_everything,
)

__all__ = [
    "MetricLogger",
    "NoopLogger",
    "SparsityLagrangian",
    "TrainConfig",
    "Trainer",
    "seed_everything",
]
