"""Training loop and optimization.

Will contain: the JEPA training loop (adapted from the vendored le-wm loop),
joint from-scratch training of both SAVi encoders (D7 — no pretrained init, no
EMA/stop-gradient asymmetry; collapse prevention is the regularizer alone), loss
assembly, optimizers/schedules, seeding, and W&B run logging (git SHA + full
config per run).

Owner: experiment-infra-engineer.
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
