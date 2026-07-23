"""Model modules: SAVi encoders, channel split, SPARTAN predictor.

Will contain: the PyTorch SAVi encoder (D2 — SAVi, not SAVi++; validated against
the official JAX repo), the channel split (per-slot temporal attention pooling for
causal parameters per D4, plus the linear kinematic head on last-step slots), the
SPARTAN transition model (implemented from arXiv:2411.06890 — must expose its
interaction graph for SHD/MCC eval), and optional auxiliary-variable conditioning.

Owner: model-architecture-engineer (method-faithfulness reviewed by
paper-to-code-translator).
"""

from scjepa.models.channel_split import (
    AttnPooling,
    CrossSlotAttnPooling,
    KinematicHead,
    TrackAwareAttnPooling,
    TrackedSlotAttentionPooling,
)
from scjepa.models.jepa import JepaOutput, SCJepa
from scjepa.models.savi import SAViEncoder
from scjepa.models.spartan import Spartan, SpartanOutput
from scjepa.models.state_jepa import StateJepa

__all__ = [
    "AttnPooling",
    "CrossSlotAttnPooling",
    "JepaOutput",
    "KinematicHead",
    "SAViEncoder",
    "SCJepa",
    "Spartan",
    "SpartanOutput",
    "StateJepa",
    "TrackAwareAttnPooling",
    "TrackedSlotAttentionPooling",
]
