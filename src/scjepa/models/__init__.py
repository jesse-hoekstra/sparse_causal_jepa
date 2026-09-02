"""Model modules for the experiment ladder (experiments.pdf §6).

State-to-state: ``ParameterEncoder`` (P_η, §6.2 Eqs. 16-26) + ``Spartan``
(f_gamma, Eqs. 27-37) composed by ``StateToStateModel`` (Eq. 38). ``SAViEncoder``
is the shared causal visual substrate for Experiments 2 and 3 (§6.3).
"""

from scjepa.models.parameter_encoder import ParameterEncoder
from scjepa.models.savi import SAViEncoder
from scjepa.models.spartan import Spartan, SpartanLayer, SpartanOutput
from scjepa.models.state_to_state import (
    StateToStateModel,
    TransitionOutput,
    build_state_to_state,
    num_valid_rollout_t2_offsets,
    sample_rollout_t2_offsets,
)

__all__ = [
    "ParameterEncoder",
    "SAViEncoder",
    "Spartan",
    "SpartanLayer",
    "SpartanOutput",
    "StateToStateModel",
    "TransitionOutput",
    "build_state_to_state",
    "num_valid_rollout_t2_offsets",
    "sample_rollout_t2_offsets",
]
