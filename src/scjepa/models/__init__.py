"""Model modules for the experiment ladder (experiments.pdf §6).

Experiment 1: ``ParameterEncoder`` (P_η, §6.2 Eqs. 16-26) + ``Spartan``
(f_gamma, Eqs. 27-37) composed by ``Experiment1Model`` (Eq. 38). ``SAViEncoder``
is the shared causal visual substrate for Experiments 2 and 3 (§6.3).
"""

from scjepa.models.experiment1 import Experiment1Model, TransitionOutput, build_experiment1
from scjepa.models.parameter_encoder import ParameterEncoder
from scjepa.models.savi import SAViEncoder
from scjepa.models.spartan import Spartan, SpartanLayer, SpartanOutput

__all__ = [
    "Experiment1Model",
    "ParameterEncoder",
    "SAViEncoder",
    "Spartan",
    "SpartanLayer",
    "SpartanOutput",
    "TransitionOutput",
    "build_experiment1",
]
