"""Expand only the gained code interface while preserving all common parameters."""

import copy
import math
from typing import Any, cast

import torch
from torch import Tensor, nn

from diagnostics.mlp_oracle_code_control.model import MLPPredictor, Prediction
from diagnostics.multiquery_control.data import module_fingerprint
from scjepa.data.datasets import tensor_fingerprint
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.models.spartan import Spartan


class OvercompletePredictor(MLPPredictor):
    """34→211→211→6 ReLU predictor; twelve code coordinates, six physical states."""

    param_dim = 12

    def __init__(self) -> None:
        """Construct the declared interface; the initializer preserves common source weights."""
        super().__init__()
        self.network[0] = nn.Linear(34, 211)

    def forward(self, states: Tensor, parameters: Tensor, actions: Tensor) -> Prediction:
        """Consume all twelve raw code coordinates without normalization or truncation."""
        batch = len(states)
        if states.shape != (batch, 1, 6) or parameters.shape != (batch, 12):
            raise ValueError("Require (B,1,6) states and (B,12) complete parameter codes")
        if actions.shape != (batch, 1, 16):
            raise ValueError("Require (B,1,16) observed actions")
        return Prediction(self.network(torch.cat((states[:, 0], actions[:, 0], parameters), -1)))


@torch.no_grad()
def initialize_overcomplete(model: SCJEPA, seed: int) -> dict[str, Any]:
    """Expand authenticated gained E6/fresh P6 in CPUfloat32 with isolated extra seeds."""
    encoder = model.parameter_encoder
    predictor = cast(nn.Module, model.predictor)
    if type(encoder) is not TransitionParameterEncoder or type(predictor) is not MLPPredictor:
        raise ValueError("Require original transition E6 and exact fresh MLP6 before expansion")
    if (encoder.state_dim, encoder.param_dim, encoder.action_dim) != (6, 6, 16):
        raise ValueError("Authenticate original six-code architecture before replacement")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Expand modules in CPUfloat32 before moving to the training device")
    old_output = cast(nn.Linear, encoder.head[4])
    old_input = cast(nn.Linear, predictor.network[0])
    before = {
        name: module_fingerprint(getattr(model, name))
        for name in ("parameter_encoder", "predictor", "online_encoder", "target_encoder")
    }
    # Only the CPU generator is touched, and its state is restored on exit. CUDA
    # generators are never seeded or used by this initialization procedure.
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        torch.default_generator.manual_seed(seed + 1)
        additional_output = nn.Linear(128, 6)
        additional_output.weight.mul_(math.sqrt(3))
        expanded_encoder = copy.deepcopy(encoder)
        expanded_encoder.param_dim = 12
        expanded_encoder.head[4] = nn.Linear(128, 12)
        output = cast(nn.Linear, expanded_encoder.head[4])
        output.weight[:6].copy_(old_output.weight)
        output.bias[:6].copy_(old_output.bias)
        output.weight[6:].copy_(additional_output.weight)
        output.bias[6:].copy_(additional_output.bias)
        expanded_predictor = OvercompletePredictor()
        expanded_predictor.network[2].load_state_dict(predictor.network[2].state_dict())
        expanded_predictor.network[4].load_state_dict(predictor.network[4].state_dict())
        first = cast(nn.Linear, expanded_predictor.network[0])
        first.weight[:, :28].copy_(old_input.weight)
        first.bias.copy_(old_input.bias)
        torch.default_generator.manual_seed(seed + 2)
        first.weight[:, 28:].uniform_(-1 / math.sqrt(28), 1 / math.sqrt(28))
    model.parameter_encoder = expanded_encoder
    model.predictor = cast(Spartan, expanded_predictor)
    if any(
        module_fingerprint(getattr(model, name)) != before[name]
        for name in ("online_encoder", "target_encoder")
    ):
        raise RuntimeError("Overcomplete initialization changed frozen state maps")
    return {
        "scientific_seed": seed,
        "additional_encoder_seed": seed + 1,
        "additional_predictor_seed": seed + 2,
        "code_dimension": 12,
        "physical_parameter_dimension": 6,
        "source_module_hashes": before,
        "encoder_sha256": module_fingerprint(expanded_encoder),
        "predictor_sha256": module_fingerprint(expanded_predictor),
        "encoder_parameter_count": sum(p.numel() for p in expanded_encoder.parameters()),
        "predictor_parameter_count": sum(p.numel() for p in expanded_predictor.parameters()),
        "additional_output_weight_sha256": tensor_fingerprint(output.weight[6:]),
        "additional_output_bias_sha256": tensor_fingerprint(output.bias[6:]),
        "additional_predictor_columns_sha256": tensor_fingerprint(first.weight[:, 28:]),
        "policy": (
            "Preserve gained E hidden blocks and first6 output rows/biases exactly; append "
            "native Linear(128,6) seed+1 rows with sqrt3 weight gain and unscaled biases. "
            "Preserve fresh P common28 columns, all biases and later layers exactly; append "
            "six seed+2 Uniform(-1/sqrt28,1/sqrt28) q columns, nonzero. No whitening, "
            "normalization or truncation. CPUfloat32; caller CPU/CUDA RNG unchanged."
        ),
    }
