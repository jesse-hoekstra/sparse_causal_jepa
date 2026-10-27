"""One deterministic archived-weight initialization policy; no architecture or RNG changes."""

import math
from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import nn

from diagnostics.multiquery_control.data import module_fingerprint
from scjepa.data.datasets import tensor_fingerprint
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder

ORIGINAL_ENCODER_SHA256 = "d861e1cf36f7d9c8a1052e5c549b9ae5d01090f048f18ce603eec08632dcf1ac"
CANDIDATE_ENCODER_SHA256 = "4ce40952f7f46527c118db5b6c9dbbf6b17e682a5afc1ab72293475d0fbbccd8"
LAYER_NAMES = ("transition.0", "transition.2", "transition.4", "head.0", "head.2", "head.4")


@torch.no_grad()
def apply_encoder_gains(model: SCJEPA) -> dict[str, Any]:
    """Rescale five hidden weights bysqrt6 and output weight bysqrt3 in CPUfloat32."""
    encoder = model.parameter_encoder
    if type(encoder) is not TransitionParameterEncoder:
        raise ValueError("Require the exact original transition encoder class")
    parameters = list(encoder.parameters())
    count = sum(parameter.numel() for parameter in parameters)
    if (encoder.state_dim, encoder.param_dim, encoder.action_dim, count) != (6, 6, 16, 70534):
        raise ValueError("Require original6state/6code/16action/70534parameter architecture")
    if any(
        parameter.device.type != "cpu" or parameter.dtype != torch.float32
        for parameter in parameters
    ):
        raise ValueError("Apply archived gains in CPUfloat32 before moving to training device")
    modules = cast(Iterable[tuple[str, nn.Module]], encoder.named_modules())
    layers = [(name, layer) for name, layer in modules if isinstance(layer, nn.Linear)]
    if tuple(name for name, _ in layers) != LAYER_NAMES:
        raise ValueError("Require exact six biased Linear layers")
    before_hash = module_fingerprint(encoder)
    fixed_before = {
        name: module_fingerprint(getattr(model, name))
        for name in ("online_encoder", "target_encoder", "predictor")
    }
    bias_before = {name: tensor_fingerprint(layer.bias) for name, layer in layers}
    changes: list[dict[str, Any]] = []
    for index, (name, layer) in enumerate(layers):
        before = layer.weight.detach().clone()
        gain = math.sqrt(3 if index == 5 else 6)
        layer.weight.mul_(gain)
        if not torch.equal(layer.weight, before * gain):
            raise RuntimeError("Weight gain differs from declared CPUfloat32 multiplication")
        changes.append(
            {
                "module": name,
                "gain": gain,
                "weight_sha256_before": tensor_fingerprint(before),
                "weight_sha256_after": tensor_fingerprint(layer.weight),
            }
        )
    bias_after = {name: tensor_fingerprint(layer.bias) for name, layer in layers}
    if bias_after != bias_before or any(
        module_fingerprint(getattr(model, name)) != value for name, value in fixed_before.items()
    ):
        raise RuntimeError("Initialization changed biases, predictor or state maps")
    return {
        "original_encoder_sha256": before_hash,
        "module_sha256": module_fingerprint(encoder),
        "parameter_count": count,
        "changes": changes,
        "bias_hashes_before": bias_before,
        "bias_hashes_after": bias_after,
        "fixed_module_hashes": fixed_before,
        "policy": (
            "Original archived fresh128-wide E, same class/forward/tensor objects. CPUfloat32 "
            "Multiply five hidden weights bymath.sqrt(6), final weight bymath.sqrt(3) in-place. "
            "All six biases and weight directions retained; no random draws or new E seed. "
            "One fixed initialization policy, not a gain sweep."
        ),
    }
