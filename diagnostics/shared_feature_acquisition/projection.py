"""Fit one shared-feature output initialization using TRAIN context groups only."""

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from diagnostics.multiquery_control.data import TensorSplit, module_fingerprint
from scjepa.data.datasets import tensor_fingerprint
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder


@dataclass(frozen=True)
class Projection:
    """Materialized float32 output rows and their TRAIN-only numerical receipt."""

    weight: Tensor
    bias: Tensor
    receipt: dict[str, Any]


@torch.no_grad()
def hidden_features(
    encoder: TransitionParameterEncoder, train: TensorSplit, *, batch_size: int = 64
) -> Tensor:
    """Extract per-context h128; group averaging occurs only in covariance estimation."""
    if set(train) != {"online", "target", "actions"}:
        raise ValueError("Projection accepts only TRAIN cached states/actions; no theta or IDs")
    if type(encoder) is not TransitionParameterEncoder or encoder.param_dim != 12:
        raise ValueError("Require original transition hidden layers and twelve-code output")
    if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in encoder.parameters()):
        raise ValueError("Projection extraction must use CPUfloat32 encoder weights")
    states, actions = train["online"], train["actions"]
    if (
        len(states) < 2
        or states.shape != (len(states), 10, 12, 6)
        or train["target"].shape != states.shape
        or actions.shape != (len(states), 10, 11, 16)
        or batch_size < 1
        or any(value.requires_grad for value in train.values())
    ):
        raise ValueError("Require frozen original G/R10/T12 TRAIN cache with at least two groups")
    pieces: list[Tensor] = []
    for at in range(0, len(states), batch_size):
        z = states[at : at + batch_size, :, :10].flatten(0, 1).float().cpu()
        a = actions[at : at + batch_size, :, :9].flatten(0, 1).to(z)
        transitions = torch.cat((z[:, :-1], a, z[:, 1:]), -1)
        h = encoder.head[:4](encoder.transition(transitions).mean(1))
        pieces.append(h.reshape(-1, 10, 128).double())
    result = torch.cat(pieces)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Nonfinite TRAIN hidden features")
    return result


def covariance(x: Tensor) -> Tensor:
    """Unbiased covariance of independent rows, without coordinate pooling."""
    centered = x - x.mean(0)
    return centered.T @ centered / (len(x) - 1)


@torch.no_grad()
def fit_projection(
    encoder: TransitionParameterEncoder, train: TensorSplit, *, batch_size: int = 64
) -> Projection:
    """Whiten total covariance, retain top12 shared modes, match original output mean/RMS."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        before = module_fingerprint(encoder)
        h = hidden_features(encoder, train, batch_size=batch_size)
        output = cast(nn.Linear, encoder.head[4])
        old = F.linear(h.float(), output.weight, output.bias).double()
        means = h.mean(1)
        centered = h - means[:, None]
        within = centered.flatten(0, 1).T @ centered.flatten(0, 1) / (len(h) * 9)
        between = covariance(means)
        shared = between - within / 10
        total = covariance(h.flatten(0, 1))
        te, tv = cast(tuple[Tensor, Tensor], torch.linalg.eigh(total))  # pyright: ignore[reportUnknownMemberType]
        floor = te.max() * 1e-6
        if not bool(floor > 0):
            raise ValueError("Degenerate total hidden covariance")
        whitening = (tv / te.clamp_min(floor).sqrt()) @ tv.T
        ge, gv = cast(tuple[Tensor, Tensor], torch.linalg.eigh(whitening @ shared @ whitening))  # pyright: ignore[reportUnknownMemberType]
        selected = gv[:, -12:].clone()
        for index in range(12):
            pivot = selected[:, index].abs().argmax()
            if bool(selected[pivot, index] < 0):
                selected[:, index].neg_()
        weight = selected.T @ whitening
        mean = h.flatten(0, 1).mean(0)
        old_mean = old.flatten(0, 1).mean(0)
        old_rms = old.flatten(0, 1).var(0, correction=0).mean().sqrt()
        proposed = (h - mean) @ weight.T
        new_rms = proposed.flatten(0, 1).var(0, correction=0).mean().sqrt()
        if not bool(old_rms > 0) or not bool(new_rms > 0):
            raise ValueError("Degenerate original or projected output variation")
        weight = weight * (old_rms / new_rms)
        bias = old_mean - weight @ mean
        weight, bias = weight.float(), bias.float()
        after = F.linear(h.float(), weight, bias).double()
        if module_fingerprint(encoder) != before:
            raise RuntimeError("Projection fitting mutated the source encoder")
        receipt: dict[str, Any] = {
            "source_encoder_sha256": before,
            "weight_sha256": tensor_fingerprint(weight),
            "bias_sha256": tensor_fingerprint(bias),
            "train_groups": len(h),
            "contexts_per_group": 10,
            "hidden_width": 128,
            "code_dimension": 12,
            "batch_size": batch_size,
            "threads": 2,
            "hidden_dtype": "float32",
            "covariance_dtype": "float64",
            "total_eigenfloor_relative": 1e-6,
            "total_eigenfloor_absolute": float(floor),
            "shared_generalized_eigenvalues": [float(value) for value in ge],
            "selected_eigenvector_order": "ascending within largest12, max-absolute pivot positive",
            "original_code_centered_rms": float(old_rms),
            "projected_code_centered_rms": float(
                after.flatten(0, 1).var(0, correction=0).mean().sqrt()
            ),
            "mean_match_max_error": float((after.flatten(0, 1).mean(0) - old_mean).abs().max()),
            "training_parameter_labels_absent": True,
            "formula": (
                "W=mean unbiased within-context covariance; S=Cov(groupmean)-W/10; "
                "T=Cov(individual h128), both unbiased. A=T_floor^-1/2; selected U are "
                "top12 eigenvectors of A S A; output W=U.T A times common scalar matching "
                "original TRAIN q centered RMS, bias matching original TRAIN q mean. "
                "No shared-eigenvalue inverse scaling, physical labels or heldout selection."
            ),
        }
        return Projection(weight, bias, receipt)
    finally:
        torch.set_num_threads(old_threads)


@torch.no_grad()
def apply_projection(model: SCJEPA, projection: Projection) -> dict[str, Any]:
    """Replace only twelve output rows/biases; keep individual-context inference unchanged."""
    encoder = model.parameter_encoder
    if type(encoder) is not TransitionParameterEncoder or encoder.param_dim != 12:
        raise ValueError("Expected original twelve-code transition encoder")
    if module_fingerprint(encoder) != projection.receipt["source_encoder_sha256"]:
        raise ValueError("Projection was fitted to a different encoder")
    if projection.weight.shape != (12, 128) or projection.bias.shape != (12,):
        raise ValueError("Projection has wrong output interface")
    before = {name: tensor_fingerprint(value) for name, value in encoder.state_dict().items()}
    fixed = {
        name: module_fingerprint(getattr(model, name))
        for name in ("predictor", "online_encoder", "target_encoder")
    }
    output = cast(nn.Linear, encoder.head[4])
    output.weight.copy_(projection.weight)
    output.bias.copy_(projection.bias)
    if any(
        tensor_fingerprint(value) != before[name]
        for name, value in encoder.state_dict().items()
        if name not in ("head.4.weight", "head.4.bias")
    ) or any(module_fingerprint(getattr(model, name)) != value for name, value in fixed.items()):
        raise RuntimeError("Projection changed a hidden layer, predictor or state map")
    return {"encoder_sha256": module_fingerprint(encoder), "fixed_module_hashes": fixed}
