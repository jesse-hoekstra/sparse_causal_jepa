"""Experiment 1's label-free shared-feature initialization and projection for two lags."""

from typing import Any, cast

import torch
from torch import Tensor, nn

from diagnostics.coupled_duffing.model import LaggedParameterEncoder, LaggedPredictor
from diagnostics.masked_mlp_learned.acquisition import group_covariances, shared_modes
from scjepa.data.datasets import tensor_fingerprint
from scjepa.models.model import SCJEPA


@torch.no_grad()
def training_features(
    model: SCJEPA, inputs: dict[str, Tensor], *, codes: bool, chunk: int = 16
) -> Tensor:
    """Collect one code/feature per TRAIN context without accepting physical labels."""
    if set(inputs) != {"observations", "actions"}:
        raise ValueError("initialization and projection accept only observations and actions")
    encoder = model.parameter_encoder
    if not isinstance(encoder, LaggedParameterEncoder):
        raise TypeError("expected a lagged transition encoder")
    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    pieces: list[Tensor] = []
    try:
        for start in range(0, len(inputs["observations"]), chunk):
            x = inputs["observations"][start : start + chunk, :, : model.context_len]
            a = inputs["actions"][start : start + chunk, :, : model.context_len - 1]
            groups, views = x.shape[:2]
            x, a = x.to(device), a.to(device)
            states = model.online_encoder.encode_sequence(x.flatten(0, 1))
            features = encoder.features(states, a.flatten(0, 1))
            values = encoder.head[-1](features) if codes else features
            pieces.append(values.reshape(groups, views, -1).double().cpu())
    finally:
        for module, training in modes:
            module.training = training
    result = torch.cat(pieces)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("nonfinite TRAIN features")
    return result


@torch.no_grad()
def initialize_shared_output(model: SCJEPA, inputs: dict[str, Tensor]) -> dict[str, Any]:
    """Use exactly Experiment 1's whitened shared modes, RMS scaling and mean matching."""
    features = training_features(model, inputs, codes=False)
    rows = features.flatten(0, 1)
    encoder = cast(LaggedParameterEncoder, model.parameter_encoder)
    layer = cast(nn.Linear, encoder.head[-1])
    original = rows @ layer.weight.detach().double().cpu().T + layer.bias.detach().double().cpu()
    weight, values = shared_modes(features, layer.out_features)
    mean = rows.mean(0)
    proposed = (rows - mean) @ weight.T
    old_rms = original.var(0, correction=0).mean().sqrt()
    new_rms = proposed.var(0, correction=0).mean().sqrt()
    if not bool(old_rms > 0) or not bool(new_rms > 0):
        raise ValueError("degenerate shared-feature initialization")
    weight = weight * old_rms / new_rms
    bias = original.mean(0) - weight @ mean
    layer.weight.copy_(weight.to(layer.weight))
    layer.bias.copy_(bias.to(layer.bias))
    return {
        "train_groups": len(features),
        "contexts_per_group": features.shape[1],
        "code_dimension": layer.out_features,
        "shared_generalized_eigenvalues": [float(value) for value in values],
        "weight_sha256": tensor_fingerprint(layer.weight),
        "bias_sha256": tensor_fingerprint(layer.bias),
        "training_parameter_labels_absent": True,
    }


@torch.no_grad()
def project(
    source: SCJEPA, target: SCJEPA, inputs: dict[str, Tensor]
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Fold the shared-covariance projection and OLS reconstruction into existing weights."""
    old = cast(LaggedPredictor, source.predictor)
    new = cast(LaggedPredictor, target.predictor)
    width, dimension = old.param_dim, new.param_dim
    if old.learned_logits() or new.learned_logits() or not 0 < dimension < width:
        raise ValueError("projection requires dense predictors and a strictly smaller code")
    codes = training_features(source, inputs, codes=True)
    rows = codes.flatten(0, 1)
    mean = rows.mean(0)
    shared, total = group_covariances(codes)
    values, vectors = cast(tuple[Tensor, Tensor], torch.linalg.eigh(shared))  # pyright: ignore[reportUnknownMemberType]
    selected = vectors[:, -dimension:].clone()
    for column in range(dimension):
        if bool(selected[selected[:, column].abs().argmax(), column] < 0):
            selected[:, column].neg_()
    projection = selected.T
    inverse = cast(Tensor, torch.linalg.inv(projection @ total @ projection.T))  # pyright: ignore[reportUnknownMemberType]
    reconstruction = total @ projection.T @ inverse
    state = {
        key: value.clone() if isinstance(value, Tensor) else value
        for key, value in source.state_dict().items()
    }
    weight, bias = state["parameter_encoder.head.4.weight"], state["parameter_encoder.head.4.bias"]
    state["parameter_encoder.head.4.weight"] = (projection @ weight.double().cpu()).to(weight)
    state["parameter_encoder.head.4.bias"] = (projection @ (bias.double().cpu() - mean)).to(bias)
    first, first_bias = state["predictor.first.weight"], state["predictor.first.bias"]
    prefix = first.shape[1] - width
    code_columns = first[:, prefix:].double().cpu()
    state["predictor.first.weight"] = torch.cat(
        (first[:, :prefix], (code_columns @ reconstruction).to(first)), dim=1
    )
    state["predictor.first.bias"] = (first_bias.double().cpu() + code_columns @ mean).to(first_bias)
    target.load_state_dict(state, strict=True)
    projected = (rows - mean) @ projection.T
    residual = rows - mean - projected @ reconstruction.T
    return {
        "source_code_dimension": width,
        "code_dimension": dimension,
        "train_groups": len(codes),
        "contexts_per_group": codes.shape[1],
        "shared_eigenvalues": [float(value) for value in values],
        "selected_shared_variance_fraction": float(
            values[-dimension:].sum() / values.clamp_min(0).sum()
        ),
        "reconstructed_variance_fraction": float(
            1 - residual.square().sum() / (rows - mean).square().sum()
        ),
        "projection_sha256": tensor_fingerprint(projection.float()),
        "training_parameter_labels_absent": True,
    }, {"projection": projection, "reconstruction": reconstruction, "mean": mean}
