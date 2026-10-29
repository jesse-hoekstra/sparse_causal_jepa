"""Label-free code acquisition aids from the twelve-code audits, and the 12-to-6 projection.

These follow the completed acquisition record (``docs/current_experiment.md``,
"Fresh-code conditioning", "Selected mechanism controls" and the canonical
shared-seed PCA audit), where a twelve-coordinate code with these two
initializers retained all six parameters in both declared seeds:

- **Encoder gains.** The five hidden ``Linear`` weights of the transition parameter
  encoder are multiplied by sqrt(6) (variance 2/fan_in) and its output weights by
  sqrt(3) (1/fan_in); biases are unchanged. A fresh code otherwise varies about a
  thousand times less than the states and actions entering the predictor.
- **Shared-feature output rows.** From the penultimate features h of every TRAIN
  context, W is the mean unbiased within-group covariance, S = Cov(group means) - W/R
  and T = Cov(all contexts). T is whitened with a relative eigenvalue floor of 1e-6
  and the output rows become U^T T^-1/2 for the top-k eigenvectors U of
  T^-1/2 S T^-1/2 (largest-magnitude coordinate positive), rescaled by one scalar to
  the original code's centered RMS, with the bias matching its mean.
- **Projection to six coordinates.** The trained twelve-dimensional TRAIN codes are
  projected onto the six leading eigenvectors U of their group-shared covariance S
  (unwhitened, so a direction counts by its shared variance in code units, and
  within-episode nuisance is discounted): ``c = U^T (q - mu)``. The encoder output
  rows are composed with U^T and the predictor's code columns with the least-squares
  reconstruction ``q ~ mu + T U (U^T T U)^-1 c``, so the six-code model starts from
  the twelve-code one. The sparse arms then use a square code, as the
  identifiability theorem assumes.

Only observations, actions and the known group membership of the TRAIN contexts
enter; true parameters never do.
"""

import math
from typing import Any, cast

import torch
from torch import Tensor, nn

from diagnostics.isotropic_oracle.data import StateData
from diagnostics.masked_mlp_oracle.model import MaskedTrunkPredictor
from scjepa.data.datasets import tensor_fingerprint
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder

HIDDEN_GAIN = math.sqrt(6.0)
OUTPUT_GAIN = math.sqrt(3.0)
EIGENFLOOR = 1e-6


def _encoder(model: SCJEPA) -> TransitionParameterEncoder:
    encoder = model.parameter_encoder
    if type(encoder) is not TransitionParameterEncoder:
        raise TypeError("expected the transition parameter encoder")
    return encoder


def _linears(encoder: TransitionParameterEncoder) -> tuple[list[nn.Linear], nn.Linear]:
    layers = [m for m in (*encoder.transition, *encoder.head) if isinstance(m, nn.Linear)]
    if len(layers) != 6:
        raise ValueError("expected five hidden and one output Linear layer")
    return layers[:5], layers[5]


@torch.no_grad()
def apply_encoder_gains(model: SCJEPA) -> dict[str, Any]:
    """Scale hidden weights by sqrt(6) and output weights by sqrt(3); keep biases."""
    hidden, output = _linears(_encoder(model))
    for layer in hidden:
        layer.weight.mul_(HIDDEN_GAIN)
    output.weight.mul_(OUTPUT_GAIN)
    return {"hidden_weight_gain": HIDDEN_GAIN, "output_weight_gain": OUTPUT_GAIN}


@torch.no_grad()
def train_contexts(
    model: SCJEPA, data: StateData, device: torch.device, *, codes: bool, chunk: int = 256
) -> Tensor:
    """Return float64 (G, R, D) penultimate features or codes of every TRAIN context."""
    if data.split != "train":
        raise ValueError("initializers and projections fit TRAIN contexts only")
    encoder = _encoder(model)
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    pieces: list[Tensor] = []
    try:
        for start in range(0, len(data), chunk):
            indices = torch.arange(start, min(start + chunk, len(data)))
            batch = data.batch(indices, device)
            groups, views = batch["observations"].shape[:2]
            states = model.online_encoder.encode_sequence(batch["observations"].flatten(0, 1))
            states = states[:, : model.context_len]
            actions = batch["actions"].flatten(0, 1)[:, : model.context_len - 1]
            transitions = torch.cat((states[:, :-1], actions, states[:, 1:]), dim=-1)
            values = encoder.head[:4](encoder.transition(transitions).mean(1))
            if codes:
                values = encoder.head[4](values)
            pieces.append(values.reshape(groups, views, -1).double().cpu())
    finally:
        for module, training in modes:
            module.training = training
    result = torch.cat(pieces)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("nonfinite TRAIN features")
    return result


def _covariance(rows: Tensor) -> Tensor:
    centered = rows - rows.mean(0)
    return centered.T @ centered / (len(rows) - 1)


def group_covariances(features: Tensor) -> tuple[Tensor, Tensor]:
    """Return the group-shared covariance S = Cov(means) - W/R and the total covariance T."""
    groups, views, _ = features.shape
    if groups < 2 or views < 2:
        raise ValueError("group covariances need two groups of at least two contexts")
    means = features.mean(1)
    centered = (features - means[:, None]).flatten(0, 1)
    within = centered.T @ centered / (groups * (views - 1))
    return _covariance(means) - within / views, _covariance(features.flatten(0, 1))


def _top(vectors: Tensor, k: int) -> Tensor:
    """Last k eigenvector columns, each with its largest-magnitude coordinate positive."""
    selected = vectors[:, -k:].clone()
    for index in range(k):
        if bool(selected[selected[:, index].abs().argmax(), index] < 0):
            selected[:, index].neg_()
    return selected


def shared_modes(features: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Return rows U^T T^-1/2 of the top-k whitened shared modes and their eigenvalues."""
    if not 1 <= k <= features.shape[-1]:
        raise ValueError("1 <= k <= width is required")
    shared, total = group_covariances(features)
    eigenvalues, vectors = cast(tuple[Tensor, Tensor], torch.linalg.eigh(total))  # pyright: ignore[reportUnknownMemberType]
    floor = eigenvalues.max() * EIGENFLOOR
    if not bool(floor > 0):
        raise ValueError("degenerate total covariance")
    whitening = (vectors / eigenvalues.clamp_min(floor).sqrt()) @ vectors.T
    values, modes = cast(
        tuple[Tensor, Tensor],
        torch.linalg.eigh(whitening @ shared @ whitening),  # pyright: ignore[reportUnknownMemberType]
    )
    return _top(modes, k).T @ whitening, values[-k:]


def _centered_rms(rows: Tensor) -> Tensor:
    return rows.var(0, correction=0).mean().sqrt()


@torch.no_grad()
def initialize_shared_output(
    model: SCJEPA, data: StateData, device: torch.device
) -> dict[str, Any]:
    """Replace only the encoder output rows by the top shared feature modes."""
    _, output = _linears(_encoder(model))
    features = train_contexts(model, data, device, codes=False)
    rows = features.flatten(0, 1)
    original = rows @ output.weight.detach().double().cpu().T + output.bias.detach().double().cpu()
    weight, values = shared_modes(features, output.out_features)
    mean = rows.mean(0)
    proposed = (rows - mean) @ weight.T
    old_rms, new_rms = _centered_rms(original), _centered_rms(proposed)
    if not bool(old_rms > 0) or not bool(new_rms > 0):
        raise ValueError("degenerate original or projected code variation")
    weight = weight * (old_rms / new_rms)
    bias = original.mean(0) - weight @ mean
    output.weight.copy_(weight.to(output.weight))
    output.bias.copy_(bias.to(output.bias))
    return {
        "train_groups": features.shape[0],
        "contexts_per_group": features.shape[1],
        "feature_width": features.shape[2],
        "code_dimension": output.out_features,
        "shared_generalized_eigenvalues": [float(v) for v in values],
        "original_code_centered_rms": float(old_rms),
        "weight_sha256": tensor_fingerprint(output.weight.detach().cpu()),
        "bias_sha256": tensor_fingerprint(output.bias.detach().cpu()),
        "training_parameter_labels_absent": True,
    }


@torch.no_grad()
def project_code(
    source: SCJEPA, target: SCJEPA, data: StateData, device: torch.device
) -> dict[str, Any]:
    """Load ``source`` into the smaller-code ``target`` through a shared-mode projection."""
    width, dim = source.predictor.param_dim, target.predictor.param_dim
    if dim >= width:
        raise ValueError("the projection must reduce the code dimension")
    predictor = cast(nn.Module, source.predictor)
    if not isinstance(predictor, MaskedTrunkPredictor):
        raise TypeError("expected the masked MLP predictor")
    if predictor.learned_logits():
        raise ValueError("project the dense endpoint, before any mask is learned")
    codes = train_contexts(source, data, device, codes=True)
    rows = codes.flatten(0, 1)
    mean = rows.mean(0)
    shared, total = group_covariances(codes)
    values, vectors = cast(tuple[Tensor, Tensor], torch.linalg.eigh(shared))  # pyright: ignore[reportUnknownMemberType]
    projection = _top(vectors, dim).T
    inverse = cast(Tensor, torch.linalg.inv(projection @ total @ projection.T))  # pyright: ignore[reportUnknownMemberType]
    reconstruction = total @ projection.T @ inverse
    state = {
        key: value.clone() if isinstance(value, Tensor) else value
        for key, value in source.state_dict().items()
    }
    names = ("parameter_encoder.head.4.weight", "parameter_encoder.head.4.bias")
    weight, bias = (state[name].double().cpu() for name in names)
    state[names[0]] = (projection @ weight).to(state[names[0]])
    state[names[1]] = (projection @ (bias - mean)).to(state[names[1]])
    first, first_bias = state["predictor.first.weight"], state["predictor.first.bias"]
    columns = first.shape[1] - width
    code_columns = first[:, columns:].double().cpu()
    state["predictor.first.weight"] = torch.cat(
        (first[:, :columns], (code_columns @ reconstruction).to(first)), dim=1
    )
    state["predictor.first.bias"] = (first_bias.double().cpu() + code_columns @ mean).to(first_bias)
    target.load_state_dict(state, strict=True)
    projected = (rows - mean) @ projection.T
    residual = rows - mean - projected @ reconstruction.T
    return {
        "source_code_dimension": width,
        "code_dimension": dim,
        "train_groups": codes.shape[0],
        "contexts_per_group": codes.shape[1],
        "shared_eigenvalues": [float(v) for v in values],
        "selected_shared_variance_fraction": float(values[-dim:].sum() / values.clamp_min(0).sum()),
        "source_code_centered_rms": float(_centered_rms(rows)),
        "projected_code_centered_rms": float(_centered_rms(projected)),
        "reconstructed_variance_fraction": float(
            1 - residual.square().sum() / (rows - mean).square().sum()
        ),
        "projection_sha256": tensor_fingerprint(projection.float()),
        "training_parameter_labels_absent": True,
    }


__all__ = [
    "apply_encoder_gains",
    "group_covariances",
    "initialize_shared_output",
    "project_code",
    "shared_modes",
    "train_contexts",
]
