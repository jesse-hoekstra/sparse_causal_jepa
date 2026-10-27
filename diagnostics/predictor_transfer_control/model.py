"""Validated centered TRAIN-PCA chart and affine E/P interface composition."""

import copy
from typing import Any, cast

import torch
from torch import Tensor, nn

from diagnostics.mlp_oracle_code_control.model import MLPPredictor
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.overcomplete_acquisition.model import OvercompletePredictor
from scjepa.data.datasets import canonical_fingerprint
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.models.spartan import Spartan


def audit_model_fingerprint(model: nn.Module) -> str:
    """Hash tensor-bearing child modules and SCJEPA prediction flags separately."""
    payload: dict[str, Any] = {
        name: module_fingerprint(child) for name, child in model.named_children()
    }
    if isinstance(model, SCJEPA):
        payload["prediction_flags"] = model.get_extra_state()
    return canonical_fingerprint(payload)


def fit_pca(train_codes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Center all TRAIN individual contexts and retain top six covariance eigenvectors."""
    if train_codes.ndim != 3 or train_codes.shape[-1] != 12:
        raise ValueError("Expected grouped individual-context twelve-dimensional TRAIN codes")
    flat = train_codes.double().flatten(0, 1)
    mean = flat.mean(0)
    centered = flat - mean
    covariance = centered.T @ centered / len(centered)
    values, vectors = cast(tuple[Tensor, Tensor], torch.linalg.eigh(covariance))  # pyright: ignore[reportUnknownMemberType]
    return mean, vectors[:, -6:].flip(1), values.flip(0)


@torch.no_grad()
def compose_pca(model: SCJEPA, mean: Tensor, basis: Tensor) -> SCJEPA:
    """Fold affine code projection/reconstruction into E output and P input weights."""
    rank = basis.shape[1]
    if mean.shape != (12,) or basis.shape != (12, rank) or rank not in (6, 12):
        raise ValueError("Expected a twelve-dimensional center and six/full-rank basis")
    torch.testing.assert_close(
        basis.T @ basis, torch.eye(rank, dtype=basis.dtype), atol=1e-10, rtol=1e-10
    )
    if not isinstance(model.parameter_encoder, TransitionParameterEncoder) or not isinstance(
        model.predictor, OvercompletePredictor
    ):
        raise TypeError("Require original transition encoder and twelve-code MLP")
    encoder, source_predictor = model.parameter_encoder, model.predictor
    result = copy.deepcopy(model)
    result_encoder = cast(TransitionParameterEncoder, result.parameter_encoder)
    old_output = cast(nn.Linear, encoder.head[4])
    old_input = cast(nn.Linear, source_predictor.network[0])
    result.parameter_encoder.param_dim = rank
    output = nn.Linear(old_output.in_features, rank).to(old_output.weight)
    output.weight.copy_(basis.T @ old_output.weight)
    output.bias.copy_(basis.T @ (old_output.bias - mean))
    result_encoder.head[4] = output
    predictor = MLPPredictor() if rank == 6 else OvercompletePredictor()
    predictor = predictor.to(old_input.weight)
    predictor.network[2].load_state_dict(source_predictor.network[2].state_dict())
    predictor.network[4].load_state_dict(source_predictor.network[4].state_dict())
    first = cast(nn.Linear, predictor.network[0])
    first.weight[:, :22].copy_(old_input.weight[:, :22])
    first.weight[:, 22:].copy_(old_input.weight[:, 22:] @ basis)
    first.bias.copy_(old_input.bias + old_input.weight[:, 22:] @ mean)
    result.predictor = cast(Spartan, predictor)
    return result.requires_grad_(False).eval()
