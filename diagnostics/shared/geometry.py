"""Preserve isotropic physical dynamics while balancing observed action coordinates.

The Paley matrix uses S_ij=Legendre(j-i) modulo23 and
H=[[1,1^T],[1,S-I]], V=H/sqrt24. V is orthogonal but not symmetric.
Raw-control laws, parameters, state transport and observation maps are unchanged.
"""

import math

import torch
from torch import Tensor

from diagnostics.shared import source_geometry as source
from scjepa.data.types import TransitionJacobians

DIMENSION = source.DIMENSION
BLOCK_DIMENSION = source.BLOCK_DIMENSION
ACTION_DIMENSION = source.ACTION_DIMENSION
PARAMETER_BOUND = source.PARAMETER_BOUND
ROTATION_SCALE = source.ROTATION_SCALE
GENERATOR_VERSION = "isotropic-oracle-paley24-observed-actions-v1"
TransportChart = source.TransportChart
graphs = source.graphs
sample_controls = source.sample_controls
rotation_matrix = source.rotation_matrix


def paley_hadamard(*, device: torch.device | str | None = None) -> Tensor:
    """Return the integer Paley24 matrix with the certified Legendre(j-i) convention."""
    quadratic_residues = {value * value % 23 for value in range(1, 23)}
    result = torch.ones(24, 24, dtype=torch.int64, device=device)
    rows = [
        [-1 if i == j else (1 if (j - i) % 23 in quadratic_residues else -1) for j in range(23)]
        for i in range(23)
    ]
    result[1:, 1:] = torch.tensor(rows, dtype=torch.int64, device=device)
    return result


def action_rotation(*, device: torch.device | str | None = None) -> Tensor:
    """Return V=Paley24/sqrt24 with equal-magnitude entries and VV^T=I."""
    return paley_hadamard(device=device).to(torch.float64) / math.sqrt(24)


def _actions(value: Tensor) -> Tensor:
    if value.ndim < 1 or value.shape[-1] != 24 or not value.is_floating_point():
        raise ValueError("Controls/actions must be floating point with final dimension24")
    return value.to(torch.float64)


def encode_controls(controls: Tensor) -> Tensor:
    """Encode row-vector controls as xi@V.T; V is deliberately nonsymmetric."""
    controls = _actions(controls)
    return controls @ action_rotation(device=controls.device).T


def decode_actions(actions: Tensor) -> Tensor:
    """Decode row-vector actions as a@V, not a@V.T."""
    actions = _actions(actions)
    return actions @ action_rotation(device=actions.device)


def _source_actions(actions: Tensor) -> Tensor:
    return source.encode_controls(decode_actions(actions))


def state_rotation(
    actions: Tensor,
    *,
    rotation_scale: float = ROTATION_SCALE,
    transport_chart: TransportChart = "isotropic",
) -> Tensor:
    """Use the unchanged Cayley transport after inverting the new observation map."""
    return source.state_rotation(
        _source_actions(actions), rotation_scale=rotation_scale, transport_chart=transport_chart
    )


def transition(
    states: Tensor,
    parameters: Tensor,
    actions: Tensor,
    *,
    rho: float = 0.7,
    rotation_scale: float = ROTATION_SCALE,
    transport_chart: TransportChart = "isotropic",
) -> Tensor:
    """Evaluate exactly the source physical equation from re-expressed observed actions."""
    return source.transition(
        states,
        parameters,
        _source_actions(actions),
        rho=rho,
        rotation_scale=rotation_scale,
        transport_chart=transport_chart,
    )


def simulate(
    initial_state: Tensor,
    parameters: Tensor,
    actions: Tensor,
    *,
    rho: float = 0.7,
    rotation_scale: float = ROTATION_SCALE,
    transport_chart: TransportChart = "isotropic",
) -> Tensor:
    """Keep source rollout equations and gradient attachment unchanged."""
    return source.simulate(
        initial_state,
        parameters,
        _source_actions(actions),
        rho=rho,
        rotation_scale=rotation_scale,
        transport_chart=transport_chart,
    )


def analytic_jacobians(
    states: Tensor,
    parameters: Tensor,
    actions: Tensor,
    *,
    rho: float = 0.7,
    rotation_scale: float = ROTATION_SCALE,
    transport_chart: TransportChart = "isotropic",
) -> TransitionJacobians:
    """Apply J_new=J_source V_source V_new.T with unchanged state/parameter derivatives."""
    original = source.analytic_jacobians(
        states,
        parameters,
        _source_actions(actions),
        rho=rho,
        rotation_scale=rotation_scale,
        transport_chart=transport_chart,
    )
    transformed = (
        original.action
        @ source.action_rotation(device=actions.device)
        @ action_rotation(device=actions.device).T
    )
    return TransitionJacobians(original.state, transformed, original.parameter)
