"""Change only innovation phase; preserve frozen Cayley transport and Paley24 actions.

The own/partner coefficient vector is PHASE_GAIN*(cos(pi*u),sin(pi*u)). Its
second moment is proportional to identity, so neither a physical parameter nor
a rotated parameter-pair direction has a privileged mean squared sensitivity.
"""

import math
from typing import cast

import torch
from torch import Tensor

from diagnostics.balanced_action_oracle import dgp as source
from diagnostics.isotropic_oracle import dgp as transport
from scjepa.data.types import TransitionJacobians

DIMENSION = source.DIMENSION
BLOCK_DIMENSION = source.BLOCK_DIMENSION
ACTION_DIMENSION = source.ACTION_DIMENSION
PARAMETER_BOUND = source.PARAMETER_BOUND
ROTATION_SCALE = source.ROTATION_SCALE
PHASE_GAIN = math.pi / (2 * PARAMETER_BOUND)
GENERATOR_VERSION = "rotating-pair-oracle-cayley-paley24-v1"
TransportChart = source.TransportChart
graphs = source.graphs
sample_controls = source.sample_controls
rotation_matrix = source.rotation_matrix
state_rotation = source.state_rotation
paley_hadamard = source.paley_hadamard
action_rotation = source.action_rotation
encode_controls = source.encode_controls
decode_actions = source.decode_actions


def _coordinates(value: Tensor, dimension: int, name: str) -> Tensor:
    if value.ndim < 1 or value.shape[-1] != dimension or not value.is_floating_point():
        raise ValueError(f"{name} must be floating point with final dimension {dimension}")
    return value.to(torch.float64)


def _scale(rho: float) -> float:
    if not math.isfinite(rho) or not 0 < rho < 1:
        raise ValueError("rho must be finite and strictly between zero and one")
    return math.sqrt(1 - rho * rho)


def _phase(parameters: Tensor, controls: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    angle = math.pi * controls[..., 18:24]
    cosine, sine = angle.cos(), angle.sin()
    phase = PHASE_GAIN * (cosine * parameters + sine * parameters.roll(3, -1))
    return phase, cosine, sine


def transition(
    states: Tensor,
    parameters: Tensor,
    actions: Tensor,
    *,
    rho: float = 0.7,
    rotation_scale: float = ROTATION_SCALE,
    transport_chart: TransportChart = "isotropic",
) -> Tensor:
    """Advance the unchanged Gaussian transport with equal-strength rotating pair phases."""
    scale = _scale(rho)
    states = _coordinates(states, DIMENSION, "states")
    parameters = _coordinates(parameters, DIMENSION, "parameters")
    controls = decode_actions(actions)
    phase, _, _ = _phase(parameters, controls)
    rotation = state_rotation(
        actions, rotation_scale=rotation_scale, transport_chart=transport_chart
    )
    innovation = controls[..., 6:12] * phase.cos() + controls[..., 12:18] * phase.sin()
    return rho * (rotation @ states.unsqueeze(-1)).squeeze(-1) + scale * innovation


def simulate(
    initial_state: Tensor,
    parameters: Tensor,
    actions: Tensor,
    *,
    rho: float = 0.7,
    rotation_scale: float = ROTATION_SCALE,
    transport_chart: TransportChart = "isotropic",
) -> Tensor:
    """Return all frames while retaining both autoregressive parameter-gradient paths."""
    _scale(rho)
    initial_state = _coordinates(initial_state, DIMENSION, "initial_state")
    parameters = _coordinates(parameters, DIMENSION, "parameters")
    actions = _coordinates(actions, ACTION_DIMENSION, "actions")
    if actions.ndim < 2:
        raise ValueError("actions require a time axis: (...,T-1,24)")
    batch_shape = cast(
        torch.Size,
        torch.broadcast_shapes(  # pyright: ignore[reportUnknownMemberType]
            initial_state.shape[:-1], parameters.shape[:-1], actions.shape[:-2]
        ),
    )
    state = initial_state.expand(*batch_shape, DIMENSION)
    history = [state]
    for index in range(actions.shape[-2]):
        state = transition(
            state,
            parameters,
            actions[..., index, :],
            rho=rho,
            rotation_scale=rotation_scale,
            transport_chart=transport_chart,
        )
        history.append(state)
    return torch.stack(history, dim=-2)


def analytic_jacobians(
    states: Tensor,
    parameters: Tensor,
    actions: Tensor,
    *,
    rho: float = 0.7,
    rotation_scale: float = ROTATION_SCALE,
    transport_chart: TransportChart = "isotropic",
) -> TransitionJacobians:
    """Preserve public transport derivatives and replace only innovation derivatives."""
    scale = _scale(rho)
    states = _coordinates(states, DIMENSION, "states")
    parameters = _coordinates(parameters, DIMENSION, "parameters")
    controls = decode_actions(actions)
    batch_shape = cast(
        torch.Size,
        torch.broadcast_shapes(  # pyright: ignore[reportUnknownMemberType]
            states.shape[:-1], parameters.shape[:-1], controls.shape[:-1]
        ),
    )
    parameters = parameters.expand(*batch_shape, DIMENSION)
    controls = controls.expand(*batch_shape, ACTION_DIMENSION)
    phase, cosine, sine = _phase(parameters, controls)
    companion = scale * (-controls[..., 6:12] * phase.sin() + controls[..., 12:18] * phase.cos())
    phase_control = PHASE_GAIN * math.pi * (-sine * parameters + cosine * parameters.roll(3, -1))
    zero = torch.zeros(*batch_shape, 6, 6, dtype=torch.float64, device=states.device)
    innovation_jacobian = torch.cat(
        (
            zero,
            torch.diag_embed(scale * phase.cos()),
            torch.diag_embed(scale * phase.sin()),
            torch.diag_embed(companion * phase_control),
        ),
        -1,
    )
    identity = torch.eye(6, dtype=torch.float64, device=states.device)
    parameter_jacobian = (
        PHASE_GAIN
        * companion[..., :, None]
        * (cosine[..., :, None] * identity + sine[..., :, None] * identity.roll(3, 0))
    )
    # At theta=0 the frozen source innovation is eta*c. Remove precisely that
    # derivative, including its public action-map chain, and add the new one.
    # State and rotation-action derivatives remain the frozen public reference.
    reference = source.analytic_jacobians(
        states,
        torch.zeros_like(parameters),
        actions,
        rho=rho,
        rotation_scale=rotation_scale,
        transport_chart=transport_chart,
    )
    old_mixing = transport.action_rotation(device=states.device)
    new_mixing = action_rotation(device=states.device)
    old_innovation = torch.cat((identity * 0, scale * identity, identity * 0, identity * 0), -1)
    correction = innovation_jacobian @ new_mixing.T - (
        old_innovation @ old_mixing @ old_mixing @ new_mixing.T
    )
    return TransitionJacobians(
        reference.state, reference.action + correction.unsqueeze(-2), parameter_jacobian
    )
