"""Gaussian state dynamics with smooth isotropic Cayley transport.

Raw controls are [gamma1(3), gamma2(3), c(6), d(6), u(6)]. The first eighteen
are independent standard Gaussians; u is uniform on (-1,1). Parameters influence
only coordinatewise innovations. The fixed rotation scale matches the original
rotation family's mean trace by quadrature, not learned-model performance.
"""

import math
from typing import Literal, cast

import torch
from torch import Tensor

from scjepa.data.types import GraphMasks, TransitionJacobians

DIMENSION = 6
BLOCK_DIMENSION = 3
ACTION_DIMENSION = 24
PARAMETER_BOUND = 0.75
ROTATION_SCALE = 0.23865401979977377
GENERATOR_VERSION = "isotropic-oracle-cayley-innovation-v1"
type TransportChart = Literal["isotropic"]


def _coordinates(value: Tensor, dimension: int, name: str) -> Tensor:
    if value.ndim < 1 or value.shape[-1] != dimension or not value.is_floating_point():
        raise ValueError(f"{name} must be floating point with final dimension {dimension}")
    return value.to(dtype=torch.float64)


def _innovation_scale(rho: float) -> float:
    if not math.isfinite(rho) or not 0 < rho < 1:
        raise ValueError("rho must be finite and strictly between zero and one")
    return math.sqrt(1 - rho * rho)


def _mate(device: torch.device) -> Tensor:
    return (torch.arange(DIMENSION, device=device) + BLOCK_DIMENSION) % DIMENSION


def action_rotation(*, device: torch.device | str | None = None) -> Tensor:
    """Return the fixed dense orthogonal involution I24-J24/12."""
    return torch.eye(ACTION_DIMENSION, dtype=torch.float64, device=device) - (
        2 / ACTION_DIMENSION
    ) * torch.ones(ACTION_DIMENSION, ACTION_DIMENSION, dtype=torch.float64, device=device)


def encode_controls(controls: Tensor) -> Tensor:
    """Expose only the dense orthogonal mixing of the independent raw controls."""
    controls = _coordinates(controls, ACTION_DIMENSION, "controls")
    return controls @ action_rotation(device=controls.device).T


def decode_actions(actions: Tensor) -> Tensor:
    """Invert mixed actions for simulator and diagnostic reference use only."""
    actions = _coordinates(actions, ACTION_DIMENSION, "actions")
    return actions @ action_rotation(device=actions.device).T


def sample_controls(shape: tuple[int, ...], generator: torch.Generator) -> Tensor:
    """Draw the declared eighteen Gaussian and six uniform independent controls."""
    return torch.cat(
        (
            torch.randn(*shape, 18, dtype=torch.float64, generator=generator),
            2 * torch.rand(*shape, 6, dtype=torch.float64, generator=generator) - 1,
        ),
        dim=-1,
    )


def _skew(vector: Tensor) -> Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
        *vector.shape[:-1], 3, 3
    )


def rotation_matrix(gamma: Tensor, *, rotation_scale: float = ROTATION_SCALE) -> Tensor:
    """Return a globally smooth Cayley rotation with denominator at least one."""
    gamma = _coordinates(gamma, 3, "gamma")
    if not math.isfinite(rotation_scale) or rotation_scale <= 0:
        raise ValueError("rotation_scale must be finite and positive")
    vector = rotation_scale * gamma
    squared = vector.square().sum(-1)[..., None, None]
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device)
    outer = vector[..., :, None] * vector[..., None, :]
    return ((1 - squared) * identity + 2 * outer + 2 * _skew(vector)) / (1 + squared)


def _state_rotation(
    controls: Tensor, rotation_scale: float, transport_chart: TransportChart
) -> Tensor:
    if transport_chart != "isotropic":
        raise ValueError("This diagnostic has only the isotropic transport candidate")
    first = rotation_matrix(controls[..., :3], rotation_scale=rotation_scale)
    second = rotation_matrix(controls[..., 3:6], rotation_scale=rotation_scale)
    zero = torch.zeros_like(first)
    return torch.cat((torch.cat((first, zero), -1), torch.cat((zero, second), -1)), -2)


def state_rotation(
    actions: Tensor,
    *,
    rotation_scale: float = ROTATION_SCALE,
    transport_chart: TransportChart = "isotropic",
) -> Tensor:
    """Return the two independently sampled isotropic state-transport blocks."""
    return _state_rotation(decode_actions(actions), rotation_scale, transport_chart)


def graphs() -> GraphMasks:
    """Return the exact18 state,144 action and12 parameter dependency supports."""
    identity = torch.eye(BLOCK_DIMENSION, dtype=torch.bool)
    parameter = torch.cat((torch.cat((identity, identity), 1),) * 2, 0)
    block = torch.ones(BLOCK_DIMENSION, BLOCK_DIMENSION, dtype=torch.bool)
    zero = torch.zeros_like(block)
    state = torch.cat((torch.cat((block, zero), 1), torch.cat((zero, block), 1)), 0)
    return GraphMasks(
        state.unsqueeze(1),
        torch.ones(DIMENSION, 1, ACTION_DIMENSION, dtype=torch.bool),
        parameter,
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
    """Advance one state with parameter-independent transport and sparse innovations."""
    scale = _innovation_scale(rho)
    states = _coordinates(states, DIMENSION, "states")
    parameters = _coordinates(parameters, DIMENSION, "parameters")
    controls = decode_actions(actions)
    rotation = _state_rotation(controls, rotation_scale, transport_chart)
    rotated = (rotation @ states.unsqueeze(-1)).squeeze(-1)
    phase = parameters + controls[..., 18:24] * parameters[..., _mate(parameters.device)]
    innovation = controls[..., 6:12] * phase.cos() + controls[..., 12:18] * phase.sin()
    return rho * rotated + scale * innovation


def simulate(
    initial_state: Tensor,
    parameters: Tensor,
    actions: Tensor,
    *,
    rho: float = 0.7,
    rotation_scale: float = ROTATION_SCALE,
    transport_chart: TransportChart = "isotropic",
) -> Tensor:
    """Return (...,T,6), retaining all autoregressive parameter-gradient paths."""
    _innovation_scale(rho)
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


def _rotated_state_derivative(states: Tensor, gamma: Tensor, rotation_scale: float) -> Tensor:
    vector = rotation_scale * gamma
    rotation = rotation_matrix(gamma, rotation_scale=rotation_scale)
    rotated = (rotation @ states.unsqueeze(-1)).squeeze(-1)
    denominator = 1 + vector.square().sum(-1)[..., None, None]
    identity = torch.eye(3, dtype=states.dtype, device=states.device)
    derivative = (
        -2 * (states + rotated)[..., :, None] * vector[..., None, :]
        + 2 * (vector * states).sum(-1)[..., None, None] * identity
        + 2 * vector[..., :, None] * states[..., None, :]
        - 2 * _skew(states)
    )
    return rotation_scale * derivative / denominator


def analytic_jacobians(
    states: Tensor,
    parameters: Tensor,
    actions: Tensor,
    *,
    rho: float = 0.7,
    rotation_scale: float = ROTATION_SCALE,
    transport_chart: TransportChart = "isotropic",
) -> TransitionJacobians:
    """Differentiate the declared equations, including the observed-action mixing."""
    scale = _innovation_scale(rho)
    states = _coordinates(states, DIMENSION, "states")
    parameters = _coordinates(parameters, DIMENSION, "parameters")
    actions = _coordinates(actions, ACTION_DIMENSION, "actions")
    batch_shape = cast(
        torch.Size,
        torch.broadcast_shapes(  # pyright: ignore[reportUnknownMemberType]
            states.shape[:-1], parameters.shape[:-1], actions.shape[:-1]
        ),
    )
    states = states.expand(*batch_shape, DIMENSION)
    parameters = parameters.expand(*batch_shape, DIMENSION)
    controls = decode_actions(actions.expand(*batch_shape, ACTION_DIMENSION))
    rotation = _state_rotation(controls, rotation_scale, transport_chart)
    mate = _mate(parameters.device)
    u = controls[..., 18:24]
    phase = parameters + u * parameters[..., mate]
    companion = scale * (-controls[..., 6:12] * phase.sin() + controls[..., 12:18] * phase.cos())
    first = _rotated_state_derivative(states[..., :3], controls[..., :3], rotation_scale)
    second = _rotated_state_derivative(states[..., 3:], controls[..., 3:6], rotation_scale)
    zero = torch.zeros_like(first)
    derivative_state = torch.cat((torch.cat((first, zero), -1), torch.cat((zero, second), -1)), -2)
    control_jacobian = torch.cat(
        (
            rho * derivative_state,
            torch.diag_embed(scale * phase.cos()),
            torch.diag_embed(scale * phase.sin()),
            torch.diag_embed(companion * parameters[..., mate]),
        ),
        -1,
    )
    identity = torch.eye(DIMENSION, dtype=states.dtype, device=states.device)
    parameter = companion[..., :, None] * (identity + u[..., :, None] * identity[mate])
    return TransitionJacobians(
        (rho * rotation).unsqueeze(-2),
        (control_jacobian @ action_rotation(device=actions.device)).unsqueeze(-2),
        parameter,
    )
