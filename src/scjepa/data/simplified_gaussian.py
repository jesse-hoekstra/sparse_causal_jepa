"""Six-state Gaussian-preserving dynamics with densely mixed observed actions.

The independent controls are [gamma1(2), gamma2(2), c(6), u(6)]. Only their
fixed orthogonal mixing is observed. Parameters are uniform, episode-constant,
and enter the angles directly; this family has no radial or tanh transformation.
Scientific simulation and derivative references use float64 and retain autograd.
"""

from typing import cast

import torch
from torch import Tensor

from scjepa.data.types import GraphMasks, TransitionJacobians

DIMENSION = 6
BLOCK_DIMENSION = 3
ACTION_DIMENSION = 16
PARAMETER_BOUND = 0.75
GENERATOR_VERSION = "simplified-gaussian-mixed-controls-paired-v1"


def _coordinates(value: Tensor, dimension: int, name: str) -> Tensor:
    if value.ndim < 1 or value.shape[-1] != dimension or not value.is_floating_point():
        raise ValueError(f"{name} must be floating point with final dimension {dimension}")
    return value.to(dtype=torch.float64)


def _mate(device: torch.device) -> Tensor:
    return (torch.arange(DIMENSION, device=device) + BLOCK_DIMENSION) % DIMENSION


def action_rotation(*, device: torch.device | str | None = None) -> Tensor:
    """Return the fixed symmetric orthogonal map V=I16-J16/8."""
    return torch.eye(ACTION_DIMENSION, dtype=torch.float64, device=device) - (
        2 / ACTION_DIMENSION
    ) * torch.ones(ACTION_DIMENSION, ACTION_DIMENSION, dtype=torch.float64, device=device)


def encode_controls(controls: Tensor) -> Tensor:
    """Mix raw controls into the only action coordinates exposed to the learner."""
    controls = _coordinates(controls, ACTION_DIMENSION, "controls")
    return controls @ action_rotation(device=controls.device).T


def decode_actions(actions: Tensor) -> Tensor:
    """Recover generator controls; V is its own inverse (reference use only)."""
    actions = _coordinates(actions, ACTION_DIMENSION, "actions")
    return actions @ action_rotation(device=actions.device).T


def _householder(vector: Tensor) -> Tensor:
    return torch.eye(BLOCK_DIMENSION, dtype=vector.dtype, device=vector.device) - (
        2 * vector[..., :, None] * vector[..., None, :] / vector.square().sum(-1)[..., None, None]
    )


def _block_rotation(gamma: Tensor) -> Tensor:
    vector = torch.cat((torch.ones_like(gamma[..., :1]), gamma), dim=-1)
    diagonal = torch.arange(1, BLOCK_DIMENSION + 1, dtype=gamma.dtype, device=gamma.device)
    return _householder(vector) @ _householder(vector * diagonal)


def _state_rotation(controls: Tensor) -> Tensor:
    first, second = _block_rotation(controls[..., :2]), _block_rotation(controls[..., 2:4])
    zero = torch.zeros_like(first)
    return torch.cat((torch.cat((first, zero), -1), torch.cat((zero, second), -1)), -2)


def state_rotation(actions: Tensor) -> Tensor:
    """Return the two 3-by-3 reflection products from decoded row controls."""
    return _state_rotation(decode_actions(actions))


def graphs() -> GraphMasks:
    """Return output-row masks with 18 state, 96 action, and 12 parameter edges."""
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


def transition(states: Tensor, parameters: Tensor, actions: Tensor) -> Tensor:
    """Apply one step to broadcastable (...,6) states/parameters and (...,16) actions."""
    states = _coordinates(states, DIMENSION, "states")
    parameters = _coordinates(parameters, DIMENSION, "parameters")
    controls = decode_actions(actions)
    b = (_state_rotation(controls) @ states.unsqueeze(-1)).squeeze(-1)
    phase = parameters + controls[..., 10:] * parameters[..., _mate(parameters.device)]
    return b * phase.cos() + controls[..., 4:10] * phase.sin()


def simulate(initial_state: Tensor, parameters: Tensor, actions: Tensor) -> Tensor:
    """Return (...,T,6), reusing the attached parameter across all T-1 transitions."""
    initial_state = _coordinates(initial_state, DIMENSION, "initial_state")
    parameters = _coordinates(parameters, DIMENSION, "parameters")
    actions = _coordinates(actions, ACTION_DIMENSION, "actions")
    if actions.ndim < 2:
        raise ValueError("actions require a time axis: (...,T-1,16)")
    batch_shape = cast(
        torch.Size,
        torch.broadcast_shapes(  # pyright: ignore[reportUnknownMemberType]
            initial_state.shape[:-1], parameters.shape[:-1], actions.shape[:-2]
        ),
    )
    state = initial_state.expand(*batch_shape, DIMENSION)
    history = [state]
    for index in range(actions.shape[-2]):
        state = transition(state, parameters, actions[..., index, :])
        history.append(state)
    return torch.stack(history, dim=-2)


def _householder_derivatives(vector: Tensor) -> Tensor:
    identity = torch.eye(BLOCK_DIMENSION, dtype=vector.dtype, device=vector.device)
    norm_squared = vector.square().sum(-1)[..., None, None, None]
    cross = torch.einsum("ki,...j->...kij", identity, vector) + torch.einsum(
        "...i,kj->...kij", vector, identity
    )
    outer = vector[..., :, None] * vector[..., None, :]
    return (
        -2 * cross / norm_squared
        + 4 * vector[..., :, None, None] * outer[..., None, :, :] / norm_squared.square()
    )


def _rotated_state_derivative(states: Tensor, gamma: Tensor) -> Tensor:
    vector = torch.cat((torch.ones_like(gamma[..., :1]), gamma), dim=-1)
    diagonal = torch.arange(1, BLOCK_DIMENSION + 1, dtype=gamma.dtype, device=gamma.device)
    scaled = vector * diagonal
    derivative = (
        _householder_derivatives(vector)[..., 1:, :, :] @ _householder(scaled)[..., None, :, :]
    )
    derivative = derivative + _householder(vector)[..., None, :, :] @ (
        _householder_derivatives(scaled)[..., 1:, :, :] * diagonal[1:, None, None]
    )
    return (derivative @ states[..., None, :, None]).squeeze(-1).transpose(-1, -2)


def analytic_jacobians(states: Tensor, parameters: Tensor, actions: Tensor) -> TransitionJacobians:
    """Differentiate exact equations without autodiff, including observed-action mixing.

    State/action axes are (...,output,lag,input), with one lag. Parameters have
    axes (...,output,parameter); zero derivatives at a point do not delete edges.
    """
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
    rotation = _state_rotation(controls)
    b = (rotation @ states.unsqueeze(-1)).squeeze(-1)
    mate = _mate(parameters.device)
    u = controls[..., 10:]
    phase = parameters + u * parameters[..., mate]
    companion = -b * phase.sin() + controls[..., 4:10] * phase.cos()
    first = _rotated_state_derivative(states[..., :3], controls[..., :2])
    second = _rotated_state_derivative(states[..., 3:], controls[..., 2:4])
    zero = torch.zeros_like(first)
    derivative_b = torch.cat((torch.cat((first, zero), -1), torch.cat((zero, second), -1)), -2)
    control_jacobian = torch.cat(
        (
            phase.cos()[..., :, None] * derivative_b,
            torch.diag_embed(phase.sin()),
            torch.diag_embed(companion * parameters[..., mate]),
        ),
        -1,
    )
    identity = torch.eye(DIMENSION, dtype=states.dtype, device=states.device)
    parameter = companion[..., :, None] * (identity + u[..., :, None] * identity[mate])
    return TransitionJacobians(
        (phase.cos()[..., :, None] * rotation).unsqueeze(-2),
        (control_jacobian @ action_rotation(device=actions.device)).unsqueeze(-2),
        parameter,
    )
