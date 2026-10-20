"""Nonlinear controlled dynamics preserving independent standard Gaussian states.

For each fixed parameter, an action-dependent orthogonal state rotation and
coordinatewise radial rotations preserve the product Gaussian distribution.
This is a separate, versioned DGP; it does not change ``controlled_vector``.
All scientific calculations use float64 and retain autograd. No sampling,
normalization, clipping, conditioning, or parameter labels enter this mechanism.
"""

from typing import cast

import torch
from torch import Tensor

from scjepa.data.controlled_vector import parameter_mask, state_mask
from scjepa.data.types import GraphMasks, TransitionJacobians

DIMENSION = 10
DELTA = 0.1
KAPPA = 0.5
GENERATOR_VERSION = "independent-gaussian-raw-householder-paired-v1"


def _coordinates(value: Tensor, name: str) -> Tensor:
    if value.ndim < 1 or value.shape[-1] != DIMENSION or not value.is_floating_point():
        raise ValueError(f"{name} must be floating point with final dimension {DIMENSION}")
    return value.to(dtype=torch.float64)


def _mate(device: torch.device) -> Tensor:
    return (torch.arange(DIMENSION, device=device) + 5) % DIMENSION


def _householder(vector: Tensor) -> Tensor:
    identity = torch.eye(5, dtype=vector.dtype, device=vector.device)
    return (
        identity
        - 2 * vector[..., :, None] * vector[..., None, :] / vector.square().sum(-1)[..., None, None]
    )


def _block_rotation(actions: Tensor) -> Tensor:
    vector = torch.cat((torch.ones_like(actions[..., :1]), actions), dim=-1)
    diagonal = torch.arange(1, 6, dtype=actions.dtype, device=actions.device)
    return _householder(vector) @ _householder(vector * diagonal)


def action_rotation(*, device: torch.device | str | None = None) -> Tensor:
    """Return the dense orthogonal action map V = I10 - J10/5."""
    return torch.eye(DIMENSION, dtype=torch.float64, device=device) - 0.2 * torch.ones(
        DIMENSION, DIMENSION, dtype=torch.float64, device=device
    )


def state_rotation(actions: Tensor) -> Tensor:
    """Return block-orthogonal U(a), using raw action coordinates 1 through 8."""
    actions = _coordinates(actions, "actions")
    first = _block_rotation(actions[..., :4])
    second = _block_rotation(actions[..., 4:8])
    zero = torch.zeros_like(first)
    return torch.cat((torch.cat((first, zero), dim=-1), torch.cat((zero, second), dim=-1)), dim=-2)


def graphs() -> GraphMasks:
    """Return output-row masks with 50 state, 100 action, and 20 parameter edges."""
    return GraphMasks(
        state=state_mask().unsqueeze(1),
        action=torch.ones(DIMENSION, 1, DIMENSION, dtype=torch.bool),
        parameter=parameter_mask("complementary"),
    )


def _radial_terms(
    states: Tensor, parameters: Tensor, actions: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    rotation = state_rotation(actions)
    b = (rotation @ states.unsqueeze(-1)).squeeze(-1)
    c = actions @ action_rotation(device=actions.device).T
    radius_squared = b.square() + c.square()
    h = KAPPA * radius_squared / (1 + radius_squared)
    alpha = DELTA * parameters.tanh()
    phase = alpha + h * alpha[..., _mate(parameters.device)]
    return rotation, b, c, radius_squared, h, alpha, phase


def transition(states: Tensor, parameters: Tensor, actions: Tensor) -> Tensor:
    """Apply the verified radial transition to broadcastable ``(...,10)`` inputs."""
    _, b, c, _, _, _, phase = _radial_terms(
        _coordinates(states, "states"),
        _coordinates(parameters, "parameters"),
        _coordinates(actions, "actions"),
    )
    return b * phase.cos() + c * phase.sin()


def simulate(initial_state: Tensor, parameters: Tensor, actions: Tensor) -> Tensor:
    """Return ``(...,T,10)``; action t generates t+1 with the same parameter."""
    initial_state = _coordinates(initial_state, "initial_state")
    parameters = _coordinates(parameters, "parameters")
    actions = _coordinates(actions, "actions")
    if actions.ndim < 2:
        raise ValueError("actions require a time axis: (...,T-1,10)")
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
    """Differentiate a reflection, with axes (...,input,output,state)."""
    identity = torch.eye(5, dtype=vector.dtype, device=vector.device)
    norm_squared = vector.square().sum(-1)[..., None, None, None]
    cross = torch.einsum("ki,...j->...kij", identity, vector) + torch.einsum(
        "...i,kj->...kij", vector, identity
    )
    outer = vector[..., :, None] * vector[..., None, :]
    return (
        -2 * cross / norm_squared
        + 4 * vector[..., :, None, None] * outer[..., None, :, :] / norm_squared.square()
    )


def _rotated_state_action_derivative(states: Tensor, actions: Tensor) -> Tensor:
    vector = torch.cat((torch.ones_like(actions[..., :1]), actions), dim=-1)
    diagonal = torch.arange(1, 6, dtype=actions.dtype, device=actions.device)
    scaled = vector * diagonal
    derivative = (
        _householder_derivatives(vector)[..., 1:, :, :] @ _householder(scaled)[..., None, :, :]
    )
    derivative = derivative + _householder(vector)[..., None, :, :] @ (
        _householder_derivatives(scaled)[..., 1:, :, :] * diagonal[1:, None, None]
    )
    return (derivative @ states[..., None, :, None]).squeeze(-1).transpose(-1, -2)


def analytic_jacobians(states: Tensor, parameters: Tensor, actions: Tensor) -> TransitionJacobians:
    """Return exact derivatives, including the action dependence of both reflections.

    State/action axes are ``(...,output,lag,input)`` with one lag; parameter
    axes are ``(...,output,parameter)``. This reference utility uses no autodiff.
    """
    states, parameters, actions = cast(
        tuple[Tensor, Tensor, Tensor],
        torch.broadcast_tensors(  # pyright: ignore[reportUnknownMemberType]
            _coordinates(states, "states"),
            _coordinates(parameters, "parameters"),
            _coordinates(actions, "actions"),
        ),
    )
    rotation, b, c, radius_squared, h, alpha, phase = _radial_terms(states, parameters, actions)
    companion = -b * phase.sin() + c * phase.cos()
    h_prime = KAPPA / (1 + radius_squared).square()
    mate = _mate(parameters.device)
    d_b = phase.cos() + 2 * b * h_prime * alpha[..., mate] * companion
    d_c = phase.sin() + 2 * c * h_prime * alpha[..., mate] * companion
    state = d_b[..., :, None] * rotation
    first = _rotated_state_action_derivative(states[..., :5], actions[..., :4])
    second = _rotated_state_action_derivative(states[..., 5:], actions[..., 4:8])
    zeros_four = torch.zeros_like(first)
    zeros_two = torch.zeros_like(first[..., :2])
    derivative_b = torch.cat(
        (
            torch.cat((first, zeros_four, zeros_two), dim=-1),
            torch.cat((zeros_four, second, zeros_two), dim=-1),
        ),
        dim=-2,
    )
    action = d_b[..., :, None] * derivative_b + d_c[..., :, None] * action_rotation(
        device=actions.device
    )
    identity = torch.eye(DIMENSION, dtype=states.dtype, device=states.device)
    alpha_prime = DELTA * (1 - parameters.tanh().square())
    parameter = (
        companion[..., :, None]
        * (identity + h[..., :, None] * identity[mate])
        * alpha_prime[..., None, :]
    )
    return TransitionJacobians(state.unsqueeze(-2), action.unsqueeze(-2), parameter)
