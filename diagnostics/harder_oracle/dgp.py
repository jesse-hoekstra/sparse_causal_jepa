"""Gaussian dynamics whose parameters affect innovations, not state contraction.

Independent controls have layout ``[gamma1(2), gamma2(2), c(6), d(6), u(6)]``.
The first sixteen controls are standard Gaussian and the last six are uniform
on (-1, 1). Only their fixed, dense orthogonal mixing is observed. For fixed
theta, standard Gaussian state marginals are preserved exactly in population.
This is an explicitly changed diagnostic DGP; the original generator is intact.
"""

import math
from typing import Literal, cast

import torch
from torch import Tensor

from scjepa.data.types import GraphMasks, TransitionJacobians

DIMENSION = 6
BLOCK_DIMENSION = 3
ACTION_DIMENSION = 22
PARAMETER_BOUND = 0.75
GENERATOR_VERSION = "harder-oracle-innovation-only-parameters-v1"
type TransportChart = Literal["physical", "mixed"]


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
    """Return the fixed dense orthogonal involution I22 - J22/11."""
    return torch.eye(ACTION_DIMENSION, dtype=torch.float64, device=device) - (
        2 / ACTION_DIMENSION
    ) * torch.ones(ACTION_DIMENSION, ACTION_DIMENSION, dtype=torch.float64, device=device)


def encode_controls(controls: Tensor) -> Tensor:
    """Expose the dense orthogonal mixing of raw controls as observed actions."""
    controls = _coordinates(controls, ACTION_DIMENSION, "controls")
    return controls @ action_rotation(device=controls.device).T


def decode_actions(actions: Tensor) -> Tensor:
    """Invert observed actions for the simulator and diagnostic references only."""
    actions = _coordinates(actions, ACTION_DIMENSION, "actions")
    return actions @ action_rotation(device=actions.device).T


def sample_controls(shape: tuple[int, ...], generator: torch.Generator) -> Tensor:
    """Draw independent float64 controls with the declared population law."""
    return torch.cat(
        (
            torch.randn(*shape, 16, dtype=torch.float64, generator=generator),
            2 * torch.rand(*shape, 6, dtype=torch.float64, generator=generator) - 1,
        ),
        dim=-1,
    )


def _householder(vector: Tensor) -> Tensor:
    return torch.eye(BLOCK_DIMENSION, dtype=vector.dtype, device=vector.device) - (
        2 * vector[..., :, None] * vector[..., None, :] / vector.square().sum(-1)[..., None, None]
    )


def _block_rotation(gamma: Tensor) -> Tensor:
    vector = torch.cat((torch.ones_like(gamma[..., :1]), gamma), dim=-1)
    diagonal = torch.arange(1, BLOCK_DIMENSION + 1, dtype=gamma.dtype, device=gamma.device)
    return _householder(vector) @ _householder(vector * diagonal)


def transport_rotation(*, device: torch.device | str | None = None) -> Tensor:
    """Return Q=blockdiag(H,H), H=I3-2J3/3, an exact within-row mixing."""
    householder = torch.eye(3, dtype=torch.float64, device=device) - (2 / 3) * torch.ones(
        3, 3, dtype=torch.float64, device=device
    )
    zero = torch.zeros_like(householder)
    return torch.cat((torch.cat((householder, zero), -1), torch.cat((zero, householder), -1)), -2)


def _state_rotation(controls: Tensor, transport_chart: TransportChart) -> Tensor:
    first, second = _block_rotation(controls[..., :2]), _block_rotation(controls[..., 2:4])
    zero = torch.zeros_like(first)
    rotation = torch.cat((torch.cat((first, zero), -1), torch.cat((zero, second), -1)), -2)
    if transport_chart == "physical":
        return rotation
    if transport_chart != "mixed":
        raise ValueError("transport_chart must be physical or mixed")
    mixing = transport_rotation(device=controls.device)
    return mixing.T @ rotation @ mixing


def state_rotation(actions: Tensor, *, transport_chart: TransportChart = "mixed") -> Tensor:
    """Return blockwise rotations in physical or deliberately mixed transport axes."""
    return _state_rotation(decode_actions(actions), transport_chart)


def graphs() -> GraphMasks:
    """Return true supports: eighteen state,132 action and twelve parameter edges."""
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
    transport_chart: TransportChart = "mixed",
) -> Tensor:
    """Apply rho*R*s + sqrt(1-rho²)*(c*cos(phi)+d*sin(phi))."""
    scale = _innovation_scale(rho)
    states = _coordinates(states, DIMENSION, "states")
    parameters = _coordinates(parameters, DIMENSION, "parameters")
    controls = decode_actions(actions)
    rotated = (_state_rotation(controls, transport_chart) @ states.unsqueeze(-1)).squeeze(-1)
    phase = parameters + controls[..., 16:22] * parameters[..., _mate(parameters.device)]
    innovation = controls[..., 4:10] * phase.cos() + controls[..., 10:16] * phase.sin()
    return rho * rotated + scale * innovation


def simulate(
    initial_state: Tensor,
    parameters: Tensor,
    actions: Tensor,
    *,
    rho: float = 0.7,
    transport_chart: TransportChart = "mixed",
) -> Tensor:
    """Return (...,T,6), preserving gradients through each autoregressive step."""
    _innovation_scale(rho)
    initial_state = _coordinates(initial_state, DIMENSION, "initial_state")
    parameters = _coordinates(parameters, DIMENSION, "parameters")
    actions = _coordinates(actions, ACTION_DIMENSION, "actions")
    if actions.ndim < 2:
        raise ValueError("actions require a time axis: (...,T-1,22)")
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
            state, parameters, actions[..., index, :], rho=rho, transport_chart=transport_chart
        )
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


def analytic_jacobians(
    states: Tensor,
    parameters: Tensor,
    actions: Tensor,
    *,
    rho: float = 0.7,
    transport_chart: TransportChart = "mixed",
) -> TransitionJacobians:
    """Return exact equation derivatives, including the dense observed-action map."""
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
    rotation = _state_rotation(controls, transport_chart)
    mate = _mate(parameters.device)
    u = controls[..., 16:22]
    phase = parameters + u * parameters[..., mate]
    companion = scale * (-controls[..., 4:10] * phase.sin() + controls[..., 10:16] * phase.cos())
    if transport_chart == "mixed":
        mixing = transport_rotation(device=states.device)
        derivative_inputs = states @ mixing.T
    else:
        mixing = torch.eye(DIMENSION, dtype=states.dtype, device=states.device)
        derivative_inputs = states
    first = _rotated_state_derivative(derivative_inputs[..., :3], controls[..., :2])
    second = _rotated_state_derivative(derivative_inputs[..., 3:], controls[..., 2:4])
    zero = torch.zeros_like(first)
    derivative_state = mixing.T @ torch.cat(
        (torch.cat((first, zero), -1), torch.cat((zero, second), -1)), -2
    )
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
