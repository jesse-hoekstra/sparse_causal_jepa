"""Deterministic displacement-only spring dynamics from manuscript Section 10.4.

These are simulation and evaluation utilities, never learner inputs. Each action
``a[t]`` belongs to transition ``z[t] -> z[t+1]``. The first action is an unused
initialization placeholder: ``z[1] = z[0] + dt * v[0]``. Valid recurrence sources
are ``t >= 1``; two input lags are ordered previous, current.
"""

import math

import torch
from torch import Tensor

from scjepa.data.types import GraphMasks, TransitionJacobians


def stiffness_matrix(
    *, dtype: torch.dtype = torch.float64, device: torch.device | str = "cpu"
) -> Tensor:
    """Return the fixed five-pair stiffness matrix, with output-coordinate rows."""
    pair_ids = torch.arange(10, device=device) // 2
    paired = (pair_ids[:, None] == pair_ids[None, :]).to(dtype)
    return 3 * torch.eye(10, dtype=dtype, device=device) - paired


def _validate_constants(dt: float, gamma: float) -> None:
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(gamma) or gamma < 0:
        raise ValueError("dt must be positive and gamma nonnegative, both finite")


def _validate_vector(value: Tensor, size: int, name: str) -> None:
    if value.ndim < 1 or value.shape[-1] != size:
        raise ValueError(f"{name} must have last dimension {size}")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use float32 or float64")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def forces(
    previous: Tensor, current: Tensor, action: Tensor, *, dt: float = 0.1, gamma: float = 0.2
) -> Tensor:
    """Calculate Eq. (101) forces; each pair receives one shared action.

    Inputs have final dimensions 10, 10, and 5 and may have broadcastable leading
    dimensions. Force evaluation does not depend on the inverse masses.
    """
    _validate_constants(dt, gamma)
    _validate_vector(previous, 10, "previous")
    _validate_vector(current, 10, "current")
    _validate_vector(action, 5, "action")
    if previous.dtype != current.dtype or action.dtype != current.dtype:
        raise ValueError("state and action dtypes must agree")
    if previous.device != current.device or action.device != current.device:
        raise ValueError("state and action devices must agree")
    stiffness = stiffness_matrix(dtype=current.dtype, device=current.device)
    return (
        -(current @ stiffness.T)
        - gamma * (current - previous) / dt
        + action.repeat_interleave(2, dim=-1)
    )


def transition(
    previous: Tensor,
    current: Tensor,
    parameters: Tensor,
    action: Tensor,
    *,
    dt: float = 0.1,
    gamma: float = 0.2,
) -> Tensor:
    """Advance one two-lag transition using constant inverse-mass coordinates."""
    _validate_vector(parameters, 10, "parameters")
    if parameters.dtype != current.dtype or parameters.device != current.device:
        raise ValueError("parameter and state dtype/device must agree")
    if not bool((parameters > 0).all()):
        raise ValueError("inverse masses must be positive")
    force = forces(previous, current, action, dt=dt, gamma=gamma)
    return 2 * current - previous + dt**2 * parameters * force


def simulate(
    initial_position: Tensor,
    initial_velocity: Tensor,
    parameters: Tensor,
    actions: Tensor,
    *,
    dt: float = 0.1,
    gamma: float = 0.2,
) -> Tensor:
    """Simulate one explicit episode, returning ``(T,10)`` displacements.

    Positions, velocities, and inverse masses each have shape ``(10,)``. Actions
    have shape ``(T-1,5)`` with ``T >= 2``. ``actions[0]`` is never applied;
    generated datasets retain the sampled value in this unused initialization slot.
    The same parameter tensor is used on every subsequent transition.
    """
    _validate_constants(dt, gamma)
    for value, name in (
        (initial_position, "initial_position"),
        (initial_velocity, "initial_velocity"),
        (parameters, "parameters"),
    ):
        _validate_vector(value, 10, name)
        if value.shape != (10,):
            raise ValueError(f"{name} must have shape (10,)")
        if value.dtype != initial_position.dtype or value.device != initial_position.device:
            raise ValueError("all simulation inputs must share dtype and device")
    _validate_vector(actions, 5, "actions")
    if actions.ndim != 2 or actions.shape[0] < 1:
        raise ValueError("actions must have shape (T-1,5), with T >= 2")
    if actions.dtype != initial_position.dtype or actions.device != initial_position.device:
        raise ValueError("all simulation inputs must share dtype and device")
    if not bool((parameters > 0).all()):
        raise ValueError("inverse masses must be positive")
    history = [initial_position, initial_position + dt * initial_velocity]
    for source in range(1, actions.shape[0]):
        history.append(
            transition(history[-2], history[-1], parameters, actions[source], dt=dt, gamma=gamma)
        )
    return torch.stack(history)


def recover_parameters(
    states: Tensor,
    actions: Tensor,
    *,
    dt: float = 0.1,
    gamma: float = 0.2,
    denominator_tolerance: float = 1e-12,
) -> Tensor:
    """Recover inverse masses from Eq. (105), as a float64 reference calculation.

    A context of ``K >= 3`` frames uses exactly source indices ``1..K-2``.
    ``actions`` must have shape ``(K-1,5)``; the unused index zero is ignored.
    Coordinates whose squared-force sums are at most ``denominator_tolerance``
    times ``max(1, largest squared-force sum)`` raise ``ValueError`` explicitly.
    This absolute/relative numerical conditioning threshold is an implementation
    choice, not a change to the sampling distribution or an identifiability test.
    Casting float32 trajectories to float64 cannot recover precision already lost.
    """
    _validate_constants(dt, gamma)
    _validate_vector(states, 10, "states")
    _validate_vector(actions, 5, "actions")
    if states.ndim != 2 or states.shape[0] < 3:
        raise ValueError("parameter recovery requires (K,10) states with K >= 3")
    if actions.shape != (states.shape[0] - 1, 5):
        raise ValueError("actions must have shape (K-1,5) for the supplied context")
    if actions.device != states.device:
        raise ValueError("states and actions must share a device")
    if not math.isfinite(denominator_tolerance) or denominator_tolerance <= 0:
        raise ValueError("denominator_tolerance must be positive and finite")
    states64, actions64 = states.to(torch.float64), actions.to(torch.float64)
    force = forces(states64[:-2], states64[1:-1], actions64[1:], dt=dt, gamma=gamma)
    force_energy = force.square().sum(dim=0)
    threshold = denominator_tolerance * force_energy.max().clamp_min(1.0)
    uninformative = force_energy <= threshold
    if not bool(torch.isfinite(force_energy).all()) or bool(uninformative.any()):
        invalid = uninformative | ~torch.isfinite(force_energy)
        coordinates = [index for index in range(10) if bool(invalid[index])]
        raise ValueError(f"uninformative or poorly conditioned force denominators: {coordinates}")
    acceleration = states64[2:] - 2 * states64[1:-1] + states64[:-2]
    return (force * acceleration).sum(dim=0) / (dt**2 * force_energy)


def graphs() -> GraphMasks:
    """Return structural boolean graphs in output, oldest-to-newest lag, input order.

    Lag-union state/action/parameter edge counts are 20, 10, and 10; summing
    individual state-lag masks instead gives 30. Masks describe the paper's
    parameter range and constants, irrespective of derivatives at one point.
    """
    identity = torch.eye(10, dtype=torch.bool)
    pair_ids = torch.arange(10) // 2
    pairs = pair_ids[:, None] == pair_ids[None, :]
    shared_actions = torch.eye(5, dtype=torch.bool).repeat_interleave(2, dim=0)
    return GraphMasks(
        state=torch.stack((identity, pairs), dim=1),
        action=torch.stack((torch.zeros_like(shared_actions), shared_actions), dim=1),
        parameter=identity,
    )


def analytic_jacobians(
    previous: Tensor,
    current: Tensor,
    parameters: Tensor,
    action: Tensor,
    *,
    dt: float = 0.1,
    gamma: float = 0.2,
) -> TransitionJacobians:
    """Return Eq. (106) and action/parameter derivatives for one explicit transition.

    Shapes are ``state=(10,2,10)``, ``action=(10,2,5)``, and
    ``parameter=(10,10)``. The oldest action derivative is exactly zero.
    """
    for value, size, name in (
        (previous, 10, "previous"),
        (current, 10, "current"),
        (parameters, 10, "parameters"),
        (action, 5, "action"),
    ):
        if value.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},)")
    # Share the transition's physical input validation without changing the inputs.
    transition(previous, current, parameters, action, dt=dt, gamma=gamma)
    identity = torch.eye(10, dtype=current.dtype, device=current.device)
    diagonal = torch.diag(parameters)
    stiffness = stiffness_matrix(dtype=current.dtype, device=current.device)
    previous_jacobian = -identity + dt * gamma * diagonal
    current_jacobian = 2 * identity - dt * gamma * diagonal - dt**2 * diagonal @ stiffness
    action_jacobian = (
        dt**2
        * parameters[:, None]
        * torch.eye(5, dtype=current.dtype, device=current.device).repeat_interleave(2, dim=0)
    )
    parameter_jacobian = dt**2 * torch.diag(forces(previous, current, action, dt=dt, gamma=gamma))
    return TransitionJacobians(
        state=torch.stack((previous_jacobian, current_jacobian), dim=1),
        action=torch.stack((torch.zeros_like(action_jacobian), action_jacobian), dim=1),
        parameter=parameter_jacobian,
    )


__all__ = [
    "analytic_jacobians",
    "forces",
    "graphs",
    "recover_parameters",
    "simulate",
    "stiffness_matrix",
    "transition",
]
