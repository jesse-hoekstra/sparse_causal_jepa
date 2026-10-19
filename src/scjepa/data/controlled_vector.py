"""Controlled vector process with stronger action-dependent coupling.

The graph setup follows the manuscript; the coefficients implement the revised
four-frame experiment documented in ``docs/data_generation.md``.
Rows of every coefficient/Jacobian are output coordinates; columns are inputs.
Scientific functions use float64, retain autograd, and accept arbitrary batch
axes. Ground truth and recovery helpers are evaluation-only, never learner input.
"""

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, sqrt
from typing import Literal, cast

import torch
from torch import Tensor

from scjepa.data.types import GraphMasks, TransitionJacobians

ParameterGraph = Literal["dense", "complementary", "diagonal"]
DIMENSION = 10
STATE_COUPLING = 0.1
PARAMETER_COUPLING = 0.25
ACTION_SCALE = 1 / sqrt(DIMENSION)


def _coordinates(value: Tensor, name: str) -> Tensor:
    if value.ndim < 1 or value.shape[-1] != DIMENSION or not value.is_floating_point():
        raise ValueError(f"{name} must be floating point with final dimension {DIMENSION}")
    return value.to(dtype=torch.float64)


def _solve(matrix: Tensor, values: Tensor) -> Tensor:
    return cast(Tensor, torch.linalg.solve(matrix, values))  # pyright: ignore[reportUnknownMemberType]


def _matrix_rank(matrix: Tensor, *, atol: float, rtol: float) -> Tensor:
    return cast(
        Tensor,
        torch.linalg.matrix_rank(matrix, atol=atol, rtol=rtol),  # pyright: ignore[reportUnknownMemberType]
    )


def state_mask(*, device: torch.device | str | None = None) -> Tensor:
    """Return H = diag(J5, J5), with output rows and input columns."""
    blocks = torch.arange(DIMENSION, device=device) // 5
    return blocks[:, None] == blocks[None, :]


def parameter_mask(variant: ParameterGraph, *, device: torch.device | str | None = None) -> Tensor:
    """Return the named parameter graph, including its diagonal."""
    if variant == "dense":
        return torch.ones(DIMENSION, DIMENSION, device=device, dtype=torch.bool)
    if variant == "complementary":
        coordinates = torch.arange(DIMENSION, device=device) % 5
        return coordinates[:, None] == coordinates[None, :]
    if variant == "diagonal":
        return torch.eye(DIMENSION, device=device, dtype=torch.bool)
    raise ValueError(f"unknown parameter graph {variant!r}")


def graphs(variant: ParameterGraph) -> GraphMasks:
    """Return structural masks, including the length-one oldest-first lag axis."""
    return GraphMasks(
        state=state_mask().unsqueeze(1),
        action=torch.ones(DIMENSION, 1, DIMENSION, dtype=torch.bool),
        parameter=parameter_mask(variant),
    )


def coefficient_matrices(actions: Tensor, variant: ParameterGraph) -> tuple[Tensor, Tensor]:
    """Build A(a) and B_G(a) for actions ``(...,10)`` in float64.

    ``F[i,j] = (1 + a[i]**2) * (1 + a[j]**3)`` and the positive parameter
    gain is ``exp(sum(a) / sqrt(10))``. Outputs have shape ``(...,10,10)``.
    """
    action = _coordinates(actions, "actions")
    identity = torch.eye(DIMENSION, dtype=action.dtype, device=action.device)
    interactions = (1 + action.square()).unsqueeze(-1) * (1 + action.pow(3)).unsqueeze(-2)
    state = 0.5 * identity + STATE_COUPLING * (state_mask(device=action.device) * interactions)
    parameter = (ACTION_SCALE * action.sum(-1)).exp()[..., None, None] * identity
    parameter = parameter + PARAMETER_COUPLING * (
        parameter_mask(variant, device=action.device) * interactions
    )
    return state, parameter


def transition(
    states: Tensor,
    parameters: Tensor,
    actions: Tensor,
    variant: ParameterGraph,
) -> Tensor:
    """Apply one deterministic transition to broadcastable ``(...,10)`` inputs."""
    states = _coordinates(states, "states")
    parameters = _coordinates(parameters, "parameters")
    state_matrix, parameter_matrix = coefficient_matrices(actions, variant)
    return (state_matrix @ states.unsqueeze(-1)).squeeze(-1) + (
        parameter_matrix @ parameters.unsqueeze(-1)
    ).squeeze(-1)


def simulate(
    initial_state: Tensor,
    parameters: Tensor,
    actions: Tensor,
    variant: ParameterGraph,
) -> Tensor:
    """Return ``(...,T,10)`` from initial states and ``(...,T-1,10)`` actions.

    Action ``t`` generates frame ``t+1``. The same supplied parameter tensor
    remains attached and is reused at every transition. No random draws occur.
    """
    initial_state = _coordinates(initial_state, "initial_state")
    parameters = _coordinates(parameters, "parameters")
    actions = _coordinates(actions, "actions")
    if actions.ndim < 2:
        raise ValueError("actions require a time axis: (...,T-1,10)")
    parameter_mask(variant)
    batch_shape = cast(
        torch.Size,
        torch.broadcast_shapes(  # pyright: ignore[reportUnknownMemberType]
            initial_state.shape[:-1], parameters.shape[:-1], actions.shape[:-2]
        ),
    )
    state = initial_state.expand(*batch_shape, DIMENSION)
    history = [state]
    for index in range(actions.shape[-2]):
        state = transition(state, parameters, actions[..., index, :], variant)
        history.append(state)
    return torch.stack(history, dim=-2)


class VectorObservationMap:
    """Fixed dense orthogonal mixing followed by ``u + 0.1*u**3``.

    A dedicated local CPU generator fixes W independently of trajectory draws.
    Supplying a persisted matrix avoids relying on QR reproducibility across
    library/platform versions. The returned matrix is a defensive copy.
    """

    def __init__(self, seed: int = 1729, matrix: Tensor | None = None) -> None:
        """Generate W once, or validate and copy a persisted orthogonal W."""
        if matrix is None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            raw = torch.randn(DIMENSION, DIMENSION, dtype=torch.float64, generator=generator)
            orthogonal, triangular = cast(
                tuple[Tensor, Tensor],
                torch.linalg.qr(raw),  # pyright: ignore[reportUnknownMemberType]
            )
            matrix = orthogonal * torch.where(triangular.diagonal() >= 0, 1.0, -1.0)
        if matrix.shape != (DIMENSION, DIMENSION) or not matrix.is_floating_point():
            raise ValueError("observation matrix must be floating point (10,10)")
        matrix = matrix.detach().to(device="cpu", dtype=torch.float64).clone()
        identity = torch.eye(DIMENSION, dtype=torch.float64)
        if not torch.isfinite(matrix).all() or not torch.allclose(
            matrix.T @ matrix, identity, atol=1e-12, rtol=1e-12
        ):
            raise ValueError("observation matrix must be finite and orthogonal")
        self._matrix = matrix
        self.seed = seed

    @property
    def matrix(self) -> Tensor:
        """Return the actual fixed float64 CPU matrix for provenance storage."""
        return self._matrix.clone()

    @property
    def fingerprint(self) -> str:
        """Return SHA-256 over shape/dtype-tagged little-endian matrix bytes."""
        payload = (
            b"controlled-vector-W:<f8:10,10:"
            + self._matrix.numpy().astype("<f8", copy=False).tobytes()
        )
        return sha256(payload).hexdigest()

    def forward(self, states: Tensor) -> Tensor:
        """Observe independent states ``(...,10)`` with the same fixed map."""
        states = _coordinates(states, "states")
        mixed = states @ self._matrix.to(device=states.device).T
        return mixed + 0.1 * mixed.pow(3)

    def inverse(self, observations: Tensor) -> Tensor:
        """Invert the cubic stably using its hyperbolic closed form.

        This avoids subtraction of nearly equal Cardano roots near zero.
        The mathematical map is a diffeomorphism; finite precision still limits
        extreme inputs, including overflow in cubic observation generation.
        """
        observations = _coordinates(observations, "observations")
        mixed = 2 * sqrt(10 / 3) * torch.sinh(torch.asinh(1.5 * sqrt(0.3) * observations) / 3)
        return mixed @ self._matrix.to(device=observations.device)


def recover_parameters(
    states: Tensor,
    next_states: Tensor,
    actions: Tensor,
    variant: ParameterGraph,
) -> Tensor:
    """Recover theta from one noiseless transition by Eq. (94), using a solve.

    Inputs are latent states (evaluation ground truth or inverse observations),
    not learner features. Actions in the paper's support ensure invertibility.
    """
    states = _coordinates(states, "states")
    next_states = _coordinates(next_states, "next_states")
    state_matrix, parameter_matrix = coefficient_matrices(actions, variant)
    residual = next_states.unsqueeze(-1) - state_matrix @ states.unsqueeze(-1)
    return _solve(parameter_matrix, residual).squeeze(-1)


def analytic_jacobians(
    states: Tensor,
    parameters: Tensor,
    actions: Tensor,
    variant: ParameterGraph,
) -> TransitionJacobians:
    """Return exact first derivatives without invoking automatic differentiation.

    Shapes are ``(...,10,1,10)`` for state/action and ``(...,10,10)``
    for parameters. Structural masks remain separate: a derivative can vanish
    at one state without removing a structural edge.
    """
    states, parameters, actions = cast(
        tuple[Tensor, Tensor, Tensor],
        torch.broadcast_tensors(  # pyright: ignore[reportUnknownMemberType]
            _coordinates(states, "states"),
            _coordinates(parameters, "parameters"),
            _coordinates(actions, "actions"),
        ),
    )
    state_matrix, parameter_matrix = coefficient_matrices(actions, variant)
    mask_state = state_mask(device=actions.device).to(dtype=actions.dtype)
    mask_parameter = parameter_mask(variant, device=actions.device).to(dtype=actions.dtype)
    # Interaction i is (1+a[i]**2) * sum_j(C[i,j] * (1+a[j]**3)).
    weighted = STATE_COUPLING * mask_state * states.unsqueeze(-2)
    weighted = weighted + PARAMETER_COUPLING * mask_parameter * parameters.unsqueeze(-2)
    self_derivative = 2 * actions * (weighted * (1 + actions.pow(3)).unsqueeze(-2)).sum(-1)
    gain = (ACTION_SCALE * actions.sum(-1)).exp()
    action_jacobian = (
        (ACTION_SCALE * gain[..., None] * parameters).unsqueeze(-1)
        + torch.diag_embed(self_derivative)
        + 3 * (1 + actions.square()).unsqueeze(-1) * weighted * actions.square().unsqueeze(-2)
    )
    return TransitionJacobians(
        state=state_matrix.unsqueeze(-2),
        action=action_jacobian.unsqueeze(-2),
        parameter=parameter_matrix,
    )


@dataclass(frozen=True)
class InfluenceDiagnostics:
    """Finite numerical checks of the paper's sufficient-influence constructions.

    These diagnostics accompany the analytical argument; they are not proofs of
    identifiability or of any population-wide assumption.
    """

    state_rank: int
    parameter_rank: int
    action_ranks: tuple[int, ...]
    expected_state_rank: int
    expected_parameter_rank: int
    expected_action_rank: int
    state_fixed_point_residual: float
    parameter_fixed_output_residual: float
    action_fixed_output_residual: float
    action_derivative_residual: float
    atol: float
    rtol: float
    samples: int
    seed: int
    passed: bool


def sufficient_influence_diagnostics(
    variant: ParameterGraph,
    *,
    samples: int = 256,
    seed: int = 7,
    atol: float = 1e-10,
    rtol: float = 1e-9,
) -> InfluenceDiagnostics:
    """Check coefficient spans using the paper's fixed-output constructions.

    Float64 SVD ranks use explicit absolute/relative tolerances. Inputs come
    from a private deterministic stream, so diagnostics do not affect datasets.
    The ten parameter basis vectors provide the action-column span at a=0.
    """
    if samples < 1 or not isfinite(atol) or not isfinite(rtol) or atol <= 0 or rtol <= 0:
        raise ValueError("samples and finite rank tolerances must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    actions = 2 * torch.rand(samples, DIMENSION, generator=generator, dtype=torch.float64) - 1
    output = torch.linspace(-0.9, 0.9, DIMENSION, dtype=torch.float64)
    parameters = torch.linspace(0.8, -0.7, DIMENSION, dtype=torch.float64)
    state_matrix, parameter_matrix = coefficient_matrices(actions, variant)
    # Eq. (96): the state and its successor both equal a fixed specified y.
    fixed_state_parameters = _solve(
        parameter_matrix, output[:, None] - state_matrix @ output[:, None]
    ).squeeze(-1)
    state_residual = float(
        (transition(output, fixed_state_parameters, actions, variant) - output).abs().max()
    )
    # Eq. (98): fixed output y and fixed theta, varying z_t with a.
    varying_state = _solve(
        state_matrix, output[:, None] - parameter_matrix @ parameters[:, None]
    ).squeeze(-1)
    parameter_residual = float(
        (transition(varying_state, parameters, actions, variant) - output).abs().max()
    )
    # At a=0, interaction derivatives vanish and each action column is theta/sqrt(10).
    parameter_basis = torch.eye(DIMENSION, dtype=torch.float64)
    zero_actions = torch.zeros_like(parameter_basis)
    zero_state_matrix, zero_parameter_matrix = coefficient_matrices(zero_actions, variant)
    action_states = _solve(
        zero_state_matrix, output[:, None] - zero_parameter_matrix @ parameter_basis[:, :, None]
    ).squeeze(-1)
    action_derivatives = analytic_jacobians(
        action_states, parameter_basis, zero_actions, variant
    ).action.squeeze(-2)
    action_residual = float(
        (transition(action_states, parameter_basis, zero_actions, variant) - output).abs().max()
    )
    derivative_residual = float(
        (action_derivatives - ACTION_SCALE * parameter_basis[:, :, None]).abs().max()
    )
    state_rank = int(_matrix_rank(state_matrix[:, state_mask()], atol=atol, rtol=rtol))
    parameter_rank = int(
        _matrix_rank(parameter_matrix[:, parameter_mask(variant)], atol=atol, rtol=rtol)
    )
    action_ranks = tuple(
        int(_matrix_rank(action_derivatives[:, :, k], atol=atol, rtol=rtol))
        for k in range(DIMENSION)
    )
    expected_state = int(state_mask().sum())
    expected_parameter = int(parameter_mask(variant).sum())
    passed = (
        state_rank == expected_state
        and parameter_rank == expected_parameter
        and all(rank == DIMENSION for rank in action_ranks)
        and max(state_residual, parameter_residual, action_residual, derivative_residual) <= atol
    )
    return InfluenceDiagnostics(
        state_rank=state_rank,
        parameter_rank=parameter_rank,
        action_ranks=action_ranks,
        expected_state_rank=expected_state,
        expected_parameter_rank=expected_parameter,
        expected_action_rank=DIMENSION,
        state_fixed_point_residual=state_residual,
        parameter_fixed_output_residual=parameter_residual,
        action_fixed_output_residual=action_residual,
        action_derivative_residual=derivative_residual,
        atol=atol,
        rtol=rtol,
        samples=samples,
        seed=seed,
        passed=passed,
    )
