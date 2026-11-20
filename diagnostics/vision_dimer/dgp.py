"""Data-generating process: two bonded colloidal dimers in an anisotropic medium, top view.

Each of two dimers is a large bead (slot ``a``) bonded to a small bead (slot ``b``).
The state is the eight in-plane bead coordinates in the medium's principal frame
``(u, v)``, dimer-major: ``z = (u_1a, v_1a, u_1b, v_1b, u_2a, v_2a, u_2b, v_2b)``.
Hidden per-system parameters are the drag coefficients along the two principal
axes for the two bead sizes, ``theta = (c_au, c_av, c_bu, c_bv)``, shared by both
dimers. Actions are one out-of-plane magnetic field ``B`` (scalar) and one in-plane
electric kick ``E_g`` per dimer, given in the camera frame: ``a = (B, E_1, E_2)``.
For bead ``i`` of dimer ``g`` with ``w_i = q_i^t - q_i^{t-1} + h^2 (e_i P E_g + F_i(q^t))``:

    w'      = R(sigma * beta_i * B) w                        (2D Cayley rotation)
    q^{t+1} = q^t + (I - h C_slot) w' - h^2 kappa phi(q^t - s_i),   C = diag(c_u, c_v),

where ``P`` maps camera-frame kicks into the medium frame, ``F_i`` is the smooth
anharmonic zero-rest-length bond force from the partner bead,
``F_a = -k_b (1 + |r|^2 / (1 + |r|^2)) r`` with ``r = q_a - q_b`` and ``F_b = -F_a``
(known constants; C-infinity, with a dense invertible Jacobian everywhere), ``phi``
the per-axis anharmonic trap law and ``s_i`` the bead's trap site. Electric kicks
and magnetic rotations both scale with each bead's charge-to-mass ratio; the two
beads of a dimer carry opposite charges. Dimers do not
interact. True supports: two dense 4x4 temporal blocks (dimers); ``B`` reaches all
eight coordinates and ``E_g`` only dimer ``g``; drag ``c_{slot,axis}`` reaches the
two beads of that slot along that axis. The medium axes are rotated 45 degrees
against the camera, and the kicks and bond are not aligned with either frame, so
no implicit cue favours the medium frame except the parameter structure.
"""

import math
from dataclasses import asdict, dataclass
from typing import cast

import torch
from torch import Tensor

DIMERS = 2
SLOTS = 2
AXES = 2
STATE_DIM = DIMERS * SLOTS * AXES  # 8
ACTION_DIM = 1 + DIMERS * AXES  # 5
PARAM_DIM = SLOTS * AXES  # 4
BLOCKS = ((0, 1, 2, 3), (4, 5, 6, 7))


@dataclass(frozen=True)
class Physics:
    """Known constants of the recurrence and the per-system parameter prior."""

    dt: float = 0.5
    field_scale: float = 0.6
    charge_to_mass: tuple[tuple[float, float], tuple[float, float]] = ((1.0, -0.6), (0.75, -0.4))
    trap_stiffness: float = 1.0
    bond_stiffness: float = 1.0
    site_separation: float = 1.0
    site_angle_degrees: float = 20.0
    medium_angle_degrees: float = 45.0
    dimer_centres: tuple[tuple[float, float], tuple[float, float]] = ((-1.3, 0.6), (1.3, -0.6))
    drag_range: tuple[float, float] = (0.1, 1.2)
    action_std: float = 1.3
    burn_in: int = 300
    initial_position_std: float = 0.5

    def __post_init__(self) -> None:
        """Require positive constants, contractive drag and the linear stability margin."""
        low, high = self.drag_range
        if min(self.dt, self.field_scale, self.trap_stiffness, self.bond_stiffness) <= 0:
            raise ValueError("time step, field scale and stiffnesses must be positive")
        if not 0 < low < high or self.dt * high >= 1:
            raise ValueError("require 0 < c_low < c_high and h * c_high < 1")
        # Undamped central-difference margin with the stiffest slopes: trap 17/8, and the
        # bond's relative mode 2 * (5/2) * bond (bond Jacobian eigenvalues <= 5/2 * bond).
        stiffest = 17 / 8 * self.trap_stiffness + 5 * self.bond_stiffness
        if self.dt**2 * stiffest >= 4 - 2 * self.dt * high:
            raise ValueError("stiffest mode violates the central-difference stability margin")
        flat = [b for pair in self.charge_to_mass for b in pair]
        magnitudes = [abs(b) for b in flat]
        # A15 needs distinct rotation magnitudes across dimers for each slot (equal |beta|
        # drops the temporal witness span to rank 25 of 32); require all magnitudes distinct.
        if 0 in flat or len(set(magnitudes)) != len(magnitudes):
            raise ValueError(
                "charge-to-mass ratios must be nonzero with pairwise distinct magnitudes"
            )

    def as_record(self) -> dict[str, object]:
        """JSON-serializable constants."""
        return asdict(self)


def phi(s: Tensor) -> Tensor:
    """Anharmonic restoring law: odd, analytic, slope in [1, 17/8]."""
    return s + s.pow(3) / (1 + s.square())


def rotation2(angle_degrees: float, like: Tensor) -> Tensor:
    """Fixed planar rotation matrix."""
    t = math.radians(angle_degrees)
    return torch.tensor(
        [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]],
        dtype=like.dtype,
        device=like.device,
    )


def cayley2(tau: Tensor) -> Tensor:
    """Planar Cayley rotation ``((1 - t^2) I + 2 t J) / (1 + t^2)``, ``(..., 2, 2)``."""
    t = tau[..., None, None]
    eye = torch.eye(2, dtype=tau.dtype, device=tau.device)
    skew = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=tau.dtype, device=tau.device)
    return ((1 - t.square()) * eye + 2 * t * skew) / (1 + t.square())


def unflatten_state(state: Tensor) -> Tensor:
    """``(..., 8) -> (..., dimer, slot, axis)``."""
    return cast(Tensor, state.unflatten(-1, (DIMERS, SLOTS, AXES)))  # pyright: ignore[reportUnknownMemberType]


def flatten_state(state: Tensor) -> Tensor:
    """``(..., dimer, slot, axis) -> (..., 8)``."""
    return state.flatten(-3)


def sites(physics: Physics, like: Tensor) -> Tensor:
    """Trap sites ``(dimer, slot, axis)`` in the medium frame."""
    to_medium = rotation2(-physics.medium_angle_degrees, like)  # camera -> medium
    offset = rotation2(physics.site_angle_degrees, like) @ torch.tensor(
        [physics.site_separation / 2, 0.0], dtype=like.dtype, device=like.device
    )
    centres = (
        torch.tensor(physics.dimer_centres, dtype=like.dtype, device=like.device) @ to_medium.T
    )
    return torch.stack((centres + offset, centres - offset), 1)


def bond_force(state: Tensor, physics: Physics) -> Tensor:
    """Smooth zero-rest-length anharmonic bond force on each bead ``(..., dimer, slot, axis)``."""
    r = state[..., 0, :] - state[..., 1, :]  # slot a minus slot b
    squared = r.square().sum(-1, keepdim=True)
    force_a = -physics.bond_stiffness * (1 + squared / (1 + squared)) * r
    return torch.stack((force_a, -force_a), -2)


def _beta_gain(physics: Physics, like: Tensor) -> tuple[Tensor, Tensor]:
    """Charge-to-mass ratios: both the kick gain and the rotation rate of each bead."""
    beta = torch.tensor(physics.charge_to_mass, dtype=like.dtype, device=like.device)
    return beta, beta


def medium_kicks(action: Tensor, physics: Physics) -> Tensor:
    """Camera-frame kicks ``(..., 5) -> (..., dimer, 2)`` in the medium frame."""
    to_medium = rotation2(-physics.medium_angle_degrees, action)
    camera = cast(Tensor, action[..., 1:].unflatten(-1, (DIMERS, AXES)))  # pyright: ignore[reportUnknownMemberType]
    return camera @ to_medium.T


def transition(
    previous: Tensor, current: Tensor, action: Tensor, theta: Tensor, physics: Physics
) -> Tensor:
    """``q^{t+1}`` from ``q^{t-1}, q^t`` ``(..., dimer, slot, axis)``, actions and theta."""
    beta, gain = _beta_gain(physics, current)
    drag = cast(Tensor, theta.unflatten(-1, (SLOTS, AXES)))[..., None, :, :]  # pyright: ignore[reportUnknownMemberType]
    kick = gain[..., None] * medium_kicks(action, physics)[..., :, None, :]
    increment = current - previous + physics.dt**2 * (kick + bond_force(current, physics))
    rotation = cayley2(physics.field_scale * beta * action[..., None, None, :1].squeeze(-1))
    rotated = (rotation @ increment[..., None])[..., 0]
    restoring = physics.dt**2 * physics.trap_stiffness * phi(current - sites(physics, current))
    return current + (1 - physics.dt * drag) * rotated - restoring


def previous_state(
    target: Tensor, current: Tensor, action: Tensor, theta: Tensor, physics: Physics
) -> Tensor:
    """Solve the recurrence for ``q^{t-1}`` given ``q^t``, the action and ``q^{t+1}`` exactly."""
    beta, gain = _beta_gain(physics, current)
    drag = cast(Tensor, theta.unflatten(-1, (SLOTS, AXES)))[..., None, :, :]  # pyright: ignore[reportUnknownMemberType]
    restoring = physics.dt**2 * physics.trap_stiffness * phi(current - sites(physics, current))
    rotated = (target - current + restoring) / (1 - physics.dt * drag)
    rotation = cayley2(physics.field_scale * beta * action[..., None, None, :1].squeeze(-1))
    increment = (rotation.transpose(-1, -2) @ rotated[..., None])[..., 0]
    kick = gain[..., None] * medium_kicks(action, physics)[..., :, None, :]
    return current + physics.dt**2 * (kick + bond_force(current, physics)) - increment


def sample_theta(groups: int, physics: Physics, generator: torch.Generator) -> Tensor:
    """Independent uniform drag coefficients, float64 ``(G, 4)``."""
    low, high = physics.drag_range
    uniform = torch.rand(groups, PARAM_DIM, generator=generator, dtype=torch.float64)
    return low + (high - low) * uniform


def sample_actions(shape: tuple[int, ...], physics: Physics, generator: torch.Generator) -> Tensor:
    """I.i.d. Gaussian commands rounded to float32, so simulator and learner agree exactly."""
    actions = physics.action_std * torch.randn(
        *shape, ACTION_DIM, generator=generator, dtype=torch.float64
    )
    return actions.float().double()


def simulate(
    theta: Tensor, views: int, frames: int, physics: Physics, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    """Simulate ``views`` stationary episodes per system after an unrecorded driven burn-in.

    Both initial lags are drawn independently around the sites (full-dimensional
    support). Returns float64 states ``(G, R, frames, 8)`` and float32-exact actions
    ``(G, R, frames - 1, 5)``; action ``t`` advances frame ``t`` to ``t + 1``.
    Nothing is clipped, filtered or rejected.
    """
    groups = len(theta)
    like = theta.new_zeros(())
    site = sites(physics, like)
    shape = (groups, views, DIMERS, SLOTS, AXES)
    previous = site + physics.initial_position_std * torch.randn(
        *shape, generator=generator, dtype=torch.float64
    )
    current = site + physics.initial_position_std * torch.randn(
        *shape, generator=generator, dtype=torch.float64
    )
    episode_theta = theta[:, None].expand(groups, views, PARAM_DIM)
    burn = sample_actions((groups, views, physics.burn_in), physics, generator)
    for step in range(physics.burn_in):
        previous, current = (
            current,
            transition(previous, current, burn[:, :, step], episode_theta, physics),
        )
    actions = sample_actions((groups, views, frames - 1), physics, generator)
    states = [current]
    for step in range(frames - 1):
        previous, current = (
            current,
            transition(previous, current, actions[:, :, step], episode_theta, physics),
        )
        states.append(current)
    trajectory = torch.stack(states, 2)
    if not bool(torch.isfinite(trajectory).all()):
        raise FloatingPointError("nonfinite trajectory; revise settings rather than filtering")
    return flatten_state(trajectory), actions


def graphs() -> dict[str, Tensor]:
    """True lag-aggregated supports: temporal (8,8), action (8,5), parameter (8,4)."""
    index = torch.arange(STATE_DIM)
    dimer, slot, axis = index // 4, (index // 2) % 2, index % 2
    temporal = dimer[:, None] == dimer[None, :]
    kick_owner = torch.arange(ACTION_DIM - 1) // AXES
    action = torch.cat(
        (torch.ones(STATE_DIM, 1, dtype=torch.bool), dimer[:, None] == kick_owner[None, :]), 1
    )
    parameter = (2 * slot + axis)[:, None] == torch.arange(PARAM_DIM)[None, :]
    return {"state": temporal, "action": action, "parameter": parameter}


IMAGING = ("rgb",)


@dataclass(frozen=True)
class Renderer:
    """The top-view RGB camera of the vision experiment.

    Medium-frame positions are rotated into the camera frame and mapped to pixel
    centres through ``centre + span * tanh(p / scale)`` (a smooth, strictly monotone
    wide-angle lens that keeps every bead in frame); image rows grow downwards, so
    the camera ``y`` is negated. Light is additive (fluorescence), so nothing occludes.
    Each dimer carries one dye, whose emission the RGB sensor sees through the fixed,
    overlapping responses ``dyes`` (rows: dimers, columns: R, G, B). Beads have their
    true sizes: the large bead's blob has width ``sigma`` and the small bead's
    ``sigma_small``, with brightness proportional to bead volume (peak proportional to
    width). Every pair of beads differs in dye or in width, so the image determines
    every bead even when beads overlap exactly. No pixel is clipped. ``imaging`` is a
    recorded field so that data manifests name the camera explicitly.
    """

    size: int = 64
    sigma: float = 2.6
    span: float = 28.0
    scale: float = 2.0
    medium_angle_degrees: float = 45.0
    imaging: str = "rgb"
    sigma_small: float = 1.8
    dyes: tuple[tuple[float, float, float], ...] = ((1.0, 0.55, 0.10), (0.10, 0.45, 1.0))

    def __post_init__(self) -> None:
        """Keep every centre strictly inside the frame and the colour model invertible."""
        if self.size < 8 or self.sigma <= 0 or self.scale <= 0:
            raise ValueError("invalid renderer settings")
        if not 0 < self.span < (self.size - 1) / 2:
            raise ValueError("span must keep blob centres inside the frame")
        if self.imaging not in IMAGING:
            raise ValueError(f"imaging must be one of {IMAGING}")
        object.__setattr__(self, "dyes", tuple(tuple(float(c) for c in d) for d in self.dyes))
        if not 0 < self.sigma_small < self.sigma:
            raise ValueError("the small bead must be drawn narrower than the large bead")
        dyes = torch.tensor(self.dyes, dtype=torch.float64)
        if dyes.shape != (DIMERS, 3) or bool((dyes < 0).any()):
            raise ValueError("one non-negative RGB response per dimer is required")
        values = cast(Tensor, torch.linalg.svdvals(dyes))  # pyright: ignore[reportUnknownMemberType]
        if float(values[-1] / values[0]) < 0.2:
            raise ValueError("the two dye colours must stay well distinguishable")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Image tensor shape ``(C, H, W)``."""
        return (3, self.size, self.size)

    def camera(self, states: Tensor) -> Tensor:
        """Camera-frame positions ``(..., 4, 2)`` of the four beads."""
        medium = unflatten_state(states).flatten(-3, -2)
        to_camera = rotation2(self.medium_angle_degrees, medium)
        return medium @ to_camera.T

    def pixel_coordinates(self, states: Tensor) -> Tensor:
        """Squashed pixel centres ``(..., 4, 2)`` as (column, row)."""
        middle = (self.size - 1) / 2
        p = middle + self.span * torch.tanh(self.camera(states) / self.scale)
        return torch.stack((p[..., 0], 2 * middle - p[..., 1]), -1)

    def __call__(self, states: Tensor) -> Tensor:
        """Render ``(..., 8)`` states as float32 images ``(..., 3, size, size)``."""
        pixels = self.pixel_coordinates(states.float())
        device = states.device
        grid = torch.arange(self.size, device=device, dtype=torch.float32)
        widths = torch.tensor(  # bead order 1a, 1b, 2a, 2b
            [self.sigma, self.sigma_small] * DIMERS, device=device, dtype=torch.float32
        )
        peaks = widths / self.sigma  # volume-proportional brightness, width-proportional peak

        def sized(centre: Tensor) -> Tensor:
            return torch.exp(-0.5 * ((grid - centre[..., None]) / widths[:, None]).square())

        columns, rows = sized(pixels[..., 0]), sized(pixels[..., 1])
        beads = peaks[:, None, None] * rows[..., :, None] * columns[..., None, :]
        shape = (*beads.shape[:-3], DIMERS, SLOTS, *beads.shape[-2:])
        dimers = beads.reshape(shape).sum(-3)  # one dye image per dimer
        dyes = torch.tensor(self.dyes, device=device, dtype=torch.float32)
        return torch.einsum("...ghw,gc->...chw", dimers, dyes)

    def as_record(self) -> dict[str, object]:
        """JSON-serializable settings, as stored in data manifests."""
        return asdict(self)


__all__ = [
    "ACTION_DIM",
    "AXES",
    "BLOCKS",
    "DIMERS",
    "IMAGING",
    "PARAM_DIM",
    "SLOTS",
    "STATE_DIM",
    "Physics",
    "Renderer",
    "bond_force",
    "cayley2",
    "flatten_state",
    "graphs",
    "medium_kicks",
    "phi",
    "previous_state",
    "sample_actions",
    "sample_theta",
    "simulate",
    "sites",
    "transition",
    "unflatten_state",
]
