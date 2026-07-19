"""Bounce: synthetic ground-truth system (Baumgartner et al., rendered to pixels).

N balls with per-episode sampled masses collide elastically in the unit box.
Because the simulator is ours, every quantity the identifiability diagnostics
need is exact and ships with each episode:

    frames    (T, 3, H, W)  rendered video — the PRIMARY observation (vision)
    states    (T, N, 4)     ground-truth kinematics [x, y, vx, vy]
    params    (N, 1)        ground-truth causal parameters (masses)
    contacts  (T-1, N, N)   time-indexed local graph: contacts[t, i, j] = True
                            iff balls i and j collided during transition t→t+1
                            (symmetric). The DIAGONAL records wall bounces, and
                            only in the radius∝mass variant, where the bounce
                            point pos < r_i depends on m_i (mass-relevant
                            self-event); with shared radius (D11) wall bounces
                            are mass-independent and the diagonal stays False.

Graph conventions for eval (derived from ``contacts``, documented here so SHD
code has one source of truth): state edge j→i at transition t iff contact
(plus self-edges — free flight); parameter edge mass_j → state_i at t iff
contact between i and j — including i = j: a ball's own mass matters during
ball-ball collisions always, and at wall bounces iff radius ∝ mass (audit G1).

Deliberate design (D11): mass is NOT rendered — all balls share one radius and
are identified by color. A single frame therefore reveals positions but never
S^ph; parameters are only observable from multi-frame behaviour (D4 rationale).

Physics: elastic impulse exchange (restitution 1), wall reflections (velocity
rule mass-independent; timing/geometry mass-dependent iff radius ∝ mass),
symplectic Euler with substeps. Everything is torch, deterministic per
(seed, index) — episodes are generated on the fly, nothing is downloaded.
"""

import math

import torch
from jaxtyping import Bool, Float
from torch import Tensor
from torch.utils.data import Dataset

_PALETTE = torch.tensor(
    [
        [0.90, 0.10, 0.10],  # red
        [0.10, 0.35, 0.95],  # blue
        [0.10, 0.75, 0.20],  # green
        [0.95, 0.75, 0.10],  # yellow
        [0.75, 0.15, 0.85],  # purple
        [0.10, 0.80, 0.80],  # cyan
        [0.95, 0.45, 0.10],  # orange
        [0.55, 0.35, 0.20],  # brown
    ]
)


def simulate_bounce(
    masses: Float[Tensor, "n 1"],
    positions: Float[Tensor, "n 2"],
    velocities: Float[Tensor, "n 2"],
    num_steps: int,
    radius: float = 0.08,
    dt: float = 0.1,
    substeps: int = 8,
    radii: Float[Tensor, " n"] | None = None,
) -> tuple[Float[Tensor, "t n 4"], Bool[Tensor, "tm1 n n"]]:
    """Roll the system forward; return state trajectory and per-transition contacts.

    Args:
        masses: Per-ball masses, (N, 1).
        positions: Initial centers in the unit box (already non-overlapping).
        velocities: Initial velocities.
        num_steps: T, number of recorded frames (T-1 transitions).
        radius: Shared ball radius (D11 default: mass is not geometric).
        dt: Time between recorded frames.
        substeps: Physics substeps per frame (collision robustness).
        radii: Optional per-ball radii (Baumgartner variant: radius ∝ mass —
            mass then also acts through contact distance/wall geometry,
            strengthening sufficient variability). Overrides ``radius``.

    Returns:
        states (T, N, 4) with rows [x, y, vx, vy]; contacts (T-1, N, N) boolean,
        symmetric off-diagonal (pair collided in that transition); diagonal True
        iff the ball wall-bounced AND radii were given (mass-relevant via r ∝ m).
    """
    num_balls = positions.shape[0]
    m = masses.squeeze(-1)  # (N,)
    r = radii if radii is not None else torch.full((num_balls,), radius)
    pos = positions.clone()
    vel = velocities.clone()
    states = torch.empty(num_steps, num_balls, 4)
    contacts = torch.zeros(num_steps - 1, num_balls, num_balls, dtype=torch.bool)
    states[0] = torch.cat([pos, vel], dim=-1)

    sub_dt = dt / substeps
    for step in range(num_steps - 1):
        for _ in range(substeps):
            pos = pos + vel * sub_dt
            # Wall reflections (unit box). The velocity rule is mass-independent,
            # but WHEN/WHERE the bounce happens depends on r_i — with r ∝ m that
            # makes ∂x_{t+1}/∂m_i ≠ 0, i.e. a GT param self-edge (audit G1).
            for axis in (0, 1):
                low = (pos[:, axis] < r) & (vel[:, axis] < 0)
                high = (pos[:, axis] > 1 - r) & (vel[:, axis] > 0)
                bounced = low | high
                vel[bounced, axis] = -vel[bounced, axis]
                if radii is not None:
                    contacts[step].diagonal()[bounced] = True
            # Pairwise elastic collisions (impulse exchange on approach).
            diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N, N, 2): x_i - x_j
            dist = diff.square().sum(dim=-1).sqrt() + torch.eye(num_balls) * 1e9
            touching = dist < r.unsqueeze(0) + r.unsqueeze(1)
            for pair in torch.nonzero(torch.triu(touching, diagonal=1)):
                i, j = int(pair[0].item()), int(pair[1].item())
                normal = diff[i, j] / dist[i, j]
                rel_speed = torch.dot(vel[i] - vel[j], normal)
                if rel_speed < 0:  # approaching — exchange impulse (e = 1)
                    impulse = -2.0 * rel_speed / (1.0 / m[i] + 1.0 / m[j])
                    vel[i] = vel[i] + (impulse / m[i]) * normal
                    vel[j] = vel[j] - (impulse / m[j]) * normal
                    contacts[step, i, j] = True
                    contacts[step, j, i] = True
        states[step + 1] = torch.cat([pos, vel], dim=-1)
    return states, contacts


def render_bounce(
    states: Float[Tensor, "t n 4"],
    resolution: int = 64,
    radius: float = 0.08,
    radii: Float[Tensor, " n"] | None = None,
) -> Float[Tensor, "t 3 h w"]:
    """Render ball positions as colored discs on a black background, in [0, 1].

    Default: equal radii for all balls (D11: mass is invisible); identity via
    _PALETTE colors, later balls drawn on top. ``radii`` renders per-ball sizes
    (Baumgartner variant — mass becomes visible in a single frame).
    """
    num_frames, num_balls = states.shape[0], states.shape[1]
    if num_balls > _PALETTE.shape[0]:
        raise ValueError(f"palette supports up to {_PALETTE.shape[0]} balls, got {num_balls}")
    axis = (torch.arange(resolution) + 0.5) / resolution
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")  # image row = y
    frames = torch.zeros(num_frames, 3, resolution, resolution)
    for t in range(num_frames):
        for ball in range(num_balls):
            cx, cy = states[t, ball, 0], states[t, ball, 1]
            ball_radius = float(radii[ball]) if radii is not None else radius
            inside = (grid_x - cx) ** 2 + (grid_y - cy) ** 2 < ball_radius**2
            frames[t, :, inside] = _PALETTE[ball].unsqueeze(-1)
    return frames


class BounceDataset(Dataset[dict[str, Tensor]]):
    """On-the-fly bounce episodes, deterministic per (seed, index).

    Each item carries pixels AND every ground-truth artifact, so the same
    dataset serves the vision pipeline (``frames``) and the diagnostics
    (``states``, ``params``, ``contacts``).
    """

    def __init__(
        self,
        num_episodes: int = 1000,
        clip_len: int = 4,
        num_balls: int = 5,
        resolution: int = 64,
        radius: float = 0.08,
        mass_range: tuple[float, float] = (0.5, 3.0),
        mass_normal: tuple[float, float] | None = None,
        radius_from_mass: bool = False,
        speed: float = 0.5,
        seed: int = 0,
        render: bool = True,
        cache: bool = False,
        preload: str | None = None,
    ) -> None:
        """Configure the generator (nothing is simulated until indexed).

        ``render=False`` skips frame rendering (GT-embedding regime: only
        ``states``/``params``/``contacts`` are consumed; big CPU saving).
        ``cache=True`` memoizes generated episodes (worth it for multi-epoch
        training; states-only episodes are tiny).
        ``preload``: path to a ``scripts/pregenerate_bounce.py`` file. Items
        are then served from its stacked tensors instead of simulated (the
        single-threaded python sim costs ~0.27 s/episode on a Grace core —
        7.6h for 100k, measured 2026-07-19). The file's generation settings
        must MATCH this constructor's (checked; a mismatch raises — silently
        serving episodes from a different physics config would break the D12
        identical-config rule between calibration and main). ``num_episodes``
        may be smaller than the file holds (a prefix is used); ``render`` must
        be False (frames are not stored).

        Baumgartner-exact variant (their App. E bounce): ``mass_normal=(mean,
        std)`` samples masses from a non-zero-mean normal (clamped to
        mass_range) instead of uniform, and ``radius_from_mass=True`` scales
        each ball's radius ∝ its mass (mean radius = ``radius``): mass then
        acts through contact geometry too, and IS visible in rendered frames —
        deliberately departing from D11's invisibility for that comparison.
        """
        if not 0 < mass_range[0] < mass_range[1]:
            raise ValueError(f"invalid mass_range {mass_range}")
        self.num_episodes = num_episodes
        self.clip_len = clip_len
        self.num_balls = num_balls
        self.resolution = resolution
        self.radius = radius
        self.mass_range = mass_range
        self.mass_normal = mass_normal
        self.radius_from_mass = radius_from_mass
        self.speed = speed
        self.seed = seed
        self.render = render
        self.cache = cache
        self._cached: dict[int, dict[str, Tensor]] = {}
        self._preloaded: dict[str, Tensor] | None = None
        if preload is not None:
            if render:
                raise ValueError("preload stores no frames; requires render=False")
            payload = torch.load(preload, weights_only=True)
            meta, expected = payload["meta"], self.generation_meta()
            stored_n = int(meta.pop("num_episodes"))
            expected.pop("num_episodes")
            if meta != expected:
                raise ValueError(
                    f"preload {preload} was generated with {meta}, "
                    f"but this dataset wants {expected} (D12: refuse silent drift)"
                )
            if num_episodes > stored_n:
                raise ValueError(f"preload holds {stored_n} episodes, need {num_episodes}")
            self._preloaded = payload["tensors"]

    def generation_meta(self) -> dict[str, object]:
        """Every setting that influences generated episodes (preload identity)."""
        return {
            "num_episodes": self.num_episodes,
            "clip_len": self.clip_len,
            "num_balls": self.num_balls,
            "radius": self.radius,
            "mass_range": tuple(self.mass_range),
            "mass_normal": tuple(self.mass_normal) if self.mass_normal is not None else None,
            "radius_from_mass": self.radius_from_mass,
            "speed": self.speed,
            "seed": self.seed,
        }

    def __len__(self) -> int:
        """Number of episodes."""
        return self.num_episodes

    def _initial_positions(
        self, generator: torch.Generator, radii: Tensor | None = None
    ) -> Tensor:
        """Rejection-sample non-overlapping centers inside the box.

        Uses each ball's OWN radius for wall margins and pairwise separation
        (audit G3: mean-radius margins let large balls start overlapping —
        72/300 baumgartner episodes began in contact).
        """
        r = radii if radii is not None else torch.full((self.num_balls,), self.radius)
        placed: list[Tensor] = []
        for i in range(self.num_balls):
            margin = float(r[i]) * 1.05
            while True:
                candidate = margin + torch.rand(2, generator=generator) * (1 - 2 * margin)
                if all(
                    (candidate - p).square().sum().sqrt() > (float(r[i]) + float(r[j])) * 1.1
                    for j, p in enumerate(placed)
                ):
                    placed.append(candidate)
                    break
        return torch.stack(placed)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Generate episode ``index``: sample params/state, simulate, render."""
        if self._preloaded is not None:
            if not 0 <= index < self.num_episodes:
                raise IndexError(index)
            return {key: tensor[index] for key, tensor in self._preloaded.items()}
        if self.cache and index in self._cached:
            return self._cached[index]
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        low, high = self.mass_range
        if self.mass_normal is not None:
            mean, std = self.mass_normal
            masses = (mean + std * torch.randn(self.num_balls, 1, generator=generator)).clamp(
                low, high
            )
        else:
            masses = low + torch.rand(self.num_balls, 1, generator=generator) * (high - low)
        # Fixed proportionality r_i = radius · m_i / mass_ref with an
        # episode-INDEPENDENT reference (audit G2): normalizing by the episode
        # mean made geometry reveal only m_i/mean(m) — and elastic impulses are
        # mass-ratio-only, so absolute mass became unidentifiable by ANY model
        # (measured MCC ceiling 0.775 vs Baumgartner's ~0.9).
        if self.radius_from_mass:
            mass_ref = (
                self.mass_normal[0]
                if self.mass_normal is not None
                else (self.mass_range[0] + self.mass_range[1]) / 2
            )
            radii = self.radius * masses.squeeze(-1) / mass_ref
        else:
            radii = None
        positions = self._initial_positions(generator, radii)
        angles = torch.rand(self.num_balls, generator=generator) * 2 * math.pi
        velocities = self.speed * torch.stack([angles.cos(), angles.sin()], dim=-1)
        states, contacts = simulate_bounce(
            masses, positions, velocities, num_steps=self.clip_len, radius=self.radius, radii=radii
        )
        item = {"states": states, "params": masses, "contacts": contacts}
        if self.render:
            item["frames"] = render_bounce(
                states, resolution=self.resolution, radius=self.radius, radii=radii
            )
        if self.cache:
            self._cached[index] = item
        return item


__all__ = ["BounceDataset", "render_bounce", "simulate_bounce"]
