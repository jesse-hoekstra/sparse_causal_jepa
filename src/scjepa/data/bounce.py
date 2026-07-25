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

# Increment whenever simulator dynamics change. Pre-generated trajectories are
# part of an experiment's data-generating process, so silently loading states
# from older collision rules would invalidate dense/main calibration matching.
_SIMULATOR_VERSION = 2


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

    def resolve_walls(step: int) -> None:
        """Reflect boundary overshoot and outward velocity at each radius wall.

        Mirroring ``x`` about ``r`` (or ``1-r``) is the exact remainder-of-step
        position for a linear wall impact. It also makes the recorded position a
        continuous function of a mass-dependent radius on a fixed bounce branch.
        A short loop handles unusually large overshoots without affecting the
        normal small-substep path.
        """
        for _ in range(4):
            any_violation = False
            for axis in (0, 1):
                low = pos[:, axis] < r
                high = pos[:, axis] > 1 - r
                violated = low | high
                if not bool(violated.any()):
                    continue
                any_violation = True

                # Preserve the distance travelled beyond the contact surface.
                pos[low, axis] = 2 * r[low] - pos[low, axis]
                pos[high, axis] = 2 * (1 - r[high]) - pos[high, axis]

                # Pair projection can rarely push a center through a wall while
                # its velocity already points inward. Correct its position but
                # only reflect velocity that is actually travelling outward.
                outward_low = low & (vel[:, axis] < 0)
                outward_high = high & (vel[:, axis] > 0)
                vel[outward_low | outward_high, axis] *= -1
                if radii is not None:
                    contacts[step].diagonal()[violated] = True
            if not any_violation:
                break

    sub_dt = dt / substeps
    for step in range(num_steps - 1):
        for _ in range(substeps):
            pos = pos + vel * sub_dt
            resolve_walls(step)

            # Pairwise elastic collisions. First project penetrations back to
            # the exact contact surface, splitting the correction by inverse
            # mass so the pair's center of mass is unchanged. Then apply the
            # e=1 impulse on approach. A few sequential passes resolve short
            # contact chains; ordinary no-contact substeps exit after one cheap
            # vectorized overlap check.
            for _ in range(max(4, 4 * num_balls)):
                diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # x_i - x_j
                dist = diff.square().sum(dim=-1).sqrt()
                min_dist = r.unsqueeze(0) + r.unsqueeze(1)
                touching = torch.triu(dist < min_dist, diagonal=1)
                pairs = torch.nonzero(touching)
                if pairs.numel() == 0:
                    break

                for pair in pairs:
                    i, j = int(pair[0].item()), int(pair[1].item())
                    delta = pos[i] - pos[j]
                    distance: Tensor = delta.square().sum().sqrt()
                    contact_distance = r[i] + r[j]
                    if distance >= contact_distance:
                        continue  # an earlier correction in this pass separated it
                    if float(distance) > 1e-12:
                        normal = delta / distance
                    else:  # degenerate coincidence: choose a deterministic separator
                        relative = vel[i] - vel[j]
                        relative_norm: Tensor = relative.square().sum().sqrt()
                        normal: Tensor = (
                            -relative / relative_norm
                            if float(relative_norm) > 1e-12
                            else pos.new_tensor([1.0, 0.0])
                        )

                    penetration: Tensor = contact_distance - distance
                    inv_i, inv_j = 1.0 / m[i], 1.0 / m[j]
                    inv_total = inv_i + inv_j
                    pos[i] = pos[i] + normal * penetration * (inv_i / inv_total)
                    pos[j] = pos[j] - normal * penetration * (inv_j / inv_total)

                    rel_speed = torch.dot(vel[i] - vel[j], normal)
                    if rel_speed < 0:  # approaching — exchange impulse (e = 1)
                        impulse = -2.0 * rel_speed / inv_total
                        vel[i] = vel[i] + (impulse / m[i]) * normal
                        vel[j] = vel[j] - (impulse / m[j]) * normal
                    contacts[step, i, j] = True
                    contacts[step, j, i] = True

                # A correction near a boundary can create a wall violation;
                # resolving it here may in turn require one more pair pass.
                resolve_walls(step)
        states[step + 1] = torch.cat([pos, vel], dim=-1)
    return states, contacts


def render_bounce(
    states: Float[Tensor, "t n 4"],
    resolution: int = 64,
    radius: float = 0.08,
    radii: Float[Tensor, " n"] | None = None,
    uniform_appearance: bool = False,
) -> Float[Tensor, "t 3 h w"]:
    """Render ball positions as discs on a black background, in [0, 1].

    Two appearance contracts, chosen by ``uniform_appearance``:

    * ``False`` (the state-to-state regime visualisation, simulator v2): each ball gets its
      own _PALETTE colour and, if ``radii`` is given, its own size. Both are
      dataset-global signatures tied to the simulator ROW INDEX, so a single
      frame reveals identity and (with ``radii``) mass.
    * ``True`` (Experiments 2/3 primary condition, experiments.pdf p.14): every
      object is drawn with "the same mass-independent glyph and rendered
      radius". All discs are white and share ``radius``; ``radii`` is ignored
      for drawing. The frames then carry "no dataset-global visual signature
      tied to simulator row index", so mass is only reachable through centre
      motion and hidden contact geometry.

    ``radii`` still describes the PHYSICAL collision radii and is what the
    simulator used; passing it here only affects how large the discs are drawn.
    """
    num_balls = states.shape[1]
    if not uniform_appearance and num_balls > _PALETTE.shape[0]:
        raise ValueError(f"palette supports up to {_PALETTE.shape[0]} balls, got {num_balls}")
    axis = (torch.arange(resolution) + 0.5) / resolution
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")  # image row = y

    if uniform_appearance:
        ball_radii = torch.full((num_balls,), radius)
        palette = torch.ones(num_balls, 3)
    else:
        ball_radii = radii.clone() if radii is not None else torch.full((num_balls,), radius)
        palette = _PALETTE[:num_balls]

    # Rasterize every (frame, ball) disc at once rather than looping T*N times:
    # for 60 frames and 5 balls that is 300 full-resolution passes per episode,
    # which the visual experiments pay on every batch.
    centre_x = states[:, :, 0].unsqueeze(-1).unsqueeze(-1)  # (T, N, 1, 1)
    centre_y = states[:, :, 1].unsqueeze(-1).unsqueeze(-1)
    squared = (grid_x - centre_x) ** 2 + (grid_y - centre_y) ** 2  # (T, N, H, W)
    inside = squared < (ball_radii.view(1, -1, 1, 1) ** 2)

    # Later balls are drawn ON TOP, so each pixel takes the HIGHEST covering
    # index; -1 marks background. This reproduces the sequential overwrite that
    # the per-ball loop performed.
    index = torch.where(
        inside, torch.arange(num_balls).view(1, -1, 1, 1).expand_as(inside), -1
    ).amax(dim=1)  # (T, H, W)
    colours = torch.cat([torch.zeros(1, 3), palette])  # row 0 = background
    return colours[index + 1].permute(0, 3, 1, 2).contiguous()


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
        render_radius_from_mass: bool | None = None,
        uniform_appearance: bool = False,
        mass_independent_init: bool = False,
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
        may be smaller than the file holds (a prefix is used). Preload files
        store no frames; with ``render=True`` the frames are rendered ON THE
        FLY from the preloaded states, which is what makes the visual
        experiments affordable — 100k x 60 rendered frames would be ~294 GB on
        disk, while the states they are drawn from are already stored and the
        physics is identical.

        Baumgartner-aligned variant (their App. E bounce): ``mass_normal=(mean,
        std)`` samples masses from a non-zero-mean normal (clamped to
        mass_range) instead of uniform, and ``radius_from_mass=True`` scales
        each ball's PHYSICAL radius ∝ its mass (mean radius = ``radius``): mass
        then acts through contact geometry too, which is what makes ABSOLUTE
        mass (not just mass ratios) identifiable at all — experiments.pdf p.14
        "Why absolute masses are observable".

        Visual-experiment knobs (experiments.pdf p.14: "For the visual regimes,
        physical and rendered radii must be separated"):

        * ``render_radius_from_mass``: how large discs are DRAWN. ``None``
          (default) follows ``radius_from_mass``, reproducing simulator-v2
          frames exactly. ``False`` draws every ball at the shared ``radius``
          while the physics keeps the mass-dependent radii — the visual regimes
          primary condition. ``True`` is the labelled visible-radius control.
        * ``uniform_appearance``: draw all balls as identical white discs
          instead of _PALETTE colours, removing the persistent per-row visual
          signature. Required alongside ``render_radius_from_mass=False`` for
          the primary visual condition.
        * ``mass_independent_init``: compute wall margins and pair separation
          from the common worst-case radius r_max = radius * mass_range[1] /
          mass_ref rather than each ball's own radius, so the initial-position
          SUPPORT no longer depends on mass. This isolates recovery evidence to
          the transition dynamics and is a required control before attributing
          recovery to them. It changes the generated states, so it is part of
          the preload identity.

        Setting ``render_radius_from_mass``/``uniform_appearance`` never changes
        states, params or contacts, so it is deliberately NOT part of
        ``generation_meta``: the same preloaded physics serves the state-to-state
        state runs and the visual runs.
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
        self.render_radius_from_mass = (
            radius_from_mass if render_radius_from_mass is None else render_radius_from_mass
        )
        self.uniform_appearance = uniform_appearance
        self.mass_independent_init = mass_independent_init
        self.speed = speed
        self.seed = seed
        self.render = render
        self.cache = cache
        self._cached: dict[int, dict[str, Tensor]] = {}
        self._preloaded: dict[str, Tensor] | None = None
        if preload is not None:
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
        """Every setting that influences generated episodes (preload identity).

        Rendering settings are absent by design: preload files store states,
        params and contacts, none of which depend on how the discs are drawn.

        ``mass_independent_init`` DOES change the states, so it appears here —
        but only when enabled, so that preload files written before the flag
        existed still compare equal. D12 must refuse silent drift without also
        refusing silent sameness.
        """
        meta: dict[str, object] = {
            "simulator_version": _SIMULATOR_VERSION,
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
        if self.mass_independent_init:
            meta["mass_independent_init"] = True
        return meta

    def physical_radii(self, masses: Float[Tensor, "n 1"]) -> Float[Tensor, " n"] | None:
        """Collision radii r_i = radius * m_i / m_ref (experiments.pdf Eq. 2), or None.

        The reference mass is episode-INDEPENDENT (audit G2): normalising by the
        episode mean would make geometry reveal only m_i/mean(m), and since
        elastic impulses are mass-ratio-only, absolute mass would stop being
        identifiable by any model.
        """
        if not self.radius_from_mass:
            return None
        return self.radius * masses.squeeze(-1) / self._mass_reference()

    def _mass_reference(self) -> float:
        """Episode-independent m_ref used by Eq. 2 (1.5 in the configured setup)."""
        if self.mass_normal is not None:
            return self.mass_normal[0]
        return (self.mass_range[0] + self.mass_range[1]) / 2

    def __len__(self) -> int:
        """Number of episodes."""
        return self.num_episodes

    def _initial_positions(self, generator: torch.Generator, radii: Tensor | None = None) -> Tensor:
        """Rejection-sample non-overlapping centers inside the box.

        Uses each ball's OWN radius for wall margins and pairwise separation
        (audit G3: mean-radius margins let large balls start overlapping —
        72/300 baumgartner episodes began in contact). The support of the
        initial positions is therefore itself mass-dependent.

        Under ``mass_independent_init`` every margin instead uses the common
        worst-case radius r_max = radius * mass_range[1] / m_ref, so the support
        is identical for every mass draw and recovery evidence must come from
        the dynamics rather than from where the balls could start.
        """
        if self.mass_independent_init:
            r_max = self.radius * self.mass_range[1] / self._mass_reference()
            r = torch.full((self.num_balls,), r_max)
        else:
            r = radii if radii is not None else torch.full((self.num_balls,), self.radius)
        for i in range(self.num_balls):
            if 2 * float(r[i]) * 1.05 >= 1.0:
                raise ValueError(
                    f"ball radius {float(r[i]):.3f} leaves no room in the unit box; "
                    f"reduce radius or mass_range"
                )
        # Greedy sequential placement can paint itself into a corner: the first
        # balls land legally and leave nowhere for the last one. That is rare at
        # mass-proportional radii but routine at the worst-case radius, where
        # every ball claims r_max, so a failed ball restarts the WHOLE
        # configuration rather than resampling into a dead layout.
        #
        # Both caps are far above what the mass-proportional path ever uses, so
        # its draw sequence — and every episode generated before this guard
        # existed — is unchanged.
        max_per_ball, max_layouts = 2_000, 20_000
        for _ in range(max_layouts):
            placed: list[Tensor] = []
            for i in range(self.num_balls):
                margin = float(r[i]) * 1.05
                for _ in range(max_per_ball):
                    candidate = margin + torch.rand(2, generator=generator) * (1 - 2 * margin)
                    if all(
                        (candidate - p).square().sum().sqrt() > (float(r[i]) + float(r[j])) * 1.1
                        for j, p in enumerate(placed)
                    ):
                        placed.append(candidate)
                        break
                else:
                    break  # dead layout — restart from an empty box
            if len(placed) == self.num_balls:
                return torch.stack(placed)
        raise RuntimeError(
            f"could not place {self.num_balls} balls of radius <= {float(r.max()):.3f} in "
            f"{max_layouts} layout attempts; the requested packing is too dense"
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Generate episode ``index``: sample params/state, simulate, render."""
        if self._preloaded is not None:
            if not 0 <= index < self.num_episodes:
                raise IndexError(index)
            item = {key: tensor[index] for key, tensor in self._preloaded.items()}
            if self.render and "frames" not in item:
                item["frames"] = self._render(item["states"], item["params"])
            return item
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
        radii = self.physical_radii(masses)
        positions = self._initial_positions(generator, radii)
        angles = torch.rand(self.num_balls, generator=generator) * 2 * math.pi
        velocities = self.speed * torch.stack([angles.cos(), angles.sin()], dim=-1)
        states, contacts = simulate_bounce(
            masses, positions, velocities, num_steps=self.clip_len, radius=self.radius, radii=radii
        )
        item = {"states": states, "params": masses, "contacts": contacts}
        if self.render:
            item["frames"] = self._render(states, masses)
        if self.cache:
            self._cached[index] = item
        return item

    def _render(
        self, states: Float[Tensor, "t n 4"], masses: Float[Tensor, "n 1"]
    ) -> Float[Tensor, "t 3 h w"]:
        """Draw an episode under the configured appearance contract.

        The drawn radii come from ``render_radius_from_mass``, which is
        independent of the ``radius_from_mass`` the physics used — that split
        is what lets the visual experiments keep mass identifiable through the
        dynamics while hiding it from any single frame.
        """
        drawn_radii = (
            self.radius * masses.squeeze(-1) / self._mass_reference()
            if self.render_radius_from_mass
            else None
        )
        return render_bounce(
            states,
            resolution=self.resolution,
            radius=self.radius,
            radii=drawn_radii,
            uniform_appearance=self.uniform_appearance,
        )


__all__ = ["BounceDataset", "render_bounce", "simulate_bounce"]
