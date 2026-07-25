"""Visual (Experiment 2/3) data condition: separated physical and rendered radii.

experiments.pdf p.14 requires that for the visual experiments the simulator keeps
Eq. 2's mass-proportional PHYSICAL radii — that is what makes absolute mass
identifiable at all — while every object is drawn with the same mass-independent
glyph and rendered radius, and the frames carry no dataset-global signature tied
to the simulator row index.
"""

import torch

from scjepa.data.bounce import BounceDataset

PHYSICS = dict(
    clip_len=16,
    num_balls=5,
    seed=5,
    mass_normal=(1.5, 0.5),
    radius_from_mass=True,
    speed=0.7,
    radius=0.08,
    resolution=64,
)


def _states_dataset(**overrides: object) -> BounceDataset:
    return BounceDataset(num_episodes=6, render=False, **{**PHYSICS, **overrides})


def _frames_dataset(**overrides: object) -> BounceDataset:
    return BounceDataset(num_episodes=6, render=True, **{**PHYSICS, **overrides})


def test_visual_condition_leaves_the_physics_untouched() -> None:
    """Rendering settings must not perturb states, params or contacts."""
    plain = _states_dataset()
    visual = _states_dataset(render_radius_from_mass=False, uniform_appearance=True)
    for index in range(3):
        for key in ("states", "params", "contacts"):
            assert torch.equal(plain[index][key], visual[index][key])


def test_rendering_settings_are_not_part_of_preload_identity() -> None:
    """The same stored physics must serve Experiment 1 and Experiment 2."""
    plain = _states_dataset()
    visual = _states_dataset(render_radius_from_mass=False, uniform_appearance=True)
    assert plain.generation_meta() == visual.generation_meta()


def test_uniform_appearance_removes_the_per_row_colour_signature() -> None:
    """Identical glyphs: the frame holds one ink colour, not one per ball."""
    frames = _frames_dataset(render_radius_from_mass=False, uniform_appearance=True)[0]["frames"]
    colours = torch.unique(frames[0].reshape(3, -1), dim=1)
    assert colours.shape[1] == 2  # background plus a single shared ink

    palette = _frames_dataset()[0]["frames"]
    assert torch.unique(palette[0].reshape(3, -1), dim=1).shape[1] > 2


def test_drawn_size_stops_tracking_mass() -> None:
    """Disc area must decouple from mass while the physics keeps its radii."""
    visible = _frames_dataset()
    hidden = _frames_dataset(render_radius_from_mass=False, uniform_appearance=True)

    def area_spread(dataset: BounceDataset) -> float:
        areas = [(dataset[i]["frames"][0].amax(0) > 0.5).float().sum() for i in range(6)]
        return float(torch.stack(areas).std())

    # Same episodes, same masses: only how they are drawn differs.
    assert area_spread(hidden) < 0.2 * area_spread(visible)


def test_physical_radii_still_depend_on_mass_in_the_visual_condition() -> None:
    """Eq. 2 must survive: this is the whole reason absolute mass is observable."""
    dataset = _states_dataset(render_radius_from_mass=False, uniform_appearance=True)
    masses = dataset[0]["params"]
    radii = dataset.physical_radii(masses)
    assert radii is not None
    assert torch.allclose(radii, 0.08 * masses.squeeze(-1) / 1.5)


def test_frames_render_on_the_fly_from_preloaded_states() -> None:
    """Preload files store no frames; asking for them must draw from the states."""
    dataset = BounceDataset(
        num_episodes=4,
        render=True,
        preload="data/bounce_train_v2_100000.pt",
        **{**PHYSICS, "clip_len": 60, "seed": 0},
        render_radius_from_mass=False,
        uniform_appearance=True,
    )
    item = dataset[0]
    assert item["frames"].shape == (60, 3, 64, 64)
    assert torch.equal(item["frames"], dataset._render(item["states"], item["params"]))


def test_mass_independent_initialization_control() -> None:
    """The control must change the states and declare itself in the preload identity."""
    default = _states_dataset()
    control = _states_dataset(mass_independent_init=True)
    assert not torch.equal(default[0]["states"], control[0]["states"])
    assert control.generation_meta()["mass_independent_init"] is True
    assert "mass_independent_init" not in default.generation_meta()


def test_mass_independent_initialization_uses_worst_case_margins() -> None:
    """Every ball is placed as if it had r_max, so the support ignores its mass."""
    control = _states_dataset(mass_independent_init=True)
    r_max = 0.08 * 3.0 / 1.5
    for index in range(4):
        centres = control[index]["states"][0, :, :2]
        assert bool((centres >= r_max * 1.05 - 1e-6).all())
        assert bool((centres <= 1 - r_max * 1.05 + 1e-6).all())
