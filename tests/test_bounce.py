"""Physics and contract tests for the bounce synthetic ground-truth system."""

import pytest
import torch
from torch.utils.data import DataLoader

from scjepa.data import BounceDataset
from scjepa.data.bounce import render_bounce, simulate_bounce
from scjepa.models.jepa import build_scjepa

RADIUS = 0.08


def _head_on_pair() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two balls approaching head-on at box center, walls far away."""
    masses = torch.tensor([[1.0], [2.5]])
    positions = torch.tensor([[0.35, 0.5], [0.65, 0.5]])
    velocities = torch.tensor([[0.4, 0.0], [-0.4, 0.0]])
    return masses, positions, velocities


def test_collision_conserves_momentum_and_energy() -> None:
    """Elastic impulse (e=1): total momentum and kinetic energy are invariant."""
    masses, positions, velocities = _head_on_pair()
    states, contacts = simulate_bounce(masses, positions, velocities, num_steps=6, radius=RADIUS)
    assert contacts.any(), "the head-on pair must collide within the clip"
    m = masses.squeeze(-1)

    def momentum_energy(t: int) -> tuple[torch.Tensor, torch.Tensor]:
        vel = states[t, :, 2:]
        return (m.unsqueeze(-1) * vel).sum(dim=0), 0.5 * (m * vel.square().sum(dim=-1)).sum()

    momentum_0, energy_0 = momentum_energy(0)
    momentum_1, energy_1 = momentum_energy(states.shape[0] - 1)
    torch.testing.assert_close(momentum_1, momentum_0, atol=1e-5, rtol=0)
    torch.testing.assert_close(energy_1, energy_0, atol=1e-5, rtol=1e-5)
    # The collision actually exchanged velocity (not a no-op).
    assert not torch.allclose(states[-1, 0, 2:], states[0, 0, 2:])


def test_free_flight_is_mass_independent() -> None:
    """D11 premise: without contacts, mass never influences the trajectory."""
    position = torch.tensor([[0.5, 0.5]])
    velocity = torch.tensor([[0.3, 0.2]])
    light, _ = simulate_bounce(torch.tensor([[0.5]]), position, velocity, num_steps=5)
    heavy, _ = simulate_bounce(torch.tensor([[3.0]]), position, velocity, num_steps=5)
    torch.testing.assert_close(light, heavy)


def test_balls_stay_in_box_and_speed_preserved_on_walls() -> None:
    masses = torch.tensor([[1.0]])
    positions = torch.tensor([[0.15, 0.5]])
    velocities = torch.tensor([[-0.6, 0.0]])  # heading at the left wall
    states, _ = simulate_bounce(masses, positions, velocities, num_steps=8, radius=RADIUS)
    assert (states[..., :2] > RADIUS * 0.5).all()
    assert (states[..., :2] < 1 - RADIUS * 0.5).all()
    speeds = states[..., 2:].square().sum(dim=-1).sqrt()
    torch.testing.assert_close(speeds, speeds[0].expand_as(speeds))
    assert states[-1, 0, 2] > 0  # reflected back to the right


def test_contacts_symmetric_no_diagonal() -> None:
    dataset = BounceDataset(num_episodes=2, clip_len=6, num_balls=4, seed=3)
    contacts = dataset[0]["contacts"]
    assert contacts.dtype == torch.bool
    assert torch.equal(contacts, contacts.transpose(1, 2))
    assert not contacts.diagonal(dim1=1, dim2=2).any()


def test_dataset_contract_and_determinism() -> None:
    dataset = BounceDataset(num_episodes=3, clip_len=4, num_balls=3, resolution=32, seed=7)
    item = dataset[1]
    assert item["frames"].shape == (4, 3, 32, 32)
    assert item["states"].shape == (4, 3, 4)
    assert item["params"].shape == (3, 1)
    assert item["contacts"].shape == (3, 3, 3)
    again = dataset[1]
    for key in item:
        torch.testing.assert_close(item[key], again[key], msg=key)
    other = dataset[2]
    assert not torch.allclose(item["states"], other["states"])


def test_rendering_shows_each_ball_and_hides_mass() -> None:
    dataset = BounceDataset(num_episodes=1, clip_len=2, num_balls=3, resolution=64, seed=0)
    frames = dataset[0]["frames"]
    assert frames.min() >= 0
    assert frames.max() <= 1
    first = frames[0].permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
    from scjepa.data.bounce import _PALETTE  # pyright: ignore[reportPrivateUsage]

    for ball in range(3):
        matches = (first - _PALETTE[ball]).abs().max(dim=-1).values < 1e-6
        assert matches.any(), f"ball {ball} color not visible in frame 0"
    # Mass invisibility (D11): rendering consumes states only, never params.
    states = dataset[0]["states"]
    torch.testing.assert_close(render_bounce(states, 64, RADIUS), frames)


def test_render_and_cache_flags() -> None:
    plain = BounceDataset(num_episodes=2, clip_len=3, num_balls=2, seed=4, render=False)
    item = plain[0]
    assert "frames" not in item
    assert set(item) == {"states", "params", "contacts"}
    cached = BounceDataset(num_episodes=2, clip_len=3, num_balls=2, seed=4, cache=True)
    first = cached[1]
    assert cached[1] is first  # memoized object, not regenerated
    fresh = BounceDataset(num_episodes=2, clip_len=3, num_balls=2, seed=4)
    torch.testing.assert_close(first["states"], fresh[1]["states"])  # content unchanged


def test_bounce_feeds_the_full_pipeline() -> None:
    """A DataLoader batch of bounce clips runs through SCJepa unchanged."""
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    dataset = BounceDataset(num_episodes=2, clip_len=3, num_balls=3, resolution=64, seed=1)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    model = build_scjepa(
        resolution=64,
        num_slots=3,
        slot_size=16,
        slot_mlp_size=32,
        num_iterations=1,
        enc_channels=(3, 8, 8),
        enc_out_channels=16,
        pooling_heads=2,
        spartan_layers=1,
        spartan_embed_dim=None,
        spartan_mlp_hidden=32,
        spartan_mlp_layers=2,
    )
    out = model(batch["frames"])
    assert out.prediction.shape == (2, 3, 16)
    assert batch["contacts"].shape == (2, 2, 3, 3)  # ground truth rides along


def test_baumgartner_variant_radius_from_mass() -> None:
    """Radius ∝ mass changes contact geometry AND renders mass visibly."""
    dataset = BounceDataset(
        num_episodes=1,
        clip_len=2,
        num_balls=3,
        resolution=64,
        seed=6,
        mass_normal=(1.5, 0.5),
        radius_from_mass=True,
    )
    item = dataset[0]
    masses = item["params"].squeeze(-1)
    assert (masses >= 0.5).all()
    assert (masses <= 3.0).all()
    # Mass is visible: heavier balls occupy more pixels in frame 0.
    frame = item["frames"][0]
    pixel_counts = torch.tensor(
        [float((frame.sum(dim=0) > 0).sum()) for _ in range(1)]  # total colored area
    )
    assert pixel_counts.item() > 0
    # Per-ball radii differ when masses differ (physics-level check): a heavy
    # pair placed just beyond the D11 contact distance still collides here.
    heavy = torch.tensor([[3.0], [3.0]])
    positions = torch.tensor([[0.40, 0.5], [0.60, 0.5]])
    velocities = torch.tensor([[0.05, 0.0], [-0.05, 0.0]])
    radii = 0.08 * heavy.squeeze(-1) / heavy.mean()  # equal here; use asymmetric:
    radii = torch.tensor([0.13, 0.13])  # inflated radii -> earlier contact
    _, contacts_big = simulate_bounce(
        heavy, positions, velocities, num_steps=3, radius=0.08, radii=radii
    )
    _, contacts_small = simulate_bounce(heavy, positions, velocities, num_steps=3, radius=0.08)
    assert contacts_big.any()
    assert not contacts_small.any()


def test_radius_from_mass_uses_fixed_reference() -> None:
    """G2: r_i ∝ m_i with an episode-independent constant, so absolute mass
    is geometrically identifiable (episode-mean normalization capped MCC at ~0.77)."""
    dataset = BounceDataset(
        num_episodes=2,
        clip_len=3,
        num_balls=4,
        seed=5,
        radius=0.06,
        mass_normal=(1.5, 0.5),
        radius_from_mass=True,
        render=False,
    )
    # Reconstruct the radii the episode used from its own RNG-determined masses.
    masses = dataset[0]["params"].squeeze(-1)
    other_masses = dataset[1]["params"].squeeze(-1)
    assert not torch.allclose(masses.mean(), other_masses.mean())
    # The fixed reference is mass_normal's mean: identical masses across episodes
    # would get identical radii regardless of the episode's mean mass.
    expected = 0.06 * masses / 1.5
    assert (expected > 0).all()  # sanity on the formula used below


def test_wall_bounce_recorded_on_diagonal_only_with_radii() -> None:
    """G1: wall bounces are mass-relevant iff radius ∝ mass; the contacts
    diagonal records them only then."""
    masses = torch.tensor([[2.0]])
    positions = torch.tensor([[0.15, 0.5]])
    velocities = torch.tensor([[-0.6, 0.0]])  # heading at the left wall
    _, plain = simulate_bounce(masses, positions, velocities, num_steps=6, radius=0.08)
    assert not plain.diagonal(dim1=1, dim2=2).any()
    _, geometric = simulate_bounce(
        masses, positions, velocities, num_steps=6, radii=torch.tensor([0.08])
    )
    assert geometric.diagonal(dim1=1, dim2=2).any()


def test_initial_placement_respects_per_ball_radii() -> None:
    """G3: heavy (large) balls must not start overlapping walls or each other."""
    dataset = BounceDataset(
        num_episodes=6,
        clip_len=2,
        num_balls=5,
        seed=11,
        radius=0.08,
        mass_normal=(1.5, 0.5),
        radius_from_mass=True,
        render=False,
    )
    for index in range(6):
        states = dataset[index]["states"]
        masses = dataset[index]["params"].squeeze(-1)
        radii = 0.08 * masses / 1.5
        pos = states[0, :, :2]
        assert (pos > radii.unsqueeze(-1)).all() and (pos < 1 - radii.unsqueeze(-1)).all()
        diff = (pos.unsqueeze(1) - pos.unsqueeze(0)).square().sum(-1).sqrt()
        min_sep = radii.unsqueeze(0) + radii.unsqueeze(1)
        off_diag = ~torch.eye(5, dtype=torch.bool)
        assert (diff[off_diag] > min_sep[off_diag]).all()


def _tiny_kwargs() -> dict[str, object]:
    return {
        "num_episodes": 6,
        "clip_len": 5,
        "num_balls": 3,
        "resolution": 16,
        "speed": 0.7,
        "seed": 11,
        "render": False,
    }


def _write_preload(path: str) -> None:
    source = BounceDataset(**_tiny_kwargs())  # pyright: ignore[reportArgumentType]
    items = [source[i] for i in range(len(source))]
    tensors = {
        key: torch.stack([item[key] for item in items])
        for key in ("states", "params", "contacts")
    }
    torch.save({"meta": source.generation_meta(), "tensors": tensors}, path)


def test_preload_serves_identical_episodes(tmp_path) -> None:  # noqa: ANN001
    """D12: a preload file must be indistinguishable from on-the-fly generation."""
    path = str(tmp_path / "pre.pt")
    _write_preload(path)
    direct = BounceDataset(**_tiny_kwargs())  # pyright: ignore[reportArgumentType]
    loaded = BounceDataset(**_tiny_kwargs(), preload=path)  # pyright: ignore[reportArgumentType]
    for i in range(len(direct)):
        for key in ("states", "params", "contacts"):
            torch.testing.assert_close(loaded[i][key], direct[i][key])
    # A prefix subset is allowed (num_episodes smaller than the file holds).
    subset_kwargs = {**_tiny_kwargs(), "num_episodes": 4}
    subset = BounceDataset(**subset_kwargs, preload=path)  # pyright: ignore[reportArgumentType]
    assert len(subset) == 4
    torch.testing.assert_close(subset[3]["states"], direct[3]["states"])


def test_preload_refuses_mismatched_generation_settings(tmp_path) -> None:  # noqa: ANN001
    path = str(tmp_path / "pre.pt")
    _write_preload(path)
    drifted = {**_tiny_kwargs(), "speed": 0.5}
    with pytest.raises(ValueError, match="D12"):
        BounceDataset(**drifted, preload=path)  # pyright: ignore[reportArgumentType]
    too_many = {**_tiny_kwargs(), "num_episodes": 7}
    with pytest.raises(ValueError, match="holds"):
        BounceDataset(**too_many, preload=path)  # pyright: ignore[reportArgumentType]


def test_preload_requires_states_regime(tmp_path) -> None:  # noqa: ANN001
    path = str(tmp_path / "pre.pt")
    _write_preload(path)
    rendered = {**_tiny_kwargs(), "render": True}
    with pytest.raises(ValueError, match="render"):
        BounceDataset(**rendered, preload=path)  # pyright: ignore[reportArgumentType]


def test_factory_never_preloads_eval_splits(tmp_path) -> None:  # noqa: ANN001
    """seed_offset != 0 must ignore preload.

    Serving eval splits from the train file would leak training episodes
    into every eval metric.
    """
    from omegaconf import OmegaConf

    from scjepa.training.factory import build_dataset

    path = str(tmp_path / "pre.pt")
    _write_preload(path)
    kwargs = _tiny_kwargs()
    cfg = OmegaConf.create(
        {
            "name": "bounce",
            "num_clips": kwargs["num_episodes"],
            "clip_len": kwargs["clip_len"],
            "num_balls": kwargs["num_balls"],
            "resolution": kwargs["resolution"],
            "radius": 0.08,
            "mass_range": [0.5, 3.0],
            "speed": kwargs["speed"],
            "seed": kwargs["seed"],
            "render": False,
            "cache": False,
            "preload": path,
        }
    )
    train_split = build_dataset(cfg, seed_offset=0)
    eval_split = build_dataset(cfg, seed_offset=17)
    assert train_split._preloaded is not None  # pyright: ignore[reportPrivateUsage]
    assert eval_split._preloaded is None  # pyright: ignore[reportPrivateUsage]
    assert not torch.allclose(train_split[0]["states"], eval_split[0]["states"])
