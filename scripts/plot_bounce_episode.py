"""Plot one Bounce episode as a 5x5 contact sheet, in either observation condition.

Two conditions, matching the two experiment configs:

``--condition states`` (the state-to-state regime, ``experiment=bounce_baumgartner``)
    A vector drawing: one disc per ball at its PHYSICAL collision radius, with a
    per-track palette colour. This illustrates the data-generating process; it is
    not what any model sees in the state-to-state regime, which reads [px, py, vx, vy] rows.

``--condition visual`` (the visual-to-state regime, ``experiment=bounce_visual``)
    The ACTUAL rendered frames, pixel for pixel, as the SAVi encoder receives
    them: identical white glyphs at one shared rendered radius, no persistent
    colour identity. Physical collision radii stay mass-dependent (Eq. 2), so
    ``--show-collision-radii`` overlays them as dashed circles — an ANNOTATION,
    absent from the data, that makes visible why most bounces happen with a gap
    between the drawn discs.

Usage:
    python scripts/plot_bounce_episode.py --condition visual --episode 0
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from scjepa.data.bounce import _PALETTE, BounceDataset

# The configured Bounce v2 physics (experiments.pdf S6.1.1). Identical in both
# conditions -- only the appearance arguments below it differ.
PHYSICS = dict(
    clip_len=60,
    num_balls=5,
    resolution=64,
    radius=0.08,
    mass_range=(0.5, 3.0),
    mass_normal=(1.5, 0.5),
    radius_from_mass=True,
    speed=0.7,
    seed=0,
)
GRID = (5, 5)
DT = 0.1


def _dataset(condition: str, episodes: int, preload: str | None) -> BounceDataset:
    """Build the dataset for one observation condition."""
    visual = condition == "visual"
    return BounceDataset(
        num_episodes=episodes,
        render=visual,
        render_radius_from_mass=None if not visual else False,
        uniform_appearance=visual,
        preload=preload,
        **PHYSICS,  # pyright: ignore[reportArgumentType]
    )


def main() -> None:
    """Render one episode's contact sheet to a PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("states", "visual"), default="visual")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--preload", default="data/bounce_train_v2_100000.pt")
    parser.add_argument("--out", default=None)
    parser.add_argument("--show-collision-radii", action="store_true")
    args = parser.parse_args()

    dataset = _dataset(args.condition, args.episode + 1, args.preload or None)
    item = dataset[args.episode]
    states, masses = item["states"], item["params"].squeeze(-1)
    radii = dataset.physical_radii(item["params"])
    assert radii is not None
    steps = states.shape[0]
    picks = [round(i * (steps - 1) / (GRID[0] * GRID[1] - 1)) for i in range(GRID[0] * GRID[1])]

    figure, axes = plt.subplots(*GRID, figsize=(15, 15.5))
    for axis, frame in zip(axes.flat, picks, strict=True):
        axis.set_title(f"t = {frame * DT:.1f} s", fontsize=11)
        axis.set_xlim(0, 1)
        axis.set_ylim(1, 0)  # image row = y, matching the renderer
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
        if args.condition == "visual":
            axis.imshow(item["frames"][frame].permute(1, 2, 0).numpy(), extent=(0, 1, 1, 0))
        else:
            axis.set_facecolor("#fafafa")
            for ball in range(states.shape[1]):
                centre = (float(states[frame, ball, 0]), float(states[frame, ball, 1]))
                axis.add_patch(
                    Circle(
                        centre,
                        float(radii[ball]),
                        facecolor=tuple(_PALETTE[ball].tolist()),
                        edgecolor="black",
                        linewidth=0.6,
                    )
                )
        if args.show_collision_radii:
            for ball in range(states.shape[1]):
                centre = (float(states[frame, ball, 0]), float(states[frame, ball, 1]))
                axis.add_patch(
                    Circle(
                        centre,
                        float(radii[ball]),
                        facecolor="none",
                        edgecolor="#ff3b30",
                        linewidth=0.9,
                        linestyle=(0, (3, 2)),
                    )
                )

    if args.condition == "visual":
        handles = [
            plt.Line2D([], [], marker="o", color="none", markerfacecolor="white",
                       markeredgecolor="black", markersize=11,
                       label="every object: identical glyph, rendered r = 0.080"),
        ]
        if args.show_collision_radii:
            handles.append(
                plt.Line2D([], [], color="#ff3b30", linestyle=(0, (3, 2)), linewidth=1.4,
                           label="physical collision radius (annotation; NOT in the data)")
            )
        caption = (
            "Visual-regime input: exactly the 64x64 frames the encoder receives. "
            "All objects share "
            "one mass-independent glyph and rendered radius,\nand carry no persistent colour "
            "identity, so a single frame cannot reveal mass. The physical collision radii below "
            "remain mass-dependent,\nwhich is why most bounces occur with a visible gap between "
            "the discs -- the 'hidden contact geometry' a sequence can reveal."
        )
    else:
        handles = [
            plt.Line2D([], [], marker="o", color="none",
                       markerfacecolor=tuple(_PALETTE[b].tolist()), markeredgecolor="black",
                       markersize=11, label=f"object {b + 1}: m = {float(masses[b]):.2f}")
            for b in range(states.shape[1])
        ]
        caption = (
            "Circle radii are the physical collision radii; "
            "colours distinguish simulator tracks."
        )

    figure.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5),
                  frameon=False, bbox_to_anchor=(0.5, 0.085), fontsize=11)
    masses_text = "   ".join(
        f"object {b + 1}: m = {float(masses[b]):.2f} (r_phys = {float(radii[b]):.3f})"
        for b in range(states.shape[1])
    )
    if args.condition == "visual":
        figure.text(0.5, 0.058, masses_text, ha="center", va="bottom", fontsize=9, color="#333333")
    figure.text(0.5, 0.008, caption, ha="center", va="bottom", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0, 0.115, 1, 1))

    tag = "v3_visual" if args.condition == "visual" else "v2"
    out = args.out or f"data/bounce_{tag}_episode_{args.episode:05d}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
