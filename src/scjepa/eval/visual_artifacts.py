"""Two compact final-report figures: slot allocations and mass recovery."""

# Matplotlib's kwargs and IO annotations are incomplete in its shipped stubs.
# pyright: reportUnknownMemberType=false

from pathlib import Path

import matplotlib

from scjepa.eval.visual_to_visual import VisualToVisualReport

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_visual_artifacts(report: VisualToVisualReport, run_dir: Path) -> tuple[Path, Path]:
    """Save one fixed episode's slot maps and the shared MCC recovery grid."""
    example = report.slot_example
    frames, allocations = example["frames"], example["allocations"]
    target_maps = example["target_allocations"]
    slots, resolution = allocations.shape[1], frames.shape[-1]
    figure, axes = plt.subplots(
        slots + 1, frames.shape[0], figsize=(9, 1.35 * (slots + 1)), layout="constrained"
    )
    heatmap = None
    for column, frame in enumerate(frames):
        axes[0, column].imshow(frame.permute(1, 2, 0).numpy(), origin="upper")
        axes[0, column].set_title(f"t = {int(example['timesteps'][column])}")
        for identity, centre in enumerate(example["true_centres"][column]):
            axes[0, column].text(
                float(centre[0]) * resolution - 0.5,
                float(centre[1]) * resolution - 0.5,
                str(identity + 1),
                color="tab:red",
                ha="center",
                va="center",
                fontsize=8,
            )
        for slot in range(slots):
            axis = axes[slot + 1, column]
            allocation = (
                allocations[column, slot].reshape(resolution, resolution).numpy() * resolution**2
            )
            target = (
                target_maps[column, slot].reshape(resolution, resolution).numpy() * resolution**2
            )
            heatmap = axis.imshow(allocation, cmap="magma", origin="upper", vmin=0, vmax=5)
            # Fixed units shared across all slots/times prevent tiny departures
            # from uniform attention looking like discovered objects.
            axis.contour(
                frame.mean(dim=0).numpy(), levels=[0.5], colors=["white"], linewidths=0.5, alpha=0.7
            )
            # A contour of the aligned EMA allocation shows branch disagreement
            # without colouring the actual input balls by simulator identity.
            if float(target.min()) < 2.0 < float(target.max()):
                axis.contour(target, levels=[2.0], colors=["cyan"], linewidths=0.6)
            if column == 0:
                axis.set_ylabel(f"slot {slot + 1}\nobject {int(example['assignment'][slot]) + 1}")
        for row in range(slots + 1):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.suptitle(
        "Input and slot allocations (fixed scale; white: input balls; cyan: EMA at 2x uniform)\n"
        "Object numbers are evaluation labels only",
        fontsize=10,
    )
    assert heatmap is not None
    figure.colorbar(
        heatmap,
        ax=axes[1:].ravel().tolist(),
        shrink=0.5,
        label="allocation / uniform",
        ticks=[0, 1, 2, 3, 4, 5],
    )
    slot_path = run_dir / "slot_tracks.png"
    figure.savefig(slot_path, dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(slots, slots, figsize=(1.6 * slots, 1.6 * slots), squeeze=False)
    best = report.recovery_matrix.argmax(dim=1)
    for mass in range(slots):
        for learned in range(slots):
            axis = axes[mass, learned]
            axis.scatter(
                report.learned_params[:, learned].numpy(),
                report.true_masses[:, mass].numpy(),
                s=2,
                alpha=0.2,
            )
            axis.text(
                0.04,
                0.94,
                f"R²={report.recovery_matrix[mass, learned]:.2f}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if int(best[mass]) == learned:
                for spine in axis.spines.values():
                    spine.set_color("tab:green")
                    spine.set_linewidth(2)
            if learned == 0:
                axis.set_ylabel(f"mass {mass + 1}")
            if mass == slots - 1:
                axis.set_xlabel(f"coordinate {learned + 1}")
    figure.suptitle(f"Mass recovery (MCC={report.metrics['mcc']:.3f}; green: best coordinate)")
    figure.tight_layout()
    recovery_path = run_dir / "recovery_grid.png"
    figure.savefig(recovery_path, dpi=150)
    plt.close(figure)
    return slot_path, recovery_path
