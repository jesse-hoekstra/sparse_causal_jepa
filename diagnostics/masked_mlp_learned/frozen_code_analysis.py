"""Collect the fixed five-seed frozen-code control without opening test data."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import torch

from diagnostics.masked_mlp_learned.frozen_code import STEPS, verify_completion
from diagnostics.masked_mlp_oracle import paper_analysis as paper
from diagnostics.shared.io import read_json, sha
from scjepa.experiments.storage import write_json

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ARMS = ("baseline", "full")
SEEDS = (0, 1, 2, 3, 4)
METRICS = (
    "state_mcc",
    "code_mcc",
    "code_r2_min",
    "code_r2_mean",
    "block_purity",
    "forecast/normalized_mse",
    "forecast/code_use_ratio",
    "geco/relative_prediction",
    "masks/state_aligned_shd",
    "masks/action_aligned_shd",
    paper.CODE_ALIGNED_SHD,
)


def analyse(batch: Path, output: Path) -> dict[str, Any]:
    """Verify every final endpoint, preserve all outcomes, and summarize paired seeds."""
    rows: list[dict[str, Any]] = []
    records: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for seed in SEEDS:
        for arm in ARMS:
            directory = batch / f"{arm}-s{seed}" / "results"
            completion = verify_completion(directory)
            if (completion["seed"], completion["condition"], completion["steps"]) != (
                seed,
                arm,
                STEPS,
            ):
                raise ValueError("endpoint differs from the five-seed fixed-budget declaration")
            metrics = read_json(directory / "validation.json")["metrics"]
            startup = read_json(directory / "startup.json")["metrics"]["validation"]
            records[arm][seed] = completion
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "additional_updates": STEPS,
                    "initial_state_mcc": startup["state_mcc"],
                    **{key: metrics[key] for key in METRICS},
                    "identity": completion["identity"],
                    "completion_sha256": sha(directory / "completion.json"),
                }
            )
        for key in (
            "baseline_parent_identity",
            "donor_parent_identity",
            "frozen_identity",
            "data_identity",
            "tau_prediction",
        ):
            if records["baseline"][seed][key] != records["full"][seed][key]:
                raise ValueError(f"paired arms differ in {key}")
        for split in ("calibration", "validation"):
            codes: list[torch.Tensor] = []
            for arm in ARMS:
                payload = cast(
                    dict[str, Any],
                    torch.load(
                        batch / f"{arm}-s{seed}" / "results" / f"{split}.pt",
                        map_location="cpu",
                        weights_only=False,
                    ),
                )
                codes.append(payload["samples"]["codes"])
            if not torch.equal(codes[0], codes[1]):
                raise ValueError("paired arms did not receive identical frozen codes")

    def values(arm: str, metric: str) -> list[float]:
        return [float(row[metric]) for row in rows if row["arm"] == arm]

    result: dict[str, Any] = {
        "schema": "masked-mlp-frozen-code-analysis-v1",
        "split": "validation",
        "seeds": list(SEEDS),
        "additional_updates": STEPS,
        "test_evaluated": False,
        "paired_frozen_codes_identical": True,
        "arms": {
            arm: {metric: paper.describe(values(arm, metric)) for metric in METRICS} for arm in ARMS
        },
        "full_minus_baseline_state_mcc": paper.paired(
            values("full", "state_mcc"), values("baseline", "state_mcc")
        ),
        "all_full_parameters_retained_r2_080": all(
            value >= 0.8 for value in values("full", "code_r2_min")
        ),
        "all_full_graphs_exact": all(
            value == 0 for metric in METRICS[-3:] for value in values("full", metric)
        ),
        "code_mcc_assignment": "fitted on calibration, scored on validation",
        "parameter_graph_assignment": "calibration-fitted linear pair matching",
        "interpretation": (
            "Frozen full-model code transfer; state learning starts from the baseline. "
            "This does not establish joint acquisition of entangled codes from scratch."
        ),
        "runs": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", result)
    with (output / "per_run.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.6), layout="constrained")
    for axis, metric, title in zip(
        axes,
        ("state_mcc", "code_mcc"),
        ("State coordinates", "Frozen parameter coordinates"),
        strict=True,
    ):
        samples = [values(arm, metric) for arm in ARMS]
        axis.boxplot(samples, positions=[1, 2], widths=0.4, whis=1.5, showfliers=False)
        offsets = np.linspace(-0.1, 0.1, len(SEEDS))
        for i, sample in enumerate(samples):
            axis.scatter(i + 1 + offsets, sample, s=26, zorder=3)
        for i in range(len(SEEDS)):
            axis.plot(
                [1 + offsets[i], 2 + offsets[i]],
                [sample[i] for sample in samples],
                color="0.7",
                alpha=0.7,
                linewidth=0.7,
                zorder=0,
            )
        axis.set_xticks([1, 2], ["Parameter gates open", "Parameter gates sparse"])
        axis.tick_params(axis="x", labelsize=8)
        axis.set(title=title, ylabel="Validation MCC", ylim=(0, 1.02))
        axis.grid(axis="y", alpha=0.2)
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"fig_frozen_code_mcc.{suffix}", dpi=300)  # pyright: ignore[reportUnknownMemberType]
    plt.close(figure)
    (output / "README.md").write_text(
        "# Frozen, pair-mixed parameter-code control\n\n"
        "Five matched seeds; 300,000 additional updates from the baseline state/predictor "
        "weights. Both arms receive identical fixed codes from the full model's frozen "
        "observation/parameter encoders, scaled on training data and rotated within "
        "identical learned-mask column pairs. Only the full arm sparsifies parameter gates.\n\n"
        "All endpoint scores use validation. Assignments and probes use calibration. "
        "The figure shows all five seeds, paired lines and standard 1.5-IQR whiskers. "
        "No test data or manuscript changes are part of this control. All outcomes appear "
        "in per_run.csv and summary.json.\n"
    )
    return result


def main() -> None:
    """Export the complete fixed-budget validation comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = analyse(arguments.batch_dir, arguments.output_dir)
    print(json.dumps(result["full_minus_baseline_state_mcc"]))


if __name__ == "__main__":
    main()
