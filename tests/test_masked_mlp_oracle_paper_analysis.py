"""Statistics, chance references, graph alignment and end-to-end paper outputs."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from diagnostics.balanced_action_oracle.data import PrepareConfig, prepare
from diagnostics.isotropic_oracle.data import PrepareConfig as SourcePrepareConfig
from diagnostics.isotropic_oracle.data import prepare as prepare_source
from diagnostics.masked_mlp_oracle import from_dense, paper_analysis
from diagnostics.masked_mlp_oracle import run as runner


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_sign_flip_test_is_exact() -> None:
    result = paper_analysis.paired([0.9, 0.8, 0.95], [0.7, 0.75, 0.8])
    # All three differences positive: only the all-positive assignment reaches the mean.
    assert result["positive"] == 3
    assert result["p_one_sided"] == pytest.approx(1 / 8)  # pyright: ignore[reportUnknownMemberType]
    assert result["p_two_sided"] == pytest.approx(2 / 8)  # pyright: ignore[reportUnknownMemberType]
    assert result["mean"] == pytest.approx(np.mean([0.2, 0.05, 0.15]))  # pyright: ignore[reportUnknownMemberType]
    tied = paper_analysis.paired([0.5, 0.5], [0.5, 0.5])
    assert tied["p_one_sided"] == 1.0


def test_chance_levels_match_their_limits() -> None:
    singletons = paper_analysis.chance_levels((1, 1, 1, 1, 1, 1), draws=50)
    assert singletons["mcc_mean"] == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]
    within = paper_analysis.chance_levels((3, 3), draws=2000)
    anywhere = paper_analysis.chance_levels((6,), draws=2000)
    assert within["purity_mean"] == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]
    assert 0.80 < within["mcc_mean"] < 0.86
    assert 0.66 < anywhere["mcc_mean"] < 0.72
    assert anywhere["purity_mean"] < 0.8


def test_graphs_are_aligned_by_the_matching() -> None:
    columns = [1, 0, 2, 3, 4, 5]
    state = torch.full((6, 6), -1.0)
    state[1, 0] = 1.0  # learned output 1 uses learned input 0
    graphs = paper_analysis._aligned_graphs(  # pyright: ignore[reportPrivateUsage]
        {"predictor.state_logits": state}, columns
    )
    # True coordinate 0 is learned 1 and true 1 is learned 0, so the edge is (0, 1).
    assert graphs["state"][0, 1]
    assert int(graphs["state"].sum()) == 1
    assert bool(graphs["parameter"].all())


def test_end_to_end_outputs_on_a_smoke_batch(tmp_path: Path) -> None:
    prepare_source(
        SourcePrepareConfig(
            str(tmp_path / "source"),
            train_groups=4,
            calibration_groups=2,
            validation_groups=2,
            test_groups=2,
            profile="smoke",
        )
    )
    prepare(PrepareConfig(str(tmp_path / "data"), str(tmp_path / "source"), profile="smoke"))
    batch = tmp_path / "batch"

    def config(condition: runner.Condition, seed: int) -> runner.TrainConfig:
        return runner.TrainConfig(
            str(batch / f"{condition}-s{seed}/results"),
            str(tmp_path / "data"),
            condition,
            profile="smoke",
            seed=seed,
            steps=3,
            batch_size=2,
            log_every=1,
            checkpoint_every=1,
            evaluate_every=2,
            calibration_dir=str(batch / "calibration/results") if condition != "dense" else None,
        )

    for seed in (0, 1):
        runner.train(config("dense", seed))
    runner.calibrate(
        runner.CalibrateConfig(
            str(batch / "calibration/results"), str(batch / "dense-s0/results"), profile="smoke"
        )
    )
    for condition in ("baseline", "full"):
        for seed in (0, 1):
            from_dense.train(
                from_dense.FromDenseConfig(
                    config(condition, seed), str(batch / f"dense-s{seed}/results"), sparse_steps=4
                )
            )
    result = paper_analysis.analyse(batch, tmp_path / "paper")
    assert result["seeds"] == [0, 1]
    for name in (
        "fig_mcc_box.pdf",
        "fig_mcc_curves.png",
        "fig_graph_recovery.pdf",
        "table_main.tex",
        "table_paired.tex",
        "per_run.csv",
        "summary.json",
        "README.md",
    ):
        assert (tmp_path / "paper" / name).is_file()
    assert result["paired"]["full - baseline"]["n"] == 2
    rows = (tmp_path / "paper/per_run.csv").read_text().splitlines()
    assert len(rows) == 1 + 3 * 2


def test_spreads_that_round_to_zero_are_printed_as_bounds() -> None:
    fmt = paper_analysis._fmt  # pyright: ignore[reportPrivateUsage]
    assert fmt({"mean": 0.99498, "sd": 0.00004}) == r"0.995 $\pm$ $<$0.001"
    assert fmt({"mean": 0.792, "sd": 0.019}) == r"0.792 $\pm$ 0.019"
    assert fmt({"mean": 0.0089, "sd": 0.00004}, 4) == r"0.0089 $\pm$ $<$0.0001"


def test_learned_code_batches_add_code_figures_and_table(tmp_path: Path) -> None:
    from diagnostics.masked_mlp_learned import run as learned

    prepare_source(
        SourcePrepareConfig(
            str(tmp_path / "source"),
            train_groups=4,
            calibration_groups=4,
            validation_groups=2,
            test_groups=2,
            profile="smoke",
        )
    )
    prepare(PrepareConfig(str(tmp_path / "data"), str(tmp_path / "source"), profile="smoke"))
    batch = tmp_path / "batch"

    def config(condition: learned.Condition, seed: int) -> learned.TrainConfig:
        sparse = condition != "dense"
        return learned.TrainConfig(
            str(batch / f"{condition}-s{seed}/results"),
            str(tmp_path / "data"),
            condition,
            profile="smoke",
            seed=seed,
            steps=3,
            batch_size=2,
            log_every=1,
            checkpoint_every=1,
            evaluate_every=2,
            calibration_dir=str(batch / "calibration/results") if sparse else None,
            init_dir=str(batch / f"projected-s{seed}/results") if sparse else None,
        )

    for seed in (0, 1):
        learned.train(config("dense", seed))
        learned.project(
            learned.ProjectConfig(
                str(batch / f"projected-s{seed}/results"),
                str(batch / f"dense-s{seed}/results"),
                str(tmp_path / "data"),
                profile="smoke",
            )
        )
    learned.calibrate(
        learned.CalibrateConfig(
            str(batch / "calibration/results"),
            str(batch / "projected-s0/results"),
            profile="smoke",
        )
    )
    for condition in ("baseline", "full"):
        for seed in (0, 1):
            learned.train(config(condition, seed))
    result = paper_analysis.analyse(batch, tmp_path / "paper")
    assert result["learned_codes"]
    for name in (
        "fig_code_retention_box.pdf",
        "fig_code_correlations_median_seed.png",
        "table_codes.tex",
        "table_projection.tex",
    ):
        assert (tmp_path / "paper" / name).is_file()
    assert sorted(result["projections"]) == ["0", "1"]
    assert result["projections"]["0"]["source_code_dim"] == 12
    assert "learned parameter codes" in (tmp_path / "paper/README.md").read_text()
