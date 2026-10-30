"""Statistics, chance references, graph alignment and end-to-end paper outputs."""

import csv
import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from diagnostics.balanced_action_oracle.data import PrepareConfig, prepare
from diagnostics.isotropic_oracle.data import PrepareConfig as SourcePrepareConfig
from diagnostics.isotropic_oracle.data import prepare as prepare_source
from diagnostics.isotropic_oracle.dgp import graphs as true_graphs
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


def test_retention_figure_displays_every_seed_including_failed_outliers() -> None:
    seeds = list(range(5))
    runs: dict[str, dict[int, paper_analysis.Run]] = {}
    expected: list[float] = []
    for arm in paper_analysis.ARMS:
        runs[arm] = {}
        for seed in seeds:
            scores = [0.92 + 0.005 * seed - 0.001 * k for k in range(6)]
            if arm == "full" and seed < 2:
                scores[0] = (-0.2, 0.75)[seed]
            expected.extend(scores)
            runs[arm][seed] = paper_analysis.Run(
                arm=arm,
                seed=seed,
                directory=Path("unused"),
                metrics={f"code_r2/theta_{k + 1}": value for k, value in enumerate(scores)},
                start=None,
                curve=[],
                penalty=[],
                columns=list(range(6)),
                correlation=np.eye(6),
                graphs={},
                group_moments=np.zeros((1, 31)),
                physical_samples_sha256="unused",
                data_identity="unused",
            )
    figure = paper_analysis.figure_code_retention(runs, seeds)
    axis: Any = figure.axes[0]
    points = cast(
        paper_analysis.Array,
        np.concatenate([np.asarray(c.get_offsets()) for c in axis.collections]),
    )
    assert len(points) == 90
    np.testing.assert_array_equal(np.sort(points[:, 1]), np.sort(expected))
    low, high = axis.get_ylim()
    assert low < min(expected) < max(expected) < high
    paper_analysis.plt.close(figure)


def test_group_bootstrap_matches_explicit_resampling_with_fixed_coordinate_matches() -> None:
    generator = torch.Generator().manual_seed(73)
    truth = torch.randn(7, 5, 6, generator=generator, dtype=torch.float64)
    permutation = [3, 2, 0, 1, 5, 4]
    columns = [2, 3, 1, 0, 5, 4]
    left = truth[:, :, permutation] + 0.2 * torch.randn(
        7, 5, 6, generator=generator, dtype=torch.float64
    )
    right = truth + 0.8 * torch.randn(7, 5, 6, generator=generator, dtype=torch.float64)
    a = paper_analysis.matched_group_moments(left, truth, columns)
    b = paper_analysis.matched_group_moments(right, truth, list(range(6)))
    result = paper_analysis.paired_group_mcc(a, b, draws=32, seed=42)
    indices = np.random.default_rng(42).integers(7, size=(32, 7))

    def direct(sample: list[int]) -> float:
        target = truth[sample].reshape(-1, 6)
        values: list[float] = []
        for states, order in ((left, columns), (right, list(range(6)))):
            correlation = paper_analysis.pearson_matrix(states[sample].reshape(-1, 6), target)
            values.append(float(correlation[torch.arange(6), order].abs().mean()))
        return values[0] - values[1]

    differences = [direct([int(i) for i in sample]) for sample in indices]
    assert result["estimate"] == pytest.approx(direct(list(range(7))), abs=1e-12)  # pyright: ignore[reportUnknownMemberType]
    assert result["ci_low"] == pytest.approx(float(np.percentile(differences, 2.5)), abs=1e-12)  # pyright: ignore[reportUnknownMemberType]
    assert result["ci_high"] == pytest.approx(float(np.percentile(differences, 97.5)), abs=1e-12)  # pyright: ignore[reportUnknownMemberType]
    identical = paper_analysis.paired_group_mcc(a, a)
    assert identical["estimate"] == identical["ci_low"] == identical["ci_high"] == 0


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


def test_learned_code_columns_are_matched_to_parameter_pairs() -> None:
    generator = torch.Generator().manual_seed(0)
    theta = torch.randn(4000, 6, generator=generator, dtype=torch.float64)
    # Codes in a scrambled order, each mixing the two parameters of one pair.
    codes = torch.stack(
        [
            theta[:, 1],
            theta[:, 3] + 0.5 * theta[:, 0],
            theta[:, 2] - 0.3 * theta[:, 5],
            theta[:, 4] + 0.8 * theta[:, 1],
            theta[:, 0] - theta[:, 3],
            theta[:, 5],
        ],
        1,
    )
    order = paper_analysis._code_order(codes, theta)  # pyright: ignore[reportPrivateUsage]
    assert {order[0], order[3]} == {1, 4}
    assert {order[1], order[4]} == {0, 3}
    assert {order[2], order[5]} == {2, 5}
    # Within a pair |correlation| decides: code 4 tracks theta_1, code 1 tracks theta_4.
    assert (order[0], order[3]) == (4, 1)
    truth = true_graphs().parameter.numpy()
    scrambled = np.zeros_like(truth)
    scrambled[:, order] = truth
    aligned = paper_analysis._code_aligned_metrics(  # pyright: ignore[reportPrivateUsage]
        scrambled[:, order]
    )
    assert aligned["masks/parameter_code_aligned_shd"] == 0.0


def test_graph_summaries_keep_row_only_and_code_aligned_scores_distinct(tmp_path: Path) -> None:
    runs: dict[str, dict[int, paper_analysis.Run]] = {}
    for arm in paper_analysis.ARMS:
        metrics = {
            "state_mcc": {"dense": 0.6, "baseline": 0.8, "full": 0.99}[arm],
            "block_purity": 0.99,
            "normalized_prediction_mse": 0.01,
            "online_state_probe/r2": 0.99,
            "masks/state_aligned_shd": 0.0,
            "masks/parameter_aligned_shd": 20.0 if arm == "full" else 24.0,
        }
        if arm == "full":
            metrics[paper_analysis.CODE_ALIGNED_SHD] = 1.0
        runs[arm] = {
            0: paper_analysis.Run(
                arm=arm,
                seed=0,
                directory=Path("unused"),
                metrics=metrics,
                start=None,
                curve=[],
                penalty=[],
                columns=list(range(6)),
                correlation=np.eye(6),
                graphs={},
                group_moments=np.zeros((1, 31)),
                physical_samples_sha256="unused",
                data_identity="unused",
            )
        }
    chance = {
        "within_block": {"mcc_mean": 0.83, "purity_mean": 1.0},
        "any_rotation": {"mcc_mean": 0.69, "purity_mean": 0.71},
    }
    result = paper_analysis.write_tables(runs, [0], tmp_path, chance)
    full = result["summary"]["full"]
    assert full["masks/parameter_aligned_shd"]["mean"] == 20.0
    assert full[paper_analysis.CODE_ALIGNED_SHD]["mean"] == 1.0
    assert result["parameter_graph_metric"] == paper_analysis.CODE_ALIGNED_SHD
    row = next(
        line
        for line in (tmp_path / "table_main.tex").read_text().splitlines()
        if line.startswith("Full &")
    )
    assert row.endswith(r"1.0 $\pm$ 0.0 \\")


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
        "group_mcc_contrasts.json",
        "README.md",
    ):
        assert (tmp_path / "paper" / name).is_file()
    assert result["paired"]["full - baseline"]["n"] == 2
    assert result["paired_group_mcc"]["full - baseline"]["0"]["groups"] == 2
    rows = (tmp_path / "paper/per_run.csv").read_text().splitlines()
    assert len(rows) == 1 + 3 * 2


def test_spreads_that_round_to_zero_are_printed_as_bounds() -> None:
    fmt = paper_analysis._fmt  # pyright: ignore[reportPrivateUsage]
    assert fmt({"mean": 0.99498, "sd": 0.00004}) == r"0.995 $\pm$ $<$0.001"
    assert fmt({"mean": 0.792, "sd": 0.019}) == r"0.792 $\pm$ 0.019"
    assert fmt({"mean": 0.0089, "sd": 0.00004}, 4) == r"0.0089 $\pm$ $<$0.0001"


@pytest.mark.parametrize("adapt", [False, True])
def test_learned_code_batches_add_code_figures_and_table(tmp_path: Path, adapt: bool) -> None:
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
            init_dir=str(batch / f"{'dense' if adapt else 'projected'}-s{seed}/results")
            if sparse
            else None,
        )

    for seed in (0, 1):
        initial = config("dense", seed)
        if adapt:
            initial = replace(initial, output_dir=str(batch / f"acquired-s{seed}/results"))
        learned.train(initial)
        learned.project(
            learned.ProjectConfig(
                str(batch / f"projected-s{seed}/results"),
                initial.output_dir,
                str(tmp_path / "data"),
                profile="smoke",
            )
        )
        if adapt:
            learned.train(
                replace(config("dense", seed), adapt_from=str(batch / f"projected-s{seed}/results"))
            )
    learned.calibrate(
        learned.CalibrateConfig(
            str(batch / "calibration/results"),
            str(batch / f"{'dense' if adapt else 'projected'}-s0/results"),
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
    with (tmp_path / "paper/per_run.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        aligned = row["masks/parameter_code_aligned_shd"]
        assert (aligned != "") == (row["arm"] == "full")
        saved = result["per_run"][row["arm"]][row["seed"]]
        assert saved["masks/parameter_aligned_shd"] == float(row["masks/parameter_aligned_shd"])
        if row["arm"] == "full":
            assert saved[paper_analysis.CODE_ALIGNED_SHD] == float(aligned)
        assert "geco/relative_prediction" in saved or row["arm"] == "dense"
    assert result["parameter_graph_metric"] == paper_analysis.CODE_ALIGNED_SHD
    assert "pair $R^2$" in (tmp_path / "paper/table_main.tex").read_text()
    if adapt:
        assert result["adaptations"]["0"]["updates"] == 3
        assert (tmp_path / "paper/table_adaptation.tex").is_file()
        assert "frozen learned states" in (tmp_path / "paper/README.md").read_text()
        completion_path = batch / "dense-s0/results/completion.json"
        completion = json.loads(completion_path.read_text())
        original = completion_path.read_text()
        completion["adaptation_source"]["identity"] = "wrong parent"
        completion_path.write_text(json.dumps(completion))
        with pytest.raises(ValueError, match="declared projection"):
            paper_analysis.load_adaptations(batch, [0, 1])
        completion_path.write_text(original)
        sparse_path = batch / "full-s0/results/completion.json"
        sparse = json.loads(sparse_path.read_text())
        sparse["dense_endpoint"]["identity"] = "different initializer"
        sparse_path.write_text(json.dumps(sparse))
        with pytest.raises(ValueError, match="reported adapted dense endpoint"):
            paper_analysis.load_adaptations(batch, [0, 1])
    else:
        assert result["adaptations"] is None
