"""Frozen test declaration, calibration isolation and unchanged source checkpoints."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from diagnostics.masked_mlp_learned import frozen_test, run
from diagnostics.masked_mlp_oracle import paper_analysis
from diagnostics.shared.data import PrepareConfig, prepare
from diagnostics.shared.io import read_json, sha
from diagnostics.shared.source_data import PrepareConfig as SourcePrepareConfig
from diagnostics.shared.source_data import prepare as prepare_source
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import write_json


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _batch(root: Path) -> None:
    prepare_source(
        SourcePrepareConfig(
            str(root / "source"),
            train_groups=4,
            calibration_groups=4,
            validation_groups=2,
            test_groups=2,
            profile="smoke",
        )
    )
    prepare(PrepareConfig(str(root / "data"), str(root / "source"), profile="smoke"))
    dense = root / "batch/dense-s0/results"
    run.train(
        run.TrainConfig(
            str(dense),
            str(root / "data"),
            "dense",
            profile="smoke",
            steps=1,
            batch_size=2,
            code_dim=6,
            initializer="native",
        )
    )
    bound = root / "bound"
    run.calibrate(run.CalibrateConfig(str(bound), str(dense), profile="smoke", tau_factor=1.0))
    for arm in ("baseline", "full"):
        run.train(
            run.TrainConfig(
                str(root / f"batch/{arm}-s0/results"),
                str(root / "data"),
                arm,
                profile="smoke",
                steps=1,
                batch_size=2,
                calibration_dir=str(bound),
                init_dir=str(dense),
            )
        )


def test_declaration_does_not_load_test_and_evaluation_preserves_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _batch(tmp_path)
    before = {str(path): sha(path) for path in (tmp_path / "batch").rglob("*") if path.is_file()}
    loaded: list[str] = []
    original = torch.load

    def record(path: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - torch.load wrapper
        loaded.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", record)
    output = tmp_path / "test"
    protocol = frozen_test.declare(
        tmp_path / "batch",
        tmp_path / "data",
        output,
        seeds=(0,),
        dense_steps=1,
        sparse_steps=1,
    )
    assert not any(
        Path(path).name in {"validation.pt", "test.pt", "test_states.pt"} for path in loaded
    )
    assert len(protocol["runs"]) == 3
    assert all(item["parameter_alignment_fit_split"] == "calibration" for item in protocol["runs"])
    alignment_path = tmp_path / "revision/alignment.json"
    loaded.clear()
    revision = frozen_test.align_parameters(output / "protocol.json", alignment_path)
    assert loaded
    assert all(Path(path).name == "calibration.pt" for path in loaded)
    for frozen, revised in zip(protocol["runs"], revision["runs"], strict=True):
        assert revised["parameter_columns"] == frozen["parameter_columns"]
    assert read_json(output / "protocol.json") == protocol
    frozen_test.evaluate_seed(output / "protocol.json", 0, device="cpu", wandb_mode="disabled")
    assert any(Path(path).name == "test.pt" for path in loaded)
    result = frozen_test.summarize(output / "protocol.json")
    assert result["groups"] == 2
    assert set(result["summary"]) == {"dense", "baseline", "full"}
    report = read_json(output / "seed0/full/report.json")
    assert report["split"] == report["physical_response"]["split"] == "test"
    assert len(report["physical_response"]["per_parameter"]) == 6
    assert report["source"]["matching"] == read_json(
        tmp_path / "batch/full-s0/results/matching.json"
    )
    assert "masks/parameter_code_aligned_shd" in report["metrics"]

    def forbidden(*_args: object) -> None:
        raise AssertionError("Plot export must not fit coordinate assignments on test data")

    monkeypatch.setattr(paper_analysis, "_code_order", forbidden)
    monkeypatch.setattr(paper_analysis, "_matched_code_correlation", forbidden)
    plotted, declaration = paper_analysis.load_test_batch(output)
    assert declaration == protocol
    revised_plots, _ = paper_analysis.load_test_batch(output, parameter_alignment=alignment_path)
    for frozen in protocol["runs"]:
        item = plotted[frozen["arm"]][frozen["seed"]]
        assert item.code_columns == [
            frozen["parameter_columns"][k] for k in paper_analysis.PAIR_ORDER
        ]
        assert item.columns == frozen["matching"]["online_columns"]
        updated = revised_plots[frozen["arm"]][frozen["seed"]]
        assert updated.metrics == item.metrics
        np.testing.assert_array_equal(updated.code_correlation, item.code_correlation)
        np.testing.assert_array_equal(updated.graphs["parameter"], item.graphs["parameter"])
    chance = {
        name: {"mcc_mean": 0.8, "mcc_p05": 0.6, "mcc_p95": 0.9}
        for name in ("any_rotation", "within_block")
    }
    figure = paper_analysis.figure_mcc(plotted, [0], chance, split="test")
    axis: Any = figure.axes[0]
    assert "(test)" in axis.get_ylabel()
    points = cast(
        paper_analysis.Array,
        np.concatenate([np.asarray(c.get_offsets()) for c in axis.collections]),
    )
    np.testing.assert_array_equal(
        points[:, 1], [plotted[arm][0].metrics["state_mcc"] for arm in paper_analysis.ARMS]
    )
    paper_analysis.plt.close(figure)
    for path, digest in before.items():
        assert sha(Path(path)) == digest
    with pytest.raises(FileExistsError):
        frozen_test.evaluate_seed(output / "protocol.json", 0, device="cpu", wandb_mode="disabled")
    altered = protocol | {"test_groups": 3}
    write_json(output / "altered.json", altered)
    with pytest.raises(ValueError, match="declaration changed"):
        frozen_test.evaluate_seed(output / "altered.json", 0, device="cpu", wandb_mode="disabled")
    # A separately recorded permutation must reorder existing columns, never refit on test.
    revised = revision["runs"][-1]
    first, second = revised["parameter_columns"][:2]
    revised["parameter_columns"][:2] = [second, first]
    revision["identity"] = canonical_fingerprint(
        {k: v for k, v in revision.items() if k != "identity"}
    )
    write_json(alignment_path, revision)
    remapped, _ = paper_analysis.load_test_batch(output, parameter_alignment=alignment_path)
    expected = plotted["full"][0].code_correlation
    assert expected is not None
    # Display order is (theta1, theta4, theta2, theta5, theta3, theta6).
    np.testing.assert_array_equal(
        remapped["full"][0].code_correlation, expected[:, [2, 1, 0, 3, 4, 5]]
    )
    revision["runs"].pop()
    revision["identity"] = canonical_fingerprint(
        {k: v for k, v in revision.items() if k != "identity"}
    )
    write_json(alignment_path, revision)
    with pytest.raises(ValueError, match="every endpoint"):
        paper_analysis.load_test_batch(output, parameter_alignment=alignment_path)
