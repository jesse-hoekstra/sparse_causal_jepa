"""Frozen test declaration, calibration isolation and unchanged source checkpoints."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from diagnostics.masked_mlp_learned import frozen_test, run
from diagnostics.shared.data import PrepareConfig, prepare
from diagnostics.shared.io import read_json, sha
from diagnostics.shared.source_data import PrepareConfig as SourcePrepareConfig
from diagnostics.shared.source_data import prepare as prepare_source
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
    assert not any(Path(path).name in {"test.pt", "test_states.pt"} for path in loaded)
    assert len(protocol["runs"]) == 3
    assert all(item["parameter_alignment_fit_split"] == "validation" for item in protocol["runs"])
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
    for path, digest in before.items():
        assert sha(Path(path)) == digest
    with pytest.raises(FileExistsError):
        frozen_test.evaluate_seed(output / "protocol.json", 0, device="cpu", wandb_mode="disabled")
    altered = protocol | {"test_groups": 3}
    write_json(output / "altered.json", altered)
    with pytest.raises(ValueError, match="declaration changed"):
        frozen_test.evaluate_seed(output / "altered.json", 0, device="cpu", wandb_mode="disabled")
