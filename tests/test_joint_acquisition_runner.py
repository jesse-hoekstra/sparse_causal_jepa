import json
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_encoder_gradient_source import write_source_fixture

from diagnostics.joint_acquisition.run import Config, run


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_checkpoint_to_report_preserves_source_and_records_limits(tmp_path: Path) -> None:
    source = write_source_fixture(tmp_path / "source")
    checkpoint = source / "seed-7/prediction_1/last.pt"
    before, rng = checkpoint.read_bytes(), torch.get_rng_state()
    output = tmp_path / "diagnostic"
    config = Config(
        source_dir=str(source),
        output_dir=str(output),
        seed=7,
        groups=2,
        steps=2,
        bootstrap_samples=20,
        verification=True,
        expected_source_step=2,
        expected_checkpoint_sha256=None,
    )
    result = run(config)
    assert checkpoint.read_bytes() == before
    assert torch.equal(rng, torch.get_rng_state())
    assert result["compatible_control"]["status"] == "unavailable"
    assert len(result["baseline"]["per_parameter"]) == 6
    assert result["scientific_status"] == "software verification only"
    assert result["code_repair"]["fit_query_indices"] == [[1, 2, 3, 4]]
    assert result["code_repair"]["heldout_query_indices"] == [[5, 6, 7, 8, 9]]
    assert all(
        row["displacement_rms_in_calibration_std"] <= 0.1 + 1e-12
        for row in result["code_repair"]["per_donor_fits"]
    )
    assert (output / "report.md").is_file()
    manifest = cast(dict[str, Any], json.loads((output / "completed.json").read_text()))
    assert manifest["identity"] == result["identity"]
    resumed = run(config, resume=True)
    assert resumed == result
    with pytest.raises(ValueError, match="incompatible resume"):
        run(replace(config, learning_rate=0.002), resume=True)
    (output / "report.md").write_text("tampered")
    with pytest.raises(ValueError, match="output identity"):
        run(config, resume=True)


def test_protocol_refuses_source_writes_and_unbounded_execution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the source"):
        Config(source_dir=str(tmp_path), output_dir=str(tmp_path / "output"))
    with pytest.raises(ValueError, match="1000 updates"):
        Config(source_dir=str(tmp_path / "source"), output_dir=str(tmp_path / "out"), steps=1001)
