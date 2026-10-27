from pathlib import Path
from typing import Any

import pytest
from test_encoder_gradient_source import write_source_fixture
from torch import Tensor

from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.joint_acquisition.run import Config, run
from diagnostics.multiquery_control.data import EvaluationData
from scjepa.models.model import SCJEPA
from scjepa.training.logging import WandbLogger


def test_failed_gate_still_publishes_measurements_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[dict[str, float]] = []
    artifacts: list[str] = []

    def log(_self: WandbLogger, _step: int, metrics: dict[str, float]) -> None:
        events.append(metrics)

    def artifact(
        _self: WandbLogger, path: Path, *, name: str, artifact_type: str = "evaluation"
    ) -> None:
        assert name.startswith("joint-acquisition-")
        assert artifact_type == "evaluation"
        assert path.is_file()
        artifacts.append(path.name)

    monkeypatch.setattr(WandbLogger, "log", log)
    monkeypatch.setattr(WandbLogger, "log_artifact", artifact)
    from diagnostics.joint_acquisition import run as runner

    original = runner.numerical_checks

    def invalid(
        model: SCJEPA, data: EvaluationData, selection: DiagnosticSelection
    ) -> tuple[dict[str, Any], dict[str, Tensor]]:
        report, raw = original(model, data, selection)
        report["passed"] = False
        return report, raw

    monkeypatch.setattr(runner, "numerical_checks", invalid)
    source = write_source_fixture(tmp_path / "source")
    result = run(
        Config(
            source_dir=str(source),
            output_dir=str(tmp_path / "diagnostic"),
            seed=7,
            groups=2,
            steps=2,
            bootstrap_samples=20,
            verification=True,
            expected_source_step=2,
            expected_checkpoint_sha256=None,
        )
    )
    assert all(
        row["status"] == "not_run_invalid_local_checks" for row in result["trajectories"].values()
    )
    metrics = {key: value for event in events for key, value in event.items()}
    assert metrics["diagnostic/shared_checks_passed"] == 0
    assert (
        metrics["baseline/theta_1/normalized_response_error"]
        == result["baseline"]["per_parameter"][0]["normalized_response_error"]["estimate"]
    )
    assert metrics["trajectory/joint/skipped"] == 1
    assert (
        metrics["retention/theta_1/selected_validation_r2"]
        == result["retention"]["selected_validation_r2"][0]
    )
    assert artifacts == ["results.json", "report.md"]
