import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.config import load_experiment_config
from scjepa.experiments.runner import experiment_plan
from scjepa.experiments.storage import json_safe, read_json

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/run_controlled_experiment.py"
CONFIGS = REPOSITORY / "configs/experiments"


def _invoke(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    # An attempted optional SDK import fails immediately rather than logging in
    # or contacting a service, even when testing the default online config.
    (tmp_path / "wandb.py").write_text('raise AssertionError("CLI test must not import W&B")\n')
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        WANDB_MODE="online",
        WANDB_DIR=str(tmp_path / "wandb"),
        PYTHONPATH=os.pathsep.join(filter(None, (str(tmp_path), environment.get("PYTHONPATH")))),
    )
    return subprocess.run(
        [sys.executable, "-u", str(SCRIPT), *arguments],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("name", ["controlled_vector", "controlled_vector_smoke"])
def test_shipped_configs_resolve_to_sigreg_only(name: str) -> None:
    config = load_experiment_config(CONFIGS / f"{name}.yaml", ["wandb.mode=disabled"])
    assert config.train.objective.state_regularizer == "sigreg"
    assert config.train.objective.parameter_regularizer == "sigreg"
    assert config.train.objective.state_regularizer_weight > 0
    assert config.train.objective.parameter_regularizer_weight > 0
    assert config.train.objective.parameter_sparsity_weight > 0
    config.validate()


def test_default_dry_run_reports_thirty_tasks_without_output_or_wandb(tmp_path: Path) -> None:
    output = tmp_path / "experiment"
    result = _invoke(tmp_path, "--dry-run", f"output_dir={output}")
    assert result.returncode == 0, result.stderr
    plan = cast(dict[str, object], json.loads(result.stdout))
    assert plan["runs"] == 30
    tasks = cast(list[dict[str, object]], plan["plan"])
    assert len(tasks) == 30
    assert {task["condition"] for task in tasks} == {"baseline", "full"}
    assert {task["graph"] for task in tasks} == {"dense", "complementary", "diagonal"}
    assert {task["seed"] for task in tasks} == set(range(5))
    assert not output.exists()
    assert not (tmp_path / "wandb").exists()


def test_task_selectors_leave_the_full_scientific_definition_unchanged(tmp_path: Path) -> None:
    config = load_experiment_config(CONFIGS / "controlled_vector.yaml")
    initial = canonical_fingerprint(json_safe(config.definition()))
    tasks = experiment_plan(config, graph="diagonal", seed=3, condition="full")
    assert len(tasks) == 1
    assert tasks[0]["graph"] == "diagonal"
    assert tasks[0]["seed"] == 3
    assert tasks[0]["condition"] == "full"
    assert canonical_fingerprint(json_safe(config.definition())) == initial
    result = _invoke(
        tmp_path,
        "--dry-run",
        "--graph",
        "diagonal",
        "--seed",
        "3",
        "--condition",
        "full",
        f"output_dir={tmp_path / 'experiment'}",
    )
    assert result.returncode == 0, result.stderr
    plan = cast(dict[str, object], json.loads(result.stdout))
    assert plan["runs"] == 1
    assert len(config.seeds) == 5
    assert len(config.graphs) == 3


@pytest.mark.parametrize(
    "override",
    ["unknown_setting=1", "train.objective.state_regularizer=visreg", "model.state_dim=9"],
)
def test_invalid_overrides_fail_before_creating_outputs(tmp_path: Path, override: str) -> None:
    output = tmp_path / "experiment"
    result = _invoke(tmp_path, "--dry-run", f"output_dir={output}", override)
    assert result.returncode != 0
    assert not output.exists()
    assert not (tmp_path / "wandb").exists()


def test_aggregate_only_requires_existing_manifest_and_creates_nothing(tmp_path: Path) -> None:
    output = tmp_path / "absent"
    result = _invoke(tmp_path, "--aggregate-only", f"output_dir={output}")
    assert result.returncode != 0
    assert "requires an existing experiment manifest" in result.stderr
    assert not output.exists()
    assert not (tmp_path / "wandb").exists()


def test_tiny_cpu_cli_produces_selected_run_and_keeps_full_pair_definition(tmp_path: Path) -> None:
    output = tmp_path / "cpu-smoke"
    result = _invoke(
        tmp_path,
        "--config",
        str(CONFIGS / "controlled_vector_smoke.yaml"),
        "--condition",
        "baseline",
        f"output_dir={output}",
        "wandb.mode=disabled",
        "train.device=cpu",
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "wandb").exists()
    summary = read_json(output / "summary.json")
    assert summary["completed_runs"] == 1
    assert summary["expected_runs"] == 2
    assert summary["complete"] is False
    assert summary["missing_runs"] == ["complementary/seed-0/full"]
    directory = output / "complementary/seed-0/baseline"
    assert read_json(directory / "status.json")["status"] == "completed"
    assert (directory / "best.pt").is_file()
    assert (directory / "last.pt").is_file()
    assert (directory / "evaluation/metrics.json").is_file()
    manifest = read_json(output / "experiment.json")
    definition = cast(dict[str, object], manifest["definition"])
    assert definition["seeds"] == [0]
    assert definition["graphs"] == ["complementary"]
    assert "condition" not in definition
    run = read_json(directory / "run_config.json")
    training = cast(dict[str, object], run["train"])
    objective = cast(dict[str, object], training["objective"])
    assert objective["parameter_sparsity_weight"] == 0.0
    assert objective["state_regularizer"] == "sigreg"
    assert objective["parameter_regularizer"] == "sigreg"
