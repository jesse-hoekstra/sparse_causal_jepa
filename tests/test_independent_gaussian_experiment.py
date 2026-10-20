"""Scientific-family selection, evaluation provenance and tiny CLI training."""

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.datasets import canonical_fingerprint
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.processes import ControlledVectorDGP
from scjepa.data.types import DGP, GraphMasks, ScientificEpisode
from scjepa.experiments.config import ExperimentConfig, load_experiment_config
from scjepa.experiments.evaluation import evaluate_model, validate_model
from scjepa.experiments.reference import external_dense_manifest
from scjepa.experiments.storage import json_safe, read_json, write_json
from scjepa.training.factory import build_model

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIGS = REPOSITORY / "configs/experiments"


@pytest.mark.parametrize("name", ["independent_gaussian", "independent_gaussian_smoke"])
def test_presets_use_three_frame_context_and_constant_lr(name: str, tmp_path: Path) -> None:
    config = load_experiment_config(CONFIGS / f"{name}.yaml", ["wandb.mode=disabled"])
    assert config.data_family == "independent_gaussian"
    assert config.definition()["data_family"] == "independent_gaussian"
    assert tuple(config.graphs) == ("complementary",)
    assert (config.model.context_len, config.trajectory_length, config.episodes_per_parameter) == (
        3,
        5,
        2,
    )
    assert config.model.history_len == 1
    assert config.model.autoregressive
    assert config.conditions == ("dense", "baseline", "full")
    assert config.dense_reference_dir is None
    assert config.train.steps == (2 if name.endswith("_smoke") else 100_000)
    data = config.data_config("complementary")
    assert data.family == "independent_gaussian"
    for condition in config.conditions:
        training = config.training_config(condition, 0, tmp_path, tau=1.0)
        assert training.lr == 0.001
        assert training.lr_final_ratio == 1.0
        assert training.objective.state_regularizer_weight == 0.1
        assert training.objective.parameter_regularizer_weight == 0.1
    assert "data_family" not in ExperimentConfig().definition()


@pytest.mark.parametrize(
    "override",
    [
        "data_family=unknown",
        "graphs=[dense]",
        "model.context_len=2",
        "trajectory_length=4",
        "episodes_per_parameter=1",
        "model.autoregressive=false",
    ],
)
def test_new_experiment_rejects_incompatible_protocol(override: str) -> None:
    with pytest.raises(ValueError, match="data_family|independent_gaussian"):
        load_experiment_config(
            CONFIGS / "independent_gaussian_smoke.yaml",
            ["wandb.mode=disabled", override],
        )


def test_evaluation_uses_registered_family_and_its_reference_graphs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_experiment_config(
        CONFIGS / "independent_gaussian_smoke.yaml", ["wandb.mode=disabled"]
    )
    data = config.data_config("complementary")
    dataset = build_dgp_dataset(data)
    graph = dataset.graphs

    def registered_graphs(_process: DGP) -> GraphMasks:
        # Make the registered graph observably different from controlled_vector's.
        return GraphMasks(graph.state, torch.zeros_like(graph.action), graph.parameter)

    def forbidden_controlled_generation(
        _process: ControlledVectorDGP, _index: int
    ) -> ScientificEpisode:
        raise AssertionError("evaluation dispatched to the old DGP")

    monkeypatch.setattr(type(dataset.process), "graphs", registered_graphs)
    monkeypatch.setattr(ControlledVectorDGP, "generate", forbidden_controlled_generation)
    model = build_model(config.condition_model("dense"))
    metrics = evaluate_model(
        model,
        data,
        VectorObservationMap().matrix,
        config.evaluation,
        tmp_path,
        objective=config.train.objective,
    )
    assert metrics["graph/action_shd"] == 100.0
    references = cast(
        dict[str, object], torch.load(tmp_path / "evaluation_references.pt", weights_only=True)
    )
    truth = cast(dict[str, Tensor], references["reference_graphs"])
    assert not bool(truth["action"].any())
    assert int(truth["state"].sum()) == 50
    assert int(truth["parameter"].sum()) == 20
    protocol = read_json(tmp_path / "protocol.json")
    for key in ("calibration_data", "test_data"):
        recorded = cast(dict[str, object], protocol[key])
        assert recorded["family"] == "independent_gaussian"
        assert recorded["trajectory_length"] == 5
        assert recorded["episodes_per_parameter"] == 2
    wrong_context = build_model(replace(config.condition_model("dense"), context_len=2))
    with pytest.raises(ValueError, match="K>=3"):
        validate_model(wrong_context, data, VectorObservationMap().matrix, config.evaluation)


def test_new_family_rejects_old_dense_reference_before_loading_its_checkpoint(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(
        CONFIGS / "independent_gaussian_smoke.yaml", ["wandb.mode=disabled"]
    )
    config.dense_reference_dir = str(tmp_path / "old")
    definition = json_safe(ExperimentConfig().definition())
    manifest: dict[str, object] = {
        "format_version": 1,
        "definition": definition,
        "config_fingerprint": canonical_fingerprint(definition),
    }
    manifest["identity"] = canonical_fingerprint(manifest)
    root = Path(config.dense_reference_dir)
    root.mkdir()
    write_json(root / "experiment.json", manifest)
    directory = root / "complementary/seed-0/dense"
    directory.mkdir(parents=True)
    write_json(directory / "run_config.json", {"data": {"family": "controlled_vector"}})
    with pytest.raises(ValueError, match="data family differs"):
        external_dense_manifest(config, {}, "complementary", 0)
    assert not (directory / "best.pt").exists()


def test_cpu_cli_trains_three_stages_and_evaluates_final_checkpoints(tmp_path: Path) -> None:
    output = tmp_path / "independent"
    (tmp_path / "wandb.py").write_text('raise AssertionError("test must not import W&B")\n')
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=os.pathsep.join(filter(None, (str(tmp_path), environment.get("PYTHONPATH")))),
    )
    arguments = [
        sys.executable,
        str(REPOSITORY / "scripts/run_controlled_experiment.py"),
        "--config",
        str(CONFIGS / "independent_gaussian_smoke.yaml"),
        f"output_dir={output}",
        "wandb.mode=disabled",
    ]
    result = subprocess.run(
        arguments,
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = read_json(output / "summary.json")
    assert summary["complete"] is True
    assert summary["completed_runs"] == 3
    manifest = read_json(output / "experiment.json")
    assert cast(dict[str, object], manifest["definition"])["data_family"] == "independent_gaussian"
    parent = output / "complementary/seed-0"
    for condition in ("dense", "baseline", "full"):
        directory = parent / condition
        record = read_json(directory / "run_config.json")
        assert cast(dict[str, object], record["data"])["family"] == "independent_gaussian"
        checkpoint = torch.load(directory / "last.pt", weights_only=False)
        assert checkpoint["step"] == 2
        assert checkpoint["model_config"]["context_len"] == 3
        assert math.isclose(checkpoint["optimizer"]["param_groups"][0]["lr"], 0.001)
        evaluated = read_json(directory / "evaluation/evaluated_checkpoint.json")
        assert evaluated["checkpoint_path"] == "last.pt"
        assert evaluated["evaluated_step"] == 2
        metrics = read_json(directory / "evaluation/metrics.json")
        assert all(math.isfinite(cast(float, value)) for value in metrics.values())
    dense_tau = read_json(parent / "dense/tau.json")
    assert read_json(parent / "baseline/run_config.json")["tau_calibration"] == dense_tau
    assert read_json(parent / "full/run_config.json")["tau_calibration"] == dense_tau
    reevaluated = subprocess.run(
        [*arguments, "--evaluate-only"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert reevaluated.returncode == 0, reevaluated.stderr
    assert (
        json.loads((parent / "full/evaluation/evaluated_checkpoint.json").read_text())[
            "checkpoint_path"
        ]
        == "last.pt"
    )
