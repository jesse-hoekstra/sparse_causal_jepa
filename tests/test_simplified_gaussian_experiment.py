"""Six-state protocol contracts and end-to-end final-checkpoint evaluation."""

import math
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

import scjepa.models.spartan as spartan_module
from scjepa.data.factory import build_dgp_dataset
from scjepa.experiments.config import load_experiment_config
from scjepa.experiments.storage import read_json
from scjepa.training.factory import build_model
from scjepa.training.loop import TrainingObjective

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIGS = REPOSITORY / "configs/experiments"


@pytest.mark.parametrize("suffix", ["", "_smoke"])
def test_six_state_presets_sparsify_action_paths_and_pair_the_sparse_conditions(
    suffix: str, tmp_path: Path
) -> None:
    config = load_experiment_config(
        CONFIGS / f"simplified_gaussian{suffix}.yaml", ["wandb.mode=disabled"]
    )
    assert config.data_family == "simplified_gaussian"
    assert (
        config.model.observation_dim,
        config.model.state_dim,
        config.model.param_dim,
        config.model.action_dim,
    ) == (6, 6, 6, 16)
    assert config.model.context_len == 10
    assert config.trajectory_length == 12
    assert config.episodes_per_parameter == 2
    assert config.conditions == ("dense", "baseline", "full")
    assert config.data_config("complementary").family == "simplified_gaussian"
    assert config.train.steps == (2 if suffix else 100_000)
    baseline = asdict(config.condition_model("baseline"))
    full = asdict(config.condition_model("full"))
    baseline["spartan_parameter_gates_open"] = False
    assert baseline == full
    baseline_objective = asdict(config.training_config("baseline", 0, tmp_path, tau=0.3).objective)
    full_objective = asdict(config.training_config("full", 0, tmp_path, tau=0.3).objective)
    baseline_objective["parameter_sparsity_weight"] = full_objective["parameter_sparsity_weight"]
    assert baseline_objective == full_objective
    for condition in config.conditions:
        model = config.condition_model(condition)
        assert not model.spartan_action_gates_open
        assert model.spartan_dense == (condition == "dense")
        assert model.spartan_parameter_gates_open == (condition == "baseline")
        train = config.training_config(condition, 0, tmp_path, tau=0.3)
        assert train.objective.action_sparsity_weight == (0 if condition == "dense" else 1)
        assert train.objective.state_sparsity_weight == (0 if condition == "dense" else 1)
        assert train.objective.parameter_sparsity_weight == (1 if condition == "full" else 0)
        assert train.lr == 0.001
        assert train.lr_final_ratio == 1.0
        assert train.objective.state_regularizer == "sigreg"
        assert train.objective.parameter_regularizer == "sigreg"


@pytest.mark.parametrize(
    "override",
    [
        "model.action_dim=6",
        "model.state_dim=10",
        "model.context_len=2",
        "trajectory_length=11",
        "trajectory_length=13",
        "episodes_per_parameter=1",
        "model.autoregressive=false",
        "graphs=[diagonal]",
        "train.objective.action_sparsity_weight=0",
        "model.spartan_action_gates_open=true",
    ],
)
def test_rejects_changes_that_break_the_approved_comparison(override: str) -> None:
    with pytest.raises(ValueError, match="simplified_gaussian|gate modes"):
        load_experiment_config(CONFIGS / "simplified_gaussian_smoke.yaml", [override])


def test_simplified_context_can_vary_while_preserving_two_forecasts() -> None:
    config = load_experiment_config(
        CONFIGS / "simplified_gaussian_smoke.yaml",
        ["model.context_len=3", "trajectory_length=5"],
    )
    assert config.model.context_len == 3
    assert config.trajectory_length == 5
    with pytest.raises(ValueError, match="independent_gaussian"):
        load_experiment_config(
            CONFIGS / "independent_gaussian_smoke.yaml",
            ["model.context_len=10", "trajectory_length=12"],
        )


@pytest.mark.parametrize("condition", ["baseline", "full"])
def test_long_context_has_two_forecasts_and_action_paths_are_prunable_and_penalized(
    condition: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_experiment_config(CONFIGS / "simplified_gaussian_smoke.yaml")
    dataset = build_dgp_dataset(config.data_config("complementary"))
    items = [dataset[index] for index in range(2)]
    observations = torch.stack([item["observations"] for item in items])
    actions = torch.stack([item["actions"] for item in items])
    model = build_model(config.condition_model(condition)).eval()
    context_inputs: list[tuple[Tensor, Tensor]] = []

    def capture_context(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        context_inputs.append((cast(Tensor, inputs[0]), cast(Tensor, inputs[1])))

    def closed_adjacency(logits: Tensor, _temperature: float, _training: bool) -> Tensor:
        return torch.zeros_like(logits)

    monkeypatch.setattr(spartan_module, "_sample_hard_adjacency", closed_adjacency)
    handle = model.parameter_encoder.register_forward_pre_hook(capture_context)
    closed = model(observations, actions)
    handle.remove()
    assert observations.shape == (2, 2, 12, 6)
    assert len(context_inputs) == 1
    assert context_inputs[0][0].shape == (4, 10, 6)
    torch.testing.assert_close(context_inputs[0][1], actions[:, :, :9].flatten(0, 1))
    assert closed.prediction_start == 9
    assert closed.predictions.shape == closed.targets.shape == (2, 2, 2, 6)
    assert len(closed.dynamics) == 2
    expected_targets = model.target_encoder.encode_sequence(
        observations[:, :, 10:12].flatten(0, 1)
    ).reshape(2, 2, 2, 6)
    torch.testing.assert_close(closed.targets, expected_targets)
    assert all(not dynamics.graphs.action.any() for dynamics in closed.dynamics)
    assert all(
        bool(dynamics.graphs.parameter.any()) == (condition == "baseline")
        for dynamics in closed.dynamics
    )

    def open_adjacency(logits: Tensor, _temperature: float, _training: bool) -> Tensor:
        return torch.ones_like(logits)

    monkeypatch.setattr(spartan_module, "_sample_hard_adjacency", open_adjacency)
    opened = model(observations, actions)
    objective_config = config.training_config(condition, 0, tmp_path, tau=0.3).objective
    _, penalized = TrainingObjective(objective_config)(opened)
    _, unpenalized = TrainingObjective(replace(objective_config, action_sparsity_weight=0))(opened)
    assert penalized["action_sparsity"] > 0
    torch.testing.assert_close(
        penalized["sparsity"] - unpenalized["sparsity"], penalized["action_sparsity"]
    )


def test_three_stage_cpu_cli_uses_six_state_data_and_final_checkpoints(tmp_path: Path) -> None:
    output = tmp_path / "experiment"
    (tmp_path / "wandb.py").write_text('raise AssertionError("W&B must remain disabled")\n')
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=os.pathsep.join((str(tmp_path), str(REPOSITORY / "src"))),
    )
    command = [
        sys.executable,
        str(REPOSITORY / "scripts/run_controlled_experiment.py"),
        "--config",
        str(CONFIGS / "simplified_gaussian_smoke.yaml"),
        "wandb.mode=disabled",
        f"output_dir={output}",
    ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert read_json(output / "summary.json")["complete"] is True
    matrix = torch.load(output / "observation_map.pt", weights_only=True)["matrix"]
    assert matrix.shape == (6, 6)
    parent = output / "complementary/seed-0"
    records: list[dict[str, object]] = []
    for condition in ("dense", "baseline", "full"):
        directory = parent / condition
        record = read_json(directory / "run_config.json")
        records.append(record)
        assert cast(dict[str, object], record["data"])["family"] == "simplified_gaussian"
        checkpoint = torch.load(directory / "last.pt", weights_only=False)
        assert checkpoint["step"] == 2
        assert checkpoint["model_config"]["context_len"] == 10
        assert cast(dict[str, object], record["data"])["trajectory_length"] == 12
        assert checkpoint["model_config"]["spartan_action_gates_open"] is False
        assert checkpoint["train_config"]["objective"]["action_sparsity_weight"] == (
            0 if condition == "dense" else 1
        )
        assert checkpoint["model_config"]["spartan_parameter_gates_open"] == (
            condition == "baseline"
        )
        evaluated = read_json(directory / "evaluation/evaluated_checkpoint.json")
        assert evaluated["checkpoint_path"] == "last.pt"
        assert evaluated["evaluated_step"] == 2
        metrics = read_json(directory / "evaluation/metrics.json")
        assert metrics["graph/action_false_positives"] == 0
        assert metrics["graph/action_shd"] == metrics["graph/action_false_negatives"]
        assert 0 <= cast(float, metrics["graph/action_shd"]) <= 6 * 16
        if condition == "dense":
            assert metrics["graph/action_shd"] == 0
        assert all(math.isfinite(cast(float, value)) for value in metrics.values())
    assert records[1]["initial_state_fingerprint"] == records[2]["initial_state_fingerprint"]
    tau = read_json(parent / "dense/tau.json")
    assert records[1]["tau_calibration"] == records[2]["tau_calibration"] == tau
    evaluated = subprocess.run(
        [*command, "--evaluate-only"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert evaluated.returncode == 0, evaluated.stdout + evaluated.stderr
