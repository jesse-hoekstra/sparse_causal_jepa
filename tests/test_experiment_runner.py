"""Tiny CPU end-to-end paired experiment, artifact, and resume contracts."""

import copy
import math
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import pytest
import torch

from scjepa.eval.recovery import ProbeConfig
from scjepa.experiments.config import ExperimentConfig, load_experiment_config
from scjepa.experiments.evaluation import EvaluationConfig
from scjepa.experiments.runner import (
    _selected_checkpoint,  # pyright: ignore[reportPrivateUsage]
    experiment_plan,
    run_experiment,
)
from scjepa.experiments.storage import file_lock, json_safe, prepare_experiment, read_json
from scjepa.models.model import SCJEPA
from scjepa.training import Trainer
from scjepa.training.factory import ModelConfig, build_model


def _config(directory: Path) -> ExperimentConfig:
    config = ExperimentConfig(
        output_dir=str(directory),
        graphs=("complementary",),
        seeds=(3,),
        train_episodes=4,
        trajectory_length=5,
        validation_every=1,
        model=ModelConfig(
            observation_dim=10,
            state_dim=10,
            param_dim=10,
            action_dim=10,
            context_len=2,
            history_len=1,
            encoder_hidden_dims=(8,),
            parameter_embed_dim=8,
            parameter_heads=2,
            spartan_layers=1,
            spartan_embed_dim=8,
            spartan_mlp_hidden=8,
            spartan_mlp_layers=2,
        ),
        evaluation=EvaluationConfig(
            calibration_episodes=4,
            validation_episodes=4,
            test_episodes=4,
            paired_episodes=4,
            batch_size=2,
            rollout_steps=2,
            validation_probe_steps=2,
            probe=ProbeConfig(hidden_dim=8, steps=2, batch_size=8),
        ),
    )
    config.train.steps = 2
    config.train.batch_size = 2
    config.train.checkpoint_every = 1
    config.train.log_every = 1
    config.train.objective.regularizer_num_projections = 4
    config.wandb.mode = "disabled"
    config.validate()
    return config


def test_real_paired_experiment_artifacts_and_resume_are_reproducible(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_experiment(config)
    assert result["complete"] is True
    assert result["completed_runs"] == result["expected_runs"] == 2
    assert result["missing_runs"] == []
    pairs = cast(
        dict[str, dict[str, dict[str, float | int | None]]], result["paired_full_minus_baseline"]
    )
    assert pairs["complementary"]
    assert all(item["n"] == 1 for item in pairs["complementary"].values())
    assert all(item["std"] is None for item in pairs["complementary"].values())
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "results.csv").is_file()
    assert (tmp_path / "experiment.json").is_file()
    saved_map = torch.load(tmp_path / "observation_map.pt", weights_only=True)["matrix"]
    paths = [tmp_path / "complementary" / "seed-3" / name for name in ("baseline", "full")]
    records = [read_json(directory / "run_config.json") for directory in paths]
    for name in ("initial_state_fingerprint", "dataset_identity", "experiment_identity"):
        assert records[0][name] == records[1][name]
        assert isinstance(records[0][name], str)
        assert records[0][name]
    saved_objectives: list[dict[str, object]] = []
    for directory in paths:
        assert read_json(directory / "status.json")["status"] == "completed"
        last = torch.load(directory / "last.pt", weights_only=False)
        best = torch.load(directory / "best.pt", weights_only=False)
        assert last["step"] == 2
        assert last["optimizer_steps"] == 2
        assert best["step"] == last["validation_state"]["best_step"]
        assert (
            last["validation_state"]["selection_metric"] == "validation/physical_prediction_nrmse"
        )
        saved_objectives.append(last["train_config"]["objective"])
        evaluation = directory / "evaluation"
        metrics = read_json(evaluation / "metrics.json")
        assert metrics["test/physical_prediction_nrmse"] >= 0  # pyright: ignore[reportOperatorIssue]
        assert metrics["test/encoded_target_decoding_nrmse"] >= 0  # pyright: ignore[reportOperatorIssue]
        assert all(math.isfinite(value) for value in cast(dict[str, float], metrics).values())
        for artifact in (
            "target_state_probe.pt",
            "parameter_consistency.pt",
            "evaluation_references.pt",
            "protocol.json",
        ):
            assert (evaluation / artifact).is_file()
        references = torch.load(evaluation / "evaluation_references.pt", weights_only=True)
        torch.testing.assert_close(references["matrix"], saved_map, rtol=0, atol=0)
        assert set(references["calibration_episode_ids"].tolist()).isdisjoint(
            references["test_episode_ids"].tolist()
        )
    assert saved_objectives[0]["parameter_sparsity_weight"] == 0.0
    assert (
        saved_objectives[1]["parameter_sparsity_weight"]
        == config.train.objective.parameter_sparsity_weight
    )
    baseline, full = copy.deepcopy(saved_objectives)
    del baseline["parameter_sparsity_weight"], full["parameter_sparsity_weight"]
    assert baseline == full
    assert baseline["state_regularizer"] == baseline["parameter_regularizer"] == "sigreg"
    modification_times = [(directory / "last.pt").stat().st_mtime_ns for directory in paths]
    assert run_experiment(config, resume=True) == result
    assert [(directory / "last.pt").stat().st_mtime_ns for directory in paths] == modification_times
    with pytest.raises((ValueError, RuntimeError), match="resume|exist|completed"):
        run_experiment(config)
    changed = copy.deepcopy(config)
    changed.train.objective.state_sparsity_weight *= 2
    with pytest.raises(ValueError, match="definition/source changed"):
        run_experiment(changed, resume=True)
    assert [(directory / "last.pt").stat().st_mtime_ns for directory in paths] == modification_times


def test_paired_train_configs_differ_only_in_parameter_penalty(tmp_path: Path) -> None:
    config = _config(tmp_path)
    baseline = config.training_config("baseline", 9, tmp_path)
    full = config.training_config("full", 9, tmp_path)
    assert baseline.objective.parameter_sparsity_weight == 0
    assert full.objective.parameter_sparsity_weight > 0
    baseline.objective.parameter_sparsity_weight = full.objective.parameter_sparsity_weight
    assert asdict(baseline) == asdict(full)
    assert config.train.objective.parameter_sparsity_weight > 0
    assert config.train.seed != 9
    with pytest.raises(ValueError, match="condition"):
        config.training_config("unknown", 9, tmp_path)


@pytest.mark.parametrize(
    "invalid",
    [
        "visreg",
        "dimensions",
        "zero_parameter",
        "zero_state",
        "seeds",
        "graphs",
        "dense",
        "history",
        "partial",
    ],
)
def test_experiment_config_rejects_incomparable_conditions(tmp_path: Path, invalid: str) -> None:
    config = _config(tmp_path)
    if invalid == "visreg":
        config.train.objective.state_regularizer = "visreg"
    elif invalid == "dimensions":
        config.model.param_dim = 7
    elif invalid == "zero_parameter":
        config.train.objective.parameter_sparsity_weight = 0
    elif invalid == "zero_state":
        config.train.objective.state_sparsity_weight = 0
    elif invalid == "seeds":
        config.seeds = (3, 3)
    elif invalid == "graphs":
        config.graphs = ("diagonal", "diagonal")
    elif invalid == "dense":
        config.model.spartan_dense = True
    elif invalid == "history":
        config.model.history_len = 2
    else:
        config.train.accumulation_steps = 3
    with pytest.raises(ValueError, match="[Ss][Ii][Gg][Rr]eg|comparison|penalt|distinct|boundary"):
        config.validate()


def test_experiment_defaults_use_sigreg_only() -> None:
    config = ExperimentConfig()
    config.validate()
    objective = config.train.objective
    assert objective.state_regularizer == objective.parameter_regularizer == "sigreg"
    assert objective.state_regularizer_weight > 0
    assert objective.parameter_regularizer_weight > 0
    assert objective.parameter_sparsity_weight > 0


def test_artifact_identity_uses_actual_matrix_and_rejects_mutation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first, matrix = prepare_experiment(config)
    second, reloaded = prepare_experiment(config)
    assert second == json_safe(first)
    torch.testing.assert_close(reloaded, matrix, rtol=0, atol=0)
    changed = matrix.roll(1, dims=0)
    torch.save({"matrix": changed}, tmp_path / "observation_map.pt")
    with pytest.raises(ValueError, match="observation map changed"):
        prepare_experiment(config)


def test_exclusive_run_lock_rejects_duplicate_ownership_and_releases(tmp_path: Path) -> None:
    path = tmp_path / ".run.lock"
    with (
        file_lock(path),
        pytest.raises(RuntimeError, match="another process owns"),
        file_lock(path),
    ):
        pytest.fail("second writer acquired the same lock")
    with file_lock(path):
        assert path.exists()


def test_strict_yaml_schema_supports_nested_overrides_and_rejects_visreg(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text("graphs: [complementary]\nseeds: [8]\n")
    config = load_experiment_config(path, ["train.steps=6", "wandb.mode=disabled"])
    assert tuple(config.graphs) == ("complementary",)
    assert tuple(config.seeds) == (8,)
    assert config.train.steps == 6
    assert config.wandb.mode == "disabled"
    with pytest.raises(ValueError, match="sigreg"):
        load_experiment_config(path, ["train.objective.parameter_regularizer=visreg"])
    invalid_rollout = replace(config.evaluation, rollout_steps=config.trajectory_length)
    config.evaluation = invalid_rollout
    with pytest.raises(ValueError, match="rollout"):
        config.validate()


def test_interrupted_run_resumes_without_overwriting_its_scientific_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    original_train = Trainer.train

    def interrupt(
        trainer: Trainer,
        validation_callback: Callable[[SCJEPA], dict[str, float]] | None = None,
        validation_every: int | None = None,
        selection_metric: str = "validation/physical_prediction_nrmse",
        stop_requested: Callable[[], bool] | None = None,
    ) -> dict[str, float]:
        return original_train(
            trainer,
            validation_callback,
            validation_every,
            selection_metric,
            stop_requested=lambda: trainer.step >= 1 or bool(stop_requested and stop_requested()),
        )

    with monkeypatch.context() as patch:
        patch.setattr(Trainer, "train", interrupt)
        result = run_experiment(config, condition="baseline")
    assert result["completed_runs"] == 0
    directory = tmp_path / "complementary" / "seed-3" / "baseline"
    assert read_json(directory / "status.json")["status"] == "interrupted"
    saved = torch.load(directory / "last.pt", weights_only=False)
    assert saved["step"] == 1
    assert not (directory / "best.pt").exists()
    identity = read_json(directory / "run_config.json")
    resumed = run_experiment(config, resume=True, condition="baseline")
    assert resumed["completed_runs"] == 1
    assert resumed["complete"] is False
    assert read_json(directory / "status.json")["status"] == "completed"
    final = torch.load(directory / "last.pt", weights_only=False)
    assert final["step"] == final["optimizer_steps"] == 2
    assert final["validation_state"]["best_step"] in {1, 2}
    assert read_json(directory / "run_config.json") == identity


def test_plan_selectors_do_not_mutate_the_comparison_definition(tmp_path: Path) -> None:
    config = _config(tmp_path)
    original = config.definition()
    all_tasks = experiment_plan(config)
    selected = experiment_plan(config, graph="complementary", seed=3, condition="full")
    assert len(all_tasks) == 2
    assert selected == [all_tasks[1]]
    assert config.definition() == original
    with pytest.raises(ValueError, match="selected graph"):
        experiment_plan(config, graph="diagonal")
    with pytest.raises(ValueError, match="selected seed"):
        experiment_plan(config, seed=9)
    with pytest.raises(ValueError, match="condition"):
        experiment_plan(config, condition="incorrect")


def test_resume_rejects_changed_checkpoint_objective(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_experiment(config, condition="baseline")
    path = tmp_path / "complementary" / "seed-3" / "baseline" / "last.pt"
    payload = torch.load(path, weights_only=False)
    payload["train_config"]["objective"]["parameter_sparsity_weight"] = 0.01
    torch.save(payload, path)
    with pytest.raises(ValueError, match="training configuration differs from experiment"):
        run_experiment(config, condition="baseline", evaluate_only=True)


def _selected_checkpoint_fixture(directory: Path) -> tuple[Trainer, Path, dict[str, object]]:
    config = _config(directory)
    trainer = Trainer(
        build_model(config.model),
        config.training_config("baseline", 3, directory),
        model_config=asdict(config.model),
        dataset_identity="controlled-vector-test-dataset",
    )
    trainer.run_metadata["experiment_run_identity"] = "baseline-run"
    trainer.step = 1
    trainer.validation_state = {
        "selection_metric": "validation/physical_prediction_nrmse",
        "best_score": 0.75,
        "best_step": 1,
        "last_step": 1,
    }
    path = directory / "best.pt"
    trainer.save_checkpoint(path)
    payload = cast(dict[str, object], torch.load(path, weights_only=False))
    # A later validation can leave an earlier model selected.
    trainer.step = 2
    trainer.validation_state["last_step"] = 2
    return trainer, path, payload


def test_selected_checkpoint_accepts_valid_earlier_best_and_runtime_changes(tmp_path: Path) -> None:
    trainer, path, original = _selected_checkpoint_fixture(tmp_path)
    trainer.config.out_dir = str(tmp_path / "other-runtime-directory")
    trainer.config.num_workers = 1
    trainer.config.log_every = 10
    state = copy.deepcopy(trainer.model.state_dict())
    selected = _selected_checkpoint(path, trainer, "baseline-run")
    assert selected["step"] == 1
    assert trainer.step == 2
    assert selected["validation_state"] == original["validation_state"]
    torch.testing.assert_close(trainer.model.state_dict(), state, rtol=0, atol=0)


def test_selected_checkpoint_rejects_another_condition_with_identical_shapes(
    tmp_path: Path,
) -> None:
    trainer, path, payload = _selected_checkpoint_fixture(tmp_path)
    metadata = cast(dict[str, object], payload["run_metadata"])
    metadata["experiment_run_identity"] = "full-run"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="different experiment run"):
        _selected_checkpoint(path, trainer, "baseline-run")


def test_selected_checkpoint_rejects_stale_same_run_best(tmp_path: Path) -> None:
    trainer, path, _ = _selected_checkpoint_fixture(tmp_path)
    trainer.validation_state.update(best_step=2, best_score=0.5)
    with pytest.raises(ValueError, match="saved validation selection"):
        _selected_checkpoint(path, trainer, "baseline-run")


@pytest.mark.parametrize("field", ["lr", "parameter_sparsity_weight"])
def test_selected_checkpoint_rejects_changed_scientific_training_configuration(
    tmp_path: Path,
    field: str,
) -> None:
    trainer, path, payload = _selected_checkpoint_fixture(tmp_path)
    training = cast(dict[str, object], payload["train_config"])
    if field == "lr":
        training["lr"] = trainer.config.lr * 2
    else:
        objective = cast(dict[str, object], training["objective"])
        objective["parameter_sparsity_weight"] = 0.1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="training configuration differs"):
        _selected_checkpoint(path, trainer, "baseline-run")


@pytest.mark.parametrize(
    "field", ["dataset_identity", "model_config", "world_size", "format_version"]
)
def test_selected_checkpoint_rejects_changed_provenance(tmp_path: Path, field: str) -> None:
    trainer, path, payload = _selected_checkpoint_fixture(tmp_path)
    payload[field] = "different" if field in {"dataset_identity", "model_config"} else 9
    torch.save(payload, path)
    with pytest.raises(ValueError, match="provenance differs"):
        _selected_checkpoint(path, trainer, "baseline-run")


def test_evaluate_only_reproduces_selected_results_without_training_or_checkpoint_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    run_experiment(config, condition="baseline")
    directory = tmp_path / "complementary" / "seed-3" / "baseline"
    metrics = read_json(directory / "evaluation" / "metrics.json")
    checkpoint_times = [(directory / name).stat().st_mtime_ns for name in ("last.pt", "best.pt")]
    selected = cast(dict[str, object], torch.load(directory / "best.pt", weights_only=False))
    selection = cast(dict[str, object], selected["validation_state"])
    assert metrics["evaluation/selected_step"] == selected["step"]
    assert metrics["evaluation/selection_physical_nrmse"] == selection["best_score"]

    def forbidden_train(*_args: object, **_kwargs: object) -> dict[str, float]:
        raise AssertionError("evaluation-only must never train the model")

    monkeypatch.setattr(Trainer, "train", forbidden_train)
    result = run_experiment(config, evaluate_only=True, condition="baseline")
    assert result["completed_runs"] == 1
    assert read_json(directory / "evaluation" / "metrics.json") == metrics
    assert [
        (directory / name).stat().st_mtime_ns for name in ("last.pt", "best.pt")
    ] == checkpoint_times
    assert read_json(directory / "status.json")["status"] == "completed"
