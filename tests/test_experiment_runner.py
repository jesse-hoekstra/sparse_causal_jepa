"""Tiny CPU dense calibration, GECO comparison, artifact and resume contracts."""

import copy
import hashlib
import math
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

import scjepa.models.spartan as spartan_module
from scjepa.data.identity import training_tensors_fingerprint
from scjepa.data.processes import ControlledVectorDGP
from scjepa.experiments import storage
from scjepa.experiments.config import ExperimentConfig, load_experiment_config
from scjepa.experiments.evaluation import EvaluationConfig
from scjepa.experiments.runner import (
    ExperimentLogger,
    _selected_checkpoint,  # pyright: ignore[reportPrivateUsage]
    _sparse_selection_score,  # pyright: ignore[reportPrivateUsage]
    experiment_plan,
    run_experiment,
)
from scjepa.experiments.storage import (
    file_lock,
    json_safe,
    prepare_experiment,
    read_json,
    write_json,
)
from scjepa.losses import RepresentationRegularizer
from scjepa.models.model import SCJEPA
from scjepa.training import Trainer
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.logging import WandbLogger


def _config(directory: Path) -> ExperimentConfig:
    config = ExperimentConfig(
        output_dir=str(directory),
        graphs=("complementary",),
        seeds=(3,),
        train_episodes=4,
        trajectory_length=4,
        validation_every=1,
        model=ModelConfig(
            observation_dim=10,
            state_dim=10,
            param_dim=10,
            action_dim=10,
            context_len=2,
            history_len=1,
            autoregressive=True,
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
            batch_size=2,
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


@pytest.mark.parametrize(
    ("closed_gates", "without_sigreg"), [(False, False), (True, False), (False, True)]
)
def test_real_three_stage_experiment_artifacts_and_resume_are_reproducible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    closed_gates: bool,
    without_sigreg: bool,
) -> None:
    config = _config(tmp_path)
    config.tau_multiplier = 1.05
    if without_sigreg:
        config.train.objective.state_regularizer_weight = 0.0
        config.train.objective.parameter_regularizer_weight = 0.0
        config.train.lr_final_ratio = 1.0

        def forbidden_sigreg(*_args: object, **_kwargs: object) -> Tensor:
            raise AssertionError("zero-weight SIGReg must not run in training or evaluation")

        monkeypatch.setattr(RepresentationRegularizer, "forward", forbidden_sigreg)
    if closed_gates:

        def closed_adjacency(logits: Tensor, _temperature: float, _training: bool) -> Tensor:
            return torch.zeros_like(logits)

        monkeypatch.setattr(spartan_module, "_sample_hard_adjacency", closed_adjacency)
    result = run_experiment(config)
    assert result["complete"] is True
    assert result["completed_runs"] == result["expected_runs"] == 3
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
    paths = [tmp_path / "complementary" / "seed-3" / name for name in ("dense", "baseline", "full")]
    records = [read_json(directory / "run_config.json") for directory in paths]
    for name in ("dataset_identity", "experiment_identity"):
        assert records[0][name] == records[1][name] == records[2][name]
        assert isinstance(records[0][name], str)
        assert records[0][name]
    dense_selected = torch.load(paths[0] / "best.pt", weights_only=False)
    warm_start_identity = training_tensors_fingerprint(
        {
            name: value
            for name, value in dense_selected["model"].items()
            if isinstance(value, Tensor)
        }
    )
    assert records[1]["initial_state_fingerprint"] == warm_start_identity
    assert records[2]["initial_state_fingerprint"] == warm_start_identity
    assert records[0]["initial_state_fingerprint"] != warm_start_identity
    saved_objectives: list[dict[str, object]] = []
    tau = read_json(paths[0] / "tau.json")
    assert tau["tau"] == cast(float, tau["calibration/base_loss"]) * config.tau_multiplier
    assert records[0]["tau_calibration"] is None
    assert records[1]["tau_calibration"] == records[2]["tau_calibration"] == tau
    for index, directory in enumerate(paths):
        assert read_json(directory / "status.json")["status"] == "completed"
        last = torch.load(directory / "last.pt", weights_only=False)
        best = torch.load(directory / "best.pt", weights_only=False)
        assert last["step"] == 2
        assert last["optimizer_steps"] == 2
        assert best["step"] == last["validation_state"]["best_step"]
        assert last["validation_state"]["selection_metric"] == (
            "validation/selection_score" if index else "validation/base_loss"
        )
        assert last["train_config"]["geco"]["enabled"] == (index != 0)
        assert last["train_config"]["geco"]["tau"] == (tau["tau"] if index else None)
        assert last["train_config"]["lr_decay_steps"] == (None if index else 2)
        expected_lr = config.train.lr * (1.0 if index else config.train.lr_final_ratio)
        assert math.isclose(last["optimizer"]["param_groups"][0]["lr"], expected_lr)
        assert records[index]["model"]["spartan_dense"] == (index == 0)  # pyright: ignore[reportIndexIssue]
        model_record = cast(dict[str, object], records[index]["model"])
        assert model_record["autoregressive"] is True
        assert last["model_config"]["autoregressive"] is True
        assert best["model_config"]["autoregressive"] is True
        assert model_record["spartan_parameter_gates_open"] == (index == 1)
        assert last["model_config"]["spartan_parameter_gates_open"] == (index == 1)
        assert best["model_config"]["spartan_parameter_gates_open"] == (index == 1)
        saved_objectives.append(last["train_config"]["objective"])
        for name in ("state_regularizer_weight", "parameter_regularizer_weight"):
            assert saved_objectives[-1][name] == getattr(config.train.objective, name)
        assert last["model_config"]["ema_decay"] == config.model.ema_decay == 0.996
        evaluation = directory / "evaluation"
        metrics = read_json(evaluation / "metrics.json")
        assert metrics["test/prediction_mse"] >= 0  # pyright: ignore[reportOperatorIssue]
        assert metrics["test/base_loss"] >= metrics["test/prediction_mse"]  # pyright: ignore[reportOperatorIssue]
        assert metrics["geco/tau"] == tau["tau"]
        for name in ("state", "parameter"):
            assert 0 <= metrics[f"recovery/{name}_mcc"] <= 1  # pyright: ignore[reportOperatorIssue]
        for name in ("state", "action", "parameter"):
            assert metrics[f"graph/{name}_shd"] >= 0  # pyright: ignore[reportOperatorIssue]
        assert all(math.isfinite(value) for value in cast(dict[str, float], metrics).values())
        for artifact in (
            "evaluation_references.pt",
            "protocol.json",
            "selected_checkpoint.json",
        ):
            assert (evaluation / artifact).is_file()
        references = torch.load(evaluation / "evaluation_references.pt", weights_only=True)
        torch.testing.assert_close(references["matrix"], saved_map, rtol=0, atol=0)
        if index == 1:
            assert references["routing_frequencies"]["parameter"].eq(1).all()
            assert references["aligned_support"]["parameter"].all()
            assert metrics["graph/parameter_shd"] == 80
        elif index == 2 and closed_gates:
            assert not references["routing_frequencies"]["parameter"].any()
            assert not references["aligned_support"]["parameter"].any()
            assert metrics["graph/parameter_shd"] == 20
        assert set(references["calibration_episode_ids"].tolist()).isdisjoint(
            references["test_episode_ids"].tolist()
        )
    assert all(
        saved_objectives[0][f"{name}_sparsity_weight"] == 0
        for name in ("state", "action", "parameter")
    )
    assert saved_objectives[1]["parameter_sparsity_weight"] == 0.0
    assert (
        saved_objectives[2]["parameter_sparsity_weight"]
        == config.train.objective.parameter_sparsity_weight
    )
    baseline, full = copy.deepcopy(saved_objectives[1:])
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
    changed = copy.deepcopy(config)
    changed.episodes_per_parameter = 1
    with pytest.raises(ValueError, match="definition/source changed"):
        run_experiment(changed, resume=True)
    assert [(directory / "last.pt").stat().st_mtime_ns for directory in paths] == modification_times


def test_sparse_runs_load_selected_dense_model_with_fresh_training_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path)
    original_train = Trainer.train
    checked: list[str] = []

    def inspect_start(
        trainer: Trainer,
        validation_callback: Callable[[SCJEPA], dict[str, float]] | None = None,
        validation_every: int | None = None,
        selection_metric: str = "validation/base_loss",
        stop_requested: Callable[[], bool] | None = None,
    ) -> dict[str, float]:
        condition = Path(trainer.config.out_dir).name
        assert trainer.dataset is not None
        sample = trainer.dataset[0]
        assert sample["observations"].shape == (2, 4, 10)
        assert sample["actions"].shape == (2, 3, 10)
        callback = validation_callback
        if condition == "dense":
            assert validation_callback is not None

            def select_first(current: SCJEPA) -> dict[str, float]:
                assert validation_callback is not None
                scores = validation_callback(current)
                scores["validation/base_loss"] = float(trainer.step)
                return scores

            callback = select_first
        else:
            dense = Path(trainer.config.out_dir).parent / "dense"
            best = torch.load(dense / "best.pt", weights_only=False)
            last = torch.load(dense / "last.pt", weights_only=False)
            assert best["step"] == 1
            assert last["step"] == 2
            assert any(
                not torch.equal(value, last["model"][name])
                for name, value in best["model"].items()
                if isinstance(value, Tensor)
            )
            torch.testing.assert_close(trainer.model.state_dict(), best["model"], rtol=0, atol=0)
            assert trainer.step == trainer.optimizer_steps == 0
            assert not trainer.optimizer.state
            assert trainer.geco is not None
            assert int(trainer.geco.num_updates) == 0
            assert float(trainer.geco.ma_error) == 0
            torch.testing.assert_close(
                trainer.geco.penalty_weight, torch.tensor(1 / config.train.geco.lambda_init)
            )
            initial_rng = torch.Generator().manual_seed(trainer.config.seed).get_state()
            torch.testing.assert_close(torch.get_rng_state(), initial_rng, rtol=0, atol=0)
            assert not torch.equal(initial_rng, best["rank_rng"][0]["rng_torch"])
            checked.append(condition)
        return original_train(trainer, callback, validation_every, selection_metric, stop_requested)

    monkeypatch.setattr(Trainer, "train", inspect_start)
    assert run_experiment(config)["complete"] is True
    assert checked == ["baseline", "full"]
    output = capsys.readouterr().out
    for condition in checked:
        lines = [line for line in output.splitlines() if line.startswith(f"{condition} warm-start")]
        assert len(lines) == 1
        assert all(
            label in lines[0] for label in ("dense=", "eval gates=", "sampled gates=", "tau=")
        )
        assert lines[0].count("change=") == 2
    run_experiment(config, resume=True)
    assert "warm-start base loss:" not in capsys.readouterr().out


def test_external_legacy_dense_reference_is_reused_without_modifying_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = _config(tmp_path / "exp1")
    definition = ExperimentConfig.definition

    def legacy_definition(config: ExperimentConfig) -> dict[str, object]:
        result = definition(config)
        cast(dict[str, object], result["model"]).pop("spartan_parameter_gates_open")
        result.pop("dense_reference_dir", None)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(storage, "source_fingerprint", lambda: "old-source")
        patch.setattr(ExperimentConfig, "definition", legacy_definition)
        run_experiment(old, condition="dense")
    reference = Path(old.output_dir) / "complementary/seed-3/dense"
    record = read_json(reference / "run_config.json")
    cast(dict[str, object], record["model"]).pop("spartan_parameter_gates_open")
    write_json(reference / "run_config.json", record)
    for name in ("best.pt", "last.pt"):
        payload = torch.load(reference / name, weights_only=False)
        payload["model_config"].pop("spartan_parameter_gates_open")
        torch.save(payload, reference / name)
    tau = read_json(reference / "tau.json")
    tau["checkpoint_sha256"] = hashlib.sha256((reference / "best.pt").read_bytes()).hexdigest()
    write_json(reference / "tau.json", tau)
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in Path(old.output_dir).rglob("*")
        if path.is_file()
    }
    config = _config(tmp_path / "exp2")
    config.dense_reference_dir = old.output_dir
    config.train.lr *= 2
    config.train.steps = 3
    monkeypatch.setattr(storage, "source_fingerprint", lambda: "updated-source")
    assert [task["condition"] for task in experiment_plan(config)] == ["baseline", "full"]
    result = run_experiment(config)
    assert result["complete"] is True
    assert result["completed_runs"] == result["expected_runs"] == 2
    assert result["missing_runs"] == []
    parent = Path(config.output_dir) / "complementary/seed-3"
    assert not (parent / "dense").exists()
    selected = torch.load(reference / "best.pt", weights_only=False)
    fingerprint = training_tensors_fingerprint(
        {name: value for name, value in selected["model"].items() if isinstance(value, Tensor)}
    )
    for condition in ("baseline", "full"):
        current = read_json(parent / condition / "run_config.json")
        assert current["initial_state_fingerprint"] == fingerprint
        assert current["tau_calibration"] == tau
        last = torch.load(parent / condition / "last.pt", weights_only=False)
        assert last["step"] == last["optimizer_steps"] == 3
        assert last["train_config"]["lr"] == config.train.lr
    assert run_experiment(config, resume=True) == result
    assert before == {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in Path(old.output_dir).rglob("*")
        if path.is_file()
    }
    with (reference / "best.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="provenance|checkpoint"):
        run_experiment(config, resume=True)


def test_external_dense_reference_rejects_old_generator_with_unchanged_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = _config(tmp_path / "old-dynamics")
    specification = ControlledVectorDGP.specification

    def legacy_specification(process: ControlledVectorDGP) -> dict[str, object]:
        result = specification(process)
        result.pop("dynamics")
        result["generator_version"] = "paper-dgps-v1"
        result["epsilon"] = 1 / 40
        return result

    with monkeypatch.context() as patch:
        patch.setattr(ControlledVectorDGP, "specification", legacy_specification)
        run_experiment(old, condition="dense")
    config = _config(tmp_path / "new-dynamics")
    config.dense_reference_dir = old.output_dir
    assert config.data_config("complementary") == old.data_config("complementary")

    def forbidden_train(*_args: object, **_kwargs: object) -> dict[str, float]:
        raise AssertionError("old generator references must fail before sparse training")

    monkeypatch.setattr(Trainer, "train", forbidden_train)
    with pytest.raises(ValueError, match="external dense reference dataset generator differs"):
        run_experiment(config)
    assert not list(Path(config.output_dir).glob("*/seed-*/*/last.pt"))


@pytest.mark.parametrize("previous_protocol", ["single_episode", "one_step", "teacher_forced"])
def test_rollout_experiment_rejects_incompatible_dense_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, previous_protocol: str
) -> None:
    old = _config(tmp_path / "previous-protocol")
    if previous_protocol == "single_episode":
        old.episodes_per_parameter = 1
    elif previous_protocol == "one_step":
        old.trajectory_length = 3
    else:
        old.model.autoregressive = False
    run_experiment(old, condition="dense")
    config = _config(tmp_path / "paired-episodes")
    config.dense_reference_dir = old.output_dir

    def forbidden_train(*_args: object, **_kwargs: object) -> dict[str, float]:
        raise AssertionError("incompatible references must fail before sparse training")

    monkeypatch.setattr(Trainer, "train", forbidden_train)
    with pytest.raises(ValueError, match="external dense reference .* differs"):
        run_experiment(config)
    assert not list(Path(config.output_dir).glob("*/seed-*/*/last.pt"))


@pytest.mark.parametrize("problem", ["architecture", "data", "batch", "map", "hash", "missing"])
def test_external_dense_reference_rejects_incompatible_or_changed_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    old = _config(tmp_path / "exp1")
    run_experiment(old, condition="dense")
    config = _config(tmp_path / "exp2")
    config.dense_reference_dir = old.output_dir
    reference = Path(old.output_dir) / "complementary/seed-3/dense"
    if problem == "architecture":
        config.model.encoder_hidden_dims = (9,)
    elif problem == "data":
        config.data_seed += 1
    elif problem == "batch":
        config.train.batch_size = 4
        config.evaluation = replace(config.evaluation, batch_size=4)
    elif problem == "map":
        path = Path(old.output_dir) / "observation_map.pt"
        matrix = torch.load(path, weights_only=True)["matrix"]
        torch.save({"matrix": matrix.roll(1, dims=0)}, path)
    elif problem == "hash":
        with (reference / "best.pt").open("ab") as stream:
            stream.write(b"changed")
    else:
        (reference / "tau.json").rename(reference / "missing-tau.json")

    def forbidden_train(*_args: object, **_kwargs: object) -> dict[str, float]:
        raise AssertionError("invalid dense references must fail before sparse training")

    monkeypatch.setattr(Trainer, "train", forbidden_train)
    with pytest.raises(ValueError, match="reference|calibration|provenance|missing"):
        run_experiment(config)
    assert not list(Path(config.output_dir).glob("*/seed-*/*/last.pt"))


def test_paired_train_configs_differ_only_in_parameter_penalty(tmp_path: Path) -> None:
    config = _config(tmp_path)
    baseline = config.training_config("baseline", 9, tmp_path, tau=0.3)
    full = config.training_config("full", 9, tmp_path, tau=0.3)
    assert baseline.objective.parameter_sparsity_weight == 0
    assert full.objective.parameter_sparsity_weight > 0
    baseline.objective.parameter_sparsity_weight = full.objective.parameter_sparsity_weight
    assert asdict(baseline) == asdict(full)
    assert config.train.objective.parameter_sparsity_weight > 0
    assert config.train.seed != 9
    assert baseline.geco.enabled
    assert full.geco.enabled
    assert baseline.geco.tau == full.geco.tau == 0.3
    assert baseline.lr_decay_steps is full.lr_decay_steps is None
    assert config.condition_model("dense").spartan_dense
    assert not config.condition_model("baseline").spartan_dense
    baseline_model, full_model = config.condition_model("baseline"), config.condition_model("full")
    assert baseline_model.spartan_parameter_gates_open
    assert not full_model.spartan_parameter_gates_open
    assert not config.condition_model("dense").spartan_parameter_gates_open
    baseline_model.spartan_parameter_gates_open = False
    assert asdict(baseline_model) == asdict(full_model)
    assert not config.model.spartan_parameter_gates_open
    with pytest.raises(ValueError, match="calibrated tau"):
        config.training_config("baseline", 9, tmp_path)
    with pytest.raises(ValueError, match="condition"):
        config.training_config("unknown", 9, tmp_path)


def test_dense_lr_decay_horizon_counts_optimizer_updates_only(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.train.steps = 12
    config.train.accumulation_steps = 3
    config.train.lr_final_ratio = 0.2
    config.validate()
    original = config.definition()
    dense = config.training_config("dense", 9, tmp_path)
    assert dense.lr_decay_steps == 4
    assert dense.lr_final_ratio == 0.2
    assert dense.lr == config.train.lr
    for condition in ("baseline", "full"):
        assert config.training_config(condition, 9, tmp_path, tau=0.3).lr_decay_steps is None
    assert config.train.lr_decay_steps is None
    assert config.definition() == original

    config.train.steps = 100_000
    config.train.accumulation_steps = 1
    config.train.lr_decay_steps = 10_000
    config.validate()
    dense = config.training_config("dense", 9, tmp_path)
    assert dense.steps == 100_000
    assert dense.lr_decay_steps == 10_000
    for condition in ("baseline", "full"):
        assert config.training_config(condition, 9, tmp_path, tau=0.3).lr_decay_steps is None


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
        "parameter_gates_open",
        "history",
        "partial",
        "zero_logit",
        "manual_tau",
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
    elif invalid == "parameter_gates_open":
        config.model.spartan_parameter_gates_open = True
    elif invalid == "history":
        config.model.history_len = 2
    elif invalid == "zero_logit":
        config.train.objective.logit_weight = 0
    elif invalid == "manual_tau":
        config.train.geco.tau = 0.3
    else:
        config.train.accumulation_steps = 3
    with pytest.raises(
        ValueError, match="[Ss][Ii][Gg][Rr]eg|comparison|penalt|distinct|boundary|tau|derived"
    ):
        config.validate()


def test_experiment_defaults_use_sigreg_only() -> None:
    config = ExperimentConfig()
    config.validate()
    objective = config.train.objective
    assert objective.state_regularizer == objective.parameter_regularizer == "sigreg"
    assert objective.state_regularizer_weight > 0
    assert objective.parameter_regularizer_weight > 0
    assert objective.parameter_sparsity_weight > 0
    assert objective.logit_weight > 0
    assert config.graphs == ("complementary",)
    assert config.seeds == (0,)
    assert config.tau_multiplier == 1.0
    assert config.episodes_per_parameter == 2
    assert config.trajectory_length == 4
    assert config.model.autoregressive
    assert config.data_config("complementary").episodes_per_parameter == 2


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
    with pytest.raises(ValueError, match="graph_support_threshold"):
        replace(config.evaluation, graph_support_threshold=1.0)


@pytest.mark.parametrize("condition", ["dense", "baseline"])
def test_interrupted_run_resumes_without_overwriting_its_scientific_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    condition: str,
) -> None:
    config = _config(tmp_path)
    if condition == "baseline":
        run_experiment(config, condition="dense")
    original_train = Trainer.train

    def interrupt(
        trainer: Trainer,
        validation_callback: Callable[[SCJEPA], dict[str, float]] | None = None,
        validation_every: int | None = None,
        selection_metric: str = "validation/base_loss",
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
        result = run_experiment(config, condition=condition)
    initial_output = capsys.readouterr().out
    assert initial_output.count("baseline warm-start base loss:") == (condition == "baseline")
    assert result["completed_runs"] == (1 if condition == "baseline" else 0)
    directory = tmp_path / "complementary" / "seed-3" / condition
    assert read_json(directory / "status.json")["status"] == "interrupted"
    saved = torch.load(directory / "last.pt", weights_only=False)
    assert saved["step"] == 1
    assert saved["model_config"]["spartan_parameter_gates_open"] == (condition == "baseline")
    assert not (directory / "best.pt").exists()
    assert not (directory / "tau.json").exists()
    if condition == "dense":
        assert not (directory.parent / "baseline" / "last.pt").exists()
    identity = read_json(directory / "run_config.json")
    checked_resume: list[int] = []

    def inspect_resume(
        trainer: Trainer,
        validation_callback: Callable[[SCJEPA], dict[str, float]] | None = None,
        validation_every: int | None = None,
        selection_metric: str = "validation/base_loss",
        stop_requested: Callable[[], bool] | None = None,
    ) -> dict[str, float]:
        if Path(trainer.config.out_dir) == directory:
            assert trainer.step == trainer.optimizer_steps == 1
            torch.testing.assert_close(trainer.model.state_dict(), saved["model"], rtol=0, atol=0)
            torch.testing.assert_close(
                trainer.optimizer.state_dict(), saved["optimizer"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                trainer.objective.state_dict(), saved["objective"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                torch.get_rng_state(), saved["rank_rng"][0]["rng_torch"], rtol=0, atol=0
            )
            checked_resume.append(trainer.step)
        return original_train(
            trainer, validation_callback, validation_every, selection_metric, stop_requested
        )

    with monkeypatch.context() as patch:
        patch.setattr(Trainer, "train", inspect_resume)
        resumed = run_experiment(config, resume=True)
    resumed_output = capsys.readouterr().out
    assert resumed_output.count("baseline warm-start base loss:") == (condition == "dense")
    assert resumed_output.count("full warm-start base loss:") == 1
    assert checked_resume == [1]
    assert resumed["completed_runs"] == 3
    assert resumed["complete"] is True
    assert read_json(directory / "status.json")["status"] == "completed"
    final = torch.load(directory / "last.pt", weights_only=False)
    assert final["step"] == final["optimizer_steps"] == 2
    assert final["model_config"]["spartan_parameter_gates_open"] == (condition == "baseline")
    assert final["validation_state"]["best_step"] in {1, 2}
    assert read_json(directory / "run_config.json") == identity


def test_plan_selectors_do_not_mutate_the_comparison_definition(tmp_path: Path) -> None:
    config = _config(tmp_path)
    original = config.definition()
    all_tasks = experiment_plan(config)
    selected = experiment_plan(config, graph="complementary", seed=3, condition="full")
    assert len(all_tasks) == 3
    assert selected == [all_tasks[2]]
    assert [item["condition"] for item in all_tasks] == ["dense", "baseline", "full"]
    assert config.definition() == original
    with pytest.raises(ValueError, match="selected graph"):
        experiment_plan(config, graph="diagonal")
    with pytest.raises(ValueError, match="selected seed"):
        experiment_plan(config, seed=9)
    with pytest.raises(ValueError, match="condition"):
        experiment_plan(config, condition="incorrect")


def test_resume_rejects_changed_checkpoint_objective(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_experiment(config)
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
        config.training_config("baseline", 3, directory, tau=0.3),
        model_config=asdict(config.model),
        dataset_identity="controlled-vector-test-dataset",
    )
    trainer.run_metadata["experiment_run_identity"] = "baseline-run"
    trainer.step = 1
    trainer.validation_state = {
        "selection_metric": "validation/selection_score",
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
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path)
    run_experiment(config)
    capsys.readouterr()
    directory = tmp_path / "complementary" / "seed-3" / "baseline"
    metrics = read_json(directory / "evaluation" / "metrics.json")
    checkpoint_times = [(directory / name).stat().st_mtime_ns for name in ("last.pt", "best.pt")]
    selected = cast(dict[str, object], torch.load(directory / "best.pt", weights_only=False))
    selection = cast(dict[str, object], selected["validation_state"])
    assert metrics["evaluation/selected_step"] == selected["step"]
    assert metrics["evaluation/selection_score"] == selection["best_score"]

    def forbidden_train(*_args: object, **_kwargs: object) -> dict[str, float]:
        raise AssertionError("evaluation-only must never train the model")

    monkeypatch.setattr(Trainer, "train", forbidden_train)
    result = run_experiment(config, evaluate_only=True, condition="baseline")
    assert "warm-start base loss:" not in capsys.readouterr().out
    assert result["completed_runs"] == 3
    assert read_json(directory / "evaluation" / "metrics.json") == metrics
    assert [
        (directory / name).stat().st_mtime_ns for name in ("last.pt", "best.pt")
    ] == checkpoint_times
    assert read_json(directory / "status.json")["status"] == "completed"


@pytest.mark.parametrize("condition", ["baseline", "full"])
def test_sparse_conditions_require_dense_calibration_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: str
) -> None:
    config = _config(tmp_path)

    def forbidden_train(*_args: object, **_kwargs: object) -> dict[str, float]:
        raise AssertionError("missing dense calibration must be rejected before training")

    monkeypatch.setattr(Trainer, "train", forbidden_train)
    with pytest.raises(ValueError, match="dense tau calibration is missing"):
        run_experiment(config, condition=condition)
    assert not (tmp_path / "complementary" / "seed-3" / condition / "last.pt").exists()


@pytest.mark.parametrize(
    "field", ["tau", "dense_run_identity", "experiment_identity", "checkpoint_sha256"]
)
def test_sparse_run_rejects_tampered_tau_record_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    config = _config(tmp_path)
    run_experiment(config, condition="dense")
    path = tmp_path / "complementary" / "seed-3" / "dense" / "tau.json"
    record = read_json(path)
    record[field] = cast(float, record[field]) * 2 if field == "tau" else "different"
    write_json(path, record)

    def forbidden_train(*_args: object, **_kwargs: object) -> dict[str, float]:
        raise AssertionError("invalid tau provenance must be rejected before training")

    monkeypatch.setattr(Trainer, "train", forbidden_train)
    with pytest.raises(ValueError, match="dense tau calibration"):
        run_experiment(config, condition="baseline")


def test_sparse_run_rejects_changed_dense_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_experiment(config, condition="dense")
    path = tmp_path / "complementary" / "seed-3" / "dense" / "best.pt"
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["step"] += 1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="dense tau calibration provenance"):
        run_experiment(config, condition="full")


def test_experiment_logger_keeps_objective_geco_and_recovery_scores_only() -> None:
    captured: list[tuple[int, dict[str, float]]] = []

    class Capture:
        def log(self, step: int, metrics: dict[str, float]) -> None:
            captured.append((step, metrics))

    logger = ExperimentLogger(cast(WandbLogger, Capture()))
    visible = {
        "loss/prediction": 0.2,
        "loss/base": 0.3,
        "loss/sparsity": 2.0,
        "geco/tau": 0.35,
        "geco/constraint": -0.05,
        "geco/sparsity_weight": 0.001,
        "validation/base_loss": 0.32,
        "validation/target_variance": 0.5,
        "validation/normalized_prediction_mse": 0.4,
        "recovery/state_mcc": 0.8,
        "graph/parameter_shd": 3.0,
        "health/skipped_steps": 0.0,
        "train/lr": 0.0001,
    }
    logger.log(4, {**visible, "attention/gate_entropy": 0.1, "grad/parameter_encoder": 2.0})
    assert captured == [(4, visible)]


@pytest.mark.parametrize("problem", ["batch_size", "partial_batch"])
def test_tau_rejects_different_sigreg_statistical_batches_before_outputs(
    tmp_path: Path, problem: str
) -> None:
    output = tmp_path / "new-run"
    config = _config(output)
    if problem == "batch_size":
        config.train.batch_size = 4
    else:
        config.evaluation = replace(config.evaluation, calibration_episodes=3)
    with pytest.raises(ValueError, match="SIGReg statistic|divisible"):
        run_experiment(config)
    assert not output.exists()


def test_completed_sparse_resume_revalidates_dense_tau(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_experiment(config)
    path = tmp_path / "complementary" / "seed-3" / "dense" / "tau.json"
    record = read_json(path)
    record["checkpoint_sha256"] = "changed"
    write_json(path, record)
    with pytest.raises(ValueError, match="dense tau calibration provenance"):
        run_experiment(config, condition="full", resume=True)


def test_sparse_checkpoint_selection_prefers_feasibility_then_sparsity() -> None:
    assert _sparse_selection_score(0.9, 10000.0, 1.0) < _sparse_selection_score(1.001, 0.0, 1.0)
    assert _sparse_selection_score(0.95, 2.0, 1.0) < _sparse_selection_score(0.1, 3.0, 1.0)
    assert _sparse_selection_score(1.1, 100.0, 1.0) < _sparse_selection_score(2.0, 0.0, 1.0)
