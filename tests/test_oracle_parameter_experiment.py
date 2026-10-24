"""Known-parameter control preserves data, paired calibration, and checkpoint semantics."""

import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from scjepa.data.factory import build_dgp_dataset
from scjepa.data.identity import training_tensors_fingerprint
from scjepa.data.types import DGPConfig
from scjepa.experiments.config import ExperimentConfig, load_experiment_config
from scjepa.experiments.oracle import OracleParameterEpisodes
from scjepa.experiments.reference import external_dense_manifest
from scjepa.experiments.runner import run_experiment
from scjepa.experiments.storage import read_json
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.training import seed_everything
from scjepa.training.factory import build_model

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIGS = REPOSITORY / "configs/experiments"
ORACLE_PRESET = CONFIGS / "simplified_gaussian_r10_oracle.yaml"


def _tiny_config(directory: Path, *, source: str = "oracle") -> ExperimentConfig:
    return load_experiment_config(
        ORACLE_PRESET,
        [
            "wandb.mode=disabled",
            f"output_dir={directory}",
            f"model.parameter_source={source}",
            "train.steps=2",
            "train.batch_size=2",
            "train_episodes=4",
            "validation_every=1",
            "train.checkpoint_every=1",
            "train.log_every=1",
            "model.encoder_hidden_dims=[8]",
            "model.spartan_layers=1",
            "model.spartan_embed_dim=8",
            "model.spartan_mlp_hidden=8",
            "model.spartan_mlp_layers=2",
            "train.objective.regularizer_num_projections=4",
            "evaluation.calibration_episodes=4",
            "evaluation.validation_episodes=2",
            "evaluation.test_episodes=2",
            "evaluation.batch_size=2",
        ],
    )


def test_oracle_preset_changes_parameter_inputs_and_removes_only_their_losses(
    tmp_path: Path,
) -> None:
    learned = load_experiment_config(CONFIGS / "simplified_gaussian_r10_consistency.yaml")
    oracle = load_experiment_config(ORACLE_PRESET)
    assert learned.model.parameter_source == "learned"
    assert oracle.model.parameter_source == "oracle"
    assert oracle.data_family == "simplified_gaussian"
    assert oracle.episodes_per_parameter == 10
    assert oracle.model.context_len == 10
    assert oracle.trajectory_length == 12
    assert oracle.train.steps == 100_000
    assert oracle.train.batch_size == oracle.evaluation.batch_size == 16
    assert oracle.output_dir != learned.output_dir
    assert oracle.dense_reference_dir is None
    assert oracle.conditions == ("dense", "baseline", "full")
    learned_model, oracle_model = asdict(learned.model), asdict(oracle.model)
    learned_model["parameter_source"] = "oracle"
    assert learned_model == oracle_model
    for condition in oracle.conditions:
        previous = asdict(learned.training_config(condition, 0, tmp_path, tau=0.3).objective)
        current = asdict(oracle.training_config(condition, 0, tmp_path, tau=0.3).objective)
        previous.update(
            state_regularizer_weight=0.4,
            parameter_regularizer_weight=0.0,
            parameter_consistency_weight=0.0,
        )
        assert current == previous
        model = oracle.condition_model(condition)
        assert model.parameter_source == "oracle"
        assert model.spartan_dense == (condition == "dense")
        assert model.spartan_parameter_gates_open == (condition == "baseline")


@pytest.mark.parametrize("family", ["controlled_vector", "independent_gaussian"])
def test_oracle_inputs_are_rejected_outside_the_explicit_experiment(family: str) -> None:
    with pytest.raises(ValueError, match="oracle.*simplified_gaussian"):
        load_experiment_config(
            CONFIGS / f"{family}_smoke.yaml",
            [
                "model.parameter_source=oracle",
                "train.objective.parameter_regularizer_weight=0",
                "train.objective.parameter_consistency_weight=0",
            ],
        )
    source = build_dgp_dataset(
        DGPConfig(family=family, trajectory_length=5, episodes_per_parameter=2)
    )
    with pytest.raises(ValueError, match="oracle.*simplified_gaussian"):
        OracleParameterEpisodes(source)


@pytest.mark.parametrize(
    "objective_field", ["parameter_regularizer_weight", "parameter_consistency_weight"]
)
def test_oracle_rejects_losses_that_treat_supplied_parameters_as_learned(
    objective_field: str,
) -> None:
    with pytest.raises(ValueError, match="oracle parameters require zero"):
        load_experiment_config(ORACLE_PRESET, [f"train.objective.{objective_field}=0.1"])


def test_oracle_dataset_adds_only_raw_parameters_without_changing_addressed_draws() -> None:
    source = build_dgp_dataset(
        DGPConfig(
            family="simplified_gaussian",
            trajectory_length=12,
            episodes_per_parameter=10,
            num_episodes=3,
        )
    )
    source_identity = source.fingerprint
    oracle = OracleParameterEpisodes(source)
    assert oracle.fingerprint != source_identity
    assert len(oracle) == len(source)
    for index in (0, 1, -1):
        original, supplied = source[index], oracle[index]
        assert set(original) == {"observations", "actions", "episode_ids"}
        assert set(supplied) == {*original, "oracle_parameters"}
        for key, value in original.items():
            torch.testing.assert_close(supplied[key], value, rtol=0, atol=0)
        parameters = supplied["oracle_parameters"]
        assert parameters.shape == (6,)
        assert parameters.dtype == original["observations"].dtype
        assert not parameters.requires_grad
        assert bool(parameters.abs().le(0.75).all())
        torch.testing.assert_close(
            parameters,
            source.ground_truth(index)["parameters"].to(original["observations"]),
            rtol=0,
            atol=0,
        )
    assert source.fingerprint == source_identity
    source.process.config = replace(source.process.config, root_seed=1)
    with pytest.raises(ValueError, match="dataset definition changed"):
        _ = oracle.fingerprint
    with pytest.raises(ValueError, match="dataset definition changed"):
        oracle[0]


def test_three_oracle_stages_bypass_encoder_share_calibration_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_config(tmp_path)

    def forbidden_parameter_encoding(*_args: object, **_kwargs: object) -> Tensor:
        raise AssertionError("oracle training and evaluation must bypass the parameter encoder")

    monkeypatch.setattr(TransitionParameterEncoder, "forward", forbidden_parameter_encoding)
    seed_everything(config.seeds[0])
    initial = build_model(config.condition_model("dense"))
    assert not any(parameter.requires_grad for parameter in initial.parameter_encoder.parameters())
    initial.train()
    assert not initial.parameter_encoder.training
    initial_parameters = {
        key: value.clone()
        for key, value in initial.state_dict().items()
        if key.startswith("parameter_encoder.") and isinstance(value, Tensor)
    }
    assert initial_parameters
    result = run_experiment(config)
    assert result["complete"] is True
    assert result["completed_runs"] == result["expected_runs"] == 3
    parent = tmp_path / "complementary/seed-0"
    tau = read_json(parent / "dense/tau.json")
    dense_selected = torch.load(parent / "dense/best.pt", weights_only=False)
    selected_fingerprint = training_tensors_fingerprint(
        {
            name: value
            for name, value in dense_selected["model"].items()
            if isinstance(value, Tensor)
        }
    )
    records: list[dict[str, object]] = []
    checkpoints: list[Path] = []
    for condition in config.conditions:
        directory = parent / condition
        records.append(read_json(directory / "run_config.json"))
        checkpoints.append(directory / "last.pt")
        checkpoint = torch.load(checkpoints[-1], weights_only=False)
        assert checkpoint["step"] == checkpoint["optimizer_steps"] == 2
        assert checkpoint["model_config"]["parameter_source"] == "oracle"
        assert checkpoint["model"]["_extra_state"]["oracle_parameters"] is True
        objective = checkpoint["train_config"]["objective"]
        assert objective["state_regularizer_weight"] == 0.4
        assert objective["parameter_regularizer_weight"] == 0
        assert objective["parameter_consistency_weight"] == 0
        for name, value in initial_parameters.items():
            torch.testing.assert_close(checkpoint["model"][name], value, rtol=0, atol=0)
        if condition != "dense":
            assert records[-1]["initial_state_fingerprint"] == selected_fingerprint
            assert records[-1]["tau_calibration"] == tau
            assert checkpoint["train_config"]["geco"]["tau"] == tau["tau"]
        evaluated = read_json(directory / "evaluation/evaluated_checkpoint.json")
        assert evaluated["checkpoint_path"] == "last.pt"
        assert evaluated["evaluated_step"] == 2
        metrics = read_json(directory / "evaluation/metrics.json")
        assert all(math.isfinite(value) for value in cast(dict[str, float], metrics).values())
    assert records[0]["tau_calibration"] is None
    assert records[0]["dataset_identity"] == records[1]["dataset_identity"]
    assert records[1]["dataset_identity"] == records[2]["dataset_identity"]
    modification_times = [path.stat().st_mtime_ns for path in checkpoints]
    assert run_experiment(config, resume=True) == result
    assert [path.stat().st_mtime_ns for path in checkpoints] == modification_times
    reevaluated = run_experiment(config, evaluate_only=True)
    assert reevaluated["complete"] is True
    assert [path.stat().st_mtime_ns for path in checkpoints] == modification_times
    reference = replace(
        config,
        output_dir=str(tmp_path / "reused"),
        dense_reference_dir=config.output_dir,
    )
    reused = run_experiment(reference)
    assert reused["complete"] is True
    assert reused["completed_runs"] == reused["expected_runs"] == 2
    for condition in ("baseline", "full"):
        record = read_json(
            Path(reference.output_dir) / "complementary/seed-0" / condition / "run_config.json"
        )
        assert record["initial_state_fingerprint"] == selected_fingerprint
        assert record["tau_calibration"] == tau
    assert [path.stat().st_mtime_ns for path in checkpoints] == modification_times


def test_learned_dense_reference_cannot_supply_an_oracle_calibration(tmp_path: Path) -> None:
    learned = _tiny_config(tmp_path / "learned", source="learned")
    run_experiment(learned, condition="dense")
    oracle = _tiny_config(tmp_path / "oracle")
    oracle.dense_reference_dir = learned.output_dir
    manifest = read_json(Path(learned.output_dir) / "experiment.json")
    with pytest.raises(ValueError, match="architecture differs"):
        external_dense_manifest(oracle, manifest, "complementary", 0)
