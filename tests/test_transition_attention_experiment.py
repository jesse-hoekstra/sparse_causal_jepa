"""Transition-attention integration preserves the matched SIGReg comparison."""

from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from scjepa.data.identity import training_tensors_fingerprint
from scjepa.experiments.config import load_experiment_config
from scjepa.experiments.reference import external_dense_manifest
from scjepa.experiments.runner import experiment_plan, run_experiment
from scjepa.experiments.storage import read_json, write_json
from scjepa.models import TransitionAttentionParameterEncoder
from scjepa.training.factory import build_model

CONFIGS = Path(__file__).resolve().parents[1] / "configs/experiments"
PRESET = CONFIGS / "simplified_gaussian_r10_transition_attention.yaml"


def test_preset_changes_only_parameter_architecture_from_sigreg04_comparison() -> None:
    previous = load_experiment_config(
        CONFIGS / "simplified_gaussian_r10_consistency.yaml",
        [
            "train.batch_size=16",
            "evaluation.batch_size=16",
            "train.objective.state_regularizer_weight=0.4",
            "train.objective.parameter_regularizer_weight=0.4",
        ],
    )
    current = load_experiment_config(PRESET)
    old_definition, new_definition = asdict(previous), asdict(current)
    for name in ("name", "output_dir", "wandb"):
        old_definition.pop(name)
        new_definition.pop(name)
    cast(dict[str, object], old_definition["model"]).update(
        parameter_encoder_kind="transition_attention", parameter_embed_dim=128
    )
    assert old_definition == new_definition
    assert current.output_dir != previous.output_dir
    assert current.dense_reference_dir is None
    assert [task["condition"] for task in experiment_plan(current)] == [
        "dense",
        "baseline",
        "full",
    ]


def test_same_seed_preserves_state_and_predictor_initialization_and_rng() -> None:
    current = load_experiment_config(PRESET).condition_model("dense")
    previous = replace(current, parameter_encoder_kind="transition", parameter_embed_dim=64)
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        torch.default_generator.manual_seed(29)
        old_model = build_model(previous)
        old_rng = torch.get_rng_state()
        torch.default_generator.manual_seed(29)
        new_model = build_model(current)
        new_rng = torch.get_rng_state()
    assert isinstance(new_model.parameter_encoder, TransitionAttentionParameterEncoder)
    for module in ("online_encoder", "target_encoder", "predictor"):
        old_state = getattr(old_model, module).state_dict()
        new_state = getattr(new_model, module).state_dict()
        torch.testing.assert_close(old_state, new_state, rtol=0, atol=0)
    torch.testing.assert_close(old_rng, new_rng, rtol=0, atol=0)


def test_factory_enforces_transition_context_and_keeps_future_frames_out_of_code() -> None:
    config = load_experiment_config(PRESET).condition_model("dense")
    with pytest.raises(ValueError, match="context_len>=2"):
        build_model(replace(config, context_len=1))
    model = build_model(config).eval()
    with pytest.raises(ValueError, match="context_len>=2"):
        model.set_extra_state({**model.get_extra_state(), "context_len": 1})
    states = torch.randn(2, 3, 12, 6, requires_grad=True)
    actions = torch.randn(2, 3, 11, 16, requires_grad=True)
    output = model(states, actions)
    assert output.parameters.shape == (2, 3, 6)
    assert output.predictions.shape == (2, 3, 2, 6)
    changed_states, changed_actions = states.detach().clone(), actions.detach().clone()
    changed_states[:, :, 10:] += 30
    changed_actions[:, :, 9:] -= 20
    changed = model(changed_states, changed_actions)
    torch.testing.assert_close(output.parameters, changed.parameters, rtol=0, atol=0)
    output.parameters.square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
    assert states.grad is not None
    assert actions.grad is not None
    assert bool(states.grad[:, :, :10].abs().sum() > 0)
    assert bool(actions.grad[:, :, :9].abs().sum() > 0)
    assert not bool(states.grad[:, :, 10:].any())
    assert not bool(actions.grad[:, :, 9:].any())


def test_three_attention_stages_share_dense_weights_and_restore_checkpoints(tmp_path: Path) -> None:
    config = load_experiment_config(
        PRESET,
        [
            "wandb.mode=disabled",
            f"output_dir={tmp_path}",
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
    result = run_experiment(config)
    assert result["complete"] is True
    assert result["completed_runs"] == result["expected_runs"] == 3
    parent = tmp_path / "complementary/seed-0"
    selected = torch.load(parent / "dense/best.pt", weights_only=False)
    initial_fingerprint = training_tensors_fingerprint(
        {name: value for name, value in selected["model"].items() if isinstance(value, Tensor)}
    )
    tau = read_json(parent / "dense/tau.json")
    for condition in config.conditions:
        directory = parent / condition
        record = read_json(directory / "run_config.json")
        checkpoint = torch.load(directory / "last.pt", weights_only=False)
        assert checkpoint["step"] == checkpoint["optimizer_steps"] == 2
        assert checkpoint["model_config"]["parameter_encoder_kind"] == "transition_attention"
        assert any("context_blocks" in name for name in checkpoint["model"])
        assert checkpoint["model_config"]["spartan_parameter_gates_open"] == (
            condition == "baseline"
        )
        if condition != "dense":
            assert record["initial_state_fingerprint"] == initial_fingerprint
            assert record["tau_calibration"] == tau
        evaluated = read_json(directory / "evaluation/evaluated_checkpoint.json")
        assert evaluated["checkpoint_path"] == "last.pt"
        assert evaluated["evaluated_step"] == 2

    # Re-enter an interrupted final stage to exercise Trainer.load_checkpoint,
    # rather than merely skipping three completed status records on resume.
    status_path = parent / "full/status.json"
    status = read_json(status_path)
    status["status"] = "interrupted"
    write_json(status_path, status)
    assert run_experiment(config, resume=True)["complete"] is True
    final_checkpoint = parent / "full/last.pt"
    saved_time = final_checkpoint.stat().st_mtime_ns
    assert run_experiment(config, evaluate_only=True)["complete"] is True
    assert final_checkpoint.stat().st_mtime_ns == saved_time

    incompatible = replace(
        config,
        model=replace(config.model, parameter_encoder_kind="transition"),
        output_dir=str(tmp_path / "incompatible"),
        dense_reference_dir=str(tmp_path),
    )
    manifest = read_json(tmp_path / "experiment.json")
    with pytest.raises(ValueError, match="architecture differs"):
        external_dense_manifest(incompatible, manifest, "complementary", 0)
