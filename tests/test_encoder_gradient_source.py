import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from diagnostics.encoder_gradient_control.source import (
    build_source_model,
    load_source,
    training_code_identity,
    validate_output_destination,
)
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.multiquery_control.run import frozen_hashes, sha
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2))


def refresh_hashes(root: Path) -> None:
    protocol = read_json(root / "protocol.json")
    for filename in ("train_inputs.pt", "evaluation_data.pt"):
        protocol["artifacts"][filename] = sha(root / filename)
    write_json(root / "protocol.json", protocol)
    arm = root / "seed-7/prediction_1"
    result = read_json(arm / "results.json")
    result["checkpoint_sha256"] = sha(arm / "last.pt")
    write_json(arm / "results.json", result)
    write_json(
        arm / "completed.json",
        {
            "identity": protocol["identity"],
            "checkpoint_sha256": sha(arm / "last.pt"),
            "results_sha256": sha(arm / "results.json"),
            "gradient_history_sha256": sha(arm / "gradient_history.json"),
        },
    )


def write_source_fixture(root: Path, *, steps: int = 2) -> Path:
    """Write a small, genuinely trained source with populated Adam moments."""
    root.mkdir(parents=True)
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(123)
        config = ModelConfig(
            observation_dim=6,
            encoder_hidden_dims=(8,),
            state_dim=6,
            param_dim=6,
            action_dim=16,
            context_len=10,
            history_len=1,
            parameter_encoder_kind="transition",
            spartan_layers=1,
            spartan_embed_dim=4,
            spartan_mlp_hidden=8,
            spartan_mlp_layers=2,
            spartan_dense=True,
            autoregressive=True,
        )
        model = build_model(config)
        model.online_encoder.requires_grad_(False).eval()
        model.target_encoder.requires_grad_(False).eval()
        observation = VectorObservationMap(dimension=6)
        metadata = {
            "generator_version": dgp.GENERATOR_VERSION,
            "observation_map_fingerprint": observation.fingerprint,
            "observation_matrix": observation.matrix,
            "state_online_sha256": module_fingerprint(model.online_encoder),
            "state_target_sha256": module_fingerprint(model.target_encoder),
            "views_per_group": 10,
            "context_length": 10,
            "trajectory_length": 12,
            "training_labels_absent": True,
        }
        evaluation: EvaluationData = {"metadata": metadata}
        for name in ("calibration", "validation", "test"):
            theta = dgp.PARAMETER_BOUND * (2 * torch.rand(4, 6, dtype=torch.float64) - 1)
            initial = torch.randn(4, 10, 6, dtype=torch.float64)
            controls = torch.cat(
                (
                    torch.randn(4, 10, 11, 10, dtype=torch.float64),
                    2 * torch.rand(4, 10, 11, 6, dtype=torch.float64) - 1,
                ),
                -1,
            )
            actions = dgp.encode_controls(controls)
            states = dgp.simulate(initial, theta[:, None], actions)
            observations = observation.forward(states).float()
            with torch.no_grad():
                online = model.online_encoder.encode_sequence(observations.flatten(0, 1))
                target = model.target_encoder.encode_sequence(observations.flatten(0, 1))
            evaluation[name] = {
                "theta": theta,
                "true_states": states,
                "true_actions": actions,
                "states_online": online.reshape(4, 10, 12, 6),
                "states_target": target.reshape(4, 10, 12, 6),
                "actions": actions.float(),
            }
        split = cast(TensorSplit, evaluation["calibration"])
        train = {
            "online": split["states_online"].clone(),
            "target": split["states_target"].clone(),
            "actions": split["actions"].clone(),
        }
        torch.save(train, root / "train_inputs.pt")
        torch.save(evaluation, root / "evaluation_data.pt")
        objective_config = ObjectiveConfig(
            state_regularizer="sigreg",
            parameter_regularizer="sigreg",
            state_regularizer_weight=0.4,
            parameter_regularizer_weight=0.4,
            parameter_consistency_weight=1.0,
            regularizer_num_projections=8,
            logit_weight=0.0001,
        )
        protocol: dict[str, Any] = {
            "identity": "source-identity",
            "code_identity": training_code_identity(),
            "config": {
                "seeds": [7],
                "prediction_weights": [1, 4, 16],
                "batch_size": 2,
                "steps": steps,
                "train_groups": 4,
                "calibration_groups": 4,
                "validation_groups": 4,
                "test_groups": 4,
                "lr": 0.001,
            },
            "model_config": asdict(config),
            "objective_config": asdict(objective_config),
            "source": {
                "sha256": "source-best-not-required",
                "observation_map_fingerprint": observation.fingerprint,
            },
            "frozen_hashes": frozen_hashes(model),
            "initializations": {"7": {"sha256": "initial-not-required"}},
            "artifacts": {},
        }
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.0)
        objective = PredictiveSupervisionObjective(objective_config, 1.0)
        clipped = 0
        for step in range(1, steps + 1):
            epoch, position = divmod(step - 1, 2)
            indices = list(GlobalBatchSampler(4, 2, 7, epoch))[position]
            batch = {name: values[indices] for name, values in train.items()}
            torch.default_generator.manual_seed(7 * 100000 + step)
            model.train()
            model.online_encoder.eval()
            model.target_encoder.eval()
            optimizer.zero_grad(set_to_none=True)
            output = cached_multiquery_forward(model, batch, queries=1)
            loss, _ = objective(output)
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            clipped += int(float(norm) > 1.0)
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        arm = root / "seed-7/prediction_1"
        arm.mkdir(parents=True)
        payload: dict[str, Any] = {
            "identity": protocol["identity"],
            "condition": "prediction_1",
            "seed": 7,
            "prediction_weight": 1.0,
            "step": steps,
            "source_sha256": protocol["source"]["sha256"],
            "initial_sha256": protocol["initializations"]["7"]["sha256"],
            "parameter_regularizer_weight": 0.4,
            "consistency_weight": 1.0,
            "queries": 1,
            "frozen_hashes": protocol["frozen_hashes"],
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "gradient_history": [],
            "clipped_updates": clipped,
            "wandb": {
                "id": "original-run-do-not-resume",
                "mode": "online",
                "project": "original-project",
                "entity": None,
            },
        }
        torch.save(payload, arm / "last.pt")
        write_json(arm / "gradient_history.json", {"identity": protocol["identity"], "history": []})
        write_json(
            arm / "results.json",
            {
                "identity": protocol["identity"],
                "condition": "prediction_1",
                "seed": 7,
                "prediction_weight": 1.0,
                "steps": steps,
                "metrics": {"prediction_mse": 0.2},
            },
        )
        write_json(root / "protocol.json", protocol)
        refresh_hashes(root)
    return root


@pytest.fixture
def source(tmp_path: Path) -> Path:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return write_source_fixture(tmp_path / "source")
    finally:
        torch.set_num_threads(previous)


def test_source_restores_independent_model_and_optimizer_copies(source: Path) -> None:
    before = {str(path): sha(path) for path in source.rglob("*") if path.is_file()}
    rng = torch.get_rng_state().clone()
    bundle = load_source(source, 7)
    assert (bundle.source_step, bundle.batch_size, bundle.train_groups) == (2, 2, 4)
    assert torch.equal(rng, torch.get_rng_state())
    model, optimizer = build_source_model(bundle, torch.device("cpu"))
    model2, optimizer2 = build_source_model(bundle, torch.device("cpu"))
    assert not any(p.requires_grad for p in model.online_encoder.parameters())
    assert not any(p.requires_grad for p in model.target_encoder.parameters())
    assert frozen_hashes(model) == bundle.protocol["frozen_hashes"]
    p = next(p for p in model.parameters() if p.requires_grad)
    p2 = next(p for p in model2.parameters() if p.requires_grad)
    assert torch.equal(p, p2)
    assert p.data_ptr() != p2.data_ptr()
    state = cast(dict[str, Any], optimizer.state[p])
    state2 = cast(dict[str, Any], optimizer2.state[p2])
    original = cast(dict[str, Any], next(iter(bundle.checkpoint["optimizer"]["state"].values())))
    assert float(state["step"]) == 2
    assert torch.equal(state["exp_avg"], state2["exp_avg"])
    state["exp_avg"].add_(1)
    assert not torch.equal(state["exp_avg"], state2["exp_avg"])
    assert torch.equal(original["exp_avg"], state2["exp_avg"])
    assert before == {str(path): sha(path) for path in source.rglob("*") if path.is_file()}


@pytest.mark.parametrize(
    "name",
    [
        "train_inputs.pt",
        "evaluation_data.pt",
        "seed-7/prediction_1/last.pt",
        "seed-7/prediction_1/results.json",
        "seed-7/prediction_1/gradient_history.json",
    ],
)
def test_artifact_tampering_is_rejected(source: Path, name: str) -> None:
    path = source / name
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch|artifact changed"):
        load_source(source, 7)


def test_source_code_identity_is_required(source: Path) -> None:
    protocol = read_json(source / "protocol.json")
    protocol["code_identity"] = "changed"
    write_json(source / "protocol.json", protocol)
    with pytest.raises(ValueError, match="training code changed"):
        load_source(source, 7)


def test_label_bearing_training_cache_is_rejected(source: Path) -> None:
    path = source / "train_inputs.pt"
    train = cast(TensorSplit, torch.load(path, weights_only=True))
    train["theta"] = torch.zeros(4, 6)
    torch.save(train, path)
    refresh_hashes(source)
    with pytest.raises(ValueError, match="only frozen states"):
        load_source(source, 7)


def test_cache_encoder_identity_uses_sorted_fingerprint(source: Path) -> None:
    path = source / "evaluation_data.pt"
    data = cast(EvaluationData, torch.load(path, weights_only=True))
    data["metadata"]["state_online_sha256"] = "incorrect"
    torch.save(data, path)
    refresh_hashes(source)
    with pytest.raises(ValueError, match="state-encoder metadata"):
        load_source(source, 7)


@pytest.mark.parametrize(
    ("key", "value"),
    [("step", 1), ("prediction_weight", 4), ("consistency_weight", 0), ("queries", 2)],
)
def test_nonmatching_checkpoint_is_rejected(source: Path, key: str, value: int) -> None:
    path = source / "seed-7/prediction_1/last.pt"
    data = cast(dict[str, Any], torch.load(path, weights_only=False))
    data[key] = value
    torch.save(data, path)
    refresh_hashes(source)
    with pytest.raises(ValueError, match="selected weight-one|completed matched"):
        load_source(source, 7)


@pytest.mark.parametrize("extra", [{"autoregressive": False}, {"context_len": 9}])
def test_checkpoint_extra_state_is_validated(source: Path, extra: dict[str, Any]) -> None:
    path = source / "seed-7/prediction_1/last.pt"
    data = cast(dict[str, Any], torch.load(path, weights_only=False))
    data["model"]["_extra_state"].update(extra)
    torch.save(data, path)
    refresh_hashes(source)
    with pytest.raises(ValueError, match="original MLP"):
        load_source(source, 7)


def test_missing_adam_moments_are_rejected(source: Path) -> None:
    path = source / "seed-7/prediction_1/last.pt"
    data = cast(dict[str, Any], torch.load(path, weights_only=False))
    data["optimizer"]["state"] = {}
    torch.save(data, path)
    refresh_hashes(source)
    with pytest.raises(ValueError, match="Adam moments"):
        load_source(source, 7)


def test_source_output_overlap_is_rejected(source: Path, tmp_path: Path) -> None:
    for output in (source, source / "new_outputs"):
        with pytest.raises(ValueError, match="outside"):
            validate_output_destination(source, output)
    link = tmp_path / "source_link"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        validate_output_destination(source, link / "new_outputs")
    validate_output_destination(source, tmp_path / "separate_outputs")
