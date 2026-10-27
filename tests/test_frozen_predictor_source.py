import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from diagnostics.frozen_predictor_control.source import (
    build_fresh_model,
    export_source_bundle,
    frozen_module_hashes,
    legacy_fingerprint,
    load_source_bundle,
    manifest_identity,
)
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiquery_control.run import sha
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2))


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def source_arguments(root: Path) -> dict[str, Path]:
    """Return export arguments for the locally constructed old-format source fixture."""
    source = root / "original"
    return {
        "checkpoint": source / "washout/last.pt",
        "source_protocol": source / "protocol.json",
        "source_evaluation": source / "washout_evaluation.json",
        "run_config": source / "run_config.json",
        "training_cache": source / "cache.pt",
        "evaluation_cache": source / "evaluation_data.pt",
        "output_dir": root / "source_bundle",
    }


def write_source_fixture(root: Path) -> Path:
    """Export a tiny old-format positive-control fixture with four independent groups."""
    source = root / "original"
    (source / "washout").mkdir(parents=True)
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
        # Distinguish the supplied, already fitted q from every ordinary fresh initialization.
        with torch.no_grad():
            for parameter in model.parameter_encoder.parameters():
                parameter.add_(0.2)
        observation = VectorObservationMap(dimension=6)
        torch.save({"matrix": observation.matrix}, source / "observation_map.pt")
        metadata = {
            "generator_version": dgp.GENERATOR_VERSION,
            "observation_map_fingerprint": observation.fingerprint,
            "state_online_sha256": module_fingerprint(model.online_encoder),
            "state_target_sha256": module_fingerprint(model.target_encoder),
            "views_per_group": 10,
            "context_length": 10,
            "trajectory_length": 12,
        }
        evaluation: EvaluationData = {"metadata": metadata}
        for name in ("train", "calibration", "validation", "test"):
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
        train_split = cast(TensorSplit, evaluation.pop("train"))
        train = {
            "online": train_split["states_online"],
            "target": train_split["states_target"],
            "actions": train_split["actions"],
        }
        torch.save(
            {"train": train, "other_evaluation_labels": train_split["theta"]}, source / "cache.pt"
        )
        torch.save(evaluation, source / "evaluation_data.pt")
        payload = {
            "arm": "mlp",
            "stage": "washout",
            "step": 10000,
            "source_sha256": "original-state-checkpoint",
            "model": model.state_dict(),
            "optimizer": {"must_not_export": True},
            "freeze_hashes": {
                name: legacy_fingerprint(getattr(model, name))
                for name in ("online_encoder", "target_encoder")
            },
        }
        checkpoint = source / "washout/last.pt"
        torch.save(payload, checkpoint)
        _write_json(
            source / "washout/completed.json",
            {
                "stage": "washout",
                "completed_steps": 10000,
                "frozen_unchanged": True,
                "last_sha256": sha(checkpoint),
            },
        )
        provenance = {
            "sha256": payload["source_sha256"],
            "run_identity": "run",
            "dataset_identity": "dataset",
            "observation_map": str(source / "observation_map.pt"),
            "observation_map_fingerprint": observation.fingerprint,
            "generator_version": dgp.GENERATOR_VERSION,
        }
        _write_json(
            source / "protocol.json",
            {
                "source": provenance,
                "source_sha256": payload["source_sha256"],
                "input_cache_sha256": sha(source / "cache.pt"),
                "evaluation_cache_sha256": sha(source / "evaluation_data.pt"),
            },
        )
        _write_json(
            source / "run_config.json",
            {
                "model": asdict(config),
                "run_identity": "run",
                "dataset_identity": "dataset",
                "train": {
                    "batch_size": 2,
                    "objective": asdict(
                        ObjectiveConfig(
                            state_regularizer="sigreg",
                            parameter_regularizer="sigreg",
                            state_regularizer_weight=0.4,
                            parameter_regularizer_weight=0.4,
                            parameter_consistency_weight=1.0,
                            regularizer_num_projections=8,
                            logit_weight=0.0001,
                        )
                    ),
                },
                "data": {
                    "family": "simplified_gaussian",
                    "episodes_per_parameter": 10,
                    "trajectory_length": 12,
                },
            },
        )
        _write_json(
            source / "washout_evaluation.json",
            {
                "checkpoint_sha256": sha(checkpoint),
                "checkpoint_stage": "washout",
                "checkpoint_step": 10000,
                "source_sha256": payload["source_sha256"],
                "evaluation_data_sha256": sha(source / "evaluation_data.pt"),
                "validation_selected_joint_readout": {"test_r2": [0.9] * 6},
                "intervention_summary": {
                    "per_parameter": [
                        {
                            "parameter": index,
                            "response_cosine": 0.7,
                            "response_energy_ratio": 0.6,
                            "counterfactual_mse_improvement": {
                                "paired_theta_group_bootstrap_95percent_interval": [0.01, 0.02]
                            },
                        }
                        for index in range(1, 7)
                    ]
                },
            },
        )
    arguments = source_arguments(root)
    export_source_bundle(**arguments)
    return arguments["output_dir"]


def _refresh_manifest(root: Path) -> None:
    manifest = _read_json(root / "manifest.json")
    manifest["artifacts"] = {name: sha(root / name) for name in manifest["artifacts"]}
    manifest["identity"] = manifest_identity(manifest)
    _write_json(root / "manifest.json", manifest)


def test_portable_export_excludes_labels_encoder_optimizer_and_preserves_sources(
    tmp_path: Path,
) -> None:
    source = write_source_fixture(tmp_path)
    original = tmp_path / "original"
    hashes = {path: sha(path) for path in original.rglob("*") if path.is_file()}
    bundle = load_source_bundle(source)
    assert set(bundle.frozen_payload) == {
        "online_encoder",
        "target_encoder",
        "predictor",
        "extra_state",
    }
    assert set(bundle.train) == {"online", "target", "actions"}
    assert bundle.manifest["provenance"]["label_assisted_predictor_preparation"] is True
    assert bundle.manifest["reference_code_stats"]["count"] == 40
    assert {path: sha(path) for path in hashes} == hashes
    renamed = tmp_path / "relocated"
    original.rename(renamed)
    assert load_source_bundle(source).manifest == bundle.manifest


def test_fresh_factory_resets_only_encoder_and_preserves_rng(tmp_path: Path) -> None:
    bundle = load_source_bundle(write_source_fixture(tmp_path))
    rng = torch.get_rng_state().clone()
    first = build_fresh_model(bundle, seed=7, device=torch.device("cpu"))
    assert torch.equal(torch.get_rng_state(), rng)
    second = build_fresh_model(bundle, seed=7, device=torch.device("cpu"))
    third = build_fresh_model(bundle, seed=8, device=torch.device("cpu"))
    q_hash = module_fingerprint(first.parameter_encoder)
    assert q_hash == module_fingerprint(second.parameter_encoder)
    assert q_hash != module_fingerprint(third.parameter_encoder)
    assert q_hash != bundle.manifest["provenance"]["source_q_sha256"]
    for model in (first, second, third):
        assert frozen_module_hashes(model) == bundle.manifest["frozen_hashes"]
        assert all(
            p.requires_grad == name.startswith("parameter_encoder.")
            for name, p in model.named_parameters()
        )
        assert not model.online_encoder.training
        assert not model.target_encoder.training
        assert not model.predictor.training
    before = copy.deepcopy(bundle.frozen_payload)
    with torch.no_grad():
        next(first.predictor.parameters()).add_(1)
    for name in ("online_encoder", "target_encoder", "predictor"):
        assert all(
            torch.equal(value, before[name][key])
            for key, value in bundle.frozen_payload[name].items()
        )


@pytest.mark.parametrize(
    "kind", ["hash", "labels", "dtype", "map", "frozen", "q_payload", "evidence"]
)
def test_loader_rejects_corrupt_or_incompatible_bundle(tmp_path: Path, kind: str) -> None:
    root = write_source_fixture(tmp_path)
    if kind in {"hash", "labels", "dtype"}:
        path = root / "train_inputs.pt"
        value = cast(TensorSplit, torch.load(path, weights_only=True))
        if kind == "labels":
            value["theta"] = torch.zeros(4, 6)
        elif kind == "dtype":
            value["online"] = value["online"].double()
        else:
            value["online"].add_(1)
        torch.save(value, path)
    elif kind == "map":
        path = root / "evaluation_data.pt"
        evaluation = cast(EvaluationData, torch.load(path, weights_only=True))
        evaluation["metadata"]["generator_version"] = "different-dgp"
        torch.save(evaluation, path)
    elif kind in {"frozen", "q_payload"}:
        path = root / "frozen_modules.pt"
        payload = cast(dict[str, Any], torch.load(path, weights_only=True))
        if kind == "q_payload":
            payload["parameter_encoder"] = {}
        else:
            next(iter(payload["predictor"].values())).add_(1)
        torch.save(payload, path)
    else:
        manifest = _read_json(root / "manifest.json")
        manifest["all_six_evidence"]["test_r2"][5] = -0.01
        _write_json(root / "manifest.json", manifest)
    if kind != "hash":
        _refresh_manifest(root)
    with pytest.raises(
        ValueError, match="(hash|Training inputs|tensor|metadata|fingerprint|payload|six)"
    ):
        load_source_bundle(root)


def test_export_requires_bound_checkpoint_and_positive_all_six_evidence(tmp_path: Path) -> None:
    write_source_fixture(tmp_path)
    arguments = source_arguments(tmp_path)
    arguments["output_dir"] = tmp_path / "new_bundle"
    evidence = _read_json(arguments["source_evaluation"])
    evidence["checkpoint_sha256"] = "wrong-checkpoint"
    _write_json(arguments["source_evaluation"], evidence)
    with pytest.raises(ValueError, match="provenance"):
        export_source_bundle(**arguments)
    assert not arguments["output_dir"].exists()
