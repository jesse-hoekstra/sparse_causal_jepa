from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from diagnostics.coupled_duffing.data import DataConfig, prepare
from diagnostics.coupled_duffing.evaluate import collect
from diagnostics.coupled_duffing.experiment import (
    SCHEMA,
    StageConfig,
    calibrate_endpoint,
    load_data,
    train,
    verify,
)
from diagnostics.coupled_duffing.model import build_model
from diagnostics.coupled_duffing.run import code_identity, objective_for
from diagnostics.shared.io import read_json, save_torch, sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import write_json


@pytest.mark.parametrize("phase", ["baseline", "full"])
def test_separate_continuation_matches_uninterrupted_training_and_preserves_source(
    tmp_path: Path,
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.set_num_threads(1)
    data, source = tmp_path / "data", tmp_path / "adaptation"
    manifest = prepare(
        data,
        DataConfig(
            train_groups=8,
            calibration_groups=8,
            validation_groups=8,
            views=2,
            context=6,
            forecast=4,
        ),
    )
    (data / "train-labels.pt").unlink()
    source.mkdir()
    model = build_model(0, 6, code_dim=4, parameter_encoder="temporal-cnn")
    objective = objective_for(
        "joint", 0.4, torch.device("cpu"), state_regularization="projected-aligned"
    )
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "phase": "adaptation",
        "seed": 0,
        "code_dim": 4,
        "parameter_encoder": "temporal-cnn",
        "state_regularization": "projected-aligned",
        "data_identity": manifest["identity"],
        "code_identity": code_identity(),
    }
    record["identity"] = canonical_fingerprint(record)
    save_torch(
        source / "last.pt",
        {
            "identity": record["identity"],
            "model": model.state_dict(),
            "objective": objective.state_dict(),
        },
    )
    _, inputs = load_data(data, torch.device("cpu"), forecast_steps=4)
    save_torch(source / "calibration.pt", collect(model, inputs["calibration"], 4))
    write_json(
        source / "completion.json",
        {
            **record,
            "hashes": {name: sha(source / name) for name in ("last.pt", "calibration.pt")},
        },
    )
    shared_calibration = tmp_path / "calibration.json"
    calibrate_endpoint(source, shared_calibration)
    implementation = code_identity()
    config = StageConfig(
        str(data),
        str(tmp_path / "uninterrupted"),
        phase,
        source_dir=str(source),
        source_code_identity=implementation,
        steps=4,
        batch_size=4,
        log_every=1,
        eval_every=2,
        device="cpu",
        wandb_mode="disabled",
        parameter_encoder="temporal-cnn",
        cache_device="cpu",
        state_regularization="projected-aligned",
        learning_rate=0.0001,
        gate_learning_rate=0.001,
        initial_gate_logit=10,
        forecast_steps=4,
        state_frames=8,
        calibration_file=str(shared_calibration),
    )
    train(config)
    short = replace(config, output_dir=str(tmp_path / "short"), steps=2)
    train(short)
    parent = Path(short.output_dir)
    parent_record = verify(parent)
    parent_checkpoint = cast(dict[str, Any], torch.load(parent / "last.pt", weights_only=False))
    continuation = replace(
        config,
        output_dir=str(tmp_path / "continued"),
        continuation_dir=str(parent),
        continuation_code_identity=implementation,
    )
    # Simulate an explicitly reviewed implementation identity change while retaining
    # the same numerical operators, source weights and original calibration record.
    monkeypatch.setattr(
        "diagnostics.coupled_duffing.experiment.code_identity",
        lambda: "compatible-continuation-test-implementation",
    )
    train(continuation)
    child = Path(continuation.output_dir)
    initial = cast(dict[str, Any], torch.load(child / "initial.pt", weights_only=False))
    final = cast(dict[str, Any], torch.load(child / "last.pt", weights_only=False))
    reference = cast(
        dict[str, Any], torch.load(Path(config.output_dir) / "last.pt", weights_only=False)
    )
    for category in ("model", "objective", "sampler", "torch_rng", "cuda_rng"):
        torch.testing.assert_close(initial[category], parent_checkpoint[category], rtol=0, atol=0)
        torch.testing.assert_close(final[category], reference[category], rtol=0, atol=0)
    for left, right in ((initial, parent_checkpoint), (final, reference)):
        torch.testing.assert_close(
            left["optimizer"]["state"], right["optimizer"]["state"], rtol=0, atol=0
        )
        assert left["optimizer"]["param_groups"] == right["optimizer"]["param_groups"]
        assert left["tau"] == right["tau"]
    assert initial["step"] == 2
    assert final["step"] == 4
    assert initial["identity"] != parent_checkpoint["identity"]
    assert verify(parent, expected_code_identity=implementation) == parent_record
    child_record = verify(child)
    assert child_record["continuation"]["identity"] == parent_record["identity"]
    assert child_record["continuation"]["checkpoint_sha256"] == parent_record["hashes"]["last.pt"]
    assert (child / "calibration.json").read_bytes() == (parent / "calibration.json").read_bytes()
    assert read_json(child / "config.json")["steps"] == 4

    # Resuming the separate continuation itself preserves its identity and state.
    (child / "last.pt").write_bytes((child / "initial.pt").read_bytes())
    train(continuation, resume=True)
    resumed = cast(dict[str, Any], torch.load(child / "last.pt", weights_only=False))
    for category in ("model", "objective", "sampler", "torch_rng", "cuda_rng"):
        torch.testing.assert_close(resumed[category], final[category], rtol=0, atol=0)
    assert resumed["identity"] == final["identity"]
    assert verify(parent, expected_code_identity=implementation) == parent_record
    assert not list(tmp_path.rglob("test*.pt"))

    with pytest.raises(ValueError, match="changed training settings"):
        train(replace(continuation, output_dir=str(tmp_path / "changed"), learning_rate=0.0002))
    with pytest.raises(ValueError, match="larger total"):
        train(replace(continuation, output_dir=str(tmp_path / "same-budget"), steps=2))
    with pytest.raises(ValueError, match="separate directory"):
        train(replace(continuation, output_dir=str(parent)))
    with pytest.raises(ValueError, match="different numerical implementation"):
        train(replace(continuation, continuation_code_identity="wrong"))
    with pytest.raises(ValueError, match="different numerical implementation"):
        train(replace(continuation, continuation_code_identity=None))
    original_calibration = read_json(shared_calibration)
    changed_calibration = {**original_calibration, "tau": original_calibration["tau"] * 2}
    changed_calibration["identity"] = canonical_fingerprint(
        {k: v for k, v in changed_calibration.items() if k != "identity"}
    )
    write_json(shared_calibration, changed_calibration)
    with pytest.raises(ValueError, match="changed the shared calibration"):
        train(replace(continuation, output_dir=str(tmp_path / "changed-tau")))
    write_json(shared_calibration, original_calibration)
    (parent / "last.pt").write_bytes((parent / "initial.pt").read_bytes())
    with pytest.raises(ValueError, match="changed endpoint artifact"):
        train(replace(continuation, output_dir=str(tmp_path / "tampered")))


def test_continuation_options_require_a_sparse_parent() -> None:
    with pytest.raises(ValueError, match="sparse stages"):
        StageConfig("data", "out", "joint", continuation_dir="parent")
    with pytest.raises(ValueError, match="requires continuation_dir"):
        StageConfig("data", "out", "full", source_dir="adaptation", continuation_code_identity="id")
