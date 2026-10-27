from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from test_encoder_gradient_source import refresh_hashes, write_source_fixture
from torch import Tensor, nn

from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.group_latent_control import run as runner
from diagnostics.group_latent_control.run import (
    Config,
    Initialization,
    Phase,
    TeacherObjectiveArm,
    initialize_control,
    run,
)
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.multiquery_control.data import TensorSplit, module_fingerprint
from diagnostics.multiquery_control.run import frozen_hashes, sha
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective
from scjepa.experiments.storage import write_json
from scjepa.models.model import SCJEPA
from scjepa.training.factory import build_model
from scjepa.training.rng import preserve_rng


def source_with_initialization(root: Path) -> Path:
    """Add an authenticated genuinely fresh E/P initialization to the tiny trained source."""
    source = write_source_fixture(root)
    bundle = load_primary_source(source, 7, expected_step=2, expected_checkpoint_sha256=None)
    trained, _ = build_source_model(bundle, torch.device("cpu"))
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(7)
        fresh = build_model(cast(DictConfig, OmegaConf.create(bundle.protocol["model_config"])))
    fresh.online_encoder.load_state_dict(trained.online_encoder.state_dict())
    fresh.target_encoder.load_state_dict(trained.target_encoder.state_dict())
    name = "initial_seed_7.pt"
    torch.save(
        {
            "model": fresh.state_dict(),
            "seed": 7,
            "source_sha256": bundle.protocol["source"]["sha256"],
        },
        source / name,
    )
    protocol = bundle.protocol
    protocol["initializations"]["7"] = {"path": name, "sha256": sha(source / name)}
    protocol["artifacts"][name] = sha(source / name)
    write_json(source / "protocol.json", protocol)
    checkpoint = bundle.checkpoint
    checkpoint["initial_sha256"] = sha(source / name)
    torch.save(checkpoint, source / "seed-7/prediction_1/last.pt")
    refresh_hashes(source)
    return source


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize(
    ("interrupted_phase", "interrupt_at"),
    [("teacher", 2), ("amortization", 1), ("amortization", 2)],
)
@pytest.mark.parametrize("initialization", ["warm", "fresh_zero"])
@pytest.mark.parametrize("teacher_arm", ["original", "without_parameter_sigreg"])
def test_staged_resume_matches_uninterrupted_and_preserves_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_phase: Phase,
    interrupt_at: int,
    initialization: Initialization,
    teacher_arm: TeacherObjectiveArm,
) -> None:
    source = source_with_initialization(tmp_path / "source")
    hashes = {str(path): sha(path) for path in source.rglob("*") if path.is_file()}
    config = Config(
        source_dir=str(source),
        output_dir=str(tmp_path / "complete"),
        seed=7,
        teacher_steps=2,
        amortization_steps=2,
        washout_steps=2,
        groups=2,
        bootstrap_samples=16,
        verification=True,
        initialization=initialization,
        teacher_objective_arm=teacher_arm,
    )
    complete = run(config)
    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    original = runner.training_loss
    calls = 0

    def interrupt(
        model: SCJEPA,
        batch: TensorSplit,
        indices: Tensor,
        phase: Phase,
        table: nn.Embedding | None,
        objective: PredictiveSupervisionObjective,
        *,
        teacher_objective_arm: TeacherObjectiveArm = "original",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal calls
        if phase == interrupted_phase:
            calls += 1
            if calls == interrupt_at:
                raise RuntimeError("Operational interruption")
        return original(
            model,
            batch,
            indices,
            phase,
            table,
            objective,
            teacher_objective_arm=teacher_objective_arm,
        )

    monkeypatch.setattr(runner, "training_loss", interrupt)
    with pytest.raises(RuntimeError, match="Operational interruption"):
        run(resumed_config)
    partial = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    expected_phase = "teacher" if interrupt_at == 1 else interrupted_phase
    assert partial["phase"] == expected_phase
    assert partial["phase_step"] == (2 if interrupt_at == 1 else 1)
    if interrupted_phase == "teacher":
        assert partial["latent_optimizer"]["state"]
    monkeypatch.setattr(runner, "training_loss", original)
    resumed = run(resumed_config, resume=True)
    full = cast(dict[str, Any], torch.load(tmp_path / "complete/last.pt", weights_only=False))
    final = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    torch.testing.assert_close(full["model"], final["model"], rtol=0, atol=0)
    torch.testing.assert_close(full["optimizer"], final["optimizer"], rtol=0, atol=0)
    assert final["table"] is None
    assert final["global_step"] == 6
    assert complete["records"] == resumed["records"]
    assert complete["final_test"] == resumed["final_test"]
    assert {str(path): sha(path) for path in source.rglob("*") if path.is_file()} == hashes
    assert run(resumed_config, resume=True) == resumed
    other: Initialization = "fresh_zero" if initialization == "warm" else "warm"
    with pytest.raises(ValueError, match="Resume source, settings or code changed"):
        run(replace(resumed_config, initialization=other), resume=True)
    other_arm: TeacherObjectiveArm = (
        "original" if teacher_arm == "without_parameter_sigreg" else "without_parameter_sigreg"
    )
    with pytest.raises(ValueError, match="Resume source, settings or code changed"):
        run(replace(resumed_config, teacher_objective_arm=other_arm), resume=True)


def test_fresh_initialization_copies_only_verified_learned_modules(tmp_path: Path) -> None:
    source = source_with_initialization(tmp_path / "source")
    before = {str(path): sha(path) for path in source.rglob("*") if path.is_file()}
    bundle = load_primary_source(source, 7, expected_step=2, expected_checkpoint_sha256=None)
    model, _ = build_source_model(bundle, torch.device("cpu"))
    states = frozen_hashes(model)
    old_encoder, old_predictor = (
        module_fingerprint(model.parameter_encoder),
        module_fingerprint(model.predictor),
    )
    table, evidence, hashes = initialize_control(model, bundle, "fresh_zero")
    payload = cast(dict[str, Any], torch.load(source / "initial_seed_7.pt", weights_only=False))
    expected, _ = build_source_model(bundle, torch.device("cpu"))
    expected.load_state_dict(payload["model"])
    assert frozen_hashes(model) == states
    assert model.get_extra_state() == expected.get_extra_state()
    for name, old in (("parameter_encoder", old_encoder), ("predictor", old_predictor)):
        actual = module_fingerprint(getattr(model, name))
        assert actual == module_fingerprint(getattr(expected, name))
        assert actual != old
        assert evidence[name + "_hash"] == actual
    assert table.weight.shape == (4, 6)
    assert table.sparse
    assert torch.count_nonzero(table.weight) == 0
    assert hashes["initial_seed_7.pt"] == sha(source / "initial_seed_7.pt")
    assert {str(path): sha(path) for path in source.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("change", ["bytes", "seed", "state", "flags"])
def test_fresh_initialization_rejects_mismatch(tmp_path: Path, change: str) -> None:
    source = source_with_initialization(tmp_path / "source")
    bundle = load_primary_source(source, 7, expected_step=2, expected_checkpoint_sha256=None)
    model, _ = build_source_model(bundle, torch.device("cpu"))
    path = source / "initial_seed_7.pt"
    payload = cast(dict[str, Any], torch.load(path, weights_only=False))
    if change in ("bytes", "seed"):
        payload["seed"] = 8
    elif change == "state":
        key = next(key for key in payload["model"] if key.startswith("online_encoder."))
        payload["model"][key] += 1
    else:
        payload["model"]["_extra_state"]["autoregressive"] = False
    torch.save(payload, path)
    if change != "bytes":
        bundle.protocol["initializations"]["7"]["sha256"] = sha(path)
        bundle.protocol["artifacts"][path.name] = sha(path)
    with pytest.raises(ValueError, match="Initialization"):
        initialize_control(model, bundle, "fresh_zero")


def test_scientific_budget_cannot_be_silently_changed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="30000/10000/10000"):
        Config(
            source_dir=str(tmp_path / "source"),
            output_dir=str(tmp_path / "out"),
            teacher_steps=10000,
        )
