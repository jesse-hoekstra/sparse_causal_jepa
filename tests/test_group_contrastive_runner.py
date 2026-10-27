from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from test_group_latent_runner import source_with_initialization
from torch import Tensor

from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.group_contrastive_pretext import evaluation
from diagnostics.group_contrastive_pretext import run as runner
from diagnostics.group_contrastive_pretext.run import Config, fixed_hashes, run
from diagnostics.group_latent_control.run import initialize_control
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiquery_control.run import sha
from scjepa.models.model import SCJEPA
from scjepa.training.distributed import GlobalBatchSampler


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_fresh_encoder_only_exact_schedule_resume_and_fixed_final_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_with_initialization(tmp_path / "source")
    bundle = load_primary_source(source, 7, expected_step=2, expected_checkpoint_sha256=None)
    source_model, _ = build_source_model(bundle, torch.device("cpu"))
    _, _, hashes = initialize_control(source_model, bundle, "fresh_zero")
    frozen = fixed_hashes(source_model)
    fresh_e = module_fingerprint(source_model.parameter_encoder)
    config = Config(
        str(source), str(tmp_path / "full"), seed=7, steps=3, groups=2, verification=True
    )
    original_loss = runner.training_loss
    original_assess = runner.assess_model
    captures = 0
    calls: list[tuple[bool, bool]] = []

    def loss(model: SCJEPA, batch: TensorSplit) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal captures
        captures += 1
        assert set(batch) == {"online", "target", "actions"}
        assert fixed_hashes(model) == frozen
        if captures == 1:
            assert module_fingerprint(model.parameter_encoder) == fresh_e
        assert all(p.requires_grad for p in model.parameter_encoder.parameters())
        assert not any(p.requires_grad for p in model.predictor.parameters())
        epoch, position = divmod(captures - 1, bundle.train_groups // bundle.batch_size)
        ids = list(GlobalBatchSampler(bundle.train_groups, bundle.batch_size, 7, epoch))[position]
        for name, value in batch.items():
            torch.testing.assert_close(value, bundle.train[name][ids], rtol=0, atol=0)
        expected_rng = torch.Generator().manual_seed(7 * 100000 + captures).get_state()
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        return original_loss(model, batch)

    def assess(
        model: SCJEPA,
        data: EvaluationData,
        selection: DiagnosticSelection,
        *,
        full: bool = False,
        final: bool = False,
    ) -> dict[str, Any]:
        calls.append((full, final))
        assert fixed_hashes(model) == frozen
        if final:
            assert len(calls) >= 2
            assert calls[-2] == (True, False)
            assert (Path(config.output_dir) / "full_validation.json").exists()
        guarded = data if final else {key: value for key, value in data.items() if key != "test"}
        return original_assess(model, guarded, selection, full=full, final=final)

    monkeypatch.setattr(runner, "training_loss", loss)
    monkeypatch.setattr(runner, "assess_model", assess)
    full = run(config)
    assert calls == [(False, False)] * 4 + [(True, False), (True, True)]
    assert full["initial_encoder_sha256"] == fresh_e
    assert full["final_encoder_sha256"] != fresh_e
    assert (
        full["predictor_fit_gate_from_validation_only"]
        == full["full_validation"]["full_validation_predictor_fit_gate"]
    )
    assert full["records"][0]["groups"]["validation"] == 2
    assert full["full_validation"]["groups"]["validation"] == len(
        bundle.evaluation["validation"]["theta"]
    )
    assert full["final_test"]["groups"]["test"] == len(bundle.evaluation["test"]["theta"])
    checkpoint = cast(dict[str, Any], torch.load(tmp_path / "full/last.pt", weights_only=False))
    source_model.load_state_dict(checkpoint["model"], strict=True)
    assert fixed_hashes(source_model) == frozen
    assert all(int(value["step"]) == 3 for value in checkpoint["optimizer"]["state"].values())
    initial = cast(
        dict[str, Any], torch.load(tmp_path / "full/initial_modules.pt", weights_only=False)
    )
    archived = cast(dict[str, Any], torch.load(source / "initial_seed_7.pt", weights_only=False))
    for name, value in initial["parameter_encoder"].items():
        torch.testing.assert_close(
            value, archived["model"]["parameter_encoder." + name], rtol=0, atol=0
        )
    updates = 0

    def interrupt(model: SCJEPA, batch: TensorSplit) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal updates
        updates += 1
        if updates == 2:
            raise RuntimeError("planned contrastive interruption")
        return original_loss(model, batch)

    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    monkeypatch.setattr(runner, "training_loss", interrupt)
    monkeypatch.setattr(runner, "assess_model", original_assess)
    with pytest.raises(RuntimeError, match="planned contrastive"):
        run(resumed_config)
    assert not (tmp_path / "resumed/full_validation.json").exists()
    assert not (tmp_path / "resumed/final_test.json").exists()
    monkeypatch.setattr(runner, "training_loss", original_loss)
    resumed = run(resumed_config, resume=True)
    final = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    for key in ("model", "optimizer"):
        torch.testing.assert_close(checkpoint[key], final[key], rtol=0, atol=0)
    torch.testing.assert_close(checkpoint["rng"]["torch"], final["rng"]["torch"], rtol=0, atol=0)
    assert checkpoint["rng"]["python"] == final["rng"]["python"]
    assert checkpoint["rng"]["numpy"][0] == final["rng"]["numpy"][0]
    np.testing.assert_array_equal(checkpoint["rng"]["numpy"][1], final["rng"]["numpy"][1])
    assert checkpoint["rng"]["numpy"][2:] == final["rng"]["numpy"][2:]
    assert checkpoint["rng"]["cuda"] == final["rng"]["cuda"]
    assert full["records"] == resumed["records"]
    assert full["full_validation"] == resumed["full_validation"]
    assert full["final_test"] == resumed["final_test"]
    assert all(sha(source / name) == value for name, value in hashes.items())
    assert run(resumed_config, resume=True) == resumed
    with pytest.raises(ValueError, match="Resume settings"):
        run(replace(resumed_config, groups=3), resume=True)
    with pytest.raises(ValueError, match="separate"):
        Config(str(source), str(source / "bad"), seed=7, steps=3, groups=2, verification=True)
    result_path = tmp_path / "resumed/final_test.json"
    result_path.write_text(result_path.read_text() + " ")
    with pytest.raises(ValueError, match="Completed artifacts changed"):
        run(resumed_config, resume=True)


@pytest.mark.parametrize(
    ("validation_score", "test_score", "gate"), [(0.9, -1.0, True), (0.1, 0.99, False)]
)
def test_followup_gate_uses_full_validation_even_when_test_disagrees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_score: float,
    test_score: float,
    gate: bool,
) -> None:
    source = source_with_initialization(tmp_path / "source")
    bundle = load_primary_source(source, 7, expected_step=2, expected_checkpoint_sha256=None)
    model, _ = build_source_model(bundle, torch.device("cpu"))

    def readouts(codes: dict[str, Tensor], splits: EvaluationData) -> dict[str, Any]:
        assert len(codes["calibration"]) == len(bundle.evaluation["calibration"]["theta"])
        assert len(codes["validation"]) == len(bundle.evaluation["validation"]["theta"])
        return {
            "families": {
                "linear": {
                    "validation_r2": [validation_score] * 6,
                    "selected_hyperparameters_per_parameter": [{}] * 6,
                }
            },
            "selected_methods_per_parameter": ["linear"] * 6,
            "selected_test_r2": [test_score] * 6,
            "selected_test_group_mse_by_parameter": [[0.0] * 6] * len(splits["test"]["theta"]),
        }

    monkeypatch.setattr(evaluation, "parameter_readouts", readouts)
    selection = DiagnosticSelection((0, 1), (0, 1))
    full = evaluation.assess_model(
        model, {k: v for k, v in bundle.evaluation.items() if k != "test"}, selection, full=True
    )
    final = evaluation.assess_model(model, bundle.evaluation, selection, full=True, final=True)
    assert full["full_validation_predictor_fit_gate"] is gate
    assert final["full_validation_predictor_fit_gate"] is gate
    assert final["retention"]["selected_test_r2"] == [test_score] * 6
