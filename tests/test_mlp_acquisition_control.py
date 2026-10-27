import copy
import math
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_table_settling_control import fixture_sources
from torch import Tensor

from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.group_latent_control.run import initialize_control
from diagnostics.joint_acquisition.baseline import DiagnosticSelection, build_matched_batch
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.mlp_acquisition_control import run as runner
from diagnostics.mlp_acquisition_control.evaluation import assess_model, measurement_metrics
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.mlp_acquisition_control.run import Config, fixed_hashes, run
from diagnostics.mlp_oracle_code_control.model import initialize_mlp
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit
from diagnostics.multiquery_control.run import sha
from diagnostics.oracle_code_control.evaluation import matched_batch
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_label_free_source_resume_and_final_test_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curriculum, _ = fixture_sources(tmp_path, fresh=True)
    source = Path(
        cast(dict[str, Any], read_json(curriculum / "protocol.json"))["config"]["source_dir"]
    )
    bundle = load_primary_source(source, 7, expected_step=2, expected_checkpoint_sha256=None)
    source_model, _ = build_source_model(bundle, torch.device("cpu"))
    _, _, hashes = initialize_control(source_model, bundle, "fresh_zero")
    frozen = fixed_hashes(source_model)
    initialize_mlp(source_model, 7)
    archived = cast(dict[str, Any], torch.load(source / "initial_seed_7.pt", weights_only=False))
    config = Config(
        str(source),
        str(tmp_path / "full"),
        seed=7,
        steps=3,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    calls: list[bool] = []
    original_assess = runner.assess_model

    def spy_assess(
        model: SCJEPA,
        data: EvaluationData,
        selection: DiagnosticSelection,
        *,
        bootstrap_samples: int,
        seed: int,
        final: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Tensor]]:
        calls.append(final)
        # The validation evaluator must function without any test slot at all.
        guarded = data if final else {key: value for key, value in data.items() if key != "test"}
        return original_assess(
            model, guarded, selection, bootstrap_samples=bootstrap_samples, seed=seed, final=final
        )

    monkeypatch.setattr(runner, "assess_model", spy_assess)
    full = run(config)
    assert calls == [False] * 4 + [True]
    checkpoint = cast(dict[str, Any], torch.load(tmp_path / "full/last.pt", weights_only=False))
    initial = cast(
        dict[str, Any], torch.load(tmp_path / "full/initial_modules.pt", weights_only=False)
    )
    for name, value in initial["parameter_encoder"].items():
        torch.testing.assert_close(
            value, archived["model"]["parameter_encoder." + name], rtol=0, atol=0
        )
    torch.testing.assert_close(
        initial["predictor"], source_model.predictor.state_dict(), rtol=0, atol=0
    )
    restored = copy.deepcopy(source_model)
    restored.load_state_dict(checkpoint["model"])
    assert fixed_hashes(restored) == frozen
    for prefix in ("predictor", "parameter_encoder"):
        assert any(
            not torch.equal(value, initial[prefix][name])
            for name, value in getattr(restored, prefix).state_dict().items()
        )
    assert len(full["final_test"]["retention"]["selected_test_group_mse_by_parameter"]) == len(
        bundle.evaluation["test"]["theta"]
    )
    assert len(full["final_test"]["response"]["group_indices"]) == 2
    assert "test/theta_6/retention_r2" in measurement_metrics(full["final_test"], final=True)
    original_loss = runner.training_loss
    count = 0

    def interrupt(
        model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal count
        count += 1
        assert set(batch) == {"online", "target", "actions"}
        if count == 2:
            raise RuntimeError("interrupt")
        return original_loss(model, batch, objective)

    monkeypatch.setattr(runner, "training_loss", interrupt)
    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    with pytest.raises(RuntimeError, match="interrupt"):
        run(resumed_config)
    monkeypatch.setattr(runner, "training_loss", original_loss)
    resumed = run(resumed_config, resume=True)
    resumed_checkpoint = cast(
        dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    )
    torch.testing.assert_close(checkpoint["model"], resumed_checkpoint["model"], rtol=0, atol=0)
    torch.testing.assert_close(
        checkpoint["optimizer"], resumed_checkpoint["optimizer"], rtol=0, atol=0
    )
    assert full["records"] == resumed["records"]
    assert full["final_test"] == resumed["final_test"]
    assert all(sha(source / name) == value for name, value in hashes.items())
    previous_calls = len(calls)
    assert run(resumed_config, resume=True) == resumed
    assert len(calls) == previous_calls
    with pytest.raises(ValueError, match="Resume settings"):
        run(replace(resumed_config, bootstrap_samples=17), resume=True)
    initial["parameter_encoder"][next(iter(initial["parameter_encoder"]))] += 0.01
    torch.save(initial, tmp_path / "full/initial_modules.pt")
    with pytest.raises(ValueError, match="encoder initialization weights changed"):
        run(config, resume=True)


def test_inferred_code_evaluation_matches_original_geometry_without_oracle_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curriculum, _ = fixture_sources(tmp_path, fresh=True)
    source = Path(
        cast(dict[str, Any], read_json(curriculum / "protocol.json"))["config"]["source_dir"]
    )
    bundle = load_primary_source(source, 7, expected_step=2, expected_checkpoint_sha256=None)
    model, _ = build_source_model(bundle, torch.device("cpu"))
    initialize_mlp(model, 7)
    model.double().eval()
    selection = DiagnosticSelection((0, 1), (0, 1))
    original = build_matched_batch(
        model, bundle.evaluation, selection, "validation"
    ).select_queries("heldout")
    actual, _, _ = matched_batch(model, bundle.evaluation, "validation", (0, 1))
    for name in (
        "donor_states",
        "donor_actions",
        "changed_donor_states",
        "history",
        "targets",
        "changed_targets",
    ):
        torch.testing.assert_close(
            getattr(original, name), getattr(actual, name), rtol=1e-12, atol=1e-12
        )

    def zero_encoder(states: Tensor, _actions: Tensor) -> Tensor:
        return states.new_zeros((len(states), 6))

    monkeypatch.setattr(model.parameter_encoder, "forward", zero_encoder)
    guarded = {name: value for name, value in bundle.evaluation.items() if name != "test"}
    result, _ = assess_model(model, guarded, selection, bootstrap_samples=16, seed=7)
    assert all(
        math.isclose(row["normalized_response_error"]["estimate"], 1.0, rel_tol=1e-12)
        for row in result["response"]["per_parameter"]
    )
