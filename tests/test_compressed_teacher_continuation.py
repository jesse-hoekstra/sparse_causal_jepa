from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_compressed_teacher_curriculum import compression_source as compression_source
from test_multiquery_control import make_batch, make_model, objective_config
from test_multiteacher_curriculum import teacher_roots as teacher_roots
from test_teacher_code_compression import parent_snapshot as parent_snapshot
from torch import Tensor

from diagnostics.compressed_teacher_continuation import run as runner
from diagnostics.compressed_teacher_continuation.run import Config, ObjectiveArm, run
from diagnostics.compressed_teacher_continuation.source import load_inputs, verify_sources
from diagnostics.compressed_teacher_curriculum.run import Config as CurriculumConfig
from diagnostics.compressed_teacher_curriculum.run import run as curriculum_run
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.mlp_acquisition_control.forward import training_loss as ordinary_training_loss
from diagnostics.mlp_acquisition_control.run import fixed_hashes
from diagnostics.mlp_oracle_code_control.model import initialize_mlp
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiteacher_curriculum.forward import Heads
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


@pytest.fixture(scope="module")
def student_endpoint(
    compression_source: Path,  # noqa: F811
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        output = tmp_path_factory.mktemp("continuation-source") / "student"
        curriculum_run(
            CurriculumConfig(
                str(compression_source),
                str(output),
                seed=7,
                distill_steps=2,
                predictor_steps=2,
                washout_steps=2,
                groups=2,
                bootstrap_samples=16,
                verification=True,
            )
        )
        return output
    finally:
        torch.set_num_threads(previous)


def test_continuation_arms_only_change_optimized_auxiliary_gradients_and_preserve_rng() -> None:
    model = make_model()
    initialize_mlp(model, 7)
    model.requires_grad_(False)
    model.parameter_encoder.requires_grad_(True)
    model.predictor.requires_grad_(True)
    params = (*model.parameter_encoder.parameters(), *model.predictor.parameters())
    _, batch = make_batch(model)
    objective = AcquisitionObjective(objective_config())
    torch.default_generator.manual_seed(717)
    ordinary, terms = ordinary_training_loss(model, batch, objective)
    expected_rng = torch.get_rng_state()
    full_gradients = torch.autograd.grad(ordinary, params, retain_graph=True)
    forecast_gradients = torch.autograd.grad(terms["prediction"], params)
    assert (
        sum(
            float((first - second).square().sum())
            for first, second in zip(full_gradients, forecast_gradients, strict=True)
        )
        > 1e-12
    )
    for arm_name, expected, gradients in (
        ("original", ordinary, full_gradients),
        ("forecast_only", terms["prediction"], forecast_gradients),
    ):
        arm = cast(ObjectiveArm, arm_name)
        torch.default_generator.manual_seed(717)
        actual, report = runner.training_loss(model, batch, objective, arm)
        observed = torch.autograd.grad(actual, params)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for first, second in zip(observed, gradients, strict=True):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert torch.equal(expected_rng, torch.get_rng_state())
        for name in ("prediction", "parameter_regularizer", "parameter_consistency"):
            torch.testing.assert_close(report[name], terms[name], rtol=0, atol=0)
        with pytest.raises(ValueError, match="exclude physical labels"):
            runner.training_loss(model, {**batch, "theta": torch.zeros(2, 6)}, objective, arm)


def test_source_restores_completed_weights_and_preserves_immutable_modules(
    student_endpoint: Path,
) -> None:
    inputs = load_inputs(student_endpoint, verification=True)
    saved = cast(dict[str, Any], torch.load(student_endpoint / "last.pt", weights_only=False))
    torch.testing.assert_close(inputs.model.state_dict(), saved["model"], rtol=0, atol=0)
    assert inputs.provenance["source_step"] == 6
    assert all(not parameter.requires_grad for parameter in inputs.heads.parameters())
    assert set(inputs.bundle.train) == {"online", "target", "actions"}
    with torch.no_grad():
        next(inputs.model.parameter_encoder.parameters()).add_(0.001)
        next(inputs.model.predictor.parameters()).add_(0.001)
    verify_sources(inputs)
    with torch.no_grad():
        next(inputs.heads.parameters()).add_(0.001)
    with pytest.raises(ValueError, match="[Ff]rozen|[Hh]ead"):
        verify_sources(inputs)
    inputs = load_inputs(student_endpoint, verification=True)
    path = student_endpoint / "last.pt"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"tampered")
        with pytest.raises(ValueError, match="[Ii]mmutable|[Aa]rtifact|[Ss]ource"):
            verify_sources(inputs)
    finally:
        path.write_bytes(original)


def test_output_cannot_be_created_inside_ancestor_source(student_endpoint: Path) -> None:
    inputs = load_inputs(student_endpoint, verification=True)
    output = inputs.bundle.source_dir / "unwanted-continuation"
    config = Config(
        str(student_endpoint),
        str(output),
        seed=7,
        steps=3,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    with pytest.raises(ValueError, match="[Ii]mmutable|[Ss]ource"):
        run(config)
    assert not output.exists()


@pytest.mark.parametrize("arm", ["original", "forecast_only"])
def test_paired_continuation_starts_same_global_schedule_and_resumes_exactly(
    student_endpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: ObjectiveArm,
) -> None:
    inputs = load_inputs(student_endpoint, verification=True)
    source_step = int(inputs.provenance["source_step"])
    source_model = {
        name: value.clone() if isinstance(value, Tensor) else value
        for name, value in inputs.model.state_dict().items()
    }
    frozen = fixed_hashes(inputs.model)
    heads_hash = module_fingerprint(inputs.heads)
    config = Config(
        str(student_endpoint),
        str(tmp_path / "full"),
        objective_arm=arm,
        seed=7,
        steps=3,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    original_loss = runner.training_loss
    batches = 0

    def captured_loss(
        model: SCJEPA,
        batch: TensorSplit,
        objective: AcquisitionObjective,
        objective_arm: ObjectiveArm,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal batches
        global_step = source_step + batches + 1
        if batches == 0:
            torch.testing.assert_close(model.state_dict(), source_model, rtol=0, atol=0)
        epoch, position = divmod(
            global_step - 1, inputs.bundle.train_groups // inputs.bundle.batch_size
        )
        selected = list(
            GlobalBatchSampler(inputs.bundle.train_groups, inputs.bundle.batch_size, 7, epoch)
        )[position]
        for name, value in batch.items():
            torch.testing.assert_close(value, inputs.bundle.train[name][selected], rtol=0, atol=0)
        expected_rng = torch.Generator().manual_seed(7 * 100000 + global_step).get_state()
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert set(batch) == {"online", "target", "actions"}
        assert all(parameter.requires_grad for parameter in model.parameter_encoder.parameters())
        assert all(parameter.requires_grad for parameter in model.predictor.parameters())
        assert fixed_hashes(model) == frozen
        batches += 1
        return original_loss(model, batch, objective, objective_arm)

    original_assessment = runner.assess_model
    calls: list[bool] = []

    def assessment(
        model: SCJEPA,
        data: EvaluationData,
        selection: DiagnosticSelection,
        *,
        heads: Heads,
        bootstrap_samples: int,
        seed: int,
        final: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Tensor]]:
        calls.append(final)
        assert fixed_hashes(model) == frozen
        assert module_fingerprint(heads) == heads_hash
        assert all(not parameter.requires_grad for parameter in heads.parameters())
        guarded = data if final else {name: value for name, value in data.items() if name != "test"}
        return original_assessment(
            model,
            guarded,
            selection,
            heads=heads,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            final=final,
        )

    monkeypatch.setattr(runner, "training_loss", captured_loss)
    monkeypatch.setattr(runner, "assess_model", assessment)
    full = run(config)
    assert batches == 3
    assert calls == [False] * 4 + [True]
    checkpoint = cast(dict[str, Any], torch.load(tmp_path / "full/last.pt", weights_only=False))
    assert checkpoint["step"] == full["step"] == 9
    assert checkpoint["updates"] == full["updates"] == 3
    assert [record["step"] for record in full["records"]] == [6, 7, 8, 9]
    assert all(int(state["step"]) == 3 for state in checkpoint["optimizer"]["state"].values())
    for prefix in ("parameter_encoder.", "predictor."):
        assert any(
            not torch.equal(value, source_model[name])
            for name, value in checkpoint["model"].items()
            if name.startswith(prefix)
        )
    assert len(full["final_test"]["retention"]["selected_test_group_mse_by_parameter"]) == len(
        inputs.bundle.evaluation["test"]["theta"]
    )
    assert len(full["final_test"]["expanded_retention"]["test_group_mse_by_parameter"]) == len(
        inputs.bundle.evaluation["test"]["theta"]
    )
    updates = 0

    def interrupted_loss(
        model: SCJEPA,
        batch: TensorSplit,
        objective: AcquisitionObjective,
        objective_arm: ObjectiveArm,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal updates
        updates += 1
        if updates == 2:
            raise RuntimeError("interrupted continuation")
        return original_loss(model, batch, objective, objective_arm)

    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    monkeypatch.setattr(runner, "training_loss", interrupted_loss)
    calls.clear()
    with pytest.raises(RuntimeError, match="interrupted continuation"):
        run(resumed_config)
    assert not any(calls)
    assert not (tmp_path / "resumed/final_test.json").exists()
    monkeypatch.setattr(runner, "training_loss", original_loss)
    resumed = run(resumed_config, resume=True)
    final = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    for key in ("model", "optimizer"):
        torch.testing.assert_close(checkpoint[key], final[key], rtol=0, atol=0)
    torch.testing.assert_close(checkpoint["rng"]["torch"], final["rng"]["torch"], rtol=0, atol=0)
    assert full["records"] == resumed["records"]
    assert full["final_test"] == resumed["final_test"]
    verify_sources(inputs)
    previous = len(calls)
    assert run(resumed_config, resume=True) == resumed
    assert len(calls) == previous
    other_arm = "forecast_only" if arm == "original" else "original"
    with pytest.raises(ValueError, match="Resume settings"):
        run(replace(resumed_config, objective_arm=other_arm), resume=True)
