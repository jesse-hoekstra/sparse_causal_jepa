from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_multiquery_control import make_batch, make_model, objective_config
from test_multiteacher_curriculum import teacher_roots as teacher_roots
from test_teacher_code_compression import parent_snapshot as parent_snapshot
from torch import Tensor

from diagnostics.compressed_teacher_curriculum import evaluation as evaluator
from diagnostics.compressed_teacher_curriculum import run as runner
from diagnostics.compressed_teacher_curriculum.evaluation import expanded_readouts
from diagnostics.compressed_teacher_curriculum.run import (
    Config,
    configure_phase,
    phase_training_loss,
    run,
)
from diagnostics.compressed_teacher_curriculum.source import (
    leave_one_out_codes,
    load_inputs,
    prepare_targets,
    verify_sources,
)
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.mlp_oracle_code_control.model import initialize_mlp
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiteacher_curriculum.forward import Heads, Stage, infer_codes
from diagnostics.teacher_code_compression.run import Config as CompressionConfig
from diagnostics.teacher_code_compression.run import run as compression_run
from scjepa.losses import RepresentationRegularizer
from scjepa.models.model import SCJEPA, cross_episode_parameters


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_distillation_has_no_trainable_heads_and_matches_direct_code_loss_and_rng() -> None:
    model = make_model()
    initialize_mlp(model, 7)
    parameters = configure_phase("distill", model)
    assert {id(value) for value in parameters} == {
        id(value) for value in model.parameter_encoder.parameters()
    }
    _, batch = make_batch(model)
    targets = torch.randn(2, 10, 6, dtype=torch.float64)
    objective = AcquisitionObjective(objective_config())
    torch.default_generator.manual_seed(171)
    actual, _ = phase_training_loss("distill", model, batch, targets, objective)
    actual_rng = torch.get_rng_state()
    torch.default_generator.manual_seed(171)
    codes = infer_codes(model, batch)
    cross_episode_parameters(codes, training=model.training)
    torch.randn(6, objective.projections, dtype=torch.float64)
    regularizer = RepresentationRegularizer("sigreg", objective.projections)(codes.transpose(0, 1))
    expected = (codes - targets).square().mean() + 0.4 * regularizer
    expected = expected + 2 * codes.var(1, correction=1).mean()
    assert torch.equal(actual_rng, torch.get_rng_state())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_gradients = torch.autograd.grad(actual, parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, parameters)
    for first, second in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(first, second, rtol=1e-14, atol=1e-16)
    assert all(value.grad is None for value in model.predictor.parameters())
    with pytest.raises(ValueError, match="only frozen online/target"):
        phase_training_loss("distill", model, {**batch, "theta": targets[:, 0]}, targets, objective)
    with pytest.raises(ValueError, match="detached"):
        phase_training_loss("distill", model, batch, targets.requires_grad_(), objective)


@pytest.mark.parametrize("stage", ["distill", "predictor", "washout"])
def test_phase_updates_only_declared_modules(stage: Stage) -> None:
    model = make_model()
    initialize_mlp(model, 7)
    parameters = configure_phase(stage, model)
    _, batch = make_batch(model)
    modules = (model.online_encoder, model.target_encoder, model.parameter_encoder, model.predictor)
    before = [module_fingerprint(module) for module in modules]
    targets = torch.randn(2, 10, 6, dtype=torch.float64) if stage == "distill" else None
    loss, _ = phase_training_loss(
        stage, model, batch, targets, AcquisitionObjective(objective_config())
    )
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    changed = [
        module_fingerprint(module) != initial
        for module, initial in zip(modules, before, strict=True)
    ]
    assert changed == [False, False, stage != "predictor", stage != "distill"]


@pytest.fixture(scope="module")
def compression_source(
    parent_snapshot: Path,  # noqa: F811
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Use genuinely trained tiny compression with authenticated teacher ancestry."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        root = tmp_path_factory.mktemp("compressed-curriculum") / "compression"
        compression_run(
            CompressionConfig(
                str(parent_snapshot), str(root), seed=7, steps=3, groups=2, verification=True
            )
        )
        return root
    finally:
        torch.set_num_threads(previous)


def test_compressed_targets_are_per_context_then_leave_one_out_with_no_new_normalization(
    compression_source: Path,
) -> None:
    inputs = load_inputs(compression_source, verification=True)

    # Poison every evaluation access: target construction has TRAIN tensors already.
    class NoEvaluation(dict[str, Any]):
        def __getitem__(self, key: str) -> Any:  # noqa: ANN401
            raise AssertionError(f"TRAIN target construction accessed {key}")

    inputs = replace(inputs, bundle=replace(inputs.bundle, evaluation=NoEvaluation()))
    targets = prepare_targets(inputs, device=torch.device("cpu"), batch_size=2)
    with torch.no_grad():
        raw = torch.cat(
            [
                inputs.compressor(inputs.compression.train[start : start + 2])
                for start in range(0, len(inputs.compression.train), 2)
            ]
        )
    torch.testing.assert_close(targets.raw, raw, rtol=0, atol=0)
    expected = torch.stack(
        [
            raw[:, [other for other in range(10) if other != view]].double().mean(1)
            for view in range(10)
        ],
        1,
    )
    expected = expected.to(raw)
    torch.testing.assert_close(targets.targets, expected, rtol=0, atol=0)
    averaged_inputs = torch.stack(
        [
            inputs.compression.train[:, [other for other in range(10) if other != view]].mean(1)
            for view in range(10)
        ],
        1,
    )
    with torch.no_grad():
        incorrect_order = inputs.compressor(averaged_inputs)
    assert not torch.allclose(targets.targets, incorrect_order, rtol=1e-5, atol=1e-8)
    changed = raw.clone()
    changed[:, 3] += 100
    altered = leave_one_out_codes(changed)
    torch.testing.assert_close(altered[:, 3], expected[:, 3], rtol=0, atol=0)
    assert not torch.equal(altered[:, 2], expected[:, 2])
    assert not targets.raw.requires_grad
    assert not targets.targets.requires_grad
    assert all(not value.requires_grad for value in inputs.compressor.parameters())
    assert all(not value.requires_grad for value in inputs.heads.parameters())
    verify_sources(inputs)


def test_compressor_and_fixed_decoder_mutations_are_rejected(compression_source: Path) -> None:
    inputs = load_inputs(compression_source, verification=True)
    with torch.no_grad():
        next(inputs.heads.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="Frozen compressed"):
        verify_sources(inputs)
    inputs = load_inputs(compression_source, verification=True)
    checkpoint = compression_source / "last.pt"
    original = checkpoint.read_bytes()
    try:
        checkpoint.write_bytes(original + b"tampered")
        with pytest.raises(ValueError, match="Immutable compressed"):
            verify_sources(inputs)
    finally:
        checkpoint.write_bytes(original)


def test_output_cannot_be_created_inside_immutable_ancestor_source(
    compression_source: Path,
) -> None:
    inputs = load_inputs(compression_source, verification=True)
    output = inputs.bundle.source_dir / "unwanted-new-run"
    config = Config(
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
    with pytest.raises(ValueError, match="immutable|Immutable"):
        run(config)
    assert not output.exists()


@pytest.mark.parametrize("final", [False, True])
def test_expanded_features_use_only_individual_student_q_and_select_by_validation(
    compression_source: Path,
    monkeypatch: pytest.MonkeyPatch,
    final: bool,
) -> None:
    inputs = load_inputs(compression_source, verification=True)
    generator = torch.Generator().manual_seed(192)
    names = ("calibration", "validation", "test") if final else ("calibration", "validation")
    codes = {
        name: torch.randn(4, 10, 6, generator=generator, dtype=torch.float64) for name in names
    }
    splits: EvaluationData = {
        name: {"theta": torch.randn(4, 6, generator=generator, dtype=torch.float64)}
        for name in names
    }

    # Every teacher path is poisoned. Expanded features must depend on studentq alone.
    def forbidden(*args: object, **kwargs: object) -> Tensor:
        del args, kwargs
        raise AssertionError("Expanded evaluation used an original teacher/compressor")

    for teacher in inputs.compression.parent.teachers:
        monkeypatch.setattr(teacher, "forward", forbidden)
    monkeypatch.setattr(inputs.compressor, "forward", forbidden)
    expected: list[dict[str, Tensor]] = [{name: value for name, value in codes.items()}]
    with torch.no_grad():
        heads = inputs.heads.double()
        outputs = {name: heads(value) for name, value in codes.items()}
    expected.extend(
        {name: value[:, :, index] for name, value in outputs.items()} for index in range(3)
    )
    calls = 0

    def readouts(features: dict[str, Tensor], labels: EvaluationData) -> dict[str, Any]:
        nonlocal calls
        index = calls
        calls += 1
        for name in names:
            torch.testing.assert_close(features[name], expected[index][name], rtol=0, atol=0)
            torch.testing.assert_close(labels[name]["theta"], splits[name]["theta"], rtol=0, atol=0)
            assert features[name].shape == (4, 10, 6)
        if not final:
            torch.testing.assert_close(features["test"], features["validation"], rtol=0, atol=0)
        # Head1 wins validation but loses test. Selection must still choose head1.
        validation = [0.1, 0.9, 0.3, 0.4][index]
        test = [0.99, -0.5, 0.8, 0.9][index] if final else validation
        errors = [[float(index)] * 6 for _ in range(4)]
        return {
            "selected_methods_per_parameter": ["linear"] * 6,
            "selected_test_r2": [test] * 6,
            "selected_test_group_mse_by_parameter": errors,
            "families": {
                "linear": {
                    "validation_r2": [validation] * 6,
                    "test_r2": [test] * 6,
                    "selected_hyperparameters_per_parameter": [{"ridge": index}] * 6,
                }
            },
        }

    monkeypatch.setattr(evaluator, "parameter_readouts", readouts)
    report = expanded_readouts(codes, splits, inputs.heads, final=final)
    assert calls == 4
    assert report["selected_feature_maps_per_parameter"] == ["fixed_head_1"] * 6
    assert report["selection_validation_r2"] == [0.9] * 6
    score = "test" if final else "validation"
    assert report[f"selected_{score}_r2"] == ([-0.5] * 6 if final else [0.9] * 6)
    assert report[f"{score}_group_mse_by_parameter"] == [[1.0] * 6 for _ in range(4)]
    if not final:
        assert "selected_test_r2" not in report


@pytest.mark.parametrize("interrupted_stage", ["predictor", "washout"])
def test_curriculum_resumes_at_phase_boundary_and_tests_only_fixed_endpoint(
    compression_source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_stage: Stage,
) -> None:
    config = Config(
        str(compression_source),
        str(tmp_path / "full"),
        seed=7,
        distill_steps=2,
        predictor_steps=2,
        washout_steps=2,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    original_assessment = runner.assess_model
    calls: list[bool] = []
    decoder_hashes: list[str] = []

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
        decoder_hashes.append(module_fingerprint(heads))
        assert all(not parameter.requires_grad for parameter in heads.parameters())
        guarded = data if final else {key: value for key, value in data.items() if key != "test"}
        return original_assessment(
            model,
            guarded,
            selection,
            heads=heads,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            final=final,
        )

    monkeypatch.setattr(runner, "assess_model", assessment)
    full = run(config)
    assert calls == [False] * 7 + [True]
    assert len(set(decoder_hashes)) == 1
    checkpoints = {
        stage: cast(
            dict[str, Any], torch.load(tmp_path / f"full/{stage}_final.pt", weights_only=False)
        )
        for stage in ("distill", "predictor", "washout")
    }
    initial = cast(
        dict[str, Any], torch.load(tmp_path / "full/initial_modules.pt", weights_only=False)
    )
    for key, value in initial["predictor"].items():
        torch.testing.assert_close(
            checkpoints["distill"]["model"]["predictor." + key], value, rtol=0, atol=0
        )
    for key, value in checkpoints["distill"]["model"].items():
        if key.startswith("parameter_encoder."):
            torch.testing.assert_close(
                checkpoints["predictor"]["model"][key], value, rtol=0, atol=0
            )
    for checkpoint in checkpoints.values():
        assert all(int(state["step"]) == 2 for state in checkpoint["optimizer"]["state"].values())
        assert not checkpoint.get("heads")
    original_loss = runner.phase_training_loss

    def interrupted(
        stage: Stage,
        model: SCJEPA,
        batch: TensorSplit,
        targets: Tensor | None,
        objective: AcquisitionObjective,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        assert set(batch) == {"online", "target", "actions"}
        assert (targets is not None) == (stage == "distill")
        if targets is not None:
            assert targets.shape == (2, 10, 6)
            assert not targets.requires_grad
        if stage == interrupted_stage:
            raise RuntimeError("interrupted compressed curriculum")
        return original_loss(stage, model, batch, targets, objective)

    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    monkeypatch.setattr(runner, "phase_training_loss", interrupted)
    calls.clear()
    with pytest.raises(RuntimeError, match="interrupted compressed curriculum"):
        run(resumed_config)
    assert not any(calls)
    assert not (tmp_path / "resumed/final_test.json").exists()
    monkeypatch.setattr(runner, "phase_training_loss", original_loss)
    resumed = run(resumed_config, resume=True)
    final = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    for key in ("model", "optimizer"):
        torch.testing.assert_close(final[key], checkpoints["washout"][key], rtol=0, atol=0)
    torch.testing.assert_close(
        final["rng"]["torch"], checkpoints["washout"]["rng"]["torch"], rtol=0, atol=0
    )
    assert full["records"] == resumed["records"]
    assert full["final_test"] == resumed["final_test"]
    count = len(calls)
    assert run(resumed_config, resume=True) == resumed
    assert len(calls) == count
