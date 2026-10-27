from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_multiquery_control import objective_config
from test_multiteacher_curriculum import teacher_roots as teacher_roots
from torch import Tensor

from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiteacher_curriculum.forward import Heads
from diagnostics.multiteacher_curriculum.run import Config as CurriculumConfig
from diagnostics.multiteacher_curriculum.run import run as curriculum_run
from diagnostics.teacher_code_compression import run as runner
from diagnostics.teacher_code_compression.data import load_reference, verify_reference
from diagnostics.teacher_code_compression.evaluation import Evaluation, assess, prepare_evaluation
from diagnostics.teacher_code_compression.model import (
    CompressionEncoder,
    initialize_modules,
    training_loss,
)
from diagnostics.teacher_code_compression.run import Config, run
from scjepa.losses import RepresentationRegularizer
from scjepa.models.model import cross_episode_parameters


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def parent_snapshot(
    teacher_roots: tuple[Path, Path, Path, Path],  # noqa: F811
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Keep only the immutable phase-boundary files, as for an active parent run."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        source, primary, fresh, mlp = teacher_roots
        out = tmp_path_factory.mktemp("compression-parent") / "results"
        curriculum_run(
            CurriculumConfig(
                str(source),
                str(primary),
                str(fresh),
                str(mlp),
                str(out),
                seed=7,
                distill_steps=2,
                predictor_steps=2,
                washout_steps=2,
                groups=2,
                bootstrap_samples=16,
                verification=True,
            )
        )
        for name in (
            "last.pt",
            "validation.json",
            "completed.json",
            "results.json",
            "final_test.json",
            "final_test_measurements.pt",
        ):
            (out / name).unlink()
        return out
    finally:
        torch.set_num_threads(previous)


def test_compression_is_per_context_and_preserves_original_loss_and_rng() -> None:
    encoder, heads, _ = initialize_modules(7, dtype=torch.float64)
    generator = torch.Generator().manual_seed(171)
    features = torch.randn(2, 10, 18, generator=generator, dtype=torch.float64)
    targets = torch.randn(2, 10, 3, 6, generator=generator, dtype=torch.float64)
    original_codes = encoder(features)
    altered = features.clone()
    altered[:, 3] += 10
    changed_codes = encoder(altered)
    torch.testing.assert_close(original_codes[:, 2], changed_codes[:, 2], rtol=0, atol=0)
    assert not torch.equal(original_codes[:, 3], changed_codes[:, 3])
    objective = AcquisitionObjective(objective_config())
    torch.default_generator.manual_seed(992)
    actual, terms = training_loss(encoder, heads, features, targets, objective)
    actual_rng = torch.get_rng_state()
    torch.default_generator.manual_seed(992)
    codes = encoder(features)
    cross_episode_parameters(codes, training=encoder.training)
    torch.randn(6, objective.projections, dtype=torch.float64)
    regularizer = RepresentationRegularizer("sigreg", objective.projections)(codes.transpose(0, 1))
    consistency = 2 * codes.var(1, correction=1).mean()
    errors = (heads(codes) - targets).square().mean((0, 1, 3))
    expected = errors.mean() + 0.4 * regularizer + consistency
    assert torch.equal(actual_rng, torch.get_rng_state())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for index, error in enumerate(errors, 1):
        torch.testing.assert_close(terms[f"teacher_{index}_mse"], error, rtol=0, atol=0)
    parameters = tuple(encoder.parameters()) + tuple(heads.parameters())
    actual_gradients = torch.autograd.grad(actual, parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, parameters)
    for first, second in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(first, second, rtol=1e-14, atol=1e-16)
    with pytest.raises(ValueError, match="detached finite individual-context"):
        training_loss(encoder, heads, features.requires_grad_(), targets, objective)


def test_snapshot_training_normalizers_and_exact_heads_ignore_mutable_parent_outputs(
    parent_snapshot: Path,
) -> None:
    inputs = load_reference(parent_snapshot, verification=True)
    raw = cast(
        dict[str, Any], torch.load(parent_snapshot / "teacher_targets.pt", weights_only=False)
    )
    torch.testing.assert_close(
        inputs.train, ((raw["raw"] - raw["mean"]) / raw["scale"]).flatten(-2), rtol=0, atol=0
    )
    torch.testing.assert_close(inputs.targets, raw["targets"], rtol=0, atol=0)
    assert inputs.train.shape == (4, 10, 18)
    assert not inputs.reference["parent_completion_asserted"]
    assert inputs.reference["teacher_target_receipt"]["fit_split"] == "train"
    encoder, heads, _ = initialize_modules(7)
    torch.testing.assert_close(heads.state_dict(), inputs.heads_state, rtol=0, atol=0)
    before = [module_fingerprint(teacher) for teacher in inputs.parent.teachers]
    optimizer = torch.optim.Adam([*encoder.parameters(), *heads.parameters()], lr=0.001)
    objective = AcquisitionObjective(objective_config())
    loss, _ = training_loss(encoder, heads, inputs.train[:2], inputs.targets[:2], objective)
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    assert before == [module_fingerprint(teacher) for teacher in inputs.parent.teachers]
    assert all(
        parameter.grad is None
        for teacher in inputs.parent.teachers
        for parameter in teacher.parameters()
    )
    assert inputs.train.grad is None
    assert inputs.targets.grad is None
    verify_reference(inputs)


def test_validation_uses_stored_train_normalization_and_never_accesses_test(
    parent_snapshot: Path,
) -> None:
    inputs = load_reference(parent_snapshot, verification=True)

    class NoTest(dict[str, Any]):
        def __getitem__(self, key: str) -> Any:  # noqa: ANN401
            if key == "test":
                raise AssertionError("Compression diagnostic accessed independent test")
            return super().__getitem__(key)

    inputs = replace(
        inputs,
        parent=replace(
            inputs.parent,
            bundle=replace(
                inputs.parent.bundle, evaluation=NoTest(inputs.parent.bundle.evaluation)
            ),
        ),
    )
    data = prepare_evaluation(inputs, torch.device("cpu"))
    assert set(data) == {"calibration", "validation"}
    split = inputs.parent.bundle.evaluation["validation"]
    with torch.no_grad():
        raw = torch.stack(
            [
                teacher(
                    split["states_online"][:, :, :10].flatten(0, 1),
                    split["actions"][:, :, :9].flatten(0, 1),
                ).reshape(4, 10, 6)
                for teacher in inputs.parent.teachers
            ],
            2,
        )
    torch.testing.assert_close(
        data["validation"]["features"],
        ((raw - inputs.mean) / inputs.scale).flatten(-2),
        rtol=0,
        atol=0,
    )
    encoder, heads, _ = initialize_modules(7)
    report = assess(encoder, heads, data, groups=None)
    assert report["calibration_groups"] == report["validation_groups"] == 4
    assert report["contexts_per_group"] == 10
    assert "selected_validation_r2" in report
    assert "selected_test_r2" not in report


def test_snapshot_wrong_phase_counter_and_cache_mutation_are_rejected(
    parent_snapshot: Path,
) -> None:
    inputs = load_reference(parent_snapshot, verification=True)
    path = parent_snapshot / "distill_final.pt"
    original = path.read_bytes()
    try:
        payload = cast(dict[str, Any], torch.load(path, weights_only=False))
        payload["step"] += 1
        torch.save(payload, path)
        with pytest.raises(ValueError, match="Distillation boundary"):
            load_reference(parent_snapshot, verification=True)
        with pytest.raises(ValueError, match="Immutable distillation reference"):
            verify_reference(inputs)
    finally:
        path.write_bytes(original)


def test_initialization_preserves_cpu_rng() -> None:
    torch.default_generator.manual_seed(456)
    before = torch.get_rng_state()
    initialize_modules(7)
    assert torch.equal(before, torch.get_rng_state())


def test_runner_exact_resume_and_fixed_full_validation_only(
    parent_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(
        str(parent_snapshot), str(tmp_path / "full"), seed=7, steps=3, groups=2, verification=True
    )
    original_assess = runner.assess
    calls: list[int | None] = []

    def assessment(
        encoder: CompressionEncoder,
        heads: Heads,
        data: Evaluation,
        *,
        groups: int | None,
    ) -> dict[str, Any]:
        calls.append(groups)
        assert set(data) == {"calibration", "validation"}
        return original_assess(encoder, heads, data, groups=groups)

    monkeypatch.setattr(runner, "assess", assessment)
    full = run(config)
    assert calls == [2, 2, 2, 2, None]
    checkpoint = cast(dict[str, Any], torch.load(tmp_path / "full/last.pt", weights_only=False))
    assert full["final_full_validation"]["calibration_groups"] == 4
    assert full["final_full_validation"]["validation_groups"] == 4
    assert "final_test" not in full
    original_loss = runner.training_loss
    updates = 0

    def interrupted(
        encoder: CompressionEncoder,
        heads: Heads,
        features: Tensor,
        targets: Tensor,
        objective: AcquisitionObjective,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal updates
        updates += 1
        assert features.shape == (2, 10, 18)
        assert targets.shape == (2, 10, 3, 6)
        if updates == 2:
            raise RuntimeError("interrupted compression")
        return original_loss(encoder, heads, features, targets, objective)

    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    monkeypatch.setattr(runner, "training_loss", interrupted)
    calls.clear()
    with pytest.raises(RuntimeError, match="interrupted compression"):
        run(resumed_config)
    assert calls == [2, 2]
    monkeypatch.setattr(runner, "training_loss", original_loss)
    resumed = run(resumed_config, resume=True)
    final = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    for name in ("encoder", "heads", "optimizer"):
        torch.testing.assert_close(checkpoint[name], final[name], rtol=0, atol=0)
    torch.testing.assert_close(checkpoint["rng"]["torch"], final["rng"]["torch"], rtol=0, atol=0)
    assert full["records"] == resumed["records"]
    assert full["final_full_validation"] == resumed["final_full_validation"]
    before = len(calls)
    assert run(resumed_config, resume=True) == resumed
    assert len(calls) == before
    with pytest.raises(ValueError, match="must match parent distillation"):
        run(replace(resumed_config, seed=8), resume=True)
    initial_path = tmp_path / "resumed/initial_modules.pt"
    initial = cast(dict[str, Any], torch.load(initial_path, weights_only=False))
    initial["heads"][next(iter(initial["heads"]))] += 0.01
    torch.save(initial, initial_path)
    with pytest.raises(ValueError, match="Persisted initial modules changed"):
        run(resumed_config, resume=True)
