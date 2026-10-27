import copy
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_group_latent_runner import source_with_initialization
from test_multiquery_control import make_batch, make_model, objective_config
from torch import Tensor

from diagnostics.group_latent_control.run import Config as GroupConfig
from diagnostics.group_latent_control.run import run as group_run
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective, cached_forward
from diagnostics.mlp_acquisition_control.run import Config as MLPConfig
from diagnostics.mlp_acquisition_control.run import run as mlp_run
from diagnostics.mlp_oracle_code_control.model import initialize_mlp
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiquery_control.run import sha
from diagnostics.multiteacher_curriculum import run as runner
from diagnostics.multiteacher_curriculum.forward import (
    Heads,
    Stage,
    configure_phase,
    infer_codes,
    initialize_heads,
    phase_training_loss,
    pretext_loss,
)
from diagnostics.multiteacher_curriculum.run import Config, run
from diagnostics.multiteacher_curriculum.source import (
    Inputs,
    leave_one_out_targets,
    load_inputs,
    normalize_targets,
    prepare_targets,
    verify_sources,
)
from scjepa.experiments.storage import read_json, write_json
from scjepa.losses import RepresentationRegularizer
from scjepa.models.model import SCJEPA, cross_episode_parameters
from scjepa.training.rng import preserve_rng


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def teacher_roots(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path, Path]:
    """Produce genuine completed tiny label-free teachers with shared source ancestry."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        root = tmp_path_factory.mktemp("multiteacher")
        source = source_with_initialization(root / "source")
        warm, fresh, mlp = (root / name for name in ("warm", "fresh", "mlp"))
        common = GroupConfig(
            str(source),
            str(warm),
            seed=7,
            teacher_steps=2,
            amortization_steps=2,
            washout_steps=2,
            groups=2,
            bootstrap_samples=16,
            verification=True,
        )
        group_run(common)
        group_run(replace(common, output_dir=str(fresh), initialization="fresh_zero"))
        mlp_run(
            MLPConfig(
                str(source),
                str(mlp),
                seed=7,
                steps=3,
                groups=2,
                bootstrap_samples=16,
                verification=True,
            )
        )
        return source, warm, fresh, mlp
    finally:
        torch.set_num_threads(previous)


def inputs_from(roots: tuple[Path, Path, Path, Path]) -> Inputs:
    return load_inputs(*roots, 7, verification=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA RNG state")
def test_head_initialization_preserves_cpu_cuda_rng_and_existing_modules() -> None:
    device = torch.device("cuda")
    with preserve_rng(device):
        model = make_model()
        initialize_mlp(model, 7)
        model.to(device)
        modules = (
            model.online_encoder,
            model.target_encoder,
            model.parameter_encoder,
            model.predictor,
        )
        before = tuple(module_fingerprint(module) for module in modules)
        flags = copy.deepcopy(model.get_extra_state())
        torch.default_generator.manual_seed(123)
        torch.cuda.manual_seed(456)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        heads, _ = initialize_heads(model, 7)
        assert next(heads.parameters()).device == next(model.parameters()).device
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
        assert tuple(module_fingerprint(module) for module in modules) == before
        assert model.get_extra_state() == flags


def test_leave_one_out_targets_ignore_own_context_with_fixed_train_normalizers() -> None:
    generator = torch.Generator().manual_seed(971)
    raw = torch.randn(4, 10, 3, 6, generator=generator, dtype=torch.float64)
    raw = raw * torch.tensor([1.0, 2.0, 4.0])[None, None, :, None] + 3
    target = normalize_targets(raw.requires_grad_())
    torch.testing.assert_close(target.mean, raw.detach().mean((0, 1)), rtol=0, atol=0)
    torch.testing.assert_close(target.scale, raw.detach().std((0, 1), correction=0), rtol=0, atol=0)
    assert not any(
        value.requires_grad for value in (target.raw, target.mean, target.scale, target.targets)
    )
    expected = torch.stack(
        [
            (
                (raw.detach()[:, [i for i in range(10) if i != view]] - target.mean) / target.scale
            ).mean(1)
            for view in range(10)
        ],
        1,
    )
    torch.testing.assert_close(target.targets, expected, rtol=1e-14, atol=1e-14)
    changed = raw.detach().clone()
    changed[:, 3] += 1000
    altered = leave_one_out_targets(changed, target.mean, target.scale)
    torch.testing.assert_close(altered[:, 3], target.targets[:, 3], rtol=0, atol=0)
    assert not torch.equal(altered[:, 2], target.targets[:, 2])
    assert target.receipt["fit_split"] == "train"


def test_pretext_matches_balanced_teacher_loss_original_sigreg_and_rng() -> None:
    model = make_model()
    initialize_mlp(model, 7)
    heads, _ = initialize_heads(model, 7)
    configure_phase("distill", model, heads)
    _, batch = make_batch(model)
    targets = torch.randn(2, 10, 3, 6, dtype=torch.float64)
    objective = AcquisitionObjective(objective_config())
    torch.default_generator.manual_seed(171)
    actual, terms = pretext_loss(model, heads, batch, targets, objective)
    actual_rng = torch.get_rng_state()
    torch.default_generator.manual_seed(171)
    codes = infer_codes(model, batch)
    cross_episode_parameters(codes, training=model.training)
    torch.randn(6, objective.projections, dtype=torch.float64)
    regularizer = RepresentationRegularizer("sigreg", objective.projections)(codes.transpose(0, 1))
    consistency = 2 * codes.var(1, correction=1).mean()
    errors = (heads(codes) - targets).square().mean((0, 1, 3))
    expected = errors.mean() + 0.4 * regularizer + consistency
    assert torch.equal(actual_rng, torch.get_rng_state())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for index, error in enumerate(errors, 1):
        torch.testing.assert_close(terms[f"teacher_{index}_mse"], error, rtol=0, atol=0)
    parameters = tuple(
        p for module in (model, heads) for p in module.parameters() if p.requires_grad
    )
    actual_gradients = torch.autograd.grad(actual, parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, parameters)
    for first, second in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(first, second, rtol=1e-14, atol=1e-16)
    with pytest.raises(ValueError, match="detached"):
        pretext_loss(model, heads, batch, targets.requires_grad_(), objective)


@pytest.mark.parametrize("stage", ["distill", "predictor", "washout"])
def test_phase_optimizer_and_gradients_update_only_authorized_modules(stage: Stage) -> None:
    model = make_model()
    initialize_mlp(model, 7)
    heads, _ = initialize_heads(model, 7)
    active_heads = None if stage == "washout" else heads
    parameters = configure_phase(stage, model, active_heads)
    _, batch = make_batch(model)
    targets = torch.randn(2, 10, 3, 6, dtype=torch.float64) if stage == "distill" else None
    modules = {
        "online": model.online_encoder,
        "target": model.target_encoder,
        "encoder": model.parameter_encoder,
        "predictor": model.predictor,
        "heads": heads,
    }
    before = {name: copy.deepcopy(module.state_dict()) for name, module in modules.items()}
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    loss, _ = phase_training_loss(
        stage, model, active_heads, batch, targets, AcquisitionObjective(objective_config())
    )
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    expected = {
        "distill": {"encoder", "heads"},
        "predictor": {"predictor"},
        "washout": {"encoder", "predictor"},
    }[stage]
    for name, module in modules.items():
        changed = any(
            not torch.equal(value, before[name][key]) for key, value in module.state_dict().items()
        )
        assert changed == (name in expected)
        if name not in expected:
            assert all(parameter.grad is None for parameter in module.parameters())
    assert all(value.grad is None for value in batch.values())
    with pytest.raises(ValueError, match="only frozen|exclude physical"):
        phase_training_loss(
            stage,
            model,
            active_heads,
            {**batch, "theta": torch.zeros(2, 6)},
            targets,
            AcquisitionObjective(objective_config()),
        )


def test_predictor_phase_keeps_attached_rollout_and_matches_full_objective_gradient() -> None:
    model = make_model()
    initialize_mlp(model, 7)
    configure_phase("predictor", model, None)
    _, batch = make_batch(model)
    objective = AcquisitionObjective(objective_config())
    torch.default_generator.manual_seed(123)
    loss, terms = phase_training_loss("predictor", model, None, batch, None, objective)
    rng = torch.get_rng_state()
    torch.default_generator.manual_seed(123)
    output = cached_forward(model, batch)
    full, _ = objective(output)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(loss, terms["prediction"], rtol=0, atol=0)
    params = tuple(model.predictor.parameters())
    gradients = torch.autograd.grad(loss, params)
    expected = torch.autograd.grad(full, params)
    for first, second in zip(gradients, expected, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_sources_authenticate_fresh_student_and_train_only_target_generation(
    teacher_roots: tuple[Path, Path, Path, Path],
) -> None:
    inputs = inputs_from(teacher_roots)
    initial = cast(
        dict[str, Any], torch.load(teacher_roots[0] / "initial_seed_7.pt", weights_only=False)
    )
    torch.testing.assert_close(inputs.model.state_dict(), initial["model"], rtol=0, atol=0)
    assert "adaptive" in inputs.teacher_provenance["selection_scope"]
    assert all(not p.requires_grad for teacher in inputs.teachers for p in teacher.parameters())
    assert all(
        record["label_free_ancestry_verified"] for record in inputs.teacher_provenance["teachers"]
    )

    class NoEvaluation(dict[str, Any]):
        def __getitem__(self, _key: str) -> Any:  # noqa: ANN401
            raise AssertionError("Teacher-target training accessed evaluation data")

    inputs = replace(inputs, bundle=replace(inputs.bundle, evaluation=NoEvaluation()))
    before = [module_fingerprint(teacher) for teacher in inputs.teachers]
    progress: list[tuple[int, int]] = []
    targets = prepare_targets(
        inputs,
        device=torch.device("cpu"),
        batch_size=2,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert progress == [(2, 4), (4, 4)]
    assert targets.receipt["train_inputs_sha256"] == inputs.bundle.hashes["train_inputs.pt"]
    train = inputs.bundle.train
    with torch.no_grad():
        expected = torch.stack(
            [
                teacher(
                    train["online"][:, :, :10].flatten(0, 1),
                    train["actions"][:, :, :9].flatten(0, 1),
                ).reshape(4, 10, 6)
                for teacher in inputs.teachers
            ],
            2,
        )
    torch.testing.assert_close(targets.raw, expected, rtol=1e-6, atol=1e-7)
    assert before == [module_fingerprint(teacher) for teacher in inputs.teachers]
    verify_sources(inputs)
    with torch.no_grad():
        next(inputs.teachers[0].parameters()).add_(0.01)
    with pytest.raises(ValueError, match="teacher encoder changed"):
        verify_sources(inputs)


def test_source_cache_tamper_is_rejected(teacher_roots: tuple[Path, Path, Path, Path]) -> None:
    inputs = inputs_from(teacher_roots)
    path = teacher_roots[0] / "train_inputs.pt"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"tampered")
        with pytest.raises(ValueError, match="Immutable teacher or source"):
            verify_sources(inputs)
    finally:
        path.write_bytes(original)
    assert sha(path) == inputs.bundle.hashes["train_inputs.pt"]


def test_teacher_changed_state_chart_rejected_even_with_refreshed_file_hash(
    teacher_roots: tuple[Path, Path, Path, Path],
) -> None:
    root = teacher_roots[1]
    path, completion_path = root / "teacher_final.pt", root / "completed.json"
    original, original_completion = path.read_bytes(), completion_path.read_bytes()
    try:
        payload = cast(dict[str, Any], torch.load(path, weights_only=False))
        key = next(key for key in payload["model"] if key.startswith("online_encoder."))
        payload["model"][key] += 0.01
        torch.save(payload, path)
        completion = cast(dict[str, Any], read_json(completion_path))
        completion["hashes"]["teacher_final.pt"] = sha(path)
        write_json(completion_path, completion)
        with pytest.raises(ValueError, match="frozen learned-state tensors differ"):
            inputs_from(teacher_roots)
    finally:
        path.write_bytes(original)
        completion_path.write_bytes(original_completion)


@pytest.mark.parametrize(
    ("interrupted_stage", "interrupt_at"), [("distill", 2), ("predictor", 1), ("washout", 1)]
)
def test_curriculum_resume_preserves_phases_optimizers_and_fixed_final_test(
    teacher_roots: tuple[Path, Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_stage: Stage,
    interrupt_at: int,
) -> None:
    source, primary_teacher, fresh_teacher, mlp_teacher = teacher_roots
    config = Config(
        str(source),
        str(primary_teacher),
        str(fresh_teacher),
        str(mlp_teacher),
        str(tmp_path / "full"),
        seed=7,
        distill_steps=2,
        predictor_steps=2,
        washout_steps=2,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    calls: list[bool] = []
    original_assessment = runner.assess_model

    def assessment(
        model: SCJEPA,
        data: EvaluationData,
        selection: DiagnosticSelection,
        *,
        bootstrap_samples: int,
        seed: int,
        final: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Tensor]]:
        calls.append(final)
        guarded = data if final else {key: value for key, value in data.items() if key != "test"}
        return original_assessment(
            model, guarded, selection, bootstrap_samples=bootstrap_samples, seed=seed, final=final
        )

    monkeypatch.setattr(runner, "assess_model", assessment)
    full = run(config)
    assert calls == [False] * 7 + [True]
    out = tmp_path / "full"
    checkpoints = {
        phase: cast(dict[str, Any], torch.load(out / f"{phase}_final.pt", weights_only=False))
        for phase in ("distill", "predictor", "washout")
    }
    initial = cast(dict[str, Any], torch.load(out / "initial_modules.pt", weights_only=False))
    for key, value in initial["predictor"].items():
        torch.testing.assert_close(
            checkpoints["distill"]["model"]["predictor." + key], value, rtol=0, atol=0
        )
    for key, value in checkpoints["distill"]["model"].items():
        if key.startswith("parameter_encoder."):
            torch.testing.assert_close(
                checkpoints["predictor"]["model"][key], value, rtol=0, atol=0
            )
    for phase, checkpoint in checkpoints.items():
        assert checkpoint["stage"] == phase
        assert all(int(state["step"]) == 2 for state in checkpoint["optimizer"]["state"].values())
    assert checkpoints["washout"]["heads"] is None
    assert checkpoints["distill"]["heads"] is not None
    assert "adaptive" in full["scope"] or "retrospectively" in full["scope"]
    original_loss = runner.phase_training_loss
    seen = 0

    def interrupted(
        stage: Stage,
        model: SCJEPA,
        heads: Heads | None,
        batch: TensorSplit,
        targets: Tensor | None,
        objective: AcquisitionObjective,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal seen
        assert set(batch) == {"online", "target", "actions"}
        assert (targets is not None) == (stage == "distill")
        if stage == interrupted_stage:
            seen += 1
            if seen == interrupt_at:
                raise RuntimeError("interrupted curriculum")
        return original_loss(stage, model, heads, batch, targets, objective)

    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    monkeypatch.setattr(runner, "phase_training_loss", interrupted)
    calls.clear()
    with pytest.raises(RuntimeError, match="interrupted curriculum"):
        run(resumed_config)
    assert not any(calls)
    assert not (tmp_path / "resumed/final_test.json").exists()
    monkeypatch.setattr(runner, "phase_training_loss", original_loss)
    resumed = run(resumed_config, resume=True)
    final = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    torch.testing.assert_close(final["model"], checkpoints["washout"]["model"], rtol=0, atol=0)
    torch.testing.assert_close(
        final["optimizer"], checkpoints["washout"]["optimizer"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        final["rng"]["torch"], checkpoints["washout"]["rng"]["torch"], rtol=0, atol=0
    )
    assert full["records"] == resumed["records"]
    assert full["final_test"] == resumed["final_test"]
    before_calls = len(calls)
    assert run(resumed_config, resume=True) == resumed
    assert len(calls) == before_calls
    target_path = tmp_path / "resumed/teacher_targets.pt"
    target_path.write_bytes(target_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="Completed curriculum artifacts"):
        run(resumed_config, resume=True)
