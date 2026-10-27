import copy
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_encoder_gradient_source import refresh_hashes, write_source_fixture
from test_group_latent_runner import source_with_initialization
from test_multiquery_control import make_batch, make_model, objective_config
from torch import Tensor, nn

from diagnostics.group_latent_control.forward import clip_teacher_gradients, group_teacher_forward
from diagnostics.group_latent_control.run import Config as CurriculumConfig
from diagnostics.group_latent_control.run import run as curriculum_run
from diagnostics.multiquery_control.data import TensorSplit
from diagnostics.multiquery_control.run import sha
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective
from diagnostics.table_settling_control import run as runner
from diagnostics.table_settling_control.evaluation import new_episodes, paired_change
from diagnostics.table_settling_control.run import (
    Config,
    ObjectiveArm,
    TableSettlingObjective,
    run,
    train_update,
)
from diagnostics.table_settling_control.source import (
    fixed_encoder_fingerprint,
    load_inputs,
    model_fingerprint,
    validated_predictor_optimizer,
)
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.experiments.storage import read_json, write_json
from scjepa.models.model import SCJEPA


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def fixture_sources(root: Path, *, fresh: bool = False) -> tuple[Path, Path]:
    source = (source_with_initialization if fresh else write_source_fixture)(root / "primary")
    evaluation = cast(dict[str, Any], torch.load(source / "evaluation_data.pt", weights_only=False))
    evaluation["metadata"]["training_dataset_fingerprint"] = "tiny-exact-cache-fixture"
    torch.save(evaluation, source / "evaluation_data.pt")
    refresh_hashes(source)
    curriculum = root / "curriculum"
    curriculum_run(
        CurriculumConfig(
            source_dir=str(source),
            output_dir=str(curriculum),
            seed=7,
            teacher_steps=2,
            amortization_steps=2,
            washout_steps=2,
            groups=2,
            bootstrap_samples=16,
            verification=True,
            initialization="fresh_zero" if fresh else "warm",
        )
    )
    protocol = read_json(curriculum / "protocol.json")
    theta = evaluation["calibration"]["theta"]
    audit = root / "audit"
    audit.mkdir()
    evidence = {
        "source_hashes": protocol["source_hashes"],
        "training_dataset_fingerprint": "tiny-exact-cache-fixture",
    }
    identity = canonical_fingerprint(evidence)
    write_json(audit / "protocol.json", {"identity": identity, **evidence})
    write_json(
        audit / "results.json",
        {
            "identity": identity,
            "reconstruction": {
                "groups_verified": 4,
                "dataset_fingerprint": "tiny-exact-cache-fixture",
                "all_actions_equal": True,
                "all_parameter_payloads_reproduce_observations": True,
                "state_tolerance": 2e-5,
                "state_max_absolute_error": {"online": 0.0, "target": 0.0},
                "theta_tensor_fingerprint": tensor_fingerprint(theta),
            },
        },
    )
    torch.save({"identity": identity, "theta_evaluation_only": theta}, audit / "audit_inputs.pt")
    write_json(
        audit / "completed.json",
        {
            "identity": identity,
            "hashes": {
                name: sha(audit / name)
                for name in ("protocol.json", "results.json", "audit_inputs.pt")
            },
        },
    )
    return curriculum, audit


def test_only_sampled_table_rows_change_and_labels_cannot_enter_training() -> None:
    model = make_model().requires_grad_(False)
    _, batch = make_batch(model)
    table = nn.Embedding.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        torch.zeros(4, 6, dtype=torch.float64), freeze=False, sparse=True
    )  # pyright: ignore[reportUnknownMemberType]
    optimizer = torch.optim.SparseAdam(table.parameters(), lr=0.01)
    model_before = model_fingerprint(model)
    objective = PredictiveSupervisionObjective(objective_config(), 1.0)
    before = table.weight.detach().clone()
    terms = train_update(model, table, optimizer, batch, torch.tensor([0, 2]), objective)
    assert terms["gradient_norm"] > 0
    assert not torch.equal(table.weight[[0, 2]], before[[0, 2]])
    assert torch.equal(table.weight[[1, 3]], before[[1, 3]])
    assert model_fingerprint(model) == model_before
    assert all(parameter.grad is None for parameter in model.parameters())
    with pytest.raises(ValueError, match="label-free"):
        train_update(
            model,
            table,
            optimizer,
            {**batch, "theta": torch.ones(2, 6)},
            torch.tensor([0, 2]),
            objective,
        )


@pytest.mark.parametrize("arm", ["original", "without_parameter_sigreg"])
def test_matched_arm_preserves_rng_and_sparse_moments_update(arm: ObjectiveArm) -> None:
    model = make_model().requires_grad_(False)
    _, batch = make_batch(model)
    table = nn.Embedding.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        torch.randn(4, 6, dtype=torch.float64), freeze=False, sparse=True
    )
    indices = torch.tensor([0, 2])
    optimizer = torch.optim.SparseAdam(table.parameters(), lr=0.01)
    ordinary = PredictiveSupervisionObjective(objective_config(), 1.0)
    train_update(model, table, optimizer, batch, indices, ordinary)
    before = table.weight.detach().clone()
    model_before = model_fingerprint(model)
    expected_table = copy.deepcopy(table)
    expected_optimizer = torch.optim.SparseAdam(expected_table.parameters(), lr=0.01)
    expected_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    expected_optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(837)  # pyright: ignore[reportUnknownMemberType]
    original, components = ordinary(group_teacher_forward(model, batch, expected_table(indices)))
    expected = original
    if arm == "without_parameter_sigreg":
        expected = (
            components["prediction"]
            + objective_config().logit_weight * components["logit"]
            + objective_config().parameter_consistency_weight * components["parameter_consistency"]
        )
    expected.backward()  # pyright: ignore[reportUnknownMemberType]
    clip_teacher_gradients((), expected_table, 1.0)
    expected_optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(837)  # pyright: ignore[reportUnknownMemberType]
    objective = (
        TableSettlingObjective(objective_config())
        if arm == "original"
        else TableSettlingObjective(objective_config(), arm)
    )
    actual = train_update(model, table, optimizer, batch, indices, objective)
    assert actual["original_objective"] == float(original.detach())
    assert actual["optimized_objective"] == float(expected.detach())
    assert actual["effective_parameter_regularizer_weight"] == (0.4 if arm == "original" else 0.0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(table.weight, expected_table.weight, rtol=0, atol=0)
    torch.testing.assert_close(table.weight.grad, expected_table.weight.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        optimizer.state_dict(), expected_optimizer.state_dict(), rtol=0, atol=0
    )
    assert optimizer.state_dict()["state"][0]["step"] == 2
    assert torch.equal(table.weight[[1, 3]], before[[1, 3]])
    assert model_fingerprint(model) == model_before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_objective_arm_is_validated() -> None:
    with pytest.raises(ValueError, match="objective arm"):
        Config("/curriculum", "/audit", "/output", objective_arm=cast(ObjectiveArm, "other"))
    with pytest.raises(ValueError, match="objective arm"):
        TableSettlingObjective(objective_config(), cast(ObjectiveArm, "other"))


def test_new_episodes_are_reproducible_evaluation_only_and_independent() -> None:
    model = make_model().float().eval().requires_grad_(False)
    theta = torch.zeros(4, 6, dtype=torch.float64, requires_grad=True)
    before = model_fingerprint(model)
    state = torch.get_rng_state().clone()
    first = new_episodes(model, theta, torch.eye(6, dtype=torch.float64), 91)
    second = new_episodes(model, theta, torch.eye(6, dtype=torch.float64), 91)
    different = new_episodes(model, theta, torch.eye(6, dtype=torch.float64), 92)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first["actions"], different["actions"])
    assert all(not value.requires_grad for value in first.values())
    assert set(first) == {"online", "target", "actions"}
    assert torch.equal(state, torch.get_rng_state())
    assert model_fingerprint(model) == before
    assert theta.grad is None


def test_paired_bootstrap_uses_system_pairs() -> None:
    before = torch.arange(8.0).reshape(4, 2)
    result = paired_change(before, before - 0.5, samples=64, seed=3)
    assert result["mean"] == [-0.5, -0.5]
    assert result["ci95"] == [[-0.5, -0.5], [-0.5, -0.5]]


@pytest.mark.parametrize("train_predictor", [False, True])
@pytest.mark.parametrize("arm", ["original", "without_parameter_sigreg"])
def test_resume_freeze_source_integrity_and_evaluation_separation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_predictor: bool,
    arm: ObjectiveArm,
) -> None:
    curriculum, audit = fixture_sources(tmp_path)
    inputs = load_inputs(curriculum, audit, verification=True, groups=4, selection_seed=13)
    teacher = cast(dict[str, Any], torch.load(curriculum / "teacher_final.pt", weights_only=False))
    original_state = teacher["latent_optimizer"]["state"][0]
    restored_state = inputs.latent_optimizer["state"][0]
    assert restored_state["step"] == original_state["step"] == 2
    for name in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            restored_state[name], original_state[name][inputs.group_indices], rtol=0, atol=0
        )
    source_hashes = dict(inputs.file_hashes)
    config = Config(
        curriculum_dir=str(curriculum),
        audit_dir=str(audit),
        output_dir=str(tmp_path / "full"),
        seed=13,
        steps=3,
        groups=4,
        bootstrap_samples=16,
        verification=True,
        train_predictor=train_predictor,
        objective_arm=arm,
    )

    # Tiny partitions have one validation/test group; forecast evaluation remains real.
    def tiny_readout(_table: nn.Embedding, _theta: Tensor) -> dict[str, Any]:
        return {"selected_test_r2": [0.0] * 6}

    monkeypatch.setattr(runner, "readout", tiny_readout)
    full = run(config)
    original = runner.train_update
    calls = 0

    def interrupted(
        model: SCJEPA,
        table: nn.Embedding,
        optimizer: torch.optim.SparseAdam,
        batch: TensorSplit,
        indices: Tensor,
        objective: PredictiveSupervisionObjective,
        *,
        predictor_optimizer: torch.optim.Adam | None = None,
    ) -> dict[str, float]:
        nonlocal calls
        assert set(batch) == {"online", "target", "actions"}
        # Optimization must select original cached TRAIN episodes, never fresh evaluation episodes.
        selected = inputs.group_indices[indices.cpu()]
        torch.testing.assert_close(
            batch["actions"].cpu(), inputs.bundle.train["actions"][selected], rtol=0, atol=0
        )
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupt")
        return original(
            model,
            table,
            optimizer,
            batch,
            indices,
            objective,
            predictor_optimizer=predictor_optimizer,
        )

    monkeypatch.setattr(runner, "train_update", interrupted)
    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    with pytest.raises(RuntimeError, match="interrupt"):
        run(resumed_config)
    monkeypatch.setattr(runner, "train_update", original)
    resumed = run(resumed_config, resume=True)
    assert full["records"] == resumed["records"]
    first = cast(dict[str, Any], torch.load(tmp_path / "full/last.pt", weights_only=False))
    last = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    torch.testing.assert_close(first["table"], last["table"], rtol=0, atol=0)
    torch.testing.assert_close(first["optimizer"], last["optimizer"], rtol=0, atol=0)
    assert last["optimizer"]["state"][0]["step"] == 5
    if train_predictor:
        torch.testing.assert_close(first["predictor"], last["predictor"], rtol=0, atol=0)
        torch.testing.assert_close(
            first["predictor_optimizer"], last["predictor_optimizer"], rtol=0, atol=0
        )
        assert all(
            float(state["step"]) == 5 for state in last["predictor_optimizer"]["state"].values()
        )
        assert any(
            not torch.equal(value, teacher["model"]["predictor." + name])
            for name, value in last["predictor"].items()
        )
    else:
        assert last["predictor"] is None
        assert last["predictor_optimizer"] is None
    assert all(sha(Path(name)) == expected for name, expected in source_hashes.items())
    assert run(resumed_config, resume=True) == resumed
    other_arm: ObjectiveArm = "without_parameter_sigreg" if arm == "original" else "original"
    omission_config = replace(
        config,
        output_dir=str(tmp_path / "omission"),
        objective_arm=other_arm,
    )
    omission = run(omission_config)
    assert omission["identity"] != full["identity"]
    assert omission["objective_arm"] == other_arm
    assert omission["records"][0] == full["records"][0]
    first_evaluation = torch.load(tmp_path / "full/evaluation_only.pt", weights_only=False)
    omission_evaluation = torch.load(tmp_path / "omission/evaluation_only.pt", weights_only=False)
    for key in ("group_indices", "new_episodes", "theta_evaluation_only"):
        torch.testing.assert_close(first_evaluation[key], omission_evaluation[key], rtol=0, atol=0)
    assert read_json(tmp_path / "omission/protocol.json")[
        "effective_parameter_regularizer_weight"
    ] == (0 if other_arm == "without_parameter_sigreg" else 0.4)
    with pytest.raises(ValueError, match="Resume settings"):
        run(replace(resumed_config, objective_arm=other_arm), resume=True)
    with pytest.raises(ValueError, match="Resume settings"):
        run(replace(resumed_config, seed=14), resume=True)
    with pytest.raises(ValueError, match="Resume settings"):
        run(replace(resumed_config, train_predictor=not train_predictor), resume=True)
    eval_path = tmp_path / "resumed/evaluation_only.pt"
    payload = cast(dict[str, Any], torch.load(eval_path, weights_only=False))
    payload["new_episodes"]["actions"] += 1
    torch.save(payload, eval_path)
    with pytest.raises(ValueError, match="evaluation-only episodes changed"):
        run(resumed_config, resume=True)


def test_audit_mismatch_rejects_before_optimization(tmp_path: Path) -> None:
    curriculum, audit = fixture_sources(tmp_path)
    inputs = load_inputs(curriculum, audit, verification=True, groups=4)
    payload = cast(dict[str, Any], torch.load(audit / "audit_inputs.pt", weights_only=False))
    payload["theta_evaluation_only"] = payload["theta_evaluation_only"] + 0.1
    torch.save(payload, audit / "audit_inputs.pt")
    with pytest.raises(ValueError, match="Source artifact changed"):
        load_inputs(curriculum, audit, verification=True, groups=4)
    completion = cast(dict[str, Any], read_json(audit / "completed.json"))
    completion["hashes"]["audit_inputs.pt"] = sha(audit / "audit_inputs.pt")
    write_json(audit / "completed.json", completion)
    with pytest.raises(ValueError, match="evaluation labels changed"):
        load_inputs(curriculum, audit, verification=True, groups=4)
    assert torch.equal(
        inputs.theta_evaluation_only, inputs.bundle.evaluation["calibration"]["theta"]
    )


def test_fresh_curriculum_and_audit_authenticate_extra_initialization(tmp_path: Path) -> None:
    curriculum, audit = fixture_sources(tmp_path, fresh=True)
    inputs = load_inputs(curriculum, audit, verification=True, groups=4)
    assert any(path.endswith("initial_seed_7.pt") for path in inputs.file_hashes)
    assert inputs.curriculum["config"]["initialization"] == "fresh_zero"


@pytest.mark.parametrize("arm", ["original", "without_parameter_sigreg"])
def test_joint_update_matches_restored_adam_and_one_global_clip(arm: ObjectiveArm) -> None:
    model = make_model().requires_grad_(False)
    model.predictor.requires_grad_(True)
    _, batch = make_batch(model)
    batch["target"] = batch["target"] * 100
    fixed = fixed_encoder_fingerprint(model)
    table = nn.Embedding.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        torch.randn(4, 6, dtype=torch.float64), freeze=False, sparse=True
    )
    sparse = torch.optim.SparseAdam(table.parameters(), lr=0.01)
    dense = torch.optim.Adam(model.predictor.parameters(), lr=0.001)
    objective = TableSettlingObjective(objective_config(), arm)
    indices = torch.tensor([0, 2])
    train_update(model, table, sparse, batch, indices, objective, predictor_optimizer=dense)
    before = copy.deepcopy(model.predictor.state_dict())
    expected_model, expected_table = copy.deepcopy(model), copy.deepcopy(table)
    expected_dense = torch.optim.Adam(expected_model.predictor.parameters(), lr=0.001)
    expected_sparse = torch.optim.SparseAdam(expected_table.parameters(), lr=0.01)
    expected_dense.load_state_dict(copy.deepcopy(dense.state_dict()))
    expected_sparse.load_state_dict(copy.deepcopy(sparse.state_dict()))
    expected_dense.zero_grad(set_to_none=True)
    expected_sparse.zero_grad(set_to_none=True)
    torch.manual_seed(719)  # pyright: ignore[reportUnknownMemberType]
    loss, _ = objective(group_teacher_forward(expected_model, batch, expected_table(indices)))
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert expected_table.weight.grad is not None
    expected_table.weight.grad = expected_table.weight.grad.coalesce()
    gradients = [expected_table.weight.grad.values()]
    gradients.extend(p.grad for p in expected_model.predictor.parameters() if p.grad is not None)
    norm = sum(float(gradient.double().square().sum()) for gradient in gradients) ** 0.5
    assert norm > 1
    with torch.no_grad():
        for gradient in gradients:
            gradient.mul_(min(1.0, 1.0 / (norm + 1e-6)))
    expected_sparse.step()  # pyright: ignore[reportUnknownMemberType]
    expected_dense.step()  # pyright: ignore[reportUnknownMemberType]
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(719)  # pyright: ignore[reportUnknownMemberType]
    result = train_update(
        model, table, sparse, batch, indices, objective, predictor_optimizer=dense
    )
    assert abs(result["gradient_norm"] - norm) < 1e-10
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(model.state_dict(), expected_model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(table.weight, expected_table.weight, rtol=0, atol=0)
    torch.testing.assert_close(dense.state_dict(), expected_dense.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(sparse.state_dict(), expected_sparse.state_dict(), rtol=0, atol=0)
    assert fixed_encoder_fingerprint(model) == fixed
    assert any(
        not torch.equal(value, before[name]) for name, value in model.predictor.state_dict().items()
    )
    validated_predictor_optimizer(model, dense.state_dict(), 2)
    wrong = copy.deepcopy(dense.state_dict())
    wrong["param_groups"][0]["params"] = list(reversed(wrong["param_groups"][0]["params"]))
    with pytest.raises(ValueError, match="parameter order"):
        validated_predictor_optimizer(model, wrong, 2)
    wrong = copy.deepcopy(dense.state_dict())
    wrong["state"][0]["step"] += 1
    with pytest.raises(ValueError, match="step differs"):
        validated_predictor_optimizer(model, wrong, 2)
