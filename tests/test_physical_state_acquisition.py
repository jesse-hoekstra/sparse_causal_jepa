import copy
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_group_latent_runner import source_with_initialization
from test_multiquery_control import make_model
from torch import Tensor

from diagnostics.encoder_gradient_control.source import SourceBundle
from diagnostics.joint_acquisition.baseline import DiagnosticSelection, response_report
from diagnostics.multiquery_control.data import (
    EvaluationData,
    TensorSplit,
    build_cache,
    module_fingerprint,
)
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.multiquery_control.run import sha
from diagnostics.physical_state_acquisition import run as runner
from diagnostics.physical_state_acquisition.data import (
    PhysicalIdentityEncoder,
    build_physical_cache,
)
from diagnostics.physical_state_acquisition.evaluation import (
    assess_model,
    build_physical_matched_batch,
)
from diagnostics.physical_state_acquisition.run import Config, run
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.experiments.storage import write_json
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import SpartanOutput


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def source_fixture() -> tuple[SCJEPA, TensorSplit, EvaluationData, dict[str, Any]]:
    """Use exact original DGP trajectories behind nonidentity learned state encoders."""
    model = make_model().float().eval()
    observation = VectorObservationMap(dimension=6, seed=1729)
    config: dict[str, Any] = {
        "family": "simplified_gaussian",
        "root_seed": 19,
        "split": "train",
        "num_episodes": 8,
        "episodes_per_parameter": 10,
        "trajectory_length": 12,
        "dtype": "float32",
    }
    train, evaluation = build_cache(
        model,
        observation,
        config,
        train_groups=8,
        calibration_groups=4,
        validation_groups=4,
        test_groups=4,
        batch_size=2,
    )
    return model, train, evaluation, config


def physical_fixture() -> tuple[SCJEPA, TensorSplit, EvaluationData]:
    model, train, evaluation, config = source_fixture()
    physical, assessed, _ = build_physical_cache(model, train, evaluation, config, batch_size=2)
    model.online_encoder = PhysicalIdentityEncoder()
    model.target_encoder = PhysicalIdentityEncoder()
    model.online_encoder.requires_grad_(False).eval()
    model.target_encoder.requires_grad_(False).eval()
    return model, physical, assessed


def test_physical_cache_authenticates_every_group_and_preserves_label_boundary() -> None:
    model, train, evaluation, config = source_fixture()
    before_train, before_evaluation = copy.deepcopy(train), copy.deepcopy(evaluation)
    physical, assessed, receipt = build_physical_cache(
        model, train, evaluation, config, batch_size=2
    )
    assert set(physical) == {"online", "target", "actions"}
    assert receipt["training_groups_verified"] == 8
    assert receipt["all_addressed_actions_equal"]
    assert receipt["training_parameter_labels_absent"]
    assert all(not value.requires_grad for value in physical.values())
    torch.testing.assert_close(physical["online"], physical["target"], rtol=0, atol=0)
    torch.testing.assert_close(physical["actions"], train["actions"], rtol=0, atol=0)
    assert not torch.equal(physical["online"], train["online"])
    for name in ("online", "target"):
        key = f"state_{name}_sha256"
        assert assessed["metadata"][key] == module_fingerprint(PhysicalIdentityEncoder())
        assert assessed["metadata"]["source_" + key] == evaluation["metadata"][key]
    for split in ("calibration", "validation", "test"):
        torch.testing.assert_close(
            assessed[split]["states_online"],
            evaluation[split]["true_states"].float(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            assessed[split]["theta"], evaluation[split]["theta"], rtol=0, atol=0
        )
    torch.testing.assert_close(train, before_train, rtol=0, atol=0)
    for split in ("calibration", "validation", "test"):
        torch.testing.assert_close(evaluation[split], before_evaluation[split], rtol=0, atol=0)
    assert evaluation["metadata"].keys() == before_evaluation["metadata"].keys()
    for key, value in evaluation["metadata"].items():
        if isinstance(value, Tensor):
            torch.testing.assert_close(value, before_evaluation["metadata"][key], rtol=0, atol=0)
        else:
            assert value == before_evaluation["metadata"][key]
    changed = copy.deepcopy(train)
    changed["actions"][-1, -1, -1, -1] += 0.01
    with pytest.raises(ValueError, match="training actions differ"):
        build_physical_cache(model, changed, evaluation, config, batch_size=2)
    changed = copy.deepcopy(train)
    changed["target"][-1, -1, -1, -1] += 0.01
    with pytest.raises(ValueError, match="target states differ"):
        build_physical_cache(model, changed, evaluation, config, batch_size=2)
    with pytest.raises(ValueError, match="label-free"):
        build_physical_cache(model, {**train, "theta": torch.zeros(8, 6)}, evaluation, config)


def test_physical_matched_targets_equal_exact_dgp_without_observation_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, data = physical_fixture()
    model.double().eval()

    def forbidden(_self: VectorObservationMap, _states: Tensor) -> Tensor:
        raise AssertionError("Physical evaluation must not transform raw states into observations")

    monkeypatch.setattr(VectorObservationMap, "forward", forbidden)
    selection = DiagnosticSelection(calibration_indices=(0, 1), validation_indices=(0, 1))
    matched = build_physical_matched_batch(model, data, selection, "validation")
    theta = data["validation"]["theta"][:2]
    exact = dgp.simulate(matched.history[..., 0, :], theta[:, None, None], matched.query_actions)[
        ..., 1:, :
    ]
    torch.testing.assert_close(exact, matched.targets, rtol=0, atol=0)
    changed_theta = theta[:, None] + torch.diag_embed(matched.finite_steps)
    changed = dgp.simulate(
        matched.history[:, :, None, :, 0, :],
        changed_theta[:, None, :, None],
        matched.query_actions[:, :, None],
    )[..., 1:, :]
    torch.testing.assert_close(changed, matched.changed_targets, rtol=0, atol=0)
    expected_donors = data["validation"]["true_states"][:2, :1, :10]
    # Changing only the evaluation batch shape can alter float64 matmul roundoff.
    torch.testing.assert_close(matched.donor_states, expected_donors, rtol=0, atol=1e-12)
    report, _ = response_report(exact, changed, matched, bootstrap_samples=64, seed=7)
    assert report["ordinary_mse_by_horizon"] == [0.0, 0.0]
    for factor in report["per_parameter"]:
        assert factor["normalized_response_error"]["estimate"] == 0
        assert factor["predictive_benefit"]["estimate"] > 0


def test_raw_physical_training_keeps_both_rollout_steps_and_encoder_attached() -> None:
    model, train, _ = physical_fixture()
    model.train()
    outputs: list[SpartanOutput] = []

    def capture(_module: torch.nn.Module, _args: tuple[object, ...], output: SpartanOutput) -> None:
        outputs.append(output)
        output.prediction.retain_grad()

    handle = model.predictor.register_forward_hook(capture)
    try:
        batch = {key: value[:2] for key, value in train.items()}
        output = cached_multiquery_forward(model, batch, queries=1)
        assert len(outputs) == 2
        # Backpropagate only second-step prediction: first-step gradient proves attached recursion.
        output.predictions[:, :, 1].square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
    finally:
        handle.remove()
    assert outputs[0].prediction.grad is not None
    assert float(outputs[0].prediction.grad.abs().sum()) > 0
    assert (
        sum(
            float(p.grad.abs().sum())
            for p in model.parameter_encoder.parameters()
            if p.grad is not None
        )
        > 0
    )
    assert (
        sum(float(p.grad.abs().sum()) for p in model.predictor.parameters() if p.grad is not None)
        > 0
    )
    assert not list(model.online_encoder.parameters())
    assert not list(model.target_encoder.parameters())
    assert all(value.grad is None for value in train.values())


def test_validation_assessment_cannot_read_independent_test() -> None:
    model, _, data = physical_fixture()
    model.eval()

    class NoTest(dict[str, Any]):
        def __getitem__(self, key: str) -> Any:  # noqa: ANN401
            if key == "test":
                raise AssertionError("Validation assessment accessed the independent test split")
            return super().__getitem__(key)

    selection = DiagnosticSelection(
        calibration_indices=(0, 1, 2, 3), validation_indices=(0, 1, 2, 3)
    )
    report, _ = assess_model(model, NoTest(data), selection, bootstrap_samples=64, seed=7)
    assert "selected_validation_r2" in report["retention"]
    final, _ = assess_model(model, data, selection, bootstrap_samples=64, seed=7, final=True)
    assert "selected_test_r2" in final["retention"]
    assert "independent-test" in final["response"]["interpretation"]


def test_runner_uses_fresh_weights_and_resumes_exactly_before_final_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise real optimization/resume; only separately tested reconstruction is stubbed."""
    source = source_with_initialization(tmp_path / "source")
    initial = cast(dict[str, Any], torch.load(source / "initial_seed_7.pt", weights_only=False))
    source_config = tmp_path / "source_config.json"
    write_json(source_config, {"fixture": "reconstruction independently tested above"})
    hashes = {path: sha(path) for path in source.rglob("*") if path.is_file()}

    def configuration(
        _bundle: SourceBundle, _path: Path | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return {}, {"path": str(source_config), "sha256": sha(source_config)}

    def reconstruction(
        model: SCJEPA,
        train: TensorSplit,
        evaluation: EvaluationData,
        _config: dict[str, Any],
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[TensorSplit, EvaluationData, dict[str, Any]]:
        for name in ("parameter_encoder", "predictor"):
            for key, value in getattr(model, name).state_dict().items():
                torch.testing.assert_close(value, initial["model"][f"{name}.{key}"], rtol=0, atol=0)
        # This source fixture's training systems are its calibration systems.
        physical = evaluation["calibration"]["true_states"].to(train["online"])
        physical_train = {
            "online": physical.clone(),
            "target": physical.clone(),
            "actions": train["actions"].clone(),
        }
        assessed = copy.deepcopy(evaluation)
        assessed["metadata"]["state_coordinates"] = "physical_identity"
        for name in ("online", "target"):
            assessed["metadata"][f"state_{name}_sha256"] = module_fingerprint(
                PhysicalIdentityEncoder().to(train["online"])
            )
        for split in ("calibration", "validation", "test"):
            for key in ("states_online", "states_target"):
                assessed[split][key] = assessed[split]["true_states"].to(assessed[split][key])
        if progress is not None:
            progress(len(physical), len(physical))
        return physical_train, assessed, {"fixture": "reconstruction stub"}

    monkeypatch.setattr(runner, "source_data_config", configuration)
    monkeypatch.setattr(runner, "build_physical_cache", reconstruction)
    original_assessment = runner.assess_model
    final_calls: list[bool] = []

    def assessment(
        model: SCJEPA,
        data: EvaluationData,
        selection: DiagnosticSelection,
        *,
        bootstrap_samples: int,
        seed: int,
        final: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Tensor]]:
        final_calls.append(final)
        assert not list(model.online_encoder.parameters())
        assert not list(model.target_encoder.parameters())
        return original_assessment(
            model, data, selection, bootstrap_samples=bootstrap_samples, seed=seed, final=final
        )

    monkeypatch.setattr(runner, "assess_model", assessment)
    config = Config(
        str(source),
        str(tmp_path / "full"),
        seed=7,
        steps=3,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    full = run(config)
    assert final_calls == [False] * 4 + [True]
    checkpoint = cast(dict[str, Any], torch.load(tmp_path / "full/last.pt", weights_only=False))
    for name in ("parameter_encoder", "predictor"):
        assert any(
            not torch.equal(value, initial["model"][key])
            for key, value in checkpoint["model"].items()
            if key.startswith(name + ".")
        )
    original_loss = runner.training_loss
    update_calls = 0

    def interrupted(
        model: SCJEPA, batch: TensorSplit, objective: PredictiveSupervisionObjective
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal update_calls
        update_calls += 1
        if update_calls == 2:
            raise RuntimeError("interrupt")
        assert set(batch) == {"online", "target", "actions"}
        with pytest.raises(ValueError, match="no parameter labels"):
            original_loss(model, {**batch, "theta": torch.zeros(2, 6)}, objective)
        return original_loss(model, batch, objective)

    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    monkeypatch.setattr(runner, "training_loss", interrupted)
    final_calls.clear()
    with pytest.raises(RuntimeError, match="interrupt"):
        run(resumed_config)
    assert final_calls == [False, False]
    assert not (tmp_path / "resumed/final_test.json").exists()
    monkeypatch.setattr(runner, "training_loss", original_loss)
    resumed = run(resumed_config, resume=True)
    final = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    torch.testing.assert_close(checkpoint["model"], final["model"], rtol=0, atol=0)
    torch.testing.assert_close(checkpoint["optimizer"], final["optimizer"], rtol=0, atol=0)
    assert full["records"] == resumed["records"]
    assert full["final_test"] == resumed["final_test"]
    assert all(sha(path) == value for path, value in hashes.items())
    assert run(resumed_config, resume=True) == resumed
    cached = tmp_path / "resumed/physical_train_inputs.pt"
    cached.write_bytes(cached.read_bytes() + b"tampered")
    # Completed outputs must authenticate their cached inputs as well as report artifacts.
    with pytest.raises(ValueError, match="cache|artifacts"):
        run(resumed_config, resume=True)
