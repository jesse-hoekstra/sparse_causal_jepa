import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from diagnostics.coupled_duffing import experiment
from diagnostics.coupled_duffing.acquisition import (
    initialize_shared_output,
    project,
    training_features,
)
from diagnostics.coupled_duffing.data import DataConfig, generate, prepare, render
from diagnostics.coupled_duffing.experiment import (
    BUDGETS,
    StageConfig,
    calibrate_endpoint,
    experiment_data,
    project_endpoint,
    select,
    train,
    verify,
)
from diagnostics.coupled_duffing.model import ParameterArchitecture, build_model
from diagnostics.coupled_duffing.regularization import StateRegularization
from diagnostics.coupled_duffing.visual_data import prepare as prepare_visual
from diagnostics.coupled_duffing_probe.data import ProbeDataConfig
from diagnostics.coupled_duffing_probe.data import prepare as prepare_physical
from diagnostics.masked_mlp_learned.selection import training_score
from diagnostics.shared.io import read_json
from diagnostics.shared.sampling import GroupSampler
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.loop import TrainingObjective


def test_successful_experiment1_counts_and_budgets_are_preserved() -> None:
    config = experiment_data()
    assert (config.train_groups, config.calibration_groups, config.validation_groups) == (
        8192,
        512,
        256,
    )
    assert (config.views, config.context, config.forecast) == (10, 10, 2)
    assert BUDGETS == {
        "joint": 100000,
        "acquisition": 100000,
        "adaptation": 50000,
        "baseline": 350000,
        "full": 350000,
    }
    assert StageConfig("data", "out", "joint").code_dim == 8
    assert StageConfig("data", "out", "baseline", source_dir="parent").code_dim == 4


@pytest.mark.parametrize("learning_rate", [0.0, -0.001, float("nan"), float("inf")])
def test_invalid_optimizer_rate_is_rejected(learning_rate: float) -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        StageConfig("data", "out", "joint", learning_rate=learning_rate)


@pytest.mark.parametrize("gate_learning_rate", [0.0, -0.001, float("inf"), float("nan")])
def test_invalid_gate_optimizer_rate_is_rejected(gate_learning_rate: float) -> None:
    with pytest.raises(ValueError, match="gate_learning_rate"):
        StageConfig(
            "data", "out", "full", source_dir="source", gate_learning_rate=gate_learning_rate
        )


@pytest.mark.parametrize("initial_gate_logit", [float("inf"), float("nan")])
def test_invalid_gate_initialization_is_rejected(initial_gate_logit: float) -> None:
    with pytest.raises(ValueError, match="initial_gate_logit"):
        StageConfig(
            "data", "out", "full", source_dir="source", initial_gate_logit=initial_gate_logit
        )


def test_custom_gate_initialization_requires_a_sparse_stage() -> None:
    with pytest.raises(ValueError, match="sparse stages"):
        StageConfig("data", "out", "joint", initial_gate_logit=10.0)


@pytest.mark.parametrize("rate", [0.0, -0.01, 0.002, float("nan"), float("inf")])
def test_invalid_cosine_final_rate_is_rejected(rate: float) -> None:
    with pytest.raises(ValueError, match="final_learning_rate"):
        StageConfig("data", "out", "adaptation", source_dir="source", final_learning_rate=rate)


def test_cosine_schedule_requires_a_frozen_multi_update_stage() -> None:
    with pytest.raises(ValueError, match="frozen stage"):
        StageConfig("data", "out", "full", source_dir="source", final_learning_rate=0.0001)
    with pytest.raises(ValueError, match="at least two"):
        StageConfig(
            "data", "out", "adaptation", source_dir="source", steps=1, final_learning_rate=0.0001
        )


@pytest.mark.parametrize("count", [0, -1])
def test_gradient_accumulation_requires_positive_count(count: int) -> None:
    with pytest.raises(ValueError, match="gradient_accumulation"):
        StageConfig("data", "out", "full", source_dir="source", gradient_accumulation=count)


def test_gradient_accumulation_requires_a_sparse_stage() -> None:
    with pytest.raises(ValueError, match="sparse stages"):
        StageConfig("data", "out", "joint", gradient_accumulation=8)


@pytest.mark.parametrize("phase", ["joint", "acquisition", "adaptation"])
def test_separate_gate_rate_requires_a_sparse_stage(phase: str) -> None:
    with pytest.raises(ValueError, match="sparse stages"):
        StageConfig(
            "data",
            "out",
            phase,
            source_dir=None if phase == "joint" else "source",
            acquisition_seed=1000 if phase == "acquisition" else None,
            gate_learning_rate=0.001,
        )


@pytest.mark.parametrize("phase", ["joint", "baseline", "full"])
def test_target_synchronization_is_only_allowed_at_acquisition_or_adaptation(phase: str) -> None:
    with pytest.raises(ValueError, match="acquisition/adaptation"):
        StageConfig(
            "data",
            "out",
            phase,
            source_dir=None if phase == "joint" else "source",
            synchronize_target=True,
        )


@pytest.mark.parametrize("phase", ["joint", "baseline", "full"])
def test_state_caching_requires_frozen_state_encoders(phase: str) -> None:
    with pytest.raises(ValueError, match="frozen-state stages"):
        StageConfig(
            "data",
            "out",
            phase,
            source_dir=None if phase == "joint" else "source",
            cache_frozen_states=True,
        )


def test_joint_adaptation_requires_adaptation_and_disallows_fixed_state_replay() -> None:
    with pytest.raises(ValueError, match="post-projection adaptation"):
        StageConfig("data", "out", "joint", joint_adaptation=True)
    with pytest.raises(ValueError, match="frozen-state stages"):
        StageConfig(
            "data",
            "out",
            "adaptation",
            source_dir="source",
            joint_adaptation=True,
            cache_frozen_states=True,
        )
    with pytest.raises(ValueError, match="frozen stage"):
        StageConfig(
            "data",
            "out",
            "adaptation",
            source_dir="source",
            joint_adaptation=True,
            final_learning_rate=0.0001,
        )


@pytest.mark.parametrize("architecture", ["mlp", "temporal-cnn"])
def test_shared_feature_initialization_preserves_mean_and_centered_rms(
    architecture: ParameterArchitecture,
) -> None:
    torch.set_num_threads(1)
    data, labels = generate(12, 0, DataConfig(views=3, context=6, forecast=2))
    inputs = {key: data[key] for key in ("observations", "actions")}
    model = build_model(0, 6, code_dim=8, parameter_encoder=architecture)
    before = training_features(model, inputs, codes=True).flatten(0, 1)
    initialize_shared_output(model, inputs)
    after = training_features(model, inputs, codes=True).flatten(0, 1)
    torch.testing.assert_close(after.mean(0), before.mean(0), rtol=1e-4, atol=3e-7)
    torch.testing.assert_close(
        after.var(0, correction=0).mean(), before.var(0, correction=0).mean(), rtol=1e-4, atol=1e-8
    )
    with pytest.raises(ValueError, match="only observations and actions"):
        initialize_shared_output(model, {**inputs, "theta": labels["theta"]})


@pytest.mark.parametrize("architecture", ["mlp", "temporal-cnn"])
def test_projection_preserves_affine_code_transfer_and_predictor_function(
    architecture: ParameterArchitecture,
) -> None:
    torch.set_num_threads(1)
    data, _ = generate(16, 0, DataConfig(views=3, context=6, forecast=2))
    # Check the affine identity in double precision; the full pipeline below uses float32.
    inputs = {key: data[key].double() for key in ("observations", "actions")}
    source = build_model(2, 6, code_dim=8, parameter_encoder=architecture).double().eval()
    target = build_model(2, 6, parameter_encoder=architecture).double().eval()
    initialize_shared_output(source, inputs)
    codes = training_features(source, inputs, codes=True)
    record, matrices = project(source, target, inputs)
    actual = training_features(target, inputs, codes=True)
    expected = (codes - matrices["mean"]) @ matrices["projection"].T
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-12)
    assert (record["source_code_dimension"], record["code_dimension"]) == (8, 4)
    torch.testing.assert_close(
        matrices["projection"] @ matrices["projection"].T, torch.eye(4, dtype=torch.float64)
    )
    short = actual.flatten(0, 1)
    reconstructed = actual.flatten(0, 1) @ matrices["reconstruction"].T + matrices["mean"]
    states = torch.randn(len(short), 2, 2, dtype=torch.float64)
    actions = torch.randn(len(short), 2, 1, dtype=torch.float64)
    with torch.no_grad():
        a = source.predictor(states, reconstructed, actions).prediction
        b = target.predictor(states, short, actions).prediction
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
    for key, value in source.online_encoder.state_dict().items():
        torch.testing.assert_close(value, target.online_encoder.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    (
        "architecture",
        "state_regularization",
        "synchronize_target",
        "cache_frozen_states",
        "initial_gate_logit",
        "cosine_adaptation",
        "gradient_accumulation",
        "joint_adaptation",
    ),
    [
        ("mlp", "aligned", False, False, 3.0, False, 1, False),
        ("temporal-cnn", "aligned", False, False, 3.0, False, 1, False),
        ("temporal-cnn", "projected-mixture-variance", False, False, 3.0, False, 1, False),
        ("temporal-cnn", "projected-aligned", False, False, 3.0, False, 1, False),
        ("temporal-cnn", "projected-aligned", True, False, 3.0, False, 1, False),
        ("temporal-cnn", "projected-aligned", False, True, 3.0, False, 1, False),
        ("temporal-cnn", "projected-aligned", True, True, 3.0, False, 1, False),
        ("temporal-cnn", "projected-aligned", True, True, 10.0, False, 1, False),
        ("temporal-cnn", "projected-aligned", True, True, 3.0, True, 1, False),
        ("temporal-cnn", "projected-aligned", True, True, 3.0, False, 8, False),
        ("temporal-cnn", "projected-aligned", True, False, 3.0, False, 1, True),
    ],
)
def test_complete_stages_label_free_selection_and_exact_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    architecture: ParameterArchitecture,
    state_regularization: StateRegularization,
    synchronize_target: bool,
    cache_frozen_states: bool,
    initial_gate_logit: float,
    cosine_adaptation: bool,
    gradient_accumulation: int,
    joint_adaptation: bool,
) -> None:
    torch.set_num_threads(1)
    data = tmp_path / "data"
    if architecture == "temporal-cnn":
        physical = tmp_path / "physical"
        prepare_physical(
            physical,
            ProbeDataConfig(
                train_groups=12,
                calibration_groups=8,
                validation_groups=8,
                views=3,
                context=6,
                forecast=4,
                forcing="driven",
            ),
        )
        prepare_visual(data, physical)
    else:
        prepare(
            data,
            DataConfig(
                train_groups=12,
                calibration_groups=8,
                validation_groups=8,
                views=3,
                context=6,
                forecast=2,
            ),
        )
    (data / "train-labels.pt").unlink()

    def stage(
        name: str,
        phase: str,
        source: Path | None = None,
        initialization: int | None = None,
        calibration_file: Path | None = None,
    ) -> StageConfig:
        config = StageConfig(
            str(data),
            str(tmp_path / name),
            phase,
            source_dir=str(source) if source else None,
            acquisition_seed=initialization,
            steps=2,
            batch_size=4,
            log_every=1,
            eval_every=1,
            device="cpu",
            wandb_mode="disabled",
            parameter_encoder=architecture,
            cache_device="cpu",
            state_regularization=state_regularization,
            calibration_file=str(calibration_file) if calibration_file else None,
            learning_rate=0.0001 if state_regularization == "projected-aligned" else 0.001,
            gate_learning_rate=0.001
            if state_regularization == "projected-aligned"
            and not synchronize_target
            and phase in {"baseline", "full"}
            else None,
            synchronize_target=synchronize_target and phase in {"acquisition", "adaptation"},
            cache_frozen_states=cache_frozen_states and phase in {"acquisition", "adaptation"},
            initial_gate_logit=initial_gate_logit if phase in {"baseline", "full"} else 3.0,
            final_learning_rate=0.00001 if cosine_adaptation and phase == "adaptation" else None,
            gradient_accumulation=gradient_accumulation if phase in {"baseline", "full"} else 1,
            joint_adaptation=joint_adaptation and phase == "adaptation",
        )
        if phase == "full" and gradient_accumulation > 1:
            recorded: list[tuple[torch.Tensor, ...]] = []
            original_forward = experiment.forward_loss
            original_clip = torch.nn.utils.clip_grad_norm_

            def observe_minibatch(
                model: SCJEPA, batch: dict[str, torch.Tensor], objective: TrainingObjective
            ) -> tuple[torch.Tensor, dict[str, torch.Tensor], ModelOutput]:
                result = original_forward(model, batch, objective)
                parameters = [
                    p
                    for module in (model, objective)
                    for p in module.parameters()
                    if p.requires_grad
                ]
                recorded.append(torch.autograd.grad(result[0], parameters, retain_graph=True))
                return result

            def check_unclipped_mean(
                parameters: Iterable[torch.Tensor],
                max_norm: float,
                *,
                error_if_nonfinite: bool = False,
            ) -> torch.Tensor:
                supplied = list(parameters)
                assert len(recorded) == gradient_accumulation
                for index, parameter in enumerate(supplied):
                    expected = torch.stack([row[index] for row in recorded]).mean(0)
                    assert parameter.grad is not None
                    torch.testing.assert_close(parameter.grad, expected, rtol=1e-5, atol=1e-7)
                recorded.clear()
                return original_clip(supplied, max_norm, error_if_nonfinite=error_if_nonfinite)

            with monkeypatch.context() as patch:
                patch.setattr(experiment, "forward_loss", observe_minibatch)
                patch.setattr(torch.nn.utils, "clip_grad_norm_", check_unclipped_mean)
                train(config)
            assert not recorded
        else:
            train(config)
        return config

    joint = stage("joint", "joint")
    if state_regularization == "projected-aligned":
        specification = read_json(Path(joint.output_dir) / "completion.json")
        assert specification["state_projector"] == [2, 128, 128, 2]
        assert specification["sigreg_sampling"] == "absolute_time"
        assert specification["state_variance_floor_weight"] == 0.0
    joint_samples = cast(
        dict[str, torch.Tensor],
        torch.load(Path(joint.output_dir) / "validation.pt", weights_only=True),
    )
    assert read_json(Path(joint.output_dir) / "completion.json")["metrics"][
        "code_std/min"
    ] == float(joint_samples["codes"].reshape(-1, 8).std(0).min())
    first = stage("attempt0", "acquisition", Path(joint.output_dir), 1000)
    second = stage("attempt1", "acquisition", Path(joint.output_dir), 1100)
    candidates = [Path(first.output_dir), Path(second.output_dir)]
    with pytest.raises(ValueError, match="complete prespecified pool"):
        select(candidates, window=2, maximum_loss=100.0)
    selection = select(candidates, window=2, maximum_loss=100.0, expected_candidates=2)
    assert selection["selection_used_parameter_labels"] is False
    assert selection["selected"]["train_prediction_mean"] == min(
        training_score(path, steps=2, window=2, log_every=1) for path in candidates
    )
    projected = tmp_path / "projected"
    project_endpoint(data, Path(selection["selected"]["directory"]), projected, "cpu", "cpu")
    assert read_json(projected / "completion.json")["parameter_encoder"] == architecture
    adapted = stage("adapted", "adaptation", projected)
    if cache_frozen_states or joint_adaptation:
        adapted_directory = Path(adapted.output_dir)
        before_resume = cast(
            dict[str, Any], torch.load(adapted_directory / "last.pt", weights_only=False)
        )
        (adapted_directory / "last.pt").write_bytes((adapted_directory / "initial.pt").read_bytes())
        train(adapted, resume=True)
        after_resume = cast(
            dict[str, Any], torch.load(adapted_directory / "last.pt", weights_only=False)
        )
        for key, value in before_resume["model"].items():
            torch.testing.assert_close(value, after_resume["model"][key], rtol=0, atol=0)
        assert read_json(adapted_directory / "completion.json")["frozen_state_cache"] == (
            cache_frozen_states
        )
        for key, value in before_resume["objective"].items():
            torch.testing.assert_close(value, after_resume["objective"][key], rtol=0, atol=0)
        if cosine_adaptation:
            assert before_resume["optimizer"]["param_groups"][0]["lr"] == 0.00001
            assert after_resume["optimizer"]["param_groups"][0]["lr"] == 0.00001
            history = [
                json.loads(line)
                for line in (adapted_directory / "history.jsonl").read_text().splitlines()
            ]
            assert [row["optimizer/learning_rate"] for row in history[:2]] == [0.0001, 0.00001]
    calibration_file = tmp_path / "shared-calibration.json"
    calibration = calibrate_endpoint(Path(adapted.output_dir), calibration_file)
    baseline = stage(
        "baseline", "baseline", Path(adapted.output_dir), calibration_file=calibration_file
    )
    full = stage("full", "full", Path(adapted.output_dir), calibration_file=calibration_file)
    initial_pair = [
        cast(dict[str, Any], torch.load(Path(c.output_dir) / "initial.pt", weights_only=False))
        for c in (baseline, full)
    ]
    for name, value in initial_pair[0]["model"].items():
        torch.testing.assert_close(value, initial_pair[1]["model"][name], rtol=0, atol=0)
    assert initial_pair[0]["tau"] == initial_pair[1]["tau"] == calibration["tau"]
    adapted_state = cast(
        dict[str, Any], torch.load(Path(adapted.output_dir) / "last.pt", weights_only=False)
    )
    for name, value in adapted_state["model"].items():
        torch.testing.assert_close(value, initial_pair[1]["model"][name], rtol=0, atol=0)
    for name in ("state", "action", "parameter"):
        value = initial_pair[1]["model"][f"predictor.{name}_logits"]
        torch.testing.assert_close(
            value, torch.full_like(value, initial_gate_logit), rtol=0, atol=0
        )
    assert read_json(Path(full.output_dir) / "completion.json")["initial_gate_logit"] == (
        initial_gate_logit
    )
    full_final = cast(
        dict[str, Any], torch.load(Path(full.output_dir) / "last.pt", weights_only=False)
    )
    assert not torch.equal(
        initial_pair[1]["model"]["predictor.parameter_logits"],
        full_final["model"]["predictor.parameter_logits"],
    )
    for prefix in ("online_encoder.", "target_encoder.", "parameter_encoder."):
        assert any(
            not torch.equal(value, full_final["model"][name])
            for name, value in initial_pair[1]["model"].items()
            if name.startswith(prefix)
        )
    source = cast(
        dict[str, Any], torch.load(Path(joint.output_dir) / "last.pt", weights_only=False)
    )
    if synchronize_target:
        assert any(
            not torch.equal(
                value, source["model"][key.replace("target_encoder.", "online_encoder.")]
            )
            for key, value in source["model"].items()
            if key.startswith("target_encoder.")
        )
    if state_regularization.startswith("projected-"):
        initial = cast(
            dict[str, Any], torch.load(Path(joint.output_dir) / "initial.pt", weights_only=False)
        )
        assert any(
            not torch.equal(value, initial["objective"][key])
            for key, value in source["objective"].items()
            if key.startswith("projector.")
        )
    for path in (*candidates, Path(adapted.output_dir)):
        frozen = cast(dict[str, Any], torch.load(path / "last.pt", weights_only=False))
        if joint_adaptation and path == Path(adapted.output_dir):
            initial = cast(dict[str, Any], torch.load(path / "initial.pt", weights_only=False))
            for prefix in (
                "online_encoder.",
                "target_encoder.",
                "parameter_encoder.",
                "predictor.",
            ):
                assert any(
                    not torch.equal(value, frozen["model"][key])
                    for key, value in initial["model"].items()
                    if key.startswith(prefix)
                )
            assert any(
                not torch.equal(value, frozen["objective"][key])
                for key, value in initial["objective"].items()
                if key.startswith("projector.")
            )
            assert read_json(path / "completion.json")["state_encoders_frozen"] is False
            assert any(
                not torch.equal(
                    value, frozen["model"][key.replace("target_encoder.", "online_encoder.")]
                )
                for key, value in frozen["model"].items()
                if key.startswith("target_encoder.")
            )
            continue
        for key, value in source["model"].items():
            if key.startswith(("online_encoder.", "target_encoder.")):
                expected = (
                    source["model"][key.replace("target_encoder.", "online_encoder.")]
                    if synchronize_target
                    else value
                )
                torch.testing.assert_close(expected, frozen["model"][key], rtol=0, atol=0)
        for key, value in source["objective"].items():
            if key.startswith("projector."):
                torch.testing.assert_close(value, frozen["objective"][key], rtol=0, atol=0)
    directory = Path(full.output_dir)
    complete = cast(dict[str, Any], torch.load(directory / "last.pt", weights_only=False))
    (directory / "last.pt").write_bytes((directory / "initial.pt").read_bytes())
    train(full, resume=True)
    resumed = cast(dict[str, Any], torch.load(directory / "last.pt", weights_only=False))
    for key, value in complete["model"].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)
    assert resumed["objective"]["geco.num_updates"] == 2
    sampler = GroupSampler(12, full.batch_size, full.seed)
    for _ in range(full.updates * full.gradient_accumulation):
        sampler.next()
    for key, value in sampler.state_dict().items():
        actual = resumed["sampler"][key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, actual)
        else:
            assert value == actual
    specification = read_json(directory / "completion.json")
    assert specification["sigreg_systems_per_statistic"] == full.batch_size
    assert specification["systems_per_optimizer_step"] == full.batch_size * gradient_accumulation
    for key, value in complete["objective"].items():
        torch.testing.assert_close(value, resumed["objective"][key], rtol=0, atol=0)
    assert complete["tau"] == resumed["tau"]
    assert complete["optimizer"]["param_groups"][0]["lr"] == full.learning_rate
    assert resumed["optimizer"]["param_groups"][0]["lr"] == full.learning_rate
    if full.gate_learning_rate is not None:
        groups = complete["optimizer"]["param_groups"]
        resumed_groups = resumed["optimizer"]["param_groups"]
        assert len(groups) == 2
        assert groups[0]["role"] == "continuous"
        assert groups[1]["role"] == "gates"
        assert len(groups[1]["params"]) == 3
        assert groups[1]["lr"] == full.gate_learning_rate
        assert resumed_groups == groups
        initial = cast(dict[str, Any], torch.load(directory / "initial.pt", weights_only=False))
        for name in ("state", "action", "parameter"):
            key = f"predictor.{name}_logits"
            assert not torch.equal(initial["model"][key], complete["model"][key])
        assert not torch.equal(
            initial["model"]["predictor.first.weight"], complete["model"]["predictor.first.weight"]
        )
    else:
        assert len(complete["optimizer"]["param_groups"]) == 1
    assert read_json(directory / "completion.json")["learning_rate"] == full.learning_rate
    assert not list(data.glob("test*"))
    verify(directory)
    verify(Path(joint.output_dir))
    with pytest.raises(ValueError, match="different numerical implementation"):
        verify(directory, expected_code_identity="0" * 64)
    verify(
        directory, expected_code_identity=read_json(directory / "completion.json")["code_identity"]
    )


def test_visual_cache_preserves_physical_trajectories_and_separates_labels(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    physical, visual = tmp_path / "physical", tmp_path / "visual"
    source = prepare_physical(
        physical,
        ProbeDataConfig(
            train_groups=12,
            calibration_groups=8,
            validation_groups=8,
            views=3,
            context=10,
            forecast=5,
            forcing="driven",
        ),
    )
    record = prepare_visual(visual, physical)
    assert record["source_identity"] == source["identity"]
    assert record["config"]["context"] == 10
    assert record["config"]["forecast"] == 2
    assert record["forcing"] == "driven"
    assert record["true_states_in_model_inputs"] is False
    ids: list[set[int]] = []
    for split in ("train", "calibration", "validation"):
        inputs = cast(
            dict[str, torch.Tensor], torch.load(visual / f"{split}-inputs.pt", weights_only=True)
        )
        labels = cast(
            dict[str, torch.Tensor], torch.load(visual / f"{split}-labels.pt", weights_only=True)
        )
        original = cast(
            dict[str, torch.Tensor], torch.load(physical / f"{split}-inputs.pt", weights_only=True)
        )
        assert set(inputs) == {"observations", "actions", "group_ids"}
        assert inputs["observations"].shape[2:] == (12, 1, 16, 32)
        torch.testing.assert_close(inputs["observations"], render(labels["states"], DataConfig()))
        torch.testing.assert_close(labels["states"], original["observations"][:, :, :12])
        torch.testing.assert_close(inputs["actions"], original["actions"][:, :, :11])
        ids.append(set(int(value) for value in inputs["group_ids"]))
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert not list(visual.glob("test*"))
