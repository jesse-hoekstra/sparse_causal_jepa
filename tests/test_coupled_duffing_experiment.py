from pathlib import Path
from typing import Any, cast

import pytest
import torch

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
def test_target_synchronization_is_only_allowed_when_states_are_frozen(phase: str) -> None:
    with pytest.raises(ValueError, match="frozen-state"):
        StageConfig(
            "data",
            "out",
            phase,
            source_dir=None if phase == "joint" else "source",
            synchronize_target=True,
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
    ("architecture", "state_regularization", "synchronize_target"),
    [
        ("mlp", "aligned", False),
        ("temporal-cnn", "aligned", False),
        ("temporal-cnn", "projected-mixture-variance", False),
        ("temporal-cnn", "projected-aligned", False),
        ("temporal-cnn", "projected-aligned", True),
    ],
)
def test_complete_stages_label_free_selection_and_exact_resume(
    tmp_path: Path,
    architecture: ParameterArchitecture,
    state_regularization: StateRegularization,
    synchronize_target: bool,
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
        )
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
