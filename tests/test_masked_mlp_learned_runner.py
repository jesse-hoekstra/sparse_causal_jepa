"""Learned parameter codes: label-free training, projection, dense initialization, resume."""

import math
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.balanced_action_oracle.data import PrepareConfig, prepare
from diagnostics.isotropic_oracle.data import PrepareConfig as SourcePrepareConfig
from diagnostics.isotropic_oracle.data import StateData
from diagnostics.isotropic_oracle.data import prepare as prepare_source
from diagnostics.masked_mlp_learned import acquisition, bound_control
from diagnostics.masked_mlp_learned import run as runner
from diagnostics.masked_mlp_oracle.model import INITIAL_LOGIT
from diagnostics.oracle_state_learning.core import EvaluationResult
from diagnostics.oracle_state_learning.data import read_json
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.loop import TrainingObjective


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _prepare(root: Path) -> None:
    prepare_source(
        SourcePrepareConfig(
            str(root / "source"),
            train_groups=4,
            calibration_groups=4,
            validation_groups=2,
            test_groups=2,
            profile="smoke",
        )
    )
    prepare(PrepareConfig(str(root / "data"), str(root / "source"), profile="smoke"))


def _config(root: Path, condition: runner.Condition, seed: int = 0) -> runner.TrainConfig:
    sparse = condition != "dense"
    return runner.TrainConfig(
        str(root / f"{condition}-s{seed}"),
        str(root / "data"),
        condition,
        profile="smoke",
        seed=seed,
        steps=4,
        batch_size=2,
        log_every=1,
        checkpoint_every=1,
        evaluate_every=3,
        calibration_dir=str(root / "bound") if sparse else None,
        init_dir=str(root / f"projected-s{seed}") if sparse else None,
    )


def _checkpoint(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], torch.load(path, weights_only=False, map_location="cpu"))


def _dense_and_bound(root: Path) -> None:
    runner.train(_config(root, "dense"))
    runner.project(
        runner.ProjectConfig(
            str(root / "projected-s0"), str(root / "dense-s0"), str(root / "data"), profile="smoke"
        )
    )
    runner.calibrate(
        runner.CalibrateConfig(str(root / "bound"), str(root / "projected-s0"), profile="smoke")
    )


def _train_split(groups: int = 32) -> StateData:
    """Synthetic TRAIN contexts whose group offsets dominate the within-group noise."""
    generator = torch.Generator().manual_seed(1)
    offsets = 2 * torch.randn(groups, 1, 1, 6, generator=generator)
    observations = offsets + 0.1 * torch.randn(groups, 10, 12, 6, generator=generator)
    tensors = {
        "observations": observations,
        "actions": torch.randn(groups, 1, 11, 24, generator=generator)
        + 0.1 * torch.randn(groups, 10, 11, 24, generator=generator),
        "oracle_parameters": torch.zeros(groups, 6),
        "episode_ids": torch.arange(groups),
    }
    return StateData("train", tensors, None, {})


def test_training_never_feeds_true_theta_to_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(tmp_path)
    seen: list[set[str]] = []
    actual = runner.forward_loss

    def observe(
        model: SCJEPA,
        batch: dict[str, Tensor],
        objective: TrainingObjective,
        *,
        forward_seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], ModelOutput]:
        if model.training:
            seen.append(set(batch))
        return actual(model, batch, objective, forward_seed=forward_seed)

    monkeypatch.setattr(runner, "forward_loss", observe)
    runner.train(_config(tmp_path, "dense"))
    assert seen
    assert all(keys == {"observations", "actions"} for keys in seen)
    model = runner.build_control("dense", 0)
    assert model.parameter_source == "learned"
    with pytest.raises(ValueError, match="oracle"):
        model(
            torch.zeros(2, 10, 12, 6),
            torch.zeros(2, 10, 11, 24),
            oracle_parameters=torch.zeros(2, 6),
        )


def test_sparse_arms_start_from_the_dense_endpoint_and_report_codes(tmp_path: Path) -> None:
    _prepare(tmp_path)
    _dense_and_bound(tmp_path)
    trained = _checkpoint(tmp_path / "dense-s0/last.pt")["model"]
    dense_initial = _checkpoint(tmp_path / "dense-s0/initial.pt")
    assert dense_initial["initialization"]["code_dimension"] == 12
    changed = [
        key
        for key in trained
        if key.startswith("parameter_encoder.")
        and not torch.equal(trained[key], dense_initial["model"][key])
    ]
    assert changed, "the parameter encoder must learn during dense training"
    assert trained["parameter_encoder.head.4.weight"].shape == (12, 128)
    dense = _checkpoint(tmp_path / "projected-s0/last.pt")["model"]
    assert dense["parameter_encoder.head.4.weight"].shape == (6, 128)
    assert dense["predictor.first.weight"].shape[1] == 36
    for condition in ("baseline", "full"):
        completion = runner.train(_config(tmp_path, condition))
        initial = _checkpoint(tmp_path / f"{condition}-s0/initial.pt")["model"]
        assert set(initial) - set(dense) <= runner.MASK_LOGITS
        for key, value in dense.items():
            if isinstance(value, Tensor):
                torch.testing.assert_close(initial[key], value, rtol=0, atol=0)
        for key in set(initial) - set(dense):
            assert bool((initial[key] == INITIAL_LOGIT).all())
        metrics = read_json(tmp_path / f"{condition}-s0/validation.json")["metrics"]
        for key in (
            "state_mcc",
            "block_purity",
            "code_r2_min",
            "code_mcc",
            "code_pair_purity",
            "forecast/code_use_ratio",
            "code_r2/theta_6",
            "geco/relative_prediction",
        ):
            assert key in metrics
        assert completion["dense_endpoint"]["seed"] == 0
        assert completion["dense_endpoint"]["kind"] == "projection"
        assert runner.verify_completion(tmp_path / f"{condition}-s0") == completion
    names = ("dense", "projected", "baseline", "full")
    rows = runner.summarize([str(tmp_path / f"{name}-s0") for name in names])
    assert [(row["condition"], row["code_dim"]) for row in rows] == [
        ("baseline", 6),
        ("dense", 6),
        ("dense", 12),
        ("full", 6),
    ]
    with pytest.raises(ValueError, match="project"):
        runner.calibrate(
            runner.CalibrateConfig(
                str(tmp_path / "unprojected"), str(tmp_path / "dense-s0"), profile="smoke"
            )
        )


def test_tighter_bound_preserves_sources_and_initial_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(tmp_path)
    _dense_and_bound(tmp_path)
    source = tmp_path / "bound"
    before = {path: path.read_bytes() for path in source.iterdir() if path.is_file()}
    original = runner.verify_completion(source, kind="calibration")
    read = bound_control.read_json
    read_names: list[str] = []

    def calibration_only(path: Path) -> dict[str, Any]:
        read_names.append(path.name)
        assert path.name == "calibration.json"
        return read(path)

    monkeypatch.setattr(bound_control, "read_json", calibration_only)
    output = tmp_path / "tighter"
    control = bound_control.tighten_bound(
        source, output, tau_factor=0.8, hypothesis="Preserve weak predictive dependencies"
    )
    assert read_names == ["calibration.json", "calibration.json"]
    assert control["tau_prediction"] == 0.8 * original["reference"]["prediction_loss"]
    assert control["identity"] != original["identity"]
    assert control["code_identity"] == original["code_identity"] == runner.code_identity()
    assert control["reference"] == original["reference"]
    assert control["bound_control"]["original_identity"] == original["identity"]
    assert control["bound_control"]["generator_sha256"] == runner.sha(Path(bound_control.__file__))
    assert all(path.read_bytes() == value for path, value in before.items())
    assert runner.verify_completion(output, kind="calibration") == control
    with pytest.raises(FileExistsError, match="not empty"):
        bound_control.tighten_bound(source, output, tau_factor=0.8, hypothesis="Same control")
    with pytest.raises(ValueError, match="original 1.1"):
        bound_control.tighten_bound(output, tmp_path / "chained", tau_factor=0.7, hypothesis="No")
    with pytest.raises(ValueError, match="prespecified"):
        runner.CalibrateConfig("unused", "unused", tau_factor=0.8)

    completion = runner.train(replace(_config(tmp_path, "full"), calibration_dir=str(output)))
    assert completion["calibration_identity"] == control["identity"]
    initial = _checkpoint(tmp_path / "full-s0/initial.pt")["model"]
    dense = _checkpoint(tmp_path / "projected-s0/last.pt")["model"]
    for key, value in dense.items():
        if isinstance(value, Tensor):
            torch.testing.assert_close(initial[key], value, rtol=0, atol=0)
    validation = read_json(tmp_path / "full-s0/validation.json")["metrics"]
    assert math.isclose(
        validation["geco/relative_prediction"],
        validation["prediction"] / control["tau_prediction"],
        rel_tol=1e-6,
    )

    with (source / "calibration.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="artifact hash"):
        bound_control.tighten_bound(
            source, tmp_path / "tampered", tau_factor=0.8, hypothesis="Reject modified source"
        )
    assert not (tmp_path / "tampered").exists()


@pytest.mark.parametrize("factor", [0.0, -1.0, 1.1, 2.0, float("nan"), float("inf")])
def test_tighter_bound_rejects_invalid_or_wider_constraints(tmp_path: Path, factor: float) -> None:
    with pytest.raises(ValueError, match="tighter control"):
        bound_control.tighten_bound(
            tmp_path / "unused", tmp_path / "output", tau_factor=factor, hypothesis="Declared"
        )
    assert not (tmp_path / "output").exists()


def test_codes_can_be_reacquired_on_frozen_trained_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(tmp_path)
    runner.train(_config(tmp_path, "dense"))
    config = replace(
        _config(tmp_path, "dense"),
        output_dir=str(tmp_path / "acquired-s0"),
        states_from=str(tmp_path / "dense-s0"),
    )
    completion = runner.train(config)
    assert completion["state_source"]["seed"] == 0
    source = _checkpoint(tmp_path / "dense-s0/last.pt")["model"]
    fresh = runner.build_control("dense", 0, 12, "shared_feature").state_dict()
    initial = _checkpoint(tmp_path / "acquired-s0/initial.pt")
    final = _checkpoint(tmp_path / "acquired-s0/last.pt")["model"]
    for key, value in final.items():
        if key.startswith(("online_encoder.", "target_encoder.")):
            torch.testing.assert_close(value, source[key], rtol=0, atol=0)
            torch.testing.assert_close(initial["model"][key], source[key], rtol=0, atol=0)
        elif key.startswith("predictor."):
            torch.testing.assert_close(initial["model"][key], fresh[key], rtol=0, atol=0)
    assert not torch.equal(
        initial["model"]["parameter_encoder.head.4.weight"],
        fresh["parameter_encoder.head.4.weight"],
    ), "shared-feature rows are fitted on the reused states"
    assert not torch.equal(
        final["parameter_encoder.head.4.weight"],
        initial["model"]["parameter_encoder.head.4.weight"],
    )
    _check_resume(tmp_path, config, monkeypatch)
    with pytest.raises(ValueError, match="jointly trained"):
        runner.train(
            replace(
                config,
                output_dir=str(tmp_path / "twice"),
                states_from=str(tmp_path / "acquired-s0"),
            )
        )
    with pytest.raises(ValueError, match="frozen states"):
        replace(_config(tmp_path, "full"), states_from="u")


def test_projection_is_exact_for_a_six_dimensional_code() -> None:
    source = runner.build_control("dense", 0, 12, "gain")
    output = cast(torch.nn.Sequential, source.parameter_encoder.head)[4]
    assert isinstance(output, torch.nn.Linear)
    mixing = torch.randn(6, 6, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        output.weight[6:] = mixing @ output.weight[:6]
        output.bias[6:] = mixing @ output.bias[:6] + 0.3
    target = runner.build_control("dense", 0, 6)
    train = _train_split()
    receipt = acquisition.project_code(source, target, train, torch.device("cpu"))
    assert receipt["reconstructed_variance_fraction"] > 1 - 1e-6
    batch = train.batch(torch.arange(8), "cpu")
    source.eval()
    target.eval()
    with torch.no_grad():
        expected = source(batch["observations"], batch["actions"]).predictions
        actual = target(batch["observations"], batch["actions"]).predictions
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    full = runner.build_control("full", 0, 6)
    with pytest.raises(ValueError, match="dense endpoint"):
        acquisition.project_code(
            runner.build_control("full", 0, 12), full, train, torch.device("cpu")
        )


def test_projected_adaptation_preserves_states_and_supports_exact_resume_and_sparse_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(tmp_path)
    _dense_and_bound(tmp_path)
    projected = tmp_path / "projected-s0"
    source = _checkpoint(projected / "last.pt")["model"]
    original_identity = runner.verify_completion(projected)["code_identity"]
    config = replace(
        _config(tmp_path, "dense"),
        output_dir=str(tmp_path / "adapted-s0"),
        adapt_from=str(projected),
    )
    seen: list[set[str]] = []
    actual = runner.forward_loss

    def observe(
        model: SCJEPA,
        batch: dict[str, Tensor],
        objective: TrainingObjective,
        *,
        forward_seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], ModelOutput]:
        if model.training:
            seen.append(set(batch))
            assert not any(p.requires_grad for p in model.online_encoder.parameters())
            assert not any(p.requires_grad for p in model.target_encoder.parameters())
            assert all(p.requires_grad for p in model.parameter_encoder.parameters())
            assert objective.geco is None
        return actual(model, batch, objective, forward_seed=forward_seed)

    # Model transfer records the old source identity. Ordinary calibration still
    # rejects the old runtime; adaptation's own resume uses the new identity.
    with monkeypatch.context() as patch:
        patch.setattr(runner, "code_identity", lambda: "f" * 64)
        patch.setattr(runner, "forward_loss", observe)
        with pytest.raises(ValueError, match="numerical source"):
            runner.calibrate(
                runner.CalibrateConfig(str(tmp_path / "old-bound"), str(projected), profile="smoke")
            )
        completion = runner.train(config)
        assert completion["adaptation_source"]["code_identity"] == original_identity
        assert completion["code_identity"] == "f" * 64
        assert completion["code_dim"] == 6
        initial = _checkpoint(tmp_path / "adapted-s0/initial.pt")["model"]
        final = _checkpoint(tmp_path / "adapted-s0/last.pt")["model"]
        torch.testing.assert_close(initial, source, rtol=0, atol=0)
        for key, value in final.items():
            if key.startswith(runner.STATE_ENCODERS):
                torch.testing.assert_close(value, source[key], rtol=0, atol=0)
        for prefix in ("parameter_encoder.", "predictor."):
            assert any(
                key.startswith(prefix)
                and isinstance(value, Tensor)
                and not torch.equal(value, initial[key])
                for key, value in final.items()
            )
        _check_resume(tmp_path, config, monkeypatch)
        assert seen
        assert all(keys == {"observations", "actions"} for keys in seen)

        # The fixed adapted endpoint is a normal six-code dense parent for both
        # sparse arms, where the online states become trainable again.
        patch.setattr(runner, "forward_loss", actual)
        bound = tmp_path / "adapted-bound"
        runner.calibrate(runner.CalibrateConfig(str(bound), config.output_dir, profile="smoke"))
        full = replace(
            _config(tmp_path, "full"), init_dir=config.output_dir, calibration_dir=str(bound)
        )
        runner.train(full)
        sparse_initial = _checkpoint(tmp_path / "full-s0/initial.pt")["model"]
        sparse_final = _checkpoint(tmp_path / "full-s0/last.pt")["model"]
        for key, value in final.items():
            torch.testing.assert_close(sparse_initial[key], value, rtol=0, atol=0)
        assert not torch.equal(
            sparse_final["online_encoder.head.weight"], sparse_initial["online_encoder.head.weight"]
        )


def test_adaptation_requires_an_authenticated_matching_projection(tmp_path: Path) -> None:
    _prepare(tmp_path)
    _dense_and_bound(tmp_path)
    config = replace(
        _config(tmp_path, "dense"),
        output_dir=str(tmp_path / "adapted-s0"),
        adapt_from=str(tmp_path / "projected-s0"),
    )
    with pytest.raises(ValueError, match="seed-matched"):
        runner.train(replace(config, seed=1, output_dir=str(tmp_path / "wrong-seed")))
    with pytest.raises(ValueError, match="stage differs"):
        runner.train(
            replace(
                config,
                adapt_from=str(tmp_path / "dense-s0"),
                output_dir=str(tmp_path / "wrong-kind"),
            )
        )
    for override in ({"states_from": "u"}, {"code_dim": 12}, {"acquisition_seed": 10}):
        with pytest.raises(ValueError, match="adaptation inherits"):
            replace(config, **override)
    with pytest.raises(ValueError, match="one of"):
        runner.TrainConfig("u", "u", "dense", adapt_from="u", steps=100_000)
    declared = runner.TrainConfig("u", "u", "dense", adapt_from="u", steps=50_000)
    assert declared.model_code_dim == 6
    assert declared.model_initializer == "native"
    with (tmp_path / "projected-s0/projection.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="artifact hash"):
        runner.train(config)


def test_acquisition_restart_changes_initialization_not_states_or_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(tmp_path)
    runner.train(_config(tmp_path, "dense"))
    config = replace(
        _config(tmp_path, "dense"),
        output_dir=str(tmp_path / "restart-s0"),
        states_from=str(tmp_path / "dense-s0"),
        acquisition_seed=1000,
    )
    completion = runner.train(config)
    source = _checkpoint(tmp_path / "dense-s0/last.pt")
    initial = _checkpoint(tmp_path / "restart-s0/initial.pt")["model"]
    final = _checkpoint(tmp_path / "restart-s0/last.pt")
    fresh = runner.build_control("dense", 1000, 12, "shared_feature").state_dict()
    previous = runner.build_control("dense", 0, 12, "shared_feature").state_dict()
    for key, value in initial.items():
        if key.startswith(runner.STATE_ENCODERS):
            torch.testing.assert_close(value, source["model"][key], rtol=0, atol=0)
            torch.testing.assert_close(final["model"][key], value, rtol=0, atol=0)
        elif key.startswith("predictor."):
            torch.testing.assert_close(value, fresh[key], rtol=0, atol=0)
    assert not torch.equal(initial["predictor.first.weight"], previous["predictor.first.weight"])
    torch.testing.assert_close(final["sampler"], source["sampler"], rtol=0, atol=0)
    assert completion["seed"] == completion["state_source"]["seed"] == 0
    assert (
        read_json(tmp_path / "restart-s0/config.json")["scientific_config"]["acquisition_seed"]
        == 1000
    )
    _check_resume(tmp_path, config, monkeypatch)
    with pytest.raises(ValueError, match="frozen states"):
        replace(config, states_from=None)
    with pytest.raises(ValueError, match="integer"):
        replace(config, acquisition_seed=-1)


def test_acquisition_initializers_keep_everything_else() -> None:
    native = runner.build_control("dense", 0, 12)
    gained = runner.build_control("dense", 0, 12, "shared_feature")
    for key, value in native.state_dict().items():
        other = gained.state_dict()[key]
        if key.startswith("parameter_encoder.") and key.endswith("weight"):
            gain = 3 if key == "parameter_encoder.head.4.weight" else 6
            torch.testing.assert_close(other, value * gain**0.5)
        else:
            torch.testing.assert_close(other, value, rtol=0, atol=0)
    train = _train_split()
    before = {
        key: value.clone()
        for key, value in gained.state_dict().items()
        if isinstance(value, Tensor)
    }
    old = acquisition.train_contexts(gained, train, torch.device("cpu"), codes=True)
    receipt = acquisition.initialize_shared_output(gained, train, torch.device("cpu"))
    new = acquisition.train_contexts(gained, train, torch.device("cpu"), codes=True)
    after = gained.state_dict()
    changed = {key for key, value in before.items() if not torch.equal(after[key], value)}
    assert changed == {"parameter_encoder.head.4.weight", "parameter_encoder.head.4.bias"}
    old_rows, new_rows = old.flatten(0, 1), new.flatten(0, 1)
    torch.testing.assert_close(new_rows.mean(0), old_rows.mean(0), rtol=0, atol=1e-5)
    torch.testing.assert_close(
        new_rows.var(0, correction=0).mean(),
        old_rows.var(0, correction=0).mean(),
        rtol=1e-4,
        atol=0,
    )
    assert receipt["code_dimension"] == 12
    assert receipt["training_parameter_labels_absent"]


def test_code_metrics_recover_an_informative_code() -> None:
    generator = torch.Generator().manual_seed(0)
    theta = torch.rand(400, 6, generator=generator, dtype=torch.float64) * 1.5 - 0.75
    pair_mix = torch.eye(6, dtype=torch.float64)
    pair_mix[0, 3] = pair_mix[3, 0] = 0.6

    def result(groups: slice) -> EvaluationResult:
        part = theta[groups]
        codes = (part @ pair_mix.T)[:, None].expand(-1, 3, -1)
        return EvaluationResult({}, {"codes": codes.float(), "theta": part.float()})

    metrics = runner.code_metrics(result(slice(0, 300)), result(slice(300, 400)))
    assert metrics["code_r2_min"] > 0.99
    assert metrics["code_pair_purity"] > 0.99
    assert metrics["code_mcc"] < 0.99


def test_shuffled_codes_route_codes_across_groups() -> None:
    model = runner.build_control("dense", 0)
    original = model.parameter_encoder
    states = torch.randn(3 * 2, 10, 6)
    actions = torch.randn(3 * 2, 9, 24)
    with torch.no_grad():
        codes = original(states, actions).reshape(3, 2, 6)
        with runner.shuffled_codes(model, views=2):
            routed = model.parameter_encoder(states, actions).reshape(3, 2, 6)
    torch.testing.assert_close(routed, codes.roll(1, dims=0))
    assert model.parameter_encoder is original


def test_protocol_is_checked_and_resume_is_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="dense endpoint"):
        runner.TrainConfig("u", "u", "full", calibration_dir="u")
    with pytest.raises(ValueError, match="one of"):
        runner.TrainConfig("u", "u", "dense", steps=200_000)
    with pytest.raises(ValueError, match="seed"):
        runner.TrainConfig("u", "u", "dense", seed=5)
    with pytest.raises(ValueError, match="inherit"):
        runner.TrainConfig("u", "u", "full", calibration_dir="u", init_dir="u", code_dim=12)
    _prepare(tmp_path)
    _dense_and_bound(tmp_path)
    for config in (_config(tmp_path, "dense"), _config(tmp_path, "full")):
        _check_resume(tmp_path, config, monkeypatch)


def _check_resume(root: Path, config: runner.TrainConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    complete = (
        runner.verify_completion(Path(config.output_dir))
        if config.condition == "dense"
        else runner.train(config)
    )
    interrupted = replace(config, output_dir=str(root / f"interrupted-{config.condition}"))
    actual = runner.forward_loss
    updates = 0

    def interrupt(
        model: SCJEPA,
        batch: dict[str, Tensor],
        objective: TrainingObjective,
        *,
        forward_seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], ModelOutput]:
        nonlocal updates
        if model.training:
            updates += 1
            if updates == 3:
                raise RuntimeError("deliberate interruption")
        return actual(model, batch, objective, forward_seed=forward_seed)

    with monkeypatch.context() as patch:
        patch.setattr(runner, "forward_loss", interrupt)
        with pytest.raises(RuntimeError, match="deliberate"):
            runner.train(interrupted)
    assert _checkpoint(Path(interrupted.output_dir) / "last.pt")["step"] == 2
    resumed = runner.train(interrupted, resume=True)
    assert complete["identity"] == resumed["identity"]
    left = _checkpoint(Path(config.output_dir) / "last.pt")
    right = _checkpoint(Path(interrupted.output_dir) / "last.pt")
    for key in ("model", "optimizer", "objective", "sampler", "torch_rng"):
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
