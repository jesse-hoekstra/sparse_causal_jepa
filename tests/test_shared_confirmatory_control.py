import copy
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_overcomplete_use_precision import source_pair as source_pair
from test_shared_code_calibration import shared_sources as shared_sources
from torch import Tensor

from diagnostics.global_code_calibration.objective import compose_translation
from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.multiquery_control.run import sha
from diagnostics.overcomplete_acquisition.evaluation import parameter_readouts
from diagnostics.overcomplete_use_precision.data import build_validation
from diagnostics.shared_code_calibration.run import Config as CalibrationConfig
from diagnostics.shared_code_calibration.run import run as run_calibration
from diagnostics.shared_confirmatory_control import probes
from diagnostics.shared_confirmatory_control import run as runner
from diagnostics.shared_confirmatory_control.data import build_fresh, independence
from diagnostics.shared_confirmatory_control.source import load_inputs
from diagnostics.table_settling_control.source import model_fingerprint
from scjepa.data.random import generator as parameter_generator
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


def test_frozen_probe_matches_original_selection_and_translation_preserves_predictions() -> None:
    generator = torch.Generator().manual_seed(1379)
    codes: dict[str, Tensor] = {}
    splits: dict[str, Any] = {}
    for name, groups in (("calibration", 20), ("validation", 9), ("test", 7)):
        theta = torch.randn(groups, 6, generator=generator, dtype=torch.float64)
        extra = torch.randn(groups, 6, generator=generator, dtype=torch.float64)
        code = torch.cat((theta, extra), -1)[:, None].expand(-1, 10, -1).clone()
        code += 0.03 * torch.randn(groups, 10, 12, generator=generator, dtype=torch.float64)
        code[..., 1] += 0.5 * code[..., 6].square()
        code[..., 2] = torch.tanh(code[..., 2])
        codes[name] = code
        splits[name] = {"theta": theta}
    original = parameter_readouts(codes, splits)
    frozen = probes.fit_probes(
        codes["calibration"],
        codes["validation"],
        splits["calibration"]["theta"],
        splits["validation"]["theta"],
    )
    assert frozen["selection_frozen_before_fresh_generation"] is True
    assert [row["family"] for row in frozen["selected"]] == original[
        "selected_methods_per_parameter"
    ]
    for factor, row in enumerate(frozen["selected"]):
        expected = original["families"][row["family"]]
        hyperparameters = expected["selected_hyperparameters_per_parameter"][factor]
        assert row["ridge"] == hyperparameters["ridge"]
        if "width" in hyperparameters:
            assert row["width"] == hyperparameters["width"]
        assert row["validation_r2"] == expected["validation_r2"][factor]
    expected = probes.predict(frozen, codes["test"])
    scored, errors = probes.score(expected, splits["test"]["theta"])
    torch.testing.assert_close(
        torch.tensor(scored["r2"]),
        torch.tensor(original["selected_test_r2"]),
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        errors.float(),
        torch.tensor(original["selected_test_group_mse_by_parameter"], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    assert errors.shape == (7, 6)
    assert not torch.equal(expected[:, 0], expected[:, 1])
    shift = torch.randn(12, generator=generator, dtype=torch.float64)
    actual = probes.predict(frozen, codes["test"] + shift, translation=shift)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    state = [cast(Tensor, row["weights"]).clone() for row in frozen["selected"]]
    # Fresh outcomes can change reported R², never the selected frozen readout.
    changed, _ = probes.score(expected, -splits["test"]["theta"])
    assert changed["r2"] != scored["r2"]
    for before, row in zip(state, frozen["selected"], strict=True):
        torch.testing.assert_close(before, row["weights"], rtol=0, atol=0)


@pytest.fixture(scope="module")
def confirmed_sources(
    shared_sources: tuple[Path, Path, dict[int, Path]],  # noqa: F811
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, tuple[Path, Path]]:
    """Produce both accepted calibrated sources from genuine tiny training trajectories."""
    source, precision, endpoints = shared_sources
    root = tmp_path_factory.mktemp("confirmation-sources")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        roots = (root / "first", root / "second")
        for seed, out in zip((7, 17), roots, strict=True):
            run_calibration(
                CalibrationConfig(
                    str(endpoints[seed]),
                    str(source),
                    str(precision),
                    str(out),
                    seed=seed,
                    chunk_size=2,
                    max_iterations=2,
                    max_evaluations=6,
                    verification=True,
                )
            )
        return source, precision, roots
    finally:
        torch.set_num_threads(previous)


def test_four_frozen_sources_load_without_train_arrays_and_preserve_exact_translation(
    confirmed_sources: tuple[Path, Path, tuple[Path, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, precision, roots = confirmed_sources
    original = cast(Callable[..., Any], torch.load)
    seen: list[str] = []

    def guarded(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            name = Path(file).name
            seen.append(name)
            assert name not in {
                "train_inputs.pt",
                "train_codes.pt",
                "audit_inputs.pt",
                "fit_state.pt",
            }
        return original(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded)
    inputs = load_inputs(roots, source, precision, verification=True)
    assert tuple(inputs.models) == runner.ARMS
    assert inputs.train_config["verification_fixture_missing_train_identity"] is True
    assert "evaluation_data.pt" in seen
    for prefix in ("seed1", "seed2"):
        first, second = inputs.models[prefix + "/source"], inputs.models[prefix + "/calibrated"]
        expected = compose_translation(first, inputs.translations[prefix])
        assert model_fingerprint(expected) == model_fingerprint(second)
        for model in (first, second):
            assert not model.training
            assert next(model.parameters()).dtype == torch.float64
            assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
    assert all(sha(Path(name)) == value for name, value in inputs.hashes.items())


def test_fresh_test_draw_and_exact_group_identity_exclusions(
    confirmed_sources: tuple[Path, Path, tuple[Path, Path]],
) -> None:
    source, precision, roots = confirmed_sources
    inputs = load_inputs(roots, source, precision, verification=True)
    model = inputs.models["seed1/source"]
    initial = model_fingerprint(model)
    rng = torch.get_rng_state()
    fresh = build_fresh(model, inputs.old_data["metadata"], groups=4, seed=84659, batch_size=2)
    assert torch.equal(rng, torch.get_rng_state())
    assert model_fingerprint(model) == initial
    assert set(fresh) == {"metadata", "test"}
    assert fresh["metadata"]["role"] == "confirmatory_test"
    reference = build_validation(
        model, inputs.old_data["metadata"], groups=4, seed=84659, batch_size=3
    )
    torch.testing.assert_close(fresh["test"], reference["validation"], rtol=0, atol=0)
    train_config = {
        "family": "simplified_gaussian",
        "split": "train",
        "root_seed": 987,
        "num_episodes": 3,
    }
    report = independence(fresh, inputs.old_data, inputs.precision_data, train_config)
    assert report["exact_train_overlaps"] == 0
    assert report["training_parameter_groups"] == 3
    assert report["exact_precision_overlaps"] == 0
    for name in ("calibration", "validation", "test"):
        changed = copy.deepcopy(fresh)
        changed["test"]["theta"][0] = inputs.old_data[name]["theta"][0]
        with pytest.raises(ValueError, match="overlaps archived"):
            independence(changed, inputs.old_data, inputs.precision_data, train_config)
    changed = copy.deepcopy(fresh)
    changed["test"]["theta"][0] = inputs.precision_data["validation"]["theta"][0]
    with pytest.raises(ValueError, match="precision validation"):
        independence(changed, inputs.old_data, inputs.precision_data, train_config)
    changed = copy.deepcopy(fresh)
    stream = parameter_generator(987, "train", 1, "simplified_gaussian/parameters")
    changed["test"]["theta"][0] = 0.75 * (
        2 * torch.rand(6, generator=stream, dtype=torch.float64) - 1
    )
    with pytest.raises(ValueError, match="overlaps TRAIN"):
        independence(changed, inputs.old_data, inputs.precision_data, train_config)
    changed = copy.deepcopy(fresh)
    changed["metadata"]["seed"] = inputs.precision_data["metadata"]["seed"]
    with pytest.raises(ValueError, match="precision-validation seed"):
        independence(changed, inputs.old_data, inputs.precision_data, train_config)
    changed = copy.deepcopy(fresh)
    changed["metadata"]["role"] = "validation"
    with pytest.raises(ValueError, match="confirmatory"):
        independence(changed, inputs.old_data, inputs.precision_data, train_config)


def test_confirmatory_runner_freezes_probes_first_evaluates_all_four_and_resumes_exactly(
    confirmed_sources: tuple[Path, Path, tuple[Path, Path]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, precision, roots = confirmed_sources
    active_output = tmp_path / "full"
    config = runner.Config(
        str(roots[0]),
        str(roots[1]),
        str(source),
        str(precision),
        str(active_output),
        groups=4,
        seed=84659,
        bootstrap_samples=8,
        evaluation_seed=7,
        chunk_size=2,
        verification=True,
    )
    fit, build, evaluate = runner.fit_probes, runner.build_fresh, runner.evaluate
    fits, draws, calls = 0, 0, 0
    fresh_started = False
    interruptions = False

    def checked_fit(
        cal: Tensor, val: Tensor, cal_theta: Tensor, val_theta: Tensor
    ) -> dict[str, Any]:
        nonlocal fits
        assert not fresh_started, "Probe fitting after fresh-data generation"
        assert len(cal) == len(val) == 4
        fits += 1
        return fit(cal, val, cal_theta, val_theta)

    def checked_build(
        model: SCJEPA,
        metadata: dict[str, Any],
        *,
        groups: int,
        seed: int,
        batch_size: int,
        progress: Callable[[int], None] | None = None,
    ) -> EvaluationData:
        nonlocal draws, fresh_started
        assert fits == 2
        assert (active_output / "probes.pt").exists()
        assert (active_output / "probes.receipt.json").exists()
        assert (groups, seed) == (4, 84659)
        fresh_started = True
        draws += 1
        return build(
            model, metadata, groups=groups, seed=seed, batch_size=batch_size, progress=progress
        )

    def checked_evaluate(
        model: SCJEPA,
        data: EvaluationData,
        *,
        chunk_size: int,
        bootstrap_samples: int,
        seed: int,
        progress: Callable[[int], None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Tensor], dict[str, Tensor]]:
        nonlocal calls
        calls += 1
        assert fresh_started
        assert data["metadata"]["role"] == "confirmatory_test"
        assert data["metadata"]["seed"] == 84659
        assert len(data["validation"]["theta"]) == 4
        assert seed == 7
        assert not model.training
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        if interruptions and calls == 3:
            raise RuntimeError("interrupt between fixed arm evaluations")
        return evaluate(
            model,
            data,
            chunk_size=chunk_size,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            progress=progress,
        )

    monkeypatch.setattr(runner, "fit_probes", checked_fit)
    monkeypatch.setattr(runner, "build_fresh", checked_build)
    monkeypatch.setattr(runner, "evaluate", checked_evaluate)
    full = runner.run(config)
    assert fits == 2
    assert draws == 1
    assert calls == 4
    assert tuple(full["reports"]) == runner.ARMS
    full_probes = sha(active_output / "probes.pt")
    fits, draws, calls = 0, 0, 0
    fresh_started = False
    interruptions = True
    active_output = tmp_path / "resumed"
    resumed_config = replace(config, output_dir=str(active_output))
    with pytest.raises(RuntimeError, match="between fixed arm"):
        runner.run(resumed_config)
    assert fits == 2
    assert draws == 1
    assert calls == 3
    assert not (active_output / "completed.json").exists()
    last_event = cast(dict[str, Any], read_json(active_output / "tracking.json"))["last_event_step"]
    frozen_probe_hash = sha(active_output / "probes.pt")
    fresh_hash = sha(active_output / "fresh_test.pt")
    interruptions = False
    calls = 0
    resumed = runner.run(resumed_config, resume=True)
    assert fits == 2
    assert draws == 1
    assert calls == 2
    assert sha(active_output / "probes.pt") == frozen_probe_hash
    assert sha(active_output / "fresh_test.pt") == fresh_hash
    assert frozen_probe_hash == full_probes
    assert full["reports"] == resumed["reports"]
    assert full["independence"] == resumed["independence"]
    assert (
        cast(dict[str, Any], read_json(active_output / "tracking.json"))["last_event_step"]
        > last_event
    )
    runner.run(resumed_config, resume=True)
    assert calls == 2
    assert draws == 1
    changed = cast(dict[str, Any], torch.load(active_output / "probes.pt", weights_only=False))
    changed["seed1"]["selected"][0]["weights"][0] += 1
    torch.save(changed, active_output / "probes.pt")
    with pytest.raises(ValueError, match="hash|changed|corrupt"):
        runner.run(resumed_config, resume=True)
    with pytest.raises(ValueError, match="fixed"):
        replace(config, verification=False)
    with pytest.raises(ValueError, match="distinct"):
        replace(config, calibration_seed2_dir=config.calibration_seed1_dir)
    with pytest.raises(ValueError, match="disjoint"):
        replace(config, output_dir=str(source / "bad-output"))
