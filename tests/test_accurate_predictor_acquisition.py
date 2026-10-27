from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_table_settling_control import fixture_sources
from torch import Tensor

from diagnostics.accurate_predictor_acquisition import run as runner
from diagnostics.accurate_predictor_acquisition.run import Config, fixed_hashes, run
from diagnostics.accurate_predictor_acquisition.source import load_inputs, verify_sources
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective, cached_forward
from diagnostics.mlp_oracle_code_control.model import Prediction
from diagnostics.mlp_oracle_code_control.run import run as oracle_run
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiquery_control.run import sha
from diagnostics.oracle_code_control.run import Config as OracleConfig
from scjepa.experiments.storage import read_json, write_json
from scjepa.models.model import SCJEPA


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def source_roots(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Use a genuinely trained tiny exact-code oracle with authenticated source ancestry."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        root = tmp_path_factory.mktemp("accurate-predictor")
        curriculum, audit = fixture_sources(root, fresh=True)
        source = Path(
            cast(dict[str, Any], read_json(curriculum / "protocol.json"))["config"]["source_dir"]
        )
        oracle = root / "oracle"
        oracle_run(
            OracleConfig(
                str(source),
                str(audit),
                str(oracle),
                seed=7,
                steps=3,
                groups=2,
                bootstrap_samples=16,
                verification=True,
            )
        )
        return source, oracle
    finally:
        torch.set_num_threads(previous)


def test_source_restores_fresh_encoder_and_accurate_predictor_without_loading_training_theta(
    source_roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, oracle = source_roots
    original_load = cast(Callable[..., Any], torch.load)
    loaded: list[str] = []

    def guarded_load(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            loaded.append(Path(file).name)
            assert Path(file).name != "audit_inputs.pt", (
                "E-training source loaded oracle theta labels"
            )
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded_load)
    inputs = load_inputs(source, oracle, 7, verification=True)
    initial = cast(dict[str, Any], torch.load(source / "initial_seed_7.pt", weights_only=False))
    endpoint = cast(dict[str, Any], torch.load(oracle / "last.pt", weights_only=False))
    for name, value in inputs.model.parameter_encoder.state_dict().items():
        torch.testing.assert_close(
            value, initial["model"]["parameter_encoder." + name], rtol=0, atol=0
        )
    for name, value in inputs.model.predictor.state_dict().items():
        torch.testing.assert_close(value, endpoint["model"]["predictor." + name], rtol=0, atol=0)
    assert set(inputs.bundle.train) == {"online", "target", "actions"}
    assert "audit_inputs.pt" not in loaded
    assert all(not parameter.requires_grad for parameter in inputs.model.parameters())
    verify_sources(inputs)


def test_frozen_predictor_transmits_both_attached_steps_into_encoder(
    source_roots: tuple[Path, Path],
) -> None:
    inputs = load_inputs(*source_roots, 7, verification=True)
    model = inputs.model.train()
    model.parameter_encoder.requires_grad_(True)
    before = fixed_hashes(model)
    batch = {key: value[:2] for key, value in inputs.bundle.train.items()}
    outputs: list[Tensor] = []
    arguments: list[tuple[Tensor, Tensor, Tensor]] = []

    def capture(
        _module: torch.nn.Module, args: tuple[Tensor, Tensor, Tensor], output: Prediction
    ) -> None:
        arguments.append(args)
        output.prediction.retain_grad()
        outputs.append(output.prediction)

    handle = model.predictor.register_forward_hook(capture)
    try:
        output = cached_forward(model, batch)
    finally:
        handle.remove()
    loss = (output.predictions[:, :, 1] - output.targets[:, :, 1]).square().mean()
    params = tuple(model.parameter_encoder.parameters())
    actual = torch.autograd.grad(loss, params, retain_graph=True)
    assert outputs[0].grad is not None
    assert float(outputs[0].grad.square().sum()) > 0
    detached_second = model.predictor(
        outputs[0].detach()[:, None], arguments[1][1], arguments[1][2]
    ).prediction
    detached_loss = (detached_second.reshape(2, 10, 6) - output.targets[:, :, 1]).square().mean()
    detached_gradients = torch.autograd.grad(detached_loss, params, retain_graph=True)
    assert sum(float(value.square().sum()) for value in actual) > 0
    assert (
        sum(
            float((first - second).square().sum())
            for first, second in zip(actual, detached_gradients, strict=True)
        )
        > 1e-16
    )
    optimizer = torch.optim.Adam(params, lr=0.001)
    initial_encoder = module_fingerprint(model.parameter_encoder)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    assert module_fingerprint(model.parameter_encoder) != initial_encoder
    assert fixed_hashes(model) == before
    assert all(parameter.grad is None for parameter in model.predictor.parameters())
    assert all(not parameter.requires_grad for parameter in model.predictor.parameters())
    assert all(value.grad is None for value in batch.values())
    verify_sources(inputs)


def test_source_rejects_changed_chart_and_frozen_predictor_weights(
    source_roots: tuple[Path, Path],
) -> None:
    inputs = load_inputs(*source_roots, 7, verification=True)
    with torch.no_grad():
        next(inputs.model.predictor.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="[Ff]rozen|[Pp]redictor"):
        verify_sources(inputs)
    root = source_roots[1]
    path, completion_path = root / "last.pt", root / "completed.json"
    original, original_completion = path.read_bytes(), completion_path.read_bytes()
    try:
        payload = cast(dict[str, Any], torch.load(path, weights_only=False))
        key = next(key for key in payload["model"] if key.startswith("online_encoder."))
        payload["model"][key] += 0.01
        torch.save(payload, path)
        completion = cast(dict[str, Any], read_json(completion_path))
        completion["hashes"]["last.pt"] = sha(path)
        write_json(completion_path, completion)
        with pytest.raises(ValueError, match="[Ss]tate|[Cc]hart|[Ff]rozen"):
            load_inputs(*source_roots, 7, verification=True)
    finally:
        path.write_bytes(original)
        completion_path.write_bytes(original_completion)


def test_encoder_only_training_resumes_exactly_with_fixed_final_test(
    source_roots: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, oracle = source_roots
    inputs = load_inputs(source, oracle, 7, verification=True)
    frozen = fixed_hashes(inputs.model)
    config = Config(
        str(source),
        str(oracle),
        str(tmp_path / "full"),
        seed=7,
        steps=3,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    original_assess = runner.assess_model
    calls: list[bool] = []

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
        assert fixed_hashes(model) == frozen
        guarded = data if final else {key: value for key, value in data.items() if key != "test"}
        return original_assess(
            model, guarded, selection, bootstrap_samples=bootstrap_samples, seed=seed, final=final
        )

    monkeypatch.setattr(runner, "assess_model", assessment)
    full = run(config)
    assert calls == [False] * 4 + [True]
    checkpoint = cast(dict[str, Any], torch.load(tmp_path / "full/last.pt", weights_only=False))
    initial_encoder = module_fingerprint(inputs.model.parameter_encoder)
    inputs.model.load_state_dict(checkpoint["model"])
    assert fixed_hashes(inputs.model) == frozen
    assert module_fingerprint(inputs.model.parameter_encoder) != initial_encoder
    assert len(checkpoint["optimizer"]["param_groups"][0]["params"]) == len(
        tuple(inputs.model.parameter_encoder.parameters())
    )
    original_loss = runner.training_loss
    updates = 0

    def interrupted(
        model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal updates
        updates += 1
        assert set(batch) == {"online", "target", "actions"}
        assert all(not parameter.requires_grad for parameter in model.predictor.parameters())
        if updates == 2:
            raise RuntimeError("interrupted E training")
        with pytest.raises(ValueError, match="exclude physical labels"):
            original_loss(model, {**batch, "theta": torch.zeros(2, 6)}, objective)
        return original_loss(model, batch, objective)

    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    monkeypatch.setattr(runner, "training_loss", interrupted)
    calls.clear()
    with pytest.raises(RuntimeError, match="interrupted E training"):
        run(resumed_config)
    assert not any(calls)
    assert not (tmp_path / "resumed/final_test.json").exists()
    monkeypatch.setattr(runner, "training_loss", original_loss)
    resumed = run(resumed_config, resume=True)
    final = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    for name in ("model", "optimizer"):
        torch.testing.assert_close(checkpoint[name], final[name], rtol=0, atol=0)
    torch.testing.assert_close(checkpoint["rng"]["torch"], final["rng"]["torch"], rtol=0, atol=0)
    assert full["records"] == resumed["records"]
    assert full["final_test"] == resumed["final_test"]
    assert all(sha(Path(path)) == expected for path, expected in inputs.hashes.items())
    before = len(calls)
    assert run(resumed_config, resume=True) == resumed
    assert len(calls) == before
