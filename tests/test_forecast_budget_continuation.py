import copy
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import pytest
import torch
from test_group_latent_runner import source_with_initialization
from torch import Tensor

from diagnostics.encoder_gain_acquisition import run as gain_runner
from diagnostics.forecast_budget_continuation import run as runner
from diagnostics.forecast_budget_continuation.run import Config, run
from diagnostics.forecast_budget_continuation.source import load_inputs, verify_sources
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.mlp_forecast_only_acquisition import run as original_runner
from diagnostics.mlp_forecast_only_acquisition.run import fixed_hashes, training_loss
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit
from diagnostics.multiquery_control.run import sha
from scjepa.experiments.storage import read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.loop import ObjectiveConfig


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module", params=["baseline", "gain"])
def endpoints(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, Path, Literal["baseline", "gain"]]:
    root = tmp_path_factory.mktemp("forecast-budget-" + str(request.param))
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        source = source_with_initialization(root / "source")
        for steps in (2, 3):
            output = root / str(steps)
            if request.param == "baseline":
                original_runner.run(
                    original_runner.Config(
                        str(source),
                        str(output),
                        seed=7,
                        steps=steps,
                        groups=2,
                        bootstrap_samples=16,
                        verification=True,
                    )
                )
            else:
                gain_runner.run(
                    gain_runner.Config(
                        str(source),
                        str(output),
                        seed=7,
                        steps=steps,
                        groups=2,
                        bootstrap_samples=16,
                        verification=True,
                    )
                )
        return root / "2", root / "3", cast(Literal["baseline", "gain"], request.param)
    finally:
        torch.set_num_threads(old)


def checkpoint(root: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any], torch.load(root / "last.pt", map_location="cpu", weights_only=False)
    )


def test_retained_adam_first_update_matches_uninterrupted_original_run_not_fresh_adam(
    endpoints: tuple[Path, Path, Literal["baseline", "gain"]],
    tmp_path: Path,
) -> None:
    source, uninterrupted, arm = endpoints
    inputs = load_inputs(source, arm, verification=True)
    saved = checkpoint(source)
    torch.testing.assert_close(inputs.model.state_dict(), saved["model"], rtol=0, atol=0)
    config = Config(
        str(source),
        str(tmp_path / "continued"),
        source_arm=arm,
        seed=7,
        steps=1,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    result = run(config)
    actual, expected = checkpoint(tmp_path / "continued"), checkpoint(uninterrupted)
    assert actual["step"] == 3
    assert actual["updates"] == 1
    for name in ("model", "optimizer"):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    torch.testing.assert_close(actual["rng"]["torch"], expected["rng"]["torch"], rtol=0, atol=0)
    assert (
        result["final_test"]
        == cast(dict[str, Any], read_json(uninterrupted / "results.json"))["final_test"]
    )
    # Independently reset only Adam at the exact same source weights and next global batch.
    fresh = copy.deepcopy(inputs.model)
    fresh.requires_grad_(False)
    fresh.parameter_encoder.requires_grad_(True)
    fresh.predictor.requires_grad_(True)
    parameters = [parameter for parameter in fresh.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    epoch, position = divmod(2, inputs.bundle.train_groups // inputs.bundle.batch_size)
    selected = list(
        GlobalBatchSampler(inputs.bundle.train_groups, inputs.bundle.batch_size, 7, epoch)
    )[position]
    batch = {name: value[selected] for name, value in inputs.bundle.train.items()}
    torch.default_generator.manual_seed(700003)
    fresh.train()
    fresh.online_encoder.eval()
    fresh.target_encoder.eval()
    objective = AcquisitionObjective(ObjectiveConfig(**inputs.bundle.protocol["objective_config"]))
    loss, _ = training_loss(fresh, batch, objective)
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    assert any(
        not torch.equal(value, actual["model"][name])
        for name, value in fresh.state_dict().items()
        if isinstance(value, Tensor)
    )
    assert all(int(value["step"]) == 3 for value in actual["optimizer"]["state"].values())
    assert all(int(value["step"]) == 1 for value in optimizer.state_dict()["state"].values())


def test_budget_continuation_global_batches_frozen_states_source_and_exact_resume(
    endpoints: tuple[Path, Path, Literal["baseline", "gain"]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, arm = endpoints
    inputs = load_inputs(source, arm, verification=True)
    frozen = fixed_hashes(inputs.model)
    config = Config(
        str(source),
        str(tmp_path / "full"),
        source_arm=arm,
        seed=7,
        steps=3,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    original_load = cast(Callable[..., Any], torch.load)

    def guarded_load(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            assert Path(file).name != "audit_inputs.pt"
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded_load)
    updates = 0

    def capture(
        model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal updates
        global_step = 3 + updates
        epoch, position = divmod(
            global_step - 1, inputs.bundle.train_groups // inputs.bundle.batch_size
        )
        selected = list(
            GlobalBatchSampler(inputs.bundle.train_groups, inputs.bundle.batch_size, 7, epoch)
        )[position]
        assert set(batch) == {"online", "target", "actions"}
        for name, value in batch.items():
            torch.testing.assert_close(value, inputs.bundle.train[name][selected], rtol=0, atol=0)
        assert torch.equal(
            torch.get_rng_state(), torch.Generator().manual_seed(700000 + global_step).get_state()
        )
        assert fixed_hashes(model) == frozen
        assert all(parameter.requires_grad for parameter in model.parameter_encoder.parameters())
        assert all(parameter.requires_grad for parameter in model.predictor.parameters())
        updates += 1
        return training_loss(model, batch, objective)

    original_assess = runner.assess_model
    calls: list[bool] = []

    def assess(
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
        guarded = data if final else {name: value for name, value in data.items() if name != "test"}
        return original_assess(
            model, guarded, selection, bootstrap_samples=bootstrap_samples, seed=seed, final=final
        )

    monkeypatch.setattr(runner, "training_loss", capture)
    monkeypatch.setattr(runner, "assess_model", assess)
    full = run(config)
    assert updates == 3
    assert calls == [False] * 4 + [True]
    final = checkpoint(tmp_path / "full")
    assert final["step"] == 5
    assert final["updates"] == 3
    assert [row["step"] for row in full["records"]] == [2, 3, 4, 5]
    attempts = 0

    def interrupt(
        model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("interrupt budget continuation")
        return training_loss(model, batch, objective)

    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    monkeypatch.setattr(runner, "training_loss", interrupt)
    calls.clear()
    with pytest.raises(RuntimeError, match="interrupt budget continuation"):
        run(resumed_config)
    assert not any(calls)
    assert not (tmp_path / "resumed/final_test.json").exists()
    partial = checkpoint(tmp_path / "resumed")
    assert partial["step"] == 3
    assert partial["updates"] == 1
    monkeypatch.setattr(runner, "training_loss", training_loss)
    resumed = run(resumed_config, resume=True)
    replayed = checkpoint(tmp_path / "resumed")
    for name in ("model", "optimizer"):
        torch.testing.assert_close(final[name], replayed[name], rtol=0, atol=0)
    assert torch.equal(final["rng"]["torch"], replayed["rng"]["torch"])
    assert full["records"] == resumed["records"]
    assert full["final_test"] == resumed["final_test"]
    verify_sources(inputs)
    assert all(sha(Path(name)) == digest for name, digest in inputs.hashes.items())
    count = len(calls)
    assert run(resumed_config, resume=True) == resumed
    assert len(calls) == count


def test_source_identity_freezes_ancestors_and_prevents_output_overlap(
    endpoints: tuple[Path, Path, Literal["baseline", "gain"]],
) -> None:
    source, _, arm = endpoints
    inputs = load_inputs(source, arm, verification=True)
    saved = checkpoint(source)
    torch.testing.assert_close(inputs.optimizer.state_dict(), saved["optimizer"], rtol=0, atol=0)
    torch.testing.assert_close(inputs.rng["torch"], saved["rng"]["torch"], rtol=0, atol=0)
    path = source / "last.pt"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"tampered")
        with pytest.raises(ValueError, match="[Ii]mmutable|[Ss]ource|[Aa]rtifact|[Hh]ash"):
            verify_sources(inputs)
    finally:
        path.write_bytes(original)
    other: Literal["baseline", "gain"] = "gain" if arm == "baseline" else "baseline"
    with pytest.raises(ValueError, match="[Aa]rm|[Gg]ain|[Aa]ncestry"):
        load_inputs(source, other, verification=True)
    output = inputs.bundle.source_dir / "unwanted-budget-continuation"
    config = Config(
        str(source),
        str(output),
        source_arm=arm,
        seed=7,
        steps=1,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    with pytest.raises(ValueError, match="[Ii]mmutable|[Ss]ource|[Aa]rchive"):
        run(config)
    assert not output.exists()


def test_authenticated_but_reset_adam_history_is_rejected(
    endpoints: tuple[Path, Path, Literal["baseline", "gain"]],
) -> None:
    source, _, arm = endpoints
    checkpoint_path, completion_path = source / "last.pt", source / "completed.json"
    original_checkpoint, original_completion = (
        checkpoint_path.read_bytes(),
        completion_path.read_bytes(),
    )
    try:
        saved = checkpoint(source)
        first_state = next(iter(saved["optimizer"]["state"].values()))
        first_state["step"].zero_()
        torch.save(saved, checkpoint_path)
        completion = cast(dict[str, Any], read_json(completion_path))
        completion["hashes"]["last.pt"] = sha(checkpoint_path)
        write_json(completion_path, completion)
        with pytest.raises(ValueError, match="Adam step"):
            load_inputs(source, arm, verification=True)
    finally:
        checkpoint_path.write_bytes(original_checkpoint)
        completion_path.write_bytes(original_completion)
