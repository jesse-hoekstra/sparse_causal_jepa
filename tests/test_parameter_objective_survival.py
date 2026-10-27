import copy
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_multiquery_control import make_batch, make_model, objective_config
from test_predictor_transfer_control import transfer_sources as transfer_sources
from torch import Tensor, nn

from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective, cached_forward
from diagnostics.mlp_oracle_code_control.model import Prediction, initialize_mlp
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiquery_control.run import sha
from diagnostics.parameter_objective_survival import run as runner
from diagnostics.parameter_objective_survival.forward import Arm, training_loss
from diagnostics.parameter_objective_survival.source import load_inputs, verify_sources
from diagnostics.predictor_transfer_control.run import Config as TransferConfig
from diagnostics.predictor_transfer_control.run import run as run_transfer
from scjepa.models.model import SCJEPA
from scjepa.training.distributed import GlobalBatchSampler


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("arm", ["forecast", "parameter_objective"])
def test_loss_only_difference_preserves_auxiliary_draws_and_both_attached_donor_paths(
    arm: Arm,
) -> None:
    model = make_model()
    initialize_mlp(model, 17)
    model.parameter_encoder.requires_grad_(True)
    model.predictor.requires_grad_(True)
    _, batch = make_batch(model)
    outputs: list[Tensor] = []
    donors: list[Tensor] = []
    histories: list[Tensor] = []
    frozen = [module_fingerprint(x) for x in (model.online_encoder, model.target_encoder)]

    def capture(_module: nn.Module, args: tuple[object, ...], output: Prediction) -> None:
        output.prediction.retain_grad()
        outputs.append(output.prediction)
        assert isinstance(args[0], Tensor)
        assert isinstance(args[1], Tensor)
        histories.append(args[0])
        donors.append(args[1])

    hook = model.predictor.register_forward_hook(capture)
    torch.default_generator.manual_seed(531)
    try:
        loss, terms = training_loss(model, batch, AcquisitionObjective(objective_config()), arm)
    finally:
        hook.remove()
    rng = torch.get_rng_state()
    assert len(donors) == 2
    assert donors[0] is donors[1]
    assert donors[0].requires_grad
    torch.testing.assert_close(histories[1], outputs[0][:, None], rtol=0, atol=0)
    derivative = torch.autograd.grad(outputs[1].square().mean(), outputs[0], retain_graph=True)[0]
    assert float(derivative.square().sum()) > 0
    for output in outputs:
        gradients = torch.autograd.grad(
            output.square().mean(), list(model.parameter_encoder.parameters()), retain_graph=True
        )
        assert sum(float(x.square().sum()) for x in gradients) > 0
    actual_gradients = torch.autograd.grad(
        loss, [*model.parameter_encoder.parameters(), *model.predictor.parameters()]
    )
    torch.default_generator.manual_seed(531)
    full, expected_terms = AcquisitionObjective(objective_config())(cached_forward(model, batch))
    assert torch.equal(torch.get_rng_state(), rng)
    expected = expected_terms["prediction"] if arm == "forecast" else full
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        full,
        expected_terms["prediction"]
        + 0.4 * expected_terms["parameter_regularizer"]
        + expected_terms["parameter_consistency"],
        rtol=0,
        atol=0,
    )
    expected_gradients = torch.autograd.grad(
        expected, [*model.parameter_encoder.parameters(), *model.predictor.parameters()]
    )
    for first, second in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    for name in ("prediction", "parameter_regularizer", "parameter_consistency"):
        torch.testing.assert_close(terms[name], expected_terms[name], rtol=0, atol=0)
    assert frozen == [module_fingerprint(x) for x in (model.online_encoder, model.target_encoder)]
    assert all(
        p.grad is None for x in (model.online_encoder, model.target_encoder) for p in x.parameters()
    )
    with pytest.raises(ValueError, match="exclude physical"):
        training_loss(
            model,
            {**batch, "theta": torch.zeros(2, 6)},
            AcquisitionObjective(objective_config()),
            arm,
        )
    model.online_encoder.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        training_loss(model, batch, AcquisitionObjective(objective_config()), arm)
    model.online_encoder.requires_grad_(False)
    model.parameter_encoder.requires_grad_(False)
    with pytest.raises(ValueError, match="must train"):
        training_loss(model, batch, AcquisitionObjective(objective_config()), arm)


@pytest.fixture(scope="module")
def survival_sources(
    transfer_sources: tuple[Path, Path, Path],  # noqa: F811
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    source, endpoint, audit = transfer_sources
    root = tmp_path_factory.mktemp("parameter-survival") / "parent"
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        run_transfer(
            TransferConfig(
                str(source),
                str(endpoint),
                str(audit),
                str(root),
                "mlp",
                seed=7,
                steps=2,
                groups=2,
                bootstrap_samples=8,
                verification=True,
            )
        )
    finally:
        torch.set_num_threads(previous)
    return source, root


def test_source_restores_exact_trained_six_code_weights_and_preserves_original_artifacts(
    survival_sources: tuple[Path, Path],
) -> None:
    source, endpoint = survival_sources
    inputs = load_inputs(source, endpoint, device=torch.device("cpu"), verification=True)
    saved = cast(dict[str, Any], torch.load(endpoint / "last.pt", weights_only=False))
    assert saved["optimizer"]["state"]
    torch.testing.assert_close(inputs.model.state_dict(), saved["model"], rtol=0, atol=0)
    assert inputs.provenance["source_step"] == 2
    assert inputs.model.parameter_encoder.param_dim == 6
    assert inputs.model.predictor.param_dim == 6
    assert set(inputs.bundle.train) == {"online", "target", "actions"}
    verify_sources(inputs)
    assert all(sha(Path(name)) == value for name, value in inputs.hashes.items())
    with torch.no_grad():
        next(inputs.model.target_encoder.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="state chart changed"):
        verify_sources(inputs)


@pytest.mark.parametrize("arm", ["forecast", "parameter_objective"])
def test_survival_uses_fresh_ep_adam_continued_draws_frozen_states_and_exact_resume(
    survival_sources: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: Arm,
) -> None:
    source, endpoint = survival_sources
    inputs = load_inputs(source, endpoint, device=torch.device("cpu"), verification=True)
    initial = copy.deepcopy(inputs.model.state_dict())
    fixed = {
        name: module_fingerprint(getattr(inputs.model, name))
        for name in ("online_encoder", "target_encoder")
    }
    parent_step = int(inputs.provenance["source_step"])
    active = tmp_path / "full"
    config = runner.Config(
        str(source),
        str(endpoint),
        str(active),
        arm,
        seed=7,
        steps=3,
        groups=2,
        bootstrap_samples=8,
        verification=True,
    )
    loss, assess = runner.training_loss, runner.assess_model
    count, test_calls = 0, 0
    original_load = cast(Callable[..., Any], torch.load)

    def guarded(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            path = Path(file)
            assert path.name not in {"audit_inputs.pt", "fresh_test.pt"}
            assert "shared-confirmatory" not in str(path)
        return original_load(file, *args, **kwargs)

    def no_ema(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Frozen-state survival must not update target EMA")

    def capture(
        model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective, given_arm: Arm
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal count
        count += 1
        assert given_arm == arm
        assert {name: module_fingerprint(getattr(model, name)) for name in fixed} == fixed
        assert all(
            p.requires_grad
            for block in (model.parameter_encoder, model.predictor)
            for p in block.parameters()
        )
        assert all(
            not p.requires_grad
            for block in (model.online_encoder, model.target_encoder)
            for p in block.parameters()
        )
        assert set(batch) == {"online", "target", "actions"}
        global_step = parent_step + count
        epoch, position = divmod(
            global_step - 1, inputs.bundle.train_groups // inputs.bundle.batch_size
        )
        selected = list(
            GlobalBatchSampler(inputs.bundle.train_groups, inputs.bundle.batch_size, 7, epoch)
        )[position]
        torch.testing.assert_close(
            batch,
            {name: value[selected] for name, value in inputs.bundle.train.items()},
            rtol=0,
            atol=0,
        )
        assert torch.equal(
            torch.get_rng_state(),
            torch.Generator().manual_seed(7 * 100000 + global_step).get_state(),
        )
        if count == 1:
            torch.testing.assert_close(model.state_dict(), initial, rtol=0, atol=0)
            saved = cast(dict[str, Any], torch.load(active / "last.pt", weights_only=False))
            assert saved["optimizer"]["state"] == {}
            expected = len([*model.parameter_encoder.parameters(), *model.predictor.parameters()])
            assert len(saved["optimizer"]["param_groups"][0]["params"]) == expected
            assert saved["optimizer"]["param_groups"][0]["lr"] == 0.001
            assert saved["step"] == parent_step
        return loss(model, batch, objective, given_arm)

    def checked_assess(
        model: SCJEPA,
        data: EvaluationData,
        selection: DiagnosticSelection,
        *,
        bootstrap_samples: int,
        seed: int,
        final: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Tensor]]:
        nonlocal test_calls
        if final:
            test_calls += 1
            saved = cast(dict[str, Any], torch.load(active / "last.pt", weights_only=False))
            assert saved["step"] == parent_step + 3
        return assess(
            model, data, selection, bootstrap_samples=bootstrap_samples, seed=seed, final=final
        )

    monkeypatch.setattr(torch, "load", guarded)
    monkeypatch.setattr(SCJEPA, "update_target", no_ema)
    monkeypatch.setattr(runner, "training_loss", capture)
    monkeypatch.setattr(runner, "assess_model", checked_assess)
    full = runner.run(config)
    assert count == 3
    assert test_calls == 1
    final = cast(dict[str, Any], torch.load(active / "last.pt", weights_only=False))
    assert all(int(x["step"]) == 3 for x in final["optimizer"]["state"].values())
    for prefix in ("parameter_encoder.", "predictor."):
        assert any(
            not torch.equal(value, final["model"][name])
            for name, value in initial.items()
            if name.startswith(prefix)
        )
    for name, value in initial.items():
        if name.startswith(("online_encoder.", "target_encoder.")):
            torch.testing.assert_close(value, final["model"][name], rtol=0, atol=0)
    count, test_calls = 0, 0
    active = tmp_path / "resumed"
    resumed_config = replace(config, output_dir=str(active))

    def interrupt(
        model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective, given_arm: Arm
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("survival operational interruption")
        return loss(model, batch, objective, given_arm)

    monkeypatch.setattr(runner, "training_loss", interrupt)
    with pytest.raises(RuntimeError, match="operational interruption"):
        runner.run(resumed_config)
    assert test_calls == 0
    assert not (active / "final_test.json").exists()
    monkeypatch.setattr(runner, "training_loss", loss)
    resumed = runner.run(resumed_config, resume=True)
    assert test_calls == 1
    replay = cast(dict[str, Any], torch.load(active / "last.pt", weights_only=False))
    for name in ("model", "optimizer"):
        torch.testing.assert_close(final[name], replay[name], rtol=0, atol=0)
    torch.testing.assert_close(final["rng"]["torch"], replay["rng"]["torch"], rtol=0, atol=0)
    assert full["records"] == resumed["records"]
    assert full["final_test"] == resumed["final_test"]
    runner.run(resumed_config, resume=True)
    assert test_calls == 1
    assert all(sha(Path(name)) == value for name, value in inputs.hashes.items())
    with pytest.raises(ValueError, match="overlap|disjoint|separate"):
        runner.run(replace(config, output_dir=str(endpoint / "bad-output")))
    changed = cast(dict[str, Any], torch.load(active / "initial_modules.pt", weights_only=False))
    changed["parameter_encoder"]["head.4.bias"][0] += 0.1
    torch.save(changed, active / "initial_modules.pt")
    with pytest.raises(ValueError, match="initialization weights changed"):
        runner.run(resumed_config, resume=True)
