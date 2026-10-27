import copy
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from test_group_latent_runner import source_with_initialization
from torch import Tensor

from diagnostics.encoder_gain_acquisition.model import apply_encoder_gains
from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.group_latent_control.run import initialize_control
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.mlp_oracle_code_control.model import initialize_mlp
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiquery_control.run import sha
from diagnostics.overcomplete_acquisition.model import initialize_overcomplete
from diagnostics.overcomplete_replication import run as runner
from diagnostics.overcomplete_replication import source as source_module
from diagnostics.overcomplete_replication.run import Config, fixed_hashes, run
from diagnostics.overcomplete_replication.source import load_inputs
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.factory import build_model
from scjepa.training.rng import preserve_rng

Arm = Literal["gain6", "gain12"]


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def source_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return source_with_initialization(tmp_path_factory.mktemp("replication-source") / "source")
    finally:
        torch.set_num_threads(previous)


def test_new_seed_replays_full_factory_and_preserves_common_arm_weights(source_path: Path) -> None:
    bundle = load_primary_source(source_path, 7, expected_step=2, expected_checkpoint_sha256=None)
    expected, _ = build_source_model(bundle, torch.device("cpu"))
    _, _, hashes = initialize_control(expected, bundle, "fresh_zero")
    archived_encoder = module_fingerprint(expected.parameter_encoder)
    frozen = fixed_hashes(expected)
    model_config = cast(DictConfig, OmegaConf.create(bundle.protocol["model_config"]))
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(7)
        replay = build_model(model_config)
        torch.default_generator.manual_seed(17)
        fresh = build_model(model_config)
    torch.testing.assert_close(
        replay.parameter_encoder.state_dict(),
        expected.parameter_encoder.state_dict(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        replay.predictor.state_dict(), expected.predictor.state_dict(), rtol=0, atol=0
    )
    assert module_fingerprint(fresh.parameter_encoder) != archived_encoder
    expected.parameter_encoder.load_state_dict(fresh.parameter_encoder.state_dict())
    expected_mlp = initialize_mlp(expected, 17)
    expected_gain = apply_encoder_gains(expected)
    rng = torch.get_rng_state()
    six = load_inputs(source_path, 7, 17, arm="gain6", verification=True)
    assert torch.equal(rng, torch.get_rng_state())
    twelve = load_inputs(source_path, 7, 17, arm="gain12", verification=True)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(six.model.state_dict(), expected.state_dict(), rtol=0, atol=0)
    receipt = six.replication_initialization
    assert receipt["raw_encoder_sha256"] == module_fingerprint(fresh.parameter_encoder)
    assert receipt["mlp_initialization"] == expected_mlp
    assert receipt["encoder_gain_initialization"] == expected_gain
    assert receipt["overcomplete_initialization"] is None
    expanded = initialize_overcomplete(expected, 17)
    torch.testing.assert_close(twelve.model.state_dict(), expected.state_dict(), rtol=0, atol=0)
    assert twelve.replication_initialization["overcomplete_initialization"] == expanded
    assert fixed_hashes(six.model) == fixed_hashes(twelve.model) == frozen
    for name, value in twelve.model.parameter_encoder.state_dict().items():
        common = value[:6] if name in {"head.4.weight", "head.4.bias"} else value
        torch.testing.assert_close(
            common, six.model.parameter_encoder.state_dict()[name], rtol=0, atol=0
        )
    for name, value in twelve.model.predictor.state_dict().items():
        common = value[:, :28] if name == "network.0.weight" else value
        torch.testing.assert_close(common, six.model.predictor.state_dict()[name], rtol=0, atol=0)
    initialize_mlp(replay, 7)
    apply_encoder_gains(replay)
    assert module_fingerprint(replay.parameter_encoder) != module_fingerprint(
        six.model.parameter_encoder
    )
    assert module_fingerprint(replay.predictor) != module_fingerprint(six.model.predictor)
    assert all(sha(source_path / name) == digest for name, digest in hashes.items())


def test_isolated_encoder_seed_is_not_accepted_as_original_factory_policy(
    source_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = source_module.seeded_original_model

    def wrong_factory(config: dict[str, Any], seed: int) -> SCJEPA:
        model = original(config, seed)
        with preserve_rng(torch.device("cpu")):
            torch.default_generator.manual_seed(seed)
            old = model.parameter_encoder
            model.parameter_encoder = TransitionParameterEncoder(6, 6, 16, old.max_history)
        return model

    monkeypatch.setattr(source_module, "seeded_original_model", wrong_factory)
    with pytest.raises(ValueError, match="does not reproduce archived"):
        load_inputs(source_path, 7, 17, arm="gain6", verification=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RNG preservation requires CUDA")
def test_replication_source_preserves_initialized_cpu_and_cuda_rng(source_path: Path) -> None:
    torch.cuda.manual_seed(791)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    load_inputs(source_path, 7, 17, arm="gain12", verification=True)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)


@pytest.mark.parametrize("arm", ["gain6", "gain12"])
def test_replication_new_training_seed_old_evaluation_seed_fresh_adam_and_exact_resume(
    source_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: Arm,
) -> None:
    inputs = load_inputs(source_path, 7, 17, arm=arm, verification=True)
    frozen = fixed_hashes(inputs.model)
    initial_model = copy.deepcopy(inputs.model.state_dict())
    config = Config(
        str(source_path),
        str(tmp_path / "full"),
        arm=arm,
        source_seed=7,
        seed=17,
        evaluation_seed=7,
        steps=3,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    original_load = cast(Callable[..., Any], torch.load)

    def guarded_load(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            assert Path(file).name != "audit_inputs.pt", "Replication loaded training theta"
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded_load)
    original_loss = runner.training_loss
    batches = 0

    def capture(
        model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal batches
        step = batches + 1
        if step == 1:
            torch.testing.assert_close(model.state_dict(), initial_model, rtol=0, atol=0)
            zero = cast(dict[str, Any], torch.load(tmp_path / "full/last.pt", weights_only=False))
            assert zero["step"] == 0
            assert zero["optimizer"]["state"] == {}
        epoch, position = divmod(step - 1, inputs.bundle.train_groups // inputs.bundle.batch_size)
        selected = list(
            GlobalBatchSampler(inputs.bundle.train_groups, inputs.bundle.batch_size, 17, epoch)
        )[position]
        assert set(batch) == {"online", "target", "actions"}
        for name, value in batch.items():
            torch.testing.assert_close(value, inputs.bundle.train[name][selected], rtol=0, atol=0)
        assert torch.equal(
            torch.get_rng_state(), torch.Generator().manual_seed(1700000 + step).get_state()
        )
        assert fixed_hashes(model) == frozen
        assert (
            model.parameter_encoder.param_dim
            == model.predictor.param_dim
            == (6 if arm == "gain6" else 12)
        )
        assert all(p.requires_grad for p in model.parameter_encoder.parameters())
        assert all(p.requires_grad for p in model.predictor.parameters())
        batches += 1
        return original_loss(model, batch, objective)

    assessment = runner.assess_model
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
        assert seed == 7
        calls.append(final)
        assert fixed_hashes(model) == frozen
        guarded = data if final else {name: value for name, value in data.items() if name != "test"}
        return assessment(
            model, guarded, selection, bootstrap_samples=bootstrap_samples, seed=seed, final=final
        )

    monkeypatch.setattr(runner, "training_loss", capture)
    monkeypatch.setattr(runner, "assess_model", assess)
    full = run(config)
    assert batches == 3
    assert calls == [False] * 4 + [True]
    completed = cast(dict[str, Any], torch.load(tmp_path / "full/last.pt", weights_only=False))
    assert all(int(value["step"]) == 3 for value in completed["optimizer"]["state"].values())
    for prefix in ("parameter_encoder.", "predictor."):
        assert any(
            not torch.equal(value, initial_model[name])
            for name, value in completed["model"].items()
            if name.startswith(prefix)
        )
    attempts = 0

    def interrupt(
        model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("interrupt replication")
        return original_loss(model, batch, objective)

    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    monkeypatch.setattr(runner, "training_loss", interrupt)
    calls.clear()
    with pytest.raises(RuntimeError, match="interrupt replication"):
        run(resumed_config)
    assert not any(calls)
    assert not (tmp_path / "resumed/final_test.json").exists()
    monkeypatch.setattr(runner, "training_loss", original_loss)
    resumed = run(resumed_config, resume=True)
    checkpoint = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    for name in ("model", "optimizer"):
        torch.testing.assert_close(checkpoint[name], completed[name], rtol=0, atol=0)
    assert torch.equal(checkpoint["rng"]["torch"], completed["rng"]["torch"])
    assert full["records"] == resumed["records"]
    assert full["final_test"] == resumed["final_test"]
    assert all(sha(source_path / name) == digest for name, digest in inputs.source_hashes.items())
    count = len(calls)
    assert run(resumed_config, resume=True) == resumed
    assert len(calls) == count
    with pytest.raises(ValueError, match="Resume settings"):
        run(replace(resumed_config, evaluation_seed=8), resume=True)
