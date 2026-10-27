import copy
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_group_latent_runner import source_with_initialization
from torch import Tensor

from diagnostics.encoder_gain_acquisition.model import apply_encoder_gains
from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.group_latent_control.run import initialize_control
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.mlp_oracle_code_control.model import initialize_mlp
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit
from diagnostics.multiquery_control.run import sha
from diagnostics.overcomplete_acquisition import run as runner
from diagnostics.overcomplete_acquisition.model import initialize_overcomplete
from diagnostics.overcomplete_acquisition.run import Config, fixed_hashes, run
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_twelve_code_source_integrity_batches_resume_and_final_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_with_initialization(tmp_path / "source")
    bundle = load_primary_source(source, 7, expected_step=2, expected_checkpoint_sha256=None)
    source_model, _ = build_source_model(bundle, torch.device("cpu"))
    _, _, hashes = initialize_control(source_model, bundle, "fresh_zero")
    frozen = fixed_hashes(source_model)
    initialize_mlp(source_model, 7)
    gain_initialization = apply_encoder_gains(source_model)
    expanded_initialization = initialize_overcomplete(source_model, 7)
    config = Config(
        str(source),
        str(tmp_path / "full"),
        seed=7,
        steps=3,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    original_load = cast(Callable[..., Any], torch.load)

    def guarded_load(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            assert Path(file).name != "audit_inputs.pt", "Joint source loaded training theta"
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded_load)
    original_loss = runner.training_loss
    samples: list[tuple[TensorSplit, Tensor]] = []

    def capture(
        model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
    ) -> tuple[Tensor, dict[str, Tensor]]:
        assert set(batch) == {"online", "target", "actions"}
        assert type(model.parameter_encoder) is TransitionParameterEncoder
        assert model.parameter_encoder.param_dim == model.predictor.param_dim == 12
        assert fixed_hashes(model) == frozen
        assert all(parameter.requires_grad for parameter in model.parameter_encoder.parameters())
        assert all(parameter.requires_grad for parameter in model.predictor.parameters())
        samples.append(
            ({name: value.clone() for name, value in batch.items()}, torch.get_rng_state())
        )
        return original_loss(model, batch, objective)

    calls: list[bool] = []
    original_assess = runner.assess_model

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
        guarded = data if final else {key: value for key, value in data.items() if key != "test"}
        return original_assess(
            model, guarded, selection, bootstrap_samples=bootstrap_samples, seed=seed, final=final
        )

    monkeypatch.setattr(runner, "training_loss", capture)
    monkeypatch.setattr(runner, "assess_model", assess)
    full = run(config)
    assert calls == [False] * 4 + [True]
    checkpoint = cast(dict[str, Any], torch.load(tmp_path / "full/last.pt", weights_only=False))
    initial = cast(
        dict[str, Any], torch.load(tmp_path / "full/initial_modules.pt", weights_only=False)
    )
    torch.testing.assert_close(
        initial["parameter_encoder"], source_model.parameter_encoder.state_dict(), rtol=0, atol=0
    )
    assert initial["encoder_gain_initialization"] == gain_initialization
    assert gain_initialization["parameter_count"] == 70534
    assert initial["overcomplete_initialization"] == expanded_initialization
    assert expanded_initialization["encoder_parameter_count"] == 71308
    assert expanded_initialization["predictor_parameter_count"] == 53389
    torch.testing.assert_close(
        initial["predictor"], source_model.predictor.state_dict(), rtol=0, atol=0
    )
    restored = copy.deepcopy(source_model)
    restored.load_state_dict(checkpoint["model"])
    assert fixed_hashes(restored) == frozen
    for prefix in ("predictor", "parameter_encoder"):
        assert any(
            not torch.equal(value, initial[prefix][name])
            for name, value in getattr(restored, prefix).state_dict().items()
        )
    protocol = cast(dict[str, Any], read_json(tmp_path / "full/protocol.json"))
    assert protocol["training_parameter_labels_absent"] is True
    assert len(full["final_test"]["retention"]["selected_test_group_mse_by_parameter"]) == len(
        bundle.evaluation["test"]["theta"]
    )
    assert len(full["final_test"]["response"]["group_indices"]) == 2
    assert len(samples) == 3

    # The original sampler and per-update loss-entry RNG are unchanged even
    # though the zero-weight parameter regularizer now consumes twelve rows.
    from scjepa.training.distributed import GlobalBatchSampler
    from scjepa.training.rng import preserve_rng

    for step, (batch, rng) in enumerate(samples, 1):
        epoch, position = divmod(step - 1, bundle.train_groups // bundle.batch_size)
        batches = list(GlobalBatchSampler(bundle.train_groups, bundle.batch_size, 7, epoch))
        expected_batch = {name: value[batches[position]] for name, value in bundle.train.items()}
        torch.testing.assert_close(batch, expected_batch, rtol=0, atol=0)
        with preserve_rng(torch.device("cpu")):
            torch.default_generator.manual_seed(7 * 100000 + step)
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)

    updates = 0

    def interrupt(
        model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal updates
        updates += 1
        assert set(batch) == {"online", "target", "actions"}
        if updates == 2:
            raise RuntimeError("interrupted joint forecast training")
        return original_loss(model, batch, objective)

    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    monkeypatch.setattr(runner, "training_loss", interrupt)
    calls.clear()
    with pytest.raises(RuntimeError, match="interrupted joint forecast training"):
        run(resumed_config)
    assert not any(calls)
    assert not (tmp_path / "resumed/final_test.json").exists()
    monkeypatch.setattr(runner, "training_loss", original_loss)
    resumed = run(resumed_config, resume=True)
    final = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    for key in ("model", "optimizer"):
        torch.testing.assert_close(checkpoint[key], final[key], rtol=0, atol=0)
    torch.testing.assert_close(checkpoint["rng"]["torch"], final["rng"]["torch"], rtol=0, atol=0)
    assert full["records"] == resumed["records"]
    assert full["final_test"] == resumed["final_test"]
    assert all(sha(source / name) == value for name, value in hashes.items())
    count = len(calls)
    assert run(resumed_config, resume=True) == resumed
    assert len(calls) == count
    with pytest.raises(ValueError, match="Resume settings"):
        run(replace(resumed_config, bootstrap_samples=17), resume=True)
