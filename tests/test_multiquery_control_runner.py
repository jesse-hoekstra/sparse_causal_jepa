from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.multiquery_control import run as runner
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.loop import ObjectiveConfig


def case(directory: Path) -> tuple[runner.Config, dict[str, Any], dict[str, Tensor]]:
    directory.mkdir()
    model_config = ModelConfig(
        observation_dim=6,
        encoder_hidden_dims=(8,),
        state_dim=6,
        param_dim=6,
        action_dim=16,
        context_len=10,
        history_len=1,
        parameter_encoder_kind="transition",
        spartan_layers=1,
        spartan_embed_dim=4,
        spartan_mlp_hidden=8,
        spartan_mlp_layers=2,
        spartan_dense=True,
        autoregressive=True,
    )
    torch.default_generator.manual_seed(714)
    model = build_model(model_config)
    model.online_encoder.requires_grad_(False).eval()
    model.target_encoder.requires_grad_(False).eval()
    initial = directory / "initial_seed_7.pt"
    runner.atomic_save(initial, {"model": model.state_dict(), "seed": 7, "source_sha256": "source"})
    objective = ObjectiveConfig(
        state_regularizer="sigreg",
        parameter_regularizer="sigreg",
        state_regularizer_weight=0.4,
        parameter_regularizer_weight=0.4,
        parameter_consistency_weight=1.0,
        regularizer_num_projections=8,
        logit_weight=0.0001,
    )
    protocol: dict[str, Any] = {
        "identity": "test-protocol",
        "model_config": asdict(model_config),
        "objective_config": asdict(objective),
        "initializations": {"7": {"path": initial.name, "sha256": runner.sha(initial)}},
        "frozen_hashes": runner.frozen_hashes(model),
        "source": {"sha256": "source"},
    }
    config = runner.Config(
        checkpoint="unused-by-arm-unit-test",
        output_dir=str(directory),
        seeds=(7,),
        conditions=("q3_c0", "q1_c1"),
        steps=3,
        train_groups=4,
        calibration_groups=4,
        validation_groups=4,
        test_groups=4,
        batch_size=2,
        evaluate_every=1,
        checkpoint_every=1,
        log_every=1,
    )
    train = {
        "online": torch.randn(4, 10, 12, 6),
        "target": torch.randn(4, 10, 12, 6),
        "actions": torch.randn(4, 10, 11, 16),
    }
    return config, protocol, train


def cheap_evaluation(
    model: SCJEPA,
    evaluation_data: dict[str, Any],
    objective_config: ObjectiveConfig,
    *,
    batch_size: int = 16,
    final: bool = False,
) -> dict[str, Any]:
    assert not model.training
    assert not any(p.requires_grad for p in model.online_encoder.parameters())
    assert not any(p.requires_grad for p in model.target_encoder.parameters())
    assert batch_size == 2
    assert evaluation_data == {}
    torch.default_generator.manual_seed(917)  # Deliberately perturb auxiliary RNG.
    return {"final": final, "consistency_weight": objective_config.parameter_consistency_weight}


def assert_same_state(first: object, second: object) -> None:
    if isinstance(first, Tensor):
        assert isinstance(second, Tensor)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert isinstance(second, dict)
        left, right = cast(dict[str, Any], first), cast(dict[str, Any], second)
        assert left.keys() == right.keys()
        for key in left:
            assert_same_state(left[key], right[key])
    elif isinstance(first, list | tuple):
        assert isinstance(second, list | tuple)
        left_list, right_list = cast(list[Any], first), cast(list[Any], second)
        assert len(left_list) == len(right_list)
        for left_value, right_value in zip(left_list, right_list, strict=True):
            assert_same_state(left_value, right_value)
    else:
        assert first == second


def test_resume_replays_exact_next_batch_across_epoch_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(runner, "evaluate", cheap_evaluation)
    uninterrupted = case(tmp_path / "uninterrupted")
    interrupted = case(tmp_path / "interrupted")
    try:
        runner.train_condition(
            *uninterrupted[:2], uninterrupted[2], {}, torch.device("cpu"), "q3_c0", 7, resume=False
        )
        calls = 0

        def interrupt_second(
            model: SCJEPA, batch: dict[str, Tensor], queries: int = 3
        ) -> ModelOutput:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interruption")
            return cached_multiquery_forward(model, batch, queries=queries)

        monkeypatch.setattr(runner, "cached_multiquery_forward", interrupt_second)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            runner.train_condition(
                *interrupted[:2], interrupted[2], {}, torch.device("cpu"), "q3_c0", 7, resume=False
            )
        path = Path(interrupted[0].output_dir) / "seed-7" / "q3_c0" / "last.pt"
        saved = torch.load(path, weights_only=False)
        assert saved["step"] == 1
        monkeypatch.setattr(runner, "cached_multiquery_forward", cached_multiquery_forward)
        runner.train_condition(
            *interrupted[:2], interrupted[2], {}, torch.device("cpu"), "q3_c0", 7, resume=True
        )
        resumed = torch.load(path, weights_only=False)
        original = torch.load(
            Path(uninterrupted[0].output_dir) / "seed-7" / "q3_c0" / "last.pt",
            weights_only=False,
        )
        assert resumed["step"] == 3
        assert_same_state(resumed["model"], original["model"])
        assert_same_state(resumed["optimizer"], original["optimizer"])
        assert resumed["frozen_hashes"] == interrupted[1]["frozen_hashes"]

        def must_not_train(
            _model: SCJEPA, _batch: dict[str, Tensor], queries: int = 3
        ) -> ModelOutput:
            raise AssertionError(f"Completed arm attempted to train with {queries} queries")

        monkeypatch.setattr(runner, "cached_multiquery_forward", must_not_train)
        completed = runner.train_condition(
            *interrupted[:2], interrupted[2], {}, torch.device("cpu"), "q3_c0", 7, resume=True
        )
        assert completed["steps"] == 3
        monkeypatch.setattr(runner, "cached_multiquery_forward", cached_multiquery_forward)
        fresh = runner.train_condition(
            *interrupted[:2], interrupted[2], {}, torch.device("cpu"), "q1_c1", 7, resume=True
        )
        assert fresh["steps"] == 3
        assert fresh["condition"] == "q1_c1"
    finally:
        torch.set_num_threads(threads)
