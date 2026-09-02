"""End-to-end tests for the fixed teacher-forcing-plus-T=2 trainer."""

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

import scjepa.training.loop as training_loop
from scjepa.data import BounceDataset
from scjepa.models import StateToStateModel, build_state_to_state
from scjepa.training import SparsityLagrangian, TrainConfig, Trainer

N = 3


def tiny_model(dense: bool = False, identity: bool = False) -> StateToStateModel:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    return build_state_to_state(
        num_slots=N,
        param_encoder_dim=16,
        param_encoder_heads=2,
        max_history=8,
        spartan_layers=1,
        spartan_embed_dim=32,
        spartan_mlp_hidden=32,
        spartan_mlp_layers=2,
        spartan_dense=dense,
        spartan_identity=identity,
    )


def tiny_dataset(num_episodes: int = 8, clip_len: int = 6) -> BounceDataset:
    return BounceDataset(
        num_episodes=num_episodes,
        clip_len=clip_len,
        num_balls=N,
        seed=2,
        render=False,
        cache=True,
        mass_normal=(1.5, 0.5),
        radius_from_mass=True,
    )


def tiny_config(out_dir: Path, steps: int = 3, **overrides: object) -> TrainConfig:
    defaults: dict[str, object] = {
        "steps": steps,
        "batch_size": 4,
        "sparsity_tau": 0.5,
        "context_len": 2,
        "lambda_logit": 1e-3,
        # T=6/C=2 has three valid T=2 offsets: 0,1,2.
        "lambda_rollout_t2": 1.0,
        "num_rollout_t2_anchors": 3,
        "rollout_t2_horizon": 2,
        "oe_eval_horizon": 4,
        "oe_coordinate_std": (1.0, 1.0, 1.0, 1.0),
        "log_every": 1,
        "checkpoint_every": 1000,
        "out_dir": str(out_dir),
        "seed": 0,
    }
    defaults.update(overrides)
    return TrainConfig(**defaults)  # pyright: ignore[reportArgumentType]


def test_lagrangian_dual_dynamics() -> None:
    controller = SparsityLagrangian(tau=0.1, step_size=1.0, lambda_init=1e4, momentum=0.0)
    start = float(controller.log_lambda)
    controller.update(torch.tensor(1.1))
    assert float(controller.log_lambda) == pytest.approx(start + 1.0)
    for _ in range(40):
        controller.update(torch.tensor(0.0))
    assert float(controller.log_lambda) < start - 2.0
    assert controller.penalty_weight.item() == pytest.approx(
        float(torch.exp(-controller.log_lambda))
    )


def test_training_smoke_has_finite_fixed_objective_and_no_schedule_keys(tmp_path: Path) -> None:
    trainer = Trainer(tiny_model(), tiny_dataset(), tiny_config(tmp_path))
    metrics = trainer.train()
    required = {
        "train/loss_teacher_forcing",
        "train/loss_rollout_t2_raw",
        "train/loss_rollout_t2_weighted",
        "train/loss_total",
        "train/grad_norm_teacher_forcing",
        "train/grad_norm_rollout_t2_weighted",
        "loss/logit",
        "loss/sparsity",
        "sparsity/constraint",
        "sparsity/lambda",
        "sparsity/path_density",
        "health/grad_norm",
        "health/skipped_steps",
    }
    assert required <= metrics.keys()
    assert all(torch.isfinite(torch.tensor(metrics[key])) for key in required)
    assert metrics["sparsity/constraint"] == pytest.approx(
        metrics["train/loss_teacher_forcing"]
        + metrics["train/loss_rollout_t2_weighted"]
        + metrics["loss/logit"],
        rel=1e-6,
    )
    assert not any(key.startswith("schedule/") for key in metrics)
    assert (tmp_path / "last.pt").exists()


def test_shipped_state_protocol_is_fixed_and_has_no_obsolete_schema() -> None:
    path = Path(__file__).parents[1] / "configs" / "experiment" / "bounce_baumgartner.yaml"
    preset = OmegaConf.load(path)
    assert isinstance(preset, DictConfig)
    assert int(preset.train.steps) == 300_000
    assert float(preset.train.lambda_rollout_t2) == 1.0
    assert int(preset.train.num_rollout_t2_anchors) == 8
    assert int(preset.train.rollout_t2_horizon) == 2
    assert int(preset.train.oe_eval_horizon) == 30
    assert float(preset.train.oe_tolerance_nrmse) == pytest.approx(0.10)
    assert float(preset.train.lambda_logit) == pytest.approx(1e-5)
    assert float(preset.train.sparsity_lambda_init) == pytest.approx(1e4)
    obsolete = {"rollout_curriculum", "rollout_len", "lambda_roll"}
    assert obsolete.isdisjoint(preset.train.keys())
    assert obsolete.isdisjoint(field.name for field in fields(TrainConfig))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"rollout_t2_horizon": 3}, "must equal 2"),
        ({"lambda_rollout_t2": -1.0}, "non-negative"),
        ({"num_rollout_t2_anchors": 0}, "positive"),
        ({"oe_eval_horizon": 0}, "positive"),
        ({"oe_tolerance_nrmse": -0.1}, "non-negative"),
    ],
)
def test_fixed_protocol_validation(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Trainer(tiny_model(), tiny_dataset(), tiny_config(tmp_path, **override))


def test_lambda_zero_bypasses_auxiliary_calls_and_reproduces_tf_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = Trainer(
        tiny_model(),
        tiny_dataset(),
        tiny_config(tmp_path, steps=1, lambda_rollout_t2=0.0, sparsity_enabled=False),
    )
    batch = next(trainer._batches())
    states = batch["states"]
    trainer.model.train()

    torch.manual_seed(123)  # pyright: ignore[reportUnknownMemberType]
    expected = trainer.model(states, context_len=2)
    expected_rng = torch.get_rng_state().clone()

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("lambda=0 must not construct a T=2 branch")

    monkeypatch.setattr(trainer.model, "rollout_t2_from_offsets", forbidden)
    torch.manual_seed(123)  # pyright: ignore[reportUnknownMemberType]
    actual = trainer._forward(batch)
    actual_rng = torch.get_rng_state().clone()
    torch.testing.assert_close(actual.prediction, expected.prediction)
    torch.testing.assert_close(actual.target, expected.target)
    torch.testing.assert_close(actual.causal_params, expected.causal_params)
    assert actual.rollout_t2_prediction is None
    assert torch.equal(actual_rng, expected_rng)

    metrics = trainer._train_step(batch)
    assert metrics["train/loss_rollout_t2_raw"] == 0.0
    assert metrics["train/loss_rollout_t2_weighted"] == 0.0
    assert metrics["train/loss_total"] == pytest.approx(
        metrics["train/loss_teacher_forcing"] + metrics["loss/logit"], rel=1e-6
    )


def test_training_builds_only_tf_plus_two_recurrent_calls_not_k30(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tiny_model()
    calls: list[None] = []
    hook = model.predictor.register_forward_hook(
        lambda _module, _inputs, _output: calls.append(None)
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("K=30 evaluation may not participate in training")

    monkeypatch.setattr(model, "rollout_for_evaluation", forbidden)
    try:
        trainer = Trainer(model, tiny_dataset(), tiny_config(tmp_path, steps=1))
        trainer._train_step(next(trainer._batches()))
    finally:
        hook.remove()
    assert len(calls) == 3  # one B*K TF call, then exactly two B*W calls


def test_sparsity_and_geco_are_active_from_first_fixed_update(tmp_path: Path) -> None:
    trainer = Trainer(
        tiny_model(),
        tiny_dataset(),
        tiny_config(
            tmp_path,
            steps=1,
            sparsity_enabled=True,
            sparsity_tau=-1.0,
            sparsity_momentum=0.0,
            grad_skip_threshold=float("inf"),
        ),
    )
    initial = trainer.lagrangian.log_lambda.clone()
    metrics = trainer._train_step(next(trainer._batches()))
    assert metrics["sparsity/active"] == 1.0
    assert not torch.equal(trainer.lagrangian.log_lambda, initial)
    predictive = (
        metrics["train/loss_teacher_forcing"]
        + metrics["train/loss_rollout_t2_weighted"]
        + metrics["loss/logit"]
    )
    assert metrics["train/loss_total"] > predictive


def test_checkpoint_resume_restores_t2_anchor_sampling_sequence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    trained = Trainer(tiny_model(), tiny_dataset(), tiny_config(run_dir, steps=2))
    trained.train()
    checkpoint = run_dir / "last.pt"

    def next_offsets() -> torch.Tensor:
        restored = Trainer(tiny_model(), tiny_dataset(), tiny_config(run_dir, steps=3))
        restored.load_checkpoint(checkpoint)
        output = restored._forward(next(restored._batches()))
        assert output.rollout_t2_offsets is not None
        return output.rollout_t2_offsets.clone()

    torch.testing.assert_close(next_offsets(), next_offsets())
    payload = torch.load(checkpoint, weights_only=False)
    assert tuple(payload["oe_coordinate_std"]) == (1.0, 1.0, 1.0, 1.0)
    assert "successful_updates" not in payload


def test_checkpoint_rejects_a_different_oe_ruler(tmp_path: Path) -> None:
    trainer = Trainer(tiny_model(), tiny_dataset(), tiny_config(tmp_path, steps=1))
    trainer.train()
    incompatible = Trainer(
        tiny_model(),
        tiny_dataset(),
        tiny_config(tmp_path, steps=2, oe_coordinate_std=(2.0, 1.0, 1.0, 1.0)),
    )
    with pytest.raises(ValueError, match="oe_coordinate_std"):
        incompatible.load_checkpoint(tmp_path / "last.pt")


def test_gradient_spike_rejects_primal_and_dual_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = Trainer(
        tiny_model(),
        tiny_dataset(),
        tiny_config(tmp_path, steps=1, grad_skip_threshold=10.0),
    )
    before_model = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    before_dual = trainer.lagrangian.log_lambda.clone()

    def oversized_norm(*_args: object, **_kwargs: object) -> torch.Tensor:
        return torch.tensor(11.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", oversized_norm)
    trainer._train_step(next(trainer._batches()))
    assert trainer.total_skips == 1
    assert torch.equal(trainer.lagrangian.log_lambda, before_dual)
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, before_model[key])


def test_sparsity_ablation_removes_path_term_and_freezes_dual(tmp_path: Path) -> None:
    trainer = Trainer(
        tiny_model(), tiny_dataset(), tiny_config(tmp_path, steps=1, sparsity_enabled=False)
    )
    initial = trainer.lagrangian.log_lambda.clone()
    metrics = trainer.train()
    assert torch.equal(trainer.lagrangian.log_lambda, initial)
    assert metrics["train/loss_total"] == pytest.approx(
        metrics["train/loss_teacher_forcing"]
        + metrics["train/loss_rollout_t2_weighted"]
        + metrics["loss/logit"],
        rel=1e-6,
    )


def test_periodic_eval_passes_fixed_objective_and_oe_ruler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_evaluate(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        calls.append(kwargs)
        return SimpleNamespace(metrics={"constraint_loss": 0.25})

    monkeypatch.setattr(training_loop, "evaluate_identifiability", fake_evaluate)
    trainer = Trainer(
        tiny_model(),
        tiny_dataset(),
        tiny_config(tmp_path, eval_every=1),
        eval_dataset=tiny_dataset(6),
    )
    assert trainer._eval_step() == {"eval/constraint_loss": 0.25}
    assert calls == [
        {
            "batch_size": 4,
            "device": "cpu",
            "context_len": 2,
            "lambda_logit": 1e-3,
            "lambda_rollout_t2": 1.0,
            "num_rollout_t2_anchors": 3,
            "rollout_t2_horizon": 2,
            "oe_eval_horizon": 4,
            "oe_tolerance_nrmse": 0.1,
            "oe_coordinate_std": (1.0, 1.0, 1.0, 1.0),
        }
    ]


def test_eval_requires_dataset_and_fixed_training_scales(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="eval_every"):
        Trainer(tiny_model(), tiny_dataset(), tiny_config(tmp_path, eval_every=1))
    with pytest.raises(ValueError, match="oe_coordinate_std"):
        Trainer(
            tiny_model(),
            tiny_dataset(),
            tiny_config(tmp_path, eval_every=1, oe_coordinate_std=None),
            eval_dataset=tiny_dataset(6),
        )


def test_dataset_smaller_than_one_batch_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="yields no batches"):
        Trainer(tiny_model(), tiny_dataset(4), tiny_config(tmp_path, batch_size=8))
