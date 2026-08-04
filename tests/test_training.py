"""End-to-end training tests: dual controller, smoke, resume, guards, eval."""

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
PAPER_ROLLOUT_CURRICULUM: tuple[tuple[int, int | None], ...] = (
    (0, None),
    (10_000, 2),
    (15_000, 5),
    (25_000, 10),
    (40_000, 20),
    (60_000, 30),
)


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


def tiny_dataset(num_episodes: int = 8, clip_len: int = 4) -> BounceDataset:
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
    defaults: dict[str, object] = dict(
        steps=steps,
        batch_size=4,
        sparsity_tau=0.5,
        context_len=2,
        lambda_logit=1e-3,
        # T=4, Tpar=2: the chain starts at t=1 and its last target is S̄_3,
        # so K=2 keeps the hybrid branch live in every training test.
        rollout_len=2,
        log_every=1,
        checkpoint_every=1000,
        out_dir=str(out_dir),
        seed=0,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)  # pyright: ignore[reportArgumentType]


def test_lagrangian_dual_dynamics() -> None:
    """Log lambda += alpha * MA[c - tau], UNCLAMPED in both directions (§6.1.3)."""
    controller = SparsityLagrangian(tau=0.1, step_size=1.0, lambda_init=1e6, momentum=0.0)
    start = float(controller.log_lambda)
    controller.update(torch.tensor(1.1))  # c > tau: lambda RISES above its init
    assert float(controller.log_lambda) == pytest.approx(start + 1.0)
    for _ in range(40):
        controller.update(torch.tensor(0.0))  # c < tau: lambda falls freely
    assert float(controller.log_lambda) < start - 2.0
    assert controller.penalty_weight.item() == pytest.approx(
        float(torch.exp(-controller.log_lambda))
    )


def test_training_smoke(tmp_path: Path) -> None:
    """Eq. 40 objective end-to-end: finite losses, metrics logged, checkpoint."""
    trainer = Trainer(tiny_model(), tiny_dataset(), tiny_config(tmp_path))
    metrics = trainer.train()
    for key in (
        "loss/total",
        "loss/pred",
        "loss/rollout_raw",
        "loss/rollout",
        "loss/logit",
        "loss/sparsity",
        "sparsity/constraint",
        "sparsity/lambda",
        "sparsity/path_density",
        "health/grad_norm",
        "health/grad_norm_tf",
        "health/grad_norm_rollout_raw",
        "health/grad_norm_rollout",
        "schedule/lambda_roll",
    ):
        assert key in metrics
        assert torch.isfinite(torch.tensor(metrics[key])), key
    # Hybrid §4.3 dual form: the bound covers the teacher-forced AND rollout
    # errors, scalarised, and still excludes the path penalty.
    assert metrics["sparsity/constraint"] == pytest.approx(
        metrics["loss/pred"] + metrics["loss/rollout"] + metrics["loss/logit"], rel=1e-6
    )
    assert (tmp_path / "last.pt").exists()


def test_rollout_weight_stays_one_and_raw_equals_applied(tmp_path: Path) -> None:
    """Depth changes, never lambda_roll: raw and applied rollout quantities coincide."""
    config = tiny_config(tmp_path, steps=1, lambda_roll=1.0)
    assert not hasattr(config, "lambda_roll_warmup_steps")
    trainer = Trainer(tiny_model(), tiny_dataset(), config)
    metrics = trainer._train_step(next(trainer._batches()))
    assert metrics["schedule/lambda_roll"] == pytest.approx(1.0)
    assert metrics["loss/rollout"] == pytest.approx(metrics["loss/rollout_raw"], rel=1e-6)
    assert metrics["health/grad_norm_tf"] > 0.0
    assert metrics["health/grad_norm_rollout_raw"] > 0.0
    assert metrics["health/grad_norm_rollout"] == pytest.approx(
        metrics["health/grad_norm_rollout_raw"], rel=1e-6
    )


def test_rollout_curriculum_exact_successful_update_boundaries(tmp_path: Path) -> None:
    """The next forward's K changes only at the six declared accepted-update stages."""
    trainer = Trainer(
        tiny_model(),
        tiny_dataset(),
        tiny_config(
            tmp_path,
            rollout_len=30,
            rollout_curriculum=PAPER_ROLLOUT_CURRICULUM,
        ),
    )
    boundaries = (
        (0, None),
        (9_999, None),
        (10_000, 2),
        (14_999, 2),
        (15_000, 5),
        (24_999, 5),
        (25_000, 10),
        (39_999, 10),
        (40_000, 20),
        (59_999, 20),
        (60_000, 30),
        (75_000, 30),
    )
    for successful_updates, expected_horizon in boundaries:
        trainer.successful_updates = successful_updates
        assert trainer._current_rollout_len() == expected_horizon


def test_paper_preset_declares_the_exact_curriculum() -> None:
    """The shipped Experiment-1 config must not drift from the audited schedule."""
    path = Path(__file__).parents[1] / "configs" / "experiment" / "bounce_baumgartner.yaml"
    preset = OmegaConf.load(path)
    assert isinstance(preset, DictConfig)
    assert float(preset.train.lambda_roll) == 1.0
    assert OmegaConf.to_container(preset.train.rollout_curriculum, resolve=True) == [
        {"start_update": 0, "rollout_len": None},
        {"start_update": 10_000, "rollout_len": 2},
        {"start_update": 15_000, "rollout_len": 5},
        {"start_update": 25_000, "rollout_len": 10},
        {"start_update": 40_000, "rollout_len": 20},
        {"start_update": 60_000, "rollout_len": 30},
    ]
    assert "lambda_roll_warmup_steps" not in preset.train
    assert int(preset.train.grad_skip_max_consecutive) == 50


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"rollout_curriculum": ((0, None), (1, 2)), "lambda_roll": 2.0},
            "requires lambda_roll=1.0",
        ),
        (
            {"rollout_curriculum": ((1, None), (2, 2))},
            "must start at successful update 0",
        ),
        (
            {"rollout_len": 5, "rollout_curriculum": ((0, None), (1, 2), (1, 5))},
            "starts must be strictly increasing",
        ),
        (
            {"rollout_curriculum": ((0, 5), (1, 2))},
            "horizons must be non-decreasing",
        ),
        (
            {"rollout_len": 5, "rollout_curriculum": ((0, None), (1, 2))},
            "terminal horizon must equal train.rollout_len",
        ),
        (
            {"rollout_len": 1, "rollout_curriculum": ((0, None), (1, 1))},
            "horizons must be >= 2",
        ),
    ],
)
def test_rollout_curriculum_validation(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    """Malformed curricula fail at construction instead of changing the objective silently."""
    with pytest.raises(ValueError, match=message):
        Trainer(
            tiny_model(),
            tiny_dataset(),
            tiny_config(tmp_path, **overrides),  # pyright: ignore[reportArgumentType]
        )


def test_only_accepted_updates_advance_curriculum_and_make_boundary_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clipped accepted step advances K; the following rejected attempt does not."""
    curriculum = ((0, None), (1, 2))
    trainer = Trainer(
        tiny_model(),
        tiny_dataset(),
        tiny_config(
            tmp_path,
            steps=2,
            rollout_curriculum=curriculum,
            grad_clip=1.0,
            grad_skip_threshold=3.0,
            grad_skip_max_consecutive=3,
        ),
    )
    reported_norms = iter((torch.tensor(2.0), torch.tensor(4.0)))

    def scripted_grad_norm(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        return next(reported_norms)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", scripted_grad_norm)
    metrics = trainer.train()

    # The first norm exceeds grad_clip but not the skip threshold, so its
    # optimizer step counts. The second attempt is rejected at K=2.
    assert trainer.step == 2
    assert trainer.successful_updates == 1
    assert trainer.total_skips == 1
    assert trainer._current_rollout_len() == 2
    assert metrics["schedule/rollout_len"] == pytest.approx(2.0)
    assert metrics["schedule/successful_updates"] == pytest.approx(1.0)

    boundary = tmp_path / "curriculum_success_1_before_k2.pt"
    assert boundary.exists()
    payload = torch.load(boundary, weights_only=False)
    assert payload["step"] == 1
    assert payload["successful_updates"] == 1


def test_tf_only_curriculum_stage_keeps_all_teacher_forced_suffixes(tmp_path: Path) -> None:
    """K=None disables only recurrence, not any of the full-window TF pairs."""
    dataset = tiny_dataset(clip_len=7)
    trainer = Trainer(
        tiny_model(),
        dataset,
        tiny_config(
            tmp_path,
            rollout_len=5,
            rollout_curriculum=((0, None), (1, 5)),
        ),
    )
    batch = next(iter(trainer._epoch_loader(0)))
    with torch.no_grad():
        output = trainer._forward(batch, trainer._current_rollout_len())
    assert output.rollout_prediction is None
    assert output.rollout_target is None
    assert output.prediction.shape[0] == trainer.config.batch_size * (7 - 2)
    assert output.target.shape == output.prediction.shape


def test_rollout_weights_follow_dynamic_k(tmp_path: Path) -> None:
    """Moving K=2 -> K=5 builds matching Eq. 35 weights instead of reusing K=2's."""
    trainer = Trainer(
        tiny_model(),
        tiny_dataset(clip_len=7),
        tiny_config(
            tmp_path,
            steps=2,
            rollout_len=5,
            rollout_curriculum=((0, 2), (1, 5)),
            sparsity_enabled=False,
        ),
    )
    batches = trainer._batches()
    at_k2 = trainer._train_step(next(batches))
    assert trainer.successful_updates == 1
    assert at_k2["schedule/rollout_len"] == pytest.approx(2.0)
    at_k5 = trainer._train_step(next(batches))
    assert at_k5["schedule/rollout_len"] == pytest.approx(5.0)
    assert at_k2["loss/rollout"] == pytest.approx(at_k2["loss/rollout_raw"], rel=1e-6)
    assert at_k5["loss/rollout"] == pytest.approx(at_k5["loss/rollout_raw"], rel=1e-6)
    assert set(trainer._rollout_weight_cache) == {2, 5}
    assert trainer._rollout_weight_cache[2].shape == (2,)
    assert trainer._rollout_weight_cache[5].shape == (5,)


def test_sparsity_and_dual_wait_for_terminal_curriculum_k(tmp_path: Path) -> None:
    """The K=30-calibrated GECO constraint must not see easier prefix stages."""
    trainer = Trainer(
        tiny_model(),
        tiny_dataset(clip_len=7),
        tiny_config(
            tmp_path,
            steps=3,
            rollout_len=5,
            rollout_curriculum=((0, None), (1, 2), (2, 5)),
            sparsity_enabled=True,
            sparsity_tau=-1.0,
            sparsity_step_size=0.1,
            sparsity_momentum=0.0,
            grad_skip_threshold=float("inf"),
        ),
    )
    batches = trainer._batches()
    initial_log_lambda = trainer.lagrangian.log_lambda.clone()
    initial_ma_error = trainer.lagrangian.ma_error.clone()

    tf_metrics = trainer._train_step(next(batches))
    assert tf_metrics["sparsity/active"] == 0.0
    assert tf_metrics["loss/total"] == pytest.approx(
        tf_metrics["loss/pred"] + tf_metrics["loss/logit"], rel=1e-6
    )
    torch.testing.assert_close(trainer.lagrangian.log_lambda, initial_log_lambda)
    torch.testing.assert_close(trainer.lagrangian.ma_error, initial_ma_error)

    k2_metrics = trainer._train_step(next(batches))
    assert k2_metrics["schedule/rollout_len"] == 2.0
    assert k2_metrics["sparsity/active"] == 0.0
    assert k2_metrics["loss/total"] == pytest.approx(
        k2_metrics["loss/pred"] + k2_metrics["loss/rollout"] + k2_metrics["loss/logit"],
        rel=1e-6,
    )
    torch.testing.assert_close(trainer.lagrangian.log_lambda, initial_log_lambda)
    torch.testing.assert_close(trainer.lagrangian.ma_error, initial_ma_error)

    terminal_metrics = trainer._train_step(next(batches))
    assert terminal_metrics["schedule/rollout_len"] == 5.0
    assert terminal_metrics["sparsity/active"] == 1.0
    terminal_prediction_terms = (
        terminal_metrics["loss/pred"]
        + terminal_metrics["loss/rollout"]
        + terminal_metrics["loss/logit"]
    )
    assert terminal_metrics["loss/total"] > terminal_prediction_terms
    assert not torch.equal(trainer.lagrangian.ma_error, initial_ma_error)
    assert not torch.equal(trainer.lagrangian.log_lambda, initial_log_lambda)
    assert trainer.successful_updates == 3


def test_references_train_without_path_penalty(tmp_path: Path) -> None:
    """Dense / token-local references: sparsity disabled, lambda frozen."""
    for model in (tiny_model(dense=True), tiny_model(identity=True)):
        trainer = Trainer(model, tiny_dataset(), tiny_config(tmp_path, sparsity_enabled=False))
        metrics = trainer.train()
        assert metrics["loss/total"] == pytest.approx(
            metrics["loss/pred"] + metrics["loss/rollout"] + metrics["loss/logit"],
            rel=1e-6,
        )
        assert metrics["sparsity/lambda"] == pytest.approx(1e6)


def test_resume_is_exact(tmp_path: Path) -> None:
    """Resume exactly across the successful-update boundary from TF to K=2."""
    curriculum = ((0, None), (2, 2))
    config_a = tiny_config(tmp_path / "a", steps=4, rollout_curriculum=curriculum)
    trainer_a = Trainer(tiny_model(), tiny_dataset(), config_a)
    final_a = trainer_a.train()
    assert trainer_a.successful_updates == 4

    config_b2 = tiny_config(tmp_path / "b", steps=2, rollout_curriculum=curriculum)
    trainer_b = Trainer(tiny_model(), tiny_dataset(), config_b2)
    trainer_b.train()
    assert trainer_b.successful_updates == 2
    config_b4 = tiny_config(tmp_path / "b", steps=4, rollout_curriculum=curriculum)
    trainer_b4 = Trainer(tiny_model(), tiny_dataset(), config_b4)
    trainer_b4.load_checkpoint(tmp_path / "b" / "last.pt")
    assert trainer_b4.step == 2
    assert trainer_b4.successful_updates == 2
    assert trainer_b4._current_rollout_len() == 2
    final_b = trainer_b4.train()
    assert trainer_b4.successful_updates == 4

    for key, value in final_a.items():
        assert final_b[key] == pytest.approx(value, rel=1e-5), key


def test_checkpoint_restores_success_count_and_legacy_fallback(tmp_path: Path) -> None:
    """New checkpoints restore the counter; old ones derive it from attempts minus skips."""
    curriculum = ((0, None), (4, 2), (5, 5))
    config = tiny_config(
        tmp_path,
        rollout_len=5,
        rollout_curriculum=curriculum,
    )
    trainer = Trainer(tiny_model(), tiny_dataset(clip_len=7), config)
    trainer.step = 7
    trainer.successful_updates = 4
    trainer.total_skips = 3
    trainer.consecutive_skips = 2
    checkpoint = tmp_path / "counter.pt"
    trainer.save_checkpoint(checkpoint)

    restored = Trainer(tiny_model(), tiny_dataset(clip_len=7), config)
    restored.load_checkpoint(checkpoint)
    assert restored.step == 7
    assert restored.successful_updates == 4
    assert restored.total_skips == 3
    assert restored.consecutive_skips == 2
    assert restored._current_rollout_len() == 2

    payload = torch.load(checkpoint, weights_only=False)
    del payload["successful_updates"]
    legacy_checkpoint = tmp_path / "legacy.pt"
    torch.save(payload, legacy_checkpoint)
    legacy = Trainer(tiny_model(), tiny_dataset(clip_len=7), config)
    legacy.load_checkpoint(legacy_checkpoint)
    assert legacy.successful_updates == 4  # step 7 minus the 3 recorded skips
    assert legacy._current_rollout_len() == 2


def test_grad_skip_guard_rejects_updates_and_raises_when_persistent(tmp_path: Path) -> None:
    """D18: absurd grad norms freeze the update; persistent skips fail loudly."""
    config = tiny_config(tmp_path, steps=3, grad_skip_threshold=1e-12, grad_skip_max_consecutive=2)
    trainer = Trainer(tiny_model(), tiny_dataset(), config)
    with pytest.raises(RuntimeError, match="consecutive grad-spike skips"):
        trainer.train()
    assert trainer.total_skips >= 2


def test_rolling_checkpoints_are_kept(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, steps=4, checkpoint_keep_every=2)
    Trainer(tiny_model(), tiny_dataset(), config).train()
    assert (tmp_path / "step_2.pt").exists()
    assert (tmp_path / "step_4.pt").exists()


def test_sparsity_ablation_toggle(tmp_path: Path) -> None:
    """sparsity_enabled=false removes the path term and freezes the dual."""
    config = tiny_config(tmp_path, sparsity_enabled=False)
    trainer = Trainer(tiny_model(), tiny_dataset(), config)
    metrics = trainer.train()
    assert metrics["sparsity/lambda"] == pytest.approx(1e6)
    assert metrics["loss/total"] == pytest.approx(
        metrics["loss/pred"] + metrics["loss/rollout"] + metrics["loss/logit"], rel=1e-6
    )


def test_periodic_eval_logs_metrics(tmp_path: Path) -> None:
    """eval_every wires the harness in; the logged key set is exact."""

    class Capture:
        def __init__(self) -> None:
            self.records: list[dict[str, float]] = []

        def log(self, step: int, metrics: dict[str, float]) -> None:  # noqa: ARG002
            self.records.append(metrics)

    logger = Capture()
    config = tiny_config(tmp_path, steps=2, eval_every=2)
    trainer = Trainer(
        tiny_model(), tiny_dataset(), config, logger=logger, eval_dataset=tiny_dataset(6)
    )
    trainer.train()
    eval_records = [r for r in logger.records if any(k.startswith("eval/") for k in r)]
    assert eval_records
    expected_eval_keys = {
        "eval/pred_loss",
        "eval/rollout_loss",
        "eval/mean_abs_logit",
        "eval/gate_entropy",
        "eval/constraint_loss",
        "eval/shd",
        "eval/mcc",
        "eval/path_density",
    }
    for record in eval_records:
        assert set(record) == expected_eval_keys
    # Reference modes log the SAME full key set (constant curves are cheap;
    # missing curves have previously hidden dead runs).
    logger2 = Capture()
    trainer2 = Trainer(
        tiny_model(dense=True),
        tiny_dataset(),
        tiny_config(tmp_path / "dense", steps=2, eval_every=2, sparsity_enabled=False),
        logger=logger2,
        eval_dataset=tiny_dataset(6),
    )
    trainer2.train()
    dense_records = [r for r in logger2.records if any(k.startswith("eval/") for k in r)]
    assert dense_records
    assert set(dense_records[0]) == expected_eval_keys


def test_periodic_eval_uses_current_curriculum_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Periodic reports mirror live K; post-hoc calibration alone uses terminal K=30."""
    calls: list[dict[str, object]] = []

    def fake_evaluate(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        calls.append(kwargs)
        return SimpleNamespace(metrics={"constraint_loss": 0.25})

    monkeypatch.setattr(training_loop, "evaluate_identifiability", fake_evaluate)
    trainer = Trainer(
        tiny_model(),
        tiny_dataset(),
        tiny_config(
            tmp_path,
            rollout_len=30,
            rollout_curriculum=PAPER_ROLLOUT_CURRICULUM,
            eval_every=1,
        ),
        eval_dataset=tiny_dataset(6),
    )
    trainer.successful_updates = 15_000
    assert trainer._eval_step() == {"eval/constraint_loss": 0.25}
    assert calls == [
        {
            "batch_size": 4,
            "device": "cpu",
            "context_len": 2,
            "lambda_logit": 1e-3,
            "rollout_len": 5,
            "lambda_roll": 1.0,
        }
    ]


def test_eval_requires_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="eval_every"):
        Trainer(tiny_model(), tiny_dataset(), tiny_config(tmp_path, eval_every=1))


def test_dataset_smaller_than_one_batch_raises(tmp_path: Path) -> None:
    """drop_last=True on a too-small dataset yields ZERO batches.

    Before the guard this spun through empty epochs forever — regenerating
    data, never stepping, never erroring — so a misconfigured run looked like
    a slow one. It must fail at construction instead.
    """
    with pytest.raises(ValueError, match="yields no batches"):
        Trainer(tiny_model(), tiny_dataset(4), tiny_config(tmp_path, batch_size=8))
