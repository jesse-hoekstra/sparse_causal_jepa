"""End-to-end training tests: dual controller, smoke, resume, guards, eval."""

from pathlib import Path

import pytest
import torch

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


def tiny_dataset(num_episodes: int = 8) -> BounceDataset:
    return BounceDataset(
        num_episodes=num_episodes,
        clip_len=4,
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


def test_rollout_weight_warms_up_linearly(tmp_path: Path) -> None:
    """Continuation reaches the configured coefficient on the declared update."""
    trainer = Trainer(
        tiny_model(),
        tiny_dataset(),
        tiny_config(tmp_path, steps=4, lambda_roll=2.0, lambda_roll_warmup_steps=4),
    )
    batches = trainer._batches()
    expected = (0.5, 1.0, 1.5, 2.0)
    for step, coefficient in enumerate(expected, start=1):
        metrics = trainer._train_step(next(batches))
        assert metrics["schedule/lambda_roll"] == pytest.approx(coefficient)
        assert metrics["loss/rollout"] == pytest.approx(
            coefficient * metrics["loss/rollout_raw"], rel=1e-6
        )
        trainer.step = step


def test_branch_gradient_metrics_are_raw_and_weighted(tmp_path: Path) -> None:
    """The applied rollout norm is exactly the raw chain norm times its schedule weight."""
    trainer = Trainer(
        tiny_model(),
        tiny_dataset(),
        tiny_config(tmp_path, steps=1, lambda_roll=2.0, lambda_roll_warmup_steps=4),
    )
    metrics = trainer._train_step(next(trainer._batches()))
    assert metrics["health/grad_norm_tf"] > 0.0
    assert metrics["health/grad_norm_rollout_raw"] > 0.0
    assert metrics["health/grad_norm_rollout"] == pytest.approx(
        metrics["schedule/lambda_roll"] * metrics["health/grad_norm_rollout_raw"],
        rel=1e-6,
    )


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
    """Checkpoint at step 2, resume, and reproduce steps 3-4 bit-for-bit."""
    config_a = tiny_config(tmp_path / "a", steps=4)
    trainer_a = Trainer(tiny_model(), tiny_dataset(), config_a)
    final_a = trainer_a.train()

    config_b2 = tiny_config(tmp_path / "b", steps=2)
    trainer_b = Trainer(tiny_model(), tiny_dataset(), config_b2)
    trainer_b.train()
    config_b4 = tiny_config(tmp_path / "b", steps=4)
    trainer_b4 = Trainer(tiny_model(), tiny_dataset(), config_b4)
    trainer_b4.load_checkpoint(tmp_path / "b" / "last.pt")
    final_b = trainer_b4.train()

    for key, value in final_a.items():
        assert final_b[key] == pytest.approx(value, rel=1e-5), key


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
