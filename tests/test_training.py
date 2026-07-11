"""End-to-end tests: SCJepa composite, Lagrangian controller, training smoke + resume."""

from pathlib import Path

import pytest
import torch

from scjepa.data import RandomClipDataset
from scjepa.models import SCJepa
from scjepa.models.jepa import build_scjepa
from scjepa.training import SparsityLagrangian, TrainConfig, Trainer

RES = 64


def tiny_model() -> SCJepa:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    return build_scjepa(
        resolution=RES,
        num_slots=3,
        slot_size=16,
        slot_mlp_size=32,
        num_iterations=1,
        enc_channels=(3, 8, 8),
        enc_out_channels=16,
        pooling_heads=2,
        spartan_layers=1,
        spartan_embed_dim=None,
        spartan_mlp_hidden=32,
        spartan_mlp_layers=2,
    )


def tiny_config(out_dir: Path, steps: int = 3) -> TrainConfig:
    return TrainConfig(
        steps=steps,
        batch_size=2,
        num_projections=16,
        sparsity_tau=0.5,
        log_every=1,
        checkpoint_every=1000,
        out_dir=str(out_dir),
        seed=0,
    )


def test_scjepa_forward_contract() -> None:
    model = tiny_model()
    frames = torch.randn(2, 4, 3, RES, RES)  # Th=3 context + 1 target
    out = model(frames)
    assert out.prediction.shape == (2, 3, 16)
    assert out.target_slots.shape == (2, 3, 16)
    assert out.context_slots.shape == (2, 3, 3, 16)  # (B, Th, N, d)
    assert out.causal_params.shape == (2, 3, 16)
    assert out.kinematic_state.shape == (2, 3, 16)
    assert out.path_matrix.shape == (2, 6, 6)
    with pytest.raises(ValueError, match="L>=2"):
        model(torch.randn(2, 1, 3, RES, RES))  # needs at least context + target


def test_lagrangian_dual_dynamics() -> None:
    """λ must grow while error > τ (sparsity off) and shrink once error < τ."""
    controller = SparsityLagrangian(tau=1.0, step_size=0.1, lambda_init=10.0, momentum=0.0)
    start = controller.log_lambda.clone()
    controller.update(torch.tensor(3.0))  # error above target
    assert controller.log_lambda > start
    high = controller.log_lambda.clone()
    for _ in range(5):
        controller.update(torch.tensor(0.1))  # error below target
    assert controller.log_lambda < high
    weight = SparsityLagrangian(tau=1.0, lambda_init=100.0).penalty_weight
    torch.testing.assert_close(weight, torch.tensor(0.01))
    # λ stays inside its clamp even under sustained one-sided error (D12 lesson:
    # an unbounded dual runs to 1e13+ and stops being responsive).
    clamped = SparsityLagrangian(
        tau=1.0, step_size=10.0, lambda_init=10.0, momentum=0.0, lambda_max=100.0
    )
    for _ in range(50):
        clamped.update(torch.tensor(5.0))
    assert torch.exp(clamped.log_lambda) <= 100.0 + 1e-4


def test_training_smoke(tmp_path: Path) -> None:
    """Full objective end-to-end on CPU: finite losses, all terms present."""
    dataset = RandomClipDataset(num_clips=4, clip_len=3, resolution=RES, seed=1)
    trainer = Trainer(tiny_model(), dataset, tiny_config(tmp_path))
    metrics = trainer.train()
    expected = {
        "loss/total",
        "loss/pred",
        "loss/reg",
        "loss/sparsity",
        "sparsity/lambda",
        "sparsity/path_density",
        "health/target_slot_std_mean",
        "health/target_slot_std_min",
        "health/grad_norm",
    }
    assert expected <= metrics.keys()
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert metrics["loss/sparsity"] > 0  # hard adjacencies always leave self-paths
    assert (tmp_path / "last.pt").exists()


def test_resume_is_exact(tmp_path: Path) -> None:
    """Save at step 2, continue to 4; reload at 2, continue to 4 — identical."""
    dataset = RandomClipDataset(num_clips=4, clip_len=3, resolution=RES, seed=1)

    trainer_a = Trainer(tiny_model(), dataset, tiny_config(tmp_path / "a", steps=2))
    trainer_a.train()
    trainer_a.save_checkpoint(tmp_path / "step2.pt")
    trainer_a.config.steps = 4
    trainer_a.train()
    final_a = {k: v.clone() for k, v in trainer_a.model.state_dict().items()}

    trainer_b = Trainer(tiny_model(), dataset, tiny_config(tmp_path / "b", steps=4))
    trainer_b.load_checkpoint(tmp_path / "step2.pt")
    trainer_b.train()
    final_b = trainer_b.model.state_dict()

    assert final_a.keys() == final_b.keys()
    for key, value in final_a.items():
        torch.testing.assert_close(value, final_b[key], msg=f"mismatch in {key}")


def test_sparsity_ablation_toggle(tmp_path: Path) -> None:
    """±sparsity is a config flag: disabled ⇒ λ never updated, term not in total."""
    dataset = RandomClipDataset(num_clips=4, clip_len=3, resolution=RES, seed=1)
    config = tiny_config(tmp_path, steps=2)
    config.sparsity_enabled = False
    trainer = Trainer(tiny_model(), dataset, config)
    metrics = trainer.train()
    assert trainer.lagrangian.ma_error == 0.0  # dual never stepped
    expected_total = metrics["loss/pred"] + config.lambda_reg * metrics["loss/reg"]
    assert abs(metrics["loss/total"] - expected_total) < 1e-5 * abs(expected_total)


def test_periodic_eval_logs_metrics(tmp_path: Path) -> None:
    """eval_every emits held-out eval/* metrics through the logger (W&B path)."""
    from scjepa.data import BounceDataset
    from scjepa.models.state_jepa import build_state_jepa

    class CaptureLogger:
        def __init__(self) -> None:
            self.records: list[tuple[int, dict[str, float]]] = []

        def log(self, step: int, metrics: dict[str, float]) -> None:
            self.records.append((step, metrics))

    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    model = build_state_jepa(
        slot_size=16,
        pooling_heads=2,
        spartan_layers=1,
        spartan_embed_dim=None,
        spartan_mlp_hidden=32,
        spartan_mlp_layers=2,
    )
    dataset = BounceDataset(num_episodes=6, clip_len=6, num_balls=3, seed=1, render=False)
    eval_dataset = BounceDataset(num_episodes=4, clip_len=6, num_balls=3, seed=99, render=False)
    config = tiny_config(tmp_path, steps=2)
    config.input_key = "states"
    config.context_len = 4
    config.eval_every = 1
    logger = CaptureLogger()
    Trainer(model, dataset, config, logger, eval_dataset=eval_dataset).train()
    eval_records = [m for _, m in logger.records if any(k.startswith("eval/") for k in m)]
    assert len(eval_records) == 2  # one per step with eval_every=1
    for record in eval_records:
        assert "eval/mcc" in record
        assert "eval/shd_param" in record
    bad = tiny_config(tmp_path, steps=1)
    bad.eval_every = 1
    bad.input_key = "states"
    with pytest.raises(ValueError, match="eval_dataset"):
        Trainer(model, dataset, bad)
