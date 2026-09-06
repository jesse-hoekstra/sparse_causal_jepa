"""Two-process integration of real visual training, EMA, GECO, and checkpoint RNG."""

import math
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.multiprocessing as mp

from scjepa.data.bounce import BounceDataset
from scjepa.models.visual_to_visual import VisualToVisualOutput, build_visual_to_visual
from scjepa.training.distributed import (
    destroy_distributed,
    gather_rank_objects,
    initialize_distributed,
    rank,
)
from scjepa.training.loop import TrainConfig
from scjepa.training.visual_to_visual import VisualToVisualTrainer


class _RankVarianceTrainer(VisualToVisualTrainer):
    """Report deliberately unequal variances without changing predictive tensors."""

    def _forward(self, batch: dict[str, torch.Tensor]) -> VisualToVisualOutput:
        output = super()._forward(batch)
        return output._replace(target_variance=torch.tensor(1e-8 if rank() == 0 else 100.0))


def _trainer(directory: str, steps: int = 2) -> VisualToVisualTrainer:
    torch.manual_seed(7)  # pyright: ignore[reportUnknownMemberType]
    model = build_visual_to_visual(
        num_slots=2,
        slot_size=8,
        state_dim=8,
        param_encoder_dim=8,
        param_encoder_heads=2,
        max_history=8,
        spartan_layers=1,
        spartan_embed_dim=16,
        spartan_mlp_hidden=16,
        spartan_mlp_layers=2,
    )
    dataset = BounceDataset(
        num_episodes=8,
        clip_len=4,
        num_balls=2,
        render=True,
        uniform_appearance=True,
        render_radius_from_mass=False,
    )
    return _RankVarianceTrainer(
        model,
        dataset,
        TrainConfig(
            steps=steps,
            batch_size=4,
            context_len=2,
            num_rollout_t2_anchors=1,
            oe_eval_horizon=2,
            lambda_logit=1e-5,
            sparsity_tau=0.5,
            grad_skip_threshold=1e6,
            # Exercise one unlogged update before gathering final metrics.
            log_every=2,
            checkpoint_every=1,
            out_dir=directory,
        ),
    )


def _visual_worker(worker: int, port: int, directory: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE="2",
        RANK=str(worker),
        LOCAL_RANK=str(worker),
    )
    torch.set_num_threads(1)
    initialize_distributed("cpu")
    try:
        trainer = _trainer(directory)
        target_before = next(trainer.model.target.parameters()).detach().clone()
        metrics = trainer.train()
        assert metrics["health/skipped_steps"] == 0
        assert metrics["train/loss_rollout_t2_raw"] > 0
        assert metrics["train/seconds_per_step"] > 0
        # The raw global predictive/logit mean determines GECO, even when
        # different ranks report wildly different representation variances.
        assert math.isclose(
            metrics["sparsity/constraint"],
            metrics["train/loss_teacher_forcing"]
            + metrics["train/loss_rollout_t2_weighted"]
            + metrics["loss/logit"],
            rel_tol=1e-6,
        )
        assert metrics["collapse/target_variance"] == metrics["collapse/target/content_var"]
        assert "train/grad_norm_teacher_forcing" not in metrics
        assert not torch.equal(target_before, next(trainer.model.target.parameters()))
        assert all(parameter.grad is None for parameter in trainer.model.target.parameters())
        # train/save_checkpoint must return only after rank-zero writing finishes.
        path = Path(directory) / "last.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["world_size"] == 2
        assert payload["global_batch_size"] == 4
        assert len(payload["rank_rng"]) == 2
        assert not torch.equal(
            payload["rank_rng"][0]["rng_torch"], payload["rank_rng"][1]["rng_torch"]
        )
        # Both online and EMA replicas must equal rank zero's portable model;
        # the dual must have the same value on every process as well.
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, payload["model"][key], rtol=0, atol=0)
        torch.testing.assert_close(
            trainer.lagrangian.log_lambda, payload["lagrangian"]["log_lambda"]
        )

        expected_rng = torch.get_rng_state().clone()
        resumed = _trainer(directory, steps=3)
        resumed.load_checkpoint(path)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng)
        assert resumed.step == 2
        # A rejection on just rank one must leave *every* replica and its EMA
        # and dual unchanged. The other worker deliberately has a loose bound.
        resumed.config.grad_skip_threshold = 0.0 if worker == 1 else 1e6
        before = {key: value.clone() for key, value in resumed.model.state_dict().items()}
        dual_before = resumed.lagrangian.log_lambda.clone()
        rejected = resumed.train()
        assert rejected["health/skipped_steps"] == 1
        for key, value in resumed.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        torch.testing.assert_close(resumed.lagrangian.log_lambda, dual_before, rtol=0, atol=0)
        assert gather_rank_objects(resumed.total_skips) == [1, 1]

        # Each rank's target is constant; all content variation comes from the
        # different episode means. Monitoring must gather first, rather than
        # average local variance or trust the output's diagnostic scalar.
        suffix_states = torch.full((2, 4, 2, 8), worker * 4.0)
        diagnostic_output = cast(
            VisualToVisualOutput,
            SimpleNamespace(
                target=torch.zeros(4, 2, 8),
                causal_params=torch.zeros(2, 2, 1),
                context_states=suffix_states,
                target_states=suffix_states,
                target_variance=torch.tensor(1e-8 if worker == 0 else 100.0),
                target_assignment=torch.arange(2).expand(2, -1),
            ),
        )
        diagnostics = resumed._extra_metrics(diagnostic_output)  # pyright: ignore[reportPrivateUsage]
        assert diagnostics["collapse/target_variance"] == 4.0
        assert diagnostics["collapse/target/content_var"] == 4.0
        assert diagnostics["collapse/target/temporal_var"] == 0.0
    finally:
        destroy_distributed()


def test_two_rank_visual_training_synchronizes_ema_geco_skips_and_resume(tmp_path: Path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
    context = mp.spawn(_visual_worker, args=(port, str(tmp_path)), nprocs=2, join=False)  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
    assert context is not None
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("two-rank visual training did not finish within 90 seconds")
    finally:
        for process in context.processes:
            assert process is not None
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            assert process is not None
            process.join(timeout=5)
