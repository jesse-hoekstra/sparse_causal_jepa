"""Two CPU ranks verify generic optimizer/EMA synchronization and resume."""

import copy
import os
import sys
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor

from scjepa.data import TensorEpisodes
from scjepa.models.encoders import VectorEncoder
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.models.parameter_encoder import ParameterEncoder
from scjepa.models.spartan import Spartan
from scjepa.training import TrainConfig, Trainer
from scjepa.training.distributed import (
    destroy_distributed,
    gather_rank_objects,
    initialize_distributed,
    rank,
)


class _FailingLogger:
    def log(self, step: int, metrics: dict[str, float]) -> None:
        raise RuntimeError(f"logger rejected step {step} with {len(metrics)} metrics")


def _trainer(
    directory: str, identity: bool = False, dataset: TensorEpisodes | None = None
) -> Trainer:
    torch.manual_seed(7)  # pyright: ignore[reportUnknownMemberType]
    model = SCJEPA(
        VectorEncoder(4, 3, (8,)),
        ParameterEncoder(3, 2, dim=8, num_heads=2, max_history=4),
        Spartan(3, 2, num_layers=1, embed_dim=8, mlp_hidden_size=8, identity=identity),
        context_len=2,
        ema_decay=0.8,
    )
    return Trainer(
        model, TrainConfig(batch_size=4, accumulation_steps=2, out_dir=directory), dataset
    )


def _worker(worker: int, rendezvous: str, directory: str) -> None:
    os.environ.update(
        WORLD_SIZE="2",
        RANK=str(worker),
        LOCAL_RANK=str(worker),
        GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo",
    )
    torch.set_num_threads(1)
    # File rendezvous avoids host DNS lookup and TCPStore startup on developer
    # machines; production torchrun initialization remains independently tested.
    dist.init_process_group("gloo", init_method=rendezvous, rank=worker, world_size=2)
    initialize_distributed("cpu")
    try:
        trainer = _trainer(directory)
        batch = {"observations": torch.randn(2, 4, 4), "episode_ids": torch.arange(2) + worker * 2}
        initial_target = copy.deepcopy(trainer.model.target_encoder.state_dict())
        trainer.train_step(batch)
        assert trainer.optimizer_steps == 0
        torch.testing.assert_close(trainer.model.target_encoder.state_dict(), initial_target)
        trainer.train_step(batch)
        assert trainer.optimizer_steps == 1
        assert any(
            not torch.equal(initial_target[key], value)
            for key, value in trainer.model.target_encoder.state_dict().items()
        )
        path = Path(directory) / "shared.pt"
        trainer.save_checkpoint(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["world_size"] == 2
        assert len(payload["rank_rng"]) == 2
        assert not torch.equal(
            payload["rank_rng"][0]["rng_torch"], payload["rank_rng"][1]["rng_torch"]
        )
        torch.testing.assert_close(trainer.model.state_dict(), payload["model"], rtol=0, atol=0)
        expected_rng = torch.get_rng_state().clone()
        resumed = _trainer(directory)
        resumed.load_checkpoint(path)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        assert resumed.optimizer_steps == 1
        assert resumed.step == 2
        before = copy.deepcopy(resumed.model.state_dict())
        # A gradient threshold rejection on one rank invalidates both accumulated groups.
        resumed.config.grad_skip_threshold = 0 if worker == 1 else float("inf")
        resumed.train_step(batch)
        skipped = resumed.train_step(batch)
        assert skipped["health/skipped"] == 1
        torch.testing.assert_close(resumed.model.state_dict(), before, rtol=0, atol=0)
        assert gather_rank_objects(resumed.total_skips) == [1, 1]
        resumed.config.grad_skip_threshold = float("inf")
        original = resumed.objective.forward

        def bad_loss(output: ModelOutput) -> tuple[Tensor, dict[str, Tensor]]:
            total, terms = original(output)
            return (total * float("nan") if rank() == 1 else total), terms

        patch = pytest.MonkeyPatch()
        patch.setattr(resumed.objective, "forward", bad_loss)
        skipped = resumed.train_step(batch)
        patch.undo()
        assert skipped["health/skipped"] == 1
        torch.testing.assert_close(resumed.model.state_dict(), before, rtol=0, atol=0)
        resumed.train_step(batch)
        resumed.train_step(batch)
        assert resumed.optimizer_steps == 2
        assert gather_rank_objects(resumed.total_skips) == [2, 2]
        assert all(
            parameter.grad is None for parameter in resumed.model.target_encoder.parameters()
        )
        # Episode IDs must be distinct globally, not merely within each rank.
        resumed.config.objective.parameter_regularizer_weight = 1
        duplicate_ids = {"observations": batch["observations"], "episode_ids": torch.arange(2)}
        with pytest.raises(ValueError, match="distinct episodes"):
            resumed.train_step(duplicate_ids)
        # Identity mode genuinely leaves parameter inference and attention unused.
        # DDP discovers this from the loss, rather than unused returned diagnostics.
        identity = _trainer(directory, identity=True)
        for _ in range(4):
            identity.train_step(batch)
        assert identity.optimizer_steps == 2
        # Only rank zero evaluates; all ranks receive identical selection state
        # and participate in atomic best/last checkpoint creation.
        validation = _trainer(directory, dataset=TensorEpisodes(torch.zeros(8, 4, 4)))
        validation.config.steps = 6
        evaluated: list[int] = []

        def evaluate(model: SCJEPA) -> dict[str, float]:
            assert worker == 0
            assert model is validation.model
            assert not model.training
            assert validation.pending_microbatches == 0
            evaluated.append(validation.step)
            torch.randn(3)
            return {"validation/physical_prediction_nrmse": 1 / validation.step}

        validation.train(
            evaluate,
            validation_every=1,
            stop_requested=lambda: worker == 1 and validation.step >= 3,
        )
        assert validation.interrupted
        assert validation.step == 4
        assert gather_rank_objects(evaluated) == [[2], []]
        assert gather_rank_objects(validation.validation_state) == [
            validation.validation_state,
            validation.validation_state,
        ]
        assert validation.validation_state["best_step"] == 2
        assert torch.load(Path(directory) / "last.pt", weights_only=False)["step"] == 4

        def failing_evaluate(_model: SCJEPA) -> dict[str, float]:
            assert worker == 0
            raise ValueError("intentional rank-zero validation error")

        with pytest.raises(RuntimeError, match="validation failed on rank zero"):
            validation.train(failing_evaluate, validation_every=1)
        assert validation.model.training
        assert not validation.model.target_encoder.training
        # A checkpoint write failure must also release all ranks with an error.
        with pytest.raises(RuntimeError, match="checkpoint writing failed on rank zero"):
            validation.save_checkpoint(Path(directory) / "last.pt" / "impossible.pt")
        # A rank-zero logging failure must stop peers before their next backward.
        identity.dataset = TensorEpisodes(torch.zeros(4, 4, 4))
        identity.config.steps = identity.step + 2
        identity.logger = _FailingLogger()
        with pytest.raises(RuntimeError, match="metric logging failed on rank zero"):
            identity.train()
    finally:
        destroy_distributed()


def test_two_rank_accumulation_ema_skips_checkpoints_and_unused_paths(tmp_path: Path) -> None:
    rendezvous = (tmp_path / "rendezvous").as_uri()
    context = mp.spawn(_worker, args=(rendezvous, str(tmp_path)), nprocs=2, join=False)  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
    assert context is not None
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("two-rank generic training did not finish within 90 seconds")
    finally:
        for process in context.processes:
            assert process is not None
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
