"""Attach the fixed TRAIN teacher to the native synthetic sparse-continuation loop."""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import Tensor

from diagnostics.masked_mlp_learned import run as runner
from diagnostics.masked_mlp_learned.continuation import ContinuationConfig, continue_training
from diagnostics.shared.io import read_json, sha
from diagnostics.shared.sampling import GroupSampler
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.loop import TrainingObjective

from . import consensus_gradient as common


@torch.no_grad()
def teacher_bank(parent: Path, device: torch.device) -> tuple[Tensor, Tensor]:
    """Encode the parent's immutable raw TRAIN episodes without reading any training labels."""
    record = read_json(parent / "config.json")["config"]
    data = runner.load_data(Path(record["data_dir"]))[0]["train"]
    model = runner.build_control("full", record["seed"], 6, predictor_activation="gelu").to(device)
    saved = cast(
        dict[str, Any], torch.load(parent / "last.pt", map_location="cpu", weights_only=False)
    )
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    outputs: list[Tensor] = []
    for start in range(0, len(data), record["batch_size"]):
        idx = torch.arange(start, min(len(data), start + record["batch_size"]))
        x, a = (
            data.tensors[key][idx].to(device).flatten(0, 1) for key in ("observations", "actions")
        )
        length = model.context_len
        z = model.online_encoder.encode_sequence(x)
        theta = model.parameter_encoder(z[:, :length], a[:, : length - 1])
        outputs.append(theta.reshape(len(idx), -1, theta.shape[-1]))
    bank = torch.cat(outputs)
    return bank, common.whitening(bank)


@contextmanager
def training_mode(parent: Path, mode: str, device: torch.device) -> Generator[None, None, None]:
    """Inject only an encoder-gradient contribution between zero_grad and native backward."""
    bank, matrix = teacher_bank(parent, device)
    source = Path(__file__)
    methods = {"runtime": sha(source), "math": sha(Path(common.__file__))}
    native = runner.forward_loss
    specification = runner._scientific_config  # pyright: ignore[reportPrivateUsage]
    indices: Tensor | None = None

    class Sampled(GroupSampler):
        def next(self) -> Tensor:
            nonlocal indices
            indices = super().next()
            return indices

    def scientific(config: runner.TrainConfig) -> dict[str, Any]:
        original = specification(config)
        if isinstance(config, ContinuationConfig):
            original["frozen_teacher"] = {
                "target": mode,
                "source": methods,
                "teacher_sha256": config.continuation_checkpoint_sha256,
                "units": "full-rank TRAIN code covariance",
                "auxiliary_gradient_norm_cap": "native forecast encoder gradient norm",
            }
        return original

    def forward(
        model: SCJEPA,
        batch: Mapping[str, Tensor],
        objective: TrainingObjective,
        *,
        forward_seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], ModelOutput]:
        loss, terms, output = native(model, batch, objective, forward_seed=forward_seed)
        if model.training and torch.is_grad_enabled():
            if indices is None or len(indices) != output.parameters.shape[0]:
                raise RuntimeError("student and frozen teacher minibatches disagree")
            aux = common.anchor(output.parameters, bank[indices], matrix, mode)
            terms.update(
                common.add_capped_gradient(
                    terms["prediction"], aux, tuple(model.parameter_encoder.parameters())
                )
            )
        return loss, terms, output

    with (
        patch.object(runner, "forward_loss", forward),
        patch.object(runner, "GroupSampler", Sampled),
        patch.object(runner, "_scientific_config", scientific),
    ):
        yield


def train_child(
    parent: Path, out: Path, mode: str, *, steps: int, device: str, resume: bool
) -> dict[str, Any]:
    """Fork native optimizer, EMA, GECO and random state using the standard continuation."""
    if mode not in {"same", "other"}:
        raise ValueError("Unknown teacher policy")
    with training_mode(parent, mode, torch.device(device)):
        return continue_training(
            parent,
            out,
            additional_steps=steps,
            device=device,
            threads=1,
            wandb_group="icml-review-synthetic-consensus",
            resume=resume,
        )
