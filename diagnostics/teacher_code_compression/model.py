"""Declared code-input bottleneck and the unchanged teacher-distillation objective."""

from typing import Any

import torch
from torch import Tensor, nn

from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.multiteacher_curriculum.forward import Heads
from scjepa.models.model import cross_episode_parameters
from scjepa.training.rng import preserve_rng


class CompressionEncoder(nn.Module):
    """Map one context's normalized eighteen teacher coordinates to six coordinates."""

    def __init__(self) -> None:
        """Use biased 18→128→128→6 layers with GELU between layers."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(18, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 6)
        )

    def forward(self, features: Tensor) -> Tensor:
        """Process every context independently, with no pooling over views or groups."""
        if features.shape[-1] != 18:
            raise ValueError("Compression requires eighteen individual-context teacher coordinates")
        return self.network(features)


def initialize_modules(
    seed: int, *, dtype: torch.dtype = torch.float32
) -> tuple[CompressionEncoder, Heads, dict[str, Any]]:
    """Initialize the bottleneck at seed and identical curriculum heads at seed+1."""
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(seed)
        encoder = CompressionEncoder().to(dtype=dtype)
        torch.default_generator.manual_seed(seed + 1)
        heads = Heads().to(dtype=dtype)
    return (
        encoder,
        heads,
        {
            "encoder_seed": seed,
            "heads_seed": seed + 1,
            "policy": (
                "Independent CPU manual seeds, native Linear initialization, then source dtype"
            ),
            "encoder_architecture": "18-128-128-6 biased Linear/GELU, no context averaging",
            "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
            "encoder_sha256": module_fingerprint(encoder),
            "heads_sha256": module_fingerprint(heads),
        },
    )


def training_loss(
    encoder: CompressionEncoder,
    heads: Heads,
    teacher_inputs: Tensor,
    targets: Tensor,
    objective: AcquisitionObjective,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Keep original balanced target MSE, SIGReg.4, consistency1 and projection RNG."""
    if (
        teacher_inputs.ndim != 3
        or teacher_inputs.shape[1:] != (10, 18)
        or targets.shape != (*teacher_inputs.shape[:2], 3, 6)
        or teacher_inputs.requires_grad
        or targets.requires_grad
        or not bool(torch.isfinite(teacher_inputs).all())
        or not bool(torch.isfinite(targets).all())
        or teacher_inputs.device != targets.device
        or teacher_inputs.dtype != targets.dtype
    ):
        raise ValueError(
            "Require detached finite individual-context features and matched LOO targets"
        )
    codes = encoder(teacher_inputs)
    cross_episode_parameters(codes, training=encoder.training)
    torch.randn(6, objective.projections, device=codes.device, dtype=codes.dtype)
    regularizer = objective.regularizer(codes.transpose(0, 1))
    consistency = 2 * codes.var(dim=1, correction=1).mean()
    errors = (heads(codes) - targets).square().mean((0, 1, 3))
    reconstruction = errors.mean()
    loss = reconstruction + 0.4 * regularizer + consistency
    return loss, {
        "distillation": reconstruction,
        **{f"teacher_{index + 1}_mse": value for index, value in enumerate(errors)},
        "parameter_regularizer": regularizer,
        "parameter_consistency": consistency,
        "optimized_total": loss,
    }
