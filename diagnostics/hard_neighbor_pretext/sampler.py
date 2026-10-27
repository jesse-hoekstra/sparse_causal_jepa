"""Detached TRAIN-only group means select hard batches; never supply model features."""

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.group_contrastive_pretext.forward import cached_codes
from diagnostics.multiquery_control.data import module_fingerprint
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.models.model import SCJEPA
from scjepa.training.rng import preserve_rng

ANCHOR_NAMESPACE = 202610270501


@dataclass(frozen=True)
class Bank:
    """One immutable sampler snapshot, explicitly detached from encoder optimization."""

    means: Tensor
    refresh_step: int
    encoder_sha256: str

    def __post_init__(self) -> None:
        """Require finite detached CPUdouble six-dimensional lookup keys."""
        if (
            self.means.ndim != 2
            or self.means.shape[1] != 6
            or len(self.means) < 2
            or self.means.device.type != "cpu"
            or self.means.dtype != torch.float64
            or self.means.requires_grad
            or not bool(torch.isfinite(self.means).all())
            or self.refresh_step < 1
            or len(self.encoder_sha256) != 64
        ):
            raise ValueError("Invalid detached group bank")

    def payload(self) -> dict[str, Any]:
        """Bind exact means to refresh counter and encoder snapshot fingerprint."""
        receipt = {
            "refresh_step": self.refresh_step,
            "encoder_sha256": self.encoder_sha256,
            "means_sha256": tensor_fingerprint(self.means),
        }
        return {"means": self.means, **receipt, "identity": canonical_fingerprint(receipt)}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Bank":
        """Reject altered saved lookup values before resuming the deterministic sampler."""
        means = cast(Tensor, payload["means"])
        if tensor_fingerprint(means) != payload["means_sha256"]:
            raise ValueError("Persisted sampler bank changed")
        if (
            canonical_fingerprint(
                {key: payload[key] for key in ("refresh_step", "encoder_sha256", "means_sha256")}
            )
            != payload["identity"]
        ):
            raise ValueError("Persisted sampler bank metadata changed")
        return cls(means, int(payload["refresh_step"]), str(payload["encoder_sha256"]))


def refresh_due(step: int, warmup_steps: int = 5000, interval: int = 512) -> bool:
    """Refresh before the first hard update and every512 subsequent hard updates."""
    if step < 1 or warmup_steps < 0 or interval < 1:
        raise ValueError("Invalid hard-neighbor refresh schedule")
    return step > warmup_steps and (step - warmup_steps - 1) % interval == 0


@torch.no_grad()
def refresh_bank(
    model: SCJEPA, train: dict[str, Tensor], step: int, *, group_batch_size: int = 64
) -> Bank:
    """Encode all TRAIN contexts separately, then average only detached sampler keys."""
    if set(train) != {"online", "target", "actions"} or group_batch_size < 2:
        raise ValueError("Sampler reads only original TRAIN caches, no labels or evaluation split")
    groups = len(train["online"])
    if groups < 2:
        raise ValueError("Sampler requires multiple TRAIN groups")
    reference = next(model.parameters())
    modes = {module: module.training for module in model.modules()}
    before = module_fingerprint(model.parameter_encoder)
    pieces: list[Tensor] = []
    try:
        with preserve_rng(reference.device):
            model.parameter_encoder.eval()
            # Avoid a singleton final chunk (cached_codes intentionally requires B>=2).
            starts = list(range(0, groups, group_batch_size))
            if len(starts) > 1 and groups - starts[-1] == 1:
                starts.pop()
            for index, start in enumerate(starts):
                stop = starts[index + 1] if index + 1 < len(starts) else groups
                batch = {name: value[start:stop].to(reference) for name, value in train.items()}
                pieces.append(cached_codes(model, batch).mean(1).detach().cpu().double())
    finally:
        for module, was_training in modes.items():
            module.training = was_training
    if module_fingerprint(model.parameter_encoder) != before:
        raise RuntimeError("Bank refresh changed encoder weights")
    return Bank(torch.cat(pieces), step, before)


def hard_indices(bank: Bank, batch_size: int, seed: int, step: int) -> list[int]:
    """Uniform independent anchor plus nearest distinct groups, stable group-ID ties."""
    if not 2 <= batch_size <= len(bank.means) or seed < 0 or step < bank.refresh_step:
        raise ValueError("Invalid hard batch size, seed or bank timing")
    generator = torch.Generator().manual_seed(
        (seed * 1_000_003 + step + ANCHOR_NAMESPACE) % (2**63 - 1)
    )
    anchor = int(torch.randint(len(bank.means), (1,), generator=generator))
    distance = (bank.means - bank.means[anchor]).square().sum(-1)
    distance[anchor] = torch.inf
    closest = torch.argsort(distance, stable=True)[: batch_size - 1]
    return [anchor, *(int(value) for value in closest)]
