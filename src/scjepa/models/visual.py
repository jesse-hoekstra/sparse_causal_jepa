"""Shared visual front-end for the pixel experiments (experiments.pdf §6.3, Eqs. 41-89).

The recurrent SAVi encoder Q_psi maps a causal frame prefix to N posterior object
slots per timestep (Eq. 42), and a shared row-wise head g_omega maps each slot to
the latent state interface SPARTAN consumes (Eq. 87). The visual regimes use the
same architecture and train it separately; the visual-to-visual regime additionally keeps an EMA
copy of exactly this path as its target encoder (Eq. 109), which is why the two
modules are bundled into one object rather than wired up ad hoc per experiment.

The encoder is encoder-only: no image decoder and no pixel-reconstruction loss
(§6.3). Everything it learns arrives through the downstream prediction and
sparsity objectives.

Configured Bounce dimensions (Eqs. 45-50, 65): P = 64^2 spatial positions,
d_f = 128 feature width, conv channels 3->64->64->64->64, slot width d = 32,
L_SA = 2 correction iterations, Slot-Attention MLP width 256, predictor
Transformer L_tr = 2 / H_tr = 4 with FFN width 512, LSTM hidden 256.
"""

from typing import NamedTuple

from jaxtyping import Float
from torch import Tensor, nn

from scjepa.models.savi import SAViEncoder

__all__ = ["VisualState", "VisualStatePath"]


class VisualState(NamedTuple):
    """One causal pass over a frame prefix."""

    slots: Float[Tensor, "b t n d"]
    """Posterior object slots S~ (Eq. 42) — the parameter encoder's input."""
    states: Float[Tensor, "b t n s"]
    """Latent states S = g_omega(S~) (Eq. 87) — SPARTAN's state interface."""


class VisualStatePath(nn.Module):
    """Q_psi followed by g_omega: the complete visual-state path chi = (psi, omega).

    The visual-to-visual regime replicates this whole object as an EMA target encoder (Eq. 109),
    so the state head must live inside it rather than beside it — the target's
    representation is g_omega_bar(S~_tgt) (Eq. 113), not the raw target slots.
    """

    def __init__(
        self,
        num_slots: int = 5,
        slot_size: int = 32,
        state_dim: int = 32,
        resolution: int = 64,
        slot_mlp_size: int = 256,
        num_iterations: int = 2,
        enc_out_channels: int = 128,
        pred_num_layers: int = 2,
        pred_num_heads: int = 4,
        pred_ffn_dim: int = 512,
    ) -> None:
        """Build the encoder and the shared row-wise state head."""
        super().__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.state_dim = state_dim
        self.encoder = SAViEncoder(
            resolution=(resolution, resolution),
            num_slots=num_slots,
            slot_size=slot_size,
            slot_mlp_size=slot_mlp_size,
            num_iterations=num_iterations,
            enc_channels=(3, 64, 64, 64, 64),
            enc_out_channels=enc_out_channels,
            pred_num_layers=pred_num_layers,
            pred_num_heads=pred_num_heads,
            pred_ffn_dim=pred_ffn_dim,
            pred_rnn=True,
        )
        # Eq. 87: shared across tracks, so permuting the slot axis permutes the
        # latent states identically. No coordinate is designated as position or
        # velocity; the representation is learned only through the downstream
        # objectives and is checked afterwards with frozen held-out probes.
        self.state_head = nn.Linear(slot_size, state_dim)

    def forward(self, frames: Float[Tensor, "b t c h w"]) -> VisualState:
        """Run the causal recurrence once over the prefix (Eqs. 78/115)."""
        if frames.ndim != 5:
            raise ValueError(f"expected (B, T, C, H, W), got {tuple(frames.shape)}")
        slots = self.encoder(frames)
        return VisualState(slots=slots, states=self.state_head(slots))
