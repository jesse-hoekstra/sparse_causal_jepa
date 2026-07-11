"""SAVi encoder wrapper around the vendored SlotFormer implementation.

Wraps ``scjepa.third_party.slotformer.StoSAVi`` in **deterministic, encoder-only**
mode as the context/target encoder of the JEPA (D2): no decoder is built and no
reconstruction/KLD objective exists (D7) — training signal comes solely from the
predictive loss and the VISReg regularizer (D3), applied outside this module.

Symbol table (paper ↔ code):
    frames  x_{1:T}   (B, T, 3, H, W)  input video clip
    slots   S̃         (B, T, N, d)     slot history; S̃[:, -1] has seen all frames
    N = ``num_slots``, d = ``slot_size``, T = Th (context) or Th+1 (target branch)

Known deviations from official SAVi (Kipf et al., 2022), inherited from SlotFormer
(kept verbatim under the minimal-diff vendoring policy, D5 — see
``third_party/slotformer/PROVENANCE.md``):
    1. A small MLP (``kernel_dist_layer``) sits between the slot predictor output
       and the slot-attention input; official SAVi feeds the predictor output
       directly. Deterministic here (``kld_method='none'`` → the MLP's mean output
       is used, no sampling).
    2. A ``prior_slot_layer`` exists but is dead weight (never in the forward path).
"""

from typing import cast

import torch
from jaxtyping import Float
from torch import Tensor, nn

from scjepa.third_party.slotformer import StoSAVi


class SAViEncoder(nn.Module):
    """Sequential slot-attention encoder: video clip → slot history.

    Recurrent over time: slots at t are predicted from slots at t-1 (transformer
    +LSTM slot predictor) and corrected against frame-t features (slot attention).
    Weights are NOT shared between the context and target encoders — instantiate
    one ``SAViEncoder`` per branch (joint training, no EMA; see decisions D6/D7).
    The target branch consumes a single future frame (D9): build it with
    ``single_frame=True``, which skips the never-invoked slot predictor and
    reduces the encoder to per-image Slot Attention.
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (64, 64),
        num_slots: int = 7,
        slot_size: int = 128,
        slot_mlp_size: int = 256,
        num_iterations: int = 2,
        enc_channels: tuple[int, ...] = (3, 64, 64, 64, 64),
        enc_out_channels: int = 128,
        enc_kernel_size: int = 5,
        pred_num_layers: int = 2,
        pred_num_heads: int = 4,
        pred_ffn_dim: int = 512,
        pred_rnn: bool = True,
        single_frame: bool = False,
    ) -> None:
        """Build the encoder.

        Args:
            resolution: Input frame (H, W). The vendored CNN produces 64x64
                feature maps, so only 64x64 (stride-1 root) and 128x128
                (stride-2 root) inputs are supported upstream.
            num_slots: N, number of object slots.
            slot_size: d, slot embedding dimension.
            slot_mlp_size: Hidden size of the slot-attention update MLP.
            num_iterations: Slot-attention iterations per frame.
            enc_channels: CNN channel progression (first entry = input channels).
            enc_out_channels: Feature dimension fed to slot attention.
            enc_kernel_size: CNN kernel size.
            pred_num_layers: Transformer layers in the slot predictor.
            pred_num_heads: Attention heads in the slot predictor.
            pred_ffn_dim: Feed-forward dim in the slot predictor.
            pred_rnn: Wrap the predictor in an LSTM (SlotFormer default).
            single_frame: Accept exactly one frame per clip and build no slot
                predictor (D9 target encoder). The ``pred_*`` args are ignored.
        """
        super().__init__()
        if resolution[0] != resolution[1] or resolution[0] not in (64, 128):
            raise ValueError(f"vendored SAVi supports 64x64 or 128x128 inputs, got {resolution}")
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.single_frame = single_frame
        pred_dict = (
            None
            if single_frame
            else {
                "pred_type": "transformer",
                "pred_rnn": pred_rnn,
                "pred_norm_first": True,
                "pred_num_layers": pred_num_layers,
                "pred_num_heads": pred_num_heads,
                "pred_ffn_dim": pred_ffn_dim,
                "pred_sg_every": None,  # no stop-gradient anywhere (paper; D7)
            }
        )
        self._impl = StoSAVi(
            resolution=resolution,
            clip_len=0,  # only gates upstream's OOM-splitting path, which we bypass
            slot_dict={
                "num_slots": num_slots,
                "slot_size": slot_size,
                "slot_mlp_size": slot_mlp_size,
                "num_iterations": num_iterations,
                "kernel_mlp": True,
            },
            enc_dict={
                "enc_channels": tuple(enc_channels),
                "enc_ks": enc_kernel_size,
                "enc_out_channels": enc_out_channels,
                "enc_norm": "",
            },
            dec_dict={},  # decoder is never built (D7; PROVENANCE.md patch 2)
            pred_dict=pred_dict,  # None → no slot predictor (D9; patch 6)
            loss_dict={
                "use_post_recon_loss": False,  # encoder-only (D7)
                "kld_method": "none",  # deterministic SAVi
            },
        )

    def forward(self, frames: Float[Tensor, "b t c h w"]) -> Float[Tensor, "b t n d"]:
        """Encode a video clip into its slot history.

        Args:
            frames: Video clip, (B, T, C, H, W).

        Returns:
            Slot history S̃, (B, T, N, d). The last timestep has integrated all
            T frames via the recurrence.
        """
        if frames.ndim != 5:
            raise ValueError(f"expected (B, T, C, H, W), got shape {tuple(frames.shape)}")
        if self.single_frame and frames.shape[1] != 1:
            raise ValueError(f"single_frame encoder expects T=1, got T={frames.shape[1]}")
        # Fresh recurrent state per clip (the predictor's LSTM is stateful).
        self._impl._reset_rnn()  # pyright: ignore[reportPrivateUsage]
        _, slots, _ = cast(
            tuple[Tensor, Tensor, Tensor],
            self._impl.encode(frames),  # pyright: ignore[reportUnknownMemberType]
        )
        b, t = frames.shape[0], frames.shape[1]
        if slots.shape != (b, t, self.num_slots, self.slot_size):
            raise AssertionError(f"slot history has shape {tuple(slots.shape)}")
        return slots


__all__ = ["SAViEncoder"]


if __name__ == "__main__":
    # Smoke check: tiny forward + backward on CPU.
    enc = SAViEncoder(
        num_slots=3, slot_size=16, slot_mlp_size=32, enc_channels=(3, 8, 8), enc_out_channels=16
    )
    x = torch.randn(2, 4, 3, 64, 64, requires_grad=True)
    out = enc(x)
    out.mean().backward()
    print("slots:", tuple(out.shape), "| grad on input:", x.grad is not None)
