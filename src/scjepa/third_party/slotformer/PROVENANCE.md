# Provenance: slotformer (SAVi implementation)

## Upstream

- **URL:** <https://github.com/pairlab/SlotFormer>
- **Commit SHA:** `10a6d9b7bc05ef2397ce2586da3fc0212f8c47aa`
- **License:** MIT (Ziyi Wu, 2022) — copied verbatim as `LICENSE`.
- **Files taken:** `slotformer/base_slots/models/{savi.py, utils.py, predictor.py}`.

Secondary upstream for `nerv_shim.py` (four names inlined, see below):

- **URL:** <https://github.com/Wuziyi616/nerv>
- **Commit SHA:** `5709625763c424a8b81b06c8cc2724d6454e688c`
- **License:** MIT (Ziyi Wu, 2022) — copied verbatim as `LICENSE.nerv`.

## Why this codebase

D2 (docs/decisions.md): PyTorch SAVi for the context/target encoders. SlotFormer is the
established SAVi→dynamics pipeline on CLEVRER (our benchmark), MIT-licensed, and its SAVi
is a proven training implementation. The official google-research/slot-attention-video
repo (JAX, archived) remains the correctness reference, not a vendoring source.

## Local changes

1. `savi.py` — import of `nerv.training.BaseModel` and
   `nerv.models.{deconv_out_shape, conv_norm_act, deconv_norm_act}` replaced by
   `from .nerv_shim import ...`. Rationale: `nerv` is a full training framework; vendoring
   it for four names violates minimal vendoring (D5). Behavior-identical (shim copied from
   nerv source; `BaseModel`'s extras are training hooks never called here).
2. `savi.py` — `_build_decoder()` is now only called when
   `loss_dict['use_post_recon_loss']` is true, and the `assert self.use_post_recon_loss`
   in `_build_loss()` was removed. Rationale: this project trains **no reconstruction
   objective** (D7); encoder-only use should not carry ~1M dead decoder parameters.
   Upstream behavior is unchanged when the flag is true.
3. `nerv_shim.py` — **new file**, not upstream: `BaseModel` reduced to `nn.Module`;
   `deconv_out_shape` copied verbatim; `conv_norm_act`/`deconv_norm_act` copied minus
   options savi.py never uses (dilation/groups/dim, activation set reduced to ''/'relu';
   unsupported values still raise).
4. `__init__.py` — **new file**, not upstream: re-exports `SlotAttention`, `StoSAVi`.
5. `utils.py`, `predictor.py` — **verbatim, no changes**.
6. `savi.py` — `_build_predictor()` is only called when `pred_dict is not None`, and
   `_reset_rnn()` no-ops in that case. Rationale: the slot predictor is only invoked from
   the second frame onward; the D9 target encoder consumes a single frame, and building
   the predictor there would carry permanently untrained parameters. Upstream behavior is
   unchanged when `pred_dict` is provided.

## Known upstream quirks (kept — minimal-diff policy; handled/documented in the wrapper)

- `StoSAVi` is SlotFormer's *stochastic* SAVi. With `kld_method='none'` it is
  deterministic, but it still inserts a `kernel_dist_layer` MLP between the slot
  predictor output and the slot-attention input, and a (dead) `prior_slot_layer` —
  both absent in official SAVi. Kept for weight-compat/minimal diff; flagged in the
  wrapper docstring as a deviation from Kipf et al.'s architecture.
- `encode()`'s CNN output resolution is hardcoded to `(64, 64)`: input must be 64×64
  (stride-1 root) or 128×128 (stride-2 root). The wrapper validates this.
