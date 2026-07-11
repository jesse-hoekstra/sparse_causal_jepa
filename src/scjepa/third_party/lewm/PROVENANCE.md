# Provenance: lewm (LeJEPA world-model reference)

## Upstream

- **URL:** <https://github.com/lucas-maes/le-wm>
- **Commit SHA:** `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`
- **License:** MIT (Lucas Maes, 2026) — copied verbatim as `LICENSE`.
- **Files taken:** `jepa.py`, `module.py`.

## Role: REFERENCE, not imported

These files are vendored as the stable in-repo reference for adapting the JEPA training
process (loss assembly, EMA-free joint training) into `scjepa.training`. Nothing in
`src/scjepa` imports them.

**Why `train.py` was NOT vendored (finding, 2026-07-09):** upstream's training loop is
not plain PyTorch — it sits on `lightning`, `stable_pretraining`, and
`stable_worldmodel`. Vendoring it would drag three framework dependencies into the
project for ~145 lines whose JEPA-specific content is small. The loop *pattern* it
implements — one optimizer step over encoder+predictor jointly, loss assembled as
`loss = pred_loss + lambd * regularizer(emb)` — is what `scjepa.training` re-expresses
in the project's own plain-PyTorch/Hydra stack (module 5). The D3 evidence (regularizer
is one swappable module call) is from this file and still holds.

`module.py` contains upstream's own `SIGReg` implementation; we do NOT use it — the
regularizers come from `third_party/visreg/` (official implementation, D3). It is kept
here so the adaptation can be diffed against what le-wm actually does.

## Local changes

1. `jepa.py`, `module.py` — **verbatim, no changes**.
