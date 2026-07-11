# scjepa.third_party — vendored reference code

Reuse-first policy (decision D5, `docs/decisions.md`): we vendor and minimally adapt proven
reference code rather than rewrite it, and prefer rewriting over depending on unmaintained
packages at import time. Vendored code lives *inside* the package (`src/scjepa/third_party/`)
so it ships with the wheel and imports as `scjepa.third_party.<name>` (D5 amendment, module 2).

## Convention

Each vendored codebase gets its own directory:

```
src/scjepa/third_party/<name>/
  LICENSE          # upstream license file, copied verbatim
  PROVENANCE.md    # see below
  ...              # the vendored source, adapted as little as possible
```

`PROVENANCE.md` must record:

1. **Upstream URL** — the repository the code was taken from.
2. **Commit SHA** — the exact upstream commit that was vendored.
3. **Local changes** — an itemized list of every modification made after copying
   (file, what changed, why). "None" is a valid and preferred entry.

Keep adaptations minimal and inside this list; anything substantial belongs in a typed, tested
wrapper under `src/scjepa/` instead.

## Lint/type exemption

This directory is excluded from ruff and pyright (configured in `pyproject.toml`) and from
pre-commit hooks (`exclude: ^src/scjepa/third_party/`). This keeps diffs against upstream
reviewable. Wrappers elsewhere in `src/scjepa/` are held to the full quality bar
(pyright strict, ruff, tests).

## Vendored / planned code and license situation

| Codebase | Status | Upstream | License | Obligation notes |
|---|---|---|---|---|
| slotformer | **vendored** (see `slotformer/PROVENANCE.md`) | <https://github.com/pairlab/SlotFormer> | MIT (+ MIT `nerv` excerpts) | Keep copyright + license notices (`LICENSE`, `LICENSE.nerv`). PyTorch SAVi encoder (D2). |
| lewm | **vendored, reference-only** (see `lewm/PROVENANCE.md`) | <https://github.com/lucas-maes/le-wm> | MIT | Keep copyright + license notice. `jepa.py`/`module.py` as adaptation reference; `train.py` NOT vendored (depends on lightning/stable_pretraining/stable_worldmodel). |
| visreg | **vendored** (see `visreg/PROVENANCE.md`) | <https://github.com/HaiyuWu/visreg> | **CC BY-NC 4.0** (README badge; upstream ships no LICENSE file) | **Non-commercial only.** Fine for this academic project; flag to Jesse before any commercial use. Attribution required. VISReg + SIGReg (D3). |
| SAVi (official) | reference only | <https://github.com/google-research/slot-attention-video> | Apache 2.0 | JAX, archived — correctness *reference* for our PyTorch SAVi (D2); consulted, not vendored. Keep NOTICE/license if any files are ever copied. |