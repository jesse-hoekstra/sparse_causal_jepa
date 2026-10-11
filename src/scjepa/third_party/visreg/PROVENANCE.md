# Provenance: visreg (VISReg + SIGReg anti-collapse losses)

## Upstream

- **URL:** <https://github.com/HaiyuWu/visreg> (official VISReg implementation)
- **Commit SHA:** `074fb516935c8978fea2c930ae04909eb3697be3`
- **License:** **CC BY-NC 4.0** — declared via the README badge linking to
  <https://creativecommons.org/licenses/by-nc/4.0/>; the repo ships **no LICENSE file**
  (recorded as found, 2026-07-09). **Non-commercial use only**; attribution required.
  Attribution and non-commercial restrictions apply to reuse.
- **Files taken:** `visreg/losses/{visreg.py, sigreg.py}`.

## Use in this project

Only SIGReg is an active, explicitly weighted training choice. The retained VISReg
source is unused. The typed SIGReg wrapper uses online state samples and one
parameter vector per distinct episode.
Regularization does not establish information preservation or identifiability.

## Input contract (verified against source, savers beware)

`forward(z)` with `z` of shape `(G, B, D)`: all statistics (centering, per-dimension
std, sorted sliced projections / characteristic-function error) are over the **sample
axis `dim=1`**. `G` is a broadcast group axis. Both losses draw fresh random projection
directions each forward call, so the loss value is stochastic given fixed inputs.

## Local changes

1. `visreg.py`, `sigreg.py` — **verbatim, no changes**.
2. `__init__.py` — **new file**, not upstream: re-exports `VISReg`, `SIGReg`.
   (`swd.py`, `vicreg.py`, `barlow.py` from the same upstream directory were NOT
   vendored — not needed; they can be added later under the same provenance.)
