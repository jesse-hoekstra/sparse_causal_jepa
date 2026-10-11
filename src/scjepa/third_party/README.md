# Third-party provenance

`visreg/` retains the upstream SIGReg and VISReg implementations, unchanged,
with their exact source commit and CC BY-NC 4.0 attribution in
[PROVENANCE.md](visreg/PROVENANCE.md). Only SIGReg is enabled through the typed
adapter in `scjepa.losses`; the upstream VISReg file remains attributed but unused.

`lewm/` retains the MIT license and provenance for the training-loop pattern
adapted into this project. Its unused reference networks have been removed.

Vendored loss code is excluded from Ruff and Pyright to preserve upstream diffs;
local wrappers are checked. Record any future vendored modifications in the
corresponding provenance file and retain the upstream license notices.
