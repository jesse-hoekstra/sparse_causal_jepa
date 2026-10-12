# Repository instructions

Implement the current interfaces in `README.md` and `docs/architecture.md`.
The architecture is framewise encoding, one context-derived parameter vector,
coordinate-level finite-lag SPARTAN dynamics, and a complete frozen EMA encoder.
Keep state, action, and parameter dimensions independent. Preserve indexing,
attached parameter reuse, residual graph accounting, and successful-update-only EMA.

Future experiments supply data and evaluation through these interfaces. Do not
reintroduce object slots, recurrent state encoding, fixed scientific protocols,
physical meanings for latent coordinates, or decoder losses without user direction.
Distinguish paper assumptions from empirical guarantees and implementation choices.

Reuse existing components and retain third-party provenance/licenses. Preserve
local datasets, checkpoints, outputs, logs, and unrelated untracked work.
Run focused tests, the synthetic CPU smoke, Ruff format/lint, and strict Pyright.
No long or GPU training run is needed for architecture changes.

Scientific DGPs live under `scjepa.data`; follow `docs/data_generation.md` for
reference equations, output-row graph orientation, oldest-first lags, and the
unused spring action-zero slot. Keep labels and observation-map reference
utilities outside training items. Preserve addressable sampling and dataset
identity validation; do not silently change distributions or renderer precision.

The controlled-vector protocol lives in `scjepa.experiments`; follow
`docs/controlled_experiment.md`. The default sequence is dense tau calibration, baseline, full on complementary
data and one seed. Baseline/full differ only in parameter sparsity. Use local
content-dependent SPARTAN gates, active logit regularization and GECO for the
sparse conditions. Preserve the shared calibrated bound and controller resume.
Report MCC and aligned routing-support SHD; keep the dashboard concise.
Use SIGReg only. Keep calibration, validation and test separate, preserve paired
initialization/data and fixed observation maps, and validate checkpoint/run
provenance. An ordinary MLP plus empirical probes does not guarantee injectivity.
Use tiny CPU or offline-W&B checks; do not launch full scientific runs implicitly.
