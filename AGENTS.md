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
`docs/controlled_experiment.md`. The default sequence is dense tau calibration,
baseline, full on complementary data and one seed. The approved vector DGP modifies
the paper's coefficients and uses two four-frame episodes per independent parameter
draw (K=2, L=1), with independent initial states/actions and two-step autoregressive
prediction across episodes in both directions. Keep the partner parameter code
fixed and feed each prediction into the next step without detaching; average loss
over both steps and directions. Follow `docs/data_generation.md`. Dataset and batch
counts denote parameter groups. Apply SIGReg separately to each view across groups,
without treating two views of the same parameter as independent samples. Earlier
single-episode, one-step and teacher-forced protocols require fresh dense training
and tau calibration. Graph metrics and sparsity count each predictor call's direct
one-step routing graph, not composed paths across the rollout.
Baseline forces parameter-source columns to decoded latest-state receiver rows open in every SPARTAN layer during training and evaluation, with zero
parameter sparsity weight. Full learns those gates and adds the parameter penalty.
Both retain learned attention weights, all other learned gates, state/action
regularization, active logit regularization and GECO. Preserve the shared calibrated
bound, paired tensor initialization and controller resume. Both sparse runs start
from the exact dense best checkpoint bound to tau, including its EMA encoder,
with fresh optimizer, GECO controller and step counters. Final held-out test
evaluation and `--evaluate-only` use each condition's `last.pt`, including dense.
Validation-selected `best.pt` remains for dense calibration/warm starts and
validation diagnostics. Resume restores each sparse run's own checkpoint and
training state. Older protocols require a new
output directory because source/configuration provenance differs. An optional
`dense_reference_dir` reuses a compatible previous experiment's selected dense
checkpoint and original tau, running only baseline/full in the new output.
Keep the reference read-only and verify architecture, data, base objective,
calibration batching, observation map and checkpoint hash before reuse; older
source versions are allowed when these checks pass and the DGP version matches.
Report MCC and aligned routing-support SHD; keep the dashboard concise.
Use SIGReg only. Keep calibration, validation and test separate, preserve paired
initialization/data and fixed observation maps, and validate checkpoint/run
provenance. An ordinary MLP plus empirical probes does not guarantee injectivity.
Use tiny CPU or offline-W&B checks; do not launch full scientific runs implicitly.

The explicitly selected `independent_gaussian` experiment is a separate approved
DGP; follow `docs/independent_gaussian_experiment.md`. Use the nonlinear two-reflection
and radius-dependent rotation, raw Gaussian action coordinates, complementary
graphs only, K=3, T=5, L=1 and paired episodes with independent initial states/actions.
Keep the existing controlled-vector definitions intact. This DGP needs fresh dense
training and calibration; do not reuse an older family's checkpoint or tau.
