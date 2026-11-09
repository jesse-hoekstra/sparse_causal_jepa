# Coupled Duffing baseline screen

Validation-only visual dynamics diagnostic with two state/action lags, learned
single-context parameter codes, and fixed-open parameter gates. Input and
reference-label files are stored separately. The runner never opens training
labels or generates/evaluates a test set.

```bash
python -m diagnostics.coupled_duffing.run prepare --output-dir DATA
python -m diagnostics.coupled_duffing.run train --data-dir DATA --output-dir RUN --seed 0
```

`experiment.py` provides the complete staged acquisition procedure: shared-feature
initialization, an overcomplete code, fixed-budget acquisition restarts selected
by training forecast loss, projection, adaptation and baseline training. Use its
`prepare`, `train`, `select` and `project` subcommands for separately recorded stages.
The staged runner accepts `--parameter-encoder temporal-cnn` for temporal context
features and `--cache-device cpu` for image caches larger than GPU memory.
`python -m diagnostics.coupled_duffing.visual_data --source-dir PHYSICAL --output-dir IMAGES`
renders an existing physical diagnostic cache without changing its trajectories.

Use an immutable checkout for each batch. Model settings, calibration,
checkpoints, source identity and evaluation assignments are retained with the
run. A low state MCC is informative only alongside successful state retention,
parameter acquisition and forecasting.

The discrete physical model follows the central-acceleration/forward-damping
construction described by [Kouw (2020)](https://arxiv.org/abs/2009.00845), extended
here to two coupled oscillators with independent parameter pairs. It is not an
exact finite-time ODE flow.
