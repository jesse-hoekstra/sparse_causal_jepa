# Coupled Duffing baseline screen

Validation-only visual dynamics diagnostic with two state/action lags, learned
single-context parameter codes, and fixed-open parameter gates. Input and
reference-label files are stored separately. The runner never opens training
labels or generates/evaluates a test set.

```bash
python -m diagnostics.coupled_duffing.run prepare --output-dir DATA
python -m diagnostics.coupled_duffing.run train --data-dir DATA --output-dir RUN --seed 0
```

Use an immutable checkout for each batch. Model settings, calibration,
checkpoints, source identity and evaluation assignments are retained with the
run. A low state MCC is informative only alongside successful state retention,
parameter acquisition and forecasting.

The discrete physical model follows the central-acceleration/forward-damping
construction described by [Kouw (2020)](https://arxiv.org/abs/2009.00845), extended
here to two coupled oscillators with independent parameter pairs. It is not an
exact finite-time ODE flow.
