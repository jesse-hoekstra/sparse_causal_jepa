# Sparse causal JEPA

Code for the experiments of the accompanying paper: framewise state and context-parameter
learning with sparse finite-lag dynamics. It covers the synthetic experiment (Experiment 1), the
vision experiment (two bonded colloidal dimers filmed by an RGB camera) and the downstream
causal-inference application on the vision models.

Requires Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,analysis,logging]'
.venv/bin/python scripts/smoke.py
```

## Layout

| Path | Contents |
|---|---|
| `src/scjepa` | Models, losses, gates, GECO controller, evaluation; shared by both experiments |
| `diagnostics/shared` | Experiment 1 data generator and shared training utilities |
| `diagnostics/masked_mlp_learned` | Experiment 1: learned parameter codes, initialization, sparse training, test evaluation |
| `diagnostics/masked_mlp_oracle` | Experiment 1: masked-MLP model, oracle-parameter control, the paper's figures and tables |
| `diagnostics/vision_dimer` | Vision experiment and downstream application ([README](diagnostics/vision_dimer/README.md)) |
| `tests`, `scripts/smoke.py` | Behavioural tests and a CPU smoke check of all three model paths |

## Experiment 1 (synthetic)

The reported settings are those of the paper's hyperparameter table:

- 6 states, 24 actions, 6 parameters; rotation scale 0.2;
- 8,192 / 512 / 256 / 512 groups of 10 episodes of 12 frames;
- GELU masked-MLP predictor;
- joint 100k updates (code dimension 12), then 16 frozen-state acquisitions of 100k, selected
  on the last 5,000 TRAIN prediction losses (at most 0.025);
- projection 12 -> 6, then adaptation 50k (the dense initializer);
- tolerance tau = 1 x the calibration prediction MSE of seed 0's dense initializer, shared by
  all seeds;
- baseline and full arms of 350k updates each, run as 300k plus an exact 50k continuation;
- seeds 0-4.

Each run records a source identity (`diagnostics/shared/io.py`, `masked_mlp_learned/run.py`)
that hashes `src`, `diagnostics/shared` and the Experiment 1 training modules. These files are
unchanged since the reported runs, so the stored identities still verify.

```bash
# data, warm-up, calibration and the 300k sparse arms on five GPU hosts (SSH + tmux)
PREDICTOR_ACTIVATION=gelu bash diagnostics/masked_mlp_learned/experiment_batch.sh \
  BATCH DATA 0.2 1.0 HOST0 HOST1 HOST2 HOST3 HOST4
# exact 50k continuation of each sparse arm (arm in baseline, full; seed in 0..4)
python -m diagnostics.masked_mlp_learned.continuation --source-dir BATCH/ARM-sSEED/results \
  --output-dir BATCH350/ARM-sSEED/results --additional-steps 50000 --device cuda --threads 1
# test evaluation of the fixed endpoints: declare (no test tensors), evaluate, summarize
python -c 'from pathlib import Path; from diagnostics.masked_mlp_learned.frozen_test import declare
declare(Path("BATCH350"), Path("DATA"), Path("OUT"), sparse_steps=350000)'
python -m diagnostics.masked_mlp_learned.frozen_test evaluate --protocol OUT/protocol.json --seed SEED --device cuda
python -m diagnostics.masked_mlp_learned.frozen_test summarize --protocol OUT/protocol.json
python -m diagnostics.masked_mlp_oracle.paper_analysis --test-dir OUT --validation-dir BATCH350/paper --output-dir OUT/paper
```

The batch launcher addresses worker hosts as `$NODE_PREFIX<host>` over SSH. Each Python entry
point documents its options under `--help`.

## Vision experiment and downstream application

See the [package README](diagnostics/vision_dimer/README.md) for the settings and commands:

- the cohort is seed 20261115, with an RGB camera and the original physical constants;
- SIGReg 0.05 on the states; learning rate 1e-3, and 1e-4 for the CNN;
- the Experiment 1 schedule with code dimension 8 -> 4;
- sparse arms of 500k updates at tau = 2 x each seed's own calibration MSE;
- seeds 0-4.

## Notes

- Training never uses physical labels. Calibration fits every probe and alignment;
  validation was used during development and the test split is scored once, at the end.
- W&B logging is optional; the entity comes from the W&B login or `WANDB_ENTITY`.
- See the [third-party notices](src/scjepa/third_party/README.md) for dependency licenses and
  attribution, including SIGReg's non-commercial restrictions.
