# Controlled-vector experiment

`scripts/run_controlled_experiment.py` implements the first comparison from
Sections 10.1–10.3: three parameter graphs, paired baseline/full models, and five
initialization seeds, giving **30 runs** by default. Runs execute sequentially;
selectors allow independent server jobs. The spring data generator remains
available, but this runner evaluates vectors.

## Scientific comparison

Each episode has ten states, ten constant parameters, and ten observed actions.
The nonlinear observation map and exogenous draws are shared across graph
variants. The state graph has two five-coordinate groups. The parameter graphs
are dense, complementary (five cross-block pairs), and diagonal; their true edge
counts are 100, 20, and 10. See [data generation](data_generation.md) for equations.
Ground-truth coordinates and graph locations never enter model optimization.

An ordinary framewise MLP maps observations `(B,T,10)` to states `(B,T,10)`.
The parameter encoder reads the first two encoded frames and their observed
transition action, producing one attached `(B,10)` parameter code. The one-lag
predictor uses that code throughout prediction and rollout. With the default
16-frame episode, teacher-forced predictions and stopped-gradient EMA targets
both have shape `(B,14,10)`. Actions have shape `(B,15,10)`.

Both conditions use latent MSE, state/action sparsity, and **SIGReg on states and
episode-level parameters**. The baseline sets only the parameter-graph sparsity
weight to zero. It retains the same parameter encoder, conditioning, gates,
optimizer, EMA, initialization, data order, and SIGReg settings. The full model
uses the configured positive parameter sparsity weight. Sparsity penalizes
residual-inclusive path counts; there is no exact graph-budget enforcement.
There is no pixel decoder or reconstruction loss.

The dense setting is intended to retain state groups; complementary parameter
dependencies should refine state coordinates while retaining parameter pairs;
diagonal dependencies should refine both. These are hypotheses to test across
seeds, not success criteria enforced by the code.

## Evaluation and model selection

- Training episodes update the model without physical labels. Validation selects
  `best.pt` using **physical prediction NRMSE**, lower is better. A separate
  calibration split fits an EMA-latent-to-physical-state probe for each candidate.
  Calibration and validation preserve training RNG and module modes.
- Final evaluation fits fresh probes and coordinate matching on calibration
  episodes only, then freezes them for test. Test data never select checkpoints.
- Prediction metrics include physical MSE/NRMSE, one-step and each rollout
  horizon, latent error, encoded-target decoding error, representation standard
  deviation/effective rank, and online/EMA drift.
  NRMSE uses per-coordinate calibration state
  variances. Rollout starts from online history, shifts it correctly, and appends
  predicted target-coordinate states; EMA drift is therefore relevant.
- State/parameter recovery includes permutation-matched absolute Spearman
  correlations, full-vector nonlinear probes, and group probes. State groups
  have five coordinates; parameter groups follow each graph variant. Group
  feature subsets use calibration coordinate matching, so these scores are a
  heuristic and are not invariant to arbitrary nonlinear mixing within a group.
  Full-vector recovery provides a separate information check.
- Independent test trajectories with the same true parameter vector measure code
  consistency. Swapping their codes tests prediction preservation; cyclic
  different-parameter swaps provide a sensitivity comparison. Parameter statistics
  use one code per episode. The within/between ratio divides half the paired
  squared difference by pair-mean variance; that denominator includes finite-pair
  sampling noise.
- Graph diagnostics report structural support counts/densities, including residual
  paths. They are conditional on the supplied code, not physical causal effects
  or graph-recovery accuracy in an unverified latent coordinate alignment.

The MLP can approximate an inverse observation map over the sampled region, but
its architecture does not ensure injectivity. This explicitly adapts the paper's
invertible-encoder proposal: physical predictions use a fitted nonlinear probe,
not an exact encoder inverse. Its test decoding error is reported alongside
prediction error because probe error can affect that comparison. SIGReg, EMA,
low loss, and successful probes do not prove population identifiability.

## Run on a server with tmux and W&B

Use Python 3.12 and a PyTorch 2.9 installation appropriate for the server's CUDA
runtime. From the server checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,logging]'
wandb login
export WANDB_ENTITY=YOUR_WANDB_USER_OR_TEAM
export WANDB_PROJECT=sparse_causal_jepa
```

W&B uses this server's authenticated account. No API key belongs in configuration
or Git. Verify the complete path cheaply with the two-step CPU pair first:

```bash
OMP_NUM_THREADS=1 python scripts/run_controlled_experiment.py \
  --config configs/experiments/controlled_vector_smoke.yaml \
  wandb.mode=online output_dir=outputs/server_preflight
```

The smoke configuration defaults to offline W&B if `wandb.mode` is not supplied.
Its scores have no scientific interpretation. This check exercises W&B, training,
validation, checkpoint selection, probes, artifact logging, and paired aggregation.
Choose a fresh output directory for another independent check, or use `--resume`
to verify completed-run detection.

Inspect the full plan without creating outputs or contacting W&B:

```bash
python scripts/run_controlled_experiment.py --dry-run train.device=cuda
```

Start tmux, activate the environment inside it, and run the comparison:

```bash
tmux new -s scjepa-vector
source .venv/bin/activate
export WANDB_ENTITY=YOUR_WANDB_USER_OR_TEAM
export WANDB_PROJECT=sparse_causal_jepa
export OMP_NUM_THREADS=1
mkdir -p logs
set -o pipefail
python -u scripts/run_controlled_experiment.py \
  train.device=cuda wandb.mode=online \
  output_dir=outputs/controlled_vector 2>&1 | tee -a logs/controlled_vector.log
```

Detach with `Ctrl-b`, then `d`; reconnect with `tmux attach -t scjepa-vector`.
To run one paired seed first, add `--graph complementary --seed 0`. Later use
`--resume` without selectors to complete the same 30-run definition. Use selectors
instead of editing `graphs` or `seeds` when splitting one experiment across jobs.
Each graph/seed/condition owns a separate lock and W&B run; duplicate concurrent
jobs are rejected. Different selectors can share the experiment directory.

Resume after interruption with the same command/configuration and `--resume`:

```bash
python -u scripts/run_controlled_experiment.py --resume \
  train.device=cuda wandb.mode=online output_dir=outputs/controlled_vector
```

SIGINT/SIGTERM request a checkpoint at the next complete accumulation boundary.
Interrupted training exits with code 130. A signal during evaluation can allow
that evaluation to finish before control returns. A forced kill recovers from
the last periodic checkpoint. The OS releases ownership locks on process exit;
a leftover lock file is harmless.

For one model distributed across two GPUs, replace the Python invocation with:

```bash
torchrun --standalone --nproc_per_node=2 scripts/run_controlled_experiment.py \
  --graph complementary --seed 0 train.device=cuda wandb.mode=online \
  output_dir=outputs/vector_two_gpu
```

`train.batch_size` is global. It must be divisible by world size with at least two
episodes per rank for SIGReg. Statistics are rank-local, so changing world size
changes this objective. `execution.json` enforces the same world size for all compared runs.
Only rank zero validates, evaluates, and logs. Other ranks synchronize on results
and failures. CPU Gloo integration is tested; validate CUDA/NCCL on the server
before allocating a full run.

## Results, tracking, and reproducibility

Each run sends train/validation/test/recovery/consistency/graph scalars to W&B and
terminal JSON. W&B names include graph, condition, and seed; groups come from
`wandb.group`. `evaluation/metrics.json` is also an explicit W&B artifact (disable
with `upload_evaluation_artifacts=false`). Checkpoints and datasets are not
uploaded automatically. Tracking failures fail the run rather than silently
dropping logs. `wandb.mode=offline` or `disabled` supports local execution.

The output root contains the actual `observation_map.pt`, an experiment manifest,
`summary.json`, and `results.csv`. Per-run directories contain:

- `run_config.json`, resolved YAML, `run_metadata.json`, and `status.json`;
- `last.pt` for exact training continuation and validation-selected `best.pt`;
- `evaluation/metrics.json`, protocol/reference tensors, frozen probe weights,
  coordinate matches, parameter-code pairs, and selected-checkpoint provenance.

Summaries include per-condition mean, sample standard deviation, standard error,
and **paired full-minus-baseline differences**. They use completed runs only and
explicitly report missing runs/pairs; one seed has no estimated variation. The
aggregator verifies matched initialization, dataset, and experiment identities.

Resume checks scientific configuration, actual source bytes (including uncommitted
code), Python/PyTorch versions, persisted observation matrix, dataset, model,
batch size, and world size. Choose a new output directory after changing scientific
settings or code. Runtime device/logging destinations may change; identical random
streams do not promise bitwise equivalence across GPU hardware. Online W&B resume
reconnects to the saved run ID. Offline segments require explicit sync.

Re-evaluate selected checkpoints or refresh a partial summary:

```bash
python scripts/run_controlled_experiment.py --evaluate-only \
  --graph complementary --seed 0 train.device=cuda
python scripts/run_controlled_experiment.py --aggregate-only
```

Both require the original experiment definition. Evaluation checks the best
checkpoint's run/data/model/configuration and validation selection, and saves its
SHA256. After a crash between best/last writes, resume training first to restore
a consistent selection record. Load only trusted checkpoints.

## Explicit experiment choices

The default YAML specifies 8,192 training episodes, 16 frames, 10,000 microbatches,
batch 64, validation every 1,000 microbatches, and five seeds. Widths, learning rate,
EMA decay, SIGReg weights 0.1, sparsity weights 0.001, probe capacity/training steps,
evaluation sample counts, and eight-step rollout are **untuned implementation
choices**, not paper-prescribed optima. Select these using validation and record a
new experiment definition before inspecting final test results. Infrastructure
readiness does not establish convergence or scientific success.
