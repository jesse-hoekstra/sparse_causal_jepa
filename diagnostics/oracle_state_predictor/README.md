# Disposable oracle-state predictor controls

This folder implements two separate controls with true state coordinates and
true parameters supplied:

1. **Prediction fitting** (`run.py`): can the current dense SPARTAN learn the
   known dynamics? This command trains one dense predictor and has no graph
   recovery objective.
2. **Routing recovery** (`compare.py`): can baseline and full recover the known
   graph while retaining the prediction quality of a shared dense reference?
   This command compares dense, baseline and full with the representations fixed.

Everything is contained here. No production source, ordinary dataset interface,
experiment preset, or existing training command is changed. Delete this folder
to remove these controls; their results live separately under `outputs/diagnostics/`.

## Point 1: server command

From the `sparse_causal_jepa` repository root, after pulling these files:

```bash
source .venv/bin/activate
PYTHONPATH=src python -u -m diagnostics.oracle_state_predictor.run \
  --device cuda \
  --steps 100000 \
  --batch-size 16 \
  --seed 0 \
  --logit-weight 0.0001 \
  --output-dir outputs/diagnostics/oracle_state_predictor/seed-0-logit1e-4 \
  --wandb-mode online \
  --wandb-project sparse_causal_jepa \
  --wandb-entity jesse-hoekstra-university-of-oxford
```

This launches one dense predictor run, not the existing dense/baseline/full
sequence. W&B uses the separate group
`diagnostic-oracle-state-parameter-predictor`. Run names include the seed and
logit weight. Add `--resume` to the same command to continue `last.pt`.
The total `--steps` may be increased on resume. Changed scientific settings,
data, or source code are rejected. A fresh run refuses a nonempty output folder.

An optional **pure prediction-loss ablation** uses the same command with
`--logit-weight 0` and a new
`--output-dir outputs/diagnostics/oracle_state_predictor/seed-0-logit0`.
This distinguishes a logit-regularization tradeoff from the predictor's fitting
ability. It is a separate run, not a hidden change to the main comparison.
For another model initialization, change `--seed` and the output folder;
`--data-seed` defaults to zero so data stay matched.

## Point 1: what is trained

- The existing two-layer dense SPARTAN: embedding 64, token MLP width 64 and
  depth 3, history length 1, state/parameter dimension 6, action dimension 16.
- The same addressed simplified-Gaussian complementary DGP: K=10, T=12,
  R=10, 8192 training parameter groups, 256 validation and 512 test groups.
  Batch 16 means 160 episodes sharing 16 independent parameter vectors.
- Zero-based indices: input true `z[9]`, observed mixed actions `a[9]`, `a[10]`,
  raw true theta; targets are fixed true `z[10]`, `z[11]`.
- The first prediction is fed into the second step **without detaching** or
  substituting the true intermediate state. Theta is reused unchanged.
- AdamW, learning rate 0.001, weight decay zero, gradient clipping 1, float32,
  constant learning rate. Independent groups are shuffled without replacement;
  incomplete final batches are dropped. Inputs are precomputed once on CPU.

The exact optimized loss is

```text
mean((predicted_states - true_states)^2)
  + logit_weight * mean(SPARTAN logit penalty across the two calls)
```

The MSE averages both horizons, every group/view, and all six coordinates.
There is no learned observation encoder, parameter encoder, EMA target, SIGReg,
parameter consistency, GECO, or sparsity loss. The default attention-logit
penalty remains the same as the current experiment (weight 1e-4). True states
already have population N(0,I) marginals; no per-batch standardization is applied.
This is supervised diagnostic access to ground truth, not an unsupervised result.
Fresh predictor initialization is reproducible, but does not claim the same
initial tensor values as a full SCJEPA model, whose construction consumes RNG.

## Point 1: what to inspect

Validation is logged every 1000 updates; final test scores use `last.pt`.
`best.pt` is selected by validation prediction MSE only. Test results do not
select a model or affect training.

- `validation/normalized_prediction_mse` and `test/normalized_prediction_mse`:
  prediction MSE divided by target variance. Here `0.01` means 1% of target
  variance; these are comparable in scale to the earlier normalized error,
  though the earlier target was a learned representation.
- `horizon_1_mse`, `horizon_2_mse`, and their normalized versions distinguish
  one-step approximation from autoregressive error accumulation.
- `reference/analytic/*` checks the exact DGP on the supplied float32 inputs;
  residuals should be at rounding precision. `reference/persistence/*` repeats
  the input state as a simple forecasting baseline. References use validation
  data only and never train the predictor.
- `loss/prediction`, `loss/logit`, and `loss/total` expose any tradeoff with the
  attention-logit penalty. MCC is not reported: true coordinates are provided.

Variance is the mean population variance of the six target coordinates,
computed over the entire held-out split, not averaged from minibatch variances.
Combined variance pools both horizons; each horizon also has its own variance.

Outputs: `config.json`, `reference.json`, `history.jsonl`, `last.pt`, `best.pt`,
and `test_metrics.json`. Checkpoints include optimizer, sampler and Torch RNG
state, diagnostic/source/data identities and W&B identity. They cannot be
loaded through this diagnostic as ordinary SCJEPA checkpoints. Weights remain
local; W&B receives configuration and scalar metrics.

Accurate held-out prediction would shift attention to joint representation
learning and gate search. Persistently poor fitting would motivate predictor
or optimization diagnostics. No single numerical cutoff proves the theorem's
exact-equivalence assumptions, and this control does not yet test graph recovery.

## Point 2: dense, baseline and full graph comparison

From the repository root, launch a fresh comparison:

```bash
source .venv/bin/activate
PYTHONPATH=src python -u -m diagnostics.oracle_state_predictor.compare \
  --device cuda:0 \
  --steps 100000 \
  --batch-size 16 \
  --seed 0 \
  --logit-weight 0.0001 \
  --output-dir outputs/diagnostics/oracle_state_graphs/seed-0 \
  --wandb-mode online \
  --wandb-project sparse_causal_jepa \
  --wandb-entity jesse-hoekstra-university-of-oxford
```

By default this trains dense for 100,000 updates, then baseline and full for
100,000 updates each. `--steps` is the budget **per condition**, not the combined
budget. It creates three W&B runs in
`diagnostic-oracle-state-parameter-graphs`.

To reuse a completed point-1 dense run, use a fresh comparison output directory
and add this argument to the command above:

```bash
--dense-reference-dir outputs/diagnostics/oracle_state_predictor/seed-0-logit1e-4
```

Reuse still creates three comparison runs: dense is an evaluation-only run,
and baseline/full are new training runs. The original point-1 directory and W&B
run remain unchanged. Its `best.pt` supplies both sparse warm starts and the
calibration weights; its `last.pt` supplies the dense comparison's final scores.
For a reference selected at update 95,000 and completed at 100,000, those are
therefore different evaluated checkpoints, and both steps are recorded.

A reference must be a completed point-1 diagnostic with matching scientific
configuration, declared protocol/objective, data identities and architecture.
Both checkpoint identities and model tensor keys, shapes, dtypes and finite
values are checked; exact checkpoint hashes are recorded. Compatible historical
source fingerprints may differ from current code after these checks. The
original fingerprint, current fingerprint and that difference are retained in
the comparison metadata. Reuse does not relabel historical training as a run
under the current source version.

Add `--resume` to the comparison command to restore unfinished conditions and
skip completed conditions. Unlike the point-1 runner, a comparison requires
the same total `--steps`, scientific settings, source and dense reference on
resume. Use a new output directory for a changed comparison.

### Point 2: shared calibration and training

Inputs, two-step attached rollout, fixed true targets, predictor architecture,
data splits and optimizer settings match point 1. There are no learned state or
parameter encoders, EMA targets, SIGReg or parameter-consistency losses.

The dense `best.pt` is selected by validation **prediction MSE**. It is evaluated
on 512 independent parameter groups from the reserved `evaluation_calibration`
split, separate from training, validation and test. The common bound is

```text
base_loss = mean two-step prediction MSE + logit_weight * mean logit penalty
tau = calibration(base_loss) * tau_multiplier
```

`--tau-multiplier` defaults to `1.0`. Both sparse conditions receive the exact
same selected dense weights and tau, with fresh AdamW optimizers and GECO
controllers, paired group sampling and paired gate random streams. Resume
restores each condition's own optimizer, sampler, controller and RNG state.

| Condition | Gate behavior | Sparsity term |
|---|---|---|
| Dense | All gates open | None |
| Baseline | State/action gates learned; parameter-source gates to decoded state rows forced open in every layer | State + action path means |
| Full | State/action/parameter gates learned | State + action + parameter path means |

The sparse objective is `base_loss + GECO_sparsity_weight * sparsity_term`.
Each enabled block has coefficient one. The blocks are the existing
residual-inclusive routing path means, rather than counts of distinct edges.
GECO uses the production settings: multiplier initialization 1000 (initial
sparsity weight 0.001), step size 0.1, momentum 0.99 and log-multiplier bounds
[-12, 12]. Its pressure changes continuously in response to training base loss.

The initial validation metrics expose the effect of switching from dense to
deterministic sparse gates. `warm_start/sampled_base_loss` also reports one
validation minibatch with sampled training gates, without advancing the
training RNG. A bound calibrated on a well-fitted dense model can be difficult
for the sparse model to meet; inspect feasibility and prediction error alongside
graph recovery. A separate multiplier ablation changes the allowed prediction
tradeoff and should use a new comparison directory.

### Point 2: graph scores and outputs

The known one-step graph has 18 state edges, 96 action edges and 12 parameter
edges. Identity alignment is used: the supplied states and parameters already
have their true coordinate labels. No learned coordinate matching is applied,
and MCC would measure supplied inputs rather than learned recovery.

| Condition | Ideal state/action/parameter support counts | Corresponding state/action/parameter SHD |
|---|---|---|
| Dense | 36 / 96 / 36 | 18 / 0 / 24 |
| Baseline | 18 / 96 / 36 | 0 / 0 / 24 |
| Full | 18 / 96 / 12 | 0 / 0 / 0 |

These are diagnostic targets, not guaranteed training outcomes. Baseline's
parameter gates are deliberately open, so its parameter SHD cannot reach zero.
Support is the union of observed nonzero routing paths over the held-out inputs
and both predictor calls, with a frequency threshold of zero. Each call is
scored as a direct one-step graph; paths are not composed across the two-step
forecast. Saved matrices use output/receiver rows and input/source columns.
Input-dependent routing support is a proxy for dependency structure, not proof
of a global functional graph or the paper's identifiability assumptions.

W&B logs the following under `validation/` and `test/`:

- Prediction MSE, normalized prediction MSE and separate horizon errors.
- `base_loss`, block path means, and the condition's `sparsity` term.
- `state_edges`, `action_edges`, `parameter_edges`, their `*_shd`,
  `*_true_positives`, `*_false_positives` and `*_false_negatives`.
- For sparse conditions, `constraint = base_loss - tau` and `feasible`.

Training also logs the applied and updated GECO sparsity weights and its moving
average constraint. Validation selects `best.pt` by prediction MSE; final test
scores use `last.pt`. Test data do not select checkpoints or calibrate tau.

The comparison root contains `comparison.json`, `calibration.json` and
`comparison_summary.json`. Each condition contains `config.json`,
`history.jsonl`, `result.json` and `test_graphs.json`; the latter records routing
frequencies, binary support and the true graph. Trained conditions also contain
`last.pt` and `best.pt`. An evaluation-only dense condition refers to its
original checkpoint files instead of creating new weights.

This comparison isolates routing recovery after supplying the representations.
Success would support testing the joint representation-and-routing problem
next. Failure with accurate prediction would instead locate a difficulty in
the sparsity objective or gate search. It does not itself establish learned
state or parameter identifiability.

## Local verification

```bash
PYTHONPATH=src python -m pytest diagnostics/oracle_state_predictor
ruff check diagnostics/oracle_state_predictor
ruff format --check diagnostics/oracle_state_predictor
pyright diagnostics/oracle_state_predictor
```

Tiny offline CPU smoke (choose a fresh output folder each time):

```bash
PYTHONPATH=src python -m diagnostics.oracle_state_predictor.run \
  --steps 2 --train-groups 4 --validation-groups 3 --test-groups 3 \
  --views 2 --batch-size 2 --threads 1 --embed-dim 8 --mlp-hidden 8 \
  --evaluate-every 1 --log-every 1 --checkpoint-every 1 \
  --device cpu --wandb-mode disabled \
  --output-dir /tmp/oracle-state-predictor-smoke
```

Tiny offline comparison smoke, including fresh dense calibration and both sparse
conditions:

```bash
PYTHONPATH=src python -m diagnostics.oracle_state_predictor.compare \
  --steps 2 --train-groups 4 --validation-groups 3 --test-groups 3 \
  --calibration-groups 4 --views 2 --batch-size 2 --threads 1 \
  --embed-dim 8 --mlp-hidden 8 --evaluate-every 1 --log-every 1 \
  --checkpoint-every 1 --device cpu --wandb-mode disabled \
  --output-dir /tmp/oracle-state-graphs-smoke
```
