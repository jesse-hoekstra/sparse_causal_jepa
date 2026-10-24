# Disposable oracle-state predictor control

This folder implements **point 1 only**: can the current dense SPARTAN learn the
known dynamics when supplied both true state coordinates and true parameters?
Everything is contained here. No production source, ordinary dataset interface,
experiment preset, or existing training command is changed. Delete this folder
to remove the diagnostic; its results live separately under `outputs/diagnostics/`.

## Server command

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

## What is trained

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

## What to inspect

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
