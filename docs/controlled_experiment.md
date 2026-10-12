# Complementary experiment: dense calibration and GECO sparsification

The default command runs **three models, complementary data, seed 0**, sequentially:

| Condition | Attention | Sparsity objective |
|---|---|---|
| `dense` | All connections open | None; calibrates tau |
| `baseline` | Learned local SPARTAN gates | State and action paths |
| `full` | Learned local SPARTAN gates | State, action and parameter paths |

All three start from identical weights and use identical training episodes, data
order, and observation map. Sparse runs start from that paired initialization,
not from the trained dense checkpoint. Baseline/full differ only in the parameter
sparsity coefficient; neither removes the parameter encoder or parameter inputs.

## Model and objective

An independent framewise MLP encodes observations `(B,T,10)` into states.
The context encoder uses the first two states and their transition action to
produce one attached `(B,10)` parameter tensor, reused for all predictions.
A one-lag SPARTAN predicts the complete frozen EMA encoder's next-frame latent.
For `T=16`, predictions and targets are `(B,14,10)`; actions are `(B,15,10)`.
The state, parameter and action dimensions remain independent in the reusable model.

SPARTAN now uses content-dependent queries and keys, hard Bernoulli gates from
`sigmoid(q @ k)`, and masked attention weights from `softmax(q @ k / sqrt(d))`.
Residual-inclusive path products track local routing through all layers. The
symmetric exponential logit penalty remains active to discourage saturation.

The non-sparsity objective is

`base = latent_MSE + ws*SIGReg(states) + wp*SIGReg(parameters) + wl*logit_penalty`.

There is no pixel decoder, reconstruction loss, variational KL term or duplicate
parameter encoder. SIGReg sample axes are unchanged: episodes at each frame for
states, and one code per distinct episode for parameters.

Following the newer SPARTAN paper's loss-constrained optimization, sparse training
minimizes weighted path counts subject to `E[base] <= tau`. The implemented primal
loss is `base + sparsity / lambda`, with a positive adaptive multiplier. After
each successful optimizer update, the controller updates `log(lambda)` using a
moving average of `base - tau`. Above the bound, lambda grows and sparsity pressure
falls; below it, sparsity pressure grows. GECO and encoder EMA update once per
accepted optimizer step, averaging accumulation groups and distributed ranks.
Skipped updates advance neither controller. Full controller state is checkpointed.
The runner requires `evaluation.batch_size = train.batch_size / world_size` and
held-out split sizes divisible by that batch size: SIGReg depends on the number
of episodes in each statistical batch. For distributed runs, explicitly set the
evaluation batch to the per-rank training batch.

The selected dense checkpoint is evaluated on the independent calibration split.
Its mean **base loss**, not MSE alone, becomes tau. `tau_multiplier=1.0` uses that
value exactly; any relaxation must be an explicit configuration change. The saved
`dense/tau.json` binds tau to the dense checkpoint hash and experiment identity.
Both sparse models use exactly the same record. This is a prediction/regularization
loss constraint, not an edge-count budget.

Validation selects the dense checkpoint with the lowest base loss. For sparse
models it first prefers checkpoints with `validation/base_loss <= tau`, then the
lowest weighted path count among those. If none is feasible, it selects the lowest
base loss and reports infeasibility. The internal selection score encodes this
ordering monotonically; it is excluded from the dashboard. Test data never select
checkpoints or calibrate tau.

## What to observe

W&B has three named runs: `complementary-dense-seed0`,
`complementary-baseline-seed0`, and `complementary-full-seed0`.
MCC and SHD are logged at each validation interval and on the final held-out
test set. Calibration matching is refitted per evaluated checkpoint; recovery
scores do not enter checkpoint selection. The dashboard retains:

- Prediction MSE, base loss and weighted sparsity loss.
- Tau, constraint violation and effective sparsity weight for GECO.
- Validation/test feasibility and the count of skipped updates.
- `recovery/state_mcc` and `recovery/parameter_mcc`: held-out mean absolute Pearson
  correlation after one-to-one Hungarian coordinate matching fitted on calibration.
- `graph/state_shd`, `graph/action_shd`, `graph/parameter_shd`: false positives plus
  false negatives after coordinate alignment. A separate calibration match for
  EMA targets permutes output rows; online-state matching permutes state-input
  columns and parameter matching permutes parameter-input columns. Observed actions
  keep their known order. Lower SHD and higher MCC are better.

SHD compares the union of deterministic local routing paths over every held-out
transition with the DGP's global graph. Residual paths are included. The optional
`evaluation.graph_support_threshold` instead requires support frequency strictly
above a predeclared value; its default is zero (union). It is never fitted to graph
labels. These are attention-routing graphs, not guaranteed Jacobian or causal
support: content-dependent attention controls also affect outputs. Poor MCC or
mixing within identifiable groups also limits SHD's interpretation. In particular,
complementary parameters may remain mixed within pairs under the paper's equivalence
class; perfect coordinatewise MCC is a stricter criterion than pair recovery.

Matching tensors, supports, references and concise metrics are saved locally.
Nonlinear probes, code swaps, rollout dashboards and automatic evaluation artifact
uploads are removed from the active protocol. Reusable evaluation utilities remain.
SIGReg, EMA, low loss, MCC and SHD do not prove injectivity or identifiability.

## Run on omi-rapid-20

On the server, update the checkout to include these changes and activate its Python
3.12 environment. Install tmux if needed (`sudo apt-get install tmux`). Authenticate
W&B once with `wandb login`; use your account's entity or team slug.

```bash
tmux new -s scjepa-geco
cd ~/sparse_causal_jepa  # use the actual checkout location
source .venv/bin/activate
python -m pip install -e '.[dev,logging]'
export WANDB_PROJECT=sparse_causal_jepa
export WANDB_ENTITY=jesse-hoekstra-university-of-oxford
python -u scripts/run_controlled_experiment.py \
  train.device=cuda \
  output_dir=outputs/complementary_geco_seed0 \
  wandb.group=complementary-geco-seed0
```

Detach with **Ctrl-b, then d**; return with `tmux attach -t scjepa-geco`.
The one command completes dense calibration before starting baseline and full.
It launches no other seed or graph. The numeric optimizer, logit coefficient and
GECO controller defaults are explicit starting choices, not validated scientific
hyperparameters. Use `--dry-run` to print the three-stage plan without starting W&B.

Resume after interruption by repeating the same command with `--resume`.
Completed stages are skipped. To select a stage, add `--condition dense`,
`--condition baseline`, or `--condition full`; sparse-only execution requires an
already completed, matching dense calibration. `--evaluate-only` recomputes the
selected checkpoint's metrics without training. `--aggregate-only` refreshes local
`results.csv` and `summary.json`.

Use a **new output directory** for this protocol. Old identity-attention/fixed-weight
runs have different architecture and objective provenance and cannot be resumed
as GECO runs. Existing outputs, checkpoints and logs are preserved.

A tiny CPU check of all three stages (not a scientific run):

```bash
OMP_NUM_THREADS=1 python scripts/run_controlled_experiment.py \
  --config configs/experiments/controlled_vector_smoke.yaml \
  output_dir=/tmp/scjepa-geco-smoke wandb.mode=disabled
```

Definitions and dataset/observation-map identities are fixed in the experiment
manifest. Resume requires matching source, Python/PyTorch, world size and scientific
configuration. W&B destinations and runtime controls do not change scientific
identity. Graph variants and seed lists remain configurable for later explicit runs.

The optimization follows Section 3.3 of [Disentangling Dynamical Systems](https://arxiv.org/abs/2603.14483v2),
with latent JEPA prediction and SIGReg replacing its reconstruction/KL terms.
MCC follows the Pearson/Hungarian convention in [Lachapelle et al.](https://github.com/slachapelle/disentanglement_via_mechanism_sparsity);
our matching is fitted on calibration rather than test observations.
