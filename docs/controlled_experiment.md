# Complementary experiment: dense calibration and GECO sparsification

The default command runs **three models, complementary data, seed 0**, sequentially:

| Condition | Attention | Sparsity objective |
|---|---|---|
| `dense` | All connections open | None; calibrates tau |
| `baseline` | Parameter-to-output gates forced open; other gates learned | State and action paths |
| `full` | Learned local SPARTAN gates | State, action and parameter paths |

All three use identical paired training data, data order, and observation map.
Each independent parameter draw generates two episodes, A and B, with independent
initial states and actions. Each episode has three frames: two context frames and
one prediction target. A's parameter code predicts B's next state, and B's code
predicts A's next state, always using the target episode's current state and action.
The pairing is known during training; true parameter values and graphs are not
model inputs. The [vector DGP](data_generation.md#controlled-vector-process)
modifies the paper's coefficients to strengthen state, parameter and action effects while
retaining its graphs and DGP assumptions. Fresh dense training and tau calibration
are required for this version; earlier single-episode checkpoints and bounds are
incompatible.
Baseline and full start from the exact dense best checkpoint used to calibrate
tau, including its online and EMA state encoders, parameter encoder and predictor.
Each sparse run starts with a fresh optimizer, GECO controller and step counters;
its condition-specific gate modes remain as specified above. Baseline forces
parameter-source columns to decoded latest-state receiver rows open in every
SPARTAN layer, during training and evaluation, and uses zero parameter sparsity
weight. Full learns those gates
and adds the parameter path penalty. Both retain learned attention weights, all
other learned gates, the parameter encoder and parameter inputs, and state/action
regularization. The gate override adds no learned tensors or random draws;
both sparse runs retain identical initial tensors.

Dense training uses cosine learning-rate decay from `train.lr` to
`train.lr * train.lr_final_ratio` (defaults: 0.001 to 0.0001). The horizon is the
configured number of optimizer updates, `train.steps / train.accumulation_steps`
(10,000 by default). Only successful optimizer updates advance the decay; resume
restores its position and keeps the original horizon. The runner derives
`train.lr_decay_steps` per condition; do not override it manually. Baseline and full
retain their constant learning rate and fresh optimizers.

Set `dense_reference_dir` to an existing experiment root to reuse its selected
dense checkpoint and original tau in a new output directory. This runs only
baseline and full; no new dense run is trained. The reference files remain
untouched. Compatible dense checkpoints from older source versions are accepted
after checking the DGP version and grouping, architecture, data, base objective,
calibration batching, observation map and checkpoint hash.

Each fresh sparse run prints one startup line comparing dense, deterministic-gate
and sampled-gate base losses on the calibration split, with differences and tau.
The check matches SIGReg projection draws, preserves weights, GECO state and
training RNG, adds no W&B metrics, and is skipped on checkpoint resume or evaluation-only runs.

## Model and objective

An independent framewise MLP encodes observations `(B,2,T,10)` into states,
where `B` counts independent parameter groups and the second axis indexes A/B.
The shared context encoder uses each episode's first two states and their action,
producing attached parameter codes `(B,2,10)`. These codes are swapped across the
pair before the one-lag SPARTAN predicts the complete frozen EMA encoder's
next-frame latent. For `T=3`, predictions and targets are `(B,2,1,10)`; actions are
`(B,2,2,10)`. Prediction MSE averages both directions. No additional consistency
loss is introduced. The state, parameter and action dimensions remain independent
in the reusable model; ordinary single-episode inputs remain supported.

SPARTAN now uses content-dependent queries and keys, hard Bernoulli gates from
`sigmoid(q @ k)`, and masked attention weights from `softmax(q @ k / sqrt(d))`.
Residual-inclusive path products track local routing through all layers. The
symmetric exponential logit penalty remains active to discourage saturation.
Baseline parameter routing is dense by construction. Forced-open gates still use
learned attention weights, so they do not guarantee a nonzero functional effect
from every parameter. The runner derives `spartan_parameter_gates_open` per
condition; manual gate-mode overrides in the experiment configuration are rejected.

The non-sparsity objective is

`base = latent_MSE + ws*SIGReg(states) + wp*SIGReg(parameters) + wl*logit_penalty`.

There is no pixel decoder, reconstruction loss, variational KL term or duplicate
parameter encoder. SIGReg is calculated separately for each episode view across
`B` independent parameter groups, at each frame for states and once per view for
parameter codes, then averaged over the two views. The two codes of one parameter
are never treated as independent samples in a `2B` SIGReg batch.

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
of independent groups in each statistical batch. For distributed runs, explicitly
set the evaluation batch to the per-rank training batch. The existing configuration
counts (`train_episodes` and held-out `*_episodes`) now denote groups: defaults are
8,192 training, 512 calibration, 256 validation and 512 test groups. Each group
contains two trajectories; `batch_size=64` therefore processes 128 trajectories
while each SIGReg statistic still uses 64 independent parameter draws.

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

The default sequence has three named W&B runs: `complementary-dense-seed0`,
`complementary-baseline-seed0`, and `complementary-full-seed0`.
With `dense_reference_dir`, only the latter two are created in the new group.
MCC and SHD are logged at each validation interval and on the final held-out
test set. Calibration matching is refitted per evaluated checkpoint; recovery
scores do not enter checkpoint selection. The dashboard retains:

- Prediction MSE, base loss and weighted sparsity loss.
- `train/lr`: the learning rate for the next optimizer update.
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
For the complementary baseline, parameter SHD is therefore 80 by construction:
all 100 parameter routing edges are available, against 20 true edges. Changes in
that baseline metric cannot be used to assess learning of parameter support.

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
mkdir -p logs
set -o pipefail
python -u scripts/run_controlled_experiment.py \
  --config configs/experiments/controlled_vector.yaml \
  --condition dense \
  train.device=cuda \
  wandb.mode=online \
  wandb.group=exp5_paired_context \
  output_dir=outputs/exp5_paired_context \
  2>&1 | tee -a logs/exp5_paired_context.log
```

Detach with **Ctrl-b, then d**; return with `tmux attach -t scjepa-geco`.
This command trains and evaluates only a fresh dense model on paired episodes,
with cosine decay, then calibrates tau from its selected checkpoint. Inspect prediction and base loss
on training and validation before proceeding to sparsification. To run baseline
and full afterward, repeat the same command with `--resume` and remove
`--condition dense`; the completed dense stage is skipped and both sparse models
start from its selected checkpoint, including EMA weights. The numeric optimizer,
logit coefficient and GECO controller defaults
are explicit starting choices, not
validated scientific hyperparameters. Use `--dry-run` to print the selected stages
without starting W&B.

Resume after interruption by repeating the same command, including
`dense_reference_dir` when set, with `--resume`; sparse runs restore their own
checkpoint, optimizer and controller state.
Completed stages are skipped. To select a stage, add `--condition dense`,
`--condition baseline`, or `--condition full`; sparse-only execution requires an
already completed, compatible dense calibration. `--condition dense` is incompatible
with `dense_reference_dir`. `--evaluate-only` recomputes the selected checkpoint's
metrics without training. `--aggregate-only` refreshes local
`results.csv` and `summary.json`.

Use a **new output directory** for this protocol. Earlier GECO runs that started
sparse models from random initialization or learned the baseline parameter gates,
as well as identity-attention/fixed-weight runs, have different source and model
provenance and cannot be resumed as this comparison.
Only a dense checkpoint and calibration record from the same DGP version and
episode grouping can be reused through `dense_reference_dir`, subject to the
remaining compatibility checks. Earlier single-episode, weak-coupling and
long-trajectory experiments need fresh dense training and tau for this comparison.
Existing outputs, checkpoints and logs are preserved.

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
