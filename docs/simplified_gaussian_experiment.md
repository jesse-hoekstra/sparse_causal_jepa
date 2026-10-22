# Experiment 1: simplified Gaussian DGP, m=3

This preset implements the approved six-state experiment: two rows of three
states, six fixed system parameters, and sixteen observed action coordinates.
Select `data_family: simplified_gaussian`. The previous DGPs retain their
original definitions and checkpoints.

## DGP

Each parameter group draws six independent `theta_i ~ Uniform(-0.75,0.75)` and
contains two episodes sharing only theta. Each episode has independent
`z0 ~ N(0,I6)` and independently sampled controls at each transition:

```text
gamma1, gamma2 ~ N(0,I2)
c ~ N(0,I6)
u_i ~ Uniform(-1,1)
xi = concatenate(gamma1, gamma2, c, u)
V = I16 - (2/16) J16
observed action a = V xi; recover xi = V a inside the generator

H(v) = I3 - 2 v v^T / (v^T v)
R(gamma) = H((1,gamma)) H(diag(1,2,3) (1,gamma))
b = diag(R(gamma1), R(gamma2)) z
phi_i = theta_i + u_i theta_partner(i)
z_next_i = b_i cos(phi_i) + c_i sin(phi_i)
```

Partners are `1 <-> 4`, `2 <-> 5`, `3 <-> 6`. The model receives observed `a`,
not the separate controls or V. Action vectors are independent across time and
independent of theta and the preceding state; coordinates within a need not be
independent. No action is chosen from the state or used to cancel dynamics.

The true graphs have **18 state, 96 action, and 12 parameter edges**:
`Gz=diag(J3,J3)`, `Ga=J(6,16)`, `Gtheta=[[I3,I3],[I3,I3]]`.
State/action structure leaves ambiguity within rows. Parameter consistency
restricts each state to its column pair, whose intersection with its row is a
singleton. This meets Assumption 14(i); parameter pairs remain ambiguous under
14(ii). The action mixing also satisfies the fixed-output variability condition;
dense support alone is not the justification. These are properties of the DGP,
not guarantees that optimization meets the learned-model hypotheses.

Conditional on theta, `z_t ~ N(0,I6)` at every frame. Observations use
`x=psi(Wz)`, `psi(s)=s+0.1s^3`, with a persisted fixed orthogonal 6x6 W
(observation seed 1729). Simulation uses float64 before conversion to the
training dtype. Labels and reference matrices are excluded from training items.

## Comparison

All conditions use paired twelve-frame episodes: K=10 context frames, L=1, and
two cross-episode autoregressive predictions with the donor code held fixed.
Frames 0--9 and their nine transitions infer the parameter code; frames 10--11
are the forecast targets. The theoretical minimum context remains three frames.

| Condition | State paths | Action paths | Parameter paths |
|---|---|---|---|
| Dense | Open; no penalty | Open; no penalty | Open; no penalty |
| Baseline | Learned; penalty 1 | Learned; penalty 1 | Open; no penalty |
| Full | Learned; penalty 1 | Learned; penalty 1 | Learned; penalty 1 |

Both sparse conditions learn action gates and apply the same action sparsity
penalty. Only dense forces all gates open. Baseline separately keeps parameter
gates open; full learns them. The true dense action graph is an evaluation
reference, not an imposed mask. The penalty counts routing paths, so redundant
paths can be removed while retaining every action dependency. Whether training
retains those dependencies is measured rather than guaranteed.
Each condition uses 100,000 updates, batch size 64 parameter groups,
constant learning rate 0.001, EMA 0.996, and SIGReg weights 0.1 for both codes.
Both sparse stages start from the same selected dense checkpoint and calibrated
GECO bound. Final test results use **last.pt**, not the selected best checkpoint.

Evaluation includes calibrated MCC, matched row/pair linear recovery, graph
SHD with false-positive/negative counts, within-system parameter-code variance,
and a cross-system code-shuffle prediction diagnostic. Linear recovery is not
a test of all nonlinear reparameterizations allowed by the theorem. In the sparse
conditions, action SHD counts missing true action edges because the reference
graph is dense; monitor it and action false negatives to check retention.

## Run on the server

Copy the updated repository to the server. From its root, use the existing
Python 3.12 environment with CUDA-compatible PyTorch:

```bash
source .venv/bin/activate
export WANDB_PROJECT=sparse_causal_jepa
export WANDB_ENTITY=jesse-hoekstra-university-of-oxford

python scripts/run_controlled_experiment.py \
  --config configs/experiments/simplified_gaussian.yaml \
  --dry-run train.device=cuda

mkdir -p logs
set -o pipefail
python -u scripts/run_controlled_experiment.py \
  --config configs/experiments/simplified_gaussian.yaml \
  train.device=cuda wandb.mode=online \
  wandb.group=simplified_gaussian_m3_k10_action_100k \
  output_dir=outputs/simplified_gaussian_m3_k10_action_100k \
  2>&1 | tee -a logs/simplified_gaussian_m3_k10_action_100k.log
```

The command runs dense, baseline, and full sequentially. Run it in a persistent
terminal such as tmux. After an interruption, repeat the same command with
`--resume` before the `train.device` override. Completed stages are skipped.
Use a fresh output directory and fresh dense calibration: previous DGPs and the
earlier K=3/open-action experiment have incompatible data/configurations and tau.
Keep source/configuration unchanged when
resuming. Existing W&B credentials are used.

The small CPU smoke is an infrastructure check, not a scientific run:

```bash
OMP_NUM_THREADS=1 python scripts/run_controlled_experiment.py \
  --config configs/experiments/simplified_gaussian_smoke.yaml \
  wandb.mode=disabled output_dir=/tmp/scjepa-simplified-gaussian-k10-action-smoke
```

## Longer-context experiment: K=50

The question is whether more observed transitions let the parameter encoder
produce more consistent codes for independent episodes with the same theta.
The existing K=10 checkpoint did not become more invariant when evaluated at
K=2,5,7,10. This experiment trains with the longer context from initialization.

`configs/experiments/simplified_gaussian_k50.yaml` changes the context to 50
frames and the total episode length to 52. Frames 0--49 and actions 0--48 infer
one parameter code. Each opposite episode uses that code to predict frames
50 and 51, with actions 49 and 50. The existing maximum context capacity of 64
already accommodates K=50. Network widths, parameter count, initial seed,
DGP, optimizer, batch size, SIGReg weights and two-step forecast horizon match
the K=10 preset. The training budget remains 100,000 updates, so this compares
equal update counts rather than equal computation or observed frames.

Start with the dense condition, matching the current K=10 dense run. Use a fresh
output directory and fresh model initialization; a K=10 checkpoint is not the
starting point. Copy the new YAML to the server's existing checkout and run:

```bash
source .venv/bin/activate
export WANDB_PROJECT=sparse_causal_jepa
export WANDB_ENTITY=jesse-hoekstra-university-of-oxford

python -u scripts/run_controlled_experiment.py \
  --config configs/experiments/simplified_gaussian_k50.yaml \
  --condition dense \
  train.device=cuda wandb.mode=online
```

The preset sets W&B group `simplified_gaussian_m3_k50_action_100k` and output
directory `outputs/simplified_gaussian_m3_k50_action_100k`. Run in tmux if needed.
After an interruption, repeat this command with `--resume`. Keep training source,
configuration and Python/PyTorch versions fixed for resume. The new YAML and
standalone diagnostic do not change the training source fingerprint.

Compare the K=10 and K=50 dense runs at matching update counts. During training,
inspect `validation/parameter_within_theta_variance_ratio` together with total
parameter-code variance and normalized prediction MSE. Final evaluation also
reports `test/shuffled_parameter_prediction_mse_delta`: a positive increase
when using another system's code is evidence that the correct code helps
prediction. A lower context-variance ratio with retained total variance and
better predictive use would support the longer-context hypothesis. A reduction
in raw code variance alone is insufficient. Compare normalized prediction error
carefully because each run learns its own state representation.

The sampling law, parameter draws and observation map are matched by seed;
individual action trajectories are not matched prefixes across training runs
with different total lengths. This single-seed experiment tests a practical
training change; it does not establish a general context-length scaling law.

After training, copy `diagnostics/evaluate_context_variance.py` to the server
and measure multiple independent contexts per theta. This reads `last.pt`, its
adjacent `run_config.json`, and the experiment's persisted observation map:

```bash
python diagnostics/evaluate_context_variance.py \
  --checkpoint outputs/simplified_gaussian_m3_k50_action_100k/complementary/seed-0/dense/last.pt \
  --output-dir outputs/context_variance_k50 \
  --context-lengths 2 5 7 10 20 30 50
```

The diagnostic defaults to 512 parameter draws and 16 independent contexts per
draw, freezes the saved model, and uses nested trajectory prefixes. It saves
raw codes and variance estimates, including the noise-corrected variance
between theta groups. Its shorter-context results test that frozen K=50 model;
compare each separately trained model at its own training length for the main
K=10 versus K=50 comparison. K=2 is below the DGP's stated minimum recovery
context, and shorter lengths were not used during training.

For the same check on the K=10 checkpoint, substitute its checkpoint path,
use `--output-dir outputs/context_variance_k10`, and set
`--context-lengths 2 5 7 10`. These two diagnostics use the same seed and theta
draws but different maximum lengths, so their action prefixes are not paired.
Neither diagnostic changes the running experiment or writes to W&B.

## Optional ten-episode consistency comparison

`configs/experiments/simplified_gaussian_r10_consistency.yaml` keeps K=10, T=12
and the existing state/action/parameter sparsity comparison, but uses ten
independent episodes per theta, the transition parameter encoder, and parameter
consistency weight 1. The shared transition MLP encodes each within-context
`[state_t, action_t, state_next]` tuple and averages transition features before
its output head. It does not average parameter codes from different episodes.

Agreement is the mean squared difference over all distinct pairs of context
codes, equivalently twice their unbiased within-theta sample variance. It is
part of the calibrated base objective in dense, baseline and full. Training
assigns each recipient one different context's attached code by a random nonzero
cyclic shift; that code stays fixed through its two predictions. Evaluation uses
a deterministic shift by one. Every recovery/variance diagnostic scores the
individual context codes, and the shuffle test changes donor theta without
changing the recipient history, actions or targets.

A batch of 64 parameter groups now contains 640 episodes. SIGReg still operates
separately per view across 64 independent theta groups; the 640 episodes are not
treated as independent parameter samples. Equal update counts therefore cost
more than the original two-episode preset. This option needs a fresh output
directory, fresh dense training and its own calibration; neither the changed
encoder nor the changed base objective can reuse the original dense reference.
Select the new preset with the same runner command above. The original preset
retains its attention encoder, two episodes and zero consistency weight.
