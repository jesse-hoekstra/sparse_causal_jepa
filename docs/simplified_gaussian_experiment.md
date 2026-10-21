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
