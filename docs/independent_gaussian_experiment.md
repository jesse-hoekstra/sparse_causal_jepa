# Independent Gaussian states: server experiment

This experiment uses the approved nonlinear two-reflection and radial-rotation
DGP. It is selected explicitly with `data_family: independent_gaussian`; the
older `controlled_vector` process and presets retain their original definitions.

The complementary graphs have 50 state edges, 100 action edges, and 20 parameter
edges. Every group independently samples `theta ~ N(0,I10)` and contains two
episodes with independent `z0 ~ N(0,I10)` and independent standard Gaussian actions.
No actions depend on states or parameters, and no samples are filtered.

## Dynamics

For a nonzero five-dimensional vector, define

```text
H(v) = I5 - 2 v v^T / (v^T v)
D = diag(1,2,3,4,5)
Q(v) = H(v) H(Dv)
v1(a) = (1,a1,a2,a3,a4)
v2(a) = (1,a5,a6,a7,a8)
U(a) = diag(Q(v1(a)), Q(v2(a)))
V = I10 - 0.2 J10

b = U(a) z
c = V a
alpha_i = 0.1 tanh(theta_i)
r_i^2 = b_i^2 + c_i^2
phi_i = alpha_i + 0.5 r_i^2 / (1+r_i^2) alpha_partner(i)
z_next_i = b_i cos(phi_i) + c_i sin(phi_i)
```

The partner swaps the two halves: `1 <-> 6, ..., 5 <-> 10`. Action coordinates
are used directly, without squaring or a periodic preprocessing transform.
Actions 9 and 10 still affect every output through the dense matrix `V`.

Orthogonal mixing preserves Gaussian distributions. Conditional on a fixed
parameter, the pairs `(b_i,c_i)` are independent standard two-dimensional
Gaussians. Rotating each pair by an angle depending on its radius preserves that
distribution. Therefore `z_t | theta ~ N(0,I10)` at every frame, in exact
arithmetic. This is same-time coordinate independence; the transition retains
cross-time dependencies. Simulation uses float64, then converts observations and
actions to the training dtype.

Observations retain the fixed invertible vector map `x = psi(Wz)`,
`psi(u) = u + 0.1 u^3`. A later image experiment can replace this observation map
without changing the sampling or transition interfaces.

The construction satisfies the true-system assumptions needed for the paper's
state-identification result, including the fixed-output Jacobian spans. Parameter
recovery uses two observed transitions and is smooth on an open informative
domain containing generated contexts with probability one. The paired parameter
graph still allows within-pair parameter ambiguity under the theorem; it does not
satisfy the separate graphical criterion for full parameter disentanglement.
Learned injectivity, predictive equivalence and sufficient sparsity remain
properties to check in the fitted model.

## Protocol

Each episode contains five frames, `z0 -> z1 -> z2 -> z3 -> z4`.
The parameter encoder receives frames 0, 1, 2 and actions 0, 1 (`K=3`).
Using the other episode's parameter code, the predictor starts at frame 2 and
predicts frame 3 with action 2, then frame 4 with action 3 using its own previous
prediction (`L=1`). Predictions remain attached across both steps.

The preset runs dense, baseline and full sequentially. Each stage has 100,000
steps, batch size 64 parameter groups, constant learning rate 0.001, EMA decay
0.996, and SIGReg weights 0.1 for states and parameters. SIGReg is applied
separately by view and frame across independent parameter groups.
These optimization settings are starting choices, not validated performance
claims. Existing SPARTAN, EMA, GECO and checkpoint selection behavior is retained.

Dense `best.pt` calibrates tau and initializes both sparse stages. Every stage's
final test evaluation uses its own `last.pt`. Baseline forces parameter gates
open; full learns them. Calibration, validation and test groups remain separate.

Only the complementary **data graph** is supported for this DGP. This does not
restrict the **training condition**: dense, baseline and full all use that same
complementary dataset.

## Server command

Copy the updated source to the server first. From its root, activate a Python
3.12 environment with the server's CUDA-compatible PyTorch installation and install
the project:

```bash
source .venv/bin/activate
python -m pip install -e '.[logging]'
export WANDB_PROJECT=sparse_causal_jepa
export WANDB_ENTITY=jesse-hoekstra-university-of-oxford
```

W&B uses the existing credentials on that server; run `wandb login` if necessary.
Check the run plan without starting training or W&B:

```bash
python scripts/run_controlled_experiment.py \
  --config configs/experiments/independent_gaussian.yaml \
  --dry-run train.device=cuda
```

Then run in a persistent terminal, such as a tmux session:

```bash
mkdir -p logs
set -o pipefail
python -u scripts/run_controlled_experiment.py \
  --config configs/experiments/independent_gaussian.yaml \
  train.device=cuda \
  wandb.mode=online \
  wandb.group=independent_gaussian_100k \
  output_dir=outputs/independent_gaussian_100k \
  2>&1 | tee -a logs/independent_gaussian_100k.log
```

This command starts all three stages. To train only the dense reference first,
add `--condition dense` before the configuration overrides. To continue afterward,
repeat the full command with `--resume` and without `--condition dense`.
After an interruption, repeat the same command with `--resume`; completed stages
are skipped and interrupted stages restore their own training state.

Use a fresh output directory. A dense checkpoint or tau from `controlled_vector`
is incompatible, even though observation and latent dimensions match.
Keep the source and scientific configuration unchanged during a resumable run.
The code validates family, generator version, data, observation map, calibration,
and checkpoint identities.

The small CPU preset is an infrastructure check only:

```bash
OMP_NUM_THREADS=1 python scripts/run_controlled_experiment.py \
  --config configs/experiments/independent_gaussian_smoke.yaml \
  wandb.mode=disabled \
  output_dir=/tmp/scjepa-independent-gaussian-smoke
```

Its scores are not evidence of scientific recovery.
