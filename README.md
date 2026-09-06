# sparse_causal_jepa

Code accompanying **"Causal Identification within JEPA Using a SPARTAN"**
([manuscript](sources/SCJEPA.pdf)). There are two active experiments:

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Observations | True object states | Frame sequences of identical-looking balls |
| Context state | Simulator state | Recurrent SAVi slots and a learned state head |
| Prediction target | True next state | Stop-gradient EMA visual state |
| Parameter estimate | One per episode from the true-state context | One per episode from context slots |
| Training | Teacher forcing + eight sampled T=2 endpoints | Same local objective in learned state space |
| Preset | `bounce_baumgartner` | `bounce_visual_to_visual` |

The supervised visual-to-true-state bridge has been retired. The old three-experiment proposal
in [outdated_experiments.pdf](sources/outdated_experiments.pdf) is historical: its equation
references explain existing components, but its numbering and same-row EMA assumption are not
the current protocol. [Decisions D37–D39](docs/decisions.md) record the active design.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
python -m pytest tests/ -q
```

The project uses PyTorch, Hydra, W&B, scipy, and jaxtyping. Source code lives under
`src/scjepa/{data,models,losses,training,eval}`; runnable entry points are in `scripts/`,
Hydra presets in `configs/experiment/`, and implementation decisions in `docs/decisions.md`.
Data, model checkpoints, W&B files, and run outputs are not committed.

## Shared teacher-forcing and T=2 objective

Both experiments use 60-frame episodes and a 30-frame parameter-inference context. Infer one
attached `theta_hat` from times 0–29 and reuse it for all 30 teacher-forced transitions and all
auxiliary windows:

```text
L_pred = L_TF + lambda_rollout_t2 * L_AR2
```

For each episode, sample eight distinct offsets uniformly from 0–28. An offset `r` launches from
the observed state at `29+r` (a true state in Experiment 1, an online visual state in Experiment
2), generates the next state, feeds it back without detaching, and predicts the second state.
Only that second endpoint contributes to `L_AR2`; average the loss over batch, windows, objects,
and coordinates. `lambda_rollout_t2=0` bypasses the auxiliary branch and its random sampling.
There is no K=30 training loss or rollout curriculum. K=30 remains an evaluation diagnostic.

Experiment 1 uses the raw predictive error in its GECO constraint. Experiment 2 divides the
predictive error by detached, floored target content variance before adding the weighted logit
penalty to the constraint; the optimized predictive loss remains raw latent MSE. The path penalty
is outside the constraint in both experiments. Tau must be calibrated for each experiment's
actual objective and representation scale. Never reuse Experiment 1's historical `tau=0.02`
for the current TF+T=2 setup or for Experiment 2.

## Experiment 1: true states

The successful historical teacher-forcing run reached MCC 0.948 and SHD 4.81 (D30). Those values
predate the T=2 objective and are a reference, not a measured result for the new objective.
The full single-seed pipeline trains dense and identity references, selects a feasible tau, then
trains and evaluates the sparse model:

```bash
bash scripts/run_bounce_example.sh --run-tag=seed0 train.lambda_logit=1e-5
```

With an already selected fixed tau, the existing L40 launcher runs eight matched dense/sparse
seeds on two GPUs:

```bash
SCJEPA_BATCH_ID=exp1_8seed CUDA_VISIBLE_DEVICES=4,5 \
  bash scripts/l40_exp1_8seed_pipeline.sbatch TAU
```

See [scripts/README.md](scripts/README.md) for outputs and Slurm usage.

## Experiment 2: learn states from pixels

Both experiments reuse the same stored physics. The visual renderer draws equal-radius white
balls, so neither glyph size nor colour supplies an object identifier or a direct mass label.
Physical collision radii still depend on mass; occlusion and collision geometry may reveal
physical information. `mass_independent_init` is a separate optional control that changes the
physics distribution and needs its own preload. The canonical Experiment-1 preload must not be
regenerated on another machine.

The online branch encodes each source frame causally. The target branch is a frozen EMA copy of
the visual encoder and state head, and encodes the full sequence, including future target frames.
Its recurrent state at time `t` depends only on frames through `t`. The predictor receives the
online current state and the single parameter estimate from the initial context. During T=2
windows, its second call receives its own prediction.

**EMA does not guarantee that slot `i` tracks the same ball in both branches.** Each episode
therefore gets one detached Hungarian assignment between the branches' pre-head slot trajectories
on the shared context prefix. That permutation is fixed for the entire target sequence and every
training term. It uses no future frames, physical states, or masses. It can correct a branch-wide
permutation; it cannot repair an identity switch midway through a trajectory. Slot numbering is
local to an episode, and tracking still has to be measured. This matching is an implementation
choice consistent with the current manuscript's Hungarian matching description; the outdated
proposal's stronger same-row assumption is not used.

EMA and stop-gradient also do not prove non-collapse or recovery of a sufficient Markov state
(SCJEPA, PDF p.9, Assumption 2 and Remark 3). No reconstruction loss or anti-collapse regularizer
has been added. Low latent prediction error alone is insufficient: inspect temporal variation,
effective rank, object tracking, and held-out state probes before interpreting MCC/SHD.

Run the complete dense-calibration and sparse pipeline on one L40:

```bash
bash scripts/l40_exp2_pipeline.sbatch visual_seed0 1e-5 0
# Or on Slurm:
sbatch scripts/l40_exp2_pipeline.sbatch visual_seed0 1e-5 0
```

The first argument is a unique run tag; the remaining arguments are `LAMBDA_LOGIT [SEED] [STEPS]`
(default seed 0 and 300,000 steps). Outputs go to `outputs/bounce_exp2_visual_seed0/{dense,main}`.
The launcher requires the corresponding physics preload, calibrates tau from the dense model's
held-out normalized constraint, rejects grossly collapsed dense references, and evaluates the
sparse model on a disjoint split. Dense and sparse models learn separate target feature spaces;
equal normalized errors therefore need not imply equal physical fidelity. This is an exploratory
experiment: the inherited `1e-5` logit coefficient and visual convergence have not been
established by full runs.

## Concise results to inspect

Use the existing `eval/mcc`, `eval/shd`, and `eval/path_density` together with TF/T=2 losses and
constraint/dual health. A low SHD with low MCC can be empty-graph collapse. Experiment 2 adds
`slot_centroid_rmse`, `slot_switch_rate`, and `branch_slot_disagreement`, frozen linear probes
(`position_probe_r2`, `velocity_probe_r2`), and one small `slot_tracks.png` panel.
The probes fit on training episodes and score separate held-out episodes, using the prediction
window in both cases. They test whether true states are linearly recoverable from the learned
representation, not whether individual latent coordinates
literally equal `(x,y,vx,vy)`. The slot panel shows allocations over frames and should make merged
objects, background slots, and switching visible without adding a large gallery of plots.
Its shared color scale is relative to uniform attention, so diffuse slots cannot look sharply
localized through independent contrast stretching.

A finite CPU smoke confirms that the implementation runs; successful identification still needs
the L40 run and inspection of these held-out results.
