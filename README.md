# sparse_causal_jepa

Codebase for **"Causal Identification within JEPA Using a SPARTAN"** (Jesse Hoekstra, Oxford
Statistics; manuscript in `sources/my_paper.pdf`). It combines a temporal parameter encoder with
a SPARTAN-style sparse transition model and evaluates causal-graph and parameter recovery with SHD
and MCC. Experiment 1 operates directly on object states; Experiments 2 and 3 introduce visual
encoders and their regime-specific representation losses.

**Decision log:** [`docs/decisions.md`](docs/decisions.md) is the source of truth for settled
design decisions: a short list of standing rules (framework, vendoring, tooling, SPARTAN
interpretations, simulator contract, pipeline consistency, grad-skip guard) followed by
D27–D37, which define the metrics, Experiment-1 architecture, verified result, and current
teacher-forcing-plus-T=2 protocol.
Read it before changing anything it covers.

> **NOTE:** the historical architecture/ladder discussion below still contains pre-D29 material.
> Its removed `--identity-check`, `train.lambda_reg`, `health/target_slot_std_*`, and `mass_mcc`
> names are not current APIs. D34–D36's K=30 state-to-state training curriculum is also
> superseded. See `CLAUDE.md` and D37 in `docs/decisions.md` for the active protocol.

## Repo map

```
pyproject.toml            # packaging + ruff/pyright/pytest config in one place
.pre-commit-config.yaml   # hygiene + ruff + pyright (strict) gates
docs/decisions.md         # decision log — source of truth
sources/                  # papers (my_paper, SPARTAN, SAVi++, VISReg, ...)
src/scjepa/
  third_party/            # vendored reference code + PROVENANCE.md convention (see its README)
  models/                 # SAVi, channel split, SPARTAN           (model-architecture-engineer)
  losses/                 # predictive loss, VISReg/SIGReg, sparsity (paper-to-code-translator)
  data/                   # CLEVRER, Push-T, synthetic systems     (data-pipeline-engineer)
  training/               # loop, optim, logging                   (experiment-infra-engineer)
  eval/                   # SHD/MCC, probes, rollouts              (experiment-infra-engineer)
configs/                  # Hydra configs                          (experiment-infra-engineer)
scripts/                  # train.py, eval.py, prepare_data.py     (experiment-infra-engineer)
tests/                    # fast CPU pytest suite                  (test-and-ci-engineer)
```

`data/`, `checkpoints/`, `wandb/`, and `outputs/` are gitignored and never committed.

## Quickstart

Requires Python 3.12 (pinned in `pyproject.toml`).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Sanity checks:

```bash
python -c "import scjepa; print(scjepa.__version__)"
ruff check .
pyright
pytest
```

Stack: PyTorch · Hydra · Weights & Biases · einops · scipy · jaxtyping — exact pins and their
rationale in `pyproject.toml`.

## Current Experiment-1 predictive objective

Experiment 1 has returned to the stable 30-transition teacher-forcing foundation and adds one
fixed local-composition term:

```text
L_pred = L_TF + lambda_rollout_t2 * L_AR2
```

One `theta_hat` is inferred from `S_0,...,S_29` and reused for all 30 teacher-forced suffix
transitions and all auxiliary windows. For each episode, training samples exactly eight distinct
valid two-step offsets uniformly without replacement, independently of the other episodes in the
batch. With context length 30 and sequence length 60, the offsets are `0,...,28`; offset `r`
launches from the true anchor `S_(29+r)`, predicts `Shat_(30+r)`, feeds that generated state back,
and predicts `Shat_(31+r)`. Only the second prediction is supervised by `L_AR2`, because the first
transition is already covered by teacher forcing. The intermediate prediction is not detached.
The loss is averaged over episodes, eight windows, objects, and coordinates.

There is no K=30 training loss, horizon curriculum, warmup, gradient-cut schedule, or
full-rollout backpropagation. A deterministic K=30 chain remains only as a fixed-held-out,
no-gradient observational-equivalence diagnostic. It reports the fraction of episodes whose
worst-step coordinate-normalized NRMSE is at most `oe_tolerance_nrmse`, plus p50 and p95 of that
worst-step error. Training horizon 2 and evaluation horizon 30 intentionally differ: T=2 targets
local composition and exposure bias; the held-out K=30 diagnostic measures approximate trajectory
agreement. Neither establishes the population observational-equivalence assumption, and T=2 does
not claim to identify every physical parameter by itself.

## Worked example: identifiability on bounce (CPU)

The **bounce** system (D11) is 5 balls with per-episode sampled masses colliding elastically;
every episode carries full ground truth (frames, states, masses, time-indexed contact graph).
The GT-embedding diagnostic regime (`model.type: states`) runs the channel split + SPARTAN on
ground-truth object states — slot i ≡ ball i by construction — so the identifiability metrics
are directly meaningful.

**One command runs the current procedure** — τ calibration (fully-connected reference, sparsity
off), the main sparsity run with the calibrated τ, and terminal identifiability evaluation. Supply
the `lambda_logit` selected by the dense sweep:

```bash
LAMBDA_LOGIT=YOUR_SELECTED_VALUE
bash scripts/run_bounce_example.sh --run-tag=seed0 \
  "train.lambda_logit=${LAMBDA_LOGIT}"
#    -> prints a freshly calibrated tau, then TF/T=2 losses, OE diagnostics, SHD, MCC, path_density
#    -> saves recovery_grid.png with all mass/latent pairs and the global assignment
```

Hydra overrides are passed to every run, so the references and main run cannot diverge
in config (the D12 rule). Knobs are script flags — `--tau-factor` (τ = factor ×
fully-connected held-out constraint loss, default 1.0), `--calib-steps`,
`--main-steps` (main run only), and `--run-tag` (required for parallel launches) — a mistyped
flag errors loudly; the equivalent env vars still work as a fallback.

**What to watch:**

1. *Branch stability.* `train/loss_teacher_forcing` and `train/loss_rollout_t2_raw` should remain
   finite, as should the corresponding branch gradients when logged. The D18 skip guard remains
   active; a sustained rejected-update sequence is a failure, not a curriculum phase.
2. *Stale tau.* Tau is objective- and scale-dependent. Changing the T=2 coefficient, anchor
   count, logit coefficient, data geometry, or model invalidates the old calibration. If the
   achievable constraint remains above tau, the dual can grow without pruning. Recalibrate from
   a matching dense run.
3. *Trajectory agreement.* Watch all three fixed-held-out OE metrics together. Satisfaction is
   thresholded, so interpret it with the continuous worst-step NRMSE p50/p95 curves. They are
   diagnostics, not training losses or a population guarantee.

Healthy sparse training still shows the SPARTAN dual dynamics: `sparsity/constraint` crosses tau,
then `sparsity/lambda` reverses and decoded-state `sparsity/path_density` shrinks while the
constraint stays near tau. A short smoke run establishes finite computation and gradients only;
it does not establish convergence, parameter recovery, or observational equivalence.

## The experiment ladder (bounce) — STALE, superseded by D29

Every rung uses the same one-command runner (τ auto-calibrated per rung; overrides apply to both
runs); repeat with `train.seed=0..7` for seeded statistics. Periodic W&B evaluation contains only
the core curves: `pred_loss`, `constraint_loss`, `mean_abs_logit`, `gate_entropy`, `mass_mcc`,
`shd_state`, `shd_param_aligned`, and `path_density`. The final report also saves the full
permutation-aware recovery matrix and assignment in `recovery_grid.png` and
`recovery_alignment.json`. Healthy training always shows: `loss/logit` falls early (when enabled),
`sparsity/constraint` drops below τ, then `sparsity/lambda` falls and the decoded-state-row
`sparsity/path_density` shrinks. In learned-target runs, monitor
`health/target_slot_std_*` for collapse; in the raw-state rung it is a fixed data statistic.

```bash
# Use the coefficient selected by the dense sweep described below.
LAMBDA_LOGIT=YOUR_SELECTED_VALUE

# Rung 1 — Baumgartner-aligned environment with a true-state JEPA (radius∝mass,
# logit loss). Their Fig. 3 MCC ≈ 0.9+ is context, not a like-for-like target:
# the encoder/objective differ. Successful recovery still gives sharp marginals.
bash scripts/run_bounce_example.sh --tau-factor=1.0 \
  experiment=bounce_baumgartner "train.lambda_logit=${LAMBDA_LOGIT}"

# Rung 1-ablation — ±sparsity (their MLP/Transformer comparison; note their own
# finding: on bounce even an unregularised Transformer disentangles, so expect a
# smaller gap here than on dual particle):
bash scripts/run_bounce_example.sh --tau-factor=1.0 \
  experiment=bounce_baumgartner \
  "train.lambda_logit=${LAMBDA_LOGIT}" \
  train.sparsity_enabled=false

# Rung 2 — invisible mass (equal radii, uniform masses): identical otherwise, so
# any MCC drop vs rung 1 isolates the weaker sufficient-variability (mass acts
# only through collision impulses). MCC ≈ rung 1 -> method robust; MCC ≈ 0 -> edge found.
bash scripts/run_bounce_example.sh --tau-factor=1.0 \
  experiment=bounce_baumgartner \
  "train.lambda_logit=${LAMBDA_LOGIT}" \
  data.radius_from_mass=false data.mass_normal=null

# The resolved Stage-1 config already uses 60-step trajectories with a
# 30-step inference context. Rung 2.5 — a recurrent per-object encoder that
# isolates representation learning — remains planned.

# Rung 3 — pixels, invisible mass (the paper's claim; SAVi from pixels):
python scripts/train.py data.name=bounce data.clip_len=10 train.steps=...   # vision regime
#   NOTE: MCC/SHD eval for learned slots awaits the slot<->object alignment probe;
#   until then only training health (pred loss, collapse metrics) is reportable.

# Negative control — pixels + VISIBLE mass (radius rendered): expect prediction to
# stay good while param->state edges prune away and MCC on θ̂ collapses — the
# parameter migrates into the state channel (D13 scope condition), motivating the
# observability assumption in the manuscript.
python scripts/train.py data.name=bounce data.radius_from_mass=true ...
```

Scale: Baumgartner's setting is ~300k steps × 8 seeds (their Fig. 17/3); the D37
state-to-state preset uses the last stable 300k-step budget. The historical successful
teacher-forcing run used `lambda_logit=1e-5` and `sparsity_lambda_init=1e4`. Its `tau=0.02`
must not be reused: the dense stage must calibrate tau for
`L_TF + lambda_rollout_t2*L_AR2 + lambda_logit*L_logit`. Baumgartner does not specify the
numerical `lambda_logit`, so first run the controlled
dense-model sweep on Isambard:

```bash
sbatch --account=<PROJECT> scripts/isambard_logit_sweep.sbatch logit_seed0 0
# The job prints selected_lambda_logit and writes it to sweep_summary.json.
SWEEP=outputs/lambda_logit_sweep_logit_seed0/sweep_summary.json
LAMBDA_LOGIT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_lambda_logit"])' "$SWEEP")
sbatch --account=<PROJECT> scripts/isambard_exp1_pipeline.sbatch full_seed0 "$LAMBDA_LOGIT" 0
```

The sweep compares prediction against the `lambda_logit=0` control, rejects values degrading it
by more than 5%, and selects the smallest Pareto coefficient that obtains 90% of the best
admissible reduction in the excess logit penalty above its theoretical floor of 2. Mass recovery
is displayed as a validation diagnostic, not used by that rule. Dense attention can only screen
the coefficient; the gated pipeline must still reach τ, move λ away from its ceiling, reduce path
density/SHD, and retain `mcc`.

If compute nodes have no internet, add `wandb.mode=offline` and sync afterwards. Every final eval
writes `metrics.json`; aggregate seeded runs with
`python scripts/aggregate_runs.py 'outputs/bounce_example_rung1_seed*/main'`.
