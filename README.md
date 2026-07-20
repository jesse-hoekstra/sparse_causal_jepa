# sparse_causal_jepa

Codebase for **"Causal Identification within JEPA Using a SPARTAN"** (Jesse Hoekstra, Oxford
Statistics; manuscript in `sources/my_paper.pdf`). A joint-embedding predictive architecture over
SAVi object slots whose predictor is a SPARTAN-style sparse transition model: a per-slot temporal
attention pooling extracts time-invariant causal parameters from the slot history, a linear head
carries the kinematic state, and SPARTAN predicts the next-step slots through a sparse interaction
graph — trained end-to-end from scratch with a Hungarian-matched single-step predictive loss and a
VISReg anti-collapse regularizer, so that causal structure can be identified from the learned graph
and parameters (SHD/MCC against ground truth) without any reconstruction objective.

**Decision log:** [`docs/decisions.md`](docs/decisions.md) is the source of truth for settled
design decisions (D1 PyTorch, D2 SAVi, D3 VISReg, D4 attention pooling, D5 reuse-first,
D6 single-step loss, D7 from-scratch training, D8 packaging & tooling). Read it before changing
anything it covers.

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
pytest   # collects from tests/; 0 tests is expected until modules land
```

Stack: PyTorch · Hydra · Weights & Biases · einops · scipy · jaxtyping — exact pins and their
rationale in `pyproject.toml`.

## Worked example: identifiability on bounce (CPU)

The **bounce** system (D11) is 5 balls with per-episode sampled masses colliding elastically;
every episode carries full ground truth (frames, states, masses, time-indexed contact graph).
The GT-embedding diagnostic regime (`model.type: states`) runs the channel split + SPARTAN on
ground-truth object states — slot i ≡ ball i by construction — so the identifiability metrics
are directly meaningful.

**One command runs the whole procedure** — τ calibration (fully-connected reference, sparsity
off), the main sparsity run with the calibrated τ, and identifiability evaluation. Add
`--identity-check` for a matched mass-blind reference and automatic feasibility guard:

```bash
bash scripts/run_bounce_example.sh                                   # 5-ball default
bash scripts/run_bounce_example.sh --identity-check                 # paper-grade guard
bash scripts/run_bounce_example.sh data.num_balls=3 train.steps=25000  # smaller/faster instance
#    -> prints the calibrated tau, then pred_loss, shd_state, shd_param, mcc, path_density
#    -> saves recovery_grid.png (Fig.-5/12-style: true mass i vs slot j) in the run dir
```

Hydra overrides are passed to every run, so the references and main run cannot diverge
in config (the D12 rule). Knobs are script flags — `--tau-factor` (τ = factor ×
fully-connected held-out constraint loss, default 1.0), `--identity-check`, `--calib-steps`,
`--main-steps` (main run only), and `--run-tag` (required for parallel launches) — a mistyped
flag errors loudly; the equivalent env vars still work as a fallback.

**What to watch (two failure modes we hit while tuning this — see decisions.md D12):**

1. *Collapse-gaming the constraint.* τ is scale-dependent and the embeddings are trainable, so a
   weak regularizer (`train.lambda_reg` < 1) lets the model satisfy `pred_loss ≤ τ` by shrinking
   the target embeddings instead of learning dynamics. Symptom: `health/target_slot_std_*` falls
   together with `loss/pred` while `sparsity/lambda` collapses and SHD worsens. Keep
   `lambda_reg: 1.0`.
2. *Stale τ.* τ calibrated under any other setting (different `lambda_reg`, clip length, …) is
   meaningless: if the achievable loss sits above τ, λ_s rises to its clamp and sparsity never
   engages (symptom: `sparsity/lambda` huge, `sparsity/path_density` never falls). Recalibrate
   after any config change.

Healthy training shows the SPARTAN A.2 dynamics: `sparsity/constraint` drops below τ first,
then `sparsity/lambda` falls and decoded-state `sparsity/path_density` shrinks while the
constraint stays near τ.
Reference run (2026-07-10, `data.num_balls=3 train.steps=25000`, `TAU_FACTOR=2.0`, ~1 h CPU):
FC pred_loss 0.026 → τ 0.051; main run ends with λ 1000→436 (pruning engaged, still in
progress), target-slot std 0.83 (no collapse), held-out pred_loss 0.039, shd_state 4.4.
Full pruning and parameter recovery (MCC) need longer runs — SPARTAN's own curves span ~10⁶
steps — so treat the CPU run as the pipeline check and scale steps up (GPU/overnight) for the
paper-grade numbers. Step counts per phase: `--calib-steps` always sets the calibration length;
`--main-steps` (or a `train.steps=...` hydra override) sets the main run's.

## The experiment ladder (bounce; decisions.md D13)

Every rung uses the same one-command runner (τ auto-calibrated per rung; overrides apply to both
runs); repeat with `train.seed=0..7` for seeded statistics. Each run prints
`pred_loss, shd_state, shd_param, mcc, mcc_linear, mcc_pooled, path_density` on a held-out
split and saves
`recovery_grid.png`. Healthy training always shows: `loss/logit` falls early (when enabled),
`sparsity/constraint` drops below τ, then `sparsity/lambda` falls and the decoded-state-row
`sparsity/path_density` shrinks. In learned-target runs, monitor
`health/target_slot_std_*` for collapse; in the raw-state rung it is a fixed data statistic.

```bash
# Rung 1 — Baumgartner-aligned environment with a true-state JEPA (radius∝mass,
# logit loss). Their Fig. 3 MCC ≈ 0.9+ is context, not a like-for-like target:
# the encoder/objective differ. Successful recovery still gives sharp marginals.
bash scripts/run_bounce_example.sh --identity-check --tau-factor=1.0 \
  experiment=bounce_baumgartner

# Rung 1-ablation — ±sparsity (their MLP/Transformer comparison; note their own
# finding: on bounce even an unregularised Transformer disentangles, so expect a
# smaller gap here than on dual particle):
bash scripts/run_bounce_example.sh --identity-check --tau-factor=1.0 \
  experiment=bounce_baumgartner \
  train.sparsity_enabled=false

# Rung 2 — invisible mass (equal radii, uniform masses): identical otherwise, so
# any MCC drop vs rung 1 isolates the weaker sufficient-variability (mass acts
# only through collision impulses). MCC ≈ rung 1 -> method robust; MCC ≈ 0 -> edge found.
bash scripts/run_bounce_example.sh --identity-check --tau-factor=1.0 \
  experiment=bounce_baumgartner \
  data.radius_from_mass=false data.mass_normal=null

# The resolved Stage-1 config already uses 60-step trajectories with a
# 30-step inference context. Rung 2.5 — a recurrent per-object encoder that
# isolates representation learning — remains planned.

# Rung 3 — pixels, invisible mass (the paper's claim; SAVi from pixels):
python scripts/train.py data.name=bounce data.clip_len=10 train.steps=...   # vision regime
#   NOTE: MCC/SHD eval for learned slots awaits the slot<->object alignment probe;
#   until then only training health (pred loss, collapse metrics) is reportable.

# Negative control — pixels + VISIBLE mass (radius rendered): expect prediction to
# stay good while param->state edges prune away and MCC on S^ph collapses — the
# parameter migrates into the state channel (D13 scope condition), motivating the
# observability assumption in the manuscript.
python scripts/train.py data.name=bounce data.radius_from_mass=true ...
```

Scale: their setting is ~300k steps × 8 seeds (their Fig. 17/3) — the full grid wants a
cluster: `sbatch scripts/slurm_rung1.sbatch` runs all 8 seeds in parallel (one srun task per
seed, τ re-calibrated per seed, W&B runs named `bounce_baumgartner-{phase}-seed{N}`). If compute
nodes have no internet, add `wandb.mode=offline` and `wandb sync outputs/...` afterwards. For
parallel launches the runner needs `RUN_TAG` set per task (the sbatch script does this) — the
default timestamped output dir collides across simultaneous starts. Every eval writes
`metrics.json` into its run dir; aggregate the seeded grid (mean ± SD plus the five-number
summary behind their Fig.-3-style box plots) with:
`python scripts/aggregate_runs.py 'outputs/bounce_example_rung1_seed*/main'`.
