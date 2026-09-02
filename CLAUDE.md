# SCJEPA — "Causal Identification within JEPA Using a SPARTAN"

PyTorch research codebase for Jesse's paper (`sources/my_paper.pdf`). SPARTAN predictor
(`sources/SPARTAN.pdf`, no public code) inside a JEPA, with the bounce identifiability
experiment replicated from Baumgartner et al. (`sources/dynamical_system.pdf`).
Settled design decisions live in `docs/decisions.md` (D1–D37) and BIND all work.
Subagent roster and shared conventions: `.claude/agents/README.md`.

**STATUS: Experiment 1 has returned to its stable teacher-forcing foundation and now adds only
eight sampled T=2 autoregressive endpoints (D37).**
Under the pure teacher-forced objective, sparsification pruned the graph AND recovered the
masses (D30, run 7cq3h2ur): `mcc` 0.948, `shd` 4.81, `path_density` 0.223, L_path 12.3, zero
skipped steps in 300k, at τ=0.02 (a TEST value, ~1.5–2× dense, not the §6.1.3 τ=1.0×dense).

**D34–D36's Experiment-1 K=30 training objective and continuation schedule are superseded.**
Run `b8v5lxu2` reached uncut full BPTT but again failed through recurrent-gradient instability
while teacher-forcing gradients remained healthy. Experiment 1 now trains all 30 true-anchored
one-step transitions plus eight independently sampled, true-anchored T=2 windows per episode.
Only each window's second prediction is supervised; its first generated prediction is fed back
without detaching. One θ̂ inferred from states 0..29 is shared by every transition and window.
There is no rollout curriculum, warmup, gradient-cut schedule, accepted-update stage state, or
K=30 training/backpropagation. K=30 is evaluation-only under `torch.no_grad()`.

Current initial settings are the stable 300k-step foundation, `lambda_logit=1e-5`,
`sparsity_lambda_init=1e4`, `lambda_rollout_t2=1`, and eight anchors. τ is still on a new scale:
D30's 0.02 and every K=30-era threshold are void. The dense pipeline must freshly calibrate the
teacher-forcing-plus-T=2 constraint.

## Non-negotiable working rules
- **Verify against the paper, not the name.** Never state a name-based or memory-based guess
  about a paper as fact — open the PDF (`.venv/bin/python -c "import pypdf; ..."`) and cite the
  page/equation. Label interpretations as interpretations, in code comments too.
- **Empirically test stability claims before committing to them.** Two costly failures were
  diagnoses that "sounded right" (raise lambda_logit; steepen the penalty wall) and were
  disproven by 1500-step smoke runs in minutes. Smoke first, then edit.
- **A metric that CAN leave its failure value MUST be watched.** Frozen-from-step-5000 eval
  metrics (v2: shd/density constant for 295k steps) mean the experiment died early; don't let a
  run finish before checking the first two eval points.

## Run-health signatures (states regime, bounce)
**Reference run `7cq3h2ur` (D30) — measured under the PRE-D34 pure-TF objective.** Its four
phases are a useful qualitative reference for the GECO dual, but its NUMBERS should not be
expected to transfer: D37's constraint also carries the weighted T=2 endpoint term. Compare a new
run against the SHAPE below, not the values. These ranges are historical D30 coordinates, not a
schedule. Under D37 the path objective and GECO are active from the start of a sparse run; there
is no dynamics-pretraining stage to add to the step axis.
1. **Saturation (~0–75k):** c > τ, so `sparsity/lambda` CLIMBS (1e4 → 4.5e7), `path_density`
   → 1.0, `loss/sparsity` rises to ~2400, `shd` parks at ~42, `mcc` loiters 0.25–0.30.
   This is correct and looks exactly like the D26 never-prunes failure. Do not kill a run here.
2. **Reversal (~75–90k):** `sparsity/constraint` drops under τ; λ peaks and rolls over. Watch
   the CONSTRAINT, not either predictive branch alone — under D37 the bound is the scalarised
   TF + weighted-T=2 + logit sum, so one component can sit below τ while c remains above it.
3. **Pruning cascade (~90–155k):** λ falls ~3.5 orders, `path_density` 1.0 → 0.27,
   `loss/sparsity` 2400 → 25, `shd` 42 → 5, and **`mcc` climbs 0.46 → 0.935 inside this same
   window** — identification is concurrent with pruning, which is the experiment's whole claim.
4. **Equilibrium (~155k+):** λ settles into 40–5000 (2272 here), `eval/constraint_loss` pinned
   AT the τ boundary (0.015–0.021 vs τ=0.02), `attention/mean_abs_logit` GROWING (3.1 → 3.9)
   and `attention/gate_entropy` FALLING (0.25 → 0.19) = gates committing, not sitting in the
   p≈0.5 limbo that killed D26.

Point checks that still apply: `health/skipped_steps` should remain zero, branch gradients should
stay finite, and `eval/path_density` should eventually be strictly between 1/T and 1. Compare
`train/loss_teacher_forcing` with held-out teacher-forcing loss and watch
`train/loss_rollout_t2_raw` for a growing local-composition error; neither old K=30 branch norms
nor curriculum-stage coordinates are meaningful now. The fixed-held-out OE satisfaction/p50/p95
series should change over training; a frozen series together with frozen MCC/SHD can signal a
stalled optimizer. A large train/eval gap can still indicate memorization (run jkslgj4o
2026-07-19 was 14x at `num_clips=4000`), so rescale `data.num_clips` whenever capacity or steps
change.

Failure catalog — THREE LIVE:
1. **Logit-penalty explosion** (run n5zq9nct, 2026-07-11): loss/logit ≫ loss/pred, grad
   spikes 1e5–1e15, lambda railed at 1e6 → fixed by 1/√d on gate logits + masked-softmax
   numerics + constraint_loss calibration (commit 55b5282; Claude memory
   `project-spartan-logit-stability`).
2. **Empty-graph collapse** (run qqye6ug1, 2026-07-12): graph pruned to identity
   (density = 1/T exactly), param→state edges dead, `mcc` at the noise floor, `shd` ≈ 2.7 —
   i.e. the SECOND-BEST SHD is total failure. Full diagnosis in `docs/audits/2026-07-12-*.md`.
   Verified drivers: τ slack converted into gate-closure depth by the logit term INSIDE the
   constraint (F-9), τ calibrated on the gated model rather than a true dense one (F-8), and a
   dual step crossing its range in ~2k steps (F-10). D30 showed the cure: a τ the dense model
   actually beats, λ_logit small enough to leave commitment budget, and patience through
   phase 1. Note the logit term stays inside the constraint by design — that is Baumgartner
   Eq. 9, not an oversight (D34).
3. **Full-horizon BPTT explosion** (runs ecbjobkj and b8v5lxu2): neither a coefficient ramp nor
   D36's local-window/gradient-cut curriculum made uncut K=30 recurrent training stable.
   Teacher-forcing gradients remained ordinary while the recurrent branch dominated and the
   skip guard aborted. D37 removes K=30 from training and uses only one recurrent feedback step;
   K=30 survives as a no-gradient held-out diagnostic.

The D18 grad-spike guard stays in the loop regardless: a non-finite or absurd pre-clip grad norm
rejects the whole update (optimizer AND dual), and persistent skipping raises. Its origin was a
finite ~1e30 loss that passed `isfinite`, overflowed the backward pass to grad_norm = inf, and
let clipping multiply every gradient by zero — a run that "finished" 230k frozen steps with
byte-identical eval rows. That signature remains worth recognising even with the short T=2 path.

## Key mechanics (Experiment 1, fixed TF + T=2 objective — D37)
- **The objective has two predictive branches sharing one θ̂.** Pool θ̂ exactly once per
  episode from states 0..29 and reuse the same attached tensor for every teacher-forcing
  transition and every sampled window. Never re-encode, replace, or detach it within an episode.
  The dual form is

      L_pred = L_TF + λ_T2·L_AR2
      L = L_pred + λ_logit·L_logit + λ_s⁻¹·L_path
      c = L_pred + λ_logit·L_logit ≤ τ                 (path penalty excluded)

  `L_TF` remains the 30 true-anchored one-step predictions
  `S_29 -> S_30, ..., S_58 -> S_59`. No VISReg, Hungarian matching, or variance normalization
  is used in Experiment 1.
- **A rollout anchor is a true state that launches one independent T=2 window.** For actual
  sequence length T and context length C, a relative offset r gives

      S_(C-1+r) -> Shat_(C+r) -> Shat_(C+1+r),

  and is valid exactly when `0 <= r <= T-C-2`. At C=30/T=60 this is `r=0,...,28`.
  Sample exactly eight distinct valid offsets uniformly without replacement for every episode,
  independently across episodes. Sampling uses the ordinary checkpointed training RNG; do not
  freeze the same eight offsets across updates.
- **Only the endpoint is supervised by the auxiliary term.** The first call consumes the true
  anchor. The second call consumes the first generated prediction, not the true intermediate
  state, and that prediction is not detached. Thus the endpoint gradient passes through both
  shared transition calls. The first-step target is already supervised by `L_TF` and must not be
  added to `L_AR2` again. All eight windows share the episode's one θ̂.
- **The T=2 loss is a mean, not a sum:**

      L_AR2 = mean_(batch,window,object,coordinate)
              (Shat_(t+2) - S_(t+2))².

  Duplicating windows or changing the number used in a controlled test must not systematically
  multiply the scale. `lambda_rollout_t2=0` bypasses anchor sampling and both auxiliary
  transition calls, reproducing the teacher-forcing computation and RNG sequence exactly.
- **There is no Experiment-1 rollout training schedule.** No K=30 loss enters training, and
  there is no horizon curriculum, rollout-length warmup, gradient-cut progression,
  accepted-update stage counter, delayed sparsity activation, or full-rollout backpropagation.
  The D18 finite/pre-clip-gradient skip guard remains ordinary training-health protection.
- **H_train=2 and K_eval=30 intentionally differ.** T=2 trains one local composition and exposes
  the model once to its own output, targeting local composition error and exposure bias without
  the unstable product of 30 recurrent Jacobians. It does not claim that two steps identify all
  physical parameters. K=30 is a deterministic, evaluation-only diagnostic on fixed held-out
  episodes: in `model.eval()` under `torch.no_grad()`, start from true `S_29`, recursively feed
  predictions through `Shat_59`, and reuse the context's one θ̂. It never contributes to an
  optimizer or constraint gradient.
- **The OE metric is normalized and tolerance-based.** With fixed training-set coordinate
  standard deviations σ_d, compute each episode/step NRMSE across objects and coordinates, then
  `E_i=max_(k=1,...,30) e_(i,k)`. Log the held-out fraction with `E_i<=0.10` as
  `eval/oe_sample_satisfaction_k30` and p50/p95 of E as
  `eval/oe_k30_worst_step_nrmse_p50` and `eval/oe_k30_worst_step_nrmse_p95`. These estimate
  approximate agreement on sampled trajectories. They do not prove the population
  observational-equivalence assumption.
- **The eval harness must mirror the final predictive constraint for tau.** Dense calibration
  and sparse training use the same `lambda_rollout_t2`, anchor count, T=2 endpoint definition,
  and `lambda_logit`. Every D30 or D34–D36 tau is invalid. The K=30 OE diagnostic is not part of
  `constraint_loss` and cannot be used as tau.
- Model: `ParameterEncoder` (Eqs. 16–26: relational attention per timestep FIRST, then
  per-track temporal pooling, then a bare scalar head) + `Spartan` (Eqs. 27–37) with one
  FIXED non-trainable track key κ_i shared by state and parameter token i (buffer,
  seed-0 codebook, scale 0.02 = our choice). Dense A≡1 / token-local A≡0 references share
  the same modules and codebook. Verified endpoints: dense L_path = 6655, token-local = 5.
- Dual: log λ += α·MA[c − τ], UNCLAMPED both directions (§6.1.3). For D37 sparse
  training, the path term and GECO update are active from the start; there is no schedule gate.
  α=2e-2 is the one free numerical knob. λ₀=1e6 is the write-up's value; the restored stable
  Experiment-1 setting uses 1e4. Pre-D29 runs are NOT comparable.
- **Working configuration: λ_logit=1e-5, λ_T2=1.0, eight T=2 anchors, K_eval=30,
  300k attempted batches, λ₀=1e4, α=2e-2.** τ is not a number to carry over. Let the
  dense pipeline calibrate the final TF+T=2+logit constraint.
- Protocol τ = 1.0 × the held-out `constraint_loss` of ONE converged full-length dense run per
  architecture/seed, computed with the IDENTICAL T=2 settings (never a short reference — v2
  failure; never the gated model with sparsity off — F-8). The `--tau-max` token-local launch
  gate was DELETED in D34: its floor (0.043645 + λ_logit·2.0, run ku244l5e) was measured under
  the pure-TF constraint and is a different quantity now.
- λ_logit comes from the label-free dense sweep (§6.1.3 rule, grid recentered to
  0…1e-3 low end): feasibility arithmetic (2026-07-25) says values above ~3e-5 make the
  gated constraint set EMPTY at raw scale — gate commitment costs λ_logit·(2cosh|l|−2)
  inside a τ that has zero slack. 1e-5 is confirmed to work; |logit| should GROW past ~3
  (7cq3h2ur ended at 3.87), and gates that stay near 0.3 mean the run never committed.
  That arithmetic was done against the PURE-TF constraint and has not been recomputed for T=2,
  so treat the ~3e-5 ceiling as an untested bound rather than a measured D37 result.
- Training's predictive keys are exactly `train/loss_teacher_forcing`,
  `train/loss_rollout_t2_raw`, `train/loss_rollout_t2_weighted`, and `train/loss_total`.
  Branch norms, when enabled without material memory cost, are
  `train/grad_norm_teacher_forcing` and `train/grad_norm_rollout_t2_weighted`. Do not resurrect
  K=30 branch-gradient, stage, BPTT-depth, cut, or curriculum-progress metrics under new names.
- Gate/penalty logits are the SCALED q·k/√D_sp — now codified in the write-up (Eq. 31).
- Path matrix entries are path COUNTS (∏(A_l+I)); `path_density` = fraction of entries ≥ 0.5;
  identity-only matrix ⇒ density = 1/T (T=10 for 5-ball states regime ⇒ 0.100 exactly).
- **There is exactly ONE graph metric (D28, 2026-07-25): `eval/shd`** — SPARTAN's SHD (their
  Table 1; Baumgartner never use it), lower is better, between the LEARNED graph and the
  GROUND-TRUTH causal graph: decoded state rows x all 2N source tokens [state | params],
  range [0, 50] for 5 balls, same index set as `path_density`/the path objective.
  **NEVER read it alone** — the true graph has only 7.72 of 50 edges (measured n=2000, seed-29
  split over t ∈ I; 5 self-edges + 2.72 contact/mass edges), so the EMPTY graph scores 2.72 and
  the SATURATED one 42.28: lower-is-better favours the model that learned nothing by ~15x.
  Reference points: perfect 0 · empty/token-local 2.72 (mcc 0) · saturated/dense 42.28 ·
  **achieved 4.81 with mcc 0.948 (D30)** = 11.17 predicted edges, 91% recall, 63% precision.
  Success = `shd` falling far below the saturated 42 WITH `mcc` high (0 is ideal, 4.81 is what
  a working run looks like today); `shd`≈2.7 with `mcc`≈0 is empty-graph
  collapse (failure #2), i.e. the second-best SHD is a total failure. A frozen `shd` is not a
  model property while the graph is saturated or empty (it is then just the GT edge count).
  Superseded keys, NOT comparable: `shd_state`, `shd_param`, `shd_param_aligned`.
- **There is exactly ONE mass-recovery metric (D27, 2026-07-25): `eval/mcc`** = Baumgartner
  App. F.1 `mean_i max_j R²_ij` (`eval/parameters.py:nonlinear_mcc`), so it is directly
  comparable to their bounce reference ~0.9 (Fig. 3) — **we reach 0.948 (D30), i.e. at or above
  their bounce number.** It ramps late (their Fig. 17): in 7cq3h2ur it was still 0.29 at 80k and
  only crossed 0.9 at ~140k, so flat-low before ~100k steps is NORMAL as long as
  density/shd/lambda are still moving.
  Do NOT add a second recovery score. Runs before 2026-07-25 logged `eval/mass_mcc`
  (a stricter Hungarian one-to-one score) and `eval/shd_param_aligned` — different
  quantities, do not plot them on the same axis as the current keys.

## Commands
- **One pipeline per experiment** (each does data -> dense/tau -> sparse -> eval):
  - Exp 1 (true state): `sbatch --account=<P> scripts/isambard_exp1_pipeline.sbatch TAG LAMBDA_LOGIT [SEED]`
  - Exp 2 (visual ctx, true target): `sbatch --account=<P> scripts/isambard_exp2_pipeline.sbatch TAG LAMBDA_LOGIT [SEED] [STEPS]`
  - Exp 3 (visual ctx, EMA target): `sbatch --account=<P> scripts/isambard_exp3_pipeline.sbatch TAG LAMBDA_LOGIT [SEED] [STEPS]`
  Configs: `bounce_baumgartner` / `bounce_visual_to_state` / `bounce_visual_to_visual`. All three share
  the SAME physics preload; the visual ones render frames from it on the fly.
- **Reproduce the historical D30 pure-TF result** (~1.6 h on one GPU, 300k steps) only by
  explicitly setting `train.lambda_rollout_t2=0.0`; D37's default is TF+T=2. The historical
  `tau=0.02` is appropriate only to that exact reproduction, never to a D37 run.
- Full pipeline: `bash scripts/run_bounce_example.sh --run-tag=X ...` (dense A≡1 same
  length as main → τ = 1.0× its held-out constraint_loss → sparse run → 5000-episode eval;
  NO identity stage since D29, NO `--tau-max` gate since D34; `--calib-steps` only for smokes;
  other hydra overrides go to BOTH runs, D12).
- Cheap stability smoke (~3–5 min, CPU): 1500–3000 steps via `Trainer` directly with
  `data.num_clips=200` at paper geometry (clip_len 60, context_len 30). D37's T=2 branch is active
  from the first update, so the smoke tests the actual fixed predictive objective. Healthy means
  finite TF/T=2 losses and gradients, zero catastrophic skip sequences, and no NaNs. It does not
  establish convergence.
- **NEVER regenerate `data/bounce_train_v2_100000.pt`.** The simulator is deterministic on a
  given machine but NOT across machines: regenerating the identical config on the Mac vs the
  server diverges chaotically (measured n=200: 194/200 episodes differ, max |delta| 3.4, and
  **26% get a different contact graph** — i.e. a different ground-truth causal graph). The file
  is the physics of record for D30 and for Experiment 2. Visual runs render frames FROM it.
  The current H_train=2 and K_eval=30 both fit the 60-step clips exactly. If T ever has to grow,
  LENGTHENING `clip_len` is a pure prefix EXTENSION, not a regeneration — measured n=40, first
  60 states and 59 contacts bit-identical, max |delta| 0.0. It is still only safe as a NEW file
  generated on the SAME machine as the existing one; `clip_len` is part of the preload identity,
  so a mismatched run raises rather than silently truncating.
- Plot an episode: `python scripts/plot_bounce_episode.py --condition states|visual [--show-collision-radii]`
  (`visual` shows the real 64x64 encoder input). Note the renderer puts y DOWN (`grid_y` indexes
  image rows); the committed `data/bounce_v2_episode_00000.png` was drawn y-up, so it is
  vertically mirrored relative to the frames and the v3 figures.
- Pull W&B history: `wandb.Api().run('jesse-hoekstra-university-of-oxford/sparse-causal-jepa/<id>').scan_history(...)`
  (credentials in ~/.netrc). Runs execute on the NFS server — make sure it has the current
  commit; `git_sha` is recorded in each run's config.
- Tests: `.venv/bin/python -m pytest tests/ -q` must stay green.
  `tests/test_rollout_t2.py` pins independent valid sampling, shared θ̂, generated-state
  feedback, two-call gradients, mean scaling, and the `lambda_rollout_t2=0` TF-only path.
  `tests/test_observational_equivalence.py` pins the no-gradient K=30 diagnostic and its
  tolerance response. `tests/test_protocol_provenance.py` rejects obsolete curriculum keys.
