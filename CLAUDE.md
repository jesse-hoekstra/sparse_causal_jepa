# SCJEPA — "Causal Identification within JEPA Using a SPARTAN"

PyTorch research codebase for Jesse's paper (`sources/my_paper.pdf`). SPARTAN predictor
(`sources/SPARTAN.pdf`, no public code) inside a JEPA, with the bounce identifiability
experiment replicated from Baumgartner et al. (`sources/dynamical_system.pdf`).
Settled design decisions live in `docs/decisions.md` (D1–D30) and BIND all work.
Subagent roster and shared conventions: `.claude/agents/README.md`.

**STATUS: Experiment 1 works — the mechanisms are confirmed (D30, 2026-07-25, run 7cq3h2ur).**
Sparsification prunes the graph AND recovers the masses: `mcc` 0.948, `shd` 4.81,
`path_density` 0.223, L_path 12.3, zero skipped steps in 300k. τ=0.02 was a TEST value, not the
§6.1.3 τ=1.0×dense — fine here, because Experiment 1 is a mechanism check, not the main
experiment. D30 has the reference trajectory. Current work moves to Experiment 2.

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
**The reference is now MEASURED, not inferred: run `7cq3h2ur` (D30).** Compare any new sparse
run against its four phases rather than against a static checklist — the metrics are supposed to
move a long way, and most "failures" reported before D30 were phase 1 mistaken for death.
1. **Saturation (~0–75k):** c > τ, so `sparsity/lambda` CLIMBS (1e4 → 4.5e7), `path_density`
   → 1.0, `loss/sparsity` rises to ~2400, `shd` parks at ~42, `mcc` loiters 0.25–0.30.
   This is correct and looks exactly like the D26 never-prunes failure. Do not kill a run here.
2. **Reversal (~75–90k):** `loss/pred` drops under τ; λ peaks and rolls over.
3. **Pruning cascade (~90–155k):** λ falls ~3.5 orders, `path_density` 1.0 → 0.27,
   `loss/sparsity` 2400 → 25, `shd` 42 → 5, and **`mcc` climbs 0.46 → 0.935 inside this same
   window** — identification is concurrent with pruning, which is the experiment's whole claim.
4. **Equilibrium (~155k+):** λ settles into 40–5000 (2272 here), `eval/constraint_loss` pinned
   AT the τ boundary (0.015–0.021 vs τ=0.02), `attention/mean_abs_logit` GROWING (3.1 → 3.9)
   and `attention/gate_entropy` FALLING (0.25 → 0.19) = gates committing, not sitting in the
   p≈0.5 limbo that killed D26.

Point checks that still apply: `health/grad_norm` mostly < 1 (7cq3h2ur: max 1.42, mean 0.47) ·
`health/skipped_steps` should now be **0** — post-D29 there is no BPTT, and the mid-density
transit was crossed without a single skip, so ANY sustained skipping is a new bug, not the
expected residual it was pre-D29 · `eval/path_density` strictly between 1/T and 1 · train
`loss/pred` vs `eval/pred_loss` gap < ~3x, and at 100k clips eval should be BETTER than train
(0.0142 vs 0.019); a large gap the other way = memorization (run jkslgj4o 2026-07-19: 14x gap
at num_clips=4000 — rescale `data.num_clips` whenever capacity or steps change).

Failure catalog. **#1 and #2 are LIVE. #3–#5 are structurally unreachable in Experiment 1**
(no regularizer, no rollout, no gate-noise chaining post-D29) — kept as history, and re-check
them if a later experiment reintroduces a regularizer or a moving target:
1. **Logit-penalty explosion** (LIVE; run n5zq9nct, 2026-07-11): loss/logit ≫ loss/pred, grad
   spikes 1e5–1e15, lambda railed at 1e6 → fixed by 1/√d on gate logits + masked-softmax
   numerics + constraint_loss calibration (commit 55b5282; Claude memory
   `project-spartan-logit-stability`).
2. **Empty-graph collapse** (LIVE; run qqye6ug1, 2026-07-12): graph pruned to identity
   (density = 1/T exactly), param→state edges dead, `mcc` at the noise floor, `shd` ≈ 2.7 —
   i.e. the SECOND-BEST SHD is total failure. Full diagnosis in `docs/audits/2026-07-12-*.md`.
   Verified drivers: τ slack converted into gate-closure depth by the logit term INSIDE the
   constraint (F-9), τ calibrated on the gated model rather than a true dense one (F-8), and a
   dual step crossing its range in ~2k steps (F-10). D30 shows the cure in practice: a τ the
   dense model actually beats, λ_logit small enough to leave commitment budget, and patience
   through phase 1.
3. *(retired for Exp 1)* **Scale collapse via weak lambda_reg** — target embeddings shrink to
   satisfy the constraint; VISReg at lambda_reg=1.0 was the anchor.
4. *(retired for Exp 1)* **Zombie freeze via finite blow-up** (run 7wupt6pw, 2026-07-17):
   Tp=30 rollout amplified a grad spike to a FINITE ~1e30 MSE that passed the isfinite guard,
   BPTT overflowed, grad_norm = inf, clip multiplied all grads by max_norm/inf = 0 → weights
   frozen while the run "finished". Signature: byte-identical eval rows.
5. *(retired for Exp 1)* **Mid-density gradient detonation** (runs 0ta5ymcw/u94wqvcb,
   2026-07-17): 60 i.i.d. hard-mask resamplings in one backward graph made the step-Jacobian
   product heavy-tailed at density 0.55–0.7. Fixed by D19, then deleted with the rollout.

## Key mechanics (Experiment 1 == experiments.pdf §6.1–6.2 exactly, D29 2026-07-25)
- The training objective is 30 TEACHER-FORCED ONE-STEP predictions per episode (Eq. 7/§6.2):
  θ̂ pooled once from observations 0..29, every prediction anchored at the TRUE Z_t. There is
  no rollout, no rollout_horizon knob, no gate-noise chaining. Objective = Eq. 40
  (pred + λ_logit·logit + λ⁻¹·path); dual constraint = Eq. 13 (pred + λ_logit·logit, RAW
  units) — no VISReg, no Hungarian matching, no variance normalization in Experiment 1.
- Model: `ParameterEncoder` (Eqs. 16–26: relational attention per timestep FIRST, then
  per-track temporal pooling, then a bare scalar head) + `Spartan` (Eqs. 27–37) with one
  FIXED non-trainable track key κ_i shared by state and parameter token i (buffer,
  seed-0 codebook, scale 0.02 = our choice). Dense A≡1 / token-local A≡0 references share
  the same modules and codebook. Verified endpoints: dense L_path = 6655, token-local = 5.
- Dual: log λ += α·MA[c − τ], UNCLAMPED both directions (§6.1.3). α = 2e-2 is the one free
  numerical knob. λ₀ = 1e6 is the write-up's value and the code default; D30 used 1e4, which is
  safe precisely BECAUSE the dual is unclamped — λ₀ only sets how far phase 1 has to climb
  before the constraint binds (it rose to 4.5e7 anyway). Pre-D29 runs are NOT comparable.
- **Working configuration (D30, verified): τ = 0.02, λ_logit = 1e-5, λ₀ = 1e4, α = 2e-2.**
  Start here. τ=0.02 is a test value (~1.5–2× the dense constraint, so the run had slack), not
  a protocol number — any reportable τ must carry its provenance.
- Protocol τ (for the confirmatory table) = 1.0 × the held-out `constraint_loss` of ONE
  converged full-length dense run per architecture/seed (never a short reference — v2 failure;
  never the gated model with sparsity off — F-8). Launch gate: τ must sit below the token-local
  floor 0.043645 + λ_logit·2.0 (reused measurement ku244l5e — the A≡0 constraint is
  parameter-encoder-invariant; the 8-seed confirmatory protocol retrains it).
- λ_logit comes from the label-free dense sweep (§6.1.3 rule, grid recentered to
  0…1e-3 low end): feasibility arithmetic (2026-07-25) says values above ~3e-5 make the
  gated constraint set EMPTY at raw scale — gate commitment costs λ_logit·(2cosh|l|−2)
  inside a τ that has zero slack. 1e-5 is confirmed to work; |logit| should GROW past ~3
  (7cq3h2ur ended at 3.87), and gates that stay near 0.3 mean the run never committed.
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
- **Reproduce the D30 result** (~1.6 h on one GPU, 300k steps):
  `bash scripts/isambard_main_only.sbatch <tag> 0.02 1e-5 300000 <seed> train.sparsity_lambda_init=1e4`
  (`bash`, not `sbatch`, when already inside an `srun --pty` allocation).
- Full pipeline: `bash scripts/run_bounce_example.sh --run-tag=X ...` (dense A≡1 same
  length as main → τ = 1.0× its held-out constraint_loss → sparse run → 5000-episode eval;
  NO identity stage since D29 — pass `--tau-max` = the token-local floor instead;
  `--calib-steps` only for smokes; other hydra overrides go to BOTH runs, D12).
- Cheap stability smoke (~3 min, CPU): 1500–3000 steps via `Trainer` directly with
  `data.num_clips=200` — see Claude memory for the pattern; healthy = grad_norm < 1 throughout.
- **NEVER regenerate `data/bounce_train_v2_100000.pt`.** The simulator is deterministic on a
  given machine but NOT across machines: regenerating the identical config on the Mac vs the
  server diverges chaotically (measured n=200: 194/200 episodes differ, max |delta| 3.4, and
  **26% get a different contact graph** — i.e. a different ground-truth causal graph). The file
  is the physics of record for D30 and for Experiment 2. Visual runs render frames FROM it.
- Plot an episode: `python scripts/plot_bounce_episode.py --condition states|visual [--show-collision-radii]`
  (`visual` shows the real 64x64 encoder input). Note the renderer puts y DOWN (`grid_y` indexes
  image rows); the committed `data/bounce_v2_episode_00000.png` was drawn y-up, so it is
  vertically mirrored relative to the frames and the v3 figures.
- Pull W&B history: `wandb.Api().run('jesse-hoekstra-university-of-oxford/sparse-causal-jepa/<id>').scan_history(...)`
  (credentials in ~/.netrc). Runs execute on the NFS server — make sure it has the current
  commit; `git_sha` is recorded in each run's config.
- Tests: `.venv/bin/python -m pytest tests/ -q` (82 tests, ~1 min, must stay green).
