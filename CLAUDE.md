# SCJEPA — "Causal Identification within JEPA Using a SPARTAN"

PyTorch research codebase for Jesse's paper (`sources/my_paper.pdf`). SPARTAN predictor
(`sources/SPARTAN.pdf`, no public code) inside a JEPA, with the bounce identifiability
experiment replicated from Baumgartner et al. (`sources/dynamical_system.pdf`).
Settled design decisions live in `docs/decisions.md` (D1–D15) and BIND all work.
Subagent roster and shared conventions: `.claude/agents/README.md`.

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
Healthy: `loss/logit` ≈ 0.003–0.12 and smooth · `health/grad_norm` mostly < 1 ·
`health/skipped_steps` flat or slowly accumulating isolated skips (D18 guard; a few per 1k
steps during the mid-density transit is expected post-D19 — consecutive-run growth is the
fatal signature) ·
`sparsity/lambda` responsive in BOTH directions (settling ~40–5000 is fine) ·
`health/target_slot_std_min` ≥ ~0.1 · `eval/path_density` strictly between 1/T and 1 and still
moving after 5k steps · `eval/constraint_loss` hovering near τ (the dual holds it AT the
boundary; far below τ = over-pruned, far above = under-pruned) ·
train `loss/pred` vs `eval/pred_loss` gap < ~3x AND eval still improving (train falling with
eval flat = memorization — run jkslgj4o 2026-07-19: 14x gap, 1200 views/episode at
num_clips=4000; τ calibrated held-out is then meaningless to the train-driven dual. Rescale
`data.num_clips` whenever model capacity or steps change).
Failure catalog (all observed, all diagnosed — don't re-derive):
1. **Logit-penalty explosion** (run n5zq9nct, 2026-07-11): loss/logit ≫ loss/pred, grad spikes
   1e5–1e15, lambda railed at 1e6 → fixed by 1/√d on gate logits + masked-softmax numerics +
   constraint_loss calibration (commit 55b5282; details in Claude memory
   `project-spartan-logit-stability`).
2. **Empty-graph collapse** (run qqye6ug1, 2026-07-12): graph pruned to identity (density =
   1/T exactly, |Ā| ≈ T) by step ~2k, param→state edges dead, MCC = the eval's noise floor,
   recovery grid = identical blobs across slots. Full diagnosis in `docs/audits/2026-07-12-*.md`;
   verified mechanism, in order: (a) the objective itself — teacher-forced single-step D15
   prediction values edges at only ~7% of MSE (forced-FC 0.0596 vs forced-identity 0.0639 at
   equal budget), so the empty graph satisfies ANY realistic τ; the paper (my_paper p7/p16) and
   Baumgartner both prescribe autoregressive rollouts where mass-blindness is unsatisfiable;
   (b) F-8: no true dense reference exists — τ was calibrated on the gated stochastic model,
   undertrained (6k steps), ×2.0; (c) F-9: the logit term INSIDE the constraint converts τ slack
   into gate-closure depth (loss/logit ≈ τ − pred ≈ 0.12 observed; gate logits ≈ −4.8, reopen
   prob 0.8%) — slack doesn't just permit the empty graph, it entrenches it; (d) F-10: dual
   step 0.02 crosses its range in ~2k steps vs the papers' 10⁵–10⁶-step λ trajectories.
3. **Scale collapse via weak lambda_reg** (pre-2026-07-10, see bounce_states.yaml comment):
   target embeddings shrink to satisfy the constraint; VISReg at lambda_reg=1.0 is the anchor.
4. **Zombie freeze via finite blow-up** (run 7wupt6pw, 2026-07-17): grad-spike episode →
   predictor per-step gain > 1 → Tp=30 rollout amplifies to FINITE ~1e30 MSE (passes the
   isfinite guard) → BPTT overflows, grad_norm = inf → clip_grad_norm_ multiplies all grads by
   max_norm/inf = 0 → weights frozen, run "finishes" (47 byte-identical evals from 70k–300k).
   Fixed by D18 (skip guard + consecutive-skip raise + rolling checkpoints). Signature:
   `health/grad_norm` = inf, byte-identical eval rows, `health/skipped_steps` climbing.
5. **Mid-density gradient detonation** (runs 0ta5ymcw/u94wqvcb + local repro, 2026-07-17): at
   train path_density ~0.55–0.7, grads 1e4–1e6 on ~every batch (u94wqvcb: 100/100 skipped for
   1600 steps). Cause: pre-D19, each of the Tp=30 chain steps redrew fresh Bernoulli gate noise
   per layer — 60 i.i.d. hard-mask resamplings in ONE backward graph; the randomly rewired
   step-Jacobian product is heavy-tailed. Forward loss stays healthy; only backward explodes.
   Fixed by D19 (per-chain gate thresholds). Post-D19 residual: rare isolated spike batches
   (~3% during the mid-density transit, absorbed by the D18 skip guard) are EXPECTED; the fatal
   signatures are consecutive-run skip growth or every-batch skipping (then: Gumbel temperature
   is the next lever, not the skip limit).

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
- Dual: log λ += α·MA[c − τ], λ₀ = 1e6, UNCLAMPED both directions (§6.1.3). α = 2e-2 is the
  one free numerical knob. Pre-D29 runs/metrics are NOT comparable to new ones.
- τ = 1.0 × the held-out `constraint_loss` of ONE converged full-length dense run per
  architecture/seed (never a short reference — v2 failure; never the gated model with
  sparsity off — F-8). Launch gate: τ must sit below the token-local floor
  0.043645 + λ_logit·2.0 (reused measurement ku244l5e — the A≡0 constraint is
  parameter-encoder-invariant; the 8-seed confirmatory protocol retrains it).
- λ_logit comes from the label-free dense sweep (§6.1.3 rule, grid recentered to
  0…1e-3 low end): feasibility arithmetic (2026-07-25) says values above ~3e-5 make the
  gated constraint set EMPTY at raw scale — gate commitment costs λ_logit·(2cosh|l|−2)
  inside a τ that has zero slack. Watch the gated run's |logit| leaving ~0.3 early.
- Gate/penalty logits are the SCALED q·k/√D_sp — now codified in the write-up (Eq. 31).
- Path matrix entries are path COUNTS (∏(A_l+I)); `path_density` = fraction of entries ≥ 0.5;
  identity-only matrix ⇒ density = 1/T (T=10 for 5-ball states regime ⇒ 0.100 exactly).
- **There is exactly ONE graph metric (D28, 2026-07-25): `eval/shd`** — SPARTAN's SHD (their
  Table 1; Baumgartner never use it), lower is better, between the LEARNED graph and the
  GROUND-TRUTH causal graph: decoded state rows x all 2N source tokens [state | params],
  range [0, 50] for 5 balls, same index set as `path_density`/the path objective.
  **NEVER read it alone** — the true graph has only 7.86 of 50 edges, so (verified) the EMPTY
  graph scores 2.86 and the SATURATED one 42.14: lower-is-better favours the model that learned
  nothing by ~15x. Reference points: perfect 0 · empty/token-local 2.86 (mcc 0) · dense 42.14
  (mcc 0.948). Success = `shd` falling toward 0 WITH `mcc` high; `shd`≈2.9 with `mcc`≈0 is
  empty-graph collapse (failure #2), i.e. the second-best SHD is a total failure. A frozen
  `shd` is not a model property while the graph is saturated or empty (it is then just the GT
  edge count). Superseded keys, NOT comparable: `shd_state`, `shd_param`, `shd_param_aligned`.
- **There is exactly ONE mass-recovery metric (D27, 2026-07-25): `eval/mcc`** = Baumgartner
  App. F.1 `mean_i max_j R²_ij` (`eval/parameters.py:nonlinear_mcc`), so it is directly
  comparable to their bounce reference ~0.9 (Fig. 3). It ramps late in training (their
  Fig. 17) — flat-low before ~100k steps is normal ONLY if density/shd are still moving.
  Do NOT add a second recovery score. Runs before 2026-07-25 logged `eval/mass_mcc`
  (a stricter Hungarian one-to-one score) and `eval/shd_param_aligned` — different
  quantities, do not plot them on the same axis as the current keys.

## Commands
- Full pipeline: `bash scripts/run_bounce_example.sh --run-tag=X ...` (dense A≡1 same
  length as main → τ = 1.0× its held-out constraint_loss → sparse run → 5000-episode eval;
  NO identity stage since D29 — pass `--tau-max` = the token-local floor instead;
  `--calib-steps` only for smokes; other hydra overrides go to BOTH runs, D12).
- Cheap stability smoke (~3 min, CPU): 1500–3000 steps via `Trainer` directly with
  `data.num_clips=200` — see Claude memory for the pattern; healthy = grad_norm < 1 throughout.
- Pull W&B history: `wandb.Api().run('jesse-hoekstra-university-of-oxford/sparse-causal-jepa/<id>').scan_history(...)`
  (credentials in ~/.netrc). Runs execute on the NFS server — make sure it has the current
  commit; `git_sha` is recorded in each run's config.
- Tests: `.venv/bin/python -m pytest tests/ -q` (70 tests, ~1 min, must stay green).
