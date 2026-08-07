# SCJEPA — "Causal Identification within JEPA Using a SPARTAN"

PyTorch research codebase for Jesse's paper (`sources/my_paper.pdf`). SPARTAN predictor
(`sources/SPARTAN.pdf`, no public code) inside a JEPA, with the bounce identifiability
experiment replicated from Baumgartner et al. (`sources/dynamical_system.pdf`).
Settled design decisions live in `docs/decisions.md` (D1–D36) and BIND all work.
Subagent roster and shared conventions: `.claude/agents/README.md`.

**STATUS: Experiment 1's mechanisms are confirmed, but the OBJECTIVE and its optimisation
protocol changed in D34/D36.**
Under the pure teacher-forced objective, sparsification pruned the graph AND recovered the
masses (D30, run 7cq3h2ur): `mcc` 0.948, `shd` 4.81, `path_density` 0.223, L_path 12.3, zero
skipped steps in 300k, at τ=0.02 (a TEST value, ~1.5–2× dense, not the §6.1.3 τ=1.0×dense).

**D34 replaced that objective with the HYBRID one: teacher forcing PLUS a full-window K=30
autoregressive rollout, with the rollout inside the dual constraint. D36 keeps that final
objective but supersedes D35's prefix-horizon route with a fixed debugging curriculum:** TF-only
to 10k accepted updates; three true-anchored windows at offsets `[0,10,20]` with H=2/5/10 to
20k/30k/50k; then one continuous forward K=30 rollout with backward cuts `{10,20}`, `{15}`,
`{20}`, `{25}` to 70k/85k/100k/115k; then no cuts and exact full BPTT through the 355k-attempt
budget. Every
stage teacher-forces all 30 suffix transitions, uses λ_roll=1 whenever a rollout exists, and
reuses one unchanged attached θ̂ per episode across every step/window. The path penalty and
GECO stay inactive until the no-cut stage at 115k, leaving 240k exact terminal updates when the
healthy run has zero skips (always report the actual accepted count otherwise).
Consequences that bite immediately: τ is on a NEW SCALE and every pre-D34 τ (0.02 included)
is void; D30's four-phase trajectory was measured pre-hybrid so its NUMBERS are not comparable
(its structure should be); and no full D36 hybrid run exists yet. D36 is intentionally the
current simple debugging protocol and may be iterated only through another recorded decision.

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
phases are a property of the GECO dual and should survive the hybrid change; its NUMBERS should
not be expected to (c now carries λ_roll·L_roll, so τ, `sparsity/constraint` and the λ
trajectory are all on a new scale). Compare a new run against the SHAPE below, not the values —
the metrics are supposed to move a long way, and most "failures" reported before D30 were
phase 1 mistaken for death. These ranges are historical D30 coordinates. Under D36 the dual is
frozen until the first no-cut full-BPTT batch at the 115k boundary, so none of these GECO phases
can begin before then; do not mechanically add 115k because the dynamics-pretrained starting
point is also different.
1. **Saturation (~0–75k):** c > τ, so `sparsity/lambda` CLIMBS (1e4 → 4.5e7), `path_density`
   → 1.0, `loss/sparsity` rises to ~2400, `shd` parks at ~42, `mcc` loiters 0.25–0.30.
   This is correct and looks exactly like the D26 never-prunes failure. Do not kill a run here.
2. **Reversal (~75–90k):** `sparsity/constraint` drops under τ; λ peaks and rolls over. Watch
   the CONSTRAINT, not `loss/pred` — under D34 the bound is the scalarised sum, so `loss/pred`
   can sit well below τ while c is still above it.
3. **Pruning cascade (~90–155k):** λ falls ~3.5 orders, `path_density` 1.0 → 0.27,
   `loss/sparsity` 2400 → 25, `shd` 42 → 5, and **`mcc` climbs 0.46 → 0.935 inside this same
   window** — identification is concurrent with pruning, which is the experiment's whole claim.
4. **Equilibrium (~155k+):** λ settles into 40–5000 (2272 here), `eval/constraint_loss` pinned
   AT the τ boundary (0.015–0.021 vs τ=0.02), `attention/mean_abs_logit` GROWING (3.1 → 3.9)
   and `attention/gate_entropy` FALLING (0.25 → 0.19) = gates committing, not sitting in the
   p≈0.5 limbo that killed D26.

Point checks that still apply: `health/grad_norm` must be interpreted against the logged D36
phase, including window count/horizon or gradient-cut set; 7cq3h2ur (pure TF) ran max 1.42 /
mean 0.47, while fixed-horizon 1500-step smokes reached 4.02 at K=10 and 4.43 at K=30.
`health/skipped_steps` should remain zero, and rejected batches must leave
`schedule/successful_updates` and the phase unchanged. `sparsity/active` must remain 0 through
the cut-at-25 phase and become 1 only when the no-cut stage begins at 115k accepted updates.
Before 50k the live multi-window constraint is not the final K=30 constraint; from 50k onward
the numerical forward rollout is already the complete K=30 trajectory, although its training
gradient stays truncated until 115k. Once the terminal no-cut stage is active,
`eval/path_density` should be strictly between 1/T and 1. Train `loss/pred` vs
`eval/pred_loss` gap should stay below ~3x, and at 100k clips eval should be BETTER than train
(0.0142 vs 0.019); a large gap the other way = memorization (run jkslgj4o 2026-07-19: 14x gap
at num_clips=4000 — rescale `data.num_clips` whenever capacity or steps change).

Failure catalog — THREE LIVE (the retired rollout/regularizer entries were pruned 2026-07-27;
they were pure-rollout or VISReg-era runs that predate the current numerics and objective, see
D34 and git history if ever needed):
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
3. **Full-horizon BPTT explosion** (run ecbjobkj, 2026-08-02): the 10k λ_roll ramp did not
   fix fixed-K=30 training. Teacher-forced gradients remained ordinary while rollout gradients
   reached 1e7–1e8 and the guard aborted after 2000 consecutive skips. D36 keeps λ_roll=1,
   first covers the three trajectory regions with short true-anchored windows, then uses one
   uninterrupted K=30 forward rollout while cuts limit recurrent backward depth to
   10/15/20/25 before exact full BPTT. A cut is forward identity, never a ground-truth reset.

The D18 grad-spike guard stays in the loop regardless: a non-finite or absurd pre-clip grad norm
rejects the whole update (optimizer AND dual), and persistent skipping raises. Its origin was a
finite ~1e30 loss that passed `isfinite`, overflowed the backward pass to grad_norm = inf, and
let clipping multiply every gradient by zero — a run that "finished" 230k frozen steps with
byte-identical eval rows. That signature is worth recognising now that BPTT exists again.

## Key mechanics (Experiment 1, hybrid objective — D34/D36)
- **The objective has TWO prediction branches sharing one θ̂** (pooled exactly once per episode
  from observations 0..29). The same attached θ̂ tensor is reused unchanged at every predictor
  call in every window/chunk; never re-encode or detach it within an episode. Hybrid write-up
  Eq. 36, dual form:

      L = L_TF + λ_roll·L_roll + λ_logit·L_logit + λ⁻¹·L_path
      c = L_TF + λ_roll·L_roll + λ_logit·L_logit  ≤  τ      (path penalty excluded)

  `L_TF` (Eq. 32/39) = 30 teacher-forced one-step predictions, each anchored at the TRUE Z_t,
  t ∈ I_TF = {29,…,58}. The terminal `L_roll` (Eq. 35) = ONE autoregressive chain anchored at
  true Z_29 and rolled to Z_59 (K=30), every prefix supervised. w_1 = 0 always — at k=1 the
  chain recomputes the TF term at t=29 — and remaining weights are uniform and normalised so
  K⁻¹Σw_k=1. The three-window preparation loss is averaged across windows rather than summed,
  preserving this rollout scale and λ_roll=1. K=30 fits the existing 60-step clips exactly:
  **no preload change.** No VISReg, Hungarian matching, or variance normalization in Exp 1.
- **D36's accepted-update schedule is the current debugging protocol:**

  | Accepted updates | Rollout training branch |
  |---:|---|
  | `[0,10k)` | off; TF only |
  | `[10k,20k)` | three true-anchored windows, offsets `[0,10,20]`, H=2 |
  | `[20k,30k)` | same three windows, H=5 |
  | `[30k,50k)` | same three windows, H=10 |
  | `[50k,70k)` | continuous forward K=30; cuts `{10,20}` |
  | `[70k,85k)` | continuous forward K=30; cut `{15}` |
  | `[85k,100k)` | continuous forward K=30; cut `{20}` |
  | `[100k,115k)` | continuous forward K=30; cut `{25}` |
  | `[115k,∞)` | continuous K=30; no cuts, exact full BPTT |

  Starts are relative to Z_29, so offsets `[0,10,20]` are true anchors
  `[Z_29,Z_39,Z_49]`. TF still covers all 30 transitions in every phase. A successful update
  means `optimizer.step()` ran; rejected batches do not advance the phase. Save and resume the
  accepted counter at every boundary. `train.steps=355k` is an attempted-batch stop; it yields
  the intended 240k terminal accepted updates only in the zero-skip healthy case. This fixed
  schedule is expected to be debugged and may be
  replaced only by a new recorded decision, not ad hoc run-specific changes.
- **Cuts truncate only backward.** From 50k onward there is always one numerical trajectory
  `Z_29 -> Zhat_30 -> ... -> Zhat_59`; at a boundary, `state.detach()` is passed forward, with
  the identical predicted value and no true-state reset. Its backward derivative across that
  boundary is zero. Thus cuts `{10,20}` make three maximum-depth-10 backward paths, followed by
  maximum depths 15, 20, 25, then 30 when all cuts are removed. Predictor weights are shared
  and the common θ̂ remains attached across every chunk.
- **Both prediction terms AND the logit term sit inside the bound, and that is Baumgartner, not
  a liberty.** Their Eq. 9 (p7) is `min L_path s.t. L_rec + L_KL + L_logit ≤ L*` (L_KL absent
  for us: no cVAE), and their `L_rec` reconstructs Eq. 2's autoregressive trajectory, so a
  rollout under the bound moves TOWARD their formulation. §4.3's "bounds" (plural) are
  scalarised into one, keeping a single dual variable and a single τ.
- **The eval harness must mirror the relevant constraint exactly.** Final/post-hoc evaluation
  and τ calibration use one uninterrupted forward K=30 rollout and λ_roll=1 from the run's
  config; detach cannot change that scalar forward value. Pre-50k multi-window losses are stage
  diagnostics, not the final constraint. Exactly 115k accepted updates is only the last cut-phase
  boundary and contains zero no-cut updates; reportable τ calibration requires
  `successful_updates > 115000`; a zero-skip production run contains 240k no-cut updates, and
  provenance reports the actual number otherwise.
  Changing the terminal horizon or coefficient on one side silently invalidates τ.
- Model: `ParameterEncoder` (Eqs. 16–26: relational attention per timestep FIRST, then
  per-track temporal pooling, then a bare scalar head) + `Spartan` (Eqs. 27–37) with one
  FIXED non-trainable track key κ_i shared by state and parameter token i (buffer,
  seed-0 codebook, scale 0.02 = our choice). Dense A≡1 / token-local A≡0 references share
  the same modules and codebook. Verified endpoints: dense L_path = 6655, token-local = 5.
- Dual: log λ += α·MA[c − τ], UNCLAMPED both directions (§6.1.3), but under D36 both
  this update and the `λ⁻¹·L_path` term are inactive until the no-cut full-BPTT stage at
  115k accepted updates. The controller reaches that terminal stage with initial λ and moving
  average untouched. α=2e-2 is the one free numerical knob. λ₀=1e6 is the write-up's value
  and code default; D30 used 1e4. Pre-D29 runs are NOT comparable.
- **Working configuration: λ_logit = 1e-5, λ_roll = 1.0, terminal K = 30, accepted-update
  D36 curriculum `TF -> 3x(H2/H5/H10) -> continuous K30 cuts(10/15/20/25) -> full BPTT`,
  355k attempted batches (zero-skip target), λ₀ = 1e4, α = 2e-2.** τ is NOT
  a number you carry over — **D30's τ=0.02 is void under the hybrid constraint** (c now includes
  λ_roll·L_roll, which ran ~1.5× L_TF in the smoke). Let the pipeline calibrate it.
- Protocol τ = 1.0 × the held-out `constraint_loss` of ONE converged full-length dense run per
  architecture/seed, computed with the IDENTICAL rollout settings (never a short reference — v2
  failure; never the gated model with sparsity off — F-8). The `--tau-max` token-local launch
  gate was DELETED in D34: its floor (0.043645 + λ_logit·2.0, run ku244l5e) was measured under
  the pure-TF constraint and is a different quantity now.
- λ_logit comes from the label-free dense sweep (§6.1.3 rule, grid recentered to
  0…1e-3 low end): feasibility arithmetic (2026-07-25) says values above ~3e-5 make the
  gated constraint set EMPTY at raw scale — gate commitment costs λ_logit·(2cosh|l|−2)
  inside a τ that has zero slack. 1e-5 is confirmed to work; |logit| should GROW past ~3
  (7cq3h2ur ended at 3.87), and gates that stay near 0.3 mean the run never committed.
  That arithmetic was done against the PURE-TF constraint. The hybrid τ is a larger absolute
  number (it absorbs λ_roll·L_roll), so the ~3e-5 ceiling is if anything conservative now —
  but it has not been recomputed, so treat it as an untested bound rather than a measured one.
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
- **Reproduce the PRE-D34 D30 result** (~1.6 h on one GPU, 300k steps) — needs the rollout
  branch switched off, since the objective changed underneath it:
  `bash scripts/isambard_main_only.sbatch <tag> 0.02 1e-5 300000 <seed> train.sparsity_lambda_init=1e4 train.rollout_len=null train.rollout_curriculum=null`
  (`bash`, not `sbatch`, when already inside an `srun --pty` allocation). WITHOUT
  both rollout overrides this trains the hybrid objective at a τ calibrated for the old one.
- Full pipeline: `bash scripts/run_bounce_example.sh --run-tag=X ...` (dense A≡1 same
  length as main → τ = 1.0× its held-out constraint_loss → sparse run → 5000-episode eval;
  NO identity stage since D29, NO `--tau-max` gate since D34; `--calib-steps` only for smokes;
  other hydra overrides go to BOTH runs, D12).
- Cheap stability smoke (~3–5 min, CPU): 1500–3000 steps via `Trainer` directly with
  `data.num_clips=200` at paper geometry (clip_len 60, context_len 30). Under D36 this tests only
  the TF stage; to test rollout mechanics cheaply, explicitly select the desired window/cut
  phase rather than pretending a 1500-step run traversed the schedule. Healthy means zero
  skipped steps; compare gradient norms only at the same forward horizon and backward-cut depth.
- **NEVER regenerate `data/bounce_train_v2_100000.pt`.** The simulator is deterministic on a
  given machine but NOT across machines: regenerating the identical config on the Mac vs the
  server diverges chaotically (measured n=200: 194/200 episodes differ, max |delta| 3.4, and
  **26% get a different contact graph** — i.e. a different ground-truth causal graph). The file
  is the physics of record for D30 and for Experiment 2. Visual runs render frames FROM it.
  Nuance if T ever has to grow (D34 does NOT need it — K=30 fits 60-step clips exactly):
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
- Tests: `.venv/bin/python -m pytest tests/ -q` (135 tests, ~15 s, must stay green).
  `tests/test_rollout.py` pins the hybrid branch: that it is genuinely autoregressive (a
  degenerate rollout that quietly became K teacher-forced steps would still train and still
  look healthy), Eq. 35's normalisation, w_1 = 0, and train/eval computing the same targets.
