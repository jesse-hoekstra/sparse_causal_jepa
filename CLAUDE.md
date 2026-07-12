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
`sparsity/lambda` responsive in BOTH directions (settling ~40–5000 is fine) ·
`health/target_slot_std_min` ≥ ~0.1 · `eval/path_density` strictly between 1/T and 1 and still
moving after 5k steps · `eval/constraint_loss` hovering near τ (the dual holds it AT the
boundary; far below τ = over-pruned, far above = under-pruned).
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

## Key mechanics (verified against papers 2026-07-11/12)
- The training objective is an AUTOREGRESSIVE ROLLOUT (D16, 2026-07-12): chains feed their own
  predictions back with one shared Ŝ^ph (my_paper p7/p16; Baumgartner §3.1/B.4).
  `train.rollout_horizon` = chain length (must divide K; None = one chain; 1 = the old
  teacher-forced D15 behavior, kept as ablation). Pre-D16 runs/metrics are NOT comparable.
- τ reference = `model.spartan_dense=true` (A≡1, SPARTAN's "fully connected model") trained to
  the SAME length as the main run; factor ~1.1. Never calibrate on the gated model with
  sparsity off (F-8) or on a short run (v2 failure).
- v3 go/no-go: the converged dense reference must beat a mass-blind model's loss — if it
  doesn't, no τ can force param edges (see D16 "Open" note; watch eval/shd_param early).
- Constraint the dual compares to τ is `pred + lambda_logit·logit_penalty` (Baumgartner Eq. 9);
  the eval harness reports it as `constraint_loss` — calibrate τ on THAT, never bare pred_loss.
  Watch the logit share: at equilibrium it consumes constraint budget (v2: 0.12 of τ=0.17).
- Gate/penalty logits are the SCALED q·k/√d (interpretation — papers write unscaled q·k, which
  is untrainable at init; flagged in `src/scjepa/models/spartan.py`).
- Path matrix entries are path COUNTS (∏(A_l+I)); `path_density` = fraction of entries ≥ 0.5;
  identity-only matrix ⇒ density = 1/T (T=10 for 5-ball states regime ⇒ 0.100 exactly).
- `shd_param` = 1.4595 constant ⇔ zero learned param edges (both failed runs); any real
  param-edge learning moves it.
- MCC here = Baumgartner F.1 nonlinear MLP-R² (`eval/parameters.py`), reference SPARTAN bounce
  MCC ~0.9 (their Fig. 3), MCC ramps late in training (their Fig. 17) — flat-low before ~100k
  steps is normal ONLY if density/shd are still moving.

## Commands
- Full pipeline: `bash scripts/run_bounce_example.sh --run-tag=X --main-steps=M
  experiment=bounce_baumgartner ...` (calibration = dense A≡1, same length as main, τ = 1.1×
  its constraint_loss by default; `--calib-steps` only for smokes; other hydra overrides go to
  BOTH runs, D12).
- Cheap stability smoke (~3 min, CPU): 1500–3000 steps via `Trainer` directly with
  `data.num_clips=200` — see Claude memory for the pattern; healthy = grad_norm < 1 throughout.
- Pull W&B history: `wandb.Api().run('jesse-hoekstra-university-of-oxford/sparse-causal-jepa/<id>').scan_history(...)`
  (credentials in ~/.netrc). Runs execute on the NFS server — make sure it has the current
  commit; `git_sha` is recorded in each run's config.
- Tests: `.venv/bin/python -m pytest tests/ -q` (70 tests, ~1 min, must stay green).
