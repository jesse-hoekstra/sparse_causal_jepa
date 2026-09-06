# Architecture & engineering decisions

Living record for the codebase implementing **"Causal Identification within JEPA Using a
SPARTAN"** (`sources/SCJEPA.pdf`). The current protocol has two experiments: true states (Experiment 1) and learned visual
states with EMA targets (Experiment 2). D39 supersedes the retired three-experiment proposal.
Equation references to `outdated_experiments.pdf` identify historical component definitions;
the supplied manuscript and current implementation use the protocol described in D37–D39.

**2026-07-25: D1-D26 were condensed to the rules below.** Experiment 1 is finished (D30) and
the D29 refactor superseded most of the historical narrative, so the archaeology was removed
rather than carried forward. The full original text (984 lines) is recoverable with
`git show 3fbcdfd:docs/decisions.md`. D27-D30 are retained as the metric definitions and historical successful baseline. Later
entries state which objectives and launch settings supersede them.

## Condensed rules that still bind (from D1-D26)

- **D1 Framework.** PyTorch, not JAX. Reuse-first: the reference codebases (le-wm, visreg,
  SlotFormer, SAVi) are all PyTorch. SPARTAN has no public code and is implemented from the
  paper.
- **D5 Vendoring.** Adapted third-party code lives in `src/scjepa/third_party/<name>/` with the
  upstream license and a `PROVENANCE.md` (upstream URL, commit SHA, what changed). Recorded
  licenses: SlotFormer MIT, le-wm MIT, SAVi Apache 2.0, **visreg CC BY-NC 4.0 (non-commercial —
  flag before any commercial use)**. Vendored code is exempt from all lint/type gates. Prefer
  adapting vendored code over rewriting; prefer rewriting over depending on unmaintained
  packages at import time.
- **D8 Packaging & tooling.** Python 3.12 only, `src/` layout, package `scjepa`, ALL tool config
  in `pyproject.toml`. Gates: ruff (line length 100), pyright **strict**, pytest
  `--strict-markers`; `third_party/` excluded from all three.
- **D10 SPARTAN interpretations.** SPARTAN has no public code, so every place its text is
  ambiguous is an INTERPRETATION and must be flagged as such in a code comment. The two
  load-bearing ones: mask BEFORE softmax normalization (renormalize over unmasked entries), and
  scale gate logits by 1/sqrt(D) before both the Bernoulli sigmoid and the Eq. 11 penalty — the
  write-up now codifies the second as Eq. 31.
- **D11 Simulator contract.** Bounce is the guiding example. Every episode ships full ground
  truth — `frames`, kinematic `states`, causal `params` (masses), and the time-indexed local
  graph `contacts (T-1, N, N)` — because that is the entire reason for using synthetic systems.
- **D12 Pipeline consistency.** The pipeline stages may differ ONLY in graph mode; every other
  hydra override must be applied to BOTH, or the calibrated tau describes a different model than
  the one it constrains. A preload data file must be indistinguishable from on-the-fly
  generation (enforced in `tests/test_bounce.py`). Re-check config interactions whenever model
  capacity or step count changes — `data.num_clips` in particular (the views-per-episode
  arithmetic is what caught the 2026-07-19 memorization run).
- **D18 Grad-spike skip guard.** In `Trainer._train_step`, a batch whose PRE-clip grad norm is
  non-finite or above `train.grad_skip_threshold` (default 1e3) is rejected entirely — no
  optimizer step and no dual/EMA update, so a pathological batch cannot jolt the lambda
  controller. After `grad_skip_max_consecutive` (2000) consecutive skips the trainer RAISES: a
  dead run must die loudly rather than finish. Weights are frozen during skips and every retry
  is a fresh draw, so patience is free and the counter resets on any calm batch.
  `train.checkpoint_keep_every` (25000) keeps step-tagged fallbacks so a late failure is a
  resume, not a rerun. Watch `health/skipped_steps`: post-D29 it should be 0.

**Deleted as superseded** (full text in git): D2/D3 (SAVi choice, VISReg — no regularizer in
Experiment 1), D4/D14 (pooling variants), D6/D9 (Hungarian single-step loss, target branch),
D7 (from-scratch encoders), D13 (experiment ladder), D15/D16 (sliding-window and autoregressive
rollout objectives), D17 (variance-normalized constraint), D19 (per-chain gate noise), D20/D21
(gt-state ruler, teacher-forced one-step — now the write-up's own spec), D22/D23 (dual
schedules and the lambda clamp), D24/D25/D26 (parameter-slot architectures). All were replaced
by D29's refactor to the write-up's Experiment-1 specification.

## D27 — One mass-recovery metric: Baumgartner App. F.1 MCC (decided 2026-07-25, Jesse)

**Supersedes every earlier recovery metric.** Four similar numbers were in flight at once
(`mass_mcc`, `mean_max_correlation`, `nonlinear_mcc`, `mass_mcc_linear`), which made run
comparisons ambiguous. Only the reference paper's metric is retained. Pre-2026-07-25 runs
logged `eval/mass_mcc` — a different quantity, never plot it on the same axis as `eval/mcc`.

**The metric.** `sources/dynamical_system.pdf` App. F.1 p.39, verbatim: R² ∈ R^{I x J} is
built by fitting θ_i ≈ MLP_ij(θ̂_j) for every pair, and "The MCC metric is calculated as
MCC = 1/I Σ_i max_j(R²_ij)" — a MEAN OF MAXIMA over ground-truth rows, with NO bijection
constraint. Their probe protocol (p.40) is reproduced exactly: one-hidden-layer MLP, hidden
dim 32, 5,000 sampled points, 90/10 cross-validation split. Unspecified upstream and
therefore ours (interpretation): tanh, Adam lr 1e-2, 300 full-batch steps, negative R²
clamped to 0. One sample per episode (bounce: E x 5 learned vs E x 5 masses).

**Consequences.** (a) The score is permutation-INSENSITIVE and indifferent to which learned
coordinate carries a mass: one coordinate may be the argmax for several true masses, and the
argmax need not be the track-matched coordinate. It measures whether mass information exists
in θ̂, not where. Any binding claim needs a separate, explicitly named diagnostic — do not
smuggle one back into this metric. (b) It has no assignment output, so `align_parameter_columns`
is deleted and parameter-graph columns are compared in their natural tracked-object order
(the ζ = id contract for the true-state experiment). (c) The recovery grid's green
outline now marks each row's argmax — the cell that actually enters the sum — instead of a
frozen assignment; `recovery_alignment.json` is replaced by `mcc_matrix.json`, whose matrix
is stored in the paper's [true_mass][learned_coordinate] orientation (the old file used the
transpose). (d) W&B/metrics key is `eval/mcc`; `mass_mcc`, `mass_mcc_linear` and
`absolute_pearson` are gone.

## D28 — One graph metric: SPARTAN's SHD against the full ground-truth causal graph (decided 2026-07-25, Jesse)

**Provenance.** SHD is not a Baumgartner et al. metric (0 occurrences in dynamical_system.pdf).
It is SPARTAN's: Table 1 and §4.1 "Graph Learning" — "we evaluate the Structural Hamming
Distance, a commonly used metric in graph structure learning, between the learned graphs and
the ground-truth" — and App. D p.19 repeats the phrasing. Lower is better (their Table 1 and
Table 7 captions state it). SCJEPA.pdf p.13 and the write-up §6.7 both require a graph-error
number, so it is kept.

**The metric.** ONE number, `eval/shd`, matching that sentence literally: the learned graph
against the ground-truth causal graph, not a sub-block. Rows are the N decoded next-states,
columns are all 2N source tokens [state | params], so the range is [0, 2N²] = [0, 50] for five
balls. This is the same index set as the path objective and ρ_path (write-up Eq. 11), so the
pruning curve and the graph score describe the same object. Parameter-token ROWS stay excluded
because they are never decoded and carry no "parent of a prediction" meaning. Readout is
SPARTAN Eq. 5 exactly: Ā_ij counts paths j → i and an edge exists iff Ā_ij >= 1 (applied as
>= 0.5 on integer counts). Ground truth is the concatenation of write-up Eq. 8 (state) and
Eq. 9 (mass), in token order.

**Superseded.** The previous split into `shd_state` / `shd_param` / `shd_param_aligned` is
gone; none of those keys are logged any more and their values are not comparable to `shd`.
`gt_graphs_from_contacts` / `read_learned_graphs` became `gt_causal_graph_from_contacts` /
`read_learned_graph`, each returning the single (N, 2N) graph.

**CAUTION — SHD alone rewards learning nothing.** Measured directly from the simulator
(n=2000 episodes, seed-29 split, window t in I): the true graph has only 7.72 edges out of 50
(5 self-edges + 2.72 contact/mass edges). Reference points, verified:

  * perfect model            SHD = 0
  * EMPTY graph (A=0)        SHD = 2.72   <- the token-local reference, MCC = 0
  * SATURATED graph (A=1)    SHD = 42.28  <- the dense reference
  * achieved (D30)           SHD = 4.81   <- WITH MCC 0.948; 91% recall, 63% precision

Since lower is better, the mass-blind model beats the mass-recovering one ~15x. This is the
standard SHD failure on sparse ground truth (false positives dominate), not an implementation
bug. `shd` is therefore NOT a standalone quality signal: read it against both references and
JOINTLY with `mcc`. Success is `shd` falling toward ~0 WHILE `mcc` stays high and `pred_loss`
stays near the dense reference; failure mode #2 (empty-graph collapse) produces `shd` ~= 2.9
with `mcc` = 0, i.e. the second-best possible SHD. Also note `shd` is a deterministic function
of the ground-truth edge count whenever the learned graph is saturated or empty, so a frozen
value is not a model property at all — it only becomes evidence once density leaves its rails.

**Naming.** What is compared is REACHABILITY agreement on the thresholded path matrix, not
verified causal use. The write-up §6.7 must be updated: it currently says "state and parameter
structural Hamming distance (SHD)" with range [0, 25]; it is now one SHD with range [0, 50].

## D29 — Experiment-1 exact replication refactor (decided 2026-07-25, Jesse)

**The instruction.** The write-up §6.1–6.2 (formerly `docs/experiments.tex`, deleted in commit
3fbcdfd — recover with `git show 3fbcdfd^:docs/experiments.tex`) is the SPEC for Experiment 1.
The code was rebuilt to match it 1:1, and everything Experiment 1 does not use was deleted
rather than kept as ablation surface. This is what superseded D1–D26 (see the preamble list).

**What the code now is.** One model per §6.2: `ParameterEncoder` (Eqs. 16–26: shared linear
embed + learned temporal PE → per-timestep relational self-attention across tracks → single
shared temporal query pooling per track → unconstrained scalar head) + `Spartan` (Eqs. 27–37:
separate W_Z/W_θ projections, role embeddings, SHARED fixed non-trainable track key κ_i added to
the state and parameter token of track i, single-head hard-gated layers, MLP(x+h), path matrix,
decoded-rows path objective) composed by `StateToStateModel` (Eq. 38: same θ̂ for all 30
transitions, every prediction anchored at the true Z_t). Trainer objective is exactly Eq. 40;
dual constraint exactly Eq. 13; dual update log λ += α·MA[c−τ] with λ₀=1e6, no clamp. Deleted:
SCJepa, rollout machinery, all four pooling variants, kinematic head, Hungarian matching,
VISReg/SlotRegularizer, target-variance constraint normalization (returns with Exp 3's Eq. 123),
aux-token pathway, gate-noise chaining, sparsity warm-up, λ clamps, the synthetic smoke dataset.
Verified fidelity anchor: the dense model's path objective is EXACTLY 6655 and token-local 5
(§6.1.3's stated endpoints; regression-tested).

**Unspecified-upstream choices (flagged in code):** learned temporal PE; FFN_time hidden width
2d; track-key scale 0.02 (matched to role-embed init) from a fixed seed-0 codebook shared by all
three predictor modes; Adam; the dual step α.

**Pipeline (why the token-local training stage was dropped).** §6.1.3 gates the sparse run on
the dense constraint sitting below the token-local constraint. The token-local model (A≡0)
disconnects parameter tokens from every decoded row, so its constraint is INVARIANT to the
parameter-encoder architecture; the measured raw floor pred=0.043645 (run ku244l5e, 300k steps,
offset-17 split) therefore remains valid, and the launchers enforce the gate arithmetically
(τ ≤ 0.043645 + λ_logit·2.0) instead of retraining a third stage per pipeline. The confirmatory
8-seed protocol still trains token-local references — they are one of the three compared
checkpoints, not just a gate.

**λ_logit and τ protocol (per §6.1.3, one full dense run for τ).**
1. λ_logit: label-free DENSE sweep (grid now 0:1e-6:3e-6:1e-5:3e-5:1e-4:3e-4:1e-3 — recentered
   low because the raw-scale feasibility arithmetic of 2026-07-25 bounds usable values at
   ≈ 3e-5: gate commitment costs λ_logit·(2cosh|l|−2) INSIDE the constraint, and τ has no slack
   for it). Selection rule unchanged: zero control, ≤5% pred tolerance, smallest Pareto
   coefficient reaching 90% of the best admissible reduction of L_logit−2. Sweep runs may be
   shortened (SWEEP_STEPS) — the sweep compares dense runs to each other, it does not set τ.
2. τ: exactly ONE full-length dense run per architecture/seed inside the pipeline;
   τ = 1.0 × its held-out constraint_loss (Eq. 13 units). No factor, no identity stage.
   CAVEAT (from the 2026-07-25 feasibility analysis): τ=1.0× leaves the gated model zero slack
   — it must shed essentially all Bernoulli gate noise AND pay its commitment cost. If
   eval/constraint_loss plateaus above τ with λ falling never engaging pruning, the fallback
   order is unchanged: smaller λ_logit, then Gumbel temperature, then a LABELLED slack ablation.

## D30 — Experiment 1 works; tau=0.02 is a test value (verified 2026-07-25, run 7cq3h2ur)

Experiment 1 runs correctly. The D29 code produces the intended effect: driving the path
objective down under the GECO constraint prunes the graph AND recovers the masses. This is the
first sparse run that does both — every earlier one either never pruned or collapsed to the
empty graph. **The mechanisms work.**

Run `7cq3h2ur` (commit 3fbcdfd, 300k steps / 1.6 h, seed 0): `eval/mcc` 0.948, `eval/shd` 4.81,
`eval/path_density` 0.223, L_path 12.3 (dense 6655, token-local 5), `eval/pred_loss` 0.0142,
zero skipped steps. Read the two metrics as a pair (D27/D28): the empty graph gets a better SHD
(2.72) but mcc 0.0, so mcc 0.948 at shd 4.81 is the success signature. Config: tau=0.02,
lambda_logit=1e-5, lambda_0=1e4, alpha=2e-2.

**tau=0.02 was chosen for testing, not by protocol.** It is not the S6.1.3 tau = 1.0 x the
held-out constraint of a converged dense run (~0.011-0.014), so this run had real slack.
That is acceptable here because Experiment 1 is not the main experiment — its job was to
confirm the mechanism end-to-end, which it did. Any reportable tau must still carry its
provenance.

Mechanism, for reference when reading future runs: lambda first CLIMBS while c > tau and the
graph saturates to density 1.0 (0-75k) — this phase is indistinguishable from a never-prunes
failure, so do not judge a run before ~150k steps. It then reverses and prunes (90-155k), and
`mcc` rises 0.46 -> 0.935 inside that same window: identification is concurrent with
sparsification. lambda settles at 2272 with the constraint pinned at the tau boundary.

Retired by this result: failure modes #3-#5 (VISReg scale collapse, BPTT zombie freeze,
mid-density gradient detonation) are structurally unreachable post-D29 — no regularizer, no
rollout, no gate-noise chaining. #1 (logit explosion) and #2 (empty-graph collapse) stay live.

## D31 — Visual data condition and fixed trajectory alignment (2026-07-25, Jesse)

The visual condition renders the same simulator states with `render_radius_from_mass=False`
and `uniform_appearance=True`: every object is drawn as the same white disc. Physical radii
remain mass dependent; the rendering hides direct glyph-size and colour identifiers, not the
physical evidence in collisions and occlusion. `mass_independent_init=True` is a separate
optional control because it changes the state distribution. It is not enabled in the primary
same-physics comparison.

One stored physics file serves both active experiments. Render settings are excluded from
`generation_meta`; controls that change states, including mass-independent initialization, are
included. Frames render on demand rather than storing a roughly 294 GB frame tensor.
The canonical seed-0 preload must not be regenerated on another machine: identical simulator
settings gave different trajectories and contact graphs across machines. This is why the visual
experiment requires the recorded Experiment-1 preload.

Anonymous visual tracks need one fixed assignment over an episode wherever rows are compared.
Per-frame rematching could erase identity switches. The original supervised prediction-target
assignment is retired with that experiment; the shared utilities remain useful for evaluation.
Current training branch matching is context-only and is defined in D39. Evaluation matches slot
allocation centroids to physical trajectories without using masses or parameter recovery.

`python scripts/plot_bounce_episode.py --condition states|visual` shows the rendering conditions.
The renderer's y coordinate increases downward; old y-up plots are vertically mirrored relative
to the actual encoder input.

## D32 — Experiment 2 foundation: visual context and EMA visual targets (2026-07-25, Jesse)

The visual-to-visual experiment reuses `Spartan`, `ParameterEncoder`, the training-loop
safeguards, MCC, and SHD from Experiment 1. `models/visual.py` supplies the SAVi encoder and
row-wise state head; both are copied into the EMA target. `models/visual_to_visual.py`,
`training/visual_to_visual.py`, and `eval/visual_to_visual.py` hold regime-specific behavior.
The preset is `configs/experiment/bounce_visual_to_visual.yaml`.

The optimized predictive error is raw latent MSE. The GECO scalar normalizes the predictive
term by detached, floored target content variance and adds the weighted logit penalty. This
keeps the controller sensitive to representation scale above the floor, but neither the floor
nor EMA rules out a constant representation. A collapsed representation can still achieve
zero predictive loss. Both spatial/content and temporal collapse diagnostics are necessary.
The early dual trajectory is not expected to resemble Experiment 1's raw-state trajectory.

The initial implementation compared same-index rows based on shared EMA ancestry. That was an
assumption in `outdated_experiments.pdf` (PDF p.29), not a guarantee of physical tracking. D39
replaces it with explicit fixed context-prefix branch matching, following the current
manuscript's matching description. Source-PDF equation numbering also differs from older code
comments; the normalized constraint appears as Eq.122 on PDF p.30 of the supplied old proposal.

Visual runs have not demonstrated full convergence. The starting logit coefficient `1e-5` is
inherited from the true-state reference and remains to be screened for the visual architecture.
Parameter interventions and EMA-speed controls remain future work.

## D33 — Retired supervised visual-to-state bridge (2026-07-25; retired 2026-09-06)

The old proposal included a middle experiment predicting a fixed raw four-dimensional physical
state from visual latent inputs. The user has skipped that experiment. Its model, trainer,
evaluation entry point, preset, dedicated tests, and launcher are removed. It is not an active
regime or a prerequisite for interpreting Experiment 2. The two supported regimes are now
`state_to_state` and `visual_to_visual`; the latter is Experiment 2 throughout active code,
launchers, and documentation. Source PDFs are preserved as supplied.

## D34 — Experiment 1 objective is HYBRID: teacher forcing + a full-window autoregressive rollout, both inside the constraint (decided 2026-07-27, Jesse)

> **SUPERSEDED for Experiment 1 by D37.** The text below is retained as the historical
> motivation for adding an autoregressive branch. Experiment 1 no longer trains through a
> K=30 rollout. D39 also supersedes the historical full-K visual objective in D34b.

D29 made Experiment 1 a pure teacher-forced one-step objective and deleted the rollout. D34
puts a rollout back, on a different footing: it is now a SECOND branch alongside teacher
forcing, not a replacement for it, and it is inside the dual constraint.

> **Superseded optimisation detail:** D36 retains the exact final K=30 objective below but
> replaces fixed-K training, the later λ_roll ramp, and D35's prefix-horizon curriculum with
> a spatial-coverage-to-truncated-BPTT continuation. The fixed anchor, terminal horizon,
> weighting, and theoretical claim remain.

**Objective (hybrid write-up Eq. 36, dual form).**

    sparse:  L = L_TF + lambda_roll*L_roll + lambda_logit*L_logit + lambda^-1 * L_path
    bound:   c = L_TF + lambda_roll*L_roll + lambda_logit*L_logit  <=  tau

`L_TF` is unchanged: 30 teacher-forced one-step predictions, every one anchored at the true
Z_t (Eq. 32/39). `L_roll` is Eq. 35 — ONE autoregressive chain per episode anchored at the
true Z_29 and rolled out over the whole prediction window to Z_59, with every prefix
k = 1..K supervised (Remark 4's difference from V-JEPA 2-AC, which supervises only the
terminal state). Both branches share the single theta-hat pooled from observations 0..29;
that sharing is the point, not an optimisation (§4.4(ii)).

**K = 30, one fixed anchor at t = Tpar-1.** The write-up samples the start; we do not. A fixed
anchor makes train and eval compute the same quantity, which matters because tau is calibrated
on the eval constraint, and K=30 covers the entire window from one chain, so no sampling is
needed for coverage. Tpar-1+K = 29+30 = 59 = T-1 fits the EXISTING 60-step clips exactly:
**no preload regeneration** (measured: lengthening clip_len is a pure prefix extension, first
60 states/59 contacts bit-identical, but that is not needed here).

**w_1 = 0 always, remaining weights uniform and normalised so K^-1 sum_k w_k = 1.** At k=1 the
rollout recomputes f_gamma(Z_29, theta-hat) against Z_30 — bit-for-bit the teacher-forced term
at t=29, differing only by the gate draw. The anchor is structurally inside
I_TF = {Tpar-1,...,T-2}, so the write-up's coverage condition holds via L_TF without it.
Normalising keeps L_roll a MEAN per-step error, so lambda_roll does not silently rescale the
constraint when K changes. Eq. 35 leaves w_k free; uniform is our choice, because §4.4(ii)
requires only that every prefix be constrained and any other profile would be an unmotivated
hyperparameter. lambda_roll = 1.

**The rollout belongs INSIDE the constraint, and so does the logit term — this is Baumgartner,
verified.** Their Eq. 9 (p7) is `min L_path s.t. L_rec + L_KL + L_logit <= L*`: the logit loss
is inside the bound, and L_KL is absent for us only because Experiment 1 has no cVAE. Their
`L_rec` reconstructs p_psi(tau | theta-hat, x_0) where Eq. 2 defines tau as the AUTOREGRESSIVE
trajectory [x_0, f(x_0), f o f(x_0), ...]. Their reconstruction target is itself a
full-trajectory rollout, so bounding L_roll moves toward their formulation; pure teacher
forcing was the deviation. §4.3 specifies "upper bounds on the teacher-forced and rollout
errors" (plural); we scalarise into ONE bound with the same lambda_roll that weights the
objective, keeping one dual variable and one tau.

**tau MUST be recalibrated.** c now carries L_roll, so every pre-D34 tau (including D30's 0.02)
is on a different scale and is NOT transferable. The pipeline recalibrates automatically because
`scjepa.eval.harness` computes the identical scalarised constraint from the same config keys —
that mirroring is the single most breakable thing in this design. Changing `rollout_len` or
`lambda_roll` on one side alone silently invalidates tau.

**The token-local launch gate (`--tau-max`) is DELETED.** It compared tau against a floor
(0.043645 + lambda_logit*2.0, run ku244l5e) measured under the pure-TF constraint, which is a
different quantity now.

**Stability: measured, not argued.** 1500-step CPU smokes at paper geometry (T=60, Tpar=30,
3 SPARTAN layers): K=10 gave grad_norm max 4.02, K=30 gave max 4.43, **zero skipped steps in
both**. The per-k drift curve at K=30 SATURATES rather than compounding — error rises to ~0.11
at k~19 and falls back to 0.092 by k=30 (ratio k=30/k=1 = 1.35), because five balls in a box is
a compact bounded state space. Catalog failures #4/#5 were pure-rollout runs with NO teacher
forcing and predate the masked-softmax/denominator numerics fixes, so they do not transfer.

**Still unmeasured:** no full 300k hybrid run exists. D30's four-phase trajectory was recorded
under the pure-TF objective; the phase STRUCTURE should survive (it is a property of the GECO
dual) but none of its numbers are comparable.

### D34b — Historical visual full-K rollout (2026-07-27; superseded by D39)

The visual predictor maps its learned state width to itself, so it is composable. The former
visual training objective used one full K=30 generated chain, anchored in the online branch,
with one fixed parameter estimate and one set of episode keys. D39 replaces that training term
with eight independent T=2 endpoint windows. The old visual-specific rollout configuration keys
and any thresholds calibrated against that full-K objective are retired.

The useful collapse lesson remains: content variance and effective rank pooled over episodes
and time can look healthy when a representation varies across episodes but is frozen within
each episode. Such a representation makes the identity transition sufficient. Temporal variance
must therefore be reduced over time within each episode first. Neither the rollout nor the
variance-normalized GECO scalar prevents this degenerate solution.

## D35 — Reach the exact K=30 objective through an accepted-update horizon curriculum (decided 2026-08-02, Jesse)

> **SUPERSEDED by D36, and its K=30 target superseded for Experiment 1 by D37.** This entry is
> retained as the historical diagnosis of the failed
> coefficient ramp and the first continuation attempt. Its off/2/5/10/20/30 optimisation route,
> 60k terminal boundary, and 300k budget are no longer active. D36 leaves D34's final scientific
> objective unchanged.

D34's 1500-step fixed-horizon smokes did not predict long-run stability. The first full-weight
K=30 launch entered a sustained BPTT gradient explosion early. Continuing `lambda_roll`
linearly from zero over 10k attempted steps only delayed the same failure: in run `ecbjobkj`,
the teacher-forced branch remained healthy while the rollout branch reached gradient norms of
1e7–1e8 and the D18 guard aborted after 2000 consecutive rejected batches. Scaling the rollout
coefficient therefore does not address the number of recurrent Jacobians traversed by backward.

**The final scientific objective is unchanged.** The reportable model is still trained and
evaluated with one fixed parameter estimate, one chain from Z_29 through Z_59, terminal K=30,
dense prefix supervision, and `lambda_roll = 1.0`. The complete 60-state clip and all causal-event
coverage are retained. The curriculum is an optimisation route to that objective, not a shorter
trajectory claim. `L_TF` continues to cover all 30 suffix transitions at every stage; only the
number of autoregressive compositions changes.

**The schedule is indexed by successful optimizer updates, not attempted batches.** The horizon
used for the next batch is:

| Successful updates before the batch | Autoregressive branch |
|---:|---:|
| `[0, 10,000)` | off (teacher forcing only) |
| `[10,000, 15,000)` | K=2 |
| `[15,000, 25,000)` | K=5 |
| `[25,000, 40,000)` | K=10 |
| `[40,000, 60,000)` | K=20 |
| `[60,000, ∞)` | K=30 |

A successful update is exactly a batch for which the finite/pre-clip-gradient guard accepts the
gradients and `optimizer.step()` executes. Gradient clipping does not make an accepted batch a
skip. A rejected batch advances the ordinary attempted-step counter and skip diagnostics, but
does not advance the curriculum, the optimizer, EMA hooks, or GECO. The accepted-update counter
is checkpointed, and a checkpoint is written immediately after the final accepted update of each
stage, before the next horizon is used. Resume restores that counter; it must not infer progress
from attempted steps alone. Experiment 1 lowers the persistent-skip abort from the base default
of 2000 to 50 consecutive batches: once a horizon is stuck, its accepted-update clock cannot
advance into the next stage, so the correct recovery is the preceding boundary checkpoint.

**`lambda_roll` has no schedule.** It is 1.0 whenever the rollout branch is present. The TF-only
stage has no rollout term because no autoregressive branch exists, not because its coefficient is
zero. The removed `lambda_roll_warmup_steps` continuation must not survive as a second hidden
schedule. The resolved config records the full curriculum and retains `rollout_len: 30` as the
terminal and post-hoc-evaluation horizon.

**Path sparsity and GECO activate only at terminal K=30.** Tau is calibrated from a converged
dense model under the full K=30 constraint

    c_30 = L_TF + L_roll,30 + lambda_logit*L_logit.

Using that tau while the live loss is TF-only or has K<30 would compare different constraints and
usually create artificial slack. GECO would then lower lambda and impose pruning pressure before
the full-rollout feasibility condition was even active. Therefore, in the sparse run, both the
`lambda^-1*L_path` term and the GECO moving-average/update are disabled until the batch uses K=30;
lambda and its moving average remain at their initialized values. At K=30 both activate together.
This leaves approximately 240k exact-K=30 sparse updates in a healthy 300k-attempt run.

**Evaluation has two deliberately different roles.** Periodic in-training evaluation follows the
live curriculum horizon and is a stage-health diagnostic; its pre-K30 `constraint_loss` must not
be compared with the final tau. Dense tau calibration and final/post-hoc evaluation always use the
configured terminal K=30 and `lambda_roll=1`. A dense checkpoint that has not completed at least
one terminal-horizon update cannot provide a reportable tau, even if it can be evaluated forward
at K=30. Precisely 60,000 accepted updates is the boundary checkpoint: it has completed the K=20
stage but trained zero K=30 batches. Reportability therefore requires
`successful_updates > 60000`. Dense and sparse runs use the identical declared curriculum,
although skips may make their horizon transitions occur at different attempted/W&B steps.

**What changes and what does not.** Training is path-dependent and spends its first 60k accepted
updates on dynamics pretraining rather than the final sparse objective. This introduces the fixed
stage boundaries and reduces the fraction of a 300k budget spent with sparsity active. It does not
truncate the declared window, detach the recurrent state, change `lambda_roll`, reset to ground
truth inside the chain, or weaken final full-rollout equivalence. Tau must be recalibrated from a
fresh dense D35 run; the failed ramp run and every D34/pre-D34 tau remain invalid.

D35 applied only to Experiment 1. Its historical curriculum is superseded by D37; the visual
experiment now follows D39's independently anchored T=2 protocol.

## D36 — Cover the trajectory locally, then remove gradient cuts from one continuous K=30 rollout (decided 2026-08-07, Jesse)

> **SUPERSEDED for Experiment 1 by D37.** This entry is retained to document the exact
> multi-window/cut schedule that was tried and why it was abandoned. None of its stages,
> accepted-update transitions, gradient cuts, or K=30 training loss remains active. D39 also
> replaces the visual full-K objective with T=2 endpoint windows.

D35 increased one prefix from the fixed start, but that can leave the later trajectory regions
untrained until a long recurrent graph reaches them. Three independently true-anchored windows
solve the coverage problem, but they can still hide compounding drift because the starts at
offsets 10 and 20 reset to ground truth. The D36 route therefore had two deliberately simple
parts: learn all three regions locally, then run the exact full forward trajectory while gradually
lengthening only its backward paths. **This was D36's debugging protocol; D37 is its recorded
replacement.**

**The final objective and observational-equivalence target are unchanged.** The terminal model
still uses one autonomous chain from `Z_29` through `Z_59`, supervises every declared prefix,
uses `lambda_roll = 1.0`, and performs exact K=30 BPTT. `L_TF` remains active over all 30 true-
anchored suffix transitions in every phase. D36 supersedes only D35's optimisation route.

**One parameter estimate per episode, everywhere.** The parameter encoder runs once on the
episode context to produce `theta_hat`. The exact same attached `theta_hat` tensor is reused at
every predictor call, across all three local windows and every step of the continuous rollout.
It is never recomputed, changed, or detached within a batch/episode; different episodes may of
course have different estimates.

**The schedule is indexed by accepted optimizer updates.** Starts below are offsets relative to
the prediction anchor `Z_29`, so `[0, 10, 20]` means true anchors `[Z_29, Z_39, Z_49]`.

| Accepted updates before the batch | Rollout training branch |
|---:|---|
| `[0, 10,000)` | off; teacher forcing only |
| `[10,000, 20,000)` | three true-anchored windows, starts `[0,10,20]`, each H=2 |
| `[20,000, 30,000)` | same three true-anchored windows, each H=5 |
| `[30,000, 50,000)` | same three true-anchored windows, each H=10 |
| `[50,000, 70,000)` | one continuous forward K=30 rollout; gradient cuts after steps `{10,20}` |
| `[70,000, 85,000)` | one continuous forward K=30 rollout; gradient cut after step `{15}` |
| `[85,000, 100,000)` | one continuous forward K=30 rollout; gradient cut after step `{20}` |
| `[100,000, 115,000)` | one continuous forward K=30 rollout; gradient cut after step `{25}` |
| `[115,000, ∞)` | one continuous K=30 rollout with no cuts: exact full BPTT |

The configured stop remains 355,000 attempted batches. In the intended healthy zero-skip run all
355,000 attempts are accepted, leaving 240,000 terminal updates. Any isolated skips reduce that
number and must be reported from the checkpoint's accepted counter; persistent skips abort under
D18. Checkpoints are written at every listed boundary, and resume restores that counter.
Each training record logs both the pre-batch accepted count that selected the stage and the
post-batch count, plus the stage boundary, window count, maximum BPTT depth, starts, and cuts.
This makes the single record that crosses a boundary unambiguous during debugging.

**The three-window loss does not grow threefold.** Each window is autoregressive only within
itself and begins at its stated true anchor. The three window losses are averaged, preserving the
scale of `L_roll` and hence the meaning of `lambda_roll = 1.0`. At H=10 the three windows tile all
30 suffix transitions. This phase provides coverage, not yet a claim that errors stitch into one
autonomous trajectory.

**A gradient cut is not a state reset.** From accepted update 50k onward, the forward computation
is always the same uninterrupted numerical trajectory

    Z_29 -> Zhat_30 -> ... -> Zhat_39 -> ... -> Zhat_49 -> ... -> Zhat_59.

After a designated step, the predicted state passed to the next call is `state.detach()`. In the
forward pass `detach(state) == state`; no true state is inserted and the scalar K=30 rollout loss
is unchanged. In the backward pass its derivative with respect to the computation before the cut
is zero. Cuts `{10,20}` therefore give one continuous 30-step forward rollout but three backward
paths of maximum depth 10. A cut `{15}` gives two paths of depth 15; `{20}` and `{25}` increase
the maximum recurrent backward depth to 20 and 25. Removing the last cut at 115k restores the
exact derivative through all 30 compositions. Predictor weights remain shared across chunks, and
the undetached shared `theta_hat` receives gradients from every predictor call.

**Sparsity is deliberately postponed.** The path penalty and GECO moving-average/dual update
remain inactive through every local-window and gradient-cut phase. They activate together only
for batches in the no-cut stage beginning at 115,000 accepted updates. This cleanly separates
stabilising exact full-horizon differentiation from imposing the constrained sparse objective.
Tau must be calibrated from a fresh, converged dense D36 run under the terminal K=30 evaluation;
no D35 or earlier tau is transferable. A boundary checkpoint at exactly 115,000 accepted updates
has completed zero no-cut updates and is not reportable; reportability requires
`successful_updates > 115000`. The 355k attempted-batch budget yields the intended 240k terminal
updates in a zero-skip run; provenance records the actual terminal accepted count otherwise.

D36 applied only to Experiment 1. The current visual objective is specified in D39.

## D37 — Experiment 1 returns to teacher forcing plus eight sampled two-step endpoints (decided 2026-09-02, Jesse)

D34–D36 tried to make exact K=30 recurrent backpropagation part of Experiment 1's training
objective. That route is retired. The continuation in D36 reached its uncut stage, but the
production run `b8v5lxu2` again entered a recurrent-gradient failure: the teacher-forcing branch
remained ordinary while the rollout branch dominated the gradient and the D18 guard stopped the
run after 50 consecutive rejected updates. The forward K=30 prediction itself had not become
non-finite. The evidence therefore points to the long recurrent backward graph, not to the
existence of a usable local transition model. Adding more curriculum stages would preserve the
same failure mode while making the result depend on a complicated training schedule.

**Restore the known-stable foundation.** Experiment 1 again uses the D30 teacher-forcing
geometry and 300,000-step budget. One `theta_hat` is inferred from the 30-state context
`(S_0,...,S_29)` and reused for all 30 true-state transitions
`S_29 -> S_30, ..., S_58 -> S_59`. The historical successful values
`lambda_logit = 1e-5` and `sparsity_lambda_init = 1e4` are restored as initial settings. D30's
`tau = 0.02` is *not* restored: tau was measured for teacher forcing alone and is on the wrong
scale for the new predictive constraint.

**The only new training branch is a fixed T=2 endpoint loss.** The predictive objective is

    L_pred = L_TF + lambda_rollout_t2 * L_AR2,

with `lambda_rollout_t2 = 1.0`, `num_rollout_t2_anchors = 8`, and
`rollout_t2_horizon = 2`. Existing non-predictive terms retain their previous roles. In the
sparse GECO objective this gives

    L = L_pred + lambda_logit * L_logit + lambda_s^-1 * L_path
    c = L_pred + lambda_logit * L_logit <= tau,

where the path penalty remains outside the constraint. The dense reference and sparse run must
use identical T=2 settings, and tau must be freshly calibrated from the converged dense
teacher-forcing-plus-T=2 model. No threshold calibrated for D30 or D34–D36 is transferable.

A **rollout anchor** is the true state from which one independent two-step generated window is
launched. For sequence length `T` and context length `C`, relative offset `r` denotes

    S_(C-1+r) -> Shat_(C+r) -> Shat_(C+1+r).

Validity is computed from the actual tensor length: `0 <= r <= T-C-2`. Thus C=30 and T=60
provide 29 valid offsets, `r in {0,...,28}`. For every episode and every training batch, sample
exactly eight distinct offsets uniformly without replacement. Sampling is independent across
episodes and uses the checkpointed training random-number stream, so a fixed seed and resume
preserve the expected sequence. The offsets are not frozen across training.

Every selected window starts from its true anchor. Its first prediction consumes that true
state; its second prediction consumes the generated first prediction. The intermediate
prediction is never detached, so the endpoint gradient traverses both transition calls. Only
the second prediction is supervised by `L_AR2`: the first transition is already present in
`L_TF`. All eight windows reuse the same episode-level `theta_hat`; it is neither recomputed nor
detached. The endpoint loss is

    L_AR2 = mean_(batch,window,object,coordinate)
            (Shat_(t+2) - S_(t+2))^2.

It is a mean, not a sum: duplicating a window or changing the number of sampled windows must not
systematically rescale it. Setting `lambda_rollout_t2 = 0` bypasses sampling and both auxiliary
predictor calls, recovering the teacher-forcing computation and random-number sequence exactly.

**Training horizon H=2 and evaluation horizon K=30 differ intentionally.** The T=2 term teaches
one local composition and exposes the transition model once to its own output, addressing local
composition error and exposure bias without a long recurrent Jacobian product. It does not
claim that two steps identify every physical parameter, nor does it replace the population
observational-equivalence assumption. There is no K=30 training loss, rollout curriculum,
horizon warmup, gradient-cut schedule, accepted-update stage state, or full-rollout
backpropagation in Experiment 1.

K=30 survives only as a deterministic held-out diagnostic. In evaluation mode and under
`torch.no_grad()`, infer one `theta_hat` from `(S_0,...,S_29)`, start from true `S_29`, and feed
each generated state into the next transition through `Shat_59`. The held-out episodes are
fixed. Let `sigma_d` be fixed coordinate standard deviations computed once from the training
set (or one for coordinates already standardized). For episode i and rollout step k,

    e_(i,k) = sqrt(mean_(object,coordinate)
                   ((Shat_(i,k) - S_(i,k)) / sigma)^2),
    E_i = max_(k=1,...,30) e_(i,k).

With `oe_tolerance_nrmse = 0.10`, report

    OE_hat_0.10 = mean_i 1[E_i <= 0.10]

as `eval/oe_sample_satisfaction_k30`, together with the median and 95th percentile of `E_i` as
`eval/oe_k30_worst_step_nrmse_p50` and
`eval/oe_k30_worst_step_nrmse_p95`. This is an empirical, tolerance-based estimate of
approximate trajectory agreement on sampled held-out trajectories. It is comparable over
training because the episodes and normalization are fixed, but it is not a proof of population
observational equivalence.

The schedule-specific D34–D36 logging is removed. Experiment 1's predictive training keys are
`train/loss_teacher_forcing`, `train/loss_rollout_t2_raw`,
`train/loss_rollout_t2_weighted`, and `train/loss_total`; branch gradient norms may additionally
use `train/grad_norm_teacher_forcing` and `train/grad_norm_rollout_t2_weighted`. Ordinary
finite-gradient/skip safeguards, dual/sparsity diagnostics, MCC, and SHD remain. D37 changes
the state-to-state Experiment-1 objective when it was recorded; D39 now extends the same
T=2 sampling and endpoint supervision to the visual Experiment 2.

## D38 — Gate Experiment-1 tau with a freshly trained identity reference (decided 2026-09-03)

D29's Experiment-1 rule `tau = 1.0 * C_dense` with no identity stage is superseded. Under the
D37 teacher-forcing-plus-T=2 objective, the zero-slack threshold can make the learned-gate model
chase the dense optimum without ever entering a useful pruning regime. The old teacher-forcing
identity loss is also on the wrong objective scale and cannot certify feasibility for D37.

Every Experiment-1 pipeline therefore trains a dense reference and then an identity
(`A = 0`, token-local) reference with the same seed, data, optimization settings, objective, and
training length. Both are evaluated on 256 held-out episodes at seed offset 17. Starting from
the most permissive candidate, the pipeline selects the first factor in
`[2.0, 1.8, 1.6, 1.4]` satisfying the strict empirical gate

    C_dense < tau = factor * C_dense < C_identity.

If no candidate satisfies the inequality, the pipeline stops before sparse training. The
selected factor is a declared slack heuristic, not a value fixed by the identification theorem;
the identity comparison ensures only empirical feasibility on the matched reference split.
Terminal sparse evaluation remains on the disjoint seed-offset-29 split.


## D39 — Two experiments, visual T=2 prediction, and explicit branch alignment (2026-09-06, Jesse)

**Scope.** Keep Experiment 1 (true states). Skip the supervised visual-to-state bridge and name
the fully visual extension Experiment 2. `sources/SCJEPA.pdf` is the current manuscript;
`sources/outdated_experiments.pdf` preserves the old three-experiment proposal. Neither PDF is
rewritten. The two presets are `bounce_baumgartner` and `bounce_visual_to_visual`.

**Extend the stable local objective.** Both experiments train all 30 teacher-forced suffix
transitions plus eight independently sampled T=2 windows, using D37's endpoint-only mean loss,
`lambda_rollout_t2=1.0`, and no detached intermediate prediction. Experiment 2 anchors each
window in its observed online visual state, supervises against its aligned EMA target two frames
later, and shares one context-inferred `theta_hat` and one set of episode keys across every
transition and window. No full-K rollout contributes training gradients. K=30 is evaluation-only.

The visual optimized objective remains

    L = L_TF + lambda_rollout_t2 * L_AR2 + lambda_logit * L_logit + lambda_s^-1 * L_path.

Its GECO constraint is

    c = (L_TF + lambda_rollout_t2 * L_AR2) / sg(max(V_target, epsilon_var))
        + lambda_logit * L_logit <= tau.

Only the predictive term is variance-normalized, and only in the scalar sent to GECO. The path
penalty remains outside the constraint. Dense calibration and sparse training must share the
same final T=2 objective, data, architecture, and optimization settings. Experiment 1 retains
D38's dense/identity feasibility procedure; the initial visual pipeline uses its own dense
normalized constraint as tau, subject to a gross-collapse rejection. Dense and sparse runs learn separate target feature spaces: equal normalized
constraint values need not imply equal physical fidelity, even after passing that screen. This
calibration remains exploratory; it does not establish visual convergence or a common recovered
physical representation. Experiment-1 thresholds never transfer.

**Causal visual inputs.** The online recurrence consumes source frames through time `t`; its
parameter encoder consumes only frames 0–29. The independent target recurrence includes future
target frames in the complete training sequence, but the target state at `t+1` is causal through
that frame. The predictor never receives that future frame, target hidden state, or simulator
truth. Training uses no physical-state, mass, segmentation, or reconstruction supervision.

**EMA does not establish semantic slot identity.** The current manuscript describes Hungarian
matching (SCJEPA PDF p.7, printed p.5). Its Assumption 2 and Remark 3 (PDF p.9, printed p.7)
explicitly separate shared EMA ancestry from non-collapse, Markov sufficiency, and equality of
limiting states. The old proposal's claim that component-wise EMA removes any need to align
branches is therefore not used as an implementation guarantee.

For each episode, compute one detached Hungarian assignment from the online and target pre-head
slot trajectories over the shared context prefix. Apply that fixed permutation to the entire
target sequence, including TF targets and T=2 endpoints. This context-prefix choice is our
explicit implementation decision: it uses no future target slots, physical states, masses,
prediction residuals, or per-frame rematching. The assignment changes row addresses, not the learned latent
coordinate system; EMA still has to maintain comparable features. It can resolve a branch-wide
permutation, but cannot conceal or repair within-episode tracking switches.

Stochastic dropout in the SAVi recurrence is disabled for this deterministic matching protocol;
the target remains in evaluation mode even while the online model trains. The target encoder
and target state head update only through EMA after accepted optimizer steps. Rejected gradient
updates must not update optimizer, dual, or target.

**Representation learning remains an empirical requirement.** No new anti-collapse regularizer
is introduced. EMA and a variance floor do not exclude constant states, background-only slots,
duplicate object allocation, or states frozen over time. Dense calibration rejects gross
collapse; full-run success still requires content/temporal variation, object tracking, state
recovery, and useful parameter/graph results. Low latent MSE or a finite CPU smoke is not success.

**Concise evaluation.** Keep MCC, SHD, path density, TF/T=2 losses, and dual/gradient health.
Compare learned graphs with the matching physical contact graph at each predicted transition,
not one final contact graph broadcast across time. Match anonymous visual tracks to physical
objects geometrically for evaluation without using true masses to choose that permutation.
Add a small slot-allocation panel and tracking/switch diagnostics. Frozen linear probes fit
position and velocity on training episodes and score separate held-out episodes; state
recoverability is evidence for the representation, not a claim that each latent coordinate is
a true-state coordinate or a proof of the theoretical assumptions.

The L40 entry point is `scripts/l40_exp2_pipeline.sbatch RUN_TAG LAMBDA_LOGIT [SEED] [STEPS]`;
the corresponding Isambard entry point is `scripts/isambard_exp2_pipeline.sbatch`. Both require
the recorded physics preload. The inherited logit coefficient and visual convergence remain
exploratory until measured in full runs.

**Local validation (2026-09-06).** The 189-test CPU suite passes, including causal/matching,
T=2 gradient flow, probe isolation, per-transition graph, and mocked L40 pipeline checks.
Repository Ruff lint/format and source/script Pyright checks pass. A five-update dense smoke
used the production visual architecture and 60-frame clips (batch one): finite TF/T=2 losses,
zero skipped updates, and a successful held-out evaluation/artifact pass. The short smoke
showed diffuse slots and very low suffix target variance; it is execution validation, not
evidence of state recovery or convergence. No L40 training run or live W&B upload was performed.
