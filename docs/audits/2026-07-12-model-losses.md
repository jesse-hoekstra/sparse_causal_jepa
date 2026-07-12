# Fidelity audit: model/loss stack vs SPARTAN, Baumgartner, my_paper

Date: 2026-07-12. Auditor scope: report-only, no code edits.
Code state: git 55b5282 (the commit the v2 run qqye6ug1 ran at).
Sources read equation-by-equation: `sources/SPARTAN.pdf` (Eqs. 3–6, App. A.1–A.2, Fig. 5),
`sources/dynamical_system.pdf` (Eqs. 7–11, App. E bounce, F.1, F.4, F.5),
`sources/my_paper.pdf` (Fig. 1, §4, §4.3), `docs/decisions.md` (D3–D15).

**Headline verdict.** No sign error, no reversed dual update, no wrong-equation bug was found.
The Lagrangian scheme is implemented faithfully to SPARTAN App. A.2 / Baumgartner Eqs. 9–10 —
and *faithfully* means: **nothing in either paper bounds sparsity pressure while the constraint is
slack; the scheme is designed to prune until the constraint binds at τ.** With τ set above the
empty-graph-achievable loss, the empty graph is the *global optimum of the objective as specified
by the papers*, not a code malfunction. That confirms the run-forensics diagnosis (τ calibration
protocol) as primary. I did, however, find (a) one genuine divergence from the printed equations
that plausibly *lowers the cost of pruning* (masked-softmax renormalization, F-1), (b) one
code-adjacent protocol gap that *inflates τ* (there is no true fully-connected reference model in
the codebase, F-8), and (c) a faithful-but-underappreciated interaction (logit penalty inside the
constraint absorbs the τ slack and converts it into *deeper* gate closure, F-9 — it explains why
λ settled at ~44 instead of falling to its floor, and is checkable against the W&B curves).

---

## 0. Symbol/shape cross-check (run config: bounce_baumgartner, states regime)

| Paper symbol | Code tensor | Shape (run) | Where |
|---|---|---|---|
| S_t (kinematic state) | `flat_state` | (B·K, N=5, d=32) | state_jepa.py:74–77 |
| Ŝ^ph (causal params) | `causal_params` | (B, 5, 32), one per episode (D15) | state_jepa.py:71 |
| S_{t+1} (target) | `target_slots` | (B·K, 5, 32) | state_jepa.py:75, 83 |
| tokens into SPARTAN | `tokens` | (B·K, T=2N=10, 128) | spartan.py:259–267 |
| A_l (layer adjacency) | `adjacency` | (B·K, 10, 10), hard {0,1} | spartan.py:129 |
| Ā (path matrix) | `path_matrix` | (B·K, 10, 10) | spartan.py:277–281 |
| \|Ā\| | `sparsity` | scalar (batch mean) | spartan.py:284 |
| L_logit (Eq. 11) | `logit_penalty` | scalar | spartan.py:169–172, 290 |
| λ (dual) | `exp(log_lambda)` | buffer | lagrangian.py:62 |

Consistency with the observed signature: T = 10 tokens ⇒ eval path_density of a pure-identity
path matrix = 10/100 = **0.100 exactly**; |Ā| with only the residual diagonal ≈ T = 10, observed
11.6 ⇒ ~1.6 stray paths. shd_param = 1.4595 = mean number of GT param edges per transition when
the learned param block is all-False (`graph.py:61` — the path-matrix diagonal lands in the
*state* block, so an identity-only Ā yields zero learned param edges). All three numbers are
mutually consistent with "all gates hard-closed, in train sampling as well as eval".

---

## 1. `src/scjepa/models/spartan.py` vs SPARTAN Eqs. 3–5 / App. A.1 and Baumgartner Eq. 7, 11

### Verified faithful

| Equation | Code | Check |
|---|---|---|
| Eq. 3: A_ij ~ Bern(σ(q_i·k_j)), Gumbel-softmax | spartan.py:56–69 | Binary-concrete sampling is *exact*: `hard = (soft>0.5)` ⇔ logits+Logistic(0,1) > 0 ⇔ Bernoulli(σ(logit)); STE standard. |
| Eq. 4 residual+MLP: ŝ_i = MLP(h_i + s_i) | spartan.py:173 | Matches (MLP over h+tokens; pre-LayerNorm is an addition, standard). |
| Eq. 5: Ā = (A_L+I)···(A_1+I), parent iff Ā_ij ≥ 1 | spartan.py:277–281; graph.py:50–62 | Left-multiplication order correct; readout `>= 0.5` ≡ `>= 1` on the integer counts produced in eval. (SPARTAN p.5 prints "A_{L+1}" — an upstream typo for A_L; code uses L layers, correct.) |
| Eq. 6/Baumgartner Eq. 8: \|Ā\| = L1 of path matrix | spartan.py:284 | Sum of (nonnegative) entries, batch mean = E‖Ā‖₁. Diagonal included (constant, zero gradient — documented D10.6). |
| Eq. 11 normalization 1/(L·T²) | spartan.py:172 (per-layer mean over i,j), :290 (mean over layers) | mean∘mean = 1/(L·T²)·Σ, times λ_logit in loop.py:182. Matches Baumgartner Eq. 11 exactly up to the clamp (F-3). |
| "adjacency disallows information flows" | spartan.py:131–160 | With mask-before-softmax, masked keys/values never enter h_i (row_max is over unmasked entries only; denominator over unmasked only). Ā_ij=0 ⇒ ∂pred_i/∂token_j=0 — pinned by test per D10.1. |
| Token layout / prediction readout | spartan.py:259–267, 287 | [state 0..N−1 | param N..2N−1 | aux]; prediction from state positions only (`tokens[:, :num_slots]`), per my_paper Fig. 1. |
| App. A.1 dims | Spartan ctor 185–233 | d→embed_dim in-projection, 3-Linear MLPs, single-head per layer. Run config uses layers=2, embed=128 vs A.1's 3/512 — a documented capacity knob, not an infidelity. |

### Findings (divergences / risks)

**F-1. Softmax denominator: BOTH papers print it UNMASKED; code renormalizes over unmasked
entries only.**
SPARTAN p.4 Eq. 4 and Baumgartner p.7 Eq. 7 print, identically:
`h_i = Σ_j A_ij·exp(q_i·k_j/√dk)·v_j / Σ exp(q_i·k_j/√dk)` — the adjacency mask appears in the
*numerator only* (the denominator's summation index is misprinted as i, but its summand carries no
A). Code (spartan.py:131–160) instead computes the softmax over *unmasked* entries
(D10 choice #1). Two independent papers printing the same formula weakens the "garbled typo"
reading — this may be what the (non-public) reference implementation actually does.
- *Paper-literal consequence:* pruning edge (i,j) strictly shrinks ‖h_i‖ (surviving weights do
  NOT renormalize up), so **every pruned edge costs prediction accuracy in a graded way** — a
  continuous MSE force resisting pruning. It also leaks masked keys through the denominator,
  contradicting the papers' own causal-readout claim (D10's argument is valid).
- *Our-code consequence:* surviving weights renormalize to sum 1, so pruning all-but-one edge of
  a row is nearly free for prediction, and pruning the last edge falls back to the pure per-token
  residual (h_i = 0, MLP self-dynamics). **The MSE pushback against pruning is much weaker than
  paper-literal.** In bounce, where most transitions are free flight (self-dynamics suffice),
  this makes edge deletion almost costless outside collision frames.
- Severity: **distorts-optimization** (could contribute to how easily/completely the graph pruned;
  does not by itself change the equilibrium, which τ sets). Could it produce the observed
  signature? Contributing, not sufficient. **Surface to Jesse: paper-says-X vs code-does-Y, and
  the paper-literal X contradicts the paper's own causal claim — an email-the-authors item
  (candidate addition to the D13 Anson Lei question list).**

**F-2. Gate-logit 1/√d scaling (known flagged interpretation) — no counter-evidence found; one
nuance to record.** SPARTAN Eq. 3 writes the *gate* as σ(q·k) UNSCALED while Eq. 4's *attention*
uses q·k/√dk SCALED — i.e., the papers derive two different logits from the same q,k. Code
(spartan.py:120–133) uses one scaled logit for gate, attention, and penalty. I searched both
papers for an alternative mechanism per the brief: **no separate gate projection, no q/k
normalization, no adjacency prior/bias, no dense initialization anywhere** (see §5). Nuance:
paper-literal unscaled gates start deep in sigmoid saturation at standard init (|logit| large,
random ~half open but *sticky*); our scaled gates start at |logit| ≈ 0.3 (Bern(0.5±0.08)) and are
maximally *plastic*. So the interpretation trades "untrainable at init" (relitigated and settled)
for "gates can be herded closed very quickly once any consistent pressure exists". Severity:
distorts-optimization (speed of pruning only); consistent with pruning completing by step ~2000.

**F-3. Eq. 11 penalty applied to the scaled logits, clamped |·|≤30 with linear tail
(spartan.py:169–172).** Baumgartner's Eq. 11 is on the unscaled q·k. Self-consistent with F-2
(the penalty targets the same logit that gates and attention read); the clamp is a documented
fp32-stability deviation acting as a hard wall. Severity: cosmetic given F-2 is accepted.

**F-4. Eval discretization `logit > 0` (spartan.py:69) — unspecified upstream; interacts with the
symmetric penalty.** The Eq. 11 penalty's minimum (logit = 0, gate = Bern(0.5)) maps to CLOSED
under the strict threshold, so ties break toward the empty graph in *eval* readout. In the v2 run
this was NOT the mechanism (train-time |Ā| ≈ 11.6 shows gates were genuinely deep-closed, ≈ −5,
not hovering at 0). Severity: cosmetic here; worth remembering for runs where train density and
eval density disagree.

**F-5. Gradient-bounding numerics (unmasked row-max, `clamp(max=0)`, denom+1 for empty rows;
spartan.py:141–159).** Deliberate post-rung1 stability fixes; they bound the straight-through
*reopening* gradient into masked gates by exp(logit − row_max) ≤ 1, while the sparsity gradient
into open gates (≥ 1 path per edge) is not similarly bounded. A mild closed-vs-open gradient
asymmetry on top of the inherent hard-attention ratchet (reopening evidence only arrives on rare
Bern(σ(−5)) ≈ 0.8% sampled-open events). Severity: cosmetic-to-low; the papers' formulation has
the same ratchet.

---

## 2. `src/scjepa/training/lagrangian.py` (+ loop constraint) vs SPARTAN App. A.2 / Baumgartner Eqs. 9–10

### Verified faithful

- **Formulation.** SPARTAN Eq. 7/8 (p.15–16): min |Ā| s.t. MSE ≤ τ; max_λ min_θ |Ā| + λ(MSE−τ);
  practical rewrite "(MSE − τ) + |Ā|/λ"; λ init high; moving-average estimator of MSE−τ.
  Code implements exactly the rewrite: loop.py:186–189 (`constraint + λ_reg·reg + (1/λ)·|Ā|`;
  the constant −τ is dropped — zero gradient, cosmetic) with `penalty_weight = exp(−log λ)`
  (lagrangian.py:66–68), λ_init = 1e3 (high ✓), MA momentum 0.99 ✓.
- **Update direction.** Paper: "λ ← α ∗ exp(MSE−τ) ∗ λ" (p.16; literally read this multiplies by
  α every step even at MSE=τ — clearly a misprint for the GECO form λ ← exp(α·(MSE−τ))·λ).
  Code: `log_lambda += step_size · MA(loss − τ)` (lagrangian.py:70–77) = the sensible reading.
  Sign check: loss > τ ⇒ λ grows ⇒ sparsity weight 1/λ shrinks ⇒ "error first, pruning later" ✓.
  Baumgartner's parametrization (Eq. 10: L_path + λ_dual·(constraint − L*), λ_dual→∞ then →0) is
  the same saddle point with the opposite λ convention; the effective sparsity:constraint gradient
  *ratio* is identical, and Adam washes out the overall scale. Alternating steps ✓ (dual update
  after each θ step, loop.py:201–202, on the detached constraint ✓).
- **Constraint contents.** Baumgartner Eq. 9: L_rec + L_KL + L_logit ≤ L*. Code: constraint =
  pred + λ_logit·logit_penalty (loop.py:182–186); the KL-analog (VISReg) is deliberately OUTSIDE
  the constraint (D12) so collapse cannot be traded for sparsity — a *documented, defensible*
  divergence (their β_KL = 1e-6 makes their KL term negligible in the constraint anyway, F.5 p.47).
- **λ clamps [1e-3, 1e6]** (lagrangian.py:34–36): papers specify none; SPARTAN Fig. 5 shows
  log λ reaching ≈ −5 (1/λ ≈ 150) and ≈ +10, so our cap on the sparsity weight (max 1e3) is
  *more* conservative than the papers' unbounded behavior. Not a contributor.

### The critical question: does anything prevent edge deletion while constraint ≪ τ?

**No — in either paper or in the code, by design.** While the constraint is slack the dual
monotonically raises the relative sparsity pressure (their λ_dual → 0 / our 1/λ → up to 1e3)
until the constraint *binds*. The equilibrium is "the sparsest graph whose constraint loss equals
τ". If τ exceeds what an empty graph achieves (here: empty-graph pred ≈ 0.045–0.05 < τ = 0.17),
the empty graph satisfies the constraint and is the optimum. The papers' ONLY protection is a
correct τ ("the loss achieved by a fully connected model", SPARTAN p.16) plus, implicitly, a slow
dual (their λ trajectories span 10⁵–10⁶ steps, Fig. 5). **The code is faithful; the failure lives
in τ and the dual speed** (see F-8, F-10).

### Findings

**F-8. There is no true fully-connected reference model in the codebase — τ was calibrated on the
wrong model class.** SPARTAN sets τ to the loss of "a fully connected model" (p.16), i.e. a dense
softmax transformer (their Transformer baseline). Our calibration protocol
(bounce_states.yaml comment; run script) trains the *same gated architecture with
`sparsity_enabled=false`* — but the Bernoulli hard gates keep sampling (spartan.py:64–68), the
logit penalty keeps pulling gate logits to 0 ⇒ gates hover near Bern(0.5), injecting permanent
masking noise. This reference (a) systematically overestimates the achievable constraint loss
(0.085 at 6k steps vs 0.045 for the 300k empty-graph model), and (b) was then multiplied by 2.0.
`Spartan` has no config path to force A ≡ 1 (deterministic dense attention). Severity:
**blocks-identifiability** (via τ), and it is the code-level face of the calibration failure:
even a fully-converged calibration run of the *current* code would not measure the quantity the
papers define. Could produce the observed signature? **Yes — it is the primary enabler.**

**F-9. Slack absorption by the logit penalty (faithful to Eq. 9, but it explains the frozen
state and is checkable).** At dual equilibrium the MA error is 0 ⇒ constraint ≈ τ. Observed
pred ≈ 0.05 with τ = 0.17 ⇒ the remaining 0.12 must be carried by λ_logit·mean(eˣ+e⁻ˣ) ⇒
mean e^|logit| ≈ 0.12/1e-3 = 120 ⇒ typical gate logit ≈ −4.8 ⇒ per-sample reopening probability
σ(−4.8) ≈ 0.8%. In other words: **because the logit penalty sits inside the constraint, the dual
spends the entire τ slack on driving gate logits deeper negative** (sparsity pressure rises until
the *logit* term, not pred, fills the budget). This simultaneously explains (a) why λ settled at
~44 instead of falling to the 1e-3 floor despite pred ≪ τ, (b) why gates were deep-closed rather
than hovering, and (c) why the param channel could not recover (0.8% sampled-open events carry
negligible STE signal). Verification against W&B qqye6ug1: `loss/logit` should sit ≈ τ − pred ≈
0.12 in steady state (and `sparsity/lambda` stationarity should coincide with `loss/logit`
plateauing). This is *faithful* to Baumgartner Eq. 9 — but it means τ mis-calibration doesn't
just permit the empty graph, it actively *deepens* the closure in proportion to log(slack/λ_logit).
Severity: distorts-optimization; explains the dead-channel depth. (If Jesse wants a lever without
diverging from Eq. 9: calibrate τ against a reference whose logit term is at its own equilibrium,
or report/monitor the constraint split pred vs logit — a metrics change, not an equation change.)

**F-10. Dual step size 0.02 (config) makes phase 1 ~2k steps in a 300k-step run.** With
MA error ≈ −0.12, log λ moves −0.0024/step: 1e3 → 44 in ≈ 1.3k steps; the sparsity weight is
already ~0.01–0.1 while the model has only learned free-flight dynamics, so collision edges are
pruned before they are worth their cost. SPARTAN's Fig. 5 λ curves evolve over ~10⁶ steps (α
unspecified upstream). The 0.02 value was chosen for 20k-step CPU runs (bounce_states.yaml
comment) and inherited by the 300k run. Severity: distorts-optimization; config-level, not a code
bug, but no code guard exists (none exists in the papers either).

**F-11 (cosmetic).** `ma_error` initializes at 0 (lagrangian.py:63), slightly damping the first
~1/(1−momentum) updates. Harmless.

---

## 3. `channel_split.py` / `state_jepa.py` vs my_paper Fig. 1 / D4 / D14 / D15

- **Param tokens are per-slot tokens:** Ŝ^ph (B, N, d) is concatenated as N separate tokens after
  the N state tokens (spartan.py:259–267) ⇒ T = 2N = 10 for 5 balls ✓ (matches |Ā| ≈ 11.6 ≈ T).
- **Prediction/loss read ONLY state positions:** `out_project(tokens[:, :num_slots])`
  (spartan.py:287); loss consumes `output.prediction` only (loop.py:180). Param-token outputs
  after layer L are discarded. ✓
- **Is there any path from Ŝ^ph to the prediction other than prunable attention edges? NO.**
  The layer MLPs are per-token, residuals are per-token, `in_project` weight-sharing between
  state/param tokens shares parameters but not information, role embeddings are constants. If
  every param→(anything) gate is closed in every layer, the pooling head (`CrossSlotAttnPooling`,
  channel_split.py:115–168) and the param content receive **zero gradient from the predictive
  loss**. Moreover the regularizer is applied to `context_slots` and `target_slots` only
  (loop.py:181) — **Ŝ^ph itself is NOT regularized** (faithful to my_paper Fig. 1, which draws
  SIGReg on the SAVi slot history and target slots, not on Ŝ^ph). Remaining gradient sources into
  the param branch after pruning: the logit penalty (shapes k_j magnitudes only, no task
  information) and rare Gumbel-open STE events (≈0.8%/sample at logit −4.8). **Architectural
  single point of failure confirmed: "empty param edges" and "dead param channel / identical
  recovery blobs" are one event.** This is faithful — the same SPOF exists in Baumgartner's
  design (θ̂ influences decoding only through SPARTAN edges; their KL provides no informative
  gradient either) — so I classify it as faithful-but-load-bearing, not a divergence. It is the
  mechanism by which F-1/F-8/F-9/F-10 manifest as the observed signature.
- **Channel split spec:** D14's CrossSlotAttnPooling is the default (last-step-anchored per-slot
  queries over all Th·N tokens, temporal PE, no slot PE) — a *decided* departure from my_paper's
  D4 text (manuscript update pending per D14). KinematicHead = linear on last-step slots ✓
  (my_paper §4: "last time-step encoding … passed through a linear layer to disassociate it").
- **D15 sliding window:** Ŝ^ph pooled once from frames[:context_len], held fixed for K = L−Th
  transitions (state_jepa.py:70–80). Ordering check: `flatten(0,1)` on (B,K,…) vs
  `repeat_interleave(K, dim=0)` on (B,…) produce matching (b,k) order ✓. States regime windows
  are re-embedded per step (memoryless) ✓ D13's no-leak guarantee holds.
- **Minor:** the S_t window embeddings (`current`, steps th−1…L−2) are not themselves
  regularized; scale is anchored through the shared `context_embed` weights via the regularized
  context window. Cosmetic.

---

## 4. `losses/predictive.py`, `losses/regularizer.py`, `training/loop.py` vs my_paper loss assembly (D6, D12)

- **Hungarian MSE** (predictive.py:27–58): per-sample squared-Euclidean cost, exact
  `linear_sum_assignment`, assignment detached (piecewise-constant a.e. — correct), MSE over
  matched pairs. Permutation-invariant ✓; **no stop-gradient on target** ⇒ gradients reach BOTH
  encoders (joint training, D7 / my_paper's deliberate departure from C-JEPA/SPARTAN) ✓.
- **Regularizer both branches** ✓ (loop.py:181: context_slots + target_slots, per Fig. 1).
  Vendored `visreg.py` matches VISReg Algorithm 1: center loss, scale loss (std−1)², sliced-
  Wasserstein shape loss against sorted Gaussian quantiles, **stop-gradient on std**
  (visreg.py:31 `std.detach()`) ✓.
- **Regularizer cannot be traded against the constraint** ✓: `constraint_loss = pred + logit`
  only (loop.py:186); `lagrangian.update(constraint_loss)` (loop.py:202) never sees reg. λ_reg
  stays 1.0 in the run config (D12) ✓. (The *min* player still sums all terms — that is what
  both papers do too; D12's requirement concerns the dual comparison, which is honored.)
- **Loss assembly matches my_paper**: L(Ŝ_{t+1}, S_{t+1}) after Hungarian matching + regularizer
  on both branches + SPARTAN sparsity (Lagrangian-weighted) + Eq. 11 logit term. ✓
- **Metrics** (loop.py:204–221, harness.py): path_density counts thresholded entries incl. the
  residual diagonal — hence the 0.100 floor reads "identity-only", correctly. `constraint_loss`
  reported by the harness (harness.py:127) is the same quantity the dual compares to τ ✓ — the
  calibration script reads the right number; the problem is the reference *model* (F-8), not the
  measured quantity.

---

## 5. Explicit answer: do the papers initialize/bias the adjacency FULLY CONNECTED?

**No. Neither paper contains any dense/open initialization, gate bias, adjacency prior, separate
gate projection, or q/k normalization.** Direct evidence:
- SPARTAN p.16 + Fig. 5: "At the start of the training … the model prediction error improves.
  This allows an *increase* in the number of active edges. As the prediction error becomes low
  enough, the sparsity penalty automatically increases, and the number of edges gradually
  decreases." Their "expected number of active edges" curves START LOW (~0–0.05), RISE to
  ~0.25–0.4 over ~10⁵–10⁶ steps, then decay. So SPARTAN's graphs open *because early prediction
  error exceeds τ*, not because of an open-biased init.
- Baumgartner Eq. 7 reuses SPARTAN's sampling verbatim; App. F contains no init detail; their
  mitigation for gate immobility is the logit loss (Eq. 11, F.4), aimed at the opposite failure
  (path loss *plateauing*, i.e. inability to prune — not over-pruning).

Consequence for the audit question: our Bern(~0.5) start is not contradicted by the papers
(paper-literal unscaled gates also start ~half open, just saturated/sticky, F-2). The papers'
early-training protection against over-pruning is therefore **not an init bias but the
combination (pred error > τ at start) + (slow dual)**. In the v2 run both were absent: the
empty-graph-achievable loss was already below τ, and the dual crossed its range in ~10³ steps —
so the "opening phase" that Fig. 5 shows never happened, exactly as observed (pruned by ~2k).

---

## 6. Everything the papers leave underspecified (and what we chose)

| # | Underspecified item | Source | Our choice | Where |
|---|---|---|---|---|
| 1 | Eq. 4 softmax denominator masking (printed UNMASKED in both papers; index misprint) | SPARTAN p.4; Baumgartner p.7 | Renormalize over unmasked only (mask-before) | spartan.py:131–160; D10.1 — see F-1, escalate |
| 2 | Gate logit scaling (papers: σ(q·k) unscaled; attention scaled) | SPARTAN Eq. 3 vs 4 | One shared 1/√d-scaled logit for gate/attention/penalty | spartan.py:120–133 (flagged interpretation) |
| 3 | Gumbel/binary-concrete temperature | both | 1.0 | config `spartan_temperature` |
| 4 | Eval-time discretization of A (sample vs threshold) | both | Deterministic logit > 0 | spartan.py:69; D10.5 |
| 5 | Dual step size α | both (SPARTAN says "step size"; curves imply 10⁵–10⁶-step trajectories) | 0.02 (chosen for 20k CPU runs, inherited by 300k) | bounce_states.yaml — see F-10 |
| 6 | λ init ("high") | SPARTAN p.16 | 1e3 | config |
| 7 | λ clamps | neither | [1e-3, 1e6] (more conservative than their unbounded) | lagrangian.py:34–36 |
| 8 | MA momentum | SPARTAN ("moving-average estimator") | 0.99 | config |
| 9 | Definition of the FC reference for τ | SPARTAN p.16 ("fully connected model") | Same gated model, sparsity off, 6k steps, ×2.0 | run script — see F-8 |
| 10 | λ_logit value ("small") | Baumgartner p.7/F.4 | 1e-3 | bounce_baumgartner.yaml |
| 11 | Eq. 11 numerical range | Baumgartner | clamp |logit| ≤ 30 + linear tail | spartan.py:169–172 |
| 12 | Token roles / layout of (S_t, Ŝ^ph, U_t) | my_paper §4.1 (layout yes, roles no) | learned role embeddings; [state\|param\|aux] | spartan.py:222–233; D10.2–3 |
| 13 | Heads per layer | SPARTAN Eqs. 3–4 (one adjacency ⇒ single head) | 1 | D10.4 |
| 14 | \|Ā\| diagonal inclusion | SPARTAN Eq. 6 literal | included (constant) | spartan.py:284; D10.6 |
| 15 | Bounce per-object token content | Baumgartner App. E (never specified) | [x, y, vx, vy] | D13 (email-Anson item) |
| 16 | Mass distribution, trajectory horizon | Baumgartner App. E | N(1.5, 0.5) clamp [0.5,3]; L=40, Th=10 | bounce_baumgartner.yaml; D13 |
| 17 | Pooling architecture for Ŝ^ph | my_paper (attention pooling layer, unspecified) | CrossSlotAttnPooling (D14; supersedes D4 text) | channel_split.py:115–168 |
| 18 | Regularizer placement on Ŝ^ph / S_t | my_paper Fig. 1 (drawn on S̃ and target slots only) | Not regularized (faithful to figure) — creates the dead-channel SPOF | loop.py:181; §3 above |
| 19 | Regularizer identity | my_paper says SIGReg | VISReg (decided D3, manuscript update pending) | regularizer.py |

---

## 7. Ranked code-level suspects for the empty-graph + dead-param-channel signature

1. **F-8 — No true fully-connected reference model (τ calibrated on the gated stochastic
   architecture, undertrained, ×2.0).** The one place where "protocol failure" has a code face:
   `Spartan` cannot run dense (A ≡ 1), so even a perfect protocol could not have measured the
   paper's τ. Directly produces the empty-graph equilibrium given the (faithful) dual.
   Severity: blocks-identifiability.
2. **F-9 — Logit penalty inside the constraint converts τ slack into gate-closure depth.**
   Faithful to Baumgartner Eq. 9, but quantitatively explains λ ≈ 44 (not the floor), gate logits
   ≈ −4.8, and the irrecoverability of the param channel. Testable against W&B:
   `loss/logit ≈ τ − pred ≈ 0.12` at steady state.
3. **F-1 — Masked-softmax renormalization vs the papers' printed unmasked denominator.** Removes
   the graded MSE cost per pruned edge that the paper-literal form has; makes over-pruning cheap.
   Genuine divergence (deliberate, D10.1, with a valid causal-readout argument) — needs Jesse's
   ruling / an author query, since the paper-literal equation contradicts the paper's own claims.
4. **F-10 — Dual step size 0.02 (config) compresses phase 1 into ~2k steps**, eliminating the
   papers' implicit "learn dynamics before pruning" window (their λ evolves over 10⁵–10⁶ steps).
5. **F-2 — 1/√d gate scaling makes gates maximally plastic at init**, enabling the fast closure
   that F-10's pressure demands (papers' unscaled gates start saturated/sticky). Affects speed,
   not equilibrium; already a flagged interpretation.
6. **§3 SPOF (faithful)** — Ŝ^ph reaches the loss only through prunable edges and has no other
   loss term; once pruned, the pooling head is gradient-dead. Not an infidelity (both papers share
   it), but it is why suspects 1–5 manifest as "identical uninformative blobs for all slots".

**Explicitly excluded (checked, faithful):** dual update sign/direction; the (MSE−τ)+|Ā|/λ
rewrite; λ init high; MA estimator; constraint membership (pred + logit; VISReg outside per D12);
Eq. 3 sampling exactness; Eq. 5 order and ≥1 readout; Eq. 6/8 |Ā|; Eq. 11 normalization
1/(L·T²); Hungarian matching and its gradient flow to both encoders; VISReg internals
(center/scale/SWD, stop-grad std); token layout and state-only prediction readout; D15 window
ordering; path_density/shd accounting (the 0.100 and 1.4595 constants are exactly what an
identity-only path matrix produces — the metrics reported the failure correctly, they did not
cause it).

**Bottom line.** The optimizer, losses, and dual controller do what the papers specify. The
papers' scheme has no built-in guard against pruning while the constraint is slack — the guard IS
τ (plus a slow dual). The v2 failure is therefore experimental design as forensics concluded,
with three code-adjacent aggravators worth fixing or ruling on before the next run: a real dense
reference for τ (F-8), awareness that constraint slack is spent on gate-closure depth via the
logit term (F-9), and a decision on the Eq. 4 denominator (F-1).
