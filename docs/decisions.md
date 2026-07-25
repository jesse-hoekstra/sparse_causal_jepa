# Architecture & engineering decisions

Living record for the codebase implementing **"Causal Identification within JEPA Using a
SPARTAN"** (`sources/my_paper.pdf`). Each entry states the decision and what would make us
revisit it.

**2026-07-25: D1-D26 were condensed to the rules below.** Experiment 1 is finished (D30) and
the D29 refactor superseded most of the historical narrative, so the archaeology was removed
rather than carried forward. The full original text (984 lines) is recoverable with
`git show 3fbcdfd:docs/decisions.md`. D27-D30 are kept in full because they describe the code
as it stands today.

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
Table 7 captions state it). my_paper.pdf p.13 and the write-up §6.7 both require a graph-error
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

## D31 — Experiment-2 foundation: visual data condition + trajectory alignment (2026-07-25, Jesse)

Experiment 2 (experiments.pdf S6.5) starts from two prerequisites the write-up names itself.
Both are implemented; the Experiment-2 MODEL is not yet.

**1. Visual data condition (p.14: "physical and rendered radii must be separated").** The
physics is untouched — same simulator, same Eq. 2 mass-proportional PHYSICAL radii, same
`data/bounce_train_v2_100000.pt` preload — so mass stays identifiable exactly as in D30. Only
the drawing changes, via three independent switches on `BounceDataset`:
`render_radius_from_mass=False` (all discs drawn at the shared radius),
`uniform_appearance=True` (one white glyph, no per-simulator-row colour), and
`mass_independent_init=True` (the separate control that removes the mass-dependent
initial-position support). Measured effect on the primary condition: correlation between total
episode mass and white pixel area falls 0.96 -> 0.17, rendered-area std 137.5 px -> 4.6 px, and
distinct frame colours 6 -> 2. The 0.17 residue is occlusion driven by physical contact
geometry, which p.14 explicitly allows as legitimate evidence; it is not a glyph signature.
Config: `configs/experiment/bounce_visual.yaml`.

Rendering settings are deliberately NOT part of `generation_meta`, because they cannot change
states/params/contacts — one stored physics file therefore serves both experiments.
`mass_independent_init` DOES change the states and so is in the identity, but only when
enabled, so files written before the flag existed still compare equal. Frames are rendered ON
THE FLY from preloaded states at ~6 ms/episode (170/s/worker); storing 100k x 60 frames would
cost ~294 GB. Placement now restarts the whole layout on failure — greedy sequential sampling
essentially never packs five balls at the worst-case radius r_max = 0.16.

**2. Trajectory-level alignment (Eqs. 98-100), `src/scjepa/losses/alignment.py`.** One detached
Hungarian assignment per EPISODE between anonymous visual tracks and simulator rows, ranked on
frozen training-split coordinate scales, reused for the prediction loss, parameter evaluation
and every graph axis. Targets are moved into visual-track order; nothing is permuted inside the
predictor. Per-timestep rematching is forbidden (S6.4) and is regression-tested: an episode
whose identity switches midway keeps a positive residual instead of being matched away.

**Experiment 1 is untouched and stays independently runnable.** Every new data knob defaults to
the Experiment-1 behaviour, verified byte-identical against pre-change output; the two configs
select the two conditions (`experiment=bounce_baumgartner` vs `experiment=bounce_visual`) and
report the same `generation_meta`.

**Still needed for a runnable Experiment 2** (none of it touches Experiment 1): the
`VisualToStateModel` itself — SAVi Q_psi (S6.3; the existing wrapper already returns the Eq. 42
(B,T,5,32) contract on these frames, verified) -> state head g_omega (Eq. 87) -> parameter
encoder with the visual input map W_vis (Eq. 91) -> SPARTAN with W_S in R^{512x32} decoding to
R^4 (Eqs. 93/95); episode-level permuted track keys from a fixed codebook (S6.4, which differ
from Experiment 1's fixed per-index keys); the trainer hook that gathers targets through the
assignment; the SEPARATE evaluation alignment zeta_e (Eqs. 137-138), which matches
slot-attention mask centroids to true rendered centres rather than using prediction error; the
S6.4 readiness checks as tests; and a freshly calibrated lambda_logit and tau_2 (Eq. 103).

**Addendum (2026-07-25): there is no separate v3 `.pt`, and the preload must never be
regenerated.** v3 is a RENDERING condition, not a simulator version — the physics is unchanged,
so a regenerated states file would be redundant. Worse, it would not even be the same data: the
simulator is deterministic on one machine but NOT across machines. Regenerating the identical
config locally instead of on the server diverges chaotically (n=200: 194/200 episodes differ,
max |delta| 3.4, and 26% get a DIFFERENT contact graph, i.e. a different ground-truth causal
graph for `eval/shd`). Masses are identical; only trajectories diverge. Experiment 2 therefore
renders frames on the fly from the same `data/bounce_train_v2_100000.pt`, which guarantees it
shares Experiment 1's exact physics. The only visual control that needs its own file is
`mass_independent_init=True`, which genuinely changes the states.

Episode figures: `scripts/plot_bounce_episode.py --condition states|visual`. Rendered examples
of the Experiment-2 input are `data/bounce_v3_visual_episode_00000.png` (with the physical
collision radii annotated) and `data/bounce_v3_visual_clean_episode_00000.png` (raw encoder
input). The renderer puts y DOWN, so both are vertically mirrored relative to the older
`data/bounce_v2_episode_00000.png`, which was drawn y-up.

## D32 — Experiment 3 implemented: visual context, EMA visual target (2026-07-25, Jesse)

Experiment 3 (experiments.pdf §6.6) is written and runs end-to-end. It reuses the
Experiment-1 pieces rather than forking them: the same `Spartan` (now with an `output_dim` and
optional per-episode `track_keys`, both defaulting to Experiment-1 behaviour), the same
`ParameterEncoder` (Eq. 91's visual input map is just a slot-width input), the same `Trainer`
(three overridden seams), the same `nonlinear_mcc` and `structural_hamming_distance`.
**Experiment 1 is byte-identical** — regression-verified, and it still trains from
`experiment=bounce_baumgartner` with no code path in common beyond the shared modules.

**What is new.** `models/visual.py` (`VisualStatePath` = Q_psi + g_omega, the object that has an
EMA twin), `models/experiment3.py` (Eqs. 105-121), `training/experiment3.py` (Eq. 123 constraint,
Eq. 111 EMA step, Eq. 124 collapse diagnostics), `eval/visual_alignment.py` (Eqs. 132-138),
`eval/experiment3.py`, `scripts/eval_visual_to_visual.py`,
`configs/experiment/bounce_visual_to_visual.yaml`, `scripts/isambard_visual_pipeline.sbatch`.

**Three things differ from Experiment 1, and only three.** (a) Frames in, not true states.
(b) The dual is fed Eq. 123's VARIANCE-NORMALIZED constraint — a learned target's scale drifts,
so an unnormalized c would silently retune itself; the gradient objective (Eq. 121) keeps the
raw latent MSE, and a floor epsilon_var stops a collapsing target from making the constraint
satisfiable by shrinking. (c) The EMA target is stepped after every optimizer step.

**No matching in training.** The EMA copy is initialized from the online encoder and updated
component-wise, so it never permutes slot rows: predicted row i is compared with target row i
(Eq. 116). §6.6 forbids matching here because it is unnecessary AND could conceal disagreement
between the two recurrent trackers. The Hungarian machinery in `losses/alignment.py` belongs to
Experiment 2's true-state target and to Experiment 3's EVALUATION only.

**Getting metrics out required exposing slot allocations.** Experiment 3's tracks are anonymous,
so `mcc` and `shd` are unreadable until each track is matched to a physical object. Eqs. 134-138
do that on trajectory GEOMETRY — slot-attention mask centroids against true rendered centres —
never on the learned parameters or the true masses, so it cannot be a permutation chosen to
flatter the score. The vendored SAVi does not return its allocations, so `SAViEncoder.allocations`
recovers them with a forward hook rather than patching `third_party` (D5 minimal diff), verified
to satisfy Eq. 71 exactly and to recover a known permutation of true centres.

**Known cold-start property.** At initialization the untrained encoder's slots are nearly
constant, so V_tgt sits at its floor and Eq. 123's constraint starts in the hundreds. tau_3 is
calibrated from the dense run in the same units, so it is self-consistent, but the early dual
trajectory will look nothing like Experiment 1's — do not read D30's four phases onto it.

**Practical warning: memory.** Two SAVi branches over 60 frames at 64x64 dominate; the context
branch holds a full backward graph. `bounce_visual_to_visual.yaml` therefore sets `batch_size: 4`
(Experiment 1 used 16). Raise only if it fits, and re-check the views-per-episode arithmetic
(D12) if you do.

**Not yet done.** lambda_logit is reused from Experiment 1 (1e-5) rather than swept — defensible
because the logit penalty acts on SPARTAN's own attention logits, which the visual state
interface does not change, but §6.1.3 asks for a fresh label-free dense sweep before the
confirmatory seeds. Also absent: §6.7's parameter interventions, the multi-step latent rollout
diagnostic, the faster/slower EMA controls, and Experiment 2 itself (the ladder puts it before
this one; its foundations are D31).

## D33 — Experiment 2 implemented; one pipeline script per experiment (2026-07-25, Jesse)

Experiment 2 (experiments.pdf §6.5) is written and runs end-to-end, and the three experiments
now have separate configs, trainers, evaluations and Isambard pipelines. Experiment 1 remains
**bit-identical to commit 3fbcdfd** — re-verified after this change: same weight SHA-256 after a
full CLI run, same resolved config, same metrics to full precision.

| | Exp 1 | Exp 2 | Exp 3 |
|---|---|---|---|
| predictor input | true Z_{0:t} | frames X_{0:t} | frames X_{0:t} |
| target | true Z_{t+1} | true Z_{t+1} | sg(EMA visual), Eq. 114 |
| SPARTAN state / out | 4 / 4 | d_s=32 / 4 (Eq. 95) | d_s=32 / d_s=32 (Eq. 118) |
| train alignment | none (zeta = id) | Hungarian, Eqs. 98-100 | none (EMA row ancestry, Eq. 116) |
| constraint | Eq. 13 raw | Eq. 103 raw | Eq. 123 variance-normalized |
| target encoder / EMA | no | **no** | yes |
| collapse regularizer | no | **no** | no (monitored, Eq. 124) |
| launch | `isambard_pipeline.sbatch` | `isambard_exp2_pipeline.sbatch` | `isambard_exp3_pipeline.sbatch` |

**Experiment 2 is NOT just "Experiment 3 without the EMA".** Two further differences follow
from the fixed target and are easy to get wrong:

1. **SPARTAN decodes into the RAW 4-dim state** (Eq. 95) from a latent d_s input, so input and
   output inhabit different spaces — which is exactly why §6.5 excludes open-loop rollout from
   the primary protocol. Experiment 3's head is type-closed (Eq. 118).
2. **A trajectory-level assignment is required** (Eqs. 98-100): predictions come out in
   visual-track order, targets in simulator-row order. Experiment 3 needs none, because EMA row
   ancestry aligns its branches by construction and §6.6 forbids matching there.

Because the target is fixed raw data there is no target encoder, no EMA teacher, no learned
target geometry and no representation-collapse regularizer (§6.5, verbatim): a constant context
state cannot predict episode-varying future states, so collapse is not a trivial optimum.
`health/latent_std` is logged as a cheap tell, not as an objective.

**Two alignments in Experiment 2's evaluation, deliberately.** `pred_loss`/`constraint_loss` use
the TRAINING assignment (Eq. 99), because tau_2 is the held-out constraint of the dense reference
and must be the same quantity the dual sees. `mcc`/`shd` use the EVALUATION assignment
(Eqs. 137-138, geometric slot-centroid matching), which is blind to learned parameters, true
masses and prediction quality — the guarantee §6.7 demands and the prediction-error assignment
cannot give. Their disagreement is logged as `assignment_disagreement`: a tracking tell, not a
metric.

**sigma_a lives on the model, not the trainer.** Eq. 98's frozen training-split scales are a
buffer in `VisualToStateModel`, so they serialize into the checkpoint. A standalone evaluation that
recomputed or defaulted them would select a different assignment and report a constraint tau was
never calibrated against.

**Track keys are resampled every forward** (§6.4: "an independently sampled episode-level
permutation"), so two passes over identical input legitimately differ. Tests that need
determinism seed the RNG; do not mistake this for nondeterminism in the encoder.

**Still open for both visual experiments:** a fresh label-free lambda_logit sweep (1e-5 is
inherited from D30, defensible because the logit penalty acts on SPARTAN's own attention logits,
but not calibrated); §6.7's parameter interventions; the frozen held-out state probe (Eq. 104)
and the "incremental information beyond the state branch" check; and the §6.4 readiness suite.
The ladder's gating still applies — Experiment 2 must pass its continuation gate before
Experiment 3 is interpreted.
