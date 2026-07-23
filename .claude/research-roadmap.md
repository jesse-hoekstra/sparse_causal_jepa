# Research goals and staged experimental roadmap

**Status:** active roadmap, 2026-07-22. The `lambda_logit` sweep is currently running.

This file records the research goal, the experimental ladder, the success criteria for each
stage, and the implementation work needed to move between stages. It is a plan, not a replacement
for settled design decisions. If this file conflicts with [`docs/decisions.md`](../docs/decisions.md),
the latest non-superseded decision there wins. In particular, D25 supersedes D24's paired mass
slots, same-index attention bias, and explicit 10k-step sparsity warm-up.

## Central research question

Can a JEPA recover time-invariant physical parameters such as object masses while SPARTAN prunes
its predictor to a compact, meaningful causal dependency graph, and does that result survive when
we progressively remove access to ground-truth state representations?

The intended progression is:

1. Establish parameter recovery and genuine pruning with ground-truth context and future states.
2. Validate the permutation-safe tracking, matching, and parameter-coordinate contract on
   ground-truth trajectories before introducing pixels.
3. Replace every teacher-forced source state with a causal encoding of its source frame while
   retaining raw future states as an externally defined physical-space target.
4. Replace the raw future-state target with a learned encoding of the true future states, while
   preserving stable object correspondence and preventing representational collapse.
5. Reach the ultimate experiment: the context branch receives frames and the target encoder
   receives only the single future frame for each transition. Ground-truth states remain available
   solely for evaluation, not as model inputs or training targets.

The dense Transformer may already recover masses. That is not a failure of the research program.
The sparse model is interesting if it retains prediction and parameter recovery while discovering
a materially smaller, interpretable graph. A modest, consistent MCC advantage over the dense
model would strengthen the result, but a dramatic MCC gap is not required.

## Terms that must not be conflated

| Name | Meaning | What it establishes |
|---|---|---|
| `train.lambda_logit` | Fixed coefficient multiplying Baumgartner's attention-logit penalty. This is what the current dense sweep selects. | Controls logit magnitude without materially damaging prediction. |
| `sparsity/lambda` | Adaptive GECO dual variable. The path-pressure weight is `1 / lambda`. | Shows whether the constraint controller is moving into or out of the pruning phase. |
| `sparsity/active` | Boolean indicating that the path penalty and GECO update are enabled. With warm-up zero it is `1` from step zero in the sparse main run. | Only says the pruning mechanism is switched on; it does **not** prove that pruning happened. |
| `loss/sparsity` | Sum of multilayer path multiplicities into decoded state rows. | The raw path objective; it is not a normalized edge density. |
| `path_density` | Fraction of source-token to decoded-state-output pairs with at least one path. | Primary, interpretable pruning curve. Identity is `0.1` and dense is `1.0` for the current ten-token system. |
| `path_density_full` | Reachability density over all token-output rows, including parameter-token outputs that are not decoded. | Diagnostic only; it is not the pruning objective or primary graph metric. |

## Experimental ladder

| Stage | Context input | Prediction target | Main question | Exit gate |
|---|---|---|---|---|
| 0. Logit sweep | Raw state trajectories | Raw future states | Which label-free `lambda_logit` controls logits without damaging dense prediction? | A coefficient is selected by the fixed Pareto rule and frozen. |
| 1. True-state pipeline | Raw state trajectories | Raw future states | Can the gated model meet the dense constraint, prune, retain the correct parameter paths, and recover masses? | A valid pilot followed by reproducible paired-seed evidence. |
| 1.5. Permutation-safe state bridge | Consistently permuted GT tracks | Raw future states | Do the proposed anonymous-track architecture and trajectory matcher work before perception is added? | Stage-1 behavior survives the new coordinate/matching contract. |
| 2. Visual source-state bridge | Initial context frames for parameters; causal source frame for each transition | Raw future states | Can visual slots replace every teacher-forced source state while the target remains a fixed physical ruler? | Stable trajectory tracking/matching, prediction, pruning, and mass use. |
| 3. Jointly learned state target | The same causal source-frame contract as Stage 2 | Simultaneously learned encoding of true future states | Can the result survive a jointly moving target geometry without collapse, identity switching, or target-space shortcuts? | Joint target learning remains stable and comparable while prediction, pruning, and mass recovery pass. |
| 4. Fully visual endpoint | Causal context/source frames | Jointly learned encoding of only the single future frame | Can prediction, pruning, and mass discovery survive with no state input or state target during training? | Stable visual slot correspondence, non-collapsed joint learning, pruning, and mass use across seeds. |

## Rules shared by every stage

1. **Change one source of supervision at a time.** Stage 1.5 introduces anonymous-track matching
   and any required permutation-safe parameter architecture while inputs are still states. Stage 2
   changes source-state perception; Stage 3 introduces a jointly learned target while retaining
   true future states as its observation; Stage 4 changes the target observation to the single
   future frame. Skipping directly to Stage 4 would mix perception, tracking, target
   representation, matching, and causal discovery into one failure mode.
2. **Select hyperparameters without mass labels.** Ground-truth masses are for final diagnostics,
   never for choosing `lambda_logit`, tau, checkpoints, or seeds.
3. **Recalibrate after architectural changes.** Tau is defined by a fresh, converged dense
   reference under the exact data, model, loss, matching, and target representation used by that
   stage. A numeric tau must not be reused across Stages 1, 1.5, 2, 3, or 4.
4. **Treat `lambda_logit` as architecture-dependent.** Freeze the current sweep's choice for the
   Stage-1 pipeline and seed study. Revalidate or repeat the label-free sweep when visual encoders,
   token dimensions, matching, or target geometry change.
5. **Use matched references.** Each comparison needs the same training budget and split for dense
   `A=1`, identity `A=0`, and gated SPARTAN. The dense baseline is the same predictor with standard
   fully connected attention, not an unrelated model.
6. **Keep the final test split untouched.** Sweep selection, tau calibration, and development use
   validation splits. The common final split is evaluated only after the choices are frozen.
7. **Use paired seeds.** Dense, identity, and sparse results must share the corresponding training
   seed, data seed, evaluation split, and model/data budget. Report the per-seed pairs as well as
   summaries.
8. **Do not equate encoding with causal use.** MCC shows that the learned parameter representation
   contains mass information. Graph alignment and parameter-intervention tests are needed to show
   that the predictor actually uses it.
9. **A pilot is engineering evidence, not the paper result.** Seed 0 is the development/sweep and
   pilot seed. Any protocol change made after inspecting it is frozen before eight fresh
   confirmatory seeds, proposed as seeds 1-8. Scientific summaries include every completed
   confirmatory run, including valid optimization failures.

## Stage 0: select `lambda_logit`

### Goal

Choose the smallest non-zero attention-logit coefficient that substantially controls excessive
attention logits while preserving the predictive performance of the dense Transformer. This is a
feasibility screen for gate plasticity; it cannot establish pruning because dense attention has no
learned gates to prune.

### Current selection rule

The current scripts already encode the rule:

1. Include an exact `lambda_logit = 0` control.
2. Train every candidate as the dense `A=1` model with sparsity disabled under identical
   provenance.
3. Reject candidates whose held-out prediction loss is more than 5% worse than the zero control.
4. Compute the excess logit penalty above its theoretical minimum of `2`.
5. On the prediction/logit Pareto frontier, select the smallest non-zero coefficient achieving
   90% of the best admissible reduction in excess logit penalty.
6. Display `mass_mcc` only as a diagnostic. Do not use it in the selection rule.
7. Save the selected value and the provenance subset checked by `sweep_summary.json`, then freeze
   it for all Stage-1 references, sparse runs, and seeds. The summary does not contain every
   simulator/loss field, so manually verify the complete `resolved_config.yaml` files and a clean
   git SHA before accepting the sweep.

Relevant implementation:
[`isambard_logit_sweep.sbatch`](../scripts/isambard_logit_sweep.sbatch) and
[`summarize_logit_sweep.py`](../scripts/summarize_logit_sweep.py).

### Exit and failure conditions

- **Pass:** one non-zero candidate satisfies the fixed selection rule; the summary's checked
  provenance agrees; and manual comparison of the complete resolved configs confirms the same
  clean git SHA, simulator/data settings, model, objective, budget, and validation split.
- **Fail:** no non-zero candidate stays within the prediction tolerance or improves logit excess.
  Do not rescue the sweep by selecting the candidate with the best MCC.
- **Not yet known:** whether the selected value preserves gate plasticity. Only the Stage-1 gated
  run can answer that.

## Stage 1: full true-state pipeline

### Scientific goal

Demonstrate that the current D25 system can recover the five masses and discover a sparse
parameter-to-state dependency structure while retaining the predictive performance required by
the fully connected reference.

### Exact current system

- Five balls with raw state `[x, y, vx, vy]` used for the predictor's state input and target.
- A 30-step context is used to infer one persistent five-coordinate scalar parameter vector.
- `GlobalLatentAttnPooling` produces five anonymous global coordinates. One dataset-level
  permutation is permitted during held-out mass evaluation; per-episode rematching is forbidden.
- SPARTAN has five state tokens, five parameter tokens, three layers, and no auxiliary tokens.
- Prediction is aligned raw MSE in a fixed physical coordinate system.
- Training uses 30 teacher-forced one-step transitions (`rollout_horizon = 1`).
- The sparse run has no explicit warm-up. `sparsity/active` is on from step zero, while the initial
  GECO value `lambda = 1e6` makes the initial path weight only `1e-6`.
- The current sweep's selected `lambda_logit` is fixed before this pipeline begins.

The operative config is
[`bounce_baumgartner.yaml`](../configs/experiment/bounce_baumgartner.yaml), with D25 as the latest
protocol decision.

### Execution sequence for each seed

1. **Dense reference:** train the deterministic `A=1` model for the full budget with path sparsity
   disabled. The relevant held-out quantity is the configured prediction term plus
   `lambda_logit * logit_penalty`, not bare MSE.
2. **Identity reference:** train the deterministic `A=0` model under the same budget.
3. **Pre-main calibration and feasibility gate:** evaluate dense and identity on the same large
   validation set with paired per-episode constraints. Set tau to `1.0` times the large-set dense
   mean and require the 95% paired interval for
   `identity_constraint - dense_constraint` to lie above zero. If not, do not launch sparse.
4. **Sparse main run:** only after that gate passes, train the gated model with the fixed
   `lambda_logit`, the newly measured tau, and `sparsity_enabled = true`.
5. **Common final evaluation:** after all choices are frozen, evaluate dense, identity, and sparse
   checkpoints on the same 5,000-episode test split. The current evaluator overwrites
   `metrics.json`, `recovery_alignment.json`, and `recovery_grid.png`; add split-suffixed output
   artifacts or archive the validation artifacts before testing references. Never overwrite the
   tau/identity evidence without retaining it.
6. **Pilot first:** inspect the early validation points of seed 0 before spending the full
   multi-seed budget. Seed 0 remains development evidence if its results affect thresholds,
   schedules, architecture, or protocol. After the protocol is frozen, launch eight fresh
   confirmatory seeds, proposed as 1-8, with the same selected `lambda_logit`. Each seed still
   receives its own fresh dense tau and identity feasibility check.

The current runner estimates tau and the identity floor from only 256 episodes, accepts any strict
point-estimate gap, and immediately starts `main`. That is acceptable for the seed-0 development
pilot but not the confirmatory protocol above. Split or update the runner before seeds 1-8 so the
large-validation dense mean is the tau actually used by sparse training; a larger estimate computed
afterwards cannot retroactively validate a run.

The existing eight-task launcher uses seeds 0-7. If seed 0 remains the inspected development pilot,
update the confirmatory launcher to seeds 1-8 before submission.

An optional fourth control is the **gated model with `sparsity_enabled = false`**. It isolates the
effect of the path penalty, but it is not the dense Transformer: its Bernoulli gates still sample
and can affect prediction. Keep the names “dense `A=1`,” “identity `A=0`,” “gated without path
pressure,” and “gated sparse” distinct in plots and tables.

Typical current launch:

```bash
SWEEP=outputs/lambda_logit_sweep_logit_seed0/sweep_summary.json
LAMBDA_LOGIT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_lambda_logit"])' "$SWEEP")
sbatch --account=<PROJECT> scripts/isambard_pipeline.sbatch full_seed0 "$LAMBDA_LOGIT" 0
```

### What counts as actual pruning

`sparsity/active = 1` is necessary but not evidence. A successful pruning trajectory requires all
of the following:

1. The GECO controller's smoothed **training** constraint crosses tau, while the read-only held-out
   `eval/constraint_loss` approaches and remains near tau.
2. `sparsity/lambda` leaves its `1e6` ceiling in the downward direction, increasing `1/lambda`.
3. Held-out `eval/path_density` shows a sustained decline over at least three evaluation points,
   rather than a one-batch fluctuation.
4. Final density is materially below the dense value `1.0` but strictly above the identity-only
   value `0.1`.
5. The model retains parameter-to-state paths and improves `shd_param_aligned`; density reduction
   without structural improvement is indiscriminate pruning.
6. The final constraint remains near tau and prediction remains competitive with dense. Pruning
   to identity while losing prediction or MCC is empty-graph collapse, not success.

Use decoded-row `path_density` as the primary density. `path_density_full` includes paths ending
at parameter outputs that are never decoded and is only a diagnostic.

### What counts as parameter identification and use

Identification requires more than one high average number:

1. Evaluate the strict one-to-one global recovery on held-out episodes with the assignment chosen
   on a separate alignment fold and frozen for the test fold.
2. Inspect all 25 cells in `recovery_grid.png`. Five distinct assigned relationships should be
   present; one coordinate explaining several masses is not five-parameter identification.
3. Report `mass_mcc` and the five assigned held-out nonlinear R2 values, not only a training or
   pooled diagnostic.
4. Align parameter graph columns using that same frozen permutation and report
   `shd_param_aligned`.
5. Add a causal-use diagnostic before making the strongest claim: shuffle learned parameter
   coordinates across episodes, replace them by their training mean, or ablate retained
   parameter-to-state paths at evaluation. Prediction, especially on collision-sensitive
   transitions, should become worse. Until this is implemented, phrase the result as parameter
   information plus aligned graph recovery rather than proven intervention-level use.

### Proposed preregistered Stage-1 gates

These quantitative targets should be confirmed before inspecting the eight-seed outcome and then
kept fixed:

- The final analysis contains all 8 scheduled confirmatory seed outcomes. Incomplete, corrupt, or
  provenance-mismatched infrastructure jobs are rerun; completed optimization failures remain in
  the denominator. At least 6 of 8 sparse seeds show three or more consecutive post-crossing
  density reductions and finish with the last-three-evaluation median in `(0.10, 0.90]`.
- The ratio of the arithmetic mean sparse test prediction loss to the arithmetic mean paired dense
  test prediction loss is at most `1.05`.
- At least 6 of 8 sparse seeds have a last-three-validation median
  `constraint_loss <= 1.05 * tau`.
- Median sparse `mass_mcc` is at least `0.80`, with at least 6 of 8 seeds at or above `0.70`.
- The 95% paired interval for the arithmetic-mean difference
  `MCC_sparse - MCC_identity` lies above zero.
- Mean sparse `shd_param_aligned` is lower than both paired dense and identity means; report the
  paired differences and intervals rather than only the endpoint values.
- The strongest outcome is a positive paired arithmetic-mean
  `MCC_sparse - MCC_dense` difference whose 95% interval excludes zero.
- A core sparse-structure result is still supported if the sparse-dense MCC interval overlaps
  zero, provided `mean(MCC_sparse) / mean(MCC_dense) >= 0.95`, prediction is retained, the density
  gate above passes, and aligned graph structure improves.

For all paired intervals above, use a predeclared paired percentile bootstrap over the eight seed
pairs with 10,000 resamples and a fixed analysis seed, and display all eight raw paired differences
because the sample is small. The existing aggregation script reports marginal summaries but not
paired differences, intervals, or complete provenance. Add this analysis before the final seeded
report.

### Outcome interpretation

| Observation | Interpretation |
|---|---|
| Dense and sparse both identify masses; sparse prunes and has similar MCC. | Successful structural result: sparsity preserves identification and produces an interpretable graph. |
| Sparse has a small, consistent paired MCC advantage. | Strongest intended result; sparsity also improves or stabilizes parameter recovery. |
| Dense identifies masses; sparse MCC is materially worse after pruning. | Pruning did not preserve identification, even if density decreased. |
| MCC is high but parameter edges/ablation have no effect. | Mass is encoded but not shown to be used by the predictor. |
| Density stays near `1.0`, lambda stays capped, and constraint stays above tau. | Pruning never engaged; diagnose feasibility or gate commitment. |
| Density reaches `0.1` and MCC/prediction approach identity. | Empty-graph collapse. |
| Density falls while `shd_param_aligned` worsens. | Indiscriminate rather than causal pruning. |
| Tau is not below identity. | Invalid experiment: no feasible constraint can force parameter use. |

### Validity checks

Reserve **invalid/rerun** for external interruption, corrupt or missing artifacts, wrong/stale data,
inconsistent provenance, or execution that did not run the frozen protocol. A reproducible
non-finite trajectory, zombie/frozen optimizer, sustained gradient-skip episode, memorization gap,
or other optimization pathology under the frozen protocol is a **valid failed outcome** and stays
in the confirmatory denominator; do not rerun it until a favorable seed appears. A protocol change
made to address such a failure starts a new experiment lineage. Periodic MCC from small evaluation
batches is a noisy trend metric; final claims use the common 5,000-episode evaluation.

## Stage 1.5: permutation-safe state and matching bridge

### Why this stage is required

D25 is valid for ordered simulator tracks: its source-track embeddings, persistent state-node
addresses, and five global parameter coordinates live in a stable coordinate system. Visual slots
are recurrent within an episode but generally have no dataset-global object order. Replacing states
with frames while simultaneously changing parameter pooling, node addresses, slot count, and the
loss matcher would violate the one-factor-per-stage rule.

Stage 1.5 keeps raw state trajectories but hides their absolute row order. Apply one random object
permutation consistently to the **model's source-track axis** across the complete trajectory while
keeping prediction targets in canonical simulator order, or independently permute the target axis.
Retain the known source-target mapping only for evaluation. Applying the same permutation to both
inputs and targets would leave rows aligned and make Hungarian matching a trivial identity test.
Train with the same trajectory-level matcher intended for Stage 2. This isolates the
coordinate/matching problem before perception.

### Coordinate contract to freeze

The default Stage-1.5 bridge uses **track-attached parameters**: emit one scalar parameter from each
recurrent input track with permutation-equivariant pooling and no same-index attention bias. This
predefines which parameter token travels with which state track, even though SPARTAN must still
decide whether to retain its path. The claim is parameter-channel use and pruning for anonymous
tracks, not discovery of a global mass-to-object binding.

A purely permutation-invariant set encoder cannot keep five global scalars associated with five
arbitrarily ordered objects: at best it can represent an unordered mass set, while a state token
has no association key identifying its scalar. Therefore the visual stages must not continue D25's
dataset-global parameter coordinates. Attach one scalar to each inferred track and give its state
and parameter tokens a shared ephemeral track key that permutes with the track. The key carries no
physical or renderer-derived information; it only preserves the state-parameter association.
SPARTAN must still learn and prune the causal paths that use those parameters.

Under the default track-attached route, the D25 evaluation harness must change: after the
per-trajectory object assignment, evaluate matched
episode-object pairs shaped approximately `(episodes * objects, 1)` rather than applying the
current `(episodes, 5)` global-coordinate MCC. Apply the same episode permutation to state
destination rows, state-source columns, and parameter-source columns before graph SHD.

### Stage-1.5 exit gate

- Trajectory matching is invariant to one whole-trajectory permutation and penalizes identity
  switches within a trajectory.
- The track-attached contract and shared permutation-safe association key are named in configs,
  artifacts, and claims.
- The exact configured `constraint_loss`, not bare MSE, still separates dense from identity.
- Prediction, pruning, graph alignment, and mass recovery remain within the frozen Stage-1
  tolerances on the development run.
- Only after this bridge passes is perception introduced.

## Stage 2: causal source frames to raw future states

### Why this stage exists

This stage introduces perception while leaving the target in raw physical coordinates and keeping
the one-step teacher-forced objective. It inherits the coordinate and trajectory-matching contract
already validated in Stage 1.5. The current visual `SCJepa` does not implement this bridge: it
already predicts slots from a separate learned future-frame encoder and applies matching after
flattening transitions.

“Context frames” must be understood precisely here. Stage 1 obtains 30 one-step source anchors
from true states at times 29 through 58. The direct visual analogue is **image teacher forcing**:
run the recurrent encoder causally over `frames[:, :-1]`, pool the time-invariant parameters only
from the first `T_h=30` slots, and use the recurrent slot at each later source time to predict that
transition's raw next state. Thus later source frames are inputs for later one-step transitions;
the model is not rolling out 30 steps from only the last context frame. A context-only
autoregressive rollout is a separate experiment with a new tau and must not be silently mixed into
this stage.

### Required model contract

Introduce a hybrid visual-state model with the following data flow:

```text
causal source frames[:, :-1]
    -> recurrent visual slots (B, L-1, N, d)
    -> source slots at times T_h-1 ... L-2 (B, K, N, d)
    -> physically grounded state head (B, K, N, 4)

first T_h recurrent slots only
    -> parameter pooling once (shape depends on the Stage-1.5 coordinate contract)

grounded source states + pooled parameters
    -> SPARTAN in raw state space
    -> predicted [x, y, vx, vy]

raw simulator future states
    -> unchanged fixed targets (B, K, 5, 4)
```

The target branch has no learned encoder in Stage 2. Raw-state MSE remains the fixed prediction
ruler, so target-scale collapse is impossible. The full feasibility constraint still includes the
weighted logit term.

### Ground the visual state channel before making causal claims

A four-dimensional head is not automatically `[x, y, vx, vy]`. If it is trained only through
next-state MSE, it can emit an arbitrary four-dimensional code and can hide mass in the state
channel. That would let prediction succeed while bypassing the intended parameter path.

Use two explicitly labelled substages:

1. **Stage 2A — grounded visual state.** Train the visual state head against the true current
   `[x, y, vx, vy]` using the fixed trajectory assignment. Preserve grounding either by retaining
   that anchoring loss throughout predictor training or by freezing the complete visual
   encoder-to-state-head path. Freezing only the linear head is insufficient because a moving
   visual encoder can change what its inputs mean. At inference the model receives frames, but this
   bridge still uses state supervision during training. This is the first valid test of SPARTAN
   operating in raw state space from visual inputs.
2. **Stage 2B — ungrounded visual anchor, optional.** Remove direct current-state supervision only
   after 2A succeeds. Call the input a learned visual state code rather than raw state. Measure how
   well mass can be decoded from that code, and do not claim parameter-channel discovery unless
   leakage is ruled out and parameter interventions show separate use.

### Visual slots, parameters, and graph identity

Ground-truth state rows have persistent simulator identity; visual slots do not. Recurrent slot
continuation must provide stable within-trajectory tracks, and Stage 2 must reuse exactly the
track-attached contract selected in Stage 1.5. Do not reintroduce D25's fixed source-track position
embeddings or state-node identities for anonymous visual slots.

Start with exactly five recurrent visual object slots if the simple renderer permits it. Align each
predicted track to one true object once per trajectory and use that alignment for the grounded
state-head loss, future-state loss, parameter recovery, and graph evaluation.

If additional background/null slots are required, define an explicit dustbin or unmatched-slot
contract before training. Do not silently match five of seven slots differently at each frame.

State explicitly that the track-to-parameter binding is architectural rather than discovered. The
values of the parameters and the graph paths that use them remain learned; the shared track key
only keeps each learned parameter attached to the same inferred object trajectory.

### Trajectory-level Hungarian matching

Independent per-frame Hungarian matching can hide identity switches. Instead, for each trajectory,
restore or retain model outputs as `(B, K, N, d)` before loss computation—the current flattened
`(B * K, N, d)` representation is insufficient—and form one detached cost matrix over all `K`
predicted transitions:

```text
C[i, j] = mean over time and standardized state dimensions
          of squared_error(predicted_track_i, true_object_j)
```

Solve Hungarian once, then hold that assignment fixed for every timestep in the trajectory. The
matching decision is detached; gradients flow through the selected prediction-target pairs.
Position and velocity dimensions must be standardized or explicitly weighted so that one block
does not dominate the assignment.

Use the same trajectory assignment to:

- compute raw-state prediction loss;
- align graph destination-state rows and source-state columns;
- for the track-attached contract, also align parameter-source columns and associate the parameter
  outputs with true object masses;
- diagnose per-frame assignment switches without permitting them in the loss.

### Stage-2 implementation subgoals

1. Complete Stage 1.5 and freeze its coordinate and matching contract.
2. Add the hybrid image-teacher-forced frames-to-raw-states model and config without changing
   Stage 1.
3. Ensure the recurrent visual encoder is causal: parameters use only the first `T_h` frames; the
   anchor for transition `t -> t+1` may use source frame `t` but not target frame `t+1`; and a
   later-frame perturbation cannot affect earlier slots or pooled parameters.
4. Complete Stage 2A by retaining current-state supervision throughout training or freezing the
   complete visual encoder-to-state-head path. Report current-state error and a mass probe on this
   state channel.
5. Implement trajectory-level matching as a separate prediction-matching mode, preserving the
   `(B, K, N, d)` trajectory axis until one assignment has been chosen.
6. Add tracking diagnostics: object coverage, trajectory assignment cost, per-frame switch rate,
   position error, and velocity error.
7. Implement the evaluation contract chosen in Stage 1.5. In particular, do not feed
   track-attached `(episodes * objects, 1)` outputs into the D25 global `(episodes, 5)` MCC
   harness.
8. Demonstrate a small dense model can learn frame-to-raw-state prediction before enabling gates.
9. Run matched dense and identity references. Revalidate `lambda_logit`, recalibrate tau from the
   exact `constraint_loss`, and only then run sparse training.
10. Add parameter shuffle/mean and graph-edge ablations on the final test split.
11. Repeat the paired-seed protocol only after a pilot passes all representation and mechanism
    gates. Attempt optional Stage 2B only after Stage 2A passes.

### Mass visibility control

The current Baumgartner-aligned state experiment already makes physical radius proportional to
mass with an episode-independent reference. This removes the common mass-scale symmetry: changing
absolute masses changes collision and wall-contact geometry. Stage 1 has `render: false` and its
raw `[x,y,vx,vy]` inputs do not include radius, so this geometry is latent rather than an input
shortcut.

Before Stage 2, split physical and rendered radius settings and pin them explicitly in every
primary visual config:

```yaml
data:
  physics_radius_from_mass: true
  render_radius_from_mass: false
  object_appearance: uniform
```

Preserve mass-proportional physical radii but render every object with the same mass-independent
glyph, so absolute mass remains dynamically observable without being read directly from
appearance. Heterogeneous-appearance controls must randomize their mapping to simulator rows across
episodes rather than create a dataset-global identity code. An equal-physical-radius variant is a
useful harder control, but absent another absolute reference it supports recovery of normalized
mass ratios rather than absolute masses; it is not the contract of the current Stage-1 experiment.

### Stage-2 exit gate

Stage 2 passes only if:

- dense `constraint_loss` is below identity `constraint_loss` with the predeclared uncertainty
  margin; bare raw-state MSE alone is not the feasibility gate;
- recurrent slot tracks are stable under the one-assignment-per-trajectory diagnostic;
- the Stage-2A state head remains physically grounded, or an optional Stage-2B result is explicitly
  labelled as a learned-code experiment with state-channel mass leakage measured;
- the image-teacher-forced model predicts raw future states without target-frame leakage;
- the sparse model reaches its freshly calibrated tau and genuinely prunes without collapsing to
  identity;
- mass recovery and all relevant graph axes are aligned under the Stage-1.5 contract and remain
  reproducible;
- parameter intervention degrades relevant predictions;
- the final scientific version succeeds with mass appearance controlled, not only with visibly
  mass-coded radii.

## Stage 3: causal source frames to a learned encoding of true future states

### Scientific goal

Replace direct regression to raw future states with prediction in a learned state representation,
while retaining Stage 2's exact causal source-frame contract. This tests whether sparse parameter
discovery survives a learned target geometry. The target encoder still receives true future
states, so this stage is not yet a fully pixel-only JEPA.

### Simultaneous-learning contract

Stage 3 does **not** freeze or pretrain the target encoder. The following modules learn
simultaneously from scratch under one optimizer:

- the recurrent visual context encoder;
- the grounded visual kinematic path and context-to-latent state mapper;
- the time-invariant parameter encoder;
- SPARTAN;
- the target-state encoder.

There is no EMA teacher and no stop-gradient on the target representation. The detached Hungarian
operation selects a trajectory assignment, but the matched predictive loss sends gradients to both
the prediction branch and the target encoder. VISReg/variance-covariance regularization acts on the
learned representations to prevent collapse.

### Target and source contracts

Use a shared-across-objects target encoder with its own learned weights:

```text
g_target: true [x, y, vx, vy] at t+1 -> target z at t+1
```

Apply it independently to each true object and future timestep. It must receive only the
instantaneous true future state—not mass, object identity labels, or the full future trajectory—so
it cannot directly place the time-invariant parameter into the target slot.

Keep the context and target representation maps separate, as in the intended JEPA. To prevent the
recurrent context slot from hiding mass in the state channel, retain Stage 2A's grounded
four-dimensional bottleneck:

```text
source frame history -> recurrent slot -> grounded current [x, y, vx, vy]
grounded current state -> learned context mapper -> source z
true next state -> learned target encoder -> target z
(source z, learned parameters) -> SPARTAN -> predicted target z
```

The current-state grounding loss remains active throughout joint Stage-3 training so the visual
path can keep learning without losing its physical meaning. The context mapper and target encoder
also move, but their inputs are restricted to kinematic state; parameter information must enter
SPARTAN through the separate parameter tokens rather than through an unconstrained recurrent state
code.

### Joint objective

For one complete trajectory assignment, optimize all trainable modules jointly with:

```text
predictive loss(predicted z, target z)
+ representation regularization(context representation, target z)
+ lambda_logit * attention-logit penalty
+ (1 / lambda_GECO) * SPARTAN path count when sparsity is active
+ persistent current-state grounding loss for the visual kinematic bottleneck
```

The anti-collapse and grounding terms remain outside the GECO prediction constraint. The target
encoder is learned through the matched predictive and representation-regularization gradients; it
is not trained with mass labels.

### Moving-target calibration risk

Dense, identity, and sparse runs each jointly learn their own target encoder from the same
initialization rule, data contract, objective, and matched seed. Their learned target spaces can
still differ. Detached target-variance normalization removes a scalar rescaling shortcut but does
not guarantee identical nonlinear geometry or prediction difficulty.

This is a validity risk to measure, not a reason to freeze the target encoder. The Stage-3 protocol
therefore requires:

1. the existing learned-target normalized prediction constraint plus the weighted logit term;
2. identical target architecture, initialization seed, regularizer, optimizer, data, and budget in
   the dense, identity, and sparse legs;
3. continuous target-collapse and geometry diagnostics: per-dimension standard deviation,
   covariance spectrum/effective rank, norm, pairwise slot distance, and drift over training;
4. a held-out probe from target `z` back to raw state, evaluated identically for dense, identity,
   and sparse, to show that target difficulty has not been made trivial in one leg;
5. a predeclared comparability gate on those diagnostics before interpreting dense tau as a valid
   sparse constraint.

If dense and sparse learn materially different target information or geometry, report Stage 3 as a
confounded/negative result and revisit calibration. Do not silently repair it with a frozen target,
EMA, stop-gradient, reconstruction loss, or post-hoc target selection; each would define a
different method.

### Matching in learned target space

Predicted visual tracks remain anonymous relative to the true-state target rows. Use one Hungarian
assignment per complete target trajectory, now based on standardized latent distance, and hold it
fixed across time. As in Stage 2, retain or restore `(B, K, N, d_z)` before computing the assignment.
The assignment computation is detached, while the matched prediction loss updates both branches.

The current per-transition Hungarian loss is insufficient because it permits a different slot
permutation at every future time. Add an explicit `trajectory_hungarian` mode and retain the Stage-2
tests that penalize a mid-trajectory identity switch.

### Stage-3 implementation subgoals

1. Implement the jointly trained per-object future-state target encoder.
2. Retain the physically grounded source-state bottleneck and add the trainable context-to-`z`
   mapper without exposing recurrent slot features directly to SPARTAN's state channel.
3. Keep target outputs and predictions shaped `(B, K, N, d_z)` until trajectory matching is
   resolved.
4. Add target collapse, geometry, drift, and raw-state information probes for every reference and
   sparse leg.
5. Define the numeric target-space comparability gate on development data before confirmatory
   seeds.
6. Confirm the parameter encoder sees only the first `T_h` context frames and the target encoder
   sees only instantaneous true future states.
7. Revalidate `lambda_logit` and recalibrate tau under the exact jointly learned Stage-3 setup.
8. Run matched dense, identity, and sparse development legs, followed by the same paired-seed and
   parameter-intervention protocol used in Stages 1 and 2.

### Stage-3 exit gate

Stage 3 passes only if:

- target representations remain non-collapsed and retain future-state information;
- the trajectory assignment is stable and cannot hide slot switches;
- dense `constraint_loss` remains below identity `constraint_loss` under the jointly learned-target
  protocol and predeclared uncertainty gate;
- dense, identity, and sparse target spaces pass the predeclared information/geometry comparability
  checks;
- sparse reaches the freshly calibrated constraint and prunes to a non-identity graph;
- masses remain recoverable and are used through retained parameter paths;
- results reproduce across paired seeds;
- all reports state explicitly that the target encoder was trained simultaneously from scratch.

## Stage 4: fully visual endpoint with a single-future-frame target

### Ultimate scientific goal

Stage 3 is the immediately preceding bridge because its target encoder still observes true future
states. The ultimate experiment removes that oracle. For every transition `t -> t+1`, the target
encoder receives only the image `x[t+1]`. No true state is fed to either model branch or used in the
training loss. States, masses, and contact graphs are retained only for held-out evaluation.

This is the strongest intended result: sparse, interpretable physical-parameter use learned from
video while the JEPA predicts a separately learned visual target representation.

### Exact branch contract

```text
context branch:
    frames up to each causal source time t
        -> recurrent context slots
        -> context state slots + time-invariant parameter slots

target branch for transition t -> t+1:
    x[t+1] only, shaped (B, 1, C, H, W)
        -> separate single-frame target encoder
        -> unordered target slots at t+1

predictor:
    (context state slots, parameter slots)
        -> SPARTAN
        -> predicted target slots at t+1
```

The target encoder must not receive context frames, earlier target frames, recurrent state, true
states, masses, or object identities. It is called independently with one frame for each target
time. Encoding several target frames in a batch for efficiency is allowed; information must not
flow between those single-frame examples.

The recurrent context encoder, single-frame target encoder, parameter encoder, context heads, and
SPARTAN all train simultaneously from scratch under one optimizer. There is no pretrained or
frozen target, EMA teacher, or target stop-gradient. The predictive loss and representation
regularization update the target encoder directly.

The existing `SCJepa`/D9 single-frame target branch is the architectural starting point for this
stage, but the current per-transition matching and evaluation contracts are not sufficient for the
final trajectory-level claim.

### What changes relative to Stage 3

- Stage 3 target: `g_target(true_state[t+1])`.
- Stage 4 target: `E_target(x[t+1])`, with no access to `true_state[t+1]` during training.
- Stage 3 can use a grounded physical bottleneck to audit the context state channel.
- Stage 4 removes that state-supervised grounding from the reported method. The context state is a
  learned visual representation, so leakage of mass from recurrent context into the state channel
  becomes a central empirical risk rather than something ruled out by construction.

The Stage-3 checkpoint may be used only to debug interfaces in a clearly labelled transfer
ablation. The reported Stage-4 result trains from scratch so success cannot be attributed to prior
state supervision.

### Single-frame target correspondence

A single-frame target encoder produces an unordered set, and its slot order can change independently
at every target time. Independent per-frame Hungarian loss is a useful smoke baseline, but it can
hide both target-slot and predicted-track switches. The reported model therefore uses generic
loss-side target tracking:

1. Encode each target frame independently and retain learned slot features plus generic geometry
   from the slot assignment maps, such as centroid. Any presence/confidence statistic must come
   from pre-spatial-normalization assignment mass, logits, or entropy; a map normalized to sum to
   one for every slot is not a presence measure.
2. Associate slots across the target sequence with a detached sequential Hungarian assignment.
   The core Bounce cost uses assignment-support overlap and spatial continuity from a simple
   constant-velocity centroid prediction. A normalized learned-feature term is an optional generic
   extension for harder scenes, not a fixed renderer-identity key.
3. Match predicted context tracks to the assembled target tracks once per trajectory and hold that
   assignment fixed for the complete loss and graph alignment.

No renderer-specific appearance category, simulator label, true state, mass, or contact signal may
choose these assignments. The procedure may aggregate slots from several independently encoded
frames in the loss, but the target encoder itself still receives one frame per call. True states
and renderer masks may measure tracking quality only after the fact.

Because a learned descriptor could silently exploit a dataset-global object-index signature, the
primary Bounce run uses uniform object glyphs and a geometry-only matcher. Heterogeneous-appearance
controls must randomize mappings across episodes. This preserves the option to use general learned
visual descriptors in more complex domains without turning the Bounce renderer into the
association mechanism.

This contract is feasible when objects are localized and the observations contain enough
continuity. A genuinely ambiguous crossing of visually indistinguishable objects has no unique
label solution; in such cases report set-level parameter recovery and ambiguity-aware tracking
metrics instead of claiming recovery of arbitrary simulator identities. If the minimal matcher
fails outside ambiguous cases, the next fallback is a learned association or differentiable
transport module with an occlusion state, not a renderer-derived identity shortcut.

### Observability and channel-leak controls

The final scientific configuration must make rendered appearance independent of mass. If radius or
another single-frame cue reveals mass, both target and context state slots can encode it directly,
so parameter-token recovery no longer establishes discovery of a time-invariant factor from
dynamics. A visible-mass version remains a useful negative/control experiment, not the primary
claim.

Because the recurrent context slot has seen history, it may encode dynamics-inferred mass even when
the target frame cannot. Stage 4 therefore requires:

- a probe for mass information in the context state slots separately from the parameter slots;
- parameter shuffle, mean-replacement, and retained-edge ablations;
- evidence that removing parameter information damages collision-sensitive prediction;
- retained parameter-to-state graph paths aligned to held-out objects;
- an explicit interpretation if mass migrates into the context state channel and parameter edges
  disappear. That outcome tests—and may falsify—the intended channel split.

### Joint objective and calibration

Use Hungarian/trajectory-matched latent prediction, VISReg on both learned branches, the selected
attention-logit term, and GECO-weighted SPARTAN path sparsity. No raw-state grounding or
reconstruction term belongs in the reported Stage-4 training objective.

Re-run the label-free `lambda_logit` screen and train fresh dense, identity, and sparse references
under the exact visual target architecture. Use learned-target constraint normalization and retain
the Stage-3 target-collapse, effective-rank, drift, and cross-leg comparability checks. Tau is the
large-validation dense constraint measured before sparse training; the identity uncertainty guard
must pass under the same visual target and matching contract.

### Stage-4 implementation subgoals

1. Start from the existing separate recurrent-context and single-frame-target encoder architecture.
2. Remove all true-state tensors and state-derived auxiliary losses from the training forward/loss
   contract; keep them only in the evaluation harness.
3. Preserve the `(B, K, N, d)` target/prediction trajectory axis until target-track association and
   prediction matching are complete.
4. Implement and validate generic loss-side target tracking without simulator-state supervision.
5. Keep the target encoder strictly single-frame and verify that no information crosses target
   examples when they are batched.
6. Add separate state-channel and parameter-channel mass probes plus parameter-use interventions.
7. Pin `physics_radius_from_mass: true`, `render_radius_from_mass: false`, and exchangeable
   per-object rendering in the primary run; use visible-mass and equal-physical-radius settings only
   as separately labelled controls.
8. Revalidate `lambda_logit`, recalibrate tau, pass the identity guard, and complete a development
   dense/identity/sparse pipeline before launching confirmatory seeds.
9. Apply the same paired-seed prediction, pruning, graph, MCC, and provenance analysis as earlier
   stages.

### Stage-4 exit gate

Stage 4 passes only if:

- training uses context/source frames and one future frame per target encoder call, with no true
  state input, state target, or state-derived training loss;
- context and target encoders are separate and trained simultaneously from scratch;
- target slots are non-collapsed and their cross-time correspondence cannot be explained by
  per-frame rematching shortcuts;
- the fully visual dense constraint is below the fully visual identity constraint under the
  uncertainty gate;
- sparse reaches its fresh tau and materially prunes without collapsing to identity;
- held-out masses remain recoverable and parameter interventions demonstrate predictor use;
- the primary result holds when mass is not visible in a single frame;
- the complete outcome reproduces across the confirmatory paired seeds.

## Required tests before expensive runs

### Matching tests

- Trajectory loss is invariant to one global target permutation.
- Loss is zero when prediction equals target up to one global permutation.
- A permutation that changes halfway through a trajectory is penalized.
- Hungarian assignment is detached, while gradients reach the selected prediction pairs and, when
  intended, the target encoder.
- Known synthetic graph and parameter axes are aligned by the same frozen trajectory assignment.
- For Stage 4, independently encoded single-frame target slots can be assembled into stable target
  tracks without true states, object labels, or masses entering the assignment.
- A synthetic target-slot switch is detected by Stage-4 target tracking rather than erased by
  independent per-frame Hungarian rematching.

### Architecture and leakage tests

- Frame, raw-state, and encoded-state shapes match their declared contracts.
- The Stage-2A visual path emits dimension `4` and remains physically grounded under a persistent
  anchoring loss or a frozen complete encoder-to-head path. Stage 3 retains that bottleneck and
  learns a separate context-to-`z` mapper jointly with the target encoder.
- The reported Stage-4 path contains no true-state tensor, state-derived loss, or state-derived
  matching signal. Ground-truth states are reachable only from evaluation code.
- Parameter pooling uses only the first `T_h` context steps.
- Perturbing frames after `T_h` cannot change pooled parameters, and perturbing target frame
  `t+1` cannot change the causal source anchor used to predict it.
- Recurrent slot continuation is equivariant to one joint slot permutation.
- A deliberately inserted mid-sequence slot swap is detected.
- A Stage-4 target encoding for `x[t+1]` is unchanged when other target frames in the same batched
  computation are permuted, replaced, or removed.

### Collapse and target tests

- A constant target representation triggers the collapse sentinel.
- Stage-3 and Stage-4 target-encoder parameters receive predictive and
  representation-regularization gradients; neither target is frozen or stop-gradient.
- The Stage-3 target encoder receives only the instantaneous true future state. The Stage-4 target
  encoder receives only the instantaneous future frame. Neither can access masses, object labels,
  context history, or multiple future timesteps.
- Target effective rank, per-dimension variance, and held-out state information are reported.
- Dense, identity, and sparse use the identical jointly learned-target architecture, initialization
  rule, objective, and matched seed, and their target-space comparability diagnostics are retained.

### Integration and negative controls

- Tiny dense overfit before gated training.
- Held-out dense `constraint_loss` is below identity `constraint_loss` under the predeclared
  uncertainty gate before sparse launch.
- Sparse density decreases without violating tau.
- Shuffled parameters and mean parameters worsen relevant prediction if parameters are used.
- Randomized context frames destroy visual parameter recovery.
- Collision-rich and collision-poor transitions are reported separately.
- Visible-radius and appearance-controlled runs are distinguished.
- No dataset-global renderer signature is tied to simulator row index; the tracker passes the
  exchangeable-rendering geometry-only control.
- Every final table verifies git SHA, resolved config, simulator version, steps, data seed, and test
  offset.

## Claims ladder

The claim must match the highest completed stage:

- **Stage 1:** sparse mass/parameter discovery conditional on oracle kinematic state tracks.
- **Stage 1.5:** the same result survives the explicitly selected permutation-safe coordinate and
  trajectory-matching contract.
- **Stage 2A:** causal source frames at inference yield supervised, physically grounded state
  tracks and parameter information sufficient for future-state prediction and sparse graph
  discovery.
- **Stage 2B:** only with state-channel leakage ruled out can an ungrounded visual code support a
  stronger parameter-separation claim.
- **Stage 3:** visual predictions can be matched consistently to a simultaneously learned,
  non-collapsed encoding of true future states while retaining sparse parameter recovery.
- **Stage 4:** causal context frames predict a separately and jointly learned representation of one
  future frame, while SPARTAN prunes and uses a time-invariant mass representation, with no true
  state used as a model input, target, loss, or matching signal during training.

Across all stages, the central contribution should be phrased as progressive removal of state
inputs and state-derived training supervision while retaining sparse, interpretable parameter use.
Stage 2A still uses state supervision to ground its visual head, and Stage 3 jointly trains its
target encoder from true future states; do not call either fully self-supervised. Stage 4 is the
fully visual endpoint only if true states remain evaluation-only and do not determine training
correspondence. A large MCC advantage over dense is optional; reproducible pruning, valid
prediction, correct graph structure, and retained mass identification are not.

## Immediate execution checklist

- [ ] Let the current dense `lambda_logit` sweep finish without selecting on mass MCC.
- [ ] Inspect `sweep_summary.json`, provenance, admissibility, and the zero control.
- [ ] Freeze the selected `lambda_logit` for Stage 1.
- [ ] Run one complete dense -> identity -> sparse Stage-1 pilot.
- [ ] Check the first evaluation points for constraint motion, lambda response, density motion,
      gradients, skips, and memorization before waiting for completion.
- [ ] Add split-suffixed evaluation outputs or archive validation artifacts before evaluating a
      reference checkpoint on another split.
- [ ] Strengthen dense/identity feasibility evaluation with a common large validation set and a
      paired uncertainty interval.
- [ ] Evaluate dense, identity, and sparse on the same final 5,000-episode test split without
      overwriting calibration evidence.
- [ ] Treat seed 0 as development; if the pilot passes, freeze the protocol and launch paired
      confirmatory seeds 1-8 with fresh tau per seed.
- [ ] Add paired sparse-dense reporting and a parameter-use intervention diagnostic.
- [ ] Freeze and report the Stage-1 conclusion before implementing Stage 1.5.
- [ ] Validate the anonymous-track coordinate and trajectory-matching contract in Stage 1.5.
- [ ] Implement Stage 2A with image teacher forcing, a grounded visual state head, trajectory
      matching, split physical/rendered radii, and exchangeable rendering; attempt Stage 2B only
      afterwards.
- [ ] Implement Stage 3 with simultaneous context, target, parameter, and SPARTAN learning;
      trajectory matching; persistent source-state grounding; and target-space comparability
      diagnostics.
- [ ] Treat Stage 3 as the final state-target bridge, freeze its conclusion, and then replace the
      target observation with `x[t+1]` only.
- [ ] Implement Stage 4 with a separate jointly trained single-frame target encoder, no
      state-derived training signal, and target-track association that does not use simulator
      identities.
- [ ] Re-run the label-free logit screen and dense -> identity -> sparse calibration under the exact
      fully visual objective before the paired-seed Stage-4 study.
- [ ] Report Stage 4 as the endpoint only if the appearance-controlled mass setting passes the
      prediction, collapse, correspondence, pruning, parameter-use, and confirmatory-seed gates.
