# Adjustments required for frame-based causal physics identification

**Status:** implementation companion to `research-roadmap.md`, 2026-07-22.

## Verdict on the existing roadmap

The central objective in `research-roadmap.md` is still correct:

> Learn time-invariant physical parameters from trajectories, show that SPARTAN uses them through
> a sparse causal graph, and progressively replace true states with frames until the context
> encoder, target encoder, parameter encoder, and SPARTAN are trained jointly from visual data.

The roadmap is not yet a sufficient implementation specification for the final visual result.
The main ladder should be understood as three scientific gates:

| Gate | Model input | Prediction target | Required result |
|---|---|---|---|
| 1. True-state identification | True `[x,y,vx,vy]` histories | True next `[x,y,vx,vy]` | Recover and use the masses while pruning relative to matched dense and identity references. |
| 2. Visual context bridge | Causal frames | True next `[x,y,vx,vy]` | A frame encoder replaces the source states without losing mass recovery, parameter use, or pruning. |
| 3. Fully visual joint model | Causal frames | A jointly learned encoding of one future frame | Recover and use masses while jointly training both visual encoders and SPARTAN, with no state-derived training signal. |

The old Stage 0 remains shared setup. The old Stage 1.5 should become readiness tests for Gate 2,
not a full paper experiment. The old Stage 3, in which an encoder embeds true future states, is a
useful diagnostic if Gate 3 fails, but it need not receive a complete multi-seed dense/identity/
sparse study before the main result. Existing Stage 1, Stage 2, and Stage 4 map to Gates 1, 2, and
3 respectively.

This file records the changes needed to make those gates executable and defensible. It does not
replace the detailed Stage-1 protocol already in `research-roadmap.md`.

## Non-negotiable final claim

For the final result:

- the context branch receives frames only;
- the target/observation encoder receives exactly one image per call;
- the future-frame target call receives only `x[t+1]`;
- true states, contacts, masses, and simulator object identities are used only for held-out
  evaluation;
- the context encoder, single-frame target/observation encoder, temporal state lift, parameter
  encoder, and SPARTAN are optimized jointly;
- the primary representation has five object slots of width 32;
- SPARTAN may retain a 512-dimensional internal workspace, but predicts five 32-dimensional
  target slots;
- success requires mass information, causal use of that information, genuine graph pruning,
  valid slot correspondence, and replication across seeds.

An EMA or frozen target may be used as a labelled stability control. It is not the primary result.

## Two feasibility conditions that must be fixed first

### 1. Preserve the current absolute-mass observability without rendering it

The current `bounce_baumgartner` state experiment has already removed the equal-radius scale
symmetry. It sets `radius_from_mass: true`, with
`r_i = base_radius * m_i / mass_ref`, where `mass_ref` is fixed across episodes. Consequently,
multiplying every mass in an episode by a common factor also changes the physical collision and
wall-contact geometry. Absolute mass scale can therefore affect the trajectory, rather than only
mass ratios being identifiable. Because this experiment inherits `render: false` and its raw state
is only `[x,y,vx,vy]`, the model is not handed radius as an input shortcut.

The remaining issue arises only when frames are enabled. The current dataset passes the same
mass-dependent radii to both the simulator and renderer, so a visual model could read mass from
disc size. Preserve the existing physics contract while making rendered appearance independent of
mass: keep mass-dependent physical collision radii, but draw every object with the same radius.

The data API therefore needs separate fields such as:

```yaml
data:
  physics_radius_from_mass: true
  render_radius_from_mass: false
```

The simulator, renderer, generation metadata, cache version, and tests must treat these as
independent settings. Gate 1 retains its present `radius_from_mass: true` contract. An equal-
physical-radius experiment remains useful as a harder control, but in that control elastic
trajectories determine normalized mass ratios rather than absolute scale unless another absolute
reference is introduced. That limitation does not apply to the current Gate-1 setup.

### 2. Slot correspondence must be generic and permutation-safe

No renderer-specific appearance category, known object label, mass, contact signal, or simulator
identity may define the slot assignment. Recurrent context slots can provide candidate tracks, but
the independently encoded target frames still produce unordered slot sets. Their association must
be inferred from learned slot features and generic spatial-temporal continuity, then held fixed
over a trajectory.

The renderer must also avoid a dataset-global visual signature tied to simulator row index;
otherwise a learned feature matcher could rediscover that signature even without an explicit
hand-written rule. The primary Bounce run should use the same mass-independent glyph for every
object and a geometry-only tracker. Heterogeneous-appearance variants may randomize their mapping
across episodes and serve as generalization tests. Learned visual descriptors may then be added for
harder scenes, but they must generalize across held-out appearance changes rather than act as fixed
object addresses.

This is feasible for persistent, visible objects when the learned slots localize them and motion
provides enough continuity. It cannot recover an objectively unobservable label: if two visually
indistinguishable objects undergo an ambiguous crossing or collision, exchanging their names is an
equivalent explanation. In that case the strongest defensible result is identification of the
mass set and consistent tracks where the observations determine them, not recovery of arbitrary
simulator labels.

## Gate 1: true states to true states

### What is already implemented

`StateJepa` with `model.gt_states=true` already provides the correct fixed-ruler model:

```text
true state history -> context embedding -> five scalar parameter coordinates
true state at t + learned parameters -> SPARTAN -> true state at t+1
```

The `bounce_baumgartner` config uses five state nodes, five scalar parameter coordinates, raw
aligned MSE, one-step teacher forcing, and a 512-dimensional SPARTAN workspace. Dense `A=1`,
identity `A=0`, and gated sparse variants exist. The tests verify raw anchors and targets, parameter
gradients, dense/identity construction, and the evaluation contract.

This means Gate 1 is implemented, not yet empirically established. Code cannot ensure that the
learned coordinates are the masses; the running sweep and complete pipeline must demonstrate it.

### Adjustments before Gate 1 can be declared passed

1. Evaluate dense, identity, and sparse checkpoints on the same final 5,000-episode split. The
   current shell pipeline uses only 256 validation episodes for the dense/identity feasibility
   check and runs the final evaluation only for sparse.
2. Write split-specific artifacts so validation and test metrics cannot overwrite one another.
3. Add paired dense-versus-sparse seed differences, bootstrap intervals, and provenance checks to
   aggregation.
4. Implement parameter-use interventions: shuffle parameter vectors across episodes, replace them
   by their training mean, and disable retained parameter-to-state paths. Report collision and
   non-collision prediction changes separately.
5. Add an automated run validator for constraint satisfaction, sustained density reduction,
   non-identity final density, mass recovery, and aligned graph improvement. `sparsity/active=1`
   alone is not a pass.
6. When targets are literal states, do not apply stochastic VISReg to the fixed target tensor.
   Regularize the learned context/parameter representation only; target variance is then a data
   diagnostic rather than a collapse diagnostic.

Do not begin Gate 2 until one complete Gate-1 pilot has passed or produced an understood negative
result. If Gate 1 cannot identify masses from true states, adding pixels cannot repair the causal
identification problem.

## Gate-2 readiness: freeze the coordinate and matching contract

Do not run a complete extra sparse study merely to call this “Stage 1.5.” Implement and test the
following invariants first:

- one object permutation applied consistently through a trajectory must only permute the model's
  track axis;
- a slot switch halfway through a trajectory must be penalized;
- the prediction matcher must choose one detached assignment per trajectory, not one assignment
  per frame;
- the same assignment must align state rows, graph rows/columns, and mass evaluation;
- five slots are used for five balls until an explicit null-slot contract exists.

Use a track-attached parameter contract for the visual gates. `TrackAwareAttnPooling` emits one
scalar per recurrent track. Give the state token and parameter token belonging to a track the same
ephemeral association key, and permute that key with the complete track. The key carries no
physical value; it only preserves the binding between two representations of the same inferred
track. SPARTAN must still learn which parameter-to-state paths matter and the gates must still
prune them.

Do not reuse dataset-global SPARTAN node addresses for anonymous visual tracks. First validate the
new association mechanism on consistently permuted true-state trajectories: jointly permuting
tracks, their parameters, and their keys must only permute outputs and graph axes. Gate 1 can keep
its current five global latent coordinates; the readiness bridge deliberately changes this
interface before pixels are introduced.

## Gate 2: causal frames to true future states

### Required model

Add a dedicated `VisualStateJepa`; the current `SCJepa` cannot perform this experiment because it
always constructs a learned future-frame target encoder.

```text
frames[:, :-1]
    -> causal recurrent context encoder
    -> slots (B, L-1, 5, 32)

slots at each source time
    -> grounded state head
    -> source [x,y,vx,vy] (B, K, 5, 4)

first parameter_context=30 slot histories only
    -> five persistent scalar parameter coordinates (B, 5, 1)

source state + parameters
    -> SPARTAN (internal width 512)
    -> predicted next [x,y,vx,vy]

states[:, parameter_context:]
    -> unchanged raw targets
```

The target encoder is absent at Gate 2. Use `rollout_horizon=1`: every later source frame is
available and supplies its own teacher-forced source state. Long open-loop rollout is an evaluation
diagnostic here, not the training objective.

### Minimal code changes

1. Add `model.type: visual_state` to the factory and a `VisualStateJepa` model.
2. Allow the trainer to pass the complete batch or explicit `frames` and `states`, rather than one
   tensor selected by `input_key`.
3. Configure `KinematicHead(slot_size=32, state_size=4)` and SPARTAN with state width 4, parameter
   width 1, and internal width 512.
4. Keep `(B,K,N,4)` until a trajectory assignment is selected. Add
   `prediction_matching: trajectory_hungarian`; flatten only after matching or for graph readout.
5. Retain a current-state grounding loss throughout Gate 2. It uses the same state supervision as
   the raw future targets and makes the four-dimensional bottleneck genuinely `[x,y,vx,vy]`
   instead of an arbitrary code that can hide mass.
6. Pool parameters only from the first 30 context frames. Perturbing any later frame must not
   change the parameter vector.
7. Add a scientific Gate-2 config rather than relying on base defaults. Explicitly pin five slots,
   width 32, scalar parameters, raw constraint normalization, one-step training,
   `physics_radius_from_mass: true`, `render_radius_from_mass: false`, uniform exchangeable renderer
   appearance, and the matching mode.
8. Extend data caching or rendering so a large frame experiment does not regenerate every video
   in every reference leg.
9. Extend the evaluator to visual tracks; the current harness deliberately rejects `SCJepa`.

### Gate-2 exit conditions

- a held-out trajectory assignment gives five-object coverage and a low switch rate;
- the grounded state head predicts position and velocity accurately without future-frame leakage;
- mass is recovered from the context-derived parameter coordinates;
- mass is weak or absent in the four-dimensional state bottleneck beyond what its state values
  imply;
- dense beats identity under the exact raw-state constraint;
- sparse reaches a fresh Gate-2 tau and prunes without reaching identity;
- parameter interventions worsen collision-sensitive prediction;
- the result holds when the per-object glyph conditional on physical state contains no direct mass
  cue, using `physics_radius_from_mass: true` and `render_radius_from_mass: false`.

## Optional diagnostic: learned encoding of true future states

The existing roadmap's Stage 3 remains useful if Gate 3 fails and we need to distinguish visual
target binding from general moving-target instability. It should initially be a small dense/
identity diagnostic:

```text
visual context -> learned prediction
true future [x,y,vx,vy] -> jointly learned 32-D target embedding
```

Do not require a complete multi-seed sparse study unless this diagnostic produces a result needed
for the paper. It does not test single-frame visual target formation and introduces the same
moving-ruler problem as Gate 3.

## Gate 3: fully visual joint training

### Observation is not the dynamical state

A single-frame target slot is an observation representation, not a complete Markov state. Original
V-JEPA target features are temporally contextualized by a video encoder. The relevant production
precedent is V-JEPA 2-AC: it independently encodes images, gives an ordered history to a causal
predictor, and combines teacher-forced and short autoregressive losses. Its released configuration
uses a two-step autoregressive objective and representation normalization:

- [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985)
- [official V-JEPA 2-AC training loop](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_droid/train.py)
- [official causal predictor](https://github.com/facebookresearch/vjepa2/blob/main/src/models/ac_predictor.py)
- [released DROID configuration](https://github.com/facebookresearch/vjepa2/blob/main/configs/train/vitg16/droid-256px-8f.yaml)

V-JEPA 2-AC freezes its pretrained encoder. We adopt its observation/history/rollout structure,
not its frozen-target choice.

### Minimal final architecture

Use the single-frame target module as a jointly learned **frame observation encoder** `E_phi`. It
is called independently on observed seed frames as well as future target frames. This does not
give it temporal information: every call still receives exactly one image.

```text
independent frame code:
    {z_t^j, a_t^j} = E_phi(x_t)                   # 5 learned slots + attention maps

generic target-track association:
    {z_t^j, a_t^j}_{t=0:L} -> zbar_0:L           # learned features + motion continuity

short observable dynamics state:
    s_t = TemporalLift([zbar_{t-1}, zbar_t])      # 64 -> 32 per object

long parameter context:
    C_theta(x_0:29) -> context slots
    context slots -> theta_hat                    # (B, 5, 1)

transition and rollout:
    zhat_{t+1} = SPARTAN(s_t, theta_hat)           # (B, 5, 32)
    s_{t+1} = TemporalLift([zbar_t, zhat_{t+1}])

single-frame target:
    z_{t+1} = E_phi(x_{t+1})
```

This preserves the intended five 32-dimensional predicted target slots. Velocity lives in the
ordered relation between consecutive embeddings. It need not, and generally cannot, be recovered
from `z_t` alone. `TemporalLift` should be one small shared per-object linear layer or MLP; do not
add a second large world model. Concatenation supplies lag order for fixed `dt`; encode `dt` only
if frame stride varies.

This architecture also closes the rollout in one representation space. The current code begins in
the recurrent context-head space and then feeds target-space predictions back into the same
SPARTAN. Equal width is not sufficient to make that composition valid.

Use separate temporal budgets:

```yaml
model:
  num_slots: 5
  slot_size: 32
  spartan_embed_dim: 512
  parameter_context: 30
  dynamics_history: 2

data:
  physics_radius_from_mass: true
  render_radius_from_mass: false
  object_appearance: uniform

train:
  autoregressive_steps: 2
```

Longer open-loop horizons through multiple collisions are required for evaluation. They do not
need to be backpropagated end to end initially.

### Generic slot matching is feasible on Bounce, conditionally

The present per-transition latent Hungarian loss is insufficient because it can erase identity
switches. The current SAVi Slot Attention computes pixel-to-slot attention but discards it. Expose
the assignment maps without adding a reconstruction decoder. Their centroids provide generic
geometric evidence rather than an object label. If a confidence or presence value is needed, derive
it from pre-spatial-normalization assignment mass, logits, or entropy; the current per-slot
normalized map sums to one and cannot itself measure presence.

The minimal matching procedure is:

1. Independently encode every target frame and retain each slot's learned feature, attention-map
   centroid, and, if implemented, a valid pre-normalization confidence statistic.
2. Assemble target tracks sequentially. At time `t`, solve a detached Hungarian assignment whose
   core Bounce cost uses attention-support overlap and distance from a constant-velocity centroid
   prediction based on the preceding two assigned frames. A normalized learned-feature term is an
   optional generic extension for more complex scenes, not a fixed renderer-identity key.
3. Keep the resulting target tracks fixed, then compute one detached trajectory-level Hungarian
   assignment between predicted tracks and target tracks using normalized latent distance across
   all prediction times.
4. Hold that predictor-to-target assignment fixed for the complete trajectory loss and graph-row
   alignment. Gradients still reach the selected predictions and target slots; only the discrete
   assignments are detached.
5. Use true states and renderer masks only after training to measure object coverage, switches,
   collision failures, and the fraction of ambiguous trajectories.

The ordering chosen for the first target frame is arbitrary: the downstream tracker, loss, shared
track keys, and graph alignment must be invariant/equivariant to relabelling that encoded slot set.
Do not claim that property for the complete current SAVi encoder unless its slot initialization is
also made exchangeable and tested. Learned features will be noisy early in training, so establish
tracking on a small dense teacher-forced run before enabling sparsity. If this generic matcher
cannot maintain tracks, the current visual slots do not support the final claim. The next
defensible fallback is a learned association module or differentiable transport with an
occlusion/dustbin state—not a renderer-derived identity shortcut and not independent per-frame
rematching.

### Joint objective and target stability

Use a compact objective:

```text
teacher-forced latent prediction
+ beta * two-step autoregressive latent prediction
+ representation regularization on learned context and target slots
+ lambda_logit * attention-logit penalty
+ GECO-weighted decoded-row path sparsity when active
```

Keep MSE initially for continuity with the SPARTAN experiments. Per-slot LayerNorm before latent
comparison and every feedback step is a predeclared Gate-3 choice worth testing because V-JEPA
2-AC normalizes its representations; changing normalization or loss requires a fresh tau.

The jointly learned target creates a moving GECO ruler. Matched initializations and target-variance
normalization fix neither arbitrary nonlinear geometry nor semantic target collapse. Use this
minimal two-phase protocol:

1. **Joint representation phase:** train the context encoder, target encoder, parameter encoder,
   temporal lift, and an all-open/dense SPARTAN together from scratch.
2. **Common-start comparison:** copy the same converged joint checkpoint into continued dense,
   identity, and gated-sparse legs. Keep all modules trainable in the primary sparse leg. Measure
   tau in the shared checkpoint's target geometry before the legs diverge.
3. Run a labelled frozen-target pruning control. Agreement between the fully joint and fixed-ruler
   control makes the sparsity conclusion much stronger; the frozen version is not substituted for
   the primary claim.
4. Continuously report target drift, per-aligned-slot variance/effective rank, object-position
   information, and fixed held-out state probes. If dense and sparse targets retain materially
   different information, do not interpret their latent losses as equal-quality constraints.

Representation regularization must be measured per aligned slot across episodes and time as well
as globally. Flattening all slot addresses together can make a static slot codebook look diverse.
One-frame position should be decodable; one-frame velocity should not be a success criterion;
velocity from the two-frame temporal state should be substantially better.

### Gate-3 exit conditions

- all trainable branches receive gradients and remain non-collapsed;
- the target encoder remains one-frame-per-call and no future information reaches source seeds or
  parameter pooling;
- five target/context objects are covered and trajectory switch rate is negligible, including at
  collisions;
- velocity is recoverable from the two-frame state and not spuriously from a single frame;
- dense prediction is better than identity in the common target geometry;
- the gated model meets its fresh constraint and prunes to a non-identity graph;
- masses are recoverable from parameter coordinates but not simply from single-frame appearance;
- parameter shuffling/mean replacement/path ablation damages collision-sensitive prediction;
- open-loop error and latent norm are reported by horizon through multiple collisions;
- the result replicates across paired seeds and remains stable under held-out changes in rendered
  appearance that were not used by the matcher.

## Required implementation order

1. Finish and evaluate Gate 1; do not infer success from the code path alone.
2. Split physical and rendered radius contracts and version the dataset.
3. Add trajectory matching and its permutation/switch tests.
4. Implement `VisualStateJepa`, its config, batch contract, and visual evaluator.
5. Pass Gate 2 with raw targets and parameter-use interventions.
6. Expose slot attention maps and validate generic five-object track association.
7. Add `TemporalLift`, type-closed two-frame rollout, and teacher-forced plus two-step losses.
8. Run a tiny dense Gate-3 overfit and the joint-target collapse/content gates.
9. Perform the common-start dense/identity/sparse development comparison and frozen-target audit.
10. Freeze the protocol before confirmatory Gate-3 seeds.

## Tests required before expensive visual runs

- exact five-slot, 32-dimensional context/target/output shapes;
- hybrid model consumes frames and raw state targets but has no target encoder;
- current-state grounding is trajectory-permutation invariant;
- one trajectory permutation leaves loss unchanged; a mid-trajectory switch increases it;
- Hungarian decisions are detached while selected context/target values receive gradients;
- perturbing frames after `parameter_context` cannot change parameters;
- perturbing a future target frame cannot change earlier source codes or predictions;
- a single frame shared by two opposite-velocity simulations has the same one-frame code, while
  the two-frame temporal states differ;
- every autoregressive SPARTAN call receives a `TemporalLift` state—never a context-space anchor
  on the first step and a target-space vector thereafter;
- replacing or permuting other batch examples cannot change a frame's target encoding;
- the generic tracker is invariant to raw slot permutations and uses only learned features,
  attention geometry, and temporal continuity;
- the tracker passes an exchangeable-rendering geometry-only test, and learned-feature matching
  cannot exploit a dataset-global visual signature tied to object index;
- target/context coverage and switch diagnostics include collision frames;
- parameters, state rows, target rows, and graph axes use the same frozen trajectory assignment;
- raw-target runs omit target VISReg; joint-target runs detect per-slot/content collapse;
- dense, identity, and sparse evaluation artifacts are split-specific and comparable.

## Deliberately deferred work

Do not add these until the three gates work:

- V-JEPA multi-block masking or tubelets;
- large-scale visual pretraining;
- a learned association network, occlusion-aware min-cost-flow tracker, or variable-object-count
  matcher;
- pixel reconstruction in the training loss;
- camera-motion augmentation;
- long-horizon backpropagation through 30 visual rollout steps;
- a full confirmatory study of the optional learned-true-state-target diagnostic.

The scientifically minimal result is already substantial: jointly learn frame representations,
infer hidden physical parameters from a longer visual context, predict single-frame target slots
through a sparse SPARTAN, and verify from held-out simulator data that the learned parameters are
the true causal masses and are actually used at collisions.
