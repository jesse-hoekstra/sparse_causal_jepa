# Architecture and interfaces

## Observation and context encoding

`FramewiseEncoder.forward` accepts a batch of individual observations;
`encode_sequence` folds `(B,T)` into one batch axis and restores it after encoding.
`VectorEncoder` uses configurable MLP widths and frame-local LayerNorm.
`ImageEncoder` uses strided convolutions, frame-local GroupNorm, spatial pooling,
and a linear state head. Neither path receives actions, parameters, other frames,
or recurrent state. No latent coordinate has a predefined physical meaning.

`ParameterEncoder(states, actions)` receives `(B,K,d_z)` and `(B,K-1,d_a)`.
It adapts the retained attention/residual blocks and learned-query temporal
pooling. Learned time embeddings preserve order. The last state has no generating
context action attached; a zero action with an explicit availability indicator
marks it. Its unconstrained head emits `(B,d_theta)`, independently of `d_z`.
In no-action mode no action tensor is accepted. `K=1` is supported mechanically;
whether any chosen context identifies parameters is an experimental question.

`SCJEPA.forward(observations, actions=None, prediction_start=None)` calls the
parameter encoder once on the permitted prefix. The exact same attached tensor
is passed to every predictor call. Changes to later observations or actions do
not change it. Teacher-forced prediction at source `t` receives
`states[:, t-L+1:t+1]` and `actions[:, t-L+1:t+1]`, targeting frame `t+1`.
The default first source is `max(K,L)-1`; explicit starts must be at least that
large. This permits independently configured context and dynamics windows,
including `K<L`; the paper's informative-context assumptions remain additional
scientific requirements.

`ModelOutput` exposes predictions and targets `(B,W,d_z)`, all online states
`(B,T,d_z)`, parameters `(B,d_theta)`, source start, and a tuple of per-transition
`SpartanOutput` values. Here `W=T-1-prediction_start`.

## Sparse coordinate dynamics

SPARTAN's `forward(states_history, parameters, actions=None)` accepts
`(B,L,d_z)`, `(B,d_theta)`, and optionally `(B,L,d_a)`. It predicts `(B,d_z)`.
There are `M=L*d_z + L*d_a + d_theta` scalar input tokens:

1. State coordinates, oldest lag first, coordinate order within each lag.
2. Action coordinates in the same lag order.
3. Parameter coordinates, each appearing once.

Shared scalar-to-workspace projections, learned role/coordinate/lag identities,
within-token normalization and MLPs, and scalar heads never mix coordinate tokens.
Only sparse attention transports values between tokens. The latest-state token
for coordinate `i` decodes output `i`, retaining its local residual path.

The retained binary straight-through gates use logistic noise while training;
evaluation thresholds logits at zero. Dense mode opens every gate, and identity
mode closes every attention edge. These modes use the same modules.
The independent `parameter_gates_open` option (`spartan_parameter_gates_open` in
`ModelConfig`) forces the parameter-source columns into the decoded latest-state
receiver rows open in every layer, in both training and evaluation. Other gates
remain learned. Attention weights remain learned even for forced-open gates, so
dense parameter routing does not guarantee that every parameter has a nonzero
functional effect. The option introduces no learned tensors or random draws;
paired model tensor initialization is unchanged.
Stable masked softmax handles fully closed rows without epsilon-gradient
amplification. The optional logit penalty is `mean(exp(g)+exp(-g))`, using a linear
continuation above `abs(g)=30` and float32 arithmetic.

**Local attention:** queries, keys and values use the current coordinate-token
content. Bernoulli gates and logit regularization use unscaled `q @ k`; attention
weights use `q @ k / sqrt(d)`. Coordinate, role and lag identities remain explicit.
This follows SPARTAN's state-dependent local routing design. Extracted paths track
masked value routing and residuals; they are not a complete global functional
independence certificate because attention controls depend on the inputs.
Parameter inference mixes only its permitted context. Predictor graphs are
conditional on the supplied parameter code, not the full context-to-output graph.

Path accounting multiplies `(A_layer + I)` through the layers in float32,
including residuals. Decoded rows give separate `GraphBlocks`:

| Block | Shape | Input axis |
|---|---|---|
| state | `(B,d_z,L,d_z)` | lag, state coordinate |
| action | `(B,d_z,L,d_a)` | lag, action coordinate |
| parameter | `(B,d_z,d_theta)` | parameter coordinate |

Lag zero means the oldest input. Nonzero counts mean at least one available path.
`aggregate_lags()` takes the maximum over the lag axis; thresholding the result
at zero implements the paper's any-lag dependency convention. Counts are potential local routing
paths, not measured causal effects. The unavoidable latest-state self path is
included. Graph extraction never infers state size from parameter size.

Each sparsity term is the mean path count of its own lag-resolved block; the
empty action term is exactly zero. These are differentiable straight-through
path penalties, **not exact edge-budget constraints**. The means include constant
residual contributions. Independent objective weights permit removal of the
parameter-block penalty without changing gate modes. The controlled experiment
also forces direct parameter gates open in the baseline; the full model learns
those gates and adds the parameter-block penalty. Both regularize state/action
paths. Shared attention paths can contribute to more than one block, so other
penalties can still change gates that remain learned. Hard budgets or other
surrogates, if wanted, require an explicit future experiment choice.

## Rollout

`model.rollout(states_history, parameters, steps, actions=None)` takes an attached
parameter tensor already inferred from the allowed context. For actions, supply
`(B,L+steps-1,d_a)` beginning at the first initial-history index. Each step takes
the next `L` actions, shifts state history left, and appends its prediction.
It returns `(B,steps,d_z)` and preserves gradients by default. Evaluation callers
use `model.eval()` and `torch.no_grad()`. Rollout length never changes `L`.
The default training objective contains no rollout term or fixed horizon.

## EMA and objective

The complete online state encoder is deep-copied exactly at initialization.
Only that path has a target copy. Target parameters are frozen, excluded from
optimizer groups, evaluated under `no_grad`, and kept in evaluation mode even
when `model.train()` is called. After each successful optimizer update:

`target_parameter = decay * target_parameter + (1-decay) * online_parameter`.

Every target buffer is copied exactly from online, regardless of dtype; buffers
are not averaged. Current encoders do not use batch-running statistics. The
trainer skips EMA on accumulation microsteps, nonfinite losses/gradients, and
rejected large gradients. A rejected microbatch discards its entire accumulated
group. Full precision is the supported trainer mode; custom AMP/scaler integrations
must call `update_target()` only when their optimizer update actually occurred.

The non-sparsity objective is:

`base = MSE(predictions, stopgrad(targets)) + wlogit*logit_penalty`
`       + ws*state_reg + wp*parameter_reg`.

`path_loss = wz*state_paths + wa*action_paths + wtheta*parameter_paths`.

Generic fixed-weight training uses `base + path_loss`. With `TrainConfig.geco`
enabled, the primal loss is `base + path_loss/lambda` subject to the calibrated
constraint `E[base] <= tau`. `GECOConfig` defines the explicit bound, dual step
size, moving-average momentum, initial multiplier and log-multiplier bounds.
After each accepted optimizer update the controller averages base loss over all
accumulated microbatches and ranks, updates its moving constraint average, and
adds `step_size * moving_constraint` to `log(lambda)`. Its state is checkpointed.
Neither rejected updates nor partial accumulation advance GECO or encoder EMA.
The controlled experiment obtains tau from a separate dense calibration stage;
no graph budget or ground-truth edge count enters optimization.

SIGReg uses the retained upstream implementation through a typed wrapper. The
active choices are `none` and `sigreg`; VISReg is not an enabled training option.
State inputs
are `(T,B,d_z)` (one sample per episode at each time); parameter inputs are
`(1,B,d_theta)`, never repeated over prediction steps. Parameter regularization
requires distinct original episode IDs across each global microbatch. Both
state and parameter penalties need at least two episodes per rank. Distributed representation
statistics are **rank-local**, averaged through DDP gradients; changing world
size can change this regularizer objective. Accumulation averages microbatch
objectives rather than pooling their representation statistics.

Regularizer selection, weights, network widths, context informativeness, and
sparsity strength are explicit implementation/experiment choices. The paper
motivates SIGReg but does not prescribe a universal coefficient here. These
components do not establish the paper's information-preservation assumptions.
The supplied vector MLP is not constrained to be invertible. The
[controlled experiment](controlled_experiment.md) uses it with empirical recovery
and information-preservation checks, using held-out MCC and aligned routing-support SHD.

## Infrastructure and future experiments

`Trainer` retains AdamW, clipping and synchronized finite/skip guards, deterministic
global batch sharding, rank-aware logging, and atomic checkpoint writes. All ranks
participate in checkpointing. Checkpoints store online and target state, EMA/context
settings, optional full model configuration, objective and GECO configuration/state,
optimizer, accepted-update and skip counters, and each rank's Python/NumPy/Torch
RNG. Resume restores optimization settings and deterministic episode position;
run length/device/output/logging controls may be changed. Exact replay requires
the same model, world size, global batch and deterministic dataset identity.
Generated definitions, stored artifact contents, and tensor data are checked before
resume, including observation-map and graph-variant changes. See
[data generation](data_generation.md) for identity and storage conventions. Pending
gradients are not serialized, so saves require accumulation boundaries.

The CLI supplies a rank-zero W&B logger, with terminal JSON retained. Configuration,
dataset identity and reduced metrics are logged; model/data tensors are not uploaded.
Generic checkpoint `run_metadata` retains the tracking identity across resumption.
Logger initialization and metric failures are synchronized so peer ranks do not
continue into the next backward pass. Direct `Trainer` and smoke tests remain
independent of W&B. Logging restores training RNG state around SDK operations.

Future experiments should provide datasets yielding the documented tensors,
normalization fitted only on training data, meaningful contexts, declared losses,
and held-out scientific evaluation. Reusable probes and error/graph algorithms
accept arbitrary coordinate counts and caller-provided references. No simulator,
renderer, hidden physical-coordinate grouping, calibrated threshold, or scheduler
protocol belongs in the architecture modules.
