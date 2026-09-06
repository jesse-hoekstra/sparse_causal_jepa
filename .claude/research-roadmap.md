# Research roadmap: two experiments

Updated 2026-09-06. Read [decisions D37–D42](../docs/decisions.md) and
[README.md](../README.md) for the active protocol. Historical run failures and objective changes
remain in the decision log. The supervised visual-to-state bridge is retired.

## Experiment 1: true states

Use `bounce_baumgartner`. The successful D30 teacher-forcing run recovered masses and pruned the
graph (MCC 0.948, SHD 4.81). Current training adds eight sampled T=2 endpoint losses to all 30
teacher-forced suffix transitions, with one shared parameter estimate per episode. K=30 is a
held-out diagnostic only. Preserve the canonical physics preload and the stable baseline.
D38 specifies dense/identity feasibility selection for tau; historical thresholds are invalid
for the current objective.

## Experiment 2: visual state learning

Use `bounce_visual_to_visual` and `scripts/l40_exp2_pipeline.sbatch`. Extend Experiment 1 by
learning online states and parameters from frames and learning targets through a stop-gradient
EMA visual branch. Both training branches use causal recurrence; only the target sequence
contains future target frames. Render equal-radius white glyphs from the same physics preload.
The online branch cannot receive true states or masses for training.

Use one detached context-prefix minimum-cost branch assignment per episode and keep it for all
future targets and TF/T=2 windows. The matching inputs are pre-head slot histories, never true
states, masses, future slots, or prediction residuals. It corrects global branch permutation,
not within-episode identity switches. EMA ancestry does not prove physical identity, state
sufficiency, or non-collapse. These are empirical requirements and explicit assumptions in
SCJEPA.pdf (PDF p.9, Assumption 2 / Remark 3).

The dense run freshly calibrates this experiment's raw TF+T=2+logit constraint (D42); sparse
training uses that tau. Target variance does not divide the loss. Require the
`visual_constraint_version=raw_tf_t2_v1` config/checkpoint stamp; old normalized thresholds do
not transfer. Reject gross collapse using evaluation diagnostics, including
`min_target_variance=1e-4`. Equal raw MSE across independently learned target spaces does not
establish equal physical fidelity. There is no replacement LayerNorm, reconstruction/grounding
loss, or new anti-collapse regularizer. The inherited
lambda_logit=1e-5 is an exploratory default, not a completed visual sweep.

## Evidence required

1. Finite training and evaluation losses with healthy optimizer/EMA updates.
2. Non-degenerate spatial and temporal slots; correct object coverage and low switching.
3. Frozen train-fit/test-eval position and velocity probes that recover physical state better
   than trivial predictions. Read decodability as evidence, not coordinate identity or proof of
   the Markov-state assumption.
4. MCC and SHD interpreted together with path density and the latent prediction constraint.
   Empty or saturated graphs are failures even if an individual metric appears favorable.
5. A small frame/slot panel, core scalar curves, and final recovery plot make the run inspectable.

Full visual convergence, confirmatory seed statistics, parameter interventions, and sensitivity
to EMA speed remain future research. Do not declare success from a short smoke or a low latent MSE.
