# Current scope and implementation requirements

Updated 2026-09-06. This file formerly proposed three scientific gates and a jointly optimized
single-frame target. That proposal is superseded by the user's two-experiment setup and
[decisions D37–D42](../docs/decisions.md).

1. **Experiment 1:** infer parameters from true-state histories; train all teacher-forced suffix
   transitions plus eight sampled T=2 endpoints. Keep the successful historical baseline and
   current dense/identity feasibility procedure. K=30 is evaluation-only.
2. **Experiment 2:** infer states and parameters from causal frame sequences; predict future
   stop-gradient EMA states with the same local TF+T=2 objective. The target is a recurrent EMA
   copy of the online encoder and state head. It processes the complete sequence, including
   future targets. No true-state supervision or jointly optimized target is active.

Draw identical white glyphs with equal rendered radius from the recorded Experiment-1 physics.
Physical collision radii stay mass dependent. Do not regenerate that preload on another machine.

Match online and target pre-head slot trajectories on the shared context prefix once per episode;
hold the detached permutation for the full sequence. No simulator identity, physical state,
mass, future frame, or per-frame rematching chooses it. EMA keeps features related but does not
prove slot identity or prevent collapse; matching cannot repair within-episode tracking switches.

Measure content/temporal variance, tracking, frozen position/velocity probes, MCC, SHD, and path
density. Preserve train/test episode separation for probes and use physical labels only in
evaluation. A concise slot panel and recovery plot accompany scalar curves. A representation may
linearly encode physical state without its latent coordinates literally being that state.

Both experiments use the raw constraint
`L_TF + lambda_rollout_t2*L_AR2 + lambda_logit*L_logit <= tau`, with the path penalty outside.
Visual tau is freshly calibrated from its dense raw constraint after gross-collapse screening.
D42 removes predictive division by variance; the evaluation-only `min_target_variance=1e-4`
threshold remains a diagnostic screen. Visual configs/checkpoints carry
`visual_constraint_version=raw_tf_t2_v1`; older normalized thresholds/checkpoints do not transfer.
Dense and sparse targets can still have different feature geometry, so equal raw latent
constraints do not establish equal physical fidelity. Full visual convergence and confirmatory
seeds remain to be demonstrated. See [research-roadmap.md](research-roadmap.md).
