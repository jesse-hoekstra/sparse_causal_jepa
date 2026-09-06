# configs/

Hydra configuration tree for two bounce experiments. `config.yaml` holds shared defaults;
`experiment/bounce_baumgartner.yaml` is Experiment 1 (true states), and
`experiment/bounce_visual_to_visual.yaml` is Experiment 2 (online visual states and EMA targets).

Both presets use the following predictive settings:

```yaml
train:
  lambda_rollout_t2: 1.0
  num_rollout_t2_anchors: 8
  rollout_t2_horizon: 2
  oe_eval_horizon: 30
```

For each episode, eight distinct valid T=2 offsets are sampled independently without replacement.
Each window starts from its observed anchor, feeds the first generated prediction back without
detaching, and supervises only the second endpoint. All transitions and windows share the one
context-inferred `theta_hat` (and Experiment 2's episode-level track keys). The endpoint error is
averaged over batch, window, object, and coordinate dimensions. The horizon is validated as two;
there is no training curriculum. Setting `lambda_rollout_t2: 0` bypasses sampling and both
auxiliary calls. K=30 is evaluation-only.

Experiment 2 renders identical-looking balls from the same physics preload. Its target encoder
and state head update by EMA after accepted optimizer steps. A context-only branch assignment
fixes the target slot order once per episode; physical truth is used only in evaluation.
Both experiments use the raw constraint
`L_TF + lambda_rollout_t2*L_AR2 + lambda_logit*L_logit <= tau`, with the path penalty outside.
Experiment 2 records `visual_constraint_version: raw_tf_t2_v1` in its config and checkpoint;
legacy normalized checkpoints and tau values cannot be reused. Calibrate a fresh visual tau
from the matching dense raw constraint. `variance_floor` is removed from the model: variance
remains a diagnostic, and evaluation's `min_target_variance=1e-4` is only a collapse-screening
threshold. No replacement LayerNorm or anti-collapse regularizer is introduced.

`train.batch_size` is the global training batch, including under torchrun. The L40 Experiment-2
launcher keeps it at four and defaults to two GPUs (two episodes per GPU); one and four GPUs
are also supported. The learning rate and step count stay fixed. `resolved_config.yaml` records
the worker count and local/global batch sizes in `distributed`. DDP uses the global mean raw
constraint; target variance is gathered only when logging diagnostics. Evaluation remains single GPU.
CUDA pinned-memory loading and deferred monitoring apply automatically, without changing any
training hyperparameters. See `scripts/l40_exp2_benchmark.sbatch` for a short hardware comparison.

The supervised visual-to-state preset and visual-only K-rollout configuration keys are retired.
See `docs/decisions.md` D37–D42 for active settings and historical rationale.
