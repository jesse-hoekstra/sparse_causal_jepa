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
Its GECO constraint normalizes the predictive term by detached target content variance, while
Experiment 1's constraint uses raw state error. Calibrate tau separately with matching objective,
data, architecture, and seed. Neither old pure-TF nor full-K rollout thresholds transfer.

The supervised visual-to-state preset and visual-only K-rollout configuration keys are retired.
See `docs/decisions.md` D37–D39 for active settings and historical rationale.
