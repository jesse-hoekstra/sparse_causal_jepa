# configs/

Hydra configuration tree for the three bounce regimes. `config.yaml` holds shared defaults and
`experiment/bounce_baumgartner.yaml` is the paper-scale state-to-state Experiment-1 preset.

## Experiment 1: fixed teacher forcing plus T=2

The active predictive settings are:

```yaml
train:
  lambda_rollout_t2: 1.0
  num_rollout_t2_anchors: 8
  rollout_t2_horizon: 2
  oe_eval_horizon: 30
  oe_tolerance_nrmse: 0.10
```

`rollout_t2_horizon` is validated as exactly two; it is not a schedule. For every episode,
exactly eight distinct valid relative offsets are sampled uniformly without replacement and
independently of the other episodes. A window starts at true state `S_(C-1+r)`, feeds its first
generated prediction into the second transition without detaching it, and supervises only the
second prediction. All windows share the one `theta_hat` inferred from the episode's context.
The endpoint error is averaged over batch, window, object, and coordinate dimensions.

Setting `lambda_rollout_t2: 0` disables the auxiliary branch completely, including its random
sampling, and recovers the teacher-forcing-only computation. The dense tau-calibration run and
the sparse run must resolve identical T=2 settings. Never reuse D30's `tau=0.02` or a D34–D36
K=30 threshold; calibrate tau for the final teacher-forcing-plus-T=2 constraint.

The K=30 value belongs only to evaluation. The fixed held-out diagnostic uses a no-gradient
autoregressive chain from `S_29` to `Shat_59`, fixed training-set coordinate standard deviations,
and the configured NRMSE tolerance. The different horizons are intentional: H=2 trains local
composition and exposure bias, while K=30 measures sampled long-trajectory agreement.

There is no state-to-state rollout curriculum, rollout-length warmup, gradient-cut schedule, or
schedule-dependent checkpoint state. D34–D36 document that superseded experiment in
`docs/decisions.md`.

## Other regimes

Experiment 2 (`bounce_visual_to_state`) remains teacher-forcing-only because its latent input and
raw-state output cannot be composed by type. Experiment 3 (`bounce_visual_to_visual`) retains its
separately specified latent-space fixed-K objective; D37 does not change that protocol. Keep its
visual-only rollout keys distinct from Experiment 1's `lambda_rollout_t2` settings.

**Owner:** experiment-infra-engineer.
