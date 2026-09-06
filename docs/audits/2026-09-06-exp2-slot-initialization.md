# Experiment 2: diffuse slots after the 400-update benchmark

The user evaluated `bounce_exp2_benchmark_speed_shm_v1/gpu1` at step 400 on 64
held-out seed-offset-17 episodes, fitting the frozen state probe on 64 training episodes.
The checkpoint uses the raw TF + eight T=2 endpoints + logit objective (`raw_tf_t2_v1`).

| Diagnostic | Result |
|---|---:|
| Position probe R2 | 0.1442 |
| Velocity probe R2 | 0.04375 |
| Slot centroid RMSE (unit-image coordinates) | 0.2497 |
| Per-frame assignment switch fraction | 0.5659 |
| Branch slot disagreement | 0.5784 |
| Target content variance | 9.068e-8 |
| Target temporal variance | 7.330e-8 |
| Target effective rank | 3.236 |
| Mass MCC | 0.05325 |

The supplied `slot_tracks.png` shows nearly uniform allocations across all five slots.
White circles are contours of input balls; row object numbers are a forced evaluation
assignment. Neither is evidence of learned localization. Near-tied centroids make assignment
switches and branch disagreement unstable; they do not independently diagnose a matching bug.
The dense graph has density one by construction, so its SHD is not a sparsification result.

The reported K30 latent NRMSE of 0.03498 is not evidence of physical accuracy. Its
evaluation-only denominator is floored at 1e-4, over 1,100 times the measured target variance.
This floor is separate from training, whose objective remains raw as requested in D42.

## Initialization check

A local CPU forward pass used the actual dense preset, model seed 0, and four generated
held-out seed-offset-17 clips. No optimization occurred. Over target frames 30..59,
content variance was 9.186e-8 and allocations divided by uniform ranged only from
0.99999964 to 1.00000036. This is a small initialization check, not the full 64-episode
GPU evaluation or a paired estimate of training improvement.

A separate single-clip CPU trace measured RMS variation across slot rows at each stage:

| Frame | Transformer output | LSTM/projector output | Kernel mean | Corrected slots |
|---|---:|---:|---:|---:|
| 0 | — | — | 3.041e-1 | 2.166e-1 |
| 1 | 2.451e-1 | 7.389e-3 | 1.327e-2 | 9.111e-3 |
| 10 | 2.048e-4 | 9.993e-5 | 1.471e-4 | 1.010e-4 |
| 20 | 3.088e-6 | 1.596e-6 | 2.571e-6 | 1.860e-6 |
| 29 | 1.509e-7 | 3.943e-8 | 7.462e-8 | 8.546e-8 |
| 59 | 1.416e-7 | 1.440e-8 | 4.048e-8 | 7.725e-8 |

The LSTM/projector contracts row differences about 33-fold at the first recurrent step;
repeated recurrence makes them negligible by the start of predictive supervision. Equal
slot queries yield equal per-pixel responsibilities and uniform spatial allocations.
This explains how different input frames can still produce indistinguishable object rows.

Interpretation: distinct object slots are not evident at step 400. This is only 0.13% of the
planned 300,000 updates and is a baseline observation, not grounds for declaring the setup
unsuccessful or requiring an architecture change. The recurrent weights are trainable, so
contraction at initialization does not establish that the behavior persists after learning.
Neither observation establishes irreversible collapse or an effect caused by removing variance
division. Continue the planned dense training with the current encoder and raw objective,
using the existing evaluations every 5,000 updates to track state decoding, object grounding,
and representation variation. Assess their trajectory over training, rather than treating an
individual early checkpoint as a pass/fail gate. No production architecture, training
normalization, or objective was changed during this audit.
