# Vision experiment

Two bonded colloidal dimers move in an anisotropic medium and are filmed from above by one
RGB camera. Each dimer is a large bead bonded to a small bead; the eight in-plane bead
coordinates in the medium's principal frame are the state, the four drag coefficients (two
bead sizes, two medium axes) are the hidden parameters of a system, and one out-of-plane
magnetic field and one in-plane electric kick per dimer are the actions. The model sees only
the images and actions. Training follows the staged schedule of Experiment 1. The analyses
score the learned states and codes against the simulator, and the downstream module asks
interventional questions of the trained models.

## Modules

| File | Contents |
|---|---|
| `dgp.py` | Physics, parameter prior, simulation, exact inverse step, true graphs; the RGB camera (`Renderer`) |
| `data.py` | Cohort generation with hashed manifests, label-free rendered inputs, bit-exact split verification |
| `model.py` | CNN state encoder, lagged code encoder, gated two-lag predictor |
| `train.py` | Staged training (`train`, `cache`, `select`, `project`, `calibrate`, `fork`, `evaluate`) |
| `evaluate.py` | Pearson and Spearman MCC, block purity, state and parameter retention, graph SHD, code shuffling |
| `summarize.py` | Per-run criteria, arm means, paired sign-flip contrasts, random-rotation references |
| `analysis.py` | Figure data, chart diagnostic, code invariance (both experiments), learning curves |
| `downstream.py` | Virtual tweezer queries on trained models: hold a coordinate (Q1), kick a dimer with one bead held (Q2) |
| `paper_figures.py`, `paper_downstream.py` | The paper's figures, LaTeX tables and number macros |
| `verify_assumptions.py` | Float64 numerical checks of the theory's assumptions for this DGP |
| `run_seed.sh`, `final_evaluation.sh` | The reported pipeline, with every argument written out |

## Reported settings

The defaults of `dgp.Physics`, `dgp.Renderer`, `data.DataConfig` and `train.StageConfig` are the
reported settings, and `tests/test_vision_dimer.py` pins the cohort record. The paper's
hyperparameter table lists the architecture.

| Setting | Value |
|---|---|
| Cohort | seed 20261115; 8,192 / 512 / 256 / 512 train / calibration / validation / test groups; 10 episodes of 40 frames per group |
| Windows | 10 context and 2 forecast frames; training draws a random 12-frame window per episode and update; evaluation uses the first window |
| Camera | 64 x 64 RGB; blob widths 2.6 (large bead) and 1.8 (small bead); lens span 28, scale 2; medium axes at 45 degrees to the camera; dye responses (1, .55, .1) and (.1, .45, 1) |
| Physics | trap stiffness 1, bond stiffness 1, trap-site separation 1, dimer homes (-1.3, .6) and (1.3, -.6); drag prior U(0.1, 1.2) per coefficient; step 0.5 |
| Learning rates | 1e-3 for the code encoder, predictor and gates; 1e-4 for the CNN |
| SIGReg | weight 0.05 on the online states, in every stage that trains the CNN (joint, baseline, full) |
| Schedule per seed | joint 100k (code dimension 8) -> 16 acquisitions of 100k with frozen encoders (seeds 1000 + 100 j + seed) -> selection on the mean TRAIN prediction loss of the last 5,000 updates -> projection 8 -> 4 -> adaptation 50k (the dense initializer) |
| Sparse arms | baseline (state and action gates) and full (also parameter gates), 500k updates each, run as 300k plus an exact continuation; gate logits start at +3 |
| Tolerance | constant GECO tolerance tau = 2 x the calibration prediction MSE of the seed's own dense initializer; GECO step 0.1, momentum 0.99, initial lambda 1000, log lambda clipped to [-12, 12] |
| Batch | 16 groups x 10 episodes |
| Seeds | 0-4 |

## Reproducing the experiment

```bash
python -m diagnostics.vision_dimer.data --data-dir DATA       # train, calibration, validation
for seed in 0 1 2 3 4; do                                     # one GPU per seed
  bash diagnostics/vision_dimer/run_seed.sh "$seed" DATA RUNS
done
bash diagnostics/vision_dimer/final_evaluation.sh DATA RUNS RUNS/final-evaluation-test
python -m diagnostics.vision_dimer.verify_assumptions --out assumptions.json   # CPU only
```

`run_seed.sh` runs eight acquisitions at a time on one GPU (fourth argument) and resumes
interrupted stages. `final_evaluation.sh` checks that the stored validation split regenerates
bit for bit, generates the test split once, and scores the fifteen fixed-budget endpoints:
the dense initializer and the 500k baseline and full models of each seed. It then writes the
figure data, summaries, chart and code-invariance analyses, learning curves, the downstream
queries and the paper's figures and tables. Every probe and alignment is fitted on
calibration; the test split is only scored.

## Numerical notes

- Training disables TF32. `train evaluate` keeps PyTorch's default cuDNN TF32 convolutions, as
  in the reported test evaluation, while the analyses run in full float32. Metrics recomputed
  by `analysis figure-data` agree with the stored evaluation to about 2e-5, and the check
  tolerance is 1e-4.
- GPU kernels are not bitwise deterministic, so a retrained model reproduces the reported one
  only up to that noise. CPU runs are bitwise reproducible.
- This package reproduces the development runtime that trained the reported models bit for
  bit on CPU, for every stage and the continuation. `train.upgrade_checkpoint` lets it
  continue checkpoints written by that runtime exactly.
- W&B logging is optional (`--wandb-mode online`). The entity comes from the W&B login or
  `WANDB_ENTITY`.
