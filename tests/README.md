# tests/

CPU pytest coverage for the two active experiments:

- `test_state_to_state.py`, `test_rollout_t2.py`, and `test_observational_equivalence.py` cover
  Experiment 1, shared T=2 sampling/gradient behavior, and evaluation-only K=30 rollouts.
- `test_visual_to_visual.py` covers Experiment 2's EMA, causal visual inputs, branch alignment,
  and raw TF+T=2 predictive/GECO objective. Target variance is diagnostic only; protocol checks
  reject checkpoints lacking `visual_constraint_version=raw_tf_t2_v1`.
  `test_bounce_visual.py` pins equal glyphs and shared physics.
- `test_slot_assignment.py` compares device-side enumeration with SciPy's exact optimal cost,
  including near ties, all five-slot permutations, and gradient detachment. CUDA parity is
  exercised when a GPU is available.
- `test_visual_evaluation.py` checks physical-object diagnostics, frozen state probes, per-step
  graph truth, evaluation RNG isolation, and compact artifacts. `test_l40_exp2_launcher.py`
  exercises the launcher with mocked jobs, including preload and collapse rejection.
  `test_l40_exp2_benchmark.py` checks matched hardware-comparison settings and rejects a speed
  report when updates were skipped.
- `test_distributed.py` runs two CPU workers to verify global statistics and gradient scaling;
  `test_distributed_training.py` exercises global mean raw visual GECO, EMA synchronization,
  unanimous update rejection, and rank-specific checkpoint RNG restoration using the real
  trainer. Target variance gathers belong to logging rather than every predictive update.
- Alignment tests check fixed episode permutations; parameter/graph tests check MCC and SHD.
- Remaining tests cover simulator, model, loss, training, checkpoint, and protocol contracts.

Run `.venv/bin/python -m pytest tests/ -q`. Fast smoke tests use tiny models and synthetic
trajectories; passing them does not establish visual segmentation, convergence, or identification.
Vendored code is excluded from lint/type gates; its public wrappers remain covered.
