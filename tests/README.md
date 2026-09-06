# tests/

CPU pytest coverage for the two active experiments:

- `test_state_to_state.py`, `test_rollout_t2.py`, and `test_observational_equivalence.py` cover
  Experiment 1, shared T=2 sampling/gradient behavior, and evaluation-only K=30 rollouts.
- `test_visual_to_visual.py` covers Experiment 2's EMA, causal visual inputs, branch alignment,
  and latent predictive objective. `test_bounce_visual.py` pins equal glyphs and shared physics.
- `test_visual_evaluation.py` checks physical-object diagnostics, frozen state probes, per-step
  graph truth, evaluation RNG isolation, and compact artifacts. `test_l40_exp2_launcher.py`
  exercises the launcher with mocked jobs, including preload and collapse rejection.
- Alignment tests check fixed episode permutations; parameter/graph tests check MCC and SHD.
- Remaining tests cover simulator, model, loss, training, checkpoint, and protocol contracts.

Run `.venv/bin/python -m pytest tests/ -q`. Fast smoke tests use tiny models and synthetic
trajectories; passing them does not establish visual segmentation, convergence, or identification.
Vendored code is excluded from lint/type gates; its public wrappers remain covered.
