# Contract tests

Run `OMP_NUM_THREADS=1 pytest` from the repository root. Tests use synthetic CPU
tensors, including two-process Gloo checks; no dataset download or GPU is required.
They cover independent frames, permitted context, arbitrary dimensions, history
and action indexing, attached parameter reuse, online-only gradients, EMA timing
and checkpoints, sparse path blocks, independent penalties, numerical guards,
configuration, and generic data/evaluation helpers.

`python scripts/smoke.py` additionally checks image/vector forward/backward with
representation regularizers and gradient accumulation. These checks establish
software contracts, not scientific recovery or identifiability.

Scientific data tests compare both equations to independent float64 references,
check Jacobians/supports, recovery and sufficient-influence constructions, native
frame rendering, paired draws, split/worker determinism, dataset bundles, and
dataset-aware resume. Their CPU integrations use only tiny generated episodes.

Experiment tests cover paired initialization/data, SIGReg-only objectives,
calibration-only recovery, physical prediction and code-swap evaluation, best
checkpoint provenance, interrupted resume, aggregation, and CLI preflight.
All use tiny synthetic CPU episodes; W&B unit tests use isolated mocks.
