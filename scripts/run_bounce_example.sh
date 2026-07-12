#!/usr/bin/env bash
# Full bounce_states identifiability example in one command:
#   calibrate tau (fully connected) -> train with sparsity -> evaluate.
#
# Any hydra overrides are passed through to BOTH runs, which enforces the
# identical-config calibration rule (decisions.md D12) by construction:
#   bash scripts/run_bounce_example.sh data.num_balls=3 train.steps=25000
#
# Script flags (preferred — a typo errors loudly instead of being ignored):
#   --tau-factor=1.1   tau = factor x dense-reference held-out constraint_loss
#                      (default 1.1: SPARTAN p.16 sets tau to the FC model's loss;
#                      slack is spent on gate-closure depth — audit F-9 — so keep
#                      the factor tight)
#   --calib-steps=N    calibration run length. Default: SAME as the main run —
#                      tau from an undertrained reference is meaningless (the v2
#                      failure: 6k-step reference at 0.085 vs 0.045 achievable).
#                      Short values are for smoke tests only.
#   --main-steps=300000  MAIN run length only (default: config value)
#   --run-tag=rung1_seed0  output dir suffix; REQUIRED for parallel launches
#                          (the default timestamp collides across simultaneous starts)
# The calibration run uses model.spartan_dense=true (A≡1, deterministic dense
# attention): SPARTAN's "fully connected model" — NOT the gated model with
# sparsity off, whose Bernoulli gates keep injecting masking noise (audit F-8).
# All other arguments are hydra overrides, passed to BOTH runs (the D12
# identical-config rule; the dense/sparsity toggles are the reference's DEFINITION).
# The equivalent env vars (TAU_FACTOR, CALIB_STEPS, MAIN_STEPS, RUN_TAG, PYTHON)
# still work as a fallback; flags win.
set -euo pipefail

PY=${PYTHON:-python}
TAU_FACTOR=${TAU_FACTOR:-1.1}
CALIB_STEPS=${CALIB_STEPS:-}

HYDRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --tau-factor=*)  TAU_FACTOR="${arg#*=}" ;;
    --calib-steps=*) CALIB_STEPS="${arg#*=}" ;;
    --main-steps=*)  MAIN_STEPS="${arg#*=}" ;;
    --run-tag=*)     RUN_TAG="${arg#*=}" ;;
    --*)             echo "unknown flag: $arg" >&2; exit 2 ;;
    *)               HYDRA_ARGS+=("$arg") ;;
  esac
done

BASE=outputs/bounce_example_${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}

# Calibration length: an undertrained reference inflates tau (v2 failure), so
# default to the main run's own length; --calib-steps only for smoke tests.
CALIB_STEPS_OVERRIDE=""
if [ -n "${CALIB_STEPS}" ]; then
  CALIB_STEPS_OVERRIDE="train.steps=${CALIB_STEPS}"
  echo "WARNING: --calib-steps=${CALIB_STEPS} — tau from a non-converged reference is only for smoke tests" >&2
elif [ -n "${MAIN_STEPS:-}" ]; then
  CALIB_STEPS_OVERRIDE="train.steps=${MAIN_STEPS}"
fi

echo "== step 1/3: tau calibration (dense A≡1 reference${CALIB_STEPS:+, ${CALIB_STEPS} steps}) =="
"$PY" scripts/train.py experiment=bounce_states \
  "hydra.run.dir=${BASE}/calibration" ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"} \
  model.spartan_dense=true train.sparsity_enabled=false ${CALIB_STEPS_OVERRIDE}
# tau must be calibrated on the SAME quantity the Lagrangian dual compares
# against it: pred + lambda_logit * logit_penalty (Baumgartner Eq. 9), which
# the eval harness reports as constraint_loss (== pred_loss when lambda_logit=0).
FC_LOSS=$("$PY" scripts/eval_identifiability.py "${BASE}/calibration" --episodes 256 \
  | awk '/constraint_loss/ {print $2}')
TAU=$("$PY" -c "print(round(float('${FC_LOSS}') * float('${TAU_FACTOR}'), 4))")
echo "dense-reference held-out constraint_loss=${FC_LOSS} -> tau=${TAU} (x${TAU_FACTOR})"

echo "== step 2/3: main run (sparsity on, tau=${TAU}) =="
MAIN_STEPS_OVERRIDE=""
if [ -n "${MAIN_STEPS:-}" ]; then MAIN_STEPS_OVERRIDE="train.steps=${MAIN_STEPS}"; fi
"$PY" scripts/train.py experiment=bounce_states \
  "hydra.run.dir=${BASE}/main" ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"} \
  "train.sparsity_tau=${TAU}" ${MAIN_STEPS_OVERRIDE}

echo "== step 3/3: identifiability evaluation =="
"$PY" scripts/eval_identifiability.py "${BASE}/main" --episodes 512
echo "artifacts in ${BASE}/main (checkpoint, resolved config, recovery_grid.png)"
