#!/usr/bin/env bash
# Full bounce_states identifiability example in one command:
#   calibrate tau (fully connected) -> train with sparsity -> evaluate.
#
# Any hydra overrides are passed through to BOTH runs, which enforces the
# identical-config calibration rule (decisions.md D12) by construction:
#   bash scripts/run_bounce_example.sh data.num_balls=3 train.steps=25000
#
# Script flags (preferred — a typo errors loudly instead of being ignored):
#   --tau-factor=2.0   tau = factor x fully-connected held-out pred_loss (default 1.5)
#   --calib-steps=6000 calibration run length (default 6000)
#   --main-steps=300000  MAIN run length only (default: config value)
#   --run-tag=rung1_seed0  output dir suffix; REQUIRED for parallel launches
#                          (the default timestamp collides across simultaneous starts)
# All other arguments are hydra overrides, passed to BOTH runs (the D12
# identical-config rule). The equivalent env vars (TAU_FACTOR, CALIB_STEPS,
# MAIN_STEPS, RUN_TAG, PYTHON) still work as a fallback; flags win.
set -euo pipefail

PY=${PYTHON:-python}
TAU_FACTOR=${TAU_FACTOR:-1.5}
CALIB_STEPS=${CALIB_STEPS:-6000}

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

echo "== step 1/3: tau calibration (fully connected, ${CALIB_STEPS} steps) =="
"$PY" scripts/train.py experiment=bounce_states \
  "hydra.run.dir=${BASE}/calibration" ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"} \
  train.sparsity_enabled=false "train.steps=${CALIB_STEPS}"
FC_LOSS=$("$PY" scripts/eval_identifiability.py "${BASE}/calibration" --episodes 256 \
  | awk '/pred_loss/ {print $2}')
TAU=$("$PY" -c "print(round(float('${FC_LOSS}') * float('${TAU_FACTOR}'), 4))")
echo "fully-connected held-out pred_loss=${FC_LOSS} -> tau=${TAU} (x${TAU_FACTOR})"

echo "== step 2/3: main run (sparsity on, tau=${TAU}) =="
MAIN_STEPS_OVERRIDE=""
if [ -n "${MAIN_STEPS:-}" ]; then MAIN_STEPS_OVERRIDE="train.steps=${MAIN_STEPS}"; fi
"$PY" scripts/train.py experiment=bounce_states \
  "hydra.run.dir=${BASE}/main" ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"} \
  "train.sparsity_tau=${TAU}" ${MAIN_STEPS_OVERRIDE}

echo "== step 3/3: identifiability evaluation =="
"$PY" scripts/eval_identifiability.py "${BASE}/main" --episodes 512
echo "artifacts in ${BASE}/main (checkpoint, resolved config, recovery_grid.png)"
