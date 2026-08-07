#!/usr/bin/env bash
# Experiment-1 pipeline in one command (experiments.pdf §6.1.3):
#   dense reference -> tau = 1.0 x its held-out constraint -> sparse run -> eval.
#
# Any hydra overrides are passed to BOTH runs, which enforces the
# identical-config calibration rule (D12) by construction:
#   bash scripts/run_bounce_example.sh --run-tag=x train.lambda_logit=1e-5
#
# Script flags:
#   --tau-factor=1.0     tau = factor x dense held-out constraint_loss.
#                        §6.1.3 prescribes exactly 1.0; any other value is a
#                        labelled slack ablation.
#   --calib-steps=N      dense reference length (default: same as main run).
#                        Reportable tau requires completion plus accepted
#                        updates beyond D36's exact-K=30 boundary at 115k; a short
#                        override remains a smoke/ablation, not convergence.
#   --main-steps=355000  D36 main run length (default: config value)
#   --eval-episodes=5000 final identifiability sample size (App. F.1: 5000)
#   --final-seed-offset=29 held-out TEST split; tau calibration uses offset 17
#   --eval-device=cpu
#   --run-tag=x          output dir suffix; REQUIRED for parallel launches
set -euo pipefail

PY=${PYTHON:-python}
TAU_FACTOR=${TAU_FACTOR:-1.0}
CALIB_STEPS=${CALIB_STEPS:-}
EVAL_EPISODES=${EVAL_EPISODES:-5000}
FINAL_SEED_OFFSET=${FINAL_SEED_OFFSET:-29}
EVAL_DEVICE=${EVAL_DEVICE:-cpu}

HYDRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --tau-factor=*)  TAU_FACTOR="${arg#*=}" ;;
    --calib-steps=*) CALIB_STEPS="${arg#*=}" ;;
    --main-steps=*)  MAIN_STEPS="${arg#*=}" ;;
    --eval-episodes=*) EVAL_EPISODES="${arg#*=}" ;;
    --final-seed-offset=*) FINAL_SEED_OFFSET="${arg#*=}" ;;
    --eval-device=*) EVAL_DEVICE="${arg#*=}" ;;
    --run-tag=*)     RUN_TAG="${arg#*=}" ;;
    --*)             echo "unknown flag: $arg" >&2; exit 2 ;;
    *)               HYDRA_ARGS+=("$arg") ;;
  esac
done

BASE=outputs/bounce_example_${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}

CALIB_STEPS_OVERRIDE=""
if [ -n "${CALIB_STEPS}" ]; then
  CALIB_STEPS_OVERRIDE="train.steps=${CALIB_STEPS}"
  echo "WARNING: --calib-steps=${CALIB_STEPS}; short references are smoke/ablation only, and terminal-curriculum validation still applies" >&2
elif [ -n "${MAIN_STEPS:-}" ]; then
  CALIB_STEPS_OVERRIDE="train.steps=${MAIN_STEPS}"
fi

echo "== step 1/3: dense reference (A≡1, no path penalty) =="
"$PY" scripts/train.py experiment=bounce_baumgartner \
  "hydra.run.dir=${BASE}/calibration" ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"} \
  model.spartan_dense=true train.sparsity_enabled=false \
  ${CALIB_STEPS_OVERRIDE}
# Tau is calibrated only after the accepted-update curriculum has reached and
# trained at terminal K=30. The held-out dual quantity is the complete Eq. 13:
# pred_loss + rollout_loss (already lambda_roll-weighted) +
# lambda_logit * logit_penalty.
FC_LOSS=$("$PY" scripts/eval_identifiability.py "${BASE}/calibration" --episodes 256 \
  --seed-offset 17 --device "${EVAL_DEVICE}" --require-terminal-curriculum \
  | awk '/constraint_loss/ {print $2}')
TAU=$("$PY" -c "print(float('${FC_LOSS}') * float('${TAU_FACTOR}'))")
echo "dense held-out constraint_loss=${FC_LOSS} -> tau=${TAU} (x${TAU_FACTOR})"

echo "== step 2/3: sparse run (tau=${TAU}) =="
MAIN_STEPS_OVERRIDE=""
if [ -n "${MAIN_STEPS:-}" ]; then MAIN_STEPS_OVERRIDE="train.steps=${MAIN_STEPS}"; fi
"$PY" scripts/train.py experiment=bounce_baumgartner \
  "hydra.run.dir=${BASE}/main" ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"} \
  "train.sparsity_tau=${TAU}" ${MAIN_STEPS_OVERRIDE}

echo "== step 3/3: identifiability evaluation =="
"$PY" scripts/eval_identifiability.py "${BASE}/main" \
  --episodes "${EVAL_EPISODES}" \
  --seed-offset "${FINAL_SEED_OFFSET}" \
  --device "${EVAL_DEVICE}" \
  --require-terminal-curriculum
echo "artifacts in ${BASE}/main (checkpoint, resolved config, recovery_grid.png)"
