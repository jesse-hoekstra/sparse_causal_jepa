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
#   --calib-steps=N      dense reference length (default: same as main run —
#                        tau needs a CONVERGED reference; short = smoke only)
#   --main-steps=300000  main run length (default: config value)
#   --eval-episodes=5000 final identifiability sample size (App. F.1: 5000)
#   --final-seed-offset=29 held-out TEST split; tau calibration uses offset 17
#   --eval-device=cpu
#   --tau-max=VALUE      launch gate (§6.1.3): abort unless tau is below the
#                        token-local constraint. The token-local floor is
#                        parameter-encoder-independent (A≡0 disconnects the
#                        parameter tokens from every decoded row), so the
#                        launchers pass the measured floor as a number instead
#                        of retraining the reference each pipeline (D29).
#   --run-tag=x          output dir suffix; REQUIRED for parallel launches
set -euo pipefail

PY=${PYTHON:-python}
TAU_FACTOR=${TAU_FACTOR:-1.0}
CALIB_STEPS=${CALIB_STEPS:-}
TAU_MAX=${TAU_MAX:-}
EVAL_EPISODES=${EVAL_EPISODES:-5000}
FINAL_SEED_OFFSET=${FINAL_SEED_OFFSET:-29}
EVAL_DEVICE=${EVAL_DEVICE:-cpu}

HYDRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --tau-factor=*)  TAU_FACTOR="${arg#*=}" ;;
    --tau-max=*)     TAU_MAX="${arg#*=}" ;;
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
  echo "WARNING: --calib-steps=${CALIB_STEPS} — tau from a non-converged reference is only for smoke tests" >&2
elif [ -n "${MAIN_STEPS:-}" ]; then
  CALIB_STEPS_OVERRIDE="train.steps=${MAIN_STEPS}"
fi

echo "== step 1/3: dense reference (A≡1, no path penalty) =="
"$PY" scripts/train.py experiment=bounce_baumgartner \
  "hydra.run.dir=${BASE}/calibration" ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"} \
  model.spartan_dense=true train.sparsity_enabled=false \
  ${CALIB_STEPS_OVERRIDE}
# tau is calibrated on the exact dual quantity (Eq. 13):
# constraint_loss = pred_loss + lambda_logit * logit_penalty, held out.
FC_LOSS=$("$PY" scripts/eval_identifiability.py "${BASE}/calibration" --episodes 256 \
  --seed-offset 17 --device "${EVAL_DEVICE}" \
  | awk '/constraint_loss/ {print $2}')
TAU=$("$PY" -c "print(float('${FC_LOSS}') * float('${TAU_FACTOR}'))")
echo "dense held-out constraint_loss=${FC_LOSS} -> tau=${TAU} (x${TAU_FACTOR})"
if [ -n "${TAU_MAX}" ] && [ "$("$PY" -c "print(1 if float('${TAU}') > float('${TAU_MAX}') else 0)")" = "1" ]; then
  echo "ABORT: tau=${TAU} > --tau-max=${TAU_MAX} (the token-local floor). A tau at or" >&2
  echo "above the token-local constraint is satisfiable by the empty graph and cannot" >&2
  echo "force parameter edges — inspect the dense run before spending main-run compute." >&2
  exit 3
fi

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
  --device "${EVAL_DEVICE}"
echo "artifacts in ${BASE}/main (checkpoint, resolved config, recovery_grid.png)"
