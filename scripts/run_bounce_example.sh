#!/usr/bin/env bash
# Experiment-1 pipeline in one command:
#   dense reference -> identity reference -> select the largest feasible tau
#   factor from [2.0, 1.8, 1.6, 1.4] -> sparse run -> eval.
#
# Any hydra overrides are passed to ALL runs, which enforces the
# identical-config calibration rule (D12) by construction:
#   bash scripts/run_bounce_example.sh --run-tag=x train.lambda_logit=1e-5
#
# Script flags:
#   --calib-steps=N      reference-run length (default: same as main run).
#                        Both references use this length. A short
#                        override remains a smoke/ablation, not a converged
#                        reference.
#   --main-steps=300000  main run length (default: config value)
#   --eval-episodes=5000 final identifiability sample size (App. F.1: 5000)
#   --final-seed-offset=29 held-out TEST split; reference checks use offset 17
#   --eval-device=cpu
#   --run-tag=x          output dir suffix; REQUIRED for parallel launches
set -euo pipefail

PY=${PYTHON:-python}
CALIB_STEPS=${CALIB_STEPS:-}
EVAL_EPISODES=${EVAL_EPISODES:-5000}
FINAL_SEED_OFFSET=${FINAL_SEED_OFFSET:-29}
EVAL_DEVICE=${EVAL_DEVICE:-cpu}
REFERENCE_EPISODES=256
REFERENCE_SEED_OFFSET=17
TAU_FACTORS=(2.0 1.8 1.6 1.4)

HYDRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
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
  echo "WARNING: --calib-steps=${CALIB_STEPS}; short references are smoke/ablation only" >&2
elif [ -n "${MAIN_STEPS:-}" ]; then
  CALIB_STEPS_OVERRIDE="train.steps=${MAIN_STEPS}"
fi

echo "== step 1/4: dense reference (A≡1, no path penalty) =="
"$PY" scripts/train.py experiment=bounce_baumgartner \
  ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"} "hydra.run.dir=${BASE}/calibration" \
  model.spartan_dense=true model.spartan_identity=false train.sparsity_enabled=false \
  ${CALIB_STEPS_OVERRIDE}
# Tau is calibrated on the completed fixed T=2 protocol. The held-out dual
# quantity is the complete predictive constraint: teacher-forcing loss plus
# lambda_rollout_t2 times the two-step endpoint loss, plus the logit penalty.
"$PY" scripts/eval_identifiability.py "${BASE}/calibration" \
  --episodes "${REFERENCE_EPISODES}" --seed-offset "${REFERENCE_SEED_OFFSET}" \
  --device "${EVAL_DEVICE}" --require-complete-protocol
FC_LOSS=$("$PY" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["constraint_loss"])' \
  "${BASE}/calibration/metrics.json")

echo "== step 2/4: identity reference (A≡0, token-local residual paths only) =="
"$PY" scripts/train.py experiment=bounce_baumgartner \
  ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"} "hydra.run.dir=${BASE}/identity" \
  model.spartan_dense=false model.spartan_identity=true train.sparsity_enabled=false \
  ${CALIB_STEPS_OVERRIDE}
"$PY" scripts/eval_identifiability.py "${BASE}/identity" \
  --episodes "${REFERENCE_EPISODES}" --seed-offset "${REFERENCE_SEED_OFFSET}" \
  --device "${EVAL_DEVICE}" --require-complete-protocol
IDENTITY_LOSS=$("$PY" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["constraint_loss"])' \
  "${BASE}/identity/metrics.json")

TAU_SELECTION=$("$PY" - "${FC_LOSS}" "${IDENTITY_LOSS}" "${TAU_FACTORS[@]}" <<'PY'
from decimal import Decimal, InvalidOperation
import sys

def positive_decimal(label: str, text: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise SystemExit(f"invalid {label} constraint_loss: {text}") from error
    if not value.is_finite() or value <= 0:
        raise SystemExit(f"invalid {label} constraint_loss: {text}")
    return value


dense = positive_decimal("dense", sys.argv[1])
identity = positive_decimal("identity", sys.argv[2])
factor_texts = sys.argv[3:]
for factor_text in factor_texts:
    factor = Decimal(factor_text)
    tau = factor * dense
    if dense < tau < identity:
        print(f"{factor_text} {tau.normalize()}")
        break
    print(
        f"rejecting tau factor {factor:g}: not {dense:.10g} < {tau:.10g} < {identity:.10g}",
        file=sys.stderr,
    )
else:
    ratio = identity / dense
    raise SystemExit(
        "no candidate tau is feasible: "
        f"C_dense={dense:.10g}, C_identity={identity:.10g}, "
        f"C_identity/C_dense={ratio:.6g}, candidates={factor_texts}"
    )
PY
)
read -r TAU_FACTOR TAU <<< "${TAU_SELECTION}"
echo "reference constraints: dense=${FC_LOSS}, identity=${IDENTITY_LOSS}"
echo "selected tau=${TAU} (factor=${TAU_FACTOR}); verified ${FC_LOSS} < ${TAU} < ${IDENTITY_LOSS}"

echo "== step 3/4: sparse run (tau=${TAU}, factor=${TAU_FACTOR}) =="
MAIN_STEPS_OVERRIDE=""
if [ -n "${MAIN_STEPS:-}" ]; then MAIN_STEPS_OVERRIDE="train.steps=${MAIN_STEPS}"; fi
"$PY" scripts/train.py experiment=bounce_baumgartner \
  ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"} "hydra.run.dir=${BASE}/main" \
  model.spartan_dense=false model.spartan_identity=false train.sparsity_enabled=true \
  "train.sparsity_tau=${TAU}" ${MAIN_STEPS_OVERRIDE}

echo "== step 4/4: identifiability evaluation =="
"$PY" scripts/eval_identifiability.py "${BASE}/main" \
  --episodes "${EVAL_EPISODES}" \
  --seed-offset "${FINAL_SEED_OFFSET}" \
  --device "${EVAL_DEVICE}" \
  --require-complete-protocol
echo "artifacts in ${BASE}/{calibration,identity,main}"
